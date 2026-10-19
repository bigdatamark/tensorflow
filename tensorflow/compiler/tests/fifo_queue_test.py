# [MODULE DOC]
# Tests for the first-in-first-out queue ops on XLA devices: enqueue, shape checks, dequeue,
# parallel producers and consumers, blocking behavior, multi-component elements and size. One test
# class; no import-time side effects.
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
"""Tests for tensorflow.ops.data_flow_ops.FIFOQueue."""

import time

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


class FIFOQueueTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Tests the FIFO queue running in the XLA test scope, one session per test.

  def testEnqueue(self):
    # [FUNCTION DOC]
    # Creates a queue of capacity 10 and enqueues one float; asserts the enqueue runs.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      enqueue_op.run()

  def testEnqueueWithShape(self):
    # [FUNCTION DOC]
    # A queue with fixed element shape 3x2 accepts a matching enqueue, rejects a 2x3 value with a
    # value error at graph construction, and afterwards holds exactly one element.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shapes=(3, 2))
      enqueue_correct_op = q.enqueue(([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],))
      enqueue_correct_op.run()
      with self.assertRaises(ValueError):
        q.enqueue(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],))
      self.assertEqual(1, self.evaluate(q.size()))

  def testMultipleDequeues(self):
    # [FUNCTION DOC]
    # Enqueues 1, 2, 3 and dequeues three times in one run; asserts the set of values is unchanged
    # (order not asserted).
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      self.evaluate(q.enqueue([1]))
      self.evaluate(q.enqueue([2]))
      self.evaluate(q.enqueue([3]))
      a, b, c = self.evaluate([q.dequeue(), q.dequeue(), q.dequeue()])
      self.assertAllEqual(set([1, 2, 3]), set([a, b, c]))

  def testQueuesDontShare(self):
    # [FUNCTION DOC]
    # Two separate queues are independent: each returns the element enqueued into it (2 and 1).
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      self.evaluate(q.enqueue(1))
      q2 = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      self.evaluate(q2.enqueue(2))
      self.assertAllEqual(self.evaluate(q2.dequeue()), 2)
      self.assertAllEqual(self.evaluate(q.dequeue()), 1)

  def testEnqueueDictWithoutNames(self):
    # [FUNCTION DOC]
    # Enqueuing a mapping into a queue without component names raises a value error mentioning
    # that names are required.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      with self.assertRaisesRegex(ValueError, "must have names"):
        q.enqueue({"a": 12.0})

  def testParallelEnqueue(self):
    # [FUNCTION DOC]
    # Ten threads each enqueue one value concurrently; a single thread then dequeues ten times and
    # asserts the same values come back (any order).
    with self.session() as sess, self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
      # [FUNCTION DOC]
      # HELPER: thread body that runs the given enqueue op in the session; returns nothing.
        sess.run(enqueue_op)

      threads = [
          self.checkedThread(target=enqueue, args=(e,)) for e in enqueue_ops
      ]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      # Dequeue every element using a single thread.
      results = []
      for _ in range(len(elems)):
        results.append(self.evaluate(dequeued_t))
      self.assertItemsEqual(elems, results)

  def testParallelDequeue(self):
    # [FUNCTION DOC]
    # Ten values are enqueued sequentially, then ten threads each dequeue one; asserts the
    # collected values equal the enqueued ones (any order).
    with self.session() as sess, self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Enqueue every element using a single thread.
      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      # Run one consumer thread for each element in elems.
      results = []

      def dequeue():
      # [FUNCTION DOC]
      # HELPER: thread body that dequeues one value and appends it to the shared results list.
        results.append(sess.run(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in enqueue_ops]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, results)

  def testDequeue(self):
    # [FUNCTION DOC]
    # Enqueues three values and dequeues them one by one; asserts first-in-first-out order.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      for i in range(len(elems)):
        vals = self.evaluate(dequeued_t)
        self.assertEqual([elems[i]], vals)

  def testEnqueueAndBlockingDequeue(self):
    # [FUNCTION DOC]
    # Starts a dequeuer and an enqueuer thread on a queue of capacity 3; the enqueuer sleeps briefly
    # so the dequeue blocks first. Asserts the dequeued values equal the enqueued ones in order.
    with self.session() as sess, self.test_scope():
      q = data_flow_ops.FIFOQueue(3, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      def enqueue():
      # [FUNCTION DOC]
      # HELPER: producer thread body; waits 0.1 seconds, then runs each enqueue op in order.
        # The enqueue_ops should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        for enqueue_op in enqueue_ops:
          sess.run(enqueue_op)

      results = []

      def dequeue():
      # [FUNCTION DOC]
      # HELPER: consumer thread body; dequeues as many elements as were enqueued, appending each to
      # the shared results list.
        for _ in range(len(elems)):
          results.append(sess.run(dequeued_t))

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      for elem, result in zip(elems, results):
        self.assertEqual([elem], result)

  def testMultiEnqueueAndDequeue(self):
    # [FUNCTION DOC]
    # Uses a queue whose elements have two components (integer and float); enqueues three pairs and
    # dequeues them, asserting each component and the first-in-first-out order.
    with self.session() as sess, self.test_scope():
      q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.float32))
      elems = [(5, 10.0), (10, 20.0), (15, 30.0)]
      enqueue_ops = [q.enqueue((x, y)) for x, y in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      for i in range(len(elems)):
        x_val, y_val = sess.run(dequeued_t)
        x, y = elems[i]
        self.assertEqual([x], x_val)
        self.assertEqual([y], y_val)

  def testQueueSizeEmpty(self):
    # [FUNCTION DOC]
    # Asserts a new queue reports size zero.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      self.assertEqual([0], self.evaluate(q.size()))

  def testQueueSizeAfterEnqueueAndDequeue(self):
    # [FUNCTION DOC]
    # Asserts the size result is a scalar, equals 1 after an enqueue and 0 after a dequeue.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue()
      size = q.size()
      self.assertEqual([], size.get_shape())

      enqueue_op.run()
      self.assertEqual(1, self.evaluate(size))
      dequeued_t.op.run()
      self.assertEqual(0, self.evaluate(size))


if __name__ == "__main__":
  test.main()
