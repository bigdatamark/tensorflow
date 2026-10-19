# [MODULE DOC]
# Test module for the first-in-first-out queue (data_flow_ops FIFOQueue) running on XLA test
# devices. One test class covers enqueue, shape validation, dequeue ordering, queue independence,
# multi-threaded producers and consumers, blocking dequeue, multi-component elements and queue
# size. No import-time side effects; runs the test runner when executed as a script.
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
  # XLA test case for FIFO queues. Each test creates queues inside a session and the test scope
  # and drives them with enqueue, dequeue and size operations, sometimes from several threads.

  def testEnqueue(self):
    # [FUNCTION DOC]
    # Creates a queue of capacity ten for floats and runs a single enqueue of a scalar; passes if
    # no error occurs.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      enqueue_op.run()

  def testEnqueueWithShape(self):
    # [FUNCTION DOC]
    # Creates a queue with a fixed element shape of three by two. Enqueuing a matching value works;
    # enqueuing a two-by-three value raises a value error at graph construction. Asserts the queue
    # size is one afterwards.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shapes=(3, 2))
      enqueue_correct_op = q.enqueue(([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],))
      enqueue_correct_op.run()
      with self.assertRaises(ValueError):
        q.enqueue(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],))
      self.assertEqual(1, self.evaluate(q.size()))

  def testMultipleDequeues(self):
    # [FUNCTION DOC]
    # Enqueues 1, 2 and 3 into an integer scalar queue, runs three dequeues together and asserts
    # the set of dequeued values equals the set of enqueued values.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      self.evaluate(q.enqueue([1]))
      self.evaluate(q.enqueue([2]))
      self.evaluate(q.enqueue([3]))
      a, b, c = self.evaluate([q.dequeue(), q.dequeue(), q.dequeue()])
      self.assertAllEqual(set([1, 2, 3]), set([a, b, c]))

  def testQueuesDontShare(self):
    # [FUNCTION DOC]
    # Creates two separate queues, enqueues one into the first and two into the second, and
    # asserts each queue dequeues only its own element.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      self.evaluate(q.enqueue(1))
      q2 = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      self.evaluate(q2.enqueue(2))
      self.assertAllEqual(self.evaluate(q2.dequeue()), 2)
      self.assertAllEqual(self.evaluate(q.dequeue()), 1)

  def testEnqueueDictWithoutNames(self):
    # [FUNCTION DOC]
    # Enqueues a mapping into a queue that has no component names and asserts a value error whose
    # message says the queue must have names.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      with self.assertRaisesRegex(ValueError, "must have names"):
        q.enqueue({"a": 12.0})

  def testParallelEnqueue(self):
    # [FUNCTION DOC]
    # Starts one thread per element (ten floats) each running its own enqueue, joins them, then
    # dequeues all elements from a single thread and asserts the dequeued items equal the
    # enqueued ones regardless of order.
    with self.session() as sess, self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        # [FUNCTION DOC]
        # HELPER: producer thread body.
        # Runs the given enqueue operation in the session; returns nothing.
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
    # Enqueues ten floats from one thread, then starts one consumer thread per element that each
    # runs a dequeue and appends the result; after joining, asserts the results equal the
    # enqueued items regardless of order.
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
        # HELPER: consumer thread body.
        # Runs dequeue operation(s) in the session and appends the result(s) to the shared results
        # list (one dequeue per call in the parallel test, one per element in the blocking test);
        # returns nothing.
        results.append(sess.run(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in enqueue_ops]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, results)

  def testDequeue(self):
    # [FUNCTION DOC]
    # Enqueues 10, 20 and 30 in order, then dequeues three times and asserts the elements come
    # out in first-in-first-out order.
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
    # Uses a queue of capacity three. A dequeue thread starts consuming while an enqueue thread
    # waits briefly (so the dequeue is blocked) and then enqueues three values. After joining
    # both, asserts the values were dequeued in the enqueued order.
    with self.session() as sess, self.test_scope():
      q = data_flow_ops.FIFOQueue(3, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body.
        # Sleeps briefly so that the consumer blocks first, then runs every enqueue operation in
        # order in the session; returns nothing.
        # The enqueue_ops should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        for enqueue_op in enqueue_ops:
          sess.run(enqueue_op)

      results = []

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body.
        # Runs dequeue operation(s) in the session and appends the result(s) to the shared results
        # list (one dequeue per call in the parallel test, one per element in the blocking test);
        # returns nothing.
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
    # Uses a queue whose elements have an integer and a float component, enqueues three such pairs
    # and dequeues them, asserting each component of each pair comes back in order.
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
    # Asserts the size of a newly created queue is zero.
    with self.session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      self.assertEqual([0], self.evaluate(q.size()))

  def testQueueSizeAfterEnqueueAndDequeue(self):
    # [FUNCTION DOC]
    # Asserts the size result is a scalar shape, then that the size is one after an enqueue and
    # zero after a following dequeue.
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
