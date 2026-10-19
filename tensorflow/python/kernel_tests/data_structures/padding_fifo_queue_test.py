# [MODULE DOC]
# Unit tests for the padding first-in-first-out queue in the data flow ops module, plus a small
# test case for building a queue by selecting from a list of queues. The tests cover queue
# construction (the generated graph node definition), single and batched enqueue and dequeue,
# partially known and variable component shapes with zero padding on batched dequeue, size queries,
# multi-threaded producers and consumers, blocking behaviour, closing semantics, shared queues,
# selecting a queue by index, cancellation of blocked operations on session close, and element
# data types.
# Most tests run only under the graph-mode (version 1) test runner because the queue class is not
# available in version 2 behavior; they use a cached session and run operations or evaluate
# tensors in it. There are no import-time side effects beyond the standard test entry point that
# runs all tests when the file is executed directly.
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
"""Tests for tensorflow.ops.data_flow_ops.PaddingFIFOQueue."""

import random
import time

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


@test_util.run_v1_only("PaddingFIFOQueue removed from v2")
class PaddingFIFOQueueTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the padding first-in-first-out queue (a queue whose components may have
  # partially unknown shapes and whose batched dequeues pad each component with zeros up to the
  # largest shape in the batch). Restricted to the graph-mode test runner. Instances hold no state
  # of their own; each test builds its own queue inside a cached session or graph and uses helper
  # methods inherited from the base test case (evaluate, checked threads, proto comparison, error
  # assertions).

  def testConstructor(self):
    # [FUNCTION DOC]
    # Test: constructing a queue with capacity ten, one float component and one component shape with
    # an unknown dimension. Inside a fresh graph it checks that the queue reference is a tensor and
    # that the node definition of the creating operation is exactly the expected padding queue
    # operation with component type float, shape with dimension size minus one (unknown), capacity ten
    # and empty container and shared name attributes.
    with ops.Graph().as_default():
      q = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((None,),), name="Q")
    self.assertTrue(isinstance(q.queue_ref, tensor.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'PaddingFIFOQueueV2'
      attr { key: 'component_types' value { list { type: DT_FLOAT } } }
      attr { key: 'shapes' value { list { shape { dim { size: -1 } } } } }
      attr { key: 'capacity' value { i: 10 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      """, q.queue_ref.op.node_def)

  def testMultiQueueConstructor(self):
    # [FUNCTION DOC]
    # Test: constructing a queue with two components (integer and float, both scalar shapes), capacity
    # five, an explicit shared name and an explicit operation name. Asserts the queue reference is a
    # tensor and that the node definition lists both component types, two empty shapes, capacity five
    # and the shared name given.
    with ops.Graph().as_default():
      q = data_flow_ops.PaddingFIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32), ((), ()),
          shared_name="foo",
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, tensor.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'PaddingFIFOQueueV2'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list { shape { } shape { } } } }
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: 'foo' } }
      """, q.queue_ref.op.node_def)

  def testConstructorWithShapes(self):
    # [FUNCTION DOC]
    # Test: constructing a two-component queue whose shapes are given as shape objects (four
    # dimensions for the first component, two for the second). Asserts the queue reference is a tensor
    # and that the node definition records the full dimension sizes of both shapes, capacity five and
    # default container and shared name.
    with ops.Graph().as_default():
      q = data_flow_ops.PaddingFIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32),
          shapes=(tensor_shape.TensorShape([1, 1, 2, 3]),
                  tensor_shape.TensorShape([5, 8])),
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, tensor.Tensor))
    self.assertProtoEquals("""
      name:'Q' op:'PaddingFIFOQueueV2'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list {
        shape { dim { size: 1 }
                dim { size: 1 }
                dim { size: 2 }
                dim { size: 3 } }
        shape { dim { size: 5 }
                dim { size: 8 } }
      } } }
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      """, q.queue_ref.op.node_def)

  def testEnqueue(self):
    # [FUNCTION DOC]
    # Test: the simplest enqueue. Creates a float scalar queue of capacity ten and runs one enqueue
    # operation of a single value. Passes if the operation runs without error; nothing else is
    # asserted.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      enqueue_op = q.enqueue((10.0,))
      enqueue_op.run()

  def testEnqueueWithShape(self):
    # [FUNCTION DOC]
    # Test: enqueue with a fully known shape of three by two. An element of the right shape is
    # enqueued and run successfully; building an enqueue with an element of shape two by three must
    # raise a value error at graph construction time. Finally asserts that the queue size is one,
    # showing the invalid enqueue never ran.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=((3, 2),))
      enqueue_correct_op = q.enqueue(([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],))
      enqueue_correct_op.run()
      with self.assertRaises(ValueError):
        q.enqueue(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],))
      self.assertEqual(1, q.size().eval())

  def testEnqueueManyWithShape(self):
    # [FUNCTION DOC]
    # Test: batched enqueue into a two-component integer queue with shapes scalar and length two.
    # Enqueues a batch of four elements in one operation and asserts that the queue size is then four.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(
          10, [dtypes_lib.int32, dtypes_lib.int32], shapes=[(), (2,)])
      q.enqueue_many([[1, 2, 3, 4], [[1, 1], [2, 2], [3, 3], [4, 4]]]).run()
      self.assertEqual(4, q.size().eval())

  def testParallelEnqueue(self):
    # [FUNCTION DOC]
    # Test: ten enqueue operations (one per value from ten to one hundred) are run concurrently, one
    # worker thread each, on a scalar float queue. After joining all threads, the elements are
    # dequeued one by one on the main thread and the test asserts the dequeued values are the same
    # multiset as the enqueued values (order is not checked). The default graph is first switched to
    # keep device scope stacks per thread so threads do not corrupt each other.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        # [FUNCTION DOC]
        # HELPER: thread body for testParallelEnqueue.
        # Evaluates (runs) the single enqueue operation passed in as the argument; returns nothing.
        self.evaluate(enqueue_op)

      threads = [
          self.checkedThread(
              target=enqueue, args=(e,)) for e in enqueue_ops
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
    # Test: ten values are enqueued sequentially on the main thread, then ten worker threads each
    # dequeue one element concurrently and append it to a shared results list. Asserts the results
    # are the same multiset as the enqueued values. The graph is switched to per-thread device
    # scoping beforehand.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
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
        # HELPER: thread body for testParallelDequeue.
        # Evaluates the shared dequeue tensor once and appends the resulting value to the enclosing
        # results list; takes no arguments and returns nothing.
        results.append(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in enqueue_ops]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, results)

  def testDequeue(self):
    # [FUNCTION DOC]
    # Test: three scalar floats are enqueued (each enqueue operation run separately), then dequeued
    # one at a time. Asserts each dequeued value equals, as a one-element list, the next enqueued
    # value, confirming first-in-first-out order.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
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
    # Test: a dequeue thread starts on an empty queue of capacity three and blocks while an enqueue
    # thread, after a short sleep, enqueues three values. Asserts the dequeued values come out in the
    # enqueued order. Relies on a fixed sleep to let the dequeue block first (acknowledged in a
    # comment as something to remove).
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(3, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body for testEnqueueAndBlockingDequeue.
        # Sleeps for a tenth of a second so the consumer blocks first, then evaluates each of the
        # pre-built enqueue operations in order. Returns nothing.
        # The enqueue_ops should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        for enqueue_op in enqueue_ops:
          self.evaluate(enqueue_op)

      results = []

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testEnqueueAndBlockingDequeue.
        # Evaluates the dequeue tensor once per expected element (three times) and appends each result to
        # the enclosing results list. Returns nothing; blocks while the queue is empty.
        for _ in range(len(elems)):
          results.append(self.evaluate(dequeued_t))

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
    # Test: a two-component queue (integer and float scalars) receives three element tuples, each via
    # its own enqueue operation. Dequeues tuples one at a time and asserts both components of each
    # tuple equal the enqueued pair, wrapped as one-element lists, in first-in-first-out order.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10,
                                         (dtypes_lib.int32, dtypes_lib.float32),
                                         ((), ()))
      elems = [(5, 10.0), (10, 20.0), (15, 30.0)]
      enqueue_ops = [q.enqueue((x, y)) for x, y in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      for i in range(len(elems)):
        x_val, y_val = self.evaluate(dequeued_t)
        x, y = elems[i]
        self.assertEqual([x], x_val)
        self.assertEqual([y], y_val)

  def testQueueSizeEmpty(self):
    # [FUNCTION DOC]
    # Test: a newly created queue reports size zero (returned as a one-element list).
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      self.assertEqual([0], q.size().eval())

  def testQueueSizeAfterEnqueueAndDequeue(self):
    # [FUNCTION DOC]
    # Test: the size tensor has an empty (scalar) static shape; after running one enqueue the size
    # evaluates to one, and after running the dequeue operation the size evaluates to zero.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue()
      size = q.size()
      self.assertEqual([], size.get_shape())

      enqueue_op.run()
      self.assertEqual(1, self.evaluate(size))
      dequeued_t.op.run()
      self.assertEqual(0, self.evaluate(size))

  def testEnqueueMany(self):
    # [FUNCTION DOC]
    # Test: a batch of four values is enqueued twice via the same batched enqueue operation, then
    # eight single dequeues are evaluated. Asserts each dequeued value equals the matching element of
    # the original batch, cycling through it twice, preserving order.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue()
      enqueue_op.run()
      enqueue_op.run()

      for i in range(8):
        vals = self.evaluate(dequeued_t)
        self.assertEqual([elems[i % 4]], vals)

  def testEmptyEnqueueMany(self):
    # [FUNCTION DOC]
    # Test: batched enqueue of an empty batch (zero elements with shape two by three) into a queue
    # whose component shape is two unknown dimensions. The size is zero before and still zero after
    # running the enqueue, showing an empty batch is accepted and adds nothing.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, (
          (None, None),))
      empty_t = constant_op.constant(
          [], dtype=dtypes_lib.float32, shape=[0, 2, 3])
      enqueue_op = q.enqueue_many((empty_t,))
      size_t = q.size()

      self.assertEqual([0], self.evaluate(size_t))
      enqueue_op.run()
      self.assertEqual([0], self.evaluate(size_t))

  def testEmptyDequeueMany(self):
    # [FUNCTION DOC]
    # Test: dequeue-many with a count of zero on a scalar float queue returns an empty list, both
    # when the queue is empty and after one element has been enqueued (the element is not consumed).
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, shapes=((),))
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue_many(0)

      self.assertEqual([], self.evaluate(dequeued_t).tolist())
      enqueue_op.run()
      self.assertEqual([], self.evaluate(dequeued_t).tolist())

  def testEmptyDequeueManyWithDynamicShape(self):
    # [FUNCTION DOC]
    # Test: dequeue-many of zero elements from a queue with an unknown-length vector component
    # returns an empty list, both before and after enqueueing one vector.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=((None,),))
      enqueue_op = q.enqueue(([10.0],))
      dequeued_t = q.dequeue_many(0)

      self.assertEqual([], self.evaluate(dequeued_t).tolist())
      enqueue_op.run()
      self.assertEqual([], self.evaluate(dequeued_t).tolist())

  def testEmptyDequeueUpToWithDynamicShape(self):
    # [FUNCTION DOC]
    # Test: dequeue-up-to with a count of zero from a queue with an unknown-length vector component
    # returns an empty list before and after one vector is enqueued.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=((None,),))
      enqueue_op = q.enqueue(([10.0],))
      dequeued_t = q.dequeue_up_to(0)

      self.assertEqual([], self.evaluate(dequeued_t).tolist())
      enqueue_op.run()
      self.assertEqual([], self.evaluate(dequeued_t).tolist())

  def testConstructPaddingFIFOQueueWithNoShape(self):
    # [FUNCTION DOC]
    # Test: creating the queue with the shapes argument set to nothing/null must raise a value error
    # whose message says a list of shapes must be provided when providing partial shapes. The error
    # is expected during construction or evaluation of the queue reference.
    with self.cached_session():
      with self.assertRaisesRegex(
          ValueError,
          r"When providing partial shapes, a list of shapes must be provided."):
        self.evaluate(
            data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32,
                                           None).queue_ref)

  def testMultiEnqueueMany(self):
    # [FUNCTION DOC]
    # Test: batched enqueue into a two-component queue (float scalar, integer vector of length two)
    # run twice. Eight dequeues are checked: the float component equals the matching element of the
    # batch and the integer component equals the matching pair, cycling twice through four elements.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10,
                                         (dtypes_lib.float32, dtypes_lib.int32),
                                         ((), (2,)))
      float_elems = [10.0, 20.0, 30.0, 40.0]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue()

      enqueue_op.run()
      enqueue_op.run()

      for i in range(8):
        float_val, int_val = self.evaluate(dequeued_t)
        self.assertEqual(float_elems[i % 4], float_val)
        self.assertAllEqual(int_elems[i % 4], int_val)

  def testMultiEnqueueManyWithPartiallyKnownShapes(self):
    # [FUNCTION DOC]
    # Test: same scenario as the multi-component batched enqueue test, but the integer component is
    # declared with an unknown length vector shape. Verifies that eight single dequeues return the
    # right values in first-in-first-out order across two runs of the batched enqueue.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (None,)))
      float_elems = [10.0, 20.0, 30.0, 40.0]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue()

      enqueue_op.run()
      enqueue_op.run()

      for i in range(8):
        float_val, int_val = self.evaluate(dequeued_t)
        self.assertEqual(float_elems[i % 4], float_val)
        self.assertAllEqual(int_elems[i % 4], int_val)

  def testDequeueMany(self):
    # [FUNCTION DOC]
    # Test: ten floats are enqueued in one batch, then dequeue-many with four is evaluated twice.
    # Asserts the results are the first four and the next four elements, in order.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(4)

      enqueue_op.run()

      self.assertAllEqual(elems[0:4], self.evaluate(dequeued_t))
      self.assertAllEqual(elems[4:8], self.evaluate(dequeued_t))

  def testDequeueUpToNoBlocking(self):
    # [FUNCTION DOC]
    # Test: with ten elements enqueued, dequeue-up-to four evaluated twice returns elements zero to
    # three then four to seven; there is enough data so nothing blocks and each call returns a full
    # batch.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_up_to(4)

      enqueue_op.run()

      self.assertAllEqual(elems[0:4], self.evaluate(dequeued_t))
      self.assertAllEqual(elems[4:8], self.evaluate(dequeued_t))

  def testMultiDequeueMany(self):
    # [FUNCTION DOC]
    # Test: two-component queue (float scalar, integer vector of length two) holding ten elements.
    # Dequeue-many of four is evaluated twice and a single dequeue once. Asserts values for both
    # components in order, and that the static shapes of the dequeued tensors match the actual value
    # shapes (batch dimension four for batched, none for single).
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (2,)))
      float_elems = [
          10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
      ]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                   [15, 16], [17, 18], [19, 20]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue_many(4)
      dequeued_single_t = q.dequeue()

      enqueue_op.run()

      float_val, int_val = self.evaluate(dequeued_t)
      self.assertAllEqual(float_elems[0:4], float_val)
      self.assertAllEqual(int_elems[0:4], int_val)
      self.assertEqual(float_val.shape, dequeued_t[0].get_shape())
      self.assertEqual(int_val.shape, dequeued_t[1].get_shape())

      float_val, int_val = self.evaluate(dequeued_t)
      self.assertAllEqual(float_elems[4:8], float_val)
      self.assertAllEqual(int_elems[4:8], int_val)

      float_val, int_val = self.evaluate(dequeued_single_t)
      self.assertAllEqual(float_elems[8], float_val)
      self.assertAllEqual(int_elems[8], int_val)
      self.assertEqual(float_val.shape, dequeued_single_t[0].get_shape())
      self.assertEqual(int_val.shape, dequeued_single_t[1].get_shape())

  def testMultiDequeueManyWithPartiallyKnownShapes(self):
    # [FUNCTION DOC]
    # Test: like the multi-component dequeue-many test but the integer component has an unknown
    # length shape. Checks values and order for two batched dequeues of four and one single dequeue,
    # and that the actual value shapes are compatible with (rather than equal to) the static shapes
    # of the dequeued tensors.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (None,)))
      float_elems = [
          10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
      ]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                   [15, 16], [17, 18], [19, 20]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue_many(4)
      dequeued_single_t = q.dequeue()

      enqueue_op.run()

      float_val, int_val = self.evaluate(dequeued_t)
      self.assertAllEqual(float_elems[0:4], float_val)
      self.assertAllEqual(int_elems[0:4], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(float_val.shape).is_compatible_with(
              dequeued_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(dequeued_t[
              1].get_shape()))

      float_val, int_val = self.evaluate(dequeued_t)
      self.assertAllEqual(float_elems[4:8], float_val)
      self.assertAllEqual(int_elems[4:8], int_val)

      float_val, int_val = self.evaluate(dequeued_single_t)
      self.assertAllEqual(float_elems[8], float_val)
      self.assertAllEqual(int_elems[8], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(float_val.shape).is_compatible_with(
              dequeued_single_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(
              dequeued_single_t[1].get_shape()))

  def testMultiDequeueManyWithPartiallyKnownShapesAndVariableSizeInput(self):
    # [FUNCTION DOC]
    # Test: the padding behaviour. A queue with a string component of unknown length and an integer
    # component of shape one by unknown receives six elements of increasing length. Dequeue-many of
    # five must pad shorter elements with empty strings and zeros up to the longest in the batch
    # (three), and the remaining single dequeue returns the unpadded fourth-length element. Also
    # checks static shape compatibility with the returned values.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.string, dtypes_lib.int32),
          shapes=((None,), (1, None)))
      str_elems = [["a"], ["ab"], ["abc"], ["abc", "d"], ["abc", "d", "e"],
                   ["abc", "d", "e", "f"]]

      int_elems = [[[1]], [[2]], [[3]], [[1, 2]], [[1, 2, 3]], [[1, 2, 3, 4]]]

      enqueue_ops = [q.enqueue((str_elems[i], int_elems[i])) for i in range(6)]

      dequeued_t = q.dequeue_many(5)
      dequeued_single_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()
      string_val, int_val = self.evaluate(dequeued_t)

      self.assertAllEqual([[b"a", b"", b""], [b"ab", b"", b""],
                           [b"abc", b"", b""], [b"abc", b"d", b""],
                           [b"abc", b"d", b"e"]], string_val)
      self.assertAllEqual([[[1, 0, 0]], [[2, 0, 0]], [[3, 0, 0]], [[1, 2, 0]],
                           [[1, 2, 3]]], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(string_val.shape).is_compatible_with(
              dequeued_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(dequeued_t[
              1].get_shape()))

      string_val, int_val = self.evaluate(dequeued_single_t)
      self.assertAllEqual([b"abc", b"d", b"e", b"f"], string_val)
      self.assertAllEqual([[1, 2, 3, 4]], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(string_val.shape).is_compatible_with(
              dequeued_single_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(
              dequeued_single_t[1].get_shape()))

  def testMultiDequeueUpToPartiallyKnownShapesAndVariableInputNoBlocking(self):
    # [FUNCTION DOC]
    # Test: same padding scenario as the variable-size dequeue-many test but using dequeue-up-to of
    # five, which returns a full batch without blocking because enough elements exist. Checks padded
    # string and integer batches, the following single dequeue, and shape compatibility.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.string, dtypes_lib.int32),
          shapes=((None,), (1, None)))
      str_elems = [["a"], ["ab"], ["abc"], ["abc", "d"], ["abc", "d", "e"],
                   ["abc", "d", "e", "f"]]

      int_elems = [[[1]], [[2]], [[3]], [[1, 2]], [[1, 2, 3]], [[1, 2, 3, 4]]]

      enqueue_ops = [q.enqueue((str_elems[i], int_elems[i])) for i in range(6)]

      dequeued_t = q.dequeue_up_to(5)
      dequeued_single_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()
      string_val, int_val = self.evaluate(dequeued_t)

      self.assertAllEqual([[b"a", b"", b""], [b"ab", b"", b""],
                           [b"abc", b"", b""], [b"abc", b"d", b""],
                           [b"abc", b"d", b"e"]], string_val)
      self.assertAllEqual([[[1, 0, 0]], [[2, 0, 0]], [[3, 0, 0]], [[1, 2, 0]],
                           [[1, 2, 3]]], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(string_val.shape).is_compatible_with(
              dequeued_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(dequeued_t[
              1].get_shape()))

      string_val, int_val = self.evaluate(dequeued_single_t)
      self.assertAllEqual([b"abc", b"d", b"e", b"f"], string_val)
      self.assertAllEqual([[1, 2, 3, 4]], int_val)
      self.assertTrue(
          tensor_shape.TensorShape(string_val.shape).is_compatible_with(
              dequeued_single_t[0].get_shape()))
      self.assertTrue(
          tensor_shape.TensorShape(int_val.shape).is_compatible_with(
              dequeued_single_t[1].get_shape()))

  def testHighDimension(self):
    # [FUNCTION DOC]
    # Test: a queue whose component has fully known shape of four by four by four by four. Ten such
    # elements are enqueued as one batch and dequeued with dequeue-many of ten; asserts the result
    # equals the input array.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.int32, ((4, 4, 4, 4),))
      elems = np.array([[[[[x] * 4] * 4] * 4] * 4 for x in range(10)], np.int32)
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(10)

      enqueue_op.run()
      self.assertAllEqual(dequeued_t, elems)

  def testPartiallyKnownHighDimension(self):
    # [FUNCTION DOC]
    # Test: same as the high dimension test but the declared shape has unknown second and fourth
    # dimensions. Ten elements round-trip through batched enqueue and dequeue-many unchanged.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.int32, (
          (4, None, 4, None),))
      elems = np.array([[[[[x] * 4] * 4] * 4] * 4 for x in range(10)], np.int32)
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(10)

      enqueue_op.run()
      self.assertAllEqual(dequeued_t, elems)

  def testEnqueueWrongShape(self):
    # [FUNCTION DOC]
    # Test: static shape validation at graph build time. For a two-component queue with scalar and
    # length-two shapes, enqueueing a first component of length two raises a value error, as does a
    # batched enqueue whose first component is a scalar rather than a batch.
    q = data_flow_ops.PaddingFIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32),
                                       ((), (2,)))

    with self.assertRaises(ValueError):
      q.enqueue(([1, 2], [2, 2]))

    with self.assertRaises(ValueError):
      q.enqueue_many((7, [[1, 2], [3, 4], [5, 6]]))

  def testBatchSizeMismatch(self):
    # [FUNCTION DOC]
    # Test: batched enqueue into a three-component integer queue where the components have different
    # batch sizes (three, two, three), including cases where one component is an unshaped placeholder
    # in different positions. Each must raise a value error at graph construction.
    q = data_flow_ops.PaddingFIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32,
                                            dtypes_lib.int32), ((), (), ()))

    with self.assertRaises(ValueError):
      q.enqueue_many(([1, 2, 3], [1, 2], [1, 2, 3]))

    with self.assertRaises(ValueError):
      q.enqueue_many(
          ([1, 2, 3], [1, 2], array_ops.placeholder(dtypes_lib.int32)))

    with self.assertRaises(ValueError):
      q.enqueue_many(
          (array_ops.placeholder(dtypes_lib.int32), [1, 2], [1, 2, 3]))

  def testEnqueueManyEmptyTypeConversion(self):
    # [FUNCTION DOC]
    # Test: batched enqueue of two empty lists into an integer and float queue. Asserts the
    # enqueue operation's inputs (after the queue handle) were converted to the queue's component
    # types, integer then float.
    q = data_flow_ops.PaddingFIFOQueue(10,
                                       (dtypes_lib.int32, dtypes_lib.float32), (
                                           (), ()))
    enq = q.enqueue_many(([], []))
    self.assertEqual(dtypes_lib.int32, enq.inputs[1].dtype)
    self.assertEqual(dtypes_lib.float32, enq.inputs[2].dtype)

  def testEnqueueWrongType(self):
    # [FUNCTION DOC]
    # Test: dtype validation. For an integer and float queue, enqueueing and batch enqueueing integer
    # placeholders for both components must raise a value error because the second component has the
    # wrong type.
    q = data_flow_ops.PaddingFIFOQueue(10,
                                       (dtypes_lib.int32, dtypes_lib.float32), (
                                           (), ()))

    with self.assertRaises(ValueError):
      q.enqueue((array_ops.placeholder(dtypes_lib.int32),
                 array_ops.placeholder(dtypes_lib.int32)))

    with self.assertRaises(ValueError):
      q.enqueue_many((array_ops.placeholder(dtypes_lib.int32),
                      array_ops.placeholder(dtypes_lib.int32)))

  def testEnqueueWrongPartiallyKnownShapeAtRuntime(self):
    # [FUNCTION DOC]
    # Test: a queue with component shapes two by two and unknown by three. A placeholder fed with a
    # three by four value for the second component passes static checking but at run time must raise
    # an invalid-argument error saying the expected shape was unknown by three and the actual was
    # three by four.
    with self.cached_session() as sess:
      # First dimension of second component is unknown, second
      # dimension must be 3.
      q = data_flow_ops.PaddingFIFOQueue(10,
                                         (dtypes_lib.int32, dtypes_lib.int32), (
                                             (2, 2), (None, 3)))
      elems_ok = np.array([1] * 4).reshape((2, 2)).astype(np.int32)
      elems_bad = array_ops.placeholder(dtypes_lib.int32)
      enqueue_op = q.enqueue((elems_ok, elems_bad))
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  r"Expected \[\?,3\], got \[3,4\]"):
        sess.run([enqueue_op],
                 feed_dict={elems_bad: np.array([1] * 12).reshape((3, 4))})

  def testEnqueueDequeueManyWrongPartiallyKnownShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the evaluation of the dequeue tensor sits inside the same error-assertion block
    # after the failing run call, so it is never reached (the run call raises first). It does not
    # affect the assertion but is dead code.
    # Test: batched enqueue whose second component has a placeholder fed with shape two by three by
    # four where two by unknown by three is required. Running it must raise an invalid-argument error
    # mentioning a shape mismatch in tuple component one with the expected and actual shapes.
    with self.cached_session() as sess:
      # First dimension of second component is unknown, second
      # dimension must be 3.
      q = data_flow_ops.PaddingFIFOQueue(10,
                                         (dtypes_lib.int32, dtypes_lib.int32), (
                                             (2, 2), (None, 3)))
      elems_ok = np.array([1] * 8).reshape((2, 2, 2)).astype(np.int32)
      elems_bad = array_ops.placeholder(dtypes_lib.int32)
      enqueue_op = q.enqueue_many((elems_ok, elems_bad))
      dequeued_t = q.dequeue_many(2)
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch in tuple component 1. "
          r"Expected \[2,\?,3\], got \[2,3,4\]"):
        sess.run([enqueue_op],
                 feed_dict={elems_bad: np.array([1] * 24).reshape((2, 3, 4))})
        self.evaluate(dequeued_t)

  def testParallelEnqueueMany(self):
    # [FUNCTION DOC]
    # Test: ten threads concurrently run the same batched enqueue of one hundred values into a queue
    # of capacity one thousand, then a single dequeue-many of one thousand is evaluated. Asserts the
    # dequeued values contain every enqueued value ten times (order ignored). Uses per-thread device
    # scoping.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(1000, dtypes_lib.float32, shapes=((),))
      elems = [10.0 * x for x in range(100)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(1000)

      # Enqueue 100 items in parallel on 10 threads.
      def enqueue():
        # [FUNCTION DOC]
        # HELPER: thread body for testParallelEnqueueMany.
        # Evaluates the shared batched enqueue operation once; returns nothing.
        self.evaluate(enqueue_op)

      threads = [self.checkedThread(target=enqueue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      self.assertCountEqual(self.evaluate(dequeued_t), elems * 10)

  def testParallelDequeueMany(self):
    # [FUNCTION DOC]
    # Test: one thousand values are enqueued, then ten threads each evaluate dequeue-many of one
    # hundred concurrently, extending a shared list. Asserts the combined results are the same
    # multiset as the enqueued values.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(1000, dtypes_lib.float32, shapes=((),))
      elems = [10.0 * x for x in range(1000)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(100)

      enqueue_op.run()

      # Dequeue 100 items in parallel on 10 threads.
      dequeued_elems = []

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: thread body for testParallelDequeueMany.
        # Evaluates the shared dequeue-many tensor once and extends the enclosing list of dequeued
        # elements with the returned batch; returns nothing.
        dequeued_elems.extend(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testParallelDequeueUpTo(self):
    # [FUNCTION DOC]
    # Test: one thousand values are enqueued and the queue is closed. Ten threads then each run
    # dequeue-up-to of one hundred and one concurrently from the closed queue, so the last batch is
    # a partial remainder. Asserts the union of all results equals the enqueued values.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(1000, dtypes_lib.float32, shapes=((),))
      elems = [10.0 * x for x in range(1000)]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(101)

      enqueue_op.run()
      close_op.run()

      # Dequeue up to 101 items in parallel on 10 threads, from closed queue.
      dequeued_elems = []

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: thread body for testParallelDequeueUpTo.
        # Evaluates the shared dequeue-up-to tensor once and extends the enclosing list with the batch
        # returned; returns nothing.
        dequeued_elems.extend(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testParallelEnqueueAndDequeue(self):
    # [FUNCTION DOC]
    # Test: a capacity-fifty queue is pre-filled with forty-nine values of ten, then ten enqueue
    # threads (each adding one hundred twenties) and ten dequeue threads (each taking one hundred
    # elements, asserting each is ten or twenty) run concurrently. Afterwards the remaining forty-nine
    # elements are drained and checked to be ten or twenty.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(50, dtypes_lib.float32, shapes=((),))
      initial_elements = [10.0] * 49
      q.enqueue_many((initial_elements,)).run()

      enqueue_op = q.enqueue((20.0,))
      dequeued_t = q.dequeue()

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: thread body for testParallelEnqueueAndDequeue.
        # Runs the shared enqueue operation one hundred times in a loop; returns nothing and may block
        # when the queue is full until dequeuers make room.
        for _ in range(100):
          self.evaluate(enqueue_op)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: thread body for testParallelEnqueueAndDequeue.
        # Evaluates the shared dequeue tensor one hundred times and asserts each value is either ten or
        # twenty. Returns nothing; an assertion failure surfaces through the checked thread.
        for _ in range(100):
          self.assertTrue(self.evaluate(dequeued_t) in (10.0, 20.0))

      enqueue_threads = [self.checkedThread(target=enqueue) for _ in range(10)]
      dequeue_threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for enqueue_thread in enqueue_threads:
        enqueue_thread.start()
      for dequeue_thread in dequeue_threads:
        dequeue_thread.start()
      for enqueue_thread in enqueue_threads:
        enqueue_thread.join()
      for dequeue_thread in dequeue_threads:
        dequeue_thread.join()

      # Dequeue the initial count of elements to clean up.
      cleanup_elems = q.dequeue_many(49).eval()
      for elem in cleanup_elems:
        self.assertTrue(elem in (10.0, 20.0))

  def testMixtureOfEnqueueAndEnqueueMany(self):
    # [FUNCTION DOC]
    # Test: a consumer thread dequeues two hundred fifty values expecting them in strictly increasing
    # order starting from zero, while the main thread feeds a running counter into the queue,
    # randomly choosing (with equal chance) between a single enqueue and a batched enqueue of a
    # random-size run of consecutive integers. After the queue is closed and the consumer joined, the
    # queue must be empty. Uses randomness, so the batch sizes vary between runs.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.int32, shapes=((),))
      enqueue_placeholder = array_ops.placeholder(dtypes_lib.int32, shape=())
      enqueue_op = q.enqueue((enqueue_placeholder,))
      enqueuemany_placeholder = array_ops.placeholder(
          dtypes_lib.int32, shape=(None,))
      enqueuemany_op = q.enqueue_many((enqueuemany_placeholder,))

      dequeued_t = q.dequeue()
      close_op = q.close()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testMixtureOfEnqueueAndEnqueueMany.
        # Dequeues two hundred fifty elements one at a time and asserts the i-th equals i. Returns
        # nothing.
        for i in range(250):
          self.assertEqual(i, self.evaluate(dequeued_t))

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()

      elements_enqueued = 0
      while elements_enqueued < 250:
        # With equal probability, run Enqueue or enqueue_many.
        if random.random() > 0.5:
          enqueue_op.run({enqueue_placeholder: elements_enqueued})
          elements_enqueued += 1
        else:
          count = random.randint(0, min(20, 250 - elements_enqueued))
          range_to_enqueue = np.arange(
              elements_enqueued, elements_enqueued + count, dtype=np.int32)
          enqueuemany_op.run({enqueuemany_placeholder: range_to_enqueue})
          elements_enqueued += count

      close_op.run()
      dequeue_thread.join()
      self.assertEqual(0, q.size().eval())

  def testMixtureOfDequeueAndDequeueMany(self):
    # [FUNCTION DOC]
    # Test: a producer thread enqueues the integers zero to two hundred forty-nine in one batched
    # operation into a capacity-ten queue while the main thread randomly alternates between single
    # dequeues and dequeue-many with a random count fed through a placeholder, asserting each result
    # matches the next expected integers in order. After closing the queue and joining the producer
    # the queue size must be zero. Uses randomness.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.int32, shapes=((),))
      enqueue_op = q.enqueue_many((np.arange(250, dtype=np.int32),))
      dequeued_t = q.dequeue()
      count_placeholder = array_ops.placeholder(dtypes_lib.int32, shape=())
      dequeuemany_t = q.dequeue_many(count_placeholder)

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body for testMixtureOfDequeueAndDequeueMany.
        # Evaluates the batched enqueue operation of two hundred fifty values (blocking as the queue
        # fills). Returns nothing.
        self.evaluate(enqueue_op)

      enqueue_thread = self.checkedThread(target=enqueue)
      enqueue_thread.start()

      elements_dequeued = 0
      while elements_dequeued < 250:
        # With equal probability, run Dequeue or dequeue_many.
        if random.random() > 0.5:
          self.assertEqual(elements_dequeued, self.evaluate(dequeued_t))
          elements_dequeued += 1
        else:
          count = random.randint(0, min(20, 250 - elements_dequeued))
          expected_range = np.arange(
              elements_dequeued, elements_dequeued + count, dtype=np.int32)
          self.assertAllEqual(expected_range,
                              dequeuemany_t.eval({
                                  count_placeholder: count
                              }))
          elements_dequeued += count

      q.close().run()
      enqueue_thread.join()
      self.assertEqual(0, q.size().eval())

  def testBlockingDequeueMany(self):
    # [FUNCTION DOC]
    # Test: a dequeue-many of four blocks on an empty queue in one thread while another thread, after
    # a short sleep, enqueues four values in a batch. Asserts the dequeued values equal the enqueued
    # ones in order.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(4)

      dequeued_elems = []

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body for testBlockingDequeueMany.
        # Sleeps a tenth of a second so the consumer blocks first, then evaluates the batched enqueue
        # operation. Returns nothing.
        # The enqueue_op should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.evaluate(enqueue_op)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testBlockingDequeueMany.
        # Evaluates the dequeue-many tensor (blocking until four elements are available) and extends the
        # shared list with the values converted to a plain list. Returns nothing.
        dequeued_elems.extend(self.evaluate(dequeued_t).tolist())

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      self.assertAllEqual(elems, dequeued_elems)

  def testBlockingDequeueUpTo(self):
    # [FUNCTION DOC]
    # Test: dequeue-up-to of four blocks on an empty queue in one thread; another thread enqueues four
    # values in a batch after a short sleep. Asserts the dequeued values equal the enqueued ones.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_up_to(4)

      dequeued_elems = []

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body for testBlockingDequeueUpTo.
        # Sleeps a tenth of a second so the consumer blocks first, then evaluates the batched enqueue
        # operation. Returns nothing.
        # The enqueue_op should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.evaluate(enqueue_op)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testBlockingDequeueUpTo.
        # Evaluates the dequeue-up-to tensor and extends the shared list with the result converted to a
        # plain list. Returns nothing.
        dequeued_elems.extend(self.evaluate(dequeued_t).tolist())

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      self.assertAllEqual(elems, dequeued_elems)

  def testDequeueManyWithTensorParameter(self):
    # [FUNCTION DOC]
    # Test: the count passed to dequeue-many is itself a tensor. One queue holds one hundred random
    # counts between one and ten; a second queue holds exactly as many random elements as the sum of
    # the counts. Each evaluation of a dequeue-many whose count comes from dequeuing the count queue
    # consumes the next batch; the concatenation of all batches must equal the elements enqueued.
    # Uses randomness for data only.
    with self.cached_session():
      # Define a first queue that contains integer counts.
      dequeue_counts = [random.randint(1, 10) for _ in range(100)]
      count_q = data_flow_ops.PaddingFIFOQueue(100, dtypes_lib.int32, ((),))
      enqueue_counts_op = count_q.enqueue_many((dequeue_counts,))
      total_count = sum(dequeue_counts)

      # Define a second queue that contains total_count elements.
      elems = [random.randint(0, 100) for _ in range(total_count)]
      q = data_flow_ops.PaddingFIFOQueue(total_count, dtypes_lib.int32, ((),))
      enqueue_elems_op = q.enqueue_many((elems,))

      # Define a subgraph that first dequeues a count, then DequeuesMany
      # that number of elements.
      dequeued_t = q.dequeue_many(count_q.dequeue())

      enqueue_counts_op.run()
      enqueue_elems_op.run()

      dequeued_elems = []
      for _ in dequeue_counts:
        dequeued_elems.extend(self.evaluate(dequeued_t))
      self.assertEqual(elems, dequeued_elems)

  def testDequeueFromClosedQueue(self):
    # [FUNCTION DOC]
    # Test: four values are enqueued and the queue closed. The four values can still be dequeued in
    # order, and a further dequeue must raise an out-of-range error saying the queue is closed and
    # has insufficient elements.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()
      close_op.run()
      for elem in elems:
        self.assertEqual([elem], self.evaluate(dequeued_t))

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                  "is closed and has insufficient"):
        self.evaluate(dequeued_t)

  def testBlockingDequeueFromClosedQueue(self):
    # [FUNCTION DOC]
    # Test: a thread dequeues all four enqueued values and then blocks on a fifth dequeue; the main
    # thread, after a short sleep, closes the queue. The blocked dequeue must fail with an
    # out-of-range error saying the queue is closed and has insufficient elements.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testBlockingDequeueFromClosedQueue.
        # Dequeues the four available elements asserting each in order, then expects one more dequeue to
        # raise an out-of-range error about the closed queue. Returns nothing.
        for elem in elems:
          self.assertEqual([elem], self.evaluate(dequeued_t))
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testDequeueUpToFromClosedQueueReturnsRemainder(self):
    # [FUNCTION DOC]
    # Test: with four elements and dequeue-up-to of three, a thread gets the first three and then,
    # once the queue is closed after a short sleep, receives the remaining single element instead of
    # an error. Verifies dequeue-up-to returns the partial remainder from a closed queue.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(3)

      enqueue_op.run()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testDequeueUpToFromClosedQueueReturnsRemainder.
        # Evaluates the dequeue-up-to tensor twice, asserting the first batch is the first three
        # elements and the second is the remaining last element. Returns nothing.
        self.assertAllEqual(elems[:3], self.evaluate(dequeued_t))
        self.assertAllEqual(elems[3:], self.evaluate(dequeued_t))

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueFromClosedEmptyQueue(self):
    # [FUNCTION DOC]
    # Test: a thread blocks dequeuing from an empty queue; after a short sleep the main thread closes
    # the queue and the blocked dequeue must fail with an out-of-range error about the queue being
    # closed with insufficient elements.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      close_op = q.close()
      dequeued_t = q.dequeue()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testBlockingDequeueFromClosedEmptyQueue.
        # Expects the blocking dequeue to raise an out-of-range error mentioning the closed queue and
        # insufficient elements. Returns nothing.
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueManyFromClosedQueue(self):
    # [FUNCTION DOC]
    # Test: four elements are enqueued; a thread dequeue-manys exactly four (succeeds) and then a
    # second dequeue-many blocks until the main thread closes the queue, then fails with an
    # out-of-range error about the closed queue and insufficient elements.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(4)

      enqueue_op.run()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testBlockingDequeueManyFromClosedQueue.
        # Asserts the first dequeue-many returns all four elements, then expects the second to raise an
        # out-of-range error about the closed queue. Returns nothing.
        self.assertAllEqual(elems, self.evaluate(dequeued_t))
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueManyButNotAllFromClosedQueue(self):
    # [FUNCTION DOC]
    # Test: four elements enqueued, dequeue-many of three. The first call returns three elements; the
    # second blocks (only one element remains) until the queue is closed and then must fail with an
    # out-of-range error about the closed queue.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(3)

      enqueue_op.run()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testBlockingDequeueManyButNotAllFromClosedQueue.
        # Asserts the first dequeue-many returns the first three elements, then expects the second call to
        # raise an out-of-range error about the closed queue. Returns nothing.
        self.assertAllEqual(elems[:3], self.evaluate(dequeued_t))
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testEnqueueManyLargerThanCapacityWithConcurrentDequeueMany(self):
    # [FUNCTION DOC]
    # Test: a capacity-four queue is fed a batch of four by one thread while another thread
    # dequeue-manys three; then a close runs from a third thread. The dequeuer gets the first three
    # elements, a second dequeue-many fails with an out-of-range error (queue closed, one element
    # left), and a single dequeue then returns the fourth element. Uses sleeps to order threads.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(3)
      cleanup_dequeue_t = q.dequeue()

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body for testEnqueueManyLargerThanCapacityWithConcurrentDequeueMany.
        # Evaluates the batched enqueue operation. Returns nothing.
        self.evaluate(enqueue_op)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testEnqueueManyLargerThanCapacityWithConcurrentDequeueMany.
        # Asserts the first dequeue-many of three returns the first three elements, that a second
        # dequeue-many raises an out-of-range error, and that a plain dequeue then returns the fourth
        # element. Returns nothing.
        self.assertAllEqual(elems[0:3], self.evaluate(dequeued_t))
        with self.assertRaises(errors_impl.OutOfRangeError):
          self.evaluate(dequeued_t)
        self.assertEqual(elems[3], self.evaluate(cleanup_dequeue_t))

      def close():
        # [FUNCTION DOC]
        # HELPER: closer thread body for testEnqueueManyLargerThanCapacityWithConcurrentDequeueMany.
        # Evaluates the close operation of the queue. Returns nothing.
        self.evaluate(close_op)

      enqueue_thread = self.checkedThread(target=enqueue)
      enqueue_thread.start()

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      close_thread = self.checkedThread(target=close)
      close_thread.start()

      enqueue_thread.join()
      dequeue_thread.join()
      close_thread.join()

  def testClosedBlockingDequeueManyRestoresPartialBatch(self):
    # [FUNCTION DOC]
    # Test: a capacity-four queue with two float components holds three elements. A thread blocks in a
    # dequeue-many of four, then the queue is closed and the dequeue must fail with an out-of-range
    # error. The three partially dequeued elements must have been restored in their original order:
    # single dequeues return the pairs in sequence and the final size is zero.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, (dtypes_lib.float32,
                                             dtypes_lib.float32), ((), ()))
      elems_a = [1.0, 2.0, 3.0]
      elems_b = [10.0, 20.0, 30.0]
      enqueue_op = q.enqueue_many((elems_a, elems_b))
      dequeued_a_t, dequeued_b_t = q.dequeue_many(4)
      cleanup_dequeue_a_t, cleanup_dequeue_b_t = q.dequeue()
      close_op = q.close()

      enqueue_op.run()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testClosedBlockingDequeueManyRestoresPartialBatch.
        # Expects the evaluation of the blocking dequeue-many of both components to raise an out-of-range
        # error once the queue closes. Returns nothing.
        with self.assertRaises(errors_impl.OutOfRangeError):
          self.evaluate([dequeued_a_t, dequeued_b_t])

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      close_op.run()
      dequeue_thread.join()
      # Test that the elements in the partially-dequeued batch are
      # restored in the correct order.
      for elem_a, elem_b in zip(elems_a, elems_b):
        val_a, val_b = self.evaluate([cleanup_dequeue_a_t, cleanup_dequeue_b_t])
        self.assertEqual(elem_a, val_a)
        self.assertEqual(elem_b, val_b)
      self.assertEqual(0, q.size().eval())

  def testBlockingDequeueManyFromClosedEmptyQueue(self):
    # [FUNCTION DOC]
    # Test: dequeue-many of four blocks on an empty queue in a thread; the queue is closed after a
    # short sleep and the dequeue must fail with an out-of-range error about the closed queue.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      close_op = q.close()
      dequeued_t = q.dequeue_many(4)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testBlockingDequeueManyFromClosedEmptyQueue.
        # Expects the blocking dequeue-many to raise an out-of-range error mentioning the closed queue.
        # Returns nothing.
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueUpToFromClosedEmptyQueue(self):
    # [FUNCTION DOC]
    # Test: dequeue-up-to of four blocks on an empty queue; closing the queue after a short sleep
    # makes the blocked dequeue fail with an out-of-range error about the closed queue.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(4)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testBlockingDequeueUpToFromClosedEmptyQueue.
        # Expects the blocking dequeue-up-to to raise an out-of-range error mentioning the closed queue.
        # Returns nothing.
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testEnqueueToClosedQueue(self):
    # [FUNCTION DOC]
    # Test: after one successful enqueue and closing the queue, running the enqueue again must raise a
    # cancelled error whose message says the queue is closed.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      enqueue_op = q.enqueue((10.0,))
      close_op = q.close()

      enqueue_op.run()
      close_op.run()

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegex(errors_impl.CancelledError, "is closed"):
        enqueue_op.run()

  def testEnqueueManyToClosedQueue(self):
    # [FUNCTION DOC]
    # Test: after a batched enqueue succeeds and the queue is closed, running the batched enqueue
    # again must raise a cancelled error whose message says the queue is closed.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()

      enqueue_op.run()
      close_op.run()

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegex(errors_impl.CancelledError, "is closed"):
        enqueue_op.run()

  def testBlockingEnqueueToFullQueue(self):
    # [FUNCTION DOC]
    # Test: a capacity-four queue is filled, then a thread tries one more enqueue and blocks. After a
    # short sleep the main thread dequeues the four original values in order and then the blocked
    # value fifty, which proves the blocked enqueue completed once space was freed. The thread is
    # joined at the end.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue((50.0,))
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body for testBlockingEnqueueToFullQueue.
        # Evaluates the single enqueue operation that blocks while the queue is full. Returns nothing.
        self.evaluate(blocking_enqueue_op)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()
      # The dequeue ops should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      for elem in elems:
        self.assertEqual([elem], self.evaluate(dequeued_t))
      self.assertEqual([50.0], self.evaluate(dequeued_t))
      thread.join()

  def testBlockingEnqueueManyToFullQueue(self):
    # [FUNCTION DOC]
    # Test: a full capacity-four queue; a thread issues a batched enqueue of two more values and
    # blocks. The main thread then dequeues the original four (with small sleeps) followed by fifty
    # and sixty, checking order, and joins the thread.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue_many(([50.0, 60.0],))
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body for testBlockingEnqueueManyToFullQueue.
        # Evaluates the batched enqueue operation that blocks while the queue is full. Returns nothing.
        self.evaluate(blocking_enqueue_op)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()
      # The dequeue ops should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      for elem in elems:
        self.assertEqual([elem], self.evaluate(dequeued_t))
        time.sleep(0.01)
      self.assertEqual([50.0], self.evaluate(dequeued_t))
      self.assertEqual([60.0], self.evaluate(dequeued_t))

      # Make sure the thread finishes before exiting.
      thread.join()

  def testBlockingEnqueueBeforeClose(self):
    # [FUNCTION DOC]
    # Test: with a full queue, one thread has a blocked single enqueue and, after a short sleep,
    # another thread closes the queue. A dequeue of the first element unblocks both; the blocked
    # enqueue still succeeds (it was issued before the close), so the remaining elements twenty to
    # fifty are dequeued in order and the queue ends empty.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue((50.0,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body for testBlockingEnqueueBeforeClose.
        # Evaluates the pending enqueue that blocks while the queue is full; expected to succeed once a
        # dequeue makes room. Returns nothing.
        # Expect the operation to succeed once the dequeue op runs.
        self.evaluate(blocking_enqueue_op)

      enqueue_thread = self.checkedThread(target=blocking_enqueue)
      enqueue_thread.start()

      # The close_op should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      def close():
        # [FUNCTION DOC]
        # HELPER: closer thread body for testBlockingEnqueueBeforeClose.
        # Evaluates the close operation of the queue. Returns nothing.
        self.evaluate(close_op)

      close_thread = self.checkedThread(target=close)
      close_thread.start()

      # The dequeue will unblock both threads.
      self.assertEqual(10.0, self.evaluate(dequeued_t))
      enqueue_thread.join()
      close_thread.join()

      for elem in [20.0, 30.0, 40.0, 50.0]:
        self.assertEqual(elem, self.evaluate(dequeued_t))
      self.assertEqual(0, q.size().eval())

  def testBlockingEnqueueManyBeforeClose(self):
    # [FUNCTION DOC]
    # Test: like the blocking enqueue before close test but with a batched enqueue of two values
    # blocked on a nearly full queue (three of four slots used). After closing, a dequeue frees
    # space; the pending batched enqueue completes, and the remaining values twenty, thirty, fifty
    # and sixty are dequeued in order.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(4, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue_many(([50.0, 60.0],))
      close_op = q.close()
      dequeued_t = q.dequeue()
      enqueue_op.run()

      def blocking_enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body for testBlockingEnqueueManyBeforeClose.
        # Evaluates the blocked batched enqueue operation. Returns nothing.
        self.evaluate(blocking_enqueue_op)

      enqueue_thread = self.checkedThread(target=blocking_enqueue)
      enqueue_thread.start()

      # The close_op should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      def close():
        # [FUNCTION DOC]
        # HELPER: closer thread body for testBlockingEnqueueManyBeforeClose.
        # Evaluates the close operation of the queue. Returns nothing.
        self.evaluate(close_op)

      close_thread = self.checkedThread(target=close)
      close_thread.start()

      # The dequeue will unblock both threads.
      self.assertEqual(10.0, self.evaluate(dequeued_t))
      enqueue_thread.join()
      close_thread.join()
      for elem in [20.0, 30.0, 50.0, 60.0]:
        self.assertEqual(elem, self.evaluate(dequeued_t))

  def testDoesNotLoseValue(self):
    # [FUNCTION DOC]
    # Test: a capacity-one queue holds one element; the size is read five hundred times and must be
    # one every time, showing that reading the size does not consume or lose the value.
    with self.cached_session():
      q = data_flow_ops.PaddingFIFOQueue(1, dtypes_lib.float32, ((),))
      enqueue_op = q.enqueue((10.0,))
      size_t = q.size()

      enqueue_op.run()
      for _ in range(500):
        self.assertEqual(self.evaluate(size_t), [1])

  def testSharedQueueSameSession(self):
    # [FUNCTION DOC]
    # Test: two queue objects created with the same shared name in one session refer to the same
    # underlying queue. An element enqueued through the first is visible in both sizes, can be
    # dequeued through the second (size drops to zero in both), and the reverse direction works for a
    # value enqueued through the second and dequeued through the first.
    with self.cached_session():
      q1 = data_flow_ops.PaddingFIFOQueue(
          1, dtypes_lib.float32, ((),), shared_name="shared_queue")
      q1.enqueue((10.0,)).run()

      q2 = data_flow_ops.PaddingFIFOQueue(
          1, dtypes_lib.float32, ((),), shared_name="shared_queue")

      q1_size_t = q1.size()
      q2_size_t = q2.size()

      self.assertEqual(self.evaluate(q1_size_t), [1])
      self.assertEqual(self.evaluate(q2_size_t), [1])

      self.assertEqual(q2.dequeue().eval(), [10.0])

      self.assertEqual(self.evaluate(q1_size_t), [0])
      self.assertEqual(self.evaluate(q2_size_t), [0])

      q2.enqueue((20.0,)).run()

      self.assertEqual(self.evaluate(q1_size_t), [1])
      self.assertEqual(self.evaluate(q2_size_t), [1])

      self.assertEqual(q1.dequeue().eval(), [20.0])

      self.assertEqual(self.evaluate(q1_size_t), [0])
      self.assertEqual(self.evaluate(q2_size_t), [0])

  def testIncompatibleSharedQueueErrors(self):
    # [FUNCTION DOC]
    # Test: six pairs of queues sharing a name but having mismatched definitions. For each pair the
    # first queue is created successfully and creating the second must raise an op error: different
    # capacity mentions capacity, different or extra component types mention component types, and
    # differing shapes (unknown versus known, or different dimensions) mention component shapes.
    with self.cached_session():
      q_a_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((),), shared_name="q_a")
      q_a_2 = data_flow_ops.PaddingFIFOQueue(
          15, dtypes_lib.float32, ((),), shared_name="q_a")
      q_a_1.queue_ref.op.run()
      with self.assertRaisesOpError("capacity"):
        q_a_2.queue_ref.op.run()

      q_b_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((),), shared_name="q_b")
      q_b_2 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.int32, ((),), shared_name="q_b")
      q_b_1.queue_ref.op.run()
      with self.assertRaisesOpError("component types"):
        q_b_2.queue_ref.op.run()

      q_c_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((),), shared_name="q_c")
      q_c_2 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_c")
      q_c_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_c_2.queue_ref.op.run()

      q_d_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_d")
      q_d_2 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((),), shared_name="q_d")
      q_d_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_d_2.queue_ref.op.run()

      q_e_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_e")
      q_e_2 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 4)], shared_name="q_e")
      q_e_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_e_2.queue_ref.op.run()

      q_f_1 = data_flow_ops.PaddingFIFOQueue(
          10, dtypes_lib.float32, ((),), shared_name="q_f")
      q_f_2 = data_flow_ops.PaddingFIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), ((), ()),
          shared_name="q_f")
      q_f_1.queue_ref.op.run()
      with self.assertRaisesOpError("component types"):
        q_f_2.queue_ref.op.run()

  def testSelectQueue(self):
    # [FUNCTION DOC]
    # Test: ten queues are built into a list. Twenty times a random index selects one via the
    # from-list constructor, a value is enqueued into the selected queue and dequeued back, asserting
    # it equals the original. Uses random numbers for the index.
    with self.cached_session():
      num_queues = 10
      qlist = []
      for _ in range(num_queues):
        qlist.append(
            data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),)))
      # Enqueue/Dequeue into a dynamically selected queue
      for _ in range(20):
        index = np.random.randint(num_queues)
        q = data_flow_ops.PaddingFIFOQueue.from_list(index, qlist)
        q.enqueue((10.,)).run()
        self.assertEqual(q.dequeue().eval(), 10.0)

  def testSelectQueueOutOfRange(self):
    # [FUNCTION DOC]
    # Test: selecting a queue from a two-element list with index three; dequeuing from the resulting
    # selected queue must raise an op error saying the index is not in the allowed range.
    with self.cached_session():
      q1 = data_flow_ops.PaddingFIFOQueue(10, dtypes_lib.float32, ((),))
      q2 = data_flow_ops.PaddingFIFOQueue(15, dtypes_lib.float32, ((),))
      enq_q = data_flow_ops.PaddingFIFOQueue.from_list(3, [q1, q2])
      with self.assertRaisesOpError("is not in"):
        enq_q.dequeue().eval()

  def _blockingDequeue(self, sess, dequeue_op):
    # [FUNCTION DOC]
    # HELPER: thread body for testResetOfBlockingOperation.
    # KNOWN ISSUE: the session parameter is accepted but never used; evaluation goes through the test
    # case helper rather than the passed session.
    # Evaluates the given dequeue operation and expects it to raise an op error saying it was
    # cancelled. Returns nothing.
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(dequeue_op)

  def _blockingDequeueMany(self, sess, dequeue_many_op):
    # [FUNCTION DOC]
    # HELPER: thread body for testResetOfBlockingOperation.
    # KNOWN ISSUE: the session parameter is accepted but never used; evaluation goes through the test
    # case helper rather than the passed session.
    # Evaluates the given dequeue-many operation and expects an op error saying it was cancelled.
    # Returns nothing.
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(dequeue_many_op)

  def _blockingEnqueue(self, sess, enqueue_op):
    # [FUNCTION DOC]
    # HELPER: thread body for testResetOfBlockingOperation.
    # KNOWN ISSUE: the session parameter is accepted but never used; evaluation goes through the test
    # case helper rather than the passed session.
    # Evaluates the given enqueue operation and expects an op error saying it was cancelled. Returns
    # nothing.
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(enqueue_op)

  def _blockingEnqueueMany(self, sess, enqueue_many_op):
    # [FUNCTION DOC]
    # HELPER: thread body for testResetOfBlockingOperation.
    # KNOWN ISSUE: the session parameter is accepted but never used; evaluation goes through the test
    # case helper rather than the passed session.
    # Evaluates the given batched enqueue operation and expects an op error saying it was cancelled.
    # Returns nothing.
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(enqueue_many_op)

  @test_util.run_deprecated_v1
  def testResetOfBlockingOperation(self):
    # [FUNCTION DOC]
    # Test: four threads are started, each blocking on one operation (dequeue and dequeue-many on an
    # empty queue, enqueue and batched enqueue on a full queue). After a short sleep the session is
    # closed, which must cancel all blocked operations so each thread observes a cancelled error; all
    # threads are then joined. Decorated to run in deprecated version 1 mode.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q_empty = data_flow_ops.PaddingFIFOQueue(5, dtypes_lib.float32, ((),))
      dequeue_op = q_empty.dequeue()
      dequeue_many_op = q_empty.dequeue_many(1)

      q_full = data_flow_ops.PaddingFIFOQueue(5, dtypes_lib.float32, ((),))
      sess.run(q_full.enqueue_many(([1.0, 2.0, 3.0, 4.0, 5.0],)))
      enqueue_op = q_full.enqueue((6.0,))
      enqueue_many_op = q_full.enqueue_many(([6.0],))

      threads = [
          self.checkedThread(
              self._blockingDequeue, args=(sess, dequeue_op)),
          self.checkedThread(
              self._blockingDequeueMany, args=(sess, dequeue_many_op)),
          self.checkedThread(
              self._blockingEnqueue, args=(sess, enqueue_op)),
          self.checkedThread(
              self._blockingEnqueueMany, args=(sess, enqueue_many_op))
      ]
      for t in threads:
        t.start()
      time.sleep(0.1)
      sess.close()  # Will cancel the blocked operations.
      for t in threads:
        t.join()

  def testBigEnqueueMany(self):
    # [FUNCTION DOC]
    # Test: a batched enqueue of ten elements into a capacity-five queue runs in a thread and must
    # block after filling the queue. The main thread checks, with sleeps, that the enqueue is still
    # unfinished while elements are dequeued until enough room exists, then that it completes and all
    # ten elements are eventually dequeued in order with the size shrinking by one each time.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(5, dtypes_lib.int32, ((),))
      elem = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      enq = q.enqueue_many((elem,))
      deq = q.dequeue()
      size_op = q.size()

      enq_done = []

      def blocking_enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body for testBigEnqueueMany.
        # Records a start marker in the shared flag list, evaluates the oversize batched enqueue (which
        # blocks until dequeues free space), then records a completion marker. Returns nothing.
        enq_done.append(False)
        # This will fill the queue and then block until enough dequeues happen.
        self.evaluate(enq)
        enq_done.append(True)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()

      # The enqueue should start and then block.
      results = []
      results.append(
          self.evaluate(deq))  # Will only complete after the enqueue starts.
      self.assertEqual(len(enq_done), 1)
      self.assertEqual(self.evaluate(size_op), 5)

      for _ in range(3):
        results.append(self.evaluate(deq))

      time.sleep(0.1)
      self.assertEqual(len(enq_done), 1)
      self.assertEqual(self.evaluate(size_op), 5)

      # This dequeue will unblock the thread.
      results.append(self.evaluate(deq))
      time.sleep(0.1)
      self.assertEqual(len(enq_done), 2)
      thread.join()

      for i in range(5):
        self.assertEqual(self.evaluate(size_op), 5 - i)
        results.append(self.evaluate(deq))
        self.assertEqual(self.evaluate(size_op), 5 - i - 1)

      self.assertAllEqual(elem, results)

  def testBigDequeueMany(self):
    # [FUNCTION DOC]
    # Test: a dequeue-many of four from a capacity-two queue runs in a thread and must block. Four
    # single enqueues are evaluated one at a time with sleeps, asserting the dequeue has produced no
    # results until the last one; then the thread is joined and the results equal the four values.
    with self.cached_session() as sess:
      q = data_flow_ops.PaddingFIFOQueue(2, dtypes_lib.int32, ((),))
      elem = np.arange(4, dtype=np.int32)
      enq_list = [q.enqueue((e,)) for e in elem]
      deq = q.dequeue_many(4)

      results = []

      def blocking_dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body for testBigDequeueMany.
        # Evaluates the dequeue-many of four (blocking until all four have been enqueued) and extends the
        # shared results list with the batch. Returns nothing.
        # Will only complete after 4 enqueues complete.
        results.extend(self.evaluate(deq))

      thread = self.checkedThread(target=blocking_dequeue)
      thread.start()
      # The dequeue should start and then block.
      for enq in enq_list:
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.assertEqual(len(results), 0)
        self.evaluate(enq)

      # Enough enqueued to unblock the dequeue
      thread.join()
      self.assertAllEqual(elem, results)

  def testDtypes(self):
    # [FUNCTION DOC]
    # Test: round trip for ten element types (floats, integers of several widths, unsigned byte,
    # boolean, complex). Random integer data of shape thirty-two by four by one hundred twenty-eight
    # is converted to each type (booleans from sign, complex from square roots), enqueued as one
    # batch and dequeued with dequeue-many of thirty-two; each component must equal its input.
    with self.cached_session() as sess:
      dtypes = [
          dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
          dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8, dtypes_lib.int64,
          dtypes_lib.bool, dtypes_lib.complex64, dtypes_lib.complex128
      ]
      shape = (32, 4, 128)
      q = data_flow_ops.PaddingFIFOQueue(32, dtypes, [shape[1:]] * len(dtypes))

      input_tuple = []
      for dtype in dtypes:
        np_dtype = dtype.as_numpy_dtype
        np_array = np.random.randint(-10, 10, shape)
        if dtype == dtypes_lib.bool:
          np_array = np_array > 0
        elif dtype in (dtypes_lib.complex64, dtypes_lib.complex128):
          np_array = np.sqrt(np_array.astype(np_dtype))
        else:
          np_array = np_array.astype(np_dtype)
        input_tuple.append(np_array)

      q.enqueue_many(input_tuple).run()

      output_tuple_t = q.dequeue_many(32)
      output_tuple = self.evaluate(output_tuple_t)

      for (input_elem, output_elem) in zip(input_tuple, output_tuple):
        self.assertAllEqual(input_elem, output_elem)

  def testUnknownRank(self):
    # [FUNCTION DOC]
    # Test: constructing a queue whose component shape has unknown rank must raise a value error
    # saying the shape must have a defined rank.
    with self.assertRaisesRegex(ValueError, "must have a defined rank"):
      data_flow_ops.PaddingFIFOQueue(32, [dtypes_lib.float32],
                                     [tensor_shape.TensorShape(None)])


class QueueFromListTest(test.TestCase):
  # [CLASS DOC]
  # Test case for selecting among several queues by an index tensor (the from-list constructor of
  # the base queue class) and for how the resulting queue's component shapes are merged from the
  # shapes of the candidate queues. It does not use sessions; only static shapes are checked.

  def testQueueFromListShapes(self):
    # [FUNCTION DOC]
    # Test: static shape merging for a single-component queue chosen from a list. For several lists of
    # candidate shapes the merged shape must be unknown rank if ranks differ, otherwise each
    # dimension is kept when all candidates agree and made unknown when they differ. Several
    # combinations are checked through a local comparison helper.
    which = constant_op.constant(1)

    def _cmp(expected, *shapes):
      # [FUNCTION DOC]
      # HELPER: comparison helper for testQueueFromListShapes.
      # Takes an expected shape (or nothing for unknown rank) followed by any number of candidate
      # shapes. Builds one float queue per candidate shape, selects among them with the index
      # constant defined in the enclosing test, and asserts the resulting first component shape equals
      # the expected one: unknown rank must match in rank, otherwise the dimension lists must be equal.
      # Returns nothing.
      qs = [
          data_flow_ops.PaddingFIFOQueue(10, [dtypes_lib.float32],
                                         [tensor_shape.TensorShape(s)])
          for s in shapes
      ]
      s_expected = tensor_shape.TensorShape(expected)
      s = data_flow_ops.QueueBase.from_list(which, qs).shapes[0]
      if s_expected.ndims is None:
        self.assertEqual(s_expected.ndims, s.ndims)
      else:
        self.assertEqual(s_expected.as_list(), s.as_list())

    _cmp(None, [1, None], [None])
    _cmp([None], [1], [2])
    _cmp([1, None], [1, 1], [1, 2])
    _cmp([1, None], [1, 1], [1, None])
    _cmp([None, None], [None, 1], [1, None])
    _cmp([1], [1], [1], [1])
    _cmp([None], [1], [None], [1])
    _cmp(None, [1, None], [1], [1])

  def testQueueFromListShapesMultipleComponents(self):
    # [FUNCTION DOC]
    # Test: shape merging for queues with two components. Three scenarios: all candidates unknown
    # length vectors (rank one kept, dimension unknown), a candidate with a different second
    # component rank (second component rank becomes unknown while the first stays an unknown vector)
    # and all candidates with fully known identical shapes (shapes preserved exactly).
    q_u_u = data_flow_ops.PaddingFIFOQueue(
        10, [dtypes_lib.float32, dtypes_lib.int32],
        [tensor_shape.TensorShape([None]), tensor_shape.TensorShape([None])])
    q_u_f = data_flow_ops.PaddingFIFOQueue(
        10, [dtypes_lib.float32, dtypes_lib.int32],
        [tensor_shape.TensorShape([None]), tensor_shape.TensorShape([1, 2])])
    q_f_f = data_flow_ops.PaddingFIFOQueue(
        10, [dtypes_lib.float32, dtypes_lib.int32],
        [tensor_shape.TensorShape([3, 4]), tensor_shape.TensorShape([1, 2])])
    which = constant_op.constant(1)

    s_cmp_1 = data_flow_ops.QueueBase.from_list(which,
                                                [q_u_u, q_u_u, q_u_u]).shapes
    self.assertEqual([1, 1], [x.ndims for x in s_cmp_1])
    self.assertEqual([None, None], [x.as_list()[0] for x in s_cmp_1])

    s_cmp_2 = data_flow_ops.QueueBase.from_list(which,
                                                [q_u_u, q_u_u, q_u_f]).shapes
    self.assertEqual([1, None], [x.ndims for x in s_cmp_2])
    self.assertEqual([None], s_cmp_2[0].as_list())

    s_cmp_3 = data_flow_ops.QueueBase.from_list(which, [q_f_f, q_f_f]).shapes
    self.assertEqual([2, 2], [x.ndims for x in s_cmp_3])
    self.assertEqual([[3, 4], [1, 2]], [x.as_list() for x in s_cmp_3])


if __name__ == "__main__":
  test.main()
