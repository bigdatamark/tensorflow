# [MODULE DOC]
# Unit tests for the first-in-first-out queue (FIFOQueue and its GPU-compatible variant) from the
# data flow ops module. Covers construction and the generated graph node definition, enqueue and
# dequeue of single elements and batches, shape and dtype validation, named (dictionary) components,
# closing semantics, blocking behaviour with threads, shared queues, operation timeouts, resource
# containers and a small throughput benchmark.
# Test classes: FIFOQueueTest and GPUCompatibleFIFOQueueTests run in both graph and eager modes;
# UnconvertedFIFOQueueTests and FIFOQueueParallelTests and FIFOQueueWithTimeoutTest run only under
# the version 1 graph and session API; FIFOQueueDictTest and QueueContainerTest build graphs
# directly. FIFOQueueBenchmark is a benchmark class.
# Import-time side effects: none beyond importing test dependencies; the file runs the test main
# entry point when executed as a script. This is a test module, not a helper module.
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

import gc
import random
import time

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


@test_util.run_all_in_graph_and_eager_modes
class FIFOQueueTest(test.TestCase):
  # [CLASS DOC]
  # Test case for FIFOQueue that runs every test in both graph and eager modes (via the class
  # decorator). Holds no state of its own; each test builds a fresh queue and uses the evaluate
  # helper of the test base class to run operations. Covers constructor node definitions,
  # enqueue/dequeue, shapes, dtypes, empty batches, queue-in-function use and type conversion.

  def testConstructor(self):
    # [FUNCTION DOC]
    # Test: builds a FIFOQueue of capacity 10 with one float32 component and name Q inside a fresh
    # graph, then asserts the queue reference is a tensor and that its node definition is the
    # FIFOQueueV2 op with the expected component types, empty shapes, capacity, container and
    # shared name attributes. Returns nothing; fails by assertion.
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, name="Q")
    self.assertTrue(isinstance(q.queue_ref, tensor.Tensor))
    self.assertProtoEquals("""
      name:'Q' device: "/device:CPU:*" op:'FIFOQueueV2'
      attr { key: 'component_types' value { list { type: DT_FLOAT } } }
      attr { key: 'shapes' value { list {} } }
      attr { key: 'capacity' value { i: 10 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: '' } }
      """, q.queue_ref.op.node_def)

  def testMultiQueueConstructor(self):
    # [FUNCTION DOC]
    # Test: builds a capacity-5 queue with two components (int32 and float32) and shared name foo
    # in a fresh graph and asserts the node definition lists both component types, empty shapes,
    # capacity 5 and shared name foo.
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32),
          shared_name="foo",
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, tensor.Tensor))
    self.assertProtoEquals("""
      name:'Q' device: "/device:CPU:*" op:'FIFOQueueV2'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list {} } }
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: 'foo' } }
      """, q.queue_ref.op.node_def)

  def testConstructorWithShapes(self):
    # [FUNCTION DOC]
    # Test: builds a two-component queue with explicit shapes (1x1x2x3 and 5x8) and asserts the
    # node definition carries both shapes as dimension lists along with the expected types,
    # capacity and empty container/shared name.
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32),
          shapes=(tensor_shape.TensorShape([1, 1, 2, 3]),
                  tensor_shape.TensorShape([5, 8])),
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, tensor.Tensor))
    self.assertProtoEquals("""
      name:'Q' device: "/device:CPU:*" op:'FIFOQueueV2'
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
    # Test: enqueues a single float32 element into a capacity-10 queue and evaluates the operation.
    # Expects no error; nothing else is asserted.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
    self.evaluate(q.enqueue((10.0,)))

  def testEnqueueHalf(self):
    # [FUNCTION DOC]
    # Test: same as the plain enqueue test but for a float16 queue; expects the enqueue to succeed.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float16)
    self.evaluate(q.enqueue((10.0,)))

  def testEnqueueWithShape(self):
    # [FUNCTION DOC]
    # Test: queue with fixed element shape 3x2. Enqueues a matching element successfully, then
    # expects a value error when enqueueing an element of shape 2x3, and asserts the queue size is
    # still 1 (the bad enqueue was rejected at graph construction).
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shapes=(3, 2))
    self.evaluate(q.enqueue(([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],)))
    with self.assertRaises(ValueError):
      q.enqueue(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],))
    self.assertEqual(1, self.evaluate(q.size()))

  def testEnqueueManyWithShape(self):
    # [FUNCTION DOC]
    # Test: two-component int32 queue with shapes scalar and length-2 vector. Enqueues four
    # elements in one batch call and asserts the size becomes 4.
    q = data_flow_ops.FIFOQueue(
        10, [dtypes_lib.int32, dtypes_lib.int32], shapes=[(), (2,)])
    self.evaluate(
        q.enqueue_many([[1, 2, 3, 4], [[1, 1], [2, 2], [3, 3], [4, 4]]]))
    self.assertEqual(4, self.evaluate(q.size()))

  def testMultipleDequeues(self):
    # [FUNCTION DOC]
    # Test: enqueues three scalars in one batch, evaluates three separate dequeue operations in a
    # single call and asserts the set of results equals the three enqueued values (order is not
    # asserted because the dequeues may run in any order).
    q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
    self.evaluate(q.enqueue_many([[1, 2, 3]]))
    a, b, c = self.evaluate([q.dequeue(), q.dequeue(), q.dequeue()])
    self.assertAllEqual(set([1, 2, 3]), set([a, b, c]))

  def testQueuesDontShare(self):
    # [FUNCTION DOC]
    # Test: creates two independent queues without shared names, enqueues 1 and 2 respectively and
    # asserts each queue dequeues only its own value, proving they do not share storage.
    q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
    self.evaluate(q.enqueue(1))
    q2 = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
    self.evaluate(q2.enqueue(2))
    self.assertAllEqual(self.evaluate(q2.dequeue()), 2)
    self.assertAllEqual(self.evaluate(q.dequeue()), 1)

  def testQueueInFunction(self):
    # [FUNCTION DOC]
    # Test: defines a module that owns one queue and lazily creates a second queue inside a traced
    # function which enqueues, moves an element between queues and returns a constant. Runs it,
    # then asserts the queues contain the expected values (2 and 5). When eager, deletes the
    # module, forces garbage collection and asserts that destroying each queue resource handle
    # raises a not-found error because the resources were cleaned up.

    class _M(module.Module):
      # [CLASS DOC]
      # HELPER: small module defined inside the queue-in-function test.
      # Holds a queue created eagerly in the constructor and a second queue that starts as nothing
      # and is created on first call of the traced method. Used to check that queues owned by a
      # module work inside traced functions and are released when the module is deleted.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: constructor of the local test module.
        # Creates the first queue (capacity 10, scalar int32 elements) and sets the second queue to
        # nothing. Takes no inputs and returns nothing.
        self.q1 = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
        self.q2 = None

      @def_function.function
      def uses_queues(self, x):
        # [FUNCTION DOC]
        # HELPER: traced method of the local test module.
        # Creates the second queue on first use, enqueues the given value x and x plus three into
        # it, dequeues one element from it and enqueues that into the first queue, and returns the
        # constant one. Side effects: mutates the module attribute for the second queue and adds
        # queue operations to the traced graph.
        if self.q2 is None:
          self.q2 = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
        self.q2.enqueue(x)
        self.q2.enqueue(x + 3)
        self.q1.enqueue(self.q2.dequeue())
        return 1

    m = _M()
    self.evaluate(m.uses_queues(constant_op.constant(2)))
    self.assertAllEqual(2, self.evaluate(m.q1.dequeue()))
    self.assertAllEqual(5, self.evaluate(m.q2.dequeue()))
    if context.executing_eagerly():
      q1_handle = m.q1.queue_ref
      q2_handle = m.q2.queue_ref
      del m
      gc.collect()
      # If executing eagerly, deleting the Module should clean up the queue
      # resources.
      with self.assertRaisesRegex(errors_impl.NotFoundError,
                                  r"Resource .* does not exist."):
        gen_resource_variable_ops.destroy_resource_op(
            q1_handle, ignore_lookup_error=False)
      with self.assertRaisesRegex(errors_impl.NotFoundError,
                                  r"Resource .* does not exist."):
        gen_resource_variable_ops.destroy_resource_op(
            q2_handle, ignore_lookup_error=False)

  def testEnqueueDictWithoutNames(self):
    # [FUNCTION DOC]
    # Test: on a queue without component names, enqueue and enqueue-many given a mapping must raise
    # a value error mentioning that the queue must have names.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
    with self.assertRaisesRegex(ValueError, "must have names"):
      q.enqueue({"a": 12.0})
    with self.assertRaisesRegex(ValueError, "must have names"):
      q.enqueue_many({"a": [12.0, 13.0]})

  def testDequeue(self):
    # [FUNCTION DOC]
    # Test: enqueues three floats one at a time and asserts they come out in the same order
    # (first-in-first-out), each as a one-element result.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
    elems = [10.0, 20.0, 30.0]

    for x in elems:
      self.evaluate(q.enqueue((x,)))

    for i in range(len(elems)):
      vals = self.evaluate(q.dequeue())
      self.assertEqual([elems[i]], vals)

  def testDequeueHalf(self):
    # [FUNCTION DOC]
    # Test: same as the dequeue ordering test but with a float16 queue.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float16)
    elems = [10.0, 20.0, 30.0]

    for x in elems:
      self.evaluate(q.enqueue((x,)))

    for i in range(len(elems)):
      vals = self.evaluate(q.dequeue())
      self.assertEqual([elems[i]], vals)

  def testMultiEnqueueAndDequeue(self):
    # [FUNCTION DOC]
    # Test: two-component queue (int32, float32); enqueues three tuples and asserts each dequeue
    # returns the two components of the matching tuple in order.
    q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.float32))
    elems = [(5, 10.0), (10, 20.0), (15, 30.0)]

    for x in elems:
      self.evaluate(q.enqueue(x))

    for i in range(len(elems)):
      x_val, y_val = self.evaluate(q.dequeue())
      x, y = elems[i]
      self.assertEqual([x], x_val)
      self.assertEqual([y], y_val)

  def testQueueSizeEmpty(self):
    # [FUNCTION DOC]
    # Test: a freshly created queue reports size zero.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
    self.assertEqual([0], self.evaluate(q.size()))

  def testQueueSizeAfterEnqueueAndDequeue(self):
    # [FUNCTION DOC]
    # Test: asserts the size tensor has scalar static shape, then that size is 1 after one enqueue
    # and 0 after the following dequeue.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
    self.assertEqual([], q.size().get_shape())

    self.evaluate(q.enqueue((10.0,)))
    self.assertEqual(1, self.evaluate(q.size()))
    self.evaluate(q.dequeue())
    self.assertEqual(0, self.evaluate(q.size()))

  def testEnqueueMany(self):
    # [FUNCTION DOC]
    # Test: enqueues a batch of four floats twice and asserts eight dequeues return the elements in
    # order, repeating the batch pattern.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
    elems = [10.0, 20.0, 30.0, 40.0]
    self.evaluate(q.enqueue_many((elems,)))
    self.evaluate(q.enqueue_many((elems,)))

    for i in range(8):
      vals = self.evaluate(q.dequeue())
      self.assertEqual([elems[i % 4]], vals)

  def testEmptyEnqueueMany(self):
    # [FUNCTION DOC]
    # Test: enqueue-many of an empty tensor (shape 0x2x3) must succeed and leave the queue size at
    # zero, both before and after.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
    empty_t = constant_op.constant(
        [], dtype=dtypes_lib.float32, shape=[0, 2, 3])

    self.assertEqual([0], self.evaluate(q.size()))
    self.evaluate(q.enqueue_many((empty_t,)))
    self.assertEqual([0], self.evaluate(q.size()))

  def testEmptyDequeueMany(self):
    # [FUNCTION DOC]
    # Test: on a scalar-element queue, dequeue-many of zero elements returns an empty result both
    # when the queue is empty and when it holds one element.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shapes=())

    self.assertEqual([], self.evaluate(q.dequeue_many(0)).tolist())
    self.evaluate(q.enqueue((10.0,)))
    self.assertEqual([], self.evaluate(q.dequeue_many(0)).tolist())

  def testEmptyDequeueUpTo(self):
    # [FUNCTION DOC]
    # Test: same as the empty dequeue-many test but using dequeue-up-to with count zero.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shapes=())

    self.assertEqual([], self.evaluate(q.dequeue_up_to(0)).tolist())
    self.evaluate(q.enqueue((10.0,)))
    self.assertEqual([], self.evaluate(q.dequeue_up_to(0)).tolist())

  def testEmptyDequeueManyWithNoShape(self):
    # [FUNCTION DOC]
    # Test: on a queue with unspecified element shapes, dequeue-many of zero elements must fail at
    # run time with an operation error mentioning the specified shapes.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
    # Expect the operation to fail due to the shape not being constrained.
    with self.assertRaisesOpError("specified shapes"):
      self.evaluate(q.dequeue_many(0))

  def testMultiEnqueueMany(self):
    # [FUNCTION DOC]
    # Test: two-component queue (float32, int32 vector); enqueues a four-element batch twice and
    # asserts eight dequeues return the float and int components in the expected repeating order.
    q = data_flow_ops.FIFOQueue(10, (dtypes_lib.float32, dtypes_lib.int32))
    float_elems = [10.0, 20.0, 30.0, 40.0]
    int_elems = [[1, 2], [3, 4], [5, 6], [7, 8]]

    self.evaluate(q.enqueue_many((float_elems, int_elems)))
    self.evaluate(q.enqueue_many((float_elems, int_elems)))

    for i in range(8):
      float_val, int_val = self.evaluate(q.dequeue())
      self.assertEqual(float_elems[i % 4], float_val)
      self.assertAllEqual(int_elems[i % 4], int_val)

  def testDequeueMany(self):
    # [FUNCTION DOC]
    # Test: enqueues ten floats and asserts two successive dequeue-many calls of four return the
    # first and second groups of four in order.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
    elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    self.evaluate(q.enqueue_many((elems,)))

    self.assertAllEqual(elems[0:4], self.evaluate(q.dequeue_many(4)))
    self.assertAllEqual(elems[4:8], self.evaluate(q.dequeue_many(4)))

  def testDequeueUpToNoBlocking(self):
    # [FUNCTION DOC]
    # Test: same as the dequeue-many test but with dequeue-up-to; since enough elements exist it
    # does not block and returns full groups of four.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
    elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    self.evaluate(q.enqueue_many((elems,)))

    self.assertAllEqual(elems[0:4], self.evaluate(q.dequeue_up_to(4)))
    self.assertAllEqual(elems[4:8], self.evaluate(q.dequeue_up_to(4)))

  def testMultiDequeueMany(self):
    # [FUNCTION DOC]
    # Test: two-component queue with shapes scalar and length-2. Enqueues ten elements, dequeues
    # batches of four twice and then one single element, asserting values and that result static
    # shapes match the evaluated shapes.
    q = data_flow_ops.FIFOQueue(
        10, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (2,)))
    float_elems = [
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
    ]
    int_elems = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                 [15, 16], [17, 18], [19, 20]]

    self.evaluate(q.enqueue_many((float_elems, int_elems)))

    dequeued_t = q.dequeue_many(4)
    float_val, int_val = self.evaluate(dequeued_t)
    self.assertAllEqual(float_elems[0:4], float_val)
    self.assertAllEqual(int_elems[0:4], int_val)
    self.assertEqual(float_val.shape, dequeued_t[0].get_shape())
    self.assertEqual(int_val.shape, dequeued_t[1].get_shape())

    float_val, int_val = self.evaluate(q.dequeue_many(4))
    self.assertAllEqual(float_elems[4:8], float_val)
    self.assertAllEqual(int_elems[4:8], int_val)

    dequeued_single_t = q.dequeue()
    float_val, int_val = self.evaluate(dequeued_single_t)
    self.assertAllEqual(float_elems[8], float_val)
    self.assertAllEqual(int_elems[8], int_val)
    self.assertEqual(float_val.shape, dequeued_single_t[0].get_shape())
    self.assertEqual(int_val.shape, dequeued_single_t[1].get_shape())

  def testMultiDequeueUpToNoBlocking(self):
    # [FUNCTION DOC]
    # Test: like the multi-component dequeue-many test but with dequeue-up-to; asserts returned
    # values for two batches and, in graph mode only, that the static batch dimension is unknown.
    q = data_flow_ops.FIFOQueue(
        10, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (2,)))
    float_elems = [
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
    ]
    int_elems = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                 [15, 16], [17, 18], [19, 20]]

    self.evaluate(q.enqueue_many((float_elems, int_elems)))

    dequeued_t = q.dequeue_up_to(4)
    float_val, int_val = self.evaluate(dequeued_t)
    self.assertAllEqual(float_elems[0:4], float_val)
    self.assertAllEqual(int_elems[0:4], int_val)
    if not context.executing_eagerly():
      self.assertEqual([None], dequeued_t[0].get_shape().as_list())
      self.assertEqual([None, 2], dequeued_t[1].get_shape().as_list())

    float_val, int_val = self.evaluate(q.dequeue_up_to(4))
    self.assertAllEqual(float_elems[4:8], float_val)
    self.assertAllEqual(int_elems[4:8], int_val)

  def testHighDimension(self):
    # [FUNCTION DOC]
    # Test: queue of int32 elements with shape 4x4x4x4; enqueues ten such elements in one call and
    # asserts dequeue-many of ten returns identical data.
    q = data_flow_ops.FIFOQueue(10, dtypes_lib.int32, (4, 4, 4, 4))
    elems = np.array([[[[[x] * 4] * 4] * 4] * 4 for x in range(10)], np.int32)

    self.evaluate(q.enqueue_many((elems,)))
    self.assertAllEqual(self.evaluate(q.dequeue_many(10)), elems)

  def testEnqueueWrongShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second shape is written as a parenthesised number, not a one-element tuple,
    # so it is the integer 2 rather than a vector shape; the test still passes because any mismatch
    # raises, but it does not clearly test the intended vector shape.
    # Test: two-component int32 queue; enqueue with a wrong-shaped first component and enqueue-many
    # with a scalar batch must each raise a value error.
    q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32), ((),
                                                                           (2)))

    with self.assertRaises(ValueError):
      q.enqueue(([1, 2], [2, 2]))

    with self.assertRaises(ValueError):
      q.enqueue_many((7, [[1, 2], [3, 4], [5, 6]]))

  def testEnqueueManyEmptyTypeConversion(self):
    # [FUNCTION DOC]
    # Test: queue with int32 and float32 scalar components. Inside a traced function, enqueue-many
    # of two empty lists is built and the test asserts the empty inputs were converted to int32 and
    # float32 (the operation inputs after the queue handle). Then calls the function once.
    q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.float32), (
        (), ()))

    @def_function.function
    def _f():
      # [FUNCTION DOC]
      # HELPER: traced function of the empty-batch conversion test.
      # Builds an enqueue-many of two empty lists and asserts the operation inputs after the queue
      # handle have dtypes int32 and float32. Takes no inputs, returns nothing; assertion failure
      # on mismatch.
      enq = q.enqueue_many(([], []))
      self.assertEqual(dtypes_lib.int32, enq.inputs[1].dtype)
      self.assertEqual(dtypes_lib.float32, enq.inputs[2].dtype)

    _f()

  def testEnqueueWrongType(self):
    # [FUNCTION DOC]
    # Test: two-component queue (int32, float32) with scalar shapes. Inside a traced function,
    # enqueue and enqueue-many with an int32 placeholder for the float32 component must raise a
    # value error. Calls the traced function once.
    q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.float32), (
        (), ()))

    @def_function.function
    def _f():
      # [FUNCTION DOC]
      # HELPER: traced function of the wrong-type test.
      # Asserts that enqueue and enqueue-many with int32 placeholders for the float32 component
      # each raise a value error. Takes no inputs, returns nothing.
      with self.assertRaises(ValueError):
        q.enqueue((array_ops.placeholder(dtypes_lib.int32),
                   array_ops.placeholder(dtypes_lib.int32)))

      with self.assertRaises(ValueError):
        q.enqueue_many((array_ops.placeholder(dtypes_lib.int32),
                        array_ops.placeholder(dtypes_lib.int32)))

    _f()


@test_util.run_all_in_graph_and_eager_modes
class GPUCompatibleFIFOQueueTests(test.TestCase):
  # [CLASS DOC]
  # Test case for the GPU-compatible FIFO queue variant, run in both graph and eager modes. Tests
  # execute inside a GPU-preferring device scope so queue elements may live on the GPU.

  def testEnqueueWithShape(self):
    # [FUNCTION DOC]
    # Test: same scenario as the CPU shape test, but with the GPU-compatible queue inside a GPU
    # device scope: matching shape enqueue succeeds, wrong shape raises a value error, size is 1.
    with test_util.use_gpu():
      q = data_flow_ops.GPUCompatibleFIFOQueue(
          10, dtypes_lib.float32, shapes=(3, 2))
      self.evaluate(q.enqueue(([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],)))
      with self.assertRaises(ValueError):
        q.enqueue(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],))
      self.assertEqual(1, self.evaluate(q.size()))

  def testEnqueueDequeue(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the device assertion always compares the dequeued tensor device against the
    # device of the first element (index zero) instead of the element at the current index; this
    # works only because all elements are copied to the same device.
    # Test: creates three constants copied to the GPU via an identity operation, enqueues them in a
    # GPU-compatible queue, then dequeues each, asserting it sits on the same device as the inputs
    # and has the expected value, in order.
    with test_util.use_gpu():
      q = data_flow_ops.GPUCompatibleFIFOQueue(10, dtypes_lib.float32)
      elems_numpy = [10.0, 20.0, 30.0]
      # The identity ensures constants are copied to the GPU immediately
      elems = [array_ops.identity(constant_op.constant(x))
               for x in elems_numpy]

      for x in elems:
        self.evaluate(q.enqueue((x,)))

      for i in range(len(elems)):
        dequeued_tensor = q.dequeue()
        self.assertEqual(elems[0].device, dequeued_tensor.device)
        vals = self.evaluate(dequeued_tensor)
        self.assertEqual([elems_numpy[i]], vals)


@test_util.run_v1_only(
    "These tests can likely run in 2.x with some fixes, but have not been "
    "converted yet. Currently they hold on to operations and rely on "
    "re-running them; for eager compatibility we need to 're-create' the op "
    "each time.")
class UnconvertedFIFOQueueTests(test.TestCase):
  # [CLASS DOC]
  # Collection of FIFOQueue tests that are restricted to the version 1 graph API (see the class
  # decorator) because they keep operations around and re-run them, which does not work eagerly.
  # Covers dequeue-many with tensor counts, closed queues, shared queues, incompatible shared
  # queue errors, queue selection, dtypes, named components and runtime shape mismatches.

  def testDequeueManyWithTensorParameter(self):
    # [FUNCTION DOC]
    # Test (graph mode only): a first queue holds 100 random counts; a second queue holds the sum
    # of those counts in random integers. Builds a dequeue-many whose count is itself dequeued from
    # the first queue, runs both enqueues, then repeatedly evaluates it and asserts the
    # concatenated results equal the original element list.
    with self.cached_session():
      # Define a first queue that contains integer counts.
      dequeue_counts = [random.randint(1, 10) for _ in range(100)]
      count_q = data_flow_ops.FIFOQueue(100, dtypes_lib.int32, ())
      enqueue_counts_op = count_q.enqueue_many((dequeue_counts,))
      total_count = sum(dequeue_counts)

      # Define a second queue that contains total_count elements.
      elems = [random.randint(0, 100) for _ in range(total_count)]
      q = data_flow_ops.FIFOQueue(total_count, dtypes_lib.int32, ())
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
    # Test: enqueues four floats, closes the queue, dequeues all four successfully, then expects a
    # further dequeue to raise an out-of-range error saying the queue is closed with insufficient
    # elements.
    with self.cached_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
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

  def testDoesNotLoseValue(self):
    # [FUNCTION DOC]
    # Test: capacity-1 queue with one element; evaluating the size operation 500 times must always
    # report 1, i.e. reading the size never consumes the value.
    with self.cached_session():
      q = data_flow_ops.FIFOQueue(1, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      size_t = q.size()

      enqueue_op.run()
      for _ in range(500):
        self.assertEqual(self.evaluate(size_t), [1])

  def testSharedQueueSameSession(self):
    # [FUNCTION DOC]
    # Test: two queue objects created with the same shared name in one session behave as one queue.
    # Enqueue through one and dequeue through the other, asserting sizes reported by both stay
    # consistent (1, 0, 1, 0) and values (10 and 20) pass through in order.
    with self.cached_session():
      q1 = data_flow_ops.FIFOQueue(
          1, dtypes_lib.float32, shared_name="shared_queue")
      q1.enqueue((10.0,)).run()

      q2 = data_flow_ops.FIFOQueue(
          1, dtypes_lib.float32, shared_name="shared_queue")

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
    # Test: for six shared names, creates a first queue and a second incompatible queue (different
    # capacity, component types, shapes specified versus not, differing shapes, or differing number
    # of components). Running the first queue creation succeeds; running the second must raise an
    # operation error mentioning capacity, component types or component shapes as appropriate.
    with self.cached_session():
      q_a_1 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shared_name="q_a")
      q_a_2 = data_flow_ops.FIFOQueue(15, dtypes_lib.float32, shared_name="q_a")
      q_a_1.queue_ref.op.run()
      with self.assertRaisesOpError("capacity"):
        q_a_2.queue_ref.op.run()

      q_b_1 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shared_name="q_b")
      q_b_2 = data_flow_ops.FIFOQueue(10, dtypes_lib.int32, shared_name="q_b")
      q_b_1.queue_ref.op.run()
      with self.assertRaisesOpError("component types"):
        q_b_2.queue_ref.op.run()

      q_c_1 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shared_name="q_c")
      q_c_2 = data_flow_ops.FIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_c")
      q_c_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_c_2.queue_ref.op.run()

      q_d_1 = data_flow_ops.FIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_d")
      q_d_2 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shared_name="q_d")
      q_d_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_d_2.queue_ref.op.run()

      q_e_1 = data_flow_ops.FIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_e")
      q_e_2 = data_flow_ops.FIFOQueue(
          10, dtypes_lib.float32, shapes=[(1, 1, 2, 4)], shared_name="q_e")
      q_e_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_e_2.queue_ref.op.run()

      q_f_1 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shared_name="q_f")
      q_f_2 = data_flow_ops.FIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32), shared_name="q_f")
      q_f_1.queue_ref.op.run()
      with self.assertRaisesOpError("component types"):
        q_f_2.queue_ref.op.run()

  def testSelectQueue(self):
    # [FUNCTION DOC]
    # Test: builds ten queues, then twenty times picks a random index, builds a queue from the list
    # selected by that index, enqueues 10.0 and asserts dequeuing gives 10.0.
    with self.cached_session():
      num_queues = 10
      qlist = []
      for _ in range(num_queues):
        qlist.append(data_flow_ops.FIFOQueue(10, dtypes_lib.float32))
      # Enqueue/Dequeue into a dynamically selected queue
      for _ in range(20):
        index = np.random.randint(num_queues)
        q = data_flow_ops.FIFOQueue.from_list(index, qlist)
        q.enqueue((10.,)).run()
        self.assertEqual(q.dequeue().eval(), 10.0)

  def testSelectQueueOutOfRange(self):
    # [FUNCTION DOC]
    # Test: builds a selector queue from a list of two queues with index 3 (out of range); running
    # dequeue on it must raise an operation error saying the index is not in range.
    with self.cached_session():
      q1 = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      q2 = data_flow_ops.FIFOQueue(15, dtypes_lib.float32)
      enq_q = data_flow_ops.FIFOQueue.from_list(3, [q1, q2])
      with self.assertRaisesOpError("is not in"):
        enq_q.dequeue().eval()

  def testDtypes(self):
    # [FUNCTION DOC]
    # Test: queue with eleven component types (floats, ints, unsigned, bool, complex), each with a
    # 4x128 element shape. Generates random numpy batches of 32 (boolean from positivity, complex
    # from square roots), enqueues them in one call, dequeues 32 and asserts every component
    # matches. The session variable is unused.
    with self.cached_session() as sess:
      dtypes = [
          dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
          dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8, dtypes_lib.int64,
          dtypes_lib.uint16, dtypes_lib.bool, dtypes_lib.complex64,
          dtypes_lib.complex128
      ]
      shape = (32, 4, 128)
      q = data_flow_ops.FIFOQueue(32, dtypes, [shape[1:]] * len(dtypes))

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

  def testEnqueueDequeueOneComponent(self):
    # [FUNCTION DOC]
    # Test: single-component queue named f. Asserts enqueue and enqueue-many reject non-mapping
    # inputs (error about enqueueing a dictionary) and mappings whose keys do not match the names
    # (error about matching names); then enqueues valid mappings (three singles and a batch of
    # two), dequeues one and a batch of two twice and asserts the values 10, then 20 and 30, then
    # 40 and 50.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(
          10, dtypes_lib.float32, shapes=((),), names="f")
      # Verify that enqueue() checks that when using names we must enqueue a
      # dictionary.
      with self.assertRaisesRegex(ValueError, "enqueue a dictionary"):
        enqueue_op = q.enqueue(10.0)
      with self.assertRaisesRegex(ValueError, "enqueue a dictionary"):
        enqueue_op = q.enqueue((10.0,))
      # The dictionary keys must match the queue component names.
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({})
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({"x": 12})
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({"f": 10.0, "s": "aa"})
      enqueue_op = q.enqueue({"f": 10.0})
      enqueue_op2 = q.enqueue({"f": 20.0})
      enqueue_op3 = q.enqueue({"f": 30.0})
      # Verify that enqueue_many() checks that when using names we must enqueue
      # a dictionary.
      with self.assertRaisesRegex(ValueError, "enqueue a dictionary"):
        enqueue_op4 = q.enqueue_many([40.0, 50.0])
      # The dictionary keys must match the queue component names.
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({})
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({"x": 12})
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({"f": [40.0, 50.0], "s": ["aa", "bb"]})
      enqueue_op4 = q.enqueue_many({"f": [40.0, 50.0]})
      dequeue = q.dequeue()
      dequeue_2 = q.dequeue_many(2)
      self.evaluate(enqueue_op)
      self.evaluate(enqueue_op2)
      self.evaluate(enqueue_op3)
      self.evaluate(enqueue_op4)
      f = sess.run(dequeue["f"])
      self.assertEqual(10.0, f)
      f = sess.run(dequeue_2["f"])
      self.assertEqual([20.0, 30.0], list(f))
      f = sess.run(dequeue_2["f"])
      self.assertEqual([40.0, 50.0], list(f))

  def testEnqueueDequeueMultipleComponent(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: several checks use a truth assertion with two arguments (the second argument is
    # treated as the failure message), so those checks always pass and never compare the float and
    # string values of the second and third dequeue batches; only the integer components are really
    # verified.
    # Test: three-component queue named f, i, s. Asserts non-mapping inputs and mappings with
    # missing, wrong or extra keys are rejected for enqueue and enqueue-many, then enqueues valid
    # mappings, dequeues one element and two batches of two and checks the values as described.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(
          10, (dtypes_lib.float32, dtypes_lib.int32, dtypes_lib.string),
          shapes=((), (), ()),
          names=("f", "i", "s"))
      # Verify that enqueue() checks that when using names we must enqueue a
      # dictionary.
      with self.assertRaisesRegex(ValueError, "enqueue a dictionary"):
        enqueue_op = q.enqueue((10.0, 123, "aa"))
      # The dictionary keys must match the queue component names.
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({})
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({"x": 10.0})
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({"i": 12, "s": "aa"})
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op = q.enqueue({"i": 123, "s": "aa", "f": 10.0, "x": 10.0})
      enqueue_op = q.enqueue({"i": 123, "s": "aa", "f": 10.0})
      enqueue_op2 = q.enqueue({"i": 124, "s": "bb", "f": 20.0})
      enqueue_op3 = q.enqueue({"i": 125, "s": "cc", "f": 30.0})
      # Verify that enqueue_many() checks that when using names we must enqueue
      # a dictionary.
      with self.assertRaisesRegex(ValueError, "enqueue a dictionary"):
        enqueue_op4 = q.enqueue_many(([40.0, 50.0], [126, 127], ["dd", "ee"]))
      # The dictionary keys must match the queue component names.
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({})
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({"x": [10.0, 20.0]})
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({"i": [12, 12], "s": ["aa", "bb"]})
      with self.assertRaisesRegex(ValueError, "match names of Queue"):
        enqueue_op4 = q.enqueue_many({
            "f": [40.0, 50.0],
            "i": [126, 127],
            "s": ["dd", "ee"],
            "x": [1, 2]
        })
      enqueue_op4 = q.enqueue_many({
          "f": [40.0, 50.0],
          "i": [126, 127],
          "s": ["dd", "ee"]
      })
      dequeue = q.dequeue()
      dequeue_2 = q.dequeue_many(2)
      self.evaluate(enqueue_op)
      self.evaluate(enqueue_op2)
      self.evaluate(enqueue_op3)
      self.evaluate(enqueue_op4)
      i, f, s = sess.run([dequeue["i"], dequeue["f"], dequeue["s"]])
      self.assertEqual(123, i)
      self.assertEqual(10.0, f)
      self.assertEqual(compat.as_bytes("aa"), s)
      i, f, s = sess.run([dequeue_2["i"], dequeue_2["f"], dequeue_2["s"]])
      self.assertEqual([124, 125], list(i))
      self.assertTrue([20.0, 30.0], list(f))
      self.assertTrue([compat.as_bytes("bb"), compat.as_bytes("cc")], list(s))
      i, f, s = sess.run([dequeue_2["i"], dequeue_2["f"], dequeue_2["s"]])
      self.assertEqual([126, 127], list(i))
      self.assertTrue([40.0, 50.0], list(f))
      self.assertTrue([compat.as_bytes("dd"), compat.as_bytes("ee")], list(s))

  def testBatchSizeMismatch(self):
    # [FUNCTION DOC]
    # Test: three-component queue; enqueue-many with components of different batch sizes (lists, or
    # lists mixed with placeholders of unknown shape) must raise a value error in each variant.
    q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32,
                                     dtypes_lib.int32), ((), (), ()))

    with self.assertRaises(ValueError):
      q.enqueue_many(([1, 2, 3], [1, 2], [1, 2, 3]))

    with self.assertRaises(ValueError):
      q.enqueue_many(
          ([1, 2, 3], [1, 2], array_ops.placeholder(dtypes_lib.int32)))

    with self.assertRaises(ValueError):
      q.enqueue_many(
          (array_ops.placeholder(dtypes_lib.int32), [1, 2], [1, 2, 3]))

  def testEnqueueWrongShapeAtRuntime(self):
    # [FUNCTION DOC]
    # Test: queue with component shapes 2x2 and 3x3. Enqueues a valid first component with a
    # placeholder second component fed a 3x4 array; the run must raise an invalid-argument error
    # saying expected 3x3 but got 3x4.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32), (
          (2, 2), (3, 3)))
      elems_ok = np.array([1] * 4).reshape((2, 2)).astype(np.int32)
      elems_bad = array_ops.placeholder(dtypes_lib.int32)
      enqueue_op = q.enqueue((elems_ok, elems_bad))
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  r"Expected \[3,3\], got \[3,4\]"):
        sess.run([enqueue_op],
                 feed_dict={elems_bad: np.array([1] * 12).reshape((3, 4))})

  def testEnqueueDequeueManyWrongShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the final evaluation of the dequeue operation sits inside the same assertion
    # block after the run that is expected to raise, so it is never reached; the dequeue part of
    # the test does not execute.
    # Test: queue with shapes 2x2 and 3x3. Enqueue-many with a second component fed shape 2x3x4 at
    # run time must raise an invalid-argument error about a shape mismatch in component 1.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.int32), (
          (2, 2), (3, 3)))
      elems_ok = np.array([1] * 8).reshape((2, 2, 2)).astype(np.int32)
      elems_bad = array_ops.placeholder(dtypes_lib.int32)
      enqueue_op = q.enqueue_many((elems_ok, elems_bad))
      dequeued_t = q.dequeue_many(2)
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Shape mismatch in tuple component 1. "
          r"Expected \[2,3,3\], got \[2,3,4\]"):
        sess.run([enqueue_op],
                 feed_dict={elems_bad: np.array([1] * 24).reshape((2, 3, 4))})
        self.evaluate(dequeued_t)


@test_util.run_v1_only(
    "These tests are heavily reliant on Session for parallelism. We can likely "
    "convert them to run in 2.x/eager, but it may be difficult.")
class FIFOQueueParallelTests(test.TestCase):
  # [CLASS DOC]
  # FIFOQueue tests restricted to the version 1 graph API that rely on sessions and multiple
  # threads: parallel producers and consumers, blocking enqueue/dequeue, closing a queue while
  # operations are blocked, and cancellation by closing the session. Many tests sleep briefly to
  # let threads block, which makes them timing dependent.

  def testParallelEnqueue(self):
    # [FUNCTION DOC]
    # Test (graph mode only): makes the default graph thread-local for device stacks, enqueues ten
    # distinct floats from ten threads (one element each), then dequeues all on one thread and
    # asserts the results contain the same items regardless of order.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        # [FUNCTION DOC]
        # HELPER: thread body for the parallel enqueue test.
        # Evaluates the given enqueue operation once. Returns nothing.
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
    # Test (graph mode only): enqueues ten floats on a single thread, dequeues them from ten
    # threads, and asserts the collected results contain the same items regardless of order.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
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
        # HELPER: thread body for the parallel dequeue test.
        # Evaluates one dequeue and appends the result to the shared results list. Returns nothing.
        results.append(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in enqueue_ops]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, results)

  def testEnqueueAndBlockingDequeue(self):
    # [FUNCTION DOC]
    # Test (graph mode only): starts a consumer that blocks on an empty capacity-3 queue and a
    # producer that enqueues three floats after a delay; after both finish asserts the values were
    # received in order. Timing dependent because it relies on sleeping.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(3, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body. Sleeps briefly so the dequeue blocks first, then evaluates
        # each enqueue operation in turn. Returns nothing.
        # The enqueue_ops should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        for enqueue_op in enqueue_ops:
          self.evaluate(enqueue_op)

      results = []

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body. Evaluates a dequeue once per element and appends each
        # result to the shared results list. Returns nothing.
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

  def testBigDequeueMany(self):
    # [FUNCTION DOC]
    # Test (graph mode only): dequeue-many larger than the queue capacity runs on a thread and
    # blocks; the main thread enqueues four elements one at a time (sleeping between), asserting
    # no result appears before the last enqueue, then asserts the four elements were returned.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(2, dtypes_lib.int32, ((),))
      elem = np.arange(4, dtype=np.int32)
      enq_list = [q.enqueue((e,)) for e in elem]
      deq = q.dequeue_many(4)

      results = []

      def blocking_dequeue():
        # [FUNCTION DOC]
        # HELPER: thread body that evaluates a dequeue-many of four from a capacity-2 queue, which
        # can only finish after several enqueues. Extends the shared results list. Returns nothing.
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

  def testBigEnqueueMany(self):
    # [FUNCTION DOC]
    # Test (graph mode only): enqueue-many larger than capacity blocks on a thread. The main thread
    # dequeues step by step, asserting the enqueue only completes once enough space has been freed,
    # the queue stays full meanwhile, sizes decrease at the end, and all ten values come out in
    # order.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(5, dtypes_lib.int32, ((),))
      elem = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      enq = q.enqueue_many((elem,))
      deq = q.dequeue()
      size_op = q.size()

      enq_done = []

      def blocking_enqueue():
        # [FUNCTION DOC]
        # HELPER: thread body that records a start marker, evaluates an enqueue-many of ten into a
        # capacity-5 queue (blocking until space frees up) and records a completion marker in the
        # shared list. Returns nothing.
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

  def _blockingDequeue(self, sess, dequeue_op):
    # [FUNCTION DOC]
    # HELPER: thread body for the blocking-operation reset test.
    # Runs the given dequeue operation in the given session and expects it to fail with an
    # operation error saying it was cancelled (the session is closed elsewhere). Returns nothing.
    with self.assertRaisesOpError("was cancelled"):
      sess.run(dequeue_op)

  def _blockingDequeueMany(self, sess, dequeue_many_op):
    # [FUNCTION DOC]
    # HELPER: thread body for the blocking-operation reset test.
    # Runs the given dequeue-many operation in the session and expects a cancelled operation error.
    with self.assertRaisesOpError("was cancelled"):
      sess.run(dequeue_many_op)

  def _blockingEnqueue(self, sess, enqueue_op):
    # [FUNCTION DOC]
    # HELPER: thread body for the blocking-operation reset test.
    # Runs the given enqueue operation (on a full queue) in the session and expects a cancelled
    # operation error.
    with self.assertRaisesOpError("was cancelled"):
      sess.run(enqueue_op)

  def _blockingEnqueueMany(self, sess, enqueue_many_op):
    # [FUNCTION DOC]
    # HELPER: thread body for the blocking-operation reset test.
    # Runs the given enqueue-many operation (on a full queue) in the session and expects a
    # cancelled operation error.
    with self.assertRaisesOpError("was cancelled"):
      sess.run(enqueue_many_op)

  def testResetOfBlockingOperation(self):
    # [FUNCTION DOC]
    # Test (graph mode only): builds an empty queue and a full queue, starts four threads running
    # blocking dequeue, dequeue-many, enqueue and enqueue-many, waits briefly, then closes the
    # session, which must cancel all four blocked operations. Afterwards opens a fresh session so
    # the shared cached session of the test base is not left closed.
    with self.session() as sess:
      q_empty = data_flow_ops.FIFOQueue(5, dtypes_lib.float32, ())
      dequeue_op = q_empty.dequeue()
      dequeue_many_op = q_empty.dequeue_many(1)

      q_full = data_flow_ops.FIFOQueue(5, dtypes_lib.float32)
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

    # Create a new session that hasn't been closed, so cached_session
    # isn't messed up.
    with self.session() as sess:
      pass

  def testBlockingDequeueFromClosedQueue(self):
    # [FUNCTION DOC]
    # Test (graph mode only): after four elements are enqueued, a consumer thread drains them and
    # then blocks on a fifth dequeue; the main thread closes the queue after a delay, which must
    # make the blocked dequeue fail with an out-of-range error.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body. Dequeues each of the four elements asserting order, then
        # expects an out-of-range error saying the queue is closed with insufficient elements.
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

  def testBlockingDequeueFromClosedEmptyQueue(self):
    # [FUNCTION DOC]
    # Test (graph mode only): a consumer blocks on an empty queue; closing the queue after a delay
    # must make the dequeue fail with an out-of-range closed-queue error.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      close_op = q.close()
      dequeued_t = q.dequeue()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body. Expects a dequeue on the empty queue to fail with an
        # out-of-range error once it is closed.
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
    # Test (graph mode only): four elements enqueued, a consumer dequeues them in one batch of four
    # and then blocks on another batch; closing the queue must make the second batch fail with an
    # out-of-range error.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(4)

      enqueue_op.run()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body. Dequeues four elements and asserts they equal the enqueued
        # list, then expects a further dequeue-many to fail with an out-of-range closed error.
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
    # Test (graph mode only): four elements in the queue, batches of three requested. After the
    # first batch succeeds the second, left with fewer than three elements, must fail once the
    # queue is closed.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(3)

      enqueue_op.run()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body. Dequeues a batch of three and asserts it equals the first
        # three elements, then expects the next batch of three (only one element left) to fail with
        # an out-of-range closed-queue error.
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

  def testDequeueUpToFromClosedQueueReturnsRemainder(self):
    # [FUNCTION DOC]
    # Test (graph mode only): with four elements and a closed queue, dequeue-up-to of three returns
    # three elements and then the one remaining element instead of failing, unlike dequeue-many.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(3)

      enqueue_op.run()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body. Asserts the first dequeue-up-to returns three elements and
        # the second returns the remaining one element.
        self.assertAllEqual(elems[:3], self.evaluate(dequeued_t))
        self.assertAllEqual(elems[3:], self.evaluate(dequeued_t))

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testEnqueueManyLargerThanCapacityWithConcurrentDequeueMany(self):
    # [FUNCTION DOC]
    # Test (graph mode only): runs an enqueue-many, a dequeue-many of three and a close operation
    # in separate threads on a queue whose capacity equals the batch, with a sleep before closing.
    # Verifies that the dequeue-many gets three elements, then fails after close, and the leftover
    # element can still be dequeued.
    with ops.Graph().as_default(), self.session() as sess:
      q = data_flow_ops.FIFOQueue(4, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(3)
      cleanup_dequeue_t = q.dequeue()

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body; runs the enqueue-many of four into the capacity-4 queue
        # in the session. Returns nothing.
        sess.run(enqueue_op)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body; asserts a dequeue-many of three returns the first three
        # elements, that the next dequeue-many fails with an out-of-range error after closing, and
        # that a single dequeue then returns the fourth element.
        self.assertAllEqual(elems[0:3], sess.run(dequeued_t))
        with self.assertRaises(errors_impl.OutOfRangeError):
          sess.run(dequeued_t)
        self.assertEqual(elems[3], sess.run(cleanup_dequeue_t))

      def close():
        # [FUNCTION DOC]
        # HELPER: thread body; runs the close operation in the session. Returns nothing.
        sess.run(close_op)

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
    # Test (graph mode only): two-component queue with three elements; a blocked dequeue-many of
    # four is failed by closing the queue. Then asserts the partially dequeued elements were
    # restored to the queue in the original order and that the queue ends up empty.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(4, (dtypes_lib.float32, dtypes_lib.float32), (
          (), ()))
      elems_a = [1.0, 2.0, 3.0]
      elems_b = [10.0, 20.0, 30.0]
      enqueue_op = q.enqueue_many((elems_a, elems_b))
      dequeued_a_t, dequeued_b_t = q.dequeue_many(4)
      cleanup_dequeue_a_t, cleanup_dequeue_b_t = q.dequeue()
      close_op = q.close()

      enqueue_op.run()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body; expects a dequeue-many of four from a queue holding only
        # three elements to fail with an out-of-range error after the queue is closed.
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
    # Test (graph mode only): a dequeue-many of four blocks on an empty queue and must fail with an
    # out-of-range closed-queue error once the queue is closed.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      close_op = q.close()
      dequeued_t = q.dequeue_many(4)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body; expects an out-of-range closed-queue error from the
        # dequeue-many.
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
    # Test (graph mode only): like the previous test but with dequeue-up-to of four on an empty
    # queue; closing the queue must make it fail with an out-of-range error.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(4)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body; expects an out-of-range closed-queue error from the
        # dequeue-up-to.
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
    # Test: enqueues once, closes the queue, then expects a second run of the enqueue to raise a
    # cancelled error saying the queue is closed.
    with self.cached_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      close_op = q.close()

      enqueue_op.run()
      close_op.run()

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegex(errors_impl.CancelledError, "is closed"):
        enqueue_op.run()

  def testEnqueueManyToClosedQueue(self):
    # [FUNCTION DOC]
    # Test: like the previous test but with enqueue-many; after closing, a repeat enqueue-many must
    # raise a cancelled error saying the queue is closed.
    with self.cached_session():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
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
    # Test (graph mode only): fills a capacity-4 queue, starts a thread enqueueing a fifth element
    # that blocks, then dequeues the four original elements in order followed by the blocked one
    # (50), and joins the thread.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(4, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue((50.0,))
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        # [FUNCTION DOC]
        # HELPER: thread body; evaluates the enqueue of 50 into the full queue (blocks until a
        # dequeue frees space). Returns nothing.
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
    # Test (graph mode only): like the previous test but the blocked operation is an enqueue-many of
    # 50 and 60; asserts the dequeues return the original four then 50 then 60, and joins the
    # thread.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(4, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue_many(([50.0, 60.0],))
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        # [FUNCTION DOC]
        # HELPER: thread body; evaluates an enqueue-many of two elements into the full queue and
        # blocks until space is available. Returns nothing.
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
    # Test (graph mode only): with a full queue, an enqueue blocks and then a close is issued. A
    # single dequeue (value 10) unblocks both, the pending enqueue still succeeds, the remaining
    # values 20, 30, 40 and 50 are dequeued in order, and the queue ends empty.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(4, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue((50.0,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        # [FUNCTION DOC]
        # HELPER: thread body; evaluates a blocked enqueue of 50 which is expected to succeed after
        # a dequeue even though close was requested meanwhile. Returns nothing.
        # Expect the operation to succeed once the dequeue op runs.
        self.evaluate(blocking_enqueue_op)

      enqueue_thread = self.checkedThread(target=blocking_enqueue)
      enqueue_thread.start()

      # The close_op should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.2)

      def close():
        # [FUNCTION DOC]
        # HELPER: thread body; evaluates the close operation. Returns nothing.
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
    # Test (graph mode only): like the previous test but with a blocked enqueue-many of 50 and 60
    # on a queue holding three elements; after a close and a dequeue of 10 both threads finish and
    # the remaining values 20, 30, 50, 60 are dequeued in order.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.session() as sess:
      q = data_flow_ops.FIFOQueue(4, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue_many(([50.0, 60.0],))
      close_op = q.close()
      dequeued_t = q.dequeue()
      enqueue_op.run()

      def blocking_enqueue():
        # [FUNCTION DOC]
        # HELPER: thread body; runs the blocking enqueue-many of two elements in the session.
        # Returns nothing.
        sess.run(blocking_enqueue_op)

      enqueue_thread = self.checkedThread(target=blocking_enqueue)
      enqueue_thread.start()

      # The close_op should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      def close():
        # [FUNCTION DOC]
        # HELPER: thread body; runs the close operation in the session. Returns nothing.
        sess.run(close_op)

      close_thread = self.checkedThread(target=close)
      close_thread.start()

      # The dequeue will unblock both threads.
      self.assertEqual(10.0, self.evaluate(dequeued_t))
      enqueue_thread.join()
      close_thread.join()
      for elem in [20.0, 30.0, 50.0, 60.0]:
        self.assertEqual(elem, self.evaluate(dequeued_t))

  def testParallelEnqueueMany(self):
    # [FUNCTION DOC]
    # Test (graph mode only): ten threads each enqueue the same batch of 100 floats into a
    # capacity-1000 queue; one dequeue-many of 1000 must return ten copies of the batch (compared
    # ignoring order).
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(1000, dtypes_lib.float32, shapes=())
      elems = [10.0 * x for x in range(100)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(1000)

      # Enqueue 100 items in parallel on 10 threads.
      def enqueue():
        # [FUNCTION DOC]
        # HELPER: thread body; evaluates the enqueue-many of 100 elements once. Returns nothing.
        self.evaluate(enqueue_op)

      threads = [self.checkedThread(target=enqueue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      self.assertCountEqual(self.evaluate(dequeued_t), elems * 10)

  def testParallelDequeueMany(self):
    # [FUNCTION DOC]
    # Test (graph mode only): enqueues 1000 floats, then ten threads each dequeue 100; asserts the
    # union of results equals the original items regardless of order.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(1000, dtypes_lib.float32, shapes=())
      elems = [10.0 * x for x in range(1000)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(100)

      enqueue_op.run()

      # Dequeue 100 items in parallel on 10 threads.
      dequeued_elems = []

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: thread body; evaluates a dequeue-many of 100 and appends the results to the shared
        # list. Returns nothing.
        dequeued_elems.extend(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testParallelDequeueUpTo(self):
    # [FUNCTION DOC]
    # Test (graph mode only): enqueues 1000 floats and closes the queue; ten threads each dequeue up
    # to 101 elements (the last gets the remainder); asserts all 1000 elements were collected
    # regardless of order.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(1000, dtypes_lib.float32, shapes=())
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
        # HELPER: thread body; evaluates a dequeue-up-to of 101 and appends the results to the
        # shared list. Returns nothing.
        dequeued_elems.extend(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testParallelEnqueueAndDequeue(self):
    # [FUNCTION DOC]
    # Test (graph mode only): pre-fills a capacity-50 queue with 49 tens, then runs ten producers
    # and ten consumers concurrently (each doing one hundred operations); consumers assert each
    # value is 10 or 20. Finally dequeues the 49 leftover elements and asserts the same.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(50, dtypes_lib.float32, shapes=())
      initial_elements = [10.0] * 49
      q.enqueue_many((initial_elements,)).run()

      enqueue_op = q.enqueue((20.0,))
      dequeued_t = q.dequeue()

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body; evaluates the enqueue of 20 one hundred times. Returns
        # nothing.
        for _ in range(100):
          self.evaluate(enqueue_op)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body; one hundred times evaluates a dequeue and asserts the value
        # is either 10 or 20. Returns nothing.
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
    # Test (graph mode only): a consumer thread checks ordered values 0 to 249 while the main
    # thread randomly alternates between single enqueues and random-size batch enqueues (fed via
    # placeholders) until 250 elements were sent, then closes the queue and asserts it is empty.
    # Uses randomness.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.int32, shapes=())
      enqueue_placeholder = array_ops.placeholder(dtypes_lib.int32, shape=())
      enqueue_op = q.enqueue((enqueue_placeholder,))
      enqueuemany_placeholder = array_ops.placeholder(
          dtypes_lib.int32, shape=(None,))
      enqueuemany_op = q.enqueue_many((enqueuemany_placeholder,))

      dequeued_t = q.dequeue()
      close_op = q.close()

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body; dequeues 250 elements, asserting they come out as the
        # consecutive integers starting at zero.
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
    # Test (graph mode only): a producer thread enqueues 0 to 249 while the main thread randomly
    # alternates between single dequeues and random-size dequeue-many calls (count fed through a
    # placeholder), asserting consecutive values; then closes the queue, joins and asserts it is
    # empty. Uses randomness.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.int32, shapes=())
      enqueue_op = q.enqueue_many((np.arange(250, dtype=np.int32),))
      dequeued_t = q.dequeue()
      count_placeholder = array_ops.placeholder(dtypes_lib.int32, shape=())
      dequeuemany_t = q.dequeue_many(count_placeholder)

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body; evaluates the enqueue-many of 250 consecutive integers.
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
    # Test (graph mode only): a dequeue-many of four blocks on an empty queue until a producer
    # enqueues four elements after a delay; asserts the collected elements equal the enqueued list.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(4)

      dequeued_elems = []

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body; sleeps briefly so the dequeue blocks first, then evaluates
        # the enqueue-many of four elements.
        # The enqueue_op should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.evaluate(enqueue_op)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body; evaluates a dequeue-many of four and appends the values to
        # the shared list.
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
    # Test (graph mode only): like the blocking dequeue-many test but with dequeue-up-to; the
    # blocked call returns all four elements once the producer enqueues them.
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, ())
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_up_to(4)

      dequeued_elems = []

      def enqueue():
        # [FUNCTION DOC]
        # HELPER: producer thread body; sleeps briefly so the dequeue blocks first, then evaluates
        # the enqueue-many of four elements.
        # The enqueue_op should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.evaluate(enqueue_op)

      def dequeue():
        # [FUNCTION DOC]
        # HELPER: consumer thread body; evaluates a dequeue-up-to of four and appends the values to
        # the shared list.
        dequeued_elems.extend(self.evaluate(dequeued_t).tolist())

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      self.assertAllEqual(elems, dequeued_elems)

  def testDequeueEnqueueFail(self):
    # [FUNCTION DOC]
    # Test (graph mode only): runs a dequeue together with an enqueue guarded by a control
    # dependency on a failing assertion. The run must raise an invalid-argument error whose message
    # contains the assertion text, verifying a failed dependency aborts the whole step.
    with self.cached_session() as session:
      q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      a = q.dequeue()
      b = control_flow_assert.Assert(False, ["Before enqueue"])
      with ops.control_dependencies([b]):
        c = q.enqueue(33)
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Before enqueue" in str(e)):
        session.run([a, c])


class FIFOQueueDictTest(test.TestCase):
  # [CLASS DOC]
  # Test case for FIFOQueue constructed with component names (dictionary mode). Only checks the
  # constructor node definition and the stored names. No special fixtures.

  def testConstructor(self):
    # [FUNCTION DOC]
    # Test: builds a two-component named queue (names i and j, shared name foo) and asserts the
    # node definition attributes and that the queue reports the names in order.
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32),
          names=("i", "j"),
          shared_name="foo",
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, tensor.Tensor))
    self.assertProtoEquals("""
      name:'Q' device: "/device:CPU:*" op:'FIFOQueueV2'
      attr { key: 'component_types' value { list {
        type: DT_INT32 type : DT_FLOAT
      } } }
      attr { key: 'shapes' value { list {} } }
      attr { key: 'capacity' value { i: 5 } }
      attr { key: 'container' value { s: '' } }
      attr { key: 'shared_name' value { s: 'foo' } }
      """, q.queue_ref.op.node_def)
    self.assertEqual(["i", "j"], q.names)

  def testConstructorWithShapes(self):
    # [FUNCTION DOC]
    # Test: builds a named queue (names i and f) with explicit shapes and asserts the node
    # definition shapes and the reported names.
    with ops.Graph().as_default():
      q = data_flow_ops.FIFOQueue(
          5, (dtypes_lib.int32, dtypes_lib.float32),
          names=("i", "f"),
          shapes=(tensor_shape.TensorShape([1, 1, 2, 3]),
                  tensor_shape.TensorShape([5, 8])),
          name="Q")
    self.assertTrue(isinstance(q.queue_ref, tensor.Tensor))
    self.assertProtoEquals("""
      name:'Q' device: "/device:CPU:*" op:'FIFOQueueV2'
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
    self.assertEqual(["i", "f"], q.names)


@test_util.run_v1_only(
    "These tests are heavily reliant on Session for parallelism. We can likely "
    "convert them to run in 2.x/eager, but it may be difficult.")
class FIFOQueueWithTimeoutTest(test.TestCase):
  # [CLASS DOC]
  # Version 1 graph-only tests for operation timeouts on blocking queue operations, using the
  # session configuration timeout and the per-run timeout option.

  def testDequeueWithTimeout(self):
    # [FUNCTION DOC]
    # Test (graph mode only): in a session configured with a 20 ms operation timeout, asserts the
    # queue container attribute is empty, then evaluates a dequeue on an empty queue and expects a
    # deadline-exceeded error mentioning a timeout waiting for notification.
    with self.session(
        config=config_pb2.ConfigProto(operation_timeout_in_ms=20)) as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      self.assertEqual(
          compat.as_bytes(""), q.queue_ref.op.get_attr("container"))
      dequeued_t = q.dequeue()

      # Intentionally do not run any enqueue_ops so that dequeue will block
      # until operation_timeout_in_ms.
      with self.assertRaisesRegex(errors_impl.DeadlineExceededError,
                                  "Timed out waiting for notification"):
        self.evaluate(dequeued_t)

  def testReusableAfterTimeout(self):
    # [FUNCTION DOC]
    # Test (graph mode only): two dequeue runs on an empty queue with a 10 ms per-run timeout must
    # each raise a deadline-exceeded error; afterwards an enqueue of 37 succeeds and a dequeue
    # returns 37, showing the queue is still usable after timeouts.
    with self.cached_session() as sess:
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      dequeued_t = q.dequeue()
      enqueue_op = q.enqueue(37)

      with self.assertRaisesRegex(errors_impl.DeadlineExceededError,
                                  "Timed out waiting for notification"):
        sess.run(dequeued_t, options=config_pb2.RunOptions(timeout_in_ms=10))

      with self.assertRaisesRegex(errors_impl.DeadlineExceededError,
                                  "Timed out waiting for notification"):
        sess.run(dequeued_t, options=config_pb2.RunOptions(timeout_in_ms=10))

      self.evaluate(enqueue_op)
      self.assertEqual(37, self.evaluate(dequeued_t))


class QueueContainerTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking that a queue created inside a resource container scope records that
  # container name in its node attributes.

  def testContainer(self):
    # [FUNCTION DOC]
    # Test: creates a queue inside a resource container scope named test in a fresh graph and
    # asserts the queue node container attribute equals that name.
    with ops.Graph().as_default():
      with ops.container("test"):
        q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
    self.assertEqual(
        compat.as_bytes("test"), q.queue_ref.op.get_attr("container"))


class FIFOQueueBenchmark(test.Benchmark):
  # [CLASS DOC]
  # KNOWN ISSUE: the class defines no public benchmark method (nothing whose name marks it as a
  # benchmark), so the timing helper is never invoked by the benchmark runner and this class
  # appears to be dead code.
  # Benchmark class measuring the time of one enqueue-then-dequeue cycle on a FIFOQueue in a
  # plain session. Holds no state. Provides a graph-building helper and a timing helper.
  """Benchmark FIFOQueue operations."""

  def _build_graph(self):
    # [FUNCTION DOC]
    # HELPER: graph builder for the benchmark timing helper.
    # Creates a capacity-1 float queue with an initial enqueue of 1.0 and an operation that
    # dequeues one element and enqueues it incremented by one. Takes no inputs and returns a pair:
    # the initialisation enqueue operation and the increment operation. Adds nodes to the current
    # default graph.
    """Builds a graph that enqueues and dequeues a single float.

    Returns:
      A tuple with the graph init tensor and graph output tensor.
    """
    q = data_flow_ops.FIFOQueue(1, "float")
    init = q.enqueue(1.0)
    x = q.dequeue()
    q_inc = q.enqueue(x + 1)
    return init, q_inc

  # TODO(suharshs): Add benchmarks for:
  #   - different capacities of the queue
  #   - various sizes of tensors
  #   - enqueue_many, dequeue_many
  def _run(self, num_iters):
    # [FUNCTION DOC]
    # HELPER: timing routine of the benchmark class.
    # KNOWN ISSUE: never called by any benchmark method in this file, so it is not run by the
    # benchmark runner.
    # Builds the graph in a new session, runs the initialisation, does one warm-up run, then times
    # the given number of iterations of the increment operation. Prints the seconds per iteration,
    # reports a benchmark result named fifo_queue (iterations and average wall time) and returns
    # the total duration in seconds. Input num_iters is the iteration count (integer).
    """Benchmarks enqueueing and dequeueing from a FIFOQueue.

    Args:
      num_iters: The number of iterations to run.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      init, output = self._build_graph()
    with session_lib.Session(graph=graph) as session:
      init.run()
      _ = session.run(output)  # warm up.
      start_time = time.time()
      for _ in range(num_iters):
        _ = session.run(output)
      duration = time.time() - start_time
    print("%f secs per enqueue-dequeue" % (duration / num_iters))

    self.report_benchmark(
        name="fifo_queue", iters=num_iters, wall_time=duration / num_iters)

    return duration


if __name__ == "__main__":
  test.main()
