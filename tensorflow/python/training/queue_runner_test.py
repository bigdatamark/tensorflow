# [MODULE DOC]
# Unit tests for the queue runner implementation (v1 graph mode only). Covers thread creation and
# naming, running enqueue ops to completion, exception capture, coordinator interaction (stop
# requests, grace period), multiple sessions, repeated starts, collection registration, starting all
# runners, and protocol-buffer round-trip serialization. Defines a tiny mock operation type that
# only has a name, used to trigger "not in the graph" errors. Runs through the test main entry point.
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
"""Tests for QueueRunner."""

import collections
import time

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import queue_runner_impl


_MockOp = collections.namedtuple("MockOp", ["name"])


@test_util.run_v1_only("QueueRunner removed from v2")
class QueueRunnerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the queue runner class and the start/add helper functions. Skipped outside v1
  # graph mode. Each test builds its own graph objects and uses a cached session; there is no
  # shared setup. A named tuple with only a name field stands in for an operation that is not in
  # the graph.

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: one enqueue op that increments a counter variable up to three, which then raises
    # out-of-range. Creates threads for the session, checks the single thread's name, runs and joins
    # it. Expected: no captured exceptions and the variable ends at three.
    with self.cached_session() as sess:
      # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      var = variable_v1.VariableV1(zero64)
      count_up_to = var.count_up_to(3)
      queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
      self.evaluate(variables.global_variables_initializer())
      qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
      threads = qr.create_threads(sess)
      self.assertEqual(sorted(t.name for t in threads),
                       ["QueueRunnerThread-fifo_queue-CountUpTo:0"])
      for t in threads:
        t.start()
      for t in threads:
        t.join()
      self.assertEqual(0, len(qr.exceptions_raised))
      # The variable should be 3.
      self.assertEqual(3, self.evaluate(var))

  def testTwoOps(self):
    # [FUNCTION DOC]
    # Scenario: two counter ops (limits three and thirty) on two variables in one runner. Checks the
    # two sorted thread names, starts and joins them. Expected: no captured exceptions and the
    # variables end at three and thirty.
    with self.cached_session() as sess:
      # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      var0 = variable_v1.VariableV1(zero64)
      count_up_to_3 = var0.count_up_to(3)
      var1 = variable_v1.VariableV1(zero64)
      count_up_to_30 = var1.count_up_to(30)
      queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
      qr = queue_runner_impl.QueueRunner(queue, [count_up_to_3, count_up_to_30])
      threads = qr.create_threads(sess)
      self.assertEqual(sorted(t.name for t in threads),
                       ["QueueRunnerThread-fifo_queue-CountUpTo:0",
                        "QueueRunnerThread-fifo_queue-CountUpTo_1:0"])
      self.evaluate(variables.global_variables_initializer())
      for t in threads:
        t.start()
      for t in threads:
        t.join()
      self.assertEqual(0, len(qr.exceptions_raised))
      self.assertEqual(3, self.evaluate(var0))
      self.assertEqual(30, self.evaluate(var1))

  def testExceptionsCaptured(self):
    # [FUNCTION DOC]
    # Scenario: a runner with two mock ops that are not in the graph and no coordinator. Threads are
    # started and joined. Expected: exactly two captured exceptions, each mentioning that the
    # operation is not in the graph.
    with self.cached_session() as sess:
      queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
      qr = queue_runner_impl.QueueRunner(queue, [_MockOp("i fail"),
                                                 _MockOp("so fail")])
      threads = qr.create_threads(sess)
      self.evaluate(variables.global_variables_initializer())
      for t in threads:
        t.start()
      for t in threads:
        t.join()
      exceptions = qr.exceptions_raised
      self.assertEqual(2, len(exceptions))
      self.assertTrue("Operation not in the graph" in str(exceptions[0]))
      self.assertTrue("Operation not in the graph" in str(exceptions[1]))

  def testRealDequeueEnqueue(self):
    # [FUNCTION DOC]
    # Scenario: a runner moves items from a first queue into a second queue. Two values are put in
    # the first queue, which is then closed. Expected: the runner stops cleanly with no exceptions,
    # both values appear in the second queue, and it is closed afterwards (a further dequeue raises
    # an out-of-range error saying the queue is closed).
    with self.cached_session() as sess:
      q0 = data_flow_ops.FIFOQueue(3, dtypes.float32)
      enqueue0 = q0.enqueue((10.0,))
      close0 = q0.close()
      q1 = data_flow_ops.FIFOQueue(30, dtypes.float32)
      enqueue1 = q1.enqueue((q0.dequeue(),))
      dequeue1 = q1.dequeue()
      qr = queue_runner_impl.QueueRunner(q1, [enqueue1])
      threads = qr.create_threads(sess)
      for t in threads:
        t.start()
      # Enqueue 2 values, then close queue0.
      enqueue0.run()
      enqueue0.run()
      close0.run()
      # Wait for the queue runner to terminate.
      for t in threads:
        t.join()
      # It should have terminated cleanly.
      self.assertEqual(0, len(qr.exceptions_raised))
      # The 2 values should be in queue1.
      self.assertEqual(10.0, self.evaluate(dequeue1))
      self.assertEqual(10.0, self.evaluate(dequeue1))
      # And queue1 should now be closed.
      with self.assertRaisesRegex(errors_impl.OutOfRangeError, "is closed"):
        self.evaluate(dequeue1)

  def testRespectCoordShouldStop(self):
    # [FUNCTION DOC]
    # Scenario: the coordinator has already been asked to stop before threads are created with it.
    # Expected: thread names include the enqueue thread and the close-on-stop thread; after starting
    # and joining via the coordinator there are no captured exceptions and the counter variable
    # remains zero because the enqueue loop never ran.
    with self.cached_session() as sess:
      # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      var = variable_v1.VariableV1(zero64)
      count_up_to = var.count_up_to(3)
      queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
      self.evaluate(variables.global_variables_initializer())
      qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
      # As the coordinator to stop.  The queue runner should
      # finish immediately.
      coord = coordinator.Coordinator()
      coord.request_stop()
      threads = qr.create_threads(sess, coord)
      self.assertEqual(sorted(t.name for t in threads),
                       ["QueueRunnerThread-fifo_queue-CountUpTo:0",
                        "QueueRunnerThread-fifo_queue-close_on_stop"])
      for t in threads:
        t.start()
      coord.join()
      self.assertEqual(0, len(qr.exceptions_raised))
      # The variable should be 0.
      self.assertEqual(0, self.evaluate(var))

  def testRequestStopOnException(self):
    # [FUNCTION DOC]
    # Scenario: a runner with a mock op not in the graph, created with a coordinator. Threads are
    # started. Expected: the error is forwarded to the coordinator and re-raised as a value error
    # (operation not in the graph) when the coordinator joins.
    with self.cached_session() as sess:
      queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
      qr = queue_runner_impl.QueueRunner(queue, [_MockOp("not an op")])
      coord = coordinator.Coordinator()
      threads = qr.create_threads(sess, coord)
      for t in threads:
        t.start()
      # The exception should be re-raised when joining.
      with self.assertRaisesRegex(ValueError, "Operation not in the graph"):
        coord.join()

  def testGracePeriod(self):
    # [FUNCTION DOC]
    # Scenario: a small queue whose enqueue quickly blocks, with threads started under a
    # coordinator. One element is dequeued, a short sleep occurs, then stop is requested. Expected:
    # the close-on-stop thread cancels the queue so the blocked enqueue ends, and joining with a
    # one-second grace period completes without error.
    with self.cached_session() as sess:
      # The enqueue will quickly block.
      queue = data_flow_ops.FIFOQueue(2, dtypes.float32)
      enqueue = queue.enqueue((10.0,))
      dequeue = queue.dequeue()
      qr = queue_runner_impl.QueueRunner(queue, [enqueue])
      coord = coordinator.Coordinator()
      qr.create_threads(sess, coord, start=True)
      # Dequeue one element and then request stop.
      dequeue.op.run()
      time.sleep(0.02)
      coord.request_stop()
      # We should be able to join because the RequestStop() will cause
      # the queue to be closed and the enqueue to terminate.
      coord.join(stop_grace_period_secs=1.0)

  def testMultipleSessions(self):
    # [FUNCTION DOC]
    # Scenario: create threads (without starting) for the same runner in two different sessions.
    # Expected: both calls return the same number of threads, showing running counts are tracked
    # per session.
    with self.cached_session() as sess:
      with session.Session() as other_sess:
        zero64 = constant_op.constant(0, dtype=dtypes.int64)
        var = variable_v1.VariableV1(zero64)
        count_up_to = var.count_up_to(3)
        queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
        self.evaluate(variables.global_variables_initializer())
        coord = coordinator.Coordinator()
        qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
        # NOTE that this test does not actually start the threads.
        threads = qr.create_threads(sess, coord=coord)
        other_threads = qr.create_threads(other_sess, coord=coord)
        self.assertEqual(len(threads), len(other_threads))

  def testIgnoreMultiStarts(self):
    # [FUNCTION DOC]
    # Scenario: create threads for a session (not started), then ask again for the same session.
    # Expected: the second call returns an empty list because threads for that session are still
    # counted as running.
    with self.cached_session() as sess:
      # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      var = variable_v1.VariableV1(zero64)
      count_up_to = var.count_up_to(3)
      queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
      self.evaluate(variables.global_variables_initializer())
      coord = coordinator.Coordinator()
      qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
      threads = []
      # NOTE that this test does not actually start the threads.
      threads.extend(qr.create_threads(sess, coord=coord))
      new_threads = qr.create_threads(sess, coord=coord)
      self.assertEqual([], new_threads)

  def testThreads(self):
    # [FUNCTION DOC]
    # Scenario: a runner with one good counter op and one mock op not in the graph, threads created
    # with immediate start and no coordinator. Expected: the two thread names are as predicted and,
    # after joining, exactly one exception is captured (operation not in the graph). Creating and
    # starting threads a second time yields a fresh capture list that again holds one such exception.
    with self.cached_session() as sess:
      # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      var = variable_v1.VariableV1(zero64)
      count_up_to = var.count_up_to(3)
      queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
      self.evaluate(variables.global_variables_initializer())
      qr = queue_runner_impl.QueueRunner(queue, [count_up_to,
                                                 _MockOp("bad_op")])
      threads = qr.create_threads(sess, start=True)
      self.assertEqual(sorted(t.name for t in threads),
                       ["QueueRunnerThread-fifo_queue-CountUpTo:0",
                        "QueueRunnerThread-fifo_queue-bad_op"])
      for t in threads:
        t.join()
      exceptions = qr.exceptions_raised
      self.assertEqual(1, len(exceptions))
      self.assertTrue("Operation not in the graph" in str(exceptions[0]))

      threads = qr.create_threads(sess, start=True)
      for t in threads:
        t.join()
      exceptions = qr.exceptions_raised
      self.assertEqual(1, len(exceptions))
      self.assertTrue("Operation not in the graph" in str(exceptions[0]))

  def testName(self):
    # [FUNCTION DOC]
    # Scenario: a queue created inside a name scope and a runner over a no-op. Expected: the runner
    # name is the scoped queue name, and after adding it to the default collection, the collection
    # filtered by that scope contains exactly one entry.
    with ops.name_scope("scope"):
      queue = data_flow_ops.FIFOQueue(10, dtypes.float32, name="queue")
    qr = queue_runner_impl.QueueRunner(queue, [control_flow_ops.no_op()])
    self.assertEqual("scope/queue", qr.name)
    queue_runner_impl.add_queue_runner(qr)
    self.assertEqual(
        1, len(ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS, "scope")))

  def testStartQueueRunners(self):
    # [FUNCTION DOC]
    # Scenario: a counter-based runner is added to the graph collection; in a session the variables
    # are initialized and the start-all function is called. Expected: after joining the returned
    # threads there are no captured exceptions and the counter equals three.
    # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
    zero64 = constant_op.constant(0, dtype=dtypes.int64)
    var = variable_v1.VariableV1(zero64)
    count_up_to = var.count_up_to(3)
    queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
    init_op = variables.global_variables_initializer()
    qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
    queue_runner_impl.add_queue_runner(qr)
    with self.cached_session() as sess:
      init_op.run()
      threads = queue_runner_impl.start_queue_runners(sess)
      for t in threads:
        t.join()
      self.assertEqual(0, len(qr.exceptions_raised))
      # The variable should be 3.
      self.assertEqual(3, self.evaluate(var))

  def testStartQueueRunnersRaisesIfNotASession(self):
    # [FUNCTION DOC]
    # Scenario: a runner is registered in the collection and the start-all function is called with a
    # plain string instead of a session. Expected: a type error whose message mentions a session.
    zero64 = constant_op.constant(0, dtype=dtypes.int64)
    var = variable_v1.VariableV1(zero64)
    count_up_to = var.count_up_to(3)
    queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
    init_op = variables.global_variables_initializer()
    qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
    queue_runner_impl.add_queue_runner(qr)
    with self.cached_session():
      init_op.run()
      with self.assertRaisesRegex(TypeError, "tf.Session"):
        queue_runner_impl.start_queue_runners("NotASession")

  def testStartQueueRunnersIgnoresMonitoredSession(self):
    # [FUNCTION DOC]
    # Scenario: a runner is registered and the start-all function is called with a monitored
    # session object (backward-compatibility case). Expected: it returns an empty list of threads.
    zero64 = constant_op.constant(0, dtype=dtypes.int64)
    var = variable_v1.VariableV1(zero64)
    count_up_to = var.count_up_to(3)
    queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
    init_op = variables.global_variables_initializer()
    qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
    queue_runner_impl.add_queue_runner(qr)
    with self.cached_session():
      init_op.run()
      threads = queue_runner_impl.start_queue_runners(
          monitored_session.MonitoredSession())
      self.assertFalse(threads)

  def testStartQueueRunnersNonDefaultGraph(self):
    # [FUNCTION DOC]
    # Scenario: the queue, counter and runner are built inside a separate graph, and a session is
    # opened on that graph (which is not the default outside the block). Expected: the start-all
    # function works against the session's graph; no captured exceptions and the counter is three.
    # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
    graph = ops.Graph()
    with graph.as_default():
      zero64 = constant_op.constant(0, dtype=dtypes.int64)
      var = variable_v1.VariableV1(zero64)
      count_up_to = var.count_up_to(3)
      queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
      init_op = variables.global_variables_initializer()
      qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
      queue_runner_impl.add_queue_runner(qr)
    with self.session(graph=graph) as sess:
      init_op.run()
      threads = queue_runner_impl.start_queue_runners(sess)
      for t in threads:
        t.join()
      self.assertEqual(0, len(qr.exceptions_raised))
      # The variable should be 3.
      self.assertEqual(3, self.evaluate(var))

  def testQueueRunnerSerializationRoundTrip(self):
    # [FUNCTION DOC]
    # Scenario: a runner with explicit close and cancel no-ops and two closed-exception types is
    # converted to a definition message and rebuilt from it. Expected: queue name, single enqueue
    # op, close op, cancel op and both exception types survive. Then the exception types are
    # cleared from the message (legacy format) and rebuilt; expected: all fields match and the
    # exception types default to just out-of-range.
    graph = ops.Graph()
    with graph.as_default():
      queue = data_flow_ops.FIFOQueue(10, dtypes.float32, name="queue")
      enqueue_op = control_flow_ops.no_op(name="enqueue")
      close_op = control_flow_ops.no_op(name="close")
      cancel_op = control_flow_ops.no_op(name="cancel")
      qr0 = queue_runner_impl.QueueRunner(
          queue, [enqueue_op],
          close_op,
          cancel_op,
          queue_closed_exception_types=(errors_impl.OutOfRangeError,
                                        errors_impl.CancelledError))
      qr0_proto = queue_runner_impl.QueueRunner.to_proto(qr0)
      qr0_recon = queue_runner_impl.QueueRunner.from_proto(qr0_proto)
      self.assertEqual("queue", qr0_recon.queue.name)
      self.assertEqual(1, len(qr0_recon.enqueue_ops))
      self.assertEqual(enqueue_op, qr0_recon.enqueue_ops[0])
      self.assertEqual(close_op, qr0_recon.close_op)
      self.assertEqual(cancel_op, qr0_recon.cancel_op)
      self.assertEqual(
          (errors_impl.OutOfRangeError, errors_impl.CancelledError),
          qr0_recon.queue_closed_exception_types)

      # Assert we reconstruct an OutOfRangeError for QueueRunners
      # created before QueueRunnerDef had a queue_closed_exception_types field.
      del qr0_proto.queue_closed_exception_types[:]
      qr0_legacy_recon = queue_runner_impl.QueueRunner.from_proto(qr0_proto)
      self.assertEqual("queue", qr0_legacy_recon.queue.name)
      self.assertEqual(1, len(qr0_legacy_recon.enqueue_ops))
      self.assertEqual(enqueue_op, qr0_legacy_recon.enqueue_ops[0])
      self.assertEqual(close_op, qr0_legacy_recon.close_op)
      self.assertEqual(cancel_op, qr0_legacy_recon.cancel_op)
      self.assertEqual((errors_impl.OutOfRangeError,),
                       qr0_legacy_recon.queue_closed_exception_types)


if __name__ == "__main__":
  test.main()
