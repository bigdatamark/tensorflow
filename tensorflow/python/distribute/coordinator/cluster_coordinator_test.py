# [MODULE DOC]
# Unit and integration tests for the parameter server cluster coordinator (closure queue, remote
# values, error reporting, scheduling, per-worker datasets and variables, sharded variables and
# strategy integration). Contains small test utilities (a closure subclass that builds an output
# remote value, a thread subclass that records errors, a test base class that installs it, and a
# factory that builds an in-process cluster and coordinator) followed by several test classes.
# Some test classes subclass others to re-run the same tests under different settings (closure
# queue size limit, schedule start delay environment variables). Not a helper module; the tests
# create in-process clusters and mutate module-level coordinator settings.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for coordinator.py."""

import collections
import contextlib
import functools
import gc
import os
import platform
import sys
import threading
import time
import traceback
from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import sharded_variable
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator
from tensorflow.python.training.server_lib import ClusterSpec


class ClosureWithOutput(coordinator_lib.Closure):
  # [CLASS DOC]
  # HELPER: small test utility. A coordinator closure that immediately creates its output remote
  # value on construction and keeps it in an attribute, so tests can inspect results and errors.

  def __init__(self, function, cancellation_mgr=None, args=None, kwargs=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: function (callable or traced function to run), cancellation_mgr
    # (optional cancellation manager), args (optional positional arguments), kwargs (optional
    # keyword arguments). Calls the base closure constructor, then builds and stores the output
    # remote value. Returns nothing.
    super(ClosureWithOutput, self).__init__(
        function, cancellation_mgr=cancellation_mgr, args=args, kwargs=kwargs)
    self.output_remote_value = self.build_output_remote_value()


class CoordinatedClosureQueueTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the internal coordinated closure queue: put/get/put-back, mark finished/failed,
  # waiting, done checks, error propagation and cancellation, thread safety and tagged queues.
  # Shared helpers create closures and run functions in parallel threads.

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: single closure through the queue lifecycle. Puts a closure, gets it (same object),
    # checks the queue is not done, puts it back, gets it again, marks it finished. Expected: the
    # queue is then done and waiting returns immediately.
    queue = coordinator_lib._CoordinatedClosureQueue()
    closure1 = self._create_closure(queue._cancellation_mgr)
    queue.put(closure1)
    self.assertIs(closure1, queue.get())
    self.assertFalse(queue.done())
    queue.put_back(closure1)
    self.assertEqual(closure1, queue.get())
    queue.mark_finished()
    self.assertTrue(queue.done())
    queue.wait()

  def testProcessAtLeaseOnce(self):
    # [FUNCTION DOC]
    # Scenario: five labeled closures, each sleeping three seconds then incrementing a per-label
    # counter, processed by two daemon threads that each put back the first closure they obtain
    # before processing. Action: wait on the queue repeatedly (four times). Expected: waiting
    # works repeatedly and every label is processed exactly once.
    # (The test name has a typo: "Lease" for "Least"; and the assertion is exactly once rather
    # than at least once.)
    closure_queue = coordinator_lib._CoordinatedClosureQueue()
    labels = ['A', 'B', 'C', 'D', 'E']
    processed_count = collections.defaultdict(int)

    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def process_queue():
      # [FUNCTION DOC]
      # HELPER: worker thread body. Inside the thread coordinator's stop-on-exception guard, loops
      # getting closures with a 30 second timeout until the null value is returned; puts back the
      # first closure obtained, then runs each later closure's function and marks it finished.
      with coord.stop_on_exception():
        has_been_put_back = False
        while True:
          closure = closure_queue.get(timeout=30)
          if closure is None:
            break
          if not has_been_put_back:
            has_been_put_back = True
            closure_queue.put_back(closure)
            continue
          closure._function()
          closure_queue.mark_finished()

    def get_func(label):
      # [FUNCTION DOC]
      # HELPER: factory. Input: label (string). Returns a function that sleeps three seconds then
      # increments the processed counter for that label.

      def func():
        # [FUNCTION DOC]
        # HELPER: closure body created by the factory; sleeps three seconds, increments the
        # counter of its label. Returns nothing.
        time.sleep(3)
        processed_count[label] += 1

      return func

    cm = cancellation.CancellationManager()
    for label in labels:
      closure_queue.put(ClosureWithOutput(get_func(label), cm))
    t1 = threading.Thread(target=process_queue, daemon=True)
    t1.start()
    t2 = threading.Thread(target=process_queue, daemon=True)
    t2.start()

    # Make sure multiple wait() calls are fine.
    closure_queue.wait()
    closure_queue.wait()
    closure_queue.wait()
    closure_queue.wait()

    self.assertEqual(processed_count, collections.Counter(labels))

    coord.join([t1, t2])

  def testNotifyBeforeWait(self):
    # [FUNCTION DOC]
    # Scenario: a closure is put and fully processed (got and marked finished) by a thread before
    # anyone waits. Action: join the thread, then wait on the queue. Expected: wait returns
    # without timing out.
    closure_queue = coordinator_lib._CoordinatedClosureQueue()

    def func():
      # [FUNCTION DOC]
      # HELPER: trivial closure body that logs an informational message. Returns nothing.
      logging.info('func running')

    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def process_queue():
      # [FUNCTION DOC]
      # HELPER: thread body; inside the stop-on-exception guard, gets one closure and marks it
      # finished.
      with coord.stop_on_exception():
        closure_queue.get()
        closure_queue.mark_finished()

    closure_queue.put(ClosureWithOutput(func, closure_queue._cancellation_mgr))
    t = threading.Thread(target=process_queue)
    t.start()
    coord.join([t])

    # This test asserts that waiting at the time the function has been processed
    # doesn't time out.
    closure_queue.wait()

  def _assert_one_unblock_the_other(self, first_fn, second_fn):
    # [FUNCTION DOC]
    # HELPER: shared test utility for the closure queue tests. Inputs: first_fn and second_fn
    # (callables without arguments). Runs first_fn on a background thread (asserting second_fn has
    # not finished when it starts) and second_fn on the calling thread; asserts first_fn has
    # finished by the time second_fn returns. Joins the thread. Returns nothing.
    """Asserts `second_fn` wouldn't return before `first_fn` is finished."""
    first_fn_done = threading.Event()
    second_fn_done = threading.Event()
    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def wrapped_first_fn():
      # [FUNCTION DOC]
      # HELPER: thread body; inside the stop-on-exception guard asserts the second function has
      # not yet finished, runs the first function, then sets its own done event.
      with coord.stop_on_exception():
        self.assertFalse(second_fn_done.is_set())
        first_fn()
        first_fn_done.set()

    self.assertFalse(first_fn_done.is_set())
    t = threading.Thread(target=wrapped_first_fn)
    t.start()

    second_fn()
    self.assertTrue(first_fn_done.is_set())
    second_fn_done.set()

    coord.join([t])

  def _run_two_fns_in_parallel(self, first_fn, second_fn):
    # [FUNCTION DOC]
    # HELPER: shared test utility. Inputs: first_fn and second_fn (callables without arguments).
    # Runs first_fn on a background thread and second_fn on the calling thread, then joins the
    # thread through a thread coordinator (re-raising errors from the thread). Returns nothing.
    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def wrapped_first_fn():
      # [FUNCTION DOC]
      # HELPER: thread body running the first function inside the stop-on-exception guard.
      with coord.stop_on_exception():
        first_fn()

    t = threading.Thread(target=wrapped_first_fn)
    t.start()

    second_fn()
    coord.join([t])

  def testWaitRaiseErrorAfterMarkFailure(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local event wait_finish_event, the thread coordinator and the closure
    # obtained from the queue are created but never used afterwards.
    # Scenario: one closure is put and taken from the queue (inflight). Action: one function marks
    # the queue failed with a value error while another function waits on the queue, arranged via
    # the unblock helper. Expected: the wait raises a value error and afterwards the queue is
    # done. Skipped on Windows with Python 3.8 or newer.
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue = coordinator_lib._CoordinatedClosureQueue()
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))
    closure = closure_queue.get()

    wait_finish_event = threading.Event()
    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    # Using a thread to verify that closure_queue.wait() will not return until
    # all inflight closures are finished.

    def mark_finished_fn():
      # [FUNCTION DOC]
      # KNOWN ISSUE: named "mark finished" but it actually marks the queue as failed.
      # HELPER: raises and catches a value error and reports it to the queue as a failure.
      try:
        raise ValueError('Some error.')
      except ValueError as e:
        closure_queue.mark_failed(e)

    def wait_fn():
      # [FUNCTION DOC]
      # HELPER: waits on the queue and asserts a value error is raised.
      with self.assertRaises(ValueError):
        closure_queue.wait()

    self._assert_one_unblock_the_other(mark_finished_fn, wait_fn)

    self.assertTrue(closure_queue.done())

  def _create_closure(self, cancellation_mgr):
    # [FUNCTION DOC]
    # HELPER: shared test utility. Input: cancellation_mgr (cancellation manager of a queue).
    # Returns a closure-with-output wrapping a traced function that returns the float one.

    @def_function.function()
    def some_function():
      # [FUNCTION DOC]
      # HELPER: trivial traced function returning the floating point value one.
      return 1.0

    return ClosureWithOutput(some_function, cancellation_mgr)

  def _put_two_closures_and_get_one(self):
    # [FUNCTION DOC]
    # HELPER: shared test utility. Creates a fresh closure queue, puts two closures, gets one and
    # asserts it is the first one. Returns a tuple of the queue, the first closure (now inflight)
    # and the second closure (still queued).
    closure_queue = coordinator_lib._CoordinatedClosureQueue()
    closure1 = self._create_closure(closure_queue._cancellation_mgr)
    closure_queue.put(closure1)

    closure2 = self._create_closure(closure_queue._cancellation_mgr)
    closure_queue.put(closure2)

    closure_got = closure_queue.get()  # returns closure1
    self.assertIs(closure_got, closure1)
    self.assertIsNot(closure_got, closure2)
    return closure_queue, closure1, closure2

  def testPutRaiseError(self):
    # [FUNCTION DOC]
    # Scenario: two closures put, one taken, then the queue is marked failed with a value error.
    # Expected: putting a new closure raises a value error; the queue is done; the still-queued
    # second closure's output was cancelled (fetch raises a cancelled error with the reschedule
    # message); the error is then cleared so a further put succeeds. Skipped on Windows with
    # Python 3.8 or newer.
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, closure2 = self._put_two_closures_and_get_one()

    closure_queue.mark_failed(ValueError())

    with self.assertRaises(ValueError):
      closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))

    self.assertTrue(closure_queue.done())

    with self.assertRaisesRegex(
        errors.CancelledError,
        'The corresponding function is cancelled. Please reschedule the '
        'function.'):
      closure2.output_remote_value.fetch()

    # The error is cleared.
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))

  def testWaitRaiseError(self):
    # [FUNCTION DOC]
    # Scenario: same as the put error test but the follow-up action is waiting. Expected: wait
    # raises a value error, the queue is done, the queued closure's output is cancelled, and a
    # second wait succeeds because the error was cleared. Skipped on Windows with Python 3.8+.
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, closure2 = self._put_two_closures_and_get_one()

    closure_queue.mark_failed(ValueError())

    with self.assertRaises(ValueError):
      closure_queue.wait()
    self.assertTrue(closure_queue.done())

    with self.assertRaisesRegex(
        errors.CancelledError,
        'The corresponding function is cancelled. Please reschedule the '
        'function.'):
      closure2.output_remote_value.fetch()

    # The error is cleared.
    closure_queue.wait()

  def testDoneRaiseError(self):
    # [FUNCTION DOC]
    # Scenario: two closures put, one taken; the queue is not done. After marking the queue failed
    # with a value error, asking whether it is done raises a value error. Skipped on Windows with
    # Python 3.8 or newer.
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, _ = self._put_two_closures_and_get_one()

    self.assertFalse(closure_queue.done())
    closure_queue.mark_failed(ValueError())
    with self.assertRaises(ValueError):
      closure_queue.done()

  def _set_error(self, closure_queue, closure, error):
    # [FUNCTION DOC]
    # HELPER: shared test utility. Inputs: closure_queue, closure, error (an exception object).
    # Raises and catches the error, records it on the closure's output remote value and marks the
    # queue failed with it. Returns nothing.
    try:
      raise error
    except Exception as e:  # pylint: disable=broad-except
      closure.output_remote_value._set_error(e)
      closure_queue.mark_failed(e)

  def _test_cancel_closure_when_error(self, call_wait):
    # [FUNCTION DOC]
    # HELPER: shared body of the cancel-after-error tests. Input: call_wait (boolean; if true the
    # reporting function waits on the queue, otherwise it puts a closure).
    # Scenario: three closures (two inflight, one queued); closure one fails with a value error,
    # leaving one inflight and one queued. While a thread simulates a fake cancellation error on
    # closure two, the main thread must see the original error from wait or put. Expected: the
    # original cancellation manager is cancelled, queue is empty with zero inflight and no error;
    # closure one fetch raises the original error; when putting, closure three is cancelled; closure
    # two is ready with the fake cancellation error; and the queue is back to a clean state (the
    # basic test passes on it). Skipped on Windows with Python 3.8 or newer.
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, closure1, closure2 = self._put_two_closures_and_get_one()
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))
    closure_queue.get()
    # At this moment, there are two inflight, one in queue.
    self.assertEqual(closure_queue._inflight_closure_count, 2)

    # Hold a copy of the queue's cancellation manager at this point
    initial_cm = closure_queue._cancellation_mgr

    # Simulating closure1 fails.
    self._set_error(closure_queue, closure1, ValueError('Some error.'))

    # At this moment, there are one inflight, one in queue.
    self.assertEqual(closure_queue._queue.qsize(), 1)
    self.assertEqual(closure_queue._inflight_closure_count, 1)

    closure3 = self._create_closure(closure_queue._cancellation_mgr)

    def fake_cancellation():
      # [FUNCTION DOC]
      # HELPER: records a fake cancellation value error on closure two and marks the queue failed.
      self._set_error(closure_queue, closure2,
                      ValueError('Fake cancellation error.'))

    def report_error():
      # [FUNCTION DOC]
      # HELPER: asserts that waiting on the queue (or putting closure three, depending on the
      # outer flag) raises a value error matching the original error, not the fake cancellation.
      # It should not report the fake cancellation error.
      with self.assertRaisesRegex(ValueError, 'Some error.'):
        # Verifying `wait()` or `put()` raises even if one closure is in
        # flight.
        if call_wait:
          closure_queue.wait()
        else:
          closure_queue.put(closure3)

    self._assert_one_unblock_the_other(fake_cancellation, report_error)

    # The original cancellation manager of the queue has been cancelled.
    self.assertTrue(initial_cm.is_cancelled)

    # At this moment, there is zero inflight, nothing in queue.
    self.assertTrue(closure_queue._queue.empty())
    self.assertEqual(closure_queue._inflight_closure_count, 0)
    self.assertIsNone(closure_queue._error)

    # This asserts that closure1 has errored.
    with self.assertRaisesRegex(ValueError, 'Some error.'):
      closure1.output_remote_value.fetch()

    # The following asserts that closure3 should have been cancelled.
    if not call_wait:
      with self.assertRaisesRegex(
          errors.CancelledError,
          'The corresponding function is cancelled. Please reschedule the '
          'function.'):
        closure3.output_remote_value.fetch()

    # Closure2 was an inflight closure when it got cancelled.
    self.assertEqual(closure2.output_remote_value._status,
                     remote_value.RemoteValueStatus.READY)
    with self.assertRaisesRegex(ValueError, 'Fake cancellation error.'):
      closure2.output_remote_value.fetch()

    # This asserts that the queue has a clear state.
    self.testBasic()

  def testWaitRaiseErrorAfterCancelClosure(self):
    # [FUNCTION DOC]
    # Runs the shared cancel-after-error scenario with the reporting action being a wait.
    self._test_cancel_closure_when_error(call_wait=True)

  def testPutRaiseErrorAfterCancelClosure(self):
    # [FUNCTION DOC]
    # Runs the shared cancel-after-error scenario with the reporting action being a put.
    self._test_cancel_closure_when_error(call_wait=False)

  def testStateIsRestoredAfterJoinIsCalled(self):
    # [FUNCTION DOC]
    # Scenario: two closures put, one taken (one inflight); queue marked failed. Expected: the
    # next put raises a value error and clears the stored error; a following put succeeds and the
    # error stays cleared. (Despite the name, no join is called; a put restores the state.)
    # Skipped on Windows with Python 3.8 or newer.
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, _ = self._put_two_closures_and_get_one()
    self.assertEqual(closure_queue._inflight_closure_count, 1)
    closure_queue.mark_failed(ValueError('test error'))
    with self.assertRaises(ValueError):
      closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))

    # Its error should have been cleared.
    self.assertIsNone(closure_queue._error)
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))
    self.assertIsNone(closure_queue._error)

  def testThreadSafey(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test name has a typo ("Safey" instead of "Safety").
    # Scenario: ten threads each perform twenty queue actions, alternately putting a closure back
    # and marking one finished; the main thread puts one hundred closures. Expected: waiting
    # completes without deadlock and the queue is done.
    thread_count = 10
    queue = coordinator_lib._CoordinatedClosureQueue()

    # Each thread performs 20 queue actions: 10 are `put_back` and 10 are
    # `mark_finished`.
    action_count = 20

    def func():
      # [FUNCTION DOC]
      # HELPER: thread body; performs the fixed number of get actions, putting the closure back on
      # even iterations and marking it finished on odd iterations.
      for i in range(action_count):
        closure = queue.get()
        if i % 2 == 0:
          queue.put_back(closure)
        else:
          queue.mark_finished()

    threads = [threading.Thread(target=func) for i in range(thread_count)]
    for t in threads:
      t.start()

    for _ in range(thread_count * action_count // 2):
      queue.put(self._create_closure(queue._cancellation_mgr))
    queue.wait()
    self.assertTrue(queue.done())

  def testPutGetWithTag(self):
    # [FUNCTION DOC]
    # Scenario: three closures, two put with tags (one and two) and one untagged, while another
    # function gets with tag two. Expected: the tagged get returns the closure with tag two; the
    # queue is not done; getting without a tag returns the untagged closure; after marking it
    # finished the queue is done (tagged leftover is not waited for) and wait returns.
    queue = coordinator_lib._CoordinatedClosureQueue()

    closure1 = self._create_closure(queue._cancellation_mgr)
    closure2 = self._create_closure(queue._cancellation_mgr)
    closure3 = self._create_closure(queue._cancellation_mgr)

    def put_fn():
      # [FUNCTION DOC]
      # HELPER: puts closure three with tag one, closure two with tag two, closure one untagged.
      queue.put(closure3, tag=1)
      queue.put(closure2, tag=2)
      queue.put(closure1)

    def get_fn():
      # [FUNCTION DOC]
      # HELPER: gets from the queue with tag two and asserts it is closure two.
      # The get should only return the closure with tag 2.
      self.assertIs(closure2, queue.get(tag=2))

    self._run_two_fns_in_parallel(put_fn, get_fn)

    self.assertFalse(queue.done())
    self.assertEqual(closure1, queue.get())
    queue.mark_finished()

    # It will not wait for closure3
    self.assertTrue(queue.done())
    queue.wait()


class ErrorReportingThread(threading.Thread):
  # [CLASS DOC]
  # HELPER: test utility replacing the standard thread class. Wraps the thread's target so that
  # any exception is printed and stored in a shared class-level attribute, letting the test case
  # teardown re-raise errors that happened in background threads.

  error = None

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Takes the usual thread arguments but requires a target keyword argument
    # (asserts it exists). Replaces the target with a wrapper preserving its metadata that runs it,
    # and on any exception prints the traceback and stores the exception in the shared class-level
    # error attribute (the wrapper then returns nothing). Then calls the base constructor.
    assert 'target' in kwargs
    target = kwargs['target']

    @functools.wraps(target)
    def wrapped_target(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: wrapper around the thread target. Returns the target's result; on any exception
      # prints its traceback and records it as the shared error, returning nothing.
      try:
        return target(*args, **kwargs)
      except Exception as e:  # pylint: disable=broad-except
        traceback.print_exception(*sys.exc_info())
        ErrorReportingThread.error = e

    kwargs['target'] = wrapped_target
    super(ErrorReportingThread, self).__init__(*args, **kwargs)


class TestCaseWithErrorReportingThread(test.TestCase):
  # [CLASS DOC]
  # HELPER: base test case. During the class lifetime it swaps the global thread class in the
  # threading module for the error-reporting thread, so exceptions in any thread created by the
  # code under test are captured and re-raised in teardown of each test.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Saves the original thread class on the class, replaces the threading module's thread class
    # with the error-reporting one (global side effect), then runs the base class setup.
    cls._threading_thread = threading.Thread
    threading.Thread = ErrorReportingThread
    super(TestCaseWithErrorReportingThread, cls).setUpClass()

  @classmethod
  def tearDownClass(cls):
    # [FUNCTION DOC]
    # Runs the base class teardown, then restores the original thread class in the threading
    # module.
    super(TestCaseWithErrorReportingThread, cls).tearDownClass()
    threading.Thread = cls._threading_thread

  def setUp(self):
    # [FUNCTION DOC]
    # Clears the shared recorded thread error, then runs the base setup.
    ErrorReportingThread.error = None
    super(TestCaseWithErrorReportingThread, self).setUp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Runs the base teardown, then re-raises any error recorded by a background thread during the
    # test so that it fails the test.
    super(TestCaseWithErrorReportingThread, self).tearDown()
    if ErrorReportingThread.error:
      raise ErrorReportingThread.error  # pylint: disable=raising-bad-type


def make_coordinator(num_workers, num_ps, partitioner=None):
  # [FUNCTION DOC]
  # HELPER: shared test utility that builds a cluster coordinator. Inputs: num_workers and
  # num_ps (integer counts), partitioner (optional variable partitioner for the strategy).
  # Creates an in-process gRPC cluster, adds a chief entry on an unused local port, builds a
  # simple cluster resolver, a parameter server strategy (version 2) and returns a cluster
  # coordinator for it. Side effects: starts in-process servers and picks a free port.
  # TODO(rchao): Test the internal rpc_layer version.
  cluster_def = multi_worker_test_base.create_in_process_cluster(
      num_workers=num_workers, num_ps=num_ps, rpc_layer='grpc')
  cluster_def['chief'] = [
      'localhost:%d' % test_util.pick_unused_port()
  ]
  cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
      ClusterSpec(cluster_def), rpc_layer='grpc')
  strategy = parameter_server_strategy_v2.ParameterServerStrategyV2(
      cluster_resolver, variable_partitioner=partitioner)
  return coordinator_lib.ClusterCoordinator(strategy)


class CoordinatorContextTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the coordinator dispatch context (current worker index). Shares a class-level
  # coordinator with five workers and two parameter servers.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Runs base setup, then creates a coordinator with five workers and two parameter servers and
    # stores it and its strategy on the class.
    super(CoordinatorContextTest, cls).setUpClass()
    cls.coordinator = make_coordinator(num_workers=5, num_ps=2)
    cls.strategy = cls.coordinator.strategy

  def testWorkerIndexDatasetFn(self):
    # [FUNCTION DOC]
    # Scenario: a per-worker dataset of the numbers zero to nine is sharded across the workers
    # using the current worker index. Five closures are put directly on the closure queue, each
    # with a distinct tag so that each worker executes exactly one; each sums its iterator.
    # Expected: after joining, the sorted sums equal five, seven, nine, eleven, thirteen (each
    # worker's shard sums). Uses private queue APIs and bumps the inflight count manually.
    def dataset_fn(context):
      # [FUNCTION DOC]
      # HELPER: dataset function. Input: context (input context, ignored). Returns the range
      # dataset sharded by the number of workers using the current worker index.
      del context
      dataset = dataset_ops.DatasetV2.range(10)
      worker_index = coordinator_context.get_current_worker_index()
      dataset = dataset.shard(
          num_shards=self.strategy._extended._num_workers,
          index=worker_index,
      )
      return dataset

    @def_function.function
    def per_worker_dataset_fn():
      # [FUNCTION DOC]
      # HELPER: traced function building a distributed dataset from the dataset function.
      return self.strategy.distribute_datasets_from_function(dataset_fn)

    @def_function.function
    def train_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: traced function. Input: iterator. Returns the 64-bit integer sum of all elements
      # of all batches from the iterator.
      total = constant_op.constant(0, dtype=dtypes.int64)
      for batch in iterator:
        total += math_ops.reduce_sum(batch)
      return total

    per_worker_dataset = self.coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    with self.strategy.scope():
      iterator = iter(per_worker_dataset)
      ret_vals = []
      # Use private APIs to schedule in tagged queues to ensure each worker
      # executes only one closure.
      for ix in range(5):
        closure = coordinator_lib.Closure(
            train_fn,
            self.coordinator._cluster.closure_queue._cancellation_mgr,
            args=(iterator,))
        ret = closure.build_output_remote_value()
        # The queue doesn't keep track of tagged closures as inflight by
        # default, so hack around this for the test.
        self.coordinator._cluster.closure_queue._inflight_closure_count += 1
        self.coordinator._cluster.closure_queue.put(closure, tag=ix)
        ret_vals.append(ret)
    self.coordinator.join()

    fetched_vals = [rv.fetch() for rv in ret_vals]
    expected_results = [5, 7, 9, 11, 13]
    self.assertAllClose(sorted(fetched_vals), expected_results)


class ClusterCoordinatorTest(
    TestCaseWithErrorReportingThread, parameterized.TestCase
):
  # [CLASS DOC]
  # Main tests of the cluster coordinator: scheduling functions, nested outputs, per-worker
  # datasets and iterators, per-worker values and variables, input validation, join semantics.
  # Uses a class-level coordinator with five workers and two parameter servers. Subclassed to
  # rerun all tests under different settings.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Runs base setup, then creates a coordinator (five workers, two parameter servers) and
    # stores it and its strategy on the class.
    super(ClusterCoordinatorTest, cls).setUpClass()
    cls.coordinator = make_coordinator(num_workers=5, num_ps=2)
    cls.strategy = cls.coordinator.strategy

  def testClusterCoordinatorOnlyInitOnce(self):
    # [FUNCTION DOC]
    # Scenario: constructing a second coordinator for the same strategy. Expected: it is the very
    # same coordinator object and shares the same internal cluster object.
    cluster = self.coordinator._cluster
    same_coordinator = coordinator_lib.ClusterCoordinator(self.strategy)
    self.assertIs(self.coordinator, same_coordinator)
    self.assertIs(cluster, same_coordinator._cluster)

  def testFnReturnNestedValues(self):
    # [FUNCTION DOC]
    # Scenario: a scheduled traced function returns a nested structure of tuples, lists and a
    # mapping built from a constant. Expected: fetching via the remote value or via the
    # coordinator returns the same structure with plain values.
    x = constant_op.constant(1)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function returning a nested structure derived from the captured constant.
      return x + 1, (x + 2, x + 3), [x + 4], {'v': x}

    got = self.coordinator.schedule(f)
    want = 2, (3, 4), [5], {'v': 1}
    self.assertEqual(got.fetch(), want)
    self.assertEqual(self.coordinator.fetch(got), want)

  def testFetchingRemoteValueStructure(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is unconditionally skipped at the start as flaky, so the rest of the
    # body never runs.
    # Intended scenario: schedule a function returning a nested structure five times and fetch the
    # list of remote values through the coordinator; expected: five copies of the structure.
    self.skipTest('b/171040359: flaky test')
    x = constant_op.constant(1)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function returning a nested structure derived from the captured constant.
      return x + 1, (x + 2, x + 3), [x + 4], {'v': x}

    want = 2, (3, 4), [5], {'v': 1}
    remote_value_list = [self.coordinator.schedule(f) for _ in range(5)]
    self.assertAllEqual(
        self.coordinator.fetch(remote_value_list), [want for _ in range(5)])

  def testInputFunction(self):
    # [FUNCTION DOC]
    # Scenario: a per-worker dataset yielding the single element one, and a worker function that
    # takes the next element, adds it to a strategy-scoped variable and returns it. The function
    # is scheduled twice with fresh iterators. Expected: each fetched result is a tuple holding
    # one, and the variable ends at about two.

    def input_fn():
      # [FUNCTION DOC]
      # HELPER: returns a dataset with the single element one.
      return dataset_ops.DatasetV2.range(1, 2)

    with self.strategy.scope():
      v = variables.Variable(initial_value=0, dtype=dtypes.int64)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: takes the next element of the iterator, adds it to the variable and returns it.
      x = next(iterator)
      v.assign_add(x)
      return x

    distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(distributed_dataset),))
    result = self.coordinator.fetch(result)
    self.assertEqual(result, (1,))
    result = self.coordinator.schedule(
        worker_fn, args=(iter(distributed_dataset),))
    result = self.coordinator.fetch(result)

    self.assertEqual(result, (1,))
    self.assertAlmostEqual(v.read_value(), 2, delta=1e-6)

  def testAsyncScheduleAndJoin(self):
    # [FUNCTION DOC]
    # Scenario: a dataset of ten twos and a worker function adding the next element to an integer
    # variable. Actions: join with nothing scheduled (must not hang); schedule five and join
    # (variable equals ten); schedule five more and join several times; the coordinator reports
    # done and the variable equals twenty. Skipped when XLA is enabled (non-deterministic add).
    if test_util.is_xla_enabled():
      self.skipTest('Assign_add is not deterministic across threads in XLA')

    def input_fn():
      # [FUNCTION DOC]
      # HELPER: returns a dataset from a list of ten twos.
      return dataset_ops.DatasetV2.from_tensor_slices([2] * 10)

    with self.strategy.scope():
      v = variables.Variable(initial_value=0, dtype=dtypes.int32)

    # TODO(yuefengz): the following tf.function has a return value which is None
    # in its structured_outputs.
    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: takes the next element of the iterator and adds it to the variable; returns
      # nothing (structured output is the null value).
      x = next(iterator)
      v.assign_add(x)

    distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)

    iterator = iter(distributed_dataset)

    # Verifying joining without any scheduling doesn't hang.
    self.coordinator.join()
    self.assertEqual(v.read_value().numpy(), 0)

    for _ in range(5):
      self.coordinator.schedule(worker_fn, args=(iterator,))
    self.coordinator.join()

    # With 5 addition it should be 2*5 = 10.
    self.assertEqual(v.read_value().numpy(), 10)

    for _ in range(5):
      self.coordinator.schedule(worker_fn, args=(iterator,))

    # Verifying multiple join is fine.
    self.coordinator.join()
    self.coordinator.join()
    self.coordinator.join()

    self.assertTrue(self.coordinator.done())

    # Likewise, it's now 20.
    self.assertEqual(v.read_value().numpy(), 20.)

  @parameterized.parameters(True, False)
  def testInputFunctionWithMap(self, use_input_fn):
    # [FUNCTION DOC]
    # Parameterized by use_input_fn (boolean): pass the input function itself or the dataset it
    # returns to the per-worker dataset creation. Scenario: a range dataset mapped by a function
    # adding ten, counting how many times the map function is traced. Expected: the first element
    # fetched is a tuple holding ten and the map function was traced exactly once.
    self._map_fn_tracing_count = 0

    def input_fn():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset mapped with a function that adds ten.

      def map_fn(x):
        # [FUNCTION DOC]
        # HELPER: increments the tracing counter and returns the input plus ten.
        self._map_fn_tracing_count += 1
        return x + 10

      return dataset_ops.DatasetV2.range(0, 10).map(map_fn)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: returns the next element of the iterator.
      return next(iterator)

    if use_input_fn:
      distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    else:
      distributed_dataset = self.coordinator.create_per_worker_dataset(
          input_fn())

    result = self.coordinator.schedule(
        worker_fn, args=(iter(distributed_dataset),))
    self.assertEqual(result.fetch(), (10,))
    self.assertEqual(self._map_fn_tracing_count, 1)

  def testInputFunctionCreateVariables(self):
    # [FUNCTION DOC]
    # Scenario: an input function that creates a variable. Expected: creating a per-worker
    # dataset from it raises a value error.

    def input_fn():
      # [FUNCTION DOC]
      # HELPER: creates a variable and returns its read value (variable creation is disallowed).
      v = variables.Variable(initial_value=0.0)
      return v.read_value()

    with self.assertRaises(ValueError):
      self.coordinator.create_per_worker_dataset(input_fn)

  @parameterized.parameters(True, False)
  def testDatasetsShuffledDifferently(self, use_input_fn):
    # [FUNCTION DOC]
    # Parameterized by use_input_fn (boolean): pass the input function or its result. Scenario:
    # a shuffled, batched range dataset of one hundred elements distributed by the strategy,
    # created per worker. Resets the global random seed to unset. Expected: the elements fetched
    # from the first two workers' iterators are not all equal (shuffled differently); requires at
    # least two workers.
    # This test requires at least two workers in the cluster.
    self.assertGreaterEqual(len(self.coordinator._cluster.workers), 2)

    random_seed.set_random_seed(None)

    def input_fn():
      # [FUNCTION DOC]
      # HELPER: returns a distributed dataset of a shuffled, batch-size-one range of one hundred.
      dataset = dataset_ops.DatasetV2.range(0, 100).shuffle(100).batch(1)
      return self.strategy.experimental_distribute_dataset(dataset)

    if use_input_fn:
      distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    else:
      distributed_dataset = self.coordinator.create_per_worker_dataset(
          input_fn())
    distributed_iterator = iter(distributed_dataset)
    # Get elements from the first two iterators.
    iterator_1 = distributed_iterator._values[0]
    iterator_1 = iterator_1.fetch()
    elements_in_iterator_1 = [
        self.strategy.experimental_local_results(e)
        for e in iterator_1
    ]
    iterator_2 = distributed_iterator._values[1]
    iterator_2 = iterator_2.fetch()
    elements_in_iterator_2 = [
        self.strategy.experimental_local_results(e)
        for e in iterator_2
    ]

    self.assertNotAllEqual(elements_in_iterator_1, elements_in_iterator_2)

  def testPerWorkerValue(self):
    # [FUNCTION DOC]
    # Scenario: a non-traced function creating a float variable on a worker (asserting its device
    # contains the worker job) is turned into per-worker resources; each resource's type spec is
    # manually set as a workaround. A worker function adding one is scheduled ten times with the
    # per-worker value. Expected: after join, the sum of all per-worker variable values is ten.
    var_shape = tuple()
    var_dtype = dtypes.float32
    var_name = 'var'

    # This should not be a tf.function, as variable creation is prohibited in
    # tf.functions in general.
    def create_var():
      # [FUNCTION DOC]
      # HELPER: creates a float variable with initial value zero, asserts its device name mentions
      # a worker, and returns it.
      var = variables.Variable(
          initial_value=0.0, dtype=var_dtype, name=var_name)
      self.assertIn('worker', var.device)
      return var

    worker_local_var = self.coordinator._create_per_worker_resources(create_var)
    # The following is a workaround to allow `worker_local_var` to be passed in
    # as args to the `coordinator.schedule` method which requires specs
    # to trace a tf.function, but _create_worker_resources' return values don't
    # have specs when its input function is not a tf.function. We can get rid of
    # this workaround once _create_worker_resources is able to infer the tensor
    # spec of the return value of the (non-tf) function passed in. See
    # b/154675763.
    for var in worker_local_var._values:
      var._type_spec = resource_variable_ops.VariableSpec(var_shape, var_dtype)

    @def_function.function
    def worker_fn(var):
      # [FUNCTION DOC]
      # HELPER: adds one to the given variable; returns nothing.
      var.assign_add(1.0)

    for _ in range(10):
      # Which slice of `worker_local_var` will be used will depend on which
      # worker the `worker_fn` gets scheduled on.
      self.coordinator.schedule(worker_fn, args=(worker_local_var,))
    self.coordinator.join()

    var_sum = sum(self.coordinator.fetch(worker_local_var._values))
    self.assertEqual(var_sum, 10.0)

  @parameterized.parameters(True, False)
  def testPerWorkerVariableCreation(self, define_shape):
    # [FUNCTION DOC]
    # Parameterized by define_shape (boolean): whether the variable's shape is specified. Creates a
    # per-worker variable in the strategy scope, captured by a worker function that adds one.
    # Scenario: schedule, join, and read the summed value ten times, expecting the sum to grow by
    # one each time; then schedule ten more. Expected: each worker's variable lives on its own
    # worker CPU device and the final sum equals twice the number of closures (twenty).
    var_dtype = dtypes.float32
    var_name = 'var'
    shape = [1] if define_shape else None

    with self.strategy.scope():
      var = variables.Variable(
          initial_value=[0.0], shape=shape, dtype=var_dtype, name=var_name,
          per_worker_variable=True)

    # Use per-worker variable as a capture
    @def_function.function
    def worker_fn():
      # [FUNCTION DOC]
      # HELPER: adds a one-element tensor holding one to the captured per-worker variable and
      # returns the variable.
      var.assign_add(constant_op.constant([1.0]))
      return var

    num_closures = 10
    for ix in range(num_closures):
      self.coordinator.schedule(worker_fn)
      # Read the PWV many times to ensure result is up-to-date
      self.coordinator.join()
      result_sum = sum(var.read_all()).numpy()
      self.assertEqual(result_sum, ix + 1)

    for _ in range(num_closures):
      self.coordinator.schedule(worker_fn)
    self.coordinator.join()

    # Verify placement of variables
    devices = [wv._get_values().device for wv in var._per_worker_vars._values]
    expected_devices = [f'/job:worker/replica:0/task:{ix}/device:CPU:0'
                        for ix in range(self.strategy._num_workers)]  # pylint: disable=protected-access
    self.assertAllEqual(devices, expected_devices)

    result_sum = sum(var.read_all()).numpy()
    self.assertEqual(result_sum, num_closures * 2)

  def testDisallowRemoteValueAsInput(self):
    # [FUNCTION DOC]
    # Scenario: schedule a function returning one point zero, then schedule a second function with
    # that remote value as an argument. Expected: the second schedule raises a value error
    # because remote values are not allowed as inputs.
    @def_function.function
    def func_0():
      # [FUNCTION DOC]
      # HELPER: traced function returning the float one.
      return 1.0

    @def_function.function
    def func_1(x):
      # [FUNCTION DOC]
      # HELPER: traced function returning its input plus one.
      return x + 1.0

    remote_v = self.coordinator.schedule(func_0)
    with self.assertRaises(ValueError):
      self.coordinator.schedule(func_1, args=(remote_v,))

  def testPythonFunctionNotAllowedToSchedule(self):
    # [FUNCTION DOC]
    # Scenario: scheduling a plain (not traced) Python function. Expected: a type error stating
    # that schedule only accepts a traced function or a concrete function.

    def func(a):
      # [FUNCTION DOC]
      # HELPER: plain function returning an identity of its input; never executed successfully.
      return array_ops.identity(a)

    with self.assertRaisesRegex(
        TypeError,
        '`tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` '
        'only accepts a `tf.function` or a concrete function.',
    ):
      self.coordinator.schedule(func, args=(1,))

  def testDatasetPartiallyCreatedOnCoordinator(self):
    # [FUNCTION DOC]
    # Scenario: a dataset is created on the coordinator and only shuffled inside a traced input
    # function, then used per worker. Expected: scheduling a function on its iterator and joining
    # raises a closure input error whose message says copying the input tensor failed.
    dataset = dataset_ops.DatasetV2.range(1, 10)

    @def_function.function
    def input_fn():
      # [FUNCTION DOC]
      # HELPER: traced input function returning the captured coordinator dataset shuffled.
      return dataset.shuffle(9)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: returns the next element of the iterator.
      x = next(iterator)
      return x

    per_worker_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    self.coordinator.schedule(worker_fn, args=(iter(per_worker_dataset),))

    with self.assertRaisesRegex(
        coordinator_lib.ClosureInputError,
        'error message is Failed copying input tensor from',
    ):
      self.coordinator.join()

  def testPassDatasetToCreatePerWorkerDataset(self):
    # [FUNCTION DOC]
    # Scenario: a batched dataset (range one to ten, batch size four) is passed directly (not as a
    # function) to per-worker dataset creation; a function takes the next element. Expected: the
    # fetched result equals the first batch, one through four.
    dataset = dataset_ops.DatasetV2.range(1, 11).batch(4)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: returns the next element of the iterator.
      return next(iterator)

    per_worker_dataset = self.coordinator.create_per_worker_dataset(dataset)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(per_worker_dataset),))
    result = result.fetch()
    expected_result = math_ops.range(1., 5.)

    self.assertAllEqual(result, (expected_result))

  def testMultipleDatasets(self):
    # [FUNCTION DOC]
    # Scenario: two per-worker datasets (zero to four, five to nine) with iterators; a function
    # adds the next element of each. Expected: first result is five; then using a new iterator
    # over the first dataset together with the second iterator gives a result of at least five.

    def input_fn1():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset from zero to four.
      return dataset_ops.DatasetV2.range(0, 5)

    def input_fn2():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset from five to nine.
      return dataset_ops.DatasetV2.range(5, 10)

    per_worker_dataset1 = self.coordinator.create_per_worker_dataset(input_fn1)
    per_worker_iterator1 = iter(per_worker_dataset1)
    per_worker_dataset2 = self.coordinator.create_per_worker_dataset(input_fn2)
    per_worker_iterator2 = iter(per_worker_dataset2)

    @def_function.function
    def worker_fn(iterator1, iterator2):
      # [FUNCTION DOC]
      # HELPER: returns the sum of the next elements of both iterators.
      return next(iterator1) + next(iterator2)

    result = self.coordinator.schedule(
        worker_fn, args=(per_worker_iterator1, per_worker_iterator2))
    self.assertEqual(result.fetch(), 5.0)

    per_worker_dataset3 = self.coordinator.create_per_worker_dataset(input_fn1)
    per_worker_iterator3 = iter(per_worker_dataset3)

    result = self.coordinator.schedule(
        worker_fn, args=(per_worker_iterator3, per_worker_iterator2))
    self.assertGreaterEqual(result.fetch(), 5.0)

  def testRepeatedIteratorCreation(self):
    # [FUNCTION DOC]
    # Scenario: two per-worker datasets over one to ninety-nine; ten times create new iterators and
    # schedule eleven closures summing the next elements of both, then join. Expected: each first
    # result is at least two. After deleting the iterators and collecting garbage, no worker still
    # holds a live reference to an iterator resource (all weak references are dead).

    def input_fn():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset from one to ninety-nine.
      return dataset_ops.DatasetV2.range(1, 100)

    per_worker_dataset1 = self.coordinator.create_per_worker_dataset(input_fn)
    per_worker_dataset2 = self.coordinator.create_per_worker_dataset(input_fn)

    @def_function.function
    def worker_fn(iterator1, iterator2):
      # [FUNCTION DOC]
      # HELPER: returns the sum of the next elements of both iterators.
      return next(iterator1) + next(iterator2)

    for _ in range(10):
      per_worker_iterator1 = iter(per_worker_dataset1)
      per_worker_iterator2 = iter(per_worker_dataset2)
      result = self.coordinator.schedule(
          worker_fn, args=(per_worker_iterator1, per_worker_iterator2))
      for _ in range(10):
        self.coordinator.schedule(
            worker_fn, args=(per_worker_iterator1, per_worker_iterator2))
      self.coordinator.join()
      self.assertGreaterEqual(result.fetch(), 2.0)
    del per_worker_iterator1, per_worker_iterator2
    gc.collect()

    # There shouldn't be any live iterator objects.
    for w in self.coordinator._cluster.workers:
      for r in w._resource_remote_value_refs:
        self.assertIsNone(r())


class LimitedClosureQueueSizeBasicTest(ClusterCoordinatorTest):
  # [CLASS DOC]
  # KNOWN ISSUE: the class-level setup sets the module-level closure queue maximum size to two in
  # the coordinator module and never restores it, so the limit leaks into other test classes that
  # run afterwards in the same process.
  # Reruns every test of the main coordinator test class with the closure queue size limited to two.
  """Test basic functionality works with explicit maximum closure queue size.

  Execute the same set of test cases as in `ClusterCoordinatorTest`, with an
  explicit size limit for the closure queue. Note that even when the queue size
  is set to infinite, there is still a maximum practical size (depends on host
  memory limit) that might cause the queue.put operations to be blocking when
  scheduling a large number of closures on a big cluster. These tests make sure
  that the coordinator does not run into deadlocks in such scenario.
  """

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Runs the parent setup, sets the coordinator module's global closure queue maximum size to
    # two, then creates a new coordinator (five workers, two parameter servers) replacing the
    # parent's, and stores it and its strategy on the class. Mutates module-level state.
    super(LimitedClosureQueueSizeBasicTest, cls).setUpClass()
    coordinator_lib._CLOSURE_QUEUE_MAX_SIZE = 2
    cls.coordinator = make_coordinator(num_workers=5, num_ps=2)
    cls.strategy = cls.coordinator.strategy


class ScheduleStartDelayTest(ClusterCoordinatorTest):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says setting the environment variables "will cause any failure";
  # it clearly means it will not cause any failure.
  # Reruns every test of the main coordinator test class with the schedule start delay
  # environment variables set (two and four seconds) and a three-worker cluster.
  """Test basic functionality works with worker scheduling delay.

  This is basically to make sure that setting environment variables
  `TF_COORDINATOR_SCHEDULE_START_DELAY` and
  `TF_COORDINATOR_SCHEDULE_START_DELAY_MAX` will cause any failure.
  """

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Runs the parent setup, sets the process environment variables for schedule start delay
    # (two) and its maximum (four), then creates a coordinator with three workers and two
    # parameter servers, stored with its strategy on the class.
    super(ScheduleStartDelayTest, cls).setUpClass()
    os.environ['TF_COORDINATOR_SCHEDULE_START_DELAY'] = '2'
    os.environ['TF_COORDINATOR_SCHEDULE_START_DELAY_MAX'] = '4'
    cls.coordinator = make_coordinator(num_workers=3, num_ps=2)
    cls.strategy = cls.coordinator.strategy

  @classmethod
  def tearDownClass(cls):
    # [FUNCTION DOC]
    # Removes the two schedule start delay environment variables, then runs the parent teardown.
    del os.environ['TF_COORDINATOR_SCHEDULE_START_DELAY']
    del os.environ['TF_COORDINATOR_SCHEDULE_START_DELAY_MAX']
    super(ScheduleStartDelayTest, cls).tearDownClass()


class ErrorReportingTest(TestCaseWithErrorReportingThread):
  # [CLASS DOC]
  # Tests error reporting of the coordinator: errors from scheduled functions surface from join
  # or later schedule calls, are cleared afterwards, propagate through inputs, and cancel pending
  # closures. Uses a three-worker cluster and shared traced functions: a normal one, one that
  # fails an assertion, and a long-running one.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Runs the parent setup, creates a coordinator (three workers, two parameter servers), stores
    # it and its strategy on the class, and creates a float variable named iteration, initial
    # value zero, in the strategy scope.
    super(ErrorReportingTest, cls).setUpClass()
    cls.coordinator = make_coordinator(num_workers=3, num_ps=2)
    cls.strategy = cls.coordinator.strategy

    with cls.strategy.scope():
      cls.iteration = variables.Variable(initial_value=0.0)

  @def_function.function
  def _normal_function(self):
    # [FUNCTION DOC]
    # HELPER: traced function used as a successful workload. Multiplies two random matrices,
    # increments the shared iteration variable by one and returns the mean of the product.
    x = random_ops.random_uniform((2, 10))
    y = random_ops.random_uniform((10, 2))
    self.iteration.assign_add(1.0)
    return math_ops.reduce_mean(math_ops.matmul(x, y))

  @def_function.function
  def _error_function(self):
    # [FUNCTION DOC]
    # HELPER: traced function used as a failing workload. Asserts the sum of a random matrix
    # product is non-positive (almost always false, raising an invalid-argument error at run time),
    # then increments the iteration variable and returns it.
    x = random_ops.random_uniform((2, 10))
    y = random_ops.random_uniform((10, 2))
    check_ops.assert_non_positive_v2(math_ops.reduce_sum(math_ops.matmul(x, y)))
    self.iteration.assign_add(1.0)
    return self.iteration

  @def_function.function
  def _long_function(self):
    # [FUNCTION DOC]
    # HELPER: traced function used as a long-running workload. Accumulates ten thousand products
    # of random thousand-by-thousand matrices and returns the sum; meant to be cancelled.
    x = random_ops.random_uniform((1000, 1000))
    for _ in math_ops.range(10000):
      a = random_ops.random_uniform((1000, 1000))
      b = random_ops.random_uniform((1000, 1000))
      x += math_ops.matmul(a, b)
    return x

  def testJoinRaiseError(self):
    # [FUNCTION DOC]
    # Scenario: schedule three normal functions and one failing function. Expected: join raises an
    # invalid-argument error.
    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    self.coordinator.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

  def testScheduleRaiseError(self):
    # [FUNCTION DOC]
    # Scenario: schedule three normal functions and one failing function, then keep scheduling
    # normal functions in a loop. Expected: eventually a schedule call raises an invalid-argument
    # error.
    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    self.coordinator.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      while True:
        self.coordinator.schedule(self._normal_function)

  def testScheduleRaiseErrorWithMultipleFailure(self):
    # [FUNCTION DOC]
    # Scenario: schedule normal functions and a failing function, then keep scheduling failing
    # functions. Expected: a schedule call raises an invalid-argument error, and a subsequent join
    # does not raise (the error was already consumed).
    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    self.coordinator.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      while True:
        self.coordinator.schedule(self._error_function)
    self.coordinator.join()

  def testErrorWillbeCleared(self):
    # [FUNCTION DOC]
    # Scenario: a failing function is scheduled and join raises an invalid-argument error. Then
    # three normal functions and another failing function are scheduled. Expected: the second join
    # again raises an invalid-argument error, showing the earlier error was cleared.
    self.coordinator.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    self.coordinator.schedule(self._error_function)
    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

  def testRemoteValueReturnError(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is unconditionally skipped at the start (pending a bug fix for open
    # source tests), so the remaining body never runs.
    # Intended scenario: scheduling a failing function; fetching its remote value raises an
    # invalid-argument error, and join then raises it once more (clearing the error).
    self.skipTest('TODO(b/211502459): Fix this in OSS test.')

    result = self.coordinator.schedule(self._error_function)

    with self.assertRaises(errors.InvalidArgumentError):
      result.fetch()

    # Clear the error.
    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

  def testInputError(self):
    # [FUNCTION DOC]
    # Scenario: per-worker resources are created from the failing function, then a function taking
    # them as input is scheduled. Expected: join raises a closure input error, and fetching the
    # scheduled function's result also raises a closure input error.

    worker_local_val = self.coordinator._create_per_worker_resources(
        self._error_function)

    @def_function.function
    def func(x):
      # [FUNCTION DOC]
      # HELPER: traced function returning its input plus one.
      return x + 1

    result = self.coordinator.schedule(func, args=(worker_local_val,))
    with self.assertRaises(coordinator_lib.ClosureInputError):
      self.coordinator.join()

    with self.assertRaises(coordinator_lib.ClosureInputError):
      result.fetch()

  def testErroredInputNotUsed(self):
    # [FUNCTION DOC]
    # Scenario: two sets of per-worker resources are created, one from the normal function and one
    # from the failing function; a function is scheduled using only the normal one. Expected: join
    # does not raise and the result can be fetched.
    input_0 = self.coordinator._create_per_worker_resources(
        self._normal_function)

    self.coordinator._create_per_worker_resources(
        self._error_function)

    @def_function.function
    def func(x):
      # [FUNCTION DOC]
      # HELPER: traced function returning its input plus one.
      return x + 1

    result = self.coordinator.schedule(func, args=(input_0,))

    # It should not raise.
    self.coordinator.join()
    result.fetch()

  def testCancellation(self):
    # [FUNCTION DOC]
    # Scenario: schedule three normal functions, the long-running function and a failing function.
    # Expected: join raises an invalid-argument error; the long-running function's result fetch
    # raises a cancelled error; afterwards scheduling and joining normal functions works.
    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    long_function = self.coordinator.schedule(self._long_function)
    self.coordinator.schedule(self._error_function)

    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

    with self.assertRaises(errors.CancelledError):
      long_function.fetch()

    for _ in range(3):
      self.coordinator.schedule(self._normal_function)
    self.coordinator.join()

  def testResourceCanStillbeUsedAfterCancellation(self):
    # [FUNCTION DOC]
    # Scenario: a per-worker dataset iterator is used by a scheduled function, then a failing
    # function is scheduled and join raises an invalid-argument error (cancelling pending work).
    # Expected: the same iterator resource can still be used by another scheduled function, and
    # the final join succeeds.

    def input_fn():
      # [FUNCTION DOC]
      # HELPER: returns a range dataset from zero to four.
      return dataset_ops.DatasetV2.range(0, 5)

    per_worker_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    per_worker_iterator = iter(per_worker_dataset)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: returns the next element of the iterator.
      return next(iterator)

    self.coordinator.schedule(worker_fn, args=(per_worker_iterator,))
    self.coordinator.schedule(self._error_function)

    with self.assertRaises(errors.InvalidArgumentError):
      self.coordinator.join()

    self.coordinator.schedule(worker_fn, args=(per_worker_iterator,))
    self.coordinator.join()


class LimitedClosureQueueErrorTest(ErrorReportingTest):
  # [CLASS DOC]
  # KNOWN ISSUE: the class-level setup sets the module-level closure queue maximum size to two in
  # the coordinator module and never restores it, so the limit leaks into later test classes.
  # Reruns every error reporting test with the closure queue size limited to two.
  """Test error reporting works with explicit maximum closure queue size.

  Execute the same set of test cases as in ErrorReportingTest, with an explicit
  size limit for the closure queue.
  """

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Runs the parent setup, sets the coordinator module's global closure queue maximum size to
    # two, creates a new coordinator (three workers, two parameter servers) and stores it and its
    # strategy on the class, and creates the float iteration variable in the strategy scope.
    # Mutates module-level state.
    super(LimitedClosureQueueErrorTest, cls).setUpClass()
    coordinator_lib._CLOSURE_QUEUE_MAX_SIZE = 2
    cls.coordinator = make_coordinator(num_workers=3, num_ps=2)
    cls.strategy = cls.coordinator.strategy

    with cls.coordinator.strategy.scope():
      cls.iteration = variables.Variable(initial_value=0.0)


class ShardedVariableTest(TestCaseWithErrorReportingThread):
  # [CLASS DOC]
  # Tests sharded variables used with the cluster coordinator, using a coordinator whose strategy
  # partitions variables into three fixed shards.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Runs the parent setup, then creates a coordinator with five workers and two parameter
    # servers and a fixed-shards partitioner of three shards, stored with its strategy on the class.
    super().setUpClass()
    cls.coordinator = make_coordinator(
        num_workers=5, num_ps=2,
        partitioner=sharded_variable.FixedShardsPartitioner(3))
    cls.strategy = cls.coordinator.strategy

  def testEmbeddingLookup(self):
    # [FUNCTION DOC]
    # Scenario: a sharded float variable of shape five by two is used by three traced lookup
    # functions (dense lookup, sparse lookup, safe sparse lookup with weights). Each is wrapped in
    # a closure built manually so its graph can be inspected. Expected: the number of gather
    # operations on resources equals the number of shards and there are no variable read
    # operations (the sharded variable was not converted to a tensor); fetched results match the
    # expected embedding rows.
    # Verify lookup ops use optimized implementations with ClusterCoordinator
    with self.strategy.scope():
      sv = variables.Variable(initial_value=np.arange(10).reshape((5, 2)) + 1,
                              dtype=dtypes.float32)

    @def_function.function
    def lookup():
      # [FUNCTION DOC]
      # HELPER: traced function doing a dense embedding lookup of ids zero, three and four.
      ids = constant_op.constant([0, 3, 4])
      return embedding_ops.embedding_lookup_v2(sv, ids)

    @def_function.function
    def sparse_lookup():
      # [FUNCTION DOC]
      # HELPER: traced function doing a sparse embedding lookup (combined per row) with no weights.
      sp_ids = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0], [2, 2]],
          values=[0, 3, 4, 1],
          dense_shape=[3, 3])
      return embedding_ops.embedding_lookup_sparse_v2(sv, sp_ids, None)

    @def_function.function
    def safe_sparse_lookup():
      # [FUNCTION DOC]
      # HELPER: traced function doing a safe sparse embedding lookup with sparse weights, where one
      # id is invalid (negative) and the weights include a negative value.
      sp_ids = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0], [2, 2]],
          values=[0, -1, 4, 1],
          dense_shape=[3, 3])
      sp_weights = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0], [2, 2]],
          values=[1., 1., -1., 1.],
          dense_shape=[3, 3])
      return embedding_ops.safe_embedding_lookup_sparse_v2(
          sv, sp_ids, sp_weights)

    results = []
    for func in [lookup, sparse_lookup, safe_sparse_lookup]:
      # Manually create Closure so we can inspect its graph
      closure = coordinator_lib.Closure(
          func,
          self.coordinator._cluster.closure_queue._cancellation_mgr)  # pylint: disable=protected-access
      result = closure.build_output_remote_value()
      self.coordinator._cluster.closure_queue.put(closure)

      graph = closure._concrete_function.graph
      num_gather_ops = 0
      num_rv_ops = 0
      for op in graph.get_operations():
        if op.type == 'ResourceGather':
          num_gather_ops += 1
        if op.type == 'ReadVariableOp':
          num_rv_ops += 1
      self.assertEqual(
          num_gather_ops, len(sv.variables), 'Number of ResourceGather op '
          f'({num_gather_ops}) does not match expected ({len(sv.variables)}), '
          'possibly due to ShardedVariable accidentally being converted to '
          f'tensor in {func.__name__} ops.')
      self.assertEqual(
          num_rv_ops, 0, f'Function {func.__name__} graph has some '
          'ReadVariableOps, possibly due to ShardedVariable accidentally being '
          'converted to tensor')
      results.append(result)

    self.assertAllEqual(results[0].fetch(), [[1., 2.], [7., 8.], [9., 10.]])
    self.assertAllClose(results[1].fetch(), [[4., 5.], [9., 10.], [3., 4.]])
    self.assertAllClose(results[2].fetch(), [[1., 2.], [0., 0.], [3., 4.]])


class StrategyIntegrationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests of how the parameter server strategy APIs (run, reduce, variables, aggregation, caching,
  # distributed datasets) work together with the cluster coordinator. Uses a class-level
  # coordinator with one worker and one parameter server.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Runs the parent setup, then creates a coordinator with one worker and one parameter server
    # and stores it and its strategy on the class.
    super(StrategyIntegrationTest, cls).setUpClass()
    cls.coordinator = make_coordinator(num_workers=1, num_ps=1)
    cls.strategy = cls.coordinator.strategy

  def testRunNotUsedWithClusterCoordinatorSchedule(self):
    # [FUNCTION DOC]
    # Scenario: the strategy run method is invoked directly on the coordinator (not inside a
    # scheduled function), both before and after a proper scheduled call. Expected: each direct
    # invocation logs a warning (captured at warning level) saying the method is invoked without
    # using the coordinator's schedule; the scheduled call succeeds and its fetched results match
    # the expected pairs for each replica.

    @def_function.function
    def input_fn():
      # [FUNCTION DOC]
      # HELPER: traced input function returning a range dataset with elements one and two.
      return dataset_ops.DatasetV2.range(1, 3)

    with self.strategy.scope():
      v = variables.Variable(initial_value=1, dtype=dtypes.int64)

      def replica_fn(input_tensor):
        # [FUNCTION DOC]
        # HELPER: returns the pair of input plus variable and input minus variable.
        return input_tensor + v, input_tensor - v

      @def_function.function
      def worker_fn(iterator):
        # [FUNCTION DOC]
        # HELPER: runs the replica function on the next element of the iterator via the strategy.
        return self.strategy.run(replica_fn, args=(next(iterator),))

    per_worker_dataset = self.coordinator.create_per_worker_dataset(input_fn)

    @contextlib.contextmanager
    def _assert_logs_usage_warning():
      # [FUNCTION DOC]
      # HELPER: context manager asserting that the enclosed block logs at least one warning, and
      # that the first logged message contains the text saying a strategy method is invoked
      # without using the coordinator's schedule. Yields nothing.
      with self.assertLogs(level='WARNING') as logs:
        yield

      self.assertIn(
          'A `tf.distribute.experimental.ParameterServerStrategy` method is '
          'invoked without using `ClusterCoordinator.schedule`. If you are not '
          'tracing a tf.function, this method is possibly executed on the '
          'coordinator, which can be slow. To properly dispatch functions to '
          'run on workers, methods like `run` or `reduce` should be used '
          'within a function passed to `tf.distribute.experimental.coordinator.'
          'ClusterCoordinator.schedule`.', logs.output[0])

    with _assert_logs_usage_warning():
      # Invoking `run` without `coordinator.schedule` should result in a
      # warning.
      self.strategy.run(
          replica_fn, args=(constant_op.constant(1, dtype=dtypes.int64),))

    # A proper `schedule` should succeed.
    rv = self.coordinator.schedule(worker_fn, args=(iter(per_worker_dataset),))

    with _assert_logs_usage_warning():
      # Invoking `run` without `coordinator.schedule` again should result in a
      # warning.
      self.strategy.run(
          replica_fn, args=(constant_op.constant(1, dtype=dtypes.int64),))

    all_results = [(2, 0)] * self.strategy.num_replicas_in_sync
    expected_result = []
    for i in range(self.strategy.num_replicas_in_sync):
      expected_result.append(all_results[i])

    self.assertAllEqual(
        tuple(expected_result),
        self.strategy.experimental_local_results(rv.fetch()))

  def testBasicVariableAssignment(self):
    # [FUNCTION DOC]
    # Scenario: reset the strategy's variable counter, create two variables (zero and one) in the
    # strategy scope and check the counter is two. A scheduled function adds 0.1 to the first,
    # subtracts 0.2 from the second and returns their ratio. Expected: the variables end at 0.1 and
    # 0.8.
    self.strategy.extended._variable_count = 0
    with self.strategy.scope():
      v1 = variables.Variable(initial_value=0.0)
      v2 = variables.Variable(initial_value=1.0)
    self.assertEqual(self.strategy.extended._variable_count, 2)

    @def_function.function
    def worker_fn():
      # [FUNCTION DOC]
      # HELPER: updates both variables and returns the ratio of the first to the second.
      v1.assign_add(0.1)
      v2.assign_sub(0.2)
      return v1.read_value() / v2.read_value()

    results = self.coordinator.schedule(worker_fn)
    logging.info('Results of experimental_run_v2: %f',
                 self.coordinator.fetch(results))

    self.assertAlmostEqual(v1.read_value().numpy(), 0.1, delta=1e-6)
    self.assertAlmostEqual(v2.read_value().numpy(), 0.8, delta=1e-6)

  def testRunAndReduce(self):
    # [FUNCTION DOC]
    # Scenario: inside the strategy scope (cross-replica context) a function runs a replica
    # function on a constant input three and sum-reduces the result, asserting equality with the
    # expected pair (four and two times the replica count). The scheduled function is called both
    # inside and outside the scope. Expected: the result is a remote value whose fetch equals the
    # expected pair in both cases; the replica function asserts it is not in cross-replica context.
    self.assertFalse(distribute_lib.in_cross_replica_context())
    with self.strategy.scope():
      self.assertTrue(distribute_lib.in_cross_replica_context())
      v = variables.Variable(initial_value=1.)

      expected_result = (4. * self.strategy.num_replicas_in_sync,
                         2. * self.strategy.num_replicas_in_sync)

      @def_function.function
      def worker_fn(input_tensor):
        # [FUNCTION DOC]
        # HELPER: runs the replica function with the strategy, sum-reduces the per-replica pair,
        # asserts it equals the expected result and returns the reduced pair.

        def replica_fn(input_tensor):
          # [FUNCTION DOC]
          # HELPER: asserts replica context (not cross-replica) and returns the pair of input plus
          # variable and input minus variable.
          # Within `replica_fn`, it has to be in a replica context.
          self.assertFalse(
              distribute_lib.in_cross_replica_context())
          return input_tensor + v, input_tensor - v

        run_result = self.strategy.run(replica_fn, args=(input_tensor,))
        reduced_result = self.strategy.reduce('SUM', run_result, axis=None)
        check_ops.assert_equal_v2(reduced_result, expected_result)
        return reduced_result

      # Asserting scheduling in scope has the expected behavior.
      result = self.coordinator.schedule(
          worker_fn, args=(constant_op.constant(3.),))
      self.assertIsInstance(result, coordinator_lib.RemoteValue)
      self.assertEqual(result.fetch(), expected_result)

    # Asserting scheduling out of scope has the expected behavior.
    result = self.coordinator.schedule(
        worker_fn, args=(constant_op.constant(3.),))
    self.assertEqual(result.fetch(), expected_result)

  def testRunAndReduceWithAssignAdd(self):
    # [FUNCTION DOC]
    # Same as the run-and-reduce test, plus a second variable aggregated by the first replica only
    # that gets the input added inside the replica function. The scheduled function runs twice
    # (inside and outside the scope) with input three. Expected: results equal the expected pair
    # both times and the second variable ends at six.
    self.assertFalse(distribute_lib.in_cross_replica_context())
    with self.strategy.scope():
      self.assertTrue(distribute_lib.in_cross_replica_context())
      v = variables.Variable(initial_value=1.)
      v1 = variables.Variable(
          initial_value=0.,
          aggregation=variable_scope.VariableAggregation.ONLY_FIRST_REPLICA)

      expected_result = (4. * self.strategy.num_replicas_in_sync,
                         2. * self.strategy.num_replicas_in_sync)

      @def_function.function
      def worker_fn(input_tensor):
        # [FUNCTION DOC]
        # HELPER: runs the replica function with the strategy, sum-reduces, asserts equality with
        # the expected pair and returns the reduced pair.

        def replica_fn(input_tensor):
          # [FUNCTION DOC]
          # HELPER: asserts replica context, adds the input to the aggregated variable, and returns
          # the pair of input plus variable and input minus variable.
          # Within `replica_fn`, it has to be in a replica context.
          self.assertFalse(
              distribute_lib.in_cross_replica_context())

          v1.assign_add(input_tensor)
          return input_tensor + v, input_tensor - v

        run_result = self.strategy.run(replica_fn, args=(input_tensor,))
        reduced_result = self.strategy.reduce('SUM', run_result, axis=None)
        check_ops.assert_equal_v2(reduced_result, expected_result)
        return reduced_result

      # Asserting scheduling in scope has the expected behavior.
      result = self.coordinator.schedule(
          worker_fn, args=(constant_op.constant(3.),))
      self.assertIsInstance(result, coordinator_lib.RemoteValue)
      self.assertEqual(result.fetch(), expected_result)

    # Asserting scheduling out of scope has the expected behavior.
    result = self.coordinator.schedule(
        worker_fn, args=(constant_op.constant(3.),))
    self.assertEqual(result.fetch(), expected_result)
    self.assertEqual(v1, 6.)

  def testVariableAggregation(self):
    # [FUNCTION DOC]
    # Scenario: a variable with sum aggregation is assigned, in each replica, the replica id plus
    # one. A scheduled function does this and the test joins. Expected: the variable equals the
    # sum of one through the number of replicas.
    self.assertFalse(distribute_lib.in_cross_replica_context())
    with self.strategy.scope():
      self.assertTrue(distribute_lib.in_cross_replica_context())
      v = variables.Variable(
          initial_value=1.,
          aggregation=variable_scope.VariableAggregation.SUM)

      @def_function.function
      def worker_fn():
        # [FUNCTION DOC]
        # HELPER: runs the replica function with the strategy; returns nothing.

        def replica_fn():
          # [FUNCTION DOC]
          # HELPER: assigns the variable the replica id in the sync group plus one, cast to the
          # variable's dtype.
          value = math_ops.cast(
              distribute_lib.get_replica_context()
              .replica_id_in_sync_group + 1, v.dtype)
          v.assign(value)

        self.strategy.run(replica_fn)

      self.coordinator.schedule(worker_fn)
      self.coordinator.join()
      expected_result = 0.
      for i in range(self.strategy.num_replicas_in_sync):
        expected_result = expected_result + i + 1
      self.assertEqual(v, expected_result)

  def testVariableCaching(self):
    # [FUNCTION DOC]
    # Scenario: variable reads inside the cache-variable-reads scope return the first value read
    # even after assignment (both direct reads and tensor conversions), checked directly, inside a
    # scheduled traced function (returns the cached one rather than five), and with the caching
    # scope entered around the schedule call outside the function (cached two). Also verifies
    # that reading outside the scope returns current values and that nesting the caching scope
    # raises a value error.
    self.assertFalse(distribute_lib.in_cross_replica_context())
    with self.strategy.scope():
      self.assertTrue(distribute_lib.in_cross_replica_context())
      v = variables.Variable(
          initial_value=1.,
          aggregation=variable_scope.VariableAggregation.ONLY_FIRST_REPLICA)

      # Test read value inside caching scope
      with distribute_utils.cache_variable_reads():
        v.read_value()  # Reads value 1.0
        v.assign(constant_op.constant(5.0))  # v changes to 5.0
        self.assertEqual(v.read_value(), 1.0)  # should be cached 1.0 value.

      # Reset v to 2.0
      v.assign(2.0)

      # Test convert to tensor value inside caching scope
      with distribute_utils.cache_variable_reads():
        t = v * 3.0
        self.assertEqual(t, 6.0)
        v.assign(3.0)
        t1 = v * 3.0
        self.assertEqual(t1, 6.0)  # should be cached 2.0 * 3.0 value.

      # Reset v to 1.0
      v.assign(1.0)

      # Verify caching scope inside tf.function
      @def_function.function
      def worker_fn():
        # [FUNCTION DOC]
        # HELPER: inside the caching scope, runs a replica function with the strategy and returns
        # its result (expected to be the cached value one).
        with distribute_utils.cache_variable_reads():
          def replica_fn():
            # [FUNCTION DOC]
            # HELPER: reads the variable, assigns five, reads again and returns the second read,
            # which should still be the cached original value.
            t = v.read_value()  # Reads value 1.0
            v.assign(constant_op.constant(5.0))  # v changes to 5.0
            t = v.read_value()  # should return 1.0
            return t  # Should be 1.0 instead of 5.0

          return self.strategy.run(replica_fn)

      result = self.coordinator.schedule(worker_fn)
      result = result.fetch()
      expected_result = 1.
      self.assertEqual(result, expected_result)

      # Verify that v.read_value works as expected outside of scope.
      v.assign(4.0)
      self.assertEqual(v.read_value(), 4.0)

      v.assign(constant_op.constant(2.0))  # v changes to 2.0
      # Check with scope outside of tf function and check that cache is reset
      @def_function.function
      def worker_fn1():
        # [FUNCTION DOC]
        # HELPER: runs a replica function with the strategy and returns its result (expected to be
        # the cached value two, because scheduling happens inside a caching scope).
        def replica_fn():
          # [FUNCTION DOC]
          # HELPER: reads the variable, assigns five, reads again and returns the second read,
          # which should be the cached value two.
          t = v.read_value()  # Reads value 2.0 ==> Should be cached
          v.assign(constant_op.constant(5.0))  # v changes to 5.0
          t = v.read_value()  # should return cached value 2.0
          return t  # Should be 2.0 instead of 5.0

        return self.strategy.run(replica_fn)

      with distribute_utils.cache_variable_reads():
        result = self.coordinator.schedule(worker_fn1)
      result = result.fetch()
      expected_result = 2.
      self.assertEqual(result, expected_result)

    # Verify scope nesting is not permitted.
    with self.assertRaises(ValueError):
      with distribute_utils.cache_variable_reads():
        with distribute_utils.cache_variable_reads():
          v.read_value()

  @parameterized.parameters(True, False)
  def testDistributedDatasetInsidePerWorkerDatasetFn(self, from_function):
    # [FUNCTION DOC]
    # Parameterized by from_function (boolean): the per-worker dataset function builds the
    # distributed dataset either from a dataset function or from a dataset. Scenario: batches of
    # four from one to ten; a scheduled function returns the local per-replica results of the next
    # element. Expected: the result equals the first batch split evenly across replicas.
    if from_function:

      def per_worker_dataset_fn():
        # [FUNCTION DOC]
        # HELPER: returns a distributed dataset created from a dataset function (range one to ten,
        # batch of four).
        dataset_fn = lambda _: dataset_ops.DatasetV2.range(1, 11).batch(4)
        return self.strategy.distribute_datasets_from_function(dataset_fn)
    else:

      def per_worker_dataset_fn():
        # [FUNCTION DOC]
        # HELPER: returns a distributed dataset created from a dataset (range one to ten, batch
        # of four).
        dataset = dataset_ops.DatasetV2.range(1, 11).batch(4)
        return self.strategy.experimental_distribute_dataset(dataset)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: returns the per-replica local results of the next element of the iterator.
      return self.strategy.experimental_local_results(next(iterator))

    per_worker_dataset = self.coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(per_worker_dataset),))
    result = result.fetch()
    expected_result = array_ops.split(
        math_ops.range(1., 5.),
        num_or_size_splits=self.strategy.num_replicas_in_sync,
        axis=0)

    self.assertAllEqual(result, (expected_result))

  @parameterized.parameters(True, False)
  def testPassDistributedDatasetToCreatePerWorkerDataset(self, from_function):
    # [FUNCTION DOC]
    # Parameterized by from_function (boolean): the distributed dataset is built from a dataset
    # function or from a dataset (range one to ten, batch of four) on the coordinator and passed
    # directly to per-worker dataset creation. Expected: a scheduled function returns the next
    # element's local results, equal to the first batch split evenly across replicas.
    if from_function:
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(1, 11).batch(4)
      distributed_dataset = self.strategy.distribute_datasets_from_function(
          dataset_fn)
    else:
      dataset = dataset_ops.DatasetV2.range(1, 11).batch(4)
      distributed_dataset = self.strategy.experimental_distribute_dataset(
          dataset)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: returns the per-replica local results of the next element of the iterator.
      return self.strategy.experimental_local_results(next(iterator))

    per_worker_dataset = self.coordinator.create_per_worker_dataset(
        distributed_dataset)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(per_worker_dataset),))
    result = result.fetch()
    expected_result = array_ops.split(
        math_ops.range(1., 5.),
        num_or_size_splits=self.strategy.num_replicas_in_sync,
        axis=0)

    self.assertAllEqual(result, (expected_result))

  def testDistributeDatasetsFromFunction(self):
    # [FUNCTION DOC]
    # Scenario: the per-worker dataset function distributes a dataset (range one to ten, batch of
    # one) from an input function that must receive a non-null input context. A scheduled function
    # returns the local results of the next element. Expected: one entry per replica containing
    # one plus the replica position.

    def per_worker_dataset_fn():
      # [FUNCTION DOC]
      # HELPER: returns the strategy's distributed dataset built from the input function below.

      def input_worker_device_fn(input_context):
        # [FUNCTION DOC]
        # HELPER: asserts the input context is not null and returns a range dataset from one to
        # ten with batch size one.
        self.assertIsNotNone(input_context)
        return dataset_ops.DatasetV2.range(1, 11).batch(1)

      return self.strategy.distribute_datasets_from_function(
          input_worker_device_fn)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: returns the per-replica local results of the next element of the iterator.
      result = self.strategy.experimental_local_results(next(iterator))
      return result

    distributed_dataset = self.coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(distributed_dataset),))
    result = result.fetch()
    expected_result = []
    for i in range(self.strategy.num_replicas_in_sync):
      expected_result.append([1 + i])
    self.assertAllEqual(result, expected_result)

  def testAsyncScheduleWithDistributedDataset(self):
    # [FUNCTION DOC]
    # Scenario: an endless distributed dataset of twos (batched by replica count) and a function
    # that mean-reduces the next element and adds it to a variable. Actions: join with nothing
    # scheduled; schedule five and join (variable local results equal ten); schedule five more,
    # join repeatedly; done is true and the variable equals twenty.

    def input_fn():
      # [FUNCTION DOC]
      # HELPER: returns a distributed dataset of the repeated value two, batched by the number of
      # replicas in sync.
      dataset = dataset_ops.DatasetV2.from_tensor_slices([2.]).repeat().batch(
          self.strategy.num_replicas_in_sync)
      return self.strategy.experimental_distribute_dataset(dataset)

    with self.strategy.scope():
      v = variables.Variable(initial_value=[0], dtype=dtypes.float32)

    # TODO(yuefengz): the following tf.function has a return value which is None
    # in its structured_outputs.
    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: takes the next element, mean-reduces it across replicas and adds the result to the
      # variable; returns nothing.
      x = next(iterator)
      # Reduce to convert PerReplica values to single value
      reduced_value = self.strategy.reduce('MEAN', x, axis=None)
      v.assign_add(reduced_value)

    distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)

    iterator = iter(distributed_dataset)

    # Verifying joining without any scheduling doesn't hang.
    self.coordinator.join()
    self.assertAllEqual(v.read_value(), (0,))

    for _ in range(5):
      self.coordinator.schedule(worker_fn, args=(iterator,))
    self.coordinator.join()

    # With 5 addition it should be 2*5 = 10.
    self.assertAllEqual(
        self.strategy.experimental_local_results(v.read_value()), ([[10]]))

    for _ in range(5):
      self.coordinator.schedule(worker_fn, args=(iterator,))

    # Verifying multiple join is fine.
    self.coordinator.join()
    self.coordinator.join()
    self.coordinator.join()

    self.assertTrue(self.coordinator.done())

    # Likewise, it's now 20.
    self.assertAllEqual(
        self.strategy.experimental_local_results(v.read_value()), ([[20]]))

  def testInputFunctionWithMapWithDistributedDataset(self):
    # [FUNCTION DOC]
    # Scenario: a range dataset batched by replica count and mapped by a function adding ten,
    # counting traces, distributed by the strategy, used per worker. Expected: the next element's
    # local results equal the first batch (ten onward) split across replicas and the map function
    # was traced exactly once.
    self._map_fn_tracing_count = 0

    def input_fn():
      # [FUNCTION DOC]
      # HELPER: returns the strategy-distributed, batched and mapped dataset.

      def map_fn(x):
        # [FUNCTION DOC]
        # HELPER: increments the tracing counter and returns the input plus ten.
        self._map_fn_tracing_count += 1
        return x + 10

      dataset = dataset_ops.DatasetV2.range(0, 10).batch(
          self.strategy.num_replicas_in_sync).map(map_fn)
      return self.strategy.experimental_distribute_dataset(dataset)

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: returns the next element of the iterator.
      return next(iterator)

    distributed_dataset = self.coordinator.create_per_worker_dataset(input_fn)
    result = self.coordinator.schedule(
        worker_fn, args=(iter(distributed_dataset),))

    expected_result = array_ops.split(
        math_ops.range(10., 10. + self.strategy.num_replicas_in_sync),
        num_or_size_splits=self.strategy.num_replicas_in_sync,
        axis=0)

    self.assertAllEqual(
        self.strategy.experimental_local_results(result.fetch()),
        tuple(expected_result))
    self.assertEqual(self._map_fn_tracing_count, 1)

  def testPerWorkerDistributeDatasetsElementSpec(self):
    # [FUNCTION DOC]
    # Scenario: a per-worker dataset created from a function distributing a two-element dataset.
    # Expected: its element spec equals the distributed tensor spec derived from the plain
    # dataset's element spec for the strategy (a per-replica spec when more than one replica).

    def per_worker_dataset_fn():
      # [FUNCTION DOC]
      # HELPER: returns a distributed dataset built from a function yielding a two-element dataset.
      return self.strategy.distribute_datasets_from_function(
          lambda _: dataset_ops.DatasetV2.from_tensor_slices([1, 2]))

    dataset = dataset_ops.DatasetV2.from_tensor_slices([1, 2])
    per_worker_distribute_dataset = self.coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)

    self.assertAllEqual(
        # Converts to PerReplicaSpec when num_replicas_in_sync are > 1
        input_lib._create_distributed_tensor_spec(self.strategy,
                                                  dataset.element_spec),
        per_worker_distribute_dataset.element_spec)

  def testPerWorkerDistributedIteratorTypeSpec(self):
    # [FUNCTION DOC]
    # Scenario: a per-worker distributed dataset function that counts how many times it runs; an
    # iterator is created, a concrete function of the worker function is traced with it, and a
    # function is scheduled with it. Expected: the dataset function ran exactly once (the iterator
    # type spec is available without re-creating the dataset).
    self._tracing_count = 0

    def per_worker_dataset_fn():
      # [FUNCTION DOC]
      # HELPER: increments the run counter and returns a distributed dataset of a single element
      # (range one to two).
      self._tracing_count += 1
      return self.strategy.distribute_datasets_from_function(
          lambda _: dataset_ops.DatasetV2.range(1, 2))

    @def_function.function
    def worker_fn(iterator):
      # [FUNCTION DOC]
      # HELPER: returns the next element of the iterator.
      return next(iterator)

    distributed_iterator = iter(
        self.coordinator.create_per_worker_dataset(per_worker_dataset_fn))
    worker_fn.get_concrete_function(distributed_iterator)

    self.coordinator.schedule(worker_fn, args=(distributed_iterator,))
    self.assertEqual(self._tracing_count, 1)

  def testPerWorkerDatasetBuild(self):
    # [FUNCTION DOC]
    # Scenario: a per-worker dataset created from a two-element dataset is built explicitly and
    # passed as a dataset (not an iterator) to a scheduled function that returns the first element.
    # Expected: the fetched result equals the first value, and each per-worker value's type spec
    # equals the source dataset's type spec.
    # Test that we can use Dataset type as input to worker_fn.
    @def_function.function
    def worker_fn(dataset):
      # [FUNCTION DOC]
      # HELPER: returns the first element of an iterator made from the given dataset.
      return next(iter(dataset))

    dataset_vals = [1, 2]
    dataset = dataset_ops.DatasetV2.from_tensor_slices(dataset_vals)
    per_worker_dataset = self.coordinator.create_per_worker_dataset(dataset)
    per_worker_dataset = per_worker_dataset.build()

    result = self.coordinator.schedule(worker_fn, args=(per_worker_dataset,))
    self.coordinator.join()
    result = result.fetch()
    self.assertEqual(result, dataset_vals[0])

    # Test that the build() output type specs match the input Dataset spec.
    for value in per_worker_dataset._values:
      self.assertEqual(value._type_spec, dataset._type_spec)


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
