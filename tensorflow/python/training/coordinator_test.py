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
# [MODULE DOC]
# Unit tests for the Coordinator and LooperThread classes: stop signalling, joining registered
# and explicit threads, grace period handling, exception reporting (bare exceptions and
# exception-info tuples, clean-stop types, the stop-on-exception context handler), clearing
# stops, request_stop after join, and looper thread argument passing. Contains small
# thread-body helper functions used by the tests. No import-time side effects.
"""Tests for Coordinator."""

import sys
import threading
import time

from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator


def StopOnEvent(coord, wait_for_stop, set_when_stopped):
  # [FUNCTION DOC]
  # HELPER: thread body for the coordinator tests. Waits for the first event, requests a stop
  # on the coordinator, then sets the second event to signal completion. Returns nothing.
  wait_for_stop.wait()
  coord.request_stop()
  set_when_stopped.set()


def RaiseOnEvent(coord, wait_for_stop, set_when_stopped, ex, report_exception):
  # [FUNCTION DOC]
  # HELPER: thread body for the coordinator tests. Waits for an event, raises the given
  # exception, catches runtime errors and reports them to the coordinator: as the bare
  # exception if the report flag is true, else as exception info. Finally sets the completion
  # event if one was supplied. Other exception types propagate out of the thread.
  try:
    wait_for_stop.wait()
    raise ex
  except RuntimeError as e:
    if report_exception:
      coord.request_stop(e)
    else:
      coord.request_stop(sys.exc_info())
  finally:
    if set_when_stopped:
      set_when_stopped.set()


def RaiseOnEventUsingContextHandler(coord, wait_for_stop, set_when_stopped, ex):
  # [FUNCTION DOC]
  # HELPER: thread body for the coordinator tests. Inside the coordinator's stop-on-exception
  # context, waits for an event and raises the given exception, which the context reports and
  # swallows. Afterward sets the completion event if one was supplied.
  with coord.stop_on_exception():
    wait_for_stop.wait()
    raise ex
  if set_when_stopped:
    set_when_stopped.set()


def SleepABit(n_secs, coord=None):
  # [FUNCTION DOC]
  # HELPER: thread body for the coordinator tests. Optionally registers the current thread
  # with the given coordinator, then sleeps for the given number of seconds.
  if coord:
    coord.register_thread(threading.current_thread())
  time.sleep(n_secs)


def WaitForThreadsToRegister(coord, num_threads):
  # [FUNCTION DOC]
  # HELPER: test synchronization. Busy-polls (sleeping a millisecond between checks), reading
  # the coordinator's private registered set under its lock, until exactly the given number of
  # threads are registered. Never times out.
  while True:
    with coord._lock:
      if len(coord._registered_threads) == num_threads:
        break
    time.sleep(0.001)


class CoordinatorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Coordinator class. Each test builds its own coordinator and worker
  # threads using the module-level thread-body helpers; no shared fixtures.

  def testStopAPI(self):
    # [FUNCTION DOC]
    # Scenario: single-threaded stop API. Before request_stop, should_stop is false and a short
    # wait times out false; after request_stop both report true.
    coord = coordinator.Coordinator()
    self.assertFalse(coord.should_stop())
    self.assertFalse(coord.wait_for_stop(0.01))
    coord.request_stop()
    self.assertTrue(coord.should_stop())
    self.assertTrue(coord.wait_for_stop(0.01))

  def testStopAsync(self):
    # [FUNCTION DOC]
    # Scenario: a worker thread requests a stop when signalled. Expects no stop before the
    # signal, and should_stop and wait_for_stop true after the worker reports it stopped.
    coord = coordinator.Coordinator()
    self.assertFalse(coord.should_stop())
    self.assertFalse(coord.wait_for_stop(0.1))
    wait_for_stop_ev = threading.Event()
    has_stopped_ev = threading.Event()
    t = threading.Thread(
        target=StopOnEvent, args=(coord, wait_for_stop_ev, has_stopped_ev))
    t.start()
    self.assertFalse(coord.should_stop())
    self.assertFalse(coord.wait_for_stop(0.01))
    wait_for_stop_ev.set()
    has_stopped_ev.wait()
    self.assertTrue(coord.wait_for_stop(0.05))
    self.assertTrue(coord.should_stop())

  def testJoin(self):
    # [FUNCTION DOC]
    # Scenario: three short-sleeping threads passed explicitly to join. Expects join to return
    # with all threads finished.
    coord = coordinator.Coordinator()
    threads = [
        threading.Thread(target=SleepABit, args=(0.01,)),
        threading.Thread(target=SleepABit, args=(0.02,)),
        threading.Thread(target=SleepABit, args=(0.01,))
    ]
    for t in threads:
      t.start()
    coord.join(threads)
    for t in threads:
      self.assertFalse(t.is_alive())

  def testJoinAllRegistered(self):
    # [FUNCTION DOC]
    # Scenario: three threads register themselves with the coordinator. After all registered,
    # join is called with no list and is expected to wait for all of them.
    coord = coordinator.Coordinator()
    threads = [
        threading.Thread(target=SleepABit, args=(0.01, coord)),
        threading.Thread(target=SleepABit, args=(0.02, coord)),
        threading.Thread(target=SleepABit, args=(0.01, coord))
    ]
    for t in threads:
      t.start()
    WaitForThreadsToRegister(coord, 3)
    coord.join()
    for t in threads:
      self.assertFalse(t.is_alive())

  def testJoinSomeRegistered(self):
    # [FUNCTION DOC]
    # Scenario: two of three threads register; the unregistered one is passed to join
    # explicitly. Expects join to wait for all three.
    coord = coordinator.Coordinator()
    threads = [
        threading.Thread(target=SleepABit, args=(0.01, coord)),
        threading.Thread(target=SleepABit, args=(0.02,)),
        threading.Thread(target=SleepABit, args=(0.01, coord))
    ]
    for t in threads:
      t.start()
    WaitForThreadsToRegister(coord, 2)
    # threads[1] is not registered we must pass it in.
    coord.join([threads[1]])
    for t in threads:
      self.assertFalse(t.is_alive())

  def testJoinGraceExpires(self):
    # [FUNCTION DOC]
    # Scenario: one thread requests a stop while another sleeps ten seconds. Runs the nested
    # check with three grace periods (tiny, short, one second); each must raise a runtime error
    # mentioning threads still running.

    def TestWithGracePeriod(stop_grace_period):
      # [FUNCTION DOC]
      # Nested check: creates a coordinator, starts a stopper thread and a long-sleeping daemon
      # thread, triggers the stop, waits for it, then expects join with the given grace period
      # to raise a runtime error about threads still running.
      coord = coordinator.Coordinator()
      wait_for_stop_ev = threading.Event()
      has_stopped_ev = threading.Event()
      threads = [
          threading.Thread(
              target=StopOnEvent,
              args=(coord, wait_for_stop_ev, has_stopped_ev)),
          threading.Thread(target=SleepABit, args=(10.0,))
      ]
      for t in threads:
        t.daemon = True
        t.start()
      wait_for_stop_ev.set()
      has_stopped_ev.wait()
      with self.assertRaisesRegex(RuntimeError, "threads still running"):
        coord.join(threads, stop_grace_period_secs=stop_grace_period)

    TestWithGracePeriod(1e-10)
    TestWithGracePeriod(0.002)
    TestWithGracePeriod(1.0)

  def testJoinWithoutGraceExpires(self):
    # [FUNCTION DOC]
    # Scenario: same threads as the grace-expiry test but join is told to ignore live threads.
    # Expects join to return normally without raising.
    coord = coordinator.Coordinator()
    wait_for_stop_ev = threading.Event()
    has_stopped_ev = threading.Event()
    threads = [
        threading.Thread(
            target=StopOnEvent, args=(coord, wait_for_stop_ev, has_stopped_ev)),
        threading.Thread(target=SleepABit, args=(10.0,))
    ]
    for t in threads:
      t.daemon = True
      t.start()
    wait_for_stop_ev.set()
    has_stopped_ev.wait()
    coord.join(threads, stop_grace_period_secs=1., ignore_live_threads=True)

  def testJoinRaiseReportExcInfo(self):
    # [FUNCTION DOC]
    # Scenario: two threads raise runtime errors in sequence and report them as exception info.
    # Expects join to raise only the first error ("First"), not the later one.
    coord = coordinator.Coordinator()
    ev_1 = threading.Event()
    ev_2 = threading.Event()
    threads = [
        threading.Thread(
            target=RaiseOnEvent,
            args=(coord, ev_1, ev_2, RuntimeError("First"), False)),
        threading.Thread(
            target=RaiseOnEvent,
            args=(coord, ev_2, None, RuntimeError("Too late"), False))
    ]
    for t in threads:
      t.start()

    ev_1.set()

    with self.assertRaisesRegex(RuntimeError, "First"):
      coord.join(threads)

  def testJoinRaiseReportException(self):
    # [FUNCTION DOC]
    # Scenario: like the exception-info test but threads report the bare exception. Expects
    # join to raise the first error ("First").
    coord = coordinator.Coordinator()
    ev_1 = threading.Event()
    ev_2 = threading.Event()
    threads = [
        threading.Thread(
            target=RaiseOnEvent,
            args=(coord, ev_1, ev_2, RuntimeError("First"), True)),
        threading.Thread(
            target=RaiseOnEvent,
            args=(coord, ev_2, None, RuntimeError("Too late"), True))
    ]
    for t in threads:
      t.start()

    ev_1.set()
    with self.assertRaisesRegex(RuntimeError, "First"):
      coord.join(threads)

  def testJoinIgnoresOutOfRange(self):
    # [FUNCTION DOC]
    # Scenario: a thread reports an out-of-range error (default clean-stop type). Expects join
    # to return without raising.
    # KNOWN ISSUE: the worker helper only catches runtime errors, so this out-of-range error
    # would escape the helper's handler rather than being reported; the test passes only
    # because the thread dies with an unhandled error and nothing is recorded.
    coord = coordinator.Coordinator()
    ev_1 = threading.Event()
    threads = [
        threading.Thread(
            target=RaiseOnEvent,
            args=(coord, ev_1, None,
                  errors_impl.OutOfRangeError(None, None, "First"), True))
    ]
    for t in threads:
      t.start()

    ev_1.set()
    coord.join(threads)

  def testJoinIgnoresMyExceptionType(self):
    # [FUNCTION DOC]
    # Scenario: a coordinator configured with value error as a clean-stop type, and a thread
    # raising a value error. Expects join to return without raising.
    # KNOWN ISSUE: the worker helper only catches runtime errors, so the value error is never
    # reported to the coordinator; the thread simply dies and the test passes trivially.
    coord = coordinator.Coordinator(clean_stop_exception_types=(ValueError,))
    ev_1 = threading.Event()
    threads = [
        threading.Thread(
            target=RaiseOnEvent,
            args=(coord, ev_1, None, ValueError("Clean stop"), True))
    ]
    for t in threads:
      t.start()

    ev_1.set()
    coord.join(threads)

  def testJoinRaiseReportExceptionUsingHandler(self):
    # [FUNCTION DOC]
    # Scenario: two threads raise inside the stop-on-exception context in sequence. Expects
    # join to raise the first error ("First").
    coord = coordinator.Coordinator()
    ev_1 = threading.Event()
    ev_2 = threading.Event()
    threads = [
        threading.Thread(
            target=RaiseOnEventUsingContextHandler,
            args=(coord, ev_1, ev_2, RuntimeError("First"))),
        threading.Thread(
            target=RaiseOnEventUsingContextHandler,
            args=(coord, ev_2, None, RuntimeError("Too late")))
    ]
    for t in threads:
      t.start()

    ev_1.set()
    with self.assertRaisesRegex(RuntimeError, "First"):
      coord.join(threads)

  def testClearStopClearsExceptionToo(self):
    # [FUNCTION DOC]
    # Scenario: a first thread reports an error and join raises it; clear_stop is called and a
    # second thread reports another error. Expects join to raise the second error ("Second"),
    # showing the first recorded exception was cleared.
    coord = coordinator.Coordinator()
    ev_1 = threading.Event()
    threads = [
        threading.Thread(
            target=RaiseOnEvent,
            args=(coord, ev_1, None, RuntimeError("First"), True)),
    ]
    for t in threads:
      t.start()

    with self.assertRaisesRegex(RuntimeError, "First"):
      ev_1.set()
      coord.join(threads)
    coord.clear_stop()
    threads = [
        threading.Thread(
            target=RaiseOnEvent,
            args=(coord, ev_1, None, RuntimeError("Second"), True)),
    ]
    for t in threads:
      t.start()
    with self.assertRaisesRegex(RuntimeError, "Second"):
      ev_1.set()
      coord.join(threads)

  def testRequestStopRaisesIfJoined(self):
    # [FUNCTION DOC]
    # Scenario: after join, reporting a bare exception from an except block re-raises it
    # immediately ("Too late"). After clear_stop, reporting another error is recorded normally
    # and the next join raises it ("After clear").
    coord = coordinator.Coordinator()
    # Join the coordinator right away.
    coord.join([])
    reported = False
    with self.assertRaisesRegex(RuntimeError, "Too late"):
      try:
        raise RuntimeError("Too late")
      except RuntimeError as e:
        reported = True
        coord.request_stop(e)
    self.assertTrue(reported)
    # If we clear_stop the exceptions are handled normally.
    coord.clear_stop()
    try:
      raise RuntimeError("After clear")
    except RuntimeError as e:
      coord.request_stop(e)
    with self.assertRaisesRegex(RuntimeError, "After clear"):
      coord.join([])

  def testRequestStopRaisesIfJoined_ExcInfo(self):
    # [FUNCTION DOC]
    # Scenario: same as the previous joined-coordinator test but errors are reported as
    # exception-info tuples. Expects the immediate re-raise after join and normal recording
    # after clear_stop.
    # Same as testRequestStopRaisesIfJoined but using syc.exc_info().
    coord = coordinator.Coordinator()
    # Join the coordinator right away.
    coord.join([])
    reported = False
    with self.assertRaisesRegex(RuntimeError, "Too late"):
      try:
        raise RuntimeError("Too late")
      except RuntimeError:
        reported = True
        coord.request_stop(sys.exc_info())
    self.assertTrue(reported)
    # If we clear_stop the exceptions are handled normally.
    coord.clear_stop()
    try:
      raise RuntimeError("After clear")
    except RuntimeError:
      coord.request_stop(sys.exc_info())
    with self.assertRaisesRegex(RuntimeError, "After clear"):
      coord.join([])


def _StopAt0(coord, n):
  # [FUNCTION DOC]
  # HELPER: looper target for the looper tests. If the first element of the one-element list is
  # zero, requests a stop on the coordinator; otherwise decrements that element. Mutates the list.
  if n[0] == 0:
    coord.request_stop()
  else:
    n[0] -= 1


class LooperTest(test.TestCase):
  # [CLASS DOC]
  # Test case for LooperThread argument passing (positional, keyword, mixed) using a shared
  # countdown list and the stop helper as the target.

  def testTargetArgs(self):
    # [FUNCTION DOC]
    # Scenario: starts a looper (interval zero) whose target gets the coordinator and a counter
    # list as positional arguments. After join, expects the counter to have reached zero.
    n = [3]
    coord = coordinator.Coordinator()
    thread = coordinator.LooperThread.loop(
        coord, 0, target=_StopAt0, args=(coord, n))
    coord.join([thread])
    self.assertEqual(0, n[0])

  def testTargetKwargs(self):
    # [FUNCTION DOC]
    # Scenario: same as the positional test but with keyword arguments only. Expects the
    # counter to reach zero after join.
    n = [3]
    coord = coordinator.Coordinator()
    thread = coordinator.LooperThread.loop(
        coord, 0, target=_StopAt0, kwargs={
            "coord": coord,
            "n": n
        })
    coord.join([thread])
    self.assertEqual(0, n[0])

  def testTargetMixedArgs(self):
    # [FUNCTION DOC]
    # Scenario: positional coordinator plus keyword counter. Expects the counter to reach zero
    # after join.
    n = [3]
    coord = coordinator.Coordinator()
    thread = coordinator.LooperThread.loop(
        coord, 0, target=_StopAt0, args=(coord,), kwargs={
            "n": n
        })
    coord.join([thread])
    self.assertEqual(0, n[0])


if __name__ == "__main__":
  test.main()
