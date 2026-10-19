# [MODULE DOC]
# Tests for the multi-process runner and pool runner. Defines small module-level task functions
# (returning the task type, raising an error, doing nothing, returning data or arguments,
# returning the barrier or process id, and swapping a module global) that are shipped to child
# processes, plus three test classes: the runner itself (output capture, termination, restart,
# timeouts, exit codes, barrier, in-process start), the pool runner (process reuse, exceptions,
# cluster config, initializer, global and nested pools) and a two-GPU test of GPU assignment.
# Import-time side effects: optionally registers dill pickling hooks for the test classes and
# creates a module-level pool runner for a two-worker cluster. The main block uses the runner's
# own test entry point.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `multi_process_runner`."""

import ctypes
import json
import os
import sys
import threading
import time
import unittest

from absl import logging
from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import test


try:
  import dill  # pylint:disable=g-import-not-at-top

  _REGISTER_DECORATOR = dill.register
except ImportError:
  _REGISTER_DECORATOR = lambda fn, *_: fn


def fn_that_adds_task_type_in_return_data():
  # [FUNCTION DOC]
  # HELPER: task function for the tests. Returns the task type (such as worker, ps or chief) of
  # the process it runs in.
  return multi_worker_test_base.get_task_type()


def fn_that_errors():
  # [FUNCTION DOC]
  # HELPER: task function for the tests. Always raises a value error with a fixed message.
  raise ValueError('This is an error.')


def fn_that_does_nothing():
  # [FUNCTION DOC]
  # HELPER: task function for the tests. Does nothing and returns nothing.
  pass


def fn_that_adds_simple_return_data():
  # [FUNCTION DOC]
  # HELPER: task function for the tests. Returns a fixed dummy string.
  return 'dummy_data'


def fn_that_returns_args_and_kwargs(*args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: task function for the tests. Returns a list made of the positional arguments followed
  # by (name, value) pairs for the keyword arguments.
  return list(args) + list(kwargs.items())


def fn_with_barrier():
  # [FUNCTION DOC]
  # HELPER: task function for the tests. Returns the shared barrier object obtained from the
  # runner (only valid inside a child process).
  return multi_process_runner.get_barrier()


def fn_that_returns_pid():
  # [FUNCTION DOC]
  # HELPER: task function for the tests. Returns the operating system process id of the process
  # it runs in.
  return os.getpid()


V = None


def fn_that_sets_global(val):
  # [FUNCTION DOC]
  # HELPER: task function for the tests. Input: a new value. Replaces the module-level global
  # (initially nothing) with it and returns the previous value. Mutates module state in the
  # process it runs in.
  global V
  old_val = V
  V = val
  return old_val


@combinations.generate(combinations.combine(required_gpus=0))
class MultiProcessRunnerTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the MultiProcessRunner and the run function, generated with a zero-GPU requirement.
  # Covers return values, error propagation, output capture and streaming, termination, restarts,
  # timeouts, exit codes, barrier use and in-process starting. Shared fixture: a helper that reads
  # the worker index from the cluster-config environment variable.

  def _worker_idx(self):
    # [FUNCTION DOC]
    # HELPER: returns the index of the current task, read from the cluster-config environment
    # variable (only meaningful inside a child process).
    config_task = json.loads(os.environ['TF_CONFIG'])['task']
    return config_task['index']

  def test_multi_process_runner(self):
    # [FUNCTION DOC]
    # Scenario: run a task-type-returning function on a cluster of two workers, three parameter
    # servers and one chief. Expected: the returned task types contain exactly the expected counts
    # per type (two workers, three ps, one chief).
    mpr_result = multi_process_runner.run(
        fn_that_adds_task_type_in_return_data,
        multi_worker_test_base.create_cluster_spec(
            num_workers=2, num_ps=3, has_chief=True))

    job_count_dict = {'worker': 2, 'ps': 3, 'chief': 1}
    for data in mpr_result.return_value:
      job_count_dict[data] -= 1

    self.assertEqual(job_count_dict['worker'], 0)
    self.assertEqual(job_count_dict['ps'], 0)
    self.assertEqual(job_count_dict['chief'], 0)

  def test_multi_process_runner_error_propagates_from_subprocesses(self):
    # [FUNCTION DOC]
    # Scenario: start a runner whose function always raises, on one worker and one ps with a 20
    # second max run time. Expected: join re-raises the value error with the original message.
    runner = multi_process_runner.MultiProcessRunner(
        fn_that_errors,
        multi_worker_test_base.create_cluster_spec(num_workers=1, num_ps=1),
        max_run_time=20)
    runner.start()
    with self.assertRaisesRegex(ValueError, 'This is an error.'):
      runner.join()

  def test_multi_process_runner_queue_emptied_between_runs(self):
    # [FUNCTION DOC]
    # Scenario: run a data-returning function twice-worker, then run a no-op function on the same
    # cluster spec. Expected: the first result holds the dummy data for both workers; the second
    # result has no return values, showing results do not leak between runs.
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    return_value = multi_process_runner.run(fn_that_adds_simple_return_data,
                                            cluster_spec).return_value
    self.assertTrue(return_value)
    self.assertEqual(return_value[0], 'dummy_data')
    self.assertEqual(return_value[1], 'dummy_data')
    return_value = multi_process_runner.run(fn_that_does_nothing,
                                            cluster_spec).return_value
    self.assertFalse(return_value)

  def test_multi_process_runner_args_passed_correctly(self):
    # [FUNCTION DOC]
    # Scenario: run a function that echoes its arguments with two positional arguments and one
    # keyword argument on one worker. Expected: the single return value lists both positional
    # values followed by the keyword name/value pair.
    return_value = multi_process_runner.run(
        fn_that_returns_args_and_kwargs,
        multi_worker_test_base.create_cluster_spec(num_workers=1),
        args=('a', 'b'),
        kwargs={
            'c_k': 'c_v'
        }).return_value
    self.assertEqual(return_value[0][0], 'a')
    self.assertEqual(return_value[0][1], 'b')
    self.assertEqual(return_value[0][2], ('c_k', 'c_v'))

  def test_stdout_captured(self):
    # [FUNCTION DOC]
    # Scenario: two workers each print a line and return a string, with output collection on.
    # Expected: the result's output lists contain the line prefixed with each worker's label, and
    # the return values include the returned string.

    def simple_print_func():
      # [FUNCTION DOC]
      # HELPER: task function that prints a fixed line (flushed) and returns a fixed string.
      print('This is something printed.', flush=True)
      return 'This is returned data.'

    mpr_result = multi_process_runner.run(
        simple_print_func,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        return_output=True)
    std_stream_results = mpr_result.stdout
    return_value = mpr_result.return_value
    self.assertIn('[worker-0]:    This is something printed.\n',
                  std_stream_results)
    self.assertIn('[worker-1]:    This is something printed.\n',
                  std_stream_results)
    self.assertIn('This is returned data.', return_value)

  def test_termination(self):
    # [FUNCTION DOC]
    # Scenario: two workers print ten iterations five seconds apart; after five seconds worker 0 is
    # terminated; then join. Expected: worker 0 printed iteration zero but never iteration nine,
    # while worker 1 printed both.

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function printing the worker index and iteration number ten times, sleeping
      # five seconds after each.
      for i in range(0, 10):
        print(
            'index {}, iteration {}'.format(self._worker_idx(), i), flush=True)
        time.sleep(5)

    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        return_output=True)
    mpr.start()
    time.sleep(5)
    mpr.terminate('worker', 0)

    std_stream_results = mpr.join().stdout

    # Worker 0 is terminated in the middle, so it should not have iteration 9
    # printed.
    self.assertIn('[worker-0]:    index 0, iteration 0\n', std_stream_results)
    self.assertNotIn('[worker-0]:    index 0, iteration 9\n',
                     std_stream_results)
    self.assertIn('[worker-1]:    index 1, iteration 0\n', std_stream_results)
    self.assertIn('[worker-1]:    index 1, iteration 9\n', std_stream_results)

  def test_termination_and_start_single_process(self):
    # [FUNCTION DOC]
    # Scenario: two workers print ten iterations one second apart; after three seconds worker 0 is
    # terminated and immediately restarted with start_single_process; then join. Expected: iteration
    # zero of worker 0 appears twice, and worker 0 and worker 1 both reach iteration nine.

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function printing the worker index and iteration number ten times, sleeping
      # one second after each.
      for i in range(0, 10):
        print(
            'index {}, iteration {}'.format(self._worker_idx(), i), flush=True)
        time.sleep(1)

    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        return_output=True)
    mpr.start()
    time.sleep(3)
    mpr.terminate('worker', 0)
    mpr.start_single_process('worker', 0)
    std_stream_results = mpr.join().stdout

    # Worker 0 is terminated in the middle, but a new worker 0 is added, so it
    # should still have iteration 9 printed. Moreover, iteration 0 of worker 0
    # should happen twice.
    self.assertLen(
        [s for s in std_stream_results if 'index 0, iteration 0' in s], 2)
    self.assertIn('[worker-0]:    index 0, iteration 9\n', std_stream_results)
    self.assertIn('[worker-1]:    index 1, iteration 0\n', std_stream_results)
    self.assertIn('[worker-1]:    index 1, iteration 9\n', std_stream_results)

  def test_streaming(self):
    # [FUNCTION DOC]
    # Scenario: a cluster with a chief, two workers and two ps runs a function that logs and prints
    # lines; chief dependence is turned off, and an extra worker (index 2) and ps (index 2) are
    # started afterwards. Expected: output contains logged and printed lines for the chief and for
    # worker/ps indices zero to two for all five iterations, and none for index three.

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function that for five iterations logs and prints a line with the task type,
      # worker index and iteration, sleeping one second each time.
      for i in range(5):
        logging.info('(logging) %s-%d, i: %d',
                     multi_worker_test_base.get_task_type(), self._worker_idx(),
                     i)
        print(
            '(print) {}-{}, i: {}'.format(
                multi_worker_test_base.get_task_type(), self._worker_idx(), i),
            flush=True)
        time.sleep(1)

    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=2, num_ps=2),
        return_output=True)
    mpr._dependence_on_chief = False

    mpr.start()
    mpr.start_single_process('worker', 2)
    mpr.start_single_process('ps', 2)
    mpr_result = mpr.join()

    list_to_assert = mpr_result.stdout

    for job in ['chief']:
      for iteration in range(5):
        self.assertTrue(
            any('(logging) {}-0, i: {}'.format(job, iteration) in line
                for line in list_to_assert))
        self.assertTrue(
            any('(print) {}-0, i: {}'.format(job, iteration) in line
                for line in list_to_assert))

    for job in ['worker', 'ps']:
      for iteration in range(5):
        for task in range(3):
          self.assertTrue(
              any('(logging) {}-{}, i: {}'.format(job, task, iteration) in line
                  for line in list_to_assert))
          self.assertTrue(
              any('(print) {}-{}, i: {}'.format(job, task, iteration) in line
                  for line in list_to_assert))
        task = 3
        self.assertFalse(
            any('(logging) {}-{}, i: {}'.format(job, task, iteration) in line
                for line in list_to_assert))
        self.assertFalse(
            any('(print) {}-{}, i: {}'.format(job, task, iteration) in line
                for line in list_to_assert))

  def test_start_in_process_as(self):
    # [FUNCTION DOC]
    # Scenario: a chief plus one worker; the chief runs in the main process while a helper thread
    # starts an extra evaluator task after one second. Expected: output includes lines from the
    # worker and the evaluator for all five iterations.

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function that logs the task type, index and iteration five times with a one
      # second sleep between.
      for i in range(5):
        logging.info('%s-%d, i: %d', multi_worker_test_base.get_task_type(),
                     self._worker_idx(), i)
        time.sleep(1)

    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1),
        return_output=True)

    def eval_func():
      # [FUNCTION DOC]
      # HELPER: thread body that waits one second then starts an evaluator task on the runner.
      time.sleep(1)
      mpr.start_single_process(task_type='evaluator', task_id=0)

    eval_thread = threading.Thread(target=eval_func)
    eval_thread.start()
    mpr.start_in_process_as(as_task_type='chief', as_task_id=0)
    eval_thread.join()
    list_to_assert = mpr.join().stdout
    for job in ['worker', 'evaluator']:
      for iteration in range(5):
        self.assertTrue(
            any('{}-0, i: {}'.format(job, iteration) in line
                for line in list_to_assert))

  def test_terminate_all_does_not_ignore_error(self):
    # [FUNCTION DOC]
    # Scenario: two workers run an erroring function; after sixty seconds terminate_all is called,
    # then join. Expected: join still raises the original value error instead of hiding it.
    mpr = multi_process_runner.MultiProcessRunner(
        fn_that_errors,
        multi_worker_test_base.create_cluster_spec(num_workers=2),
        return_output=True)
    mpr.start()
    time.sleep(60)
    mpr.terminate_all()
    with self.assertRaisesRegex(ValueError, 'This is an error.'):
      mpr.join()

  def test_barrier(self):
    # [FUNCTION DOC]
    # Scenario: run a function that returns the shared barrier on a chief plus one worker.
    # Expected: the run completes without error.
    multi_process_runner.run(
        fn_with_barrier,
        cluster_spec=multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1),
    )

  def test_barrier_called_in_main_process(self):
    # [FUNCTION DOC]
    # Scenario: ask for the barrier from the main (non-child) process. Expected: a value error.
    with self.assertRaises(ValueError):
      multi_process_runner.get_barrier()

  def test_stdout_available_when_timeout(self):
    # [FUNCTION DOC]
    # Scenario: one worker logs a line then sleeps far beyond a 60 second join timeout. Expected:
    # join raises the subprocess timeout error, whose attached result's output contains the logged
    # line. The test then terminates all processes.

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function that logs a line then sleeps for a very long time to force a timeout.
      logging.info('something printed')
      time.sleep(10000)  # Intentionally make the test timeout.

    with self.assertRaises(multi_process_runner.SubprocessTimeoutError) as cm:
      mpr = multi_process_runner.MultiProcessRunner(
          fn,
          multi_worker_test_base.create_cluster_spec(num_workers=1),
          return_output=True)
      mpr.start()
      mpr.join(timeout=60)
    mpr.terminate_all()

    list_to_assert = cm.exception.mpr_result.stdout
    self.assertTrue(
        any('something printed' in line for line in list_to_assert))

  def test_seg_fault_raises_error(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the skip condition is true on every modern interpreter, so this test effectively
    # never runs.
    # Scenario: one worker deliberately crashes with a segmentation fault. Expected: the
    # unexpected-exit error mentioning worker-0's exit code, with captured output containing a
    # segmentation fault message. Skipped in the open-source environment and on Python 3.7 or newer.

    if multi_process_runner.is_oss() or sys.version_info >= (3, 7):
      self.skipTest('TODO(b/171004637): Failing in OSS and Python 3.7+')

    def fn_expected_to_seg_fault():
      # [FUNCTION DOC]
      # HELPER: task function that reads from a null address to crash the process.
      ctypes.string_at(0)  # Intentionally made seg fault.

    with self.assertRaises(
        multi_process_runner.UnexpectedSubprocessExitError) as cm:
      multi_process_runner.run(
          fn_expected_to_seg_fault,
          multi_worker_test_base.create_cluster_spec(num_workers=1),
          return_output=True)
    self.assertIn('Subprocess worker-0 exited with exit code',
                  str(cm.exception))
    list_to_assert = cm.exception.mpr_result.stdout
    self.assertTrue(
        any('Segmentation fault' in line for line in list_to_assert))

  def test_seg_fault_in_chief_raises_error(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the skip condition is true on every modern interpreter, so this test effectively
    # never runs.
    # Scenario: chief plus one worker; the worker sleeps while the chief crashes with a segmentation
    # fault. Expected: the unexpected-exit error naming chief-0, with output showing a segmentation
    # fault. Skipped in the open-source environment and on Python 3.7 or newer.

    if multi_process_runner.is_oss() or sys.version_info >= (3, 7):
      self.skipTest('TODO(b/171004637): Failing in OSS and Python 3.7+')

    def fn_expected_to_seg_fault():
      # [FUNCTION DOC]
      # HELPER: task function; workers sleep a very long time, other tasks (the chief) crash the
      # process by reading a null address.
      if multi_worker_test_base.get_task_type() == 'worker':
        time.sleep(10000)
      ctypes.string_at(0)  # Intentionally made seg fault.

    with self.assertRaises(
        multi_process_runner.UnexpectedSubprocessExitError) as cm:
      multi_process_runner.run(
          fn_expected_to_seg_fault,
          multi_worker_test_base.create_cluster_spec(
              has_chief=True, num_workers=1),
          return_output=True)
    self.assertIn('Subprocess chief-0 exited with exit code',
                  str(cm.exception))
    list_to_assert = cm.exception.mpr_result.stdout
    self.assertTrue(
        any('Segmentation fault' in line for line in list_to_assert))

  def test_exit_code_is_reported_by_chief_subprocess(self):
    # [FUNCTION DOC]
    # Scenario: chief plus one worker; the worker sleeps, the chief exits with code twenty.
    # Expected: join raises the unexpected-exit error with a message naming chief-0 and code 20.

    def fn_expected_to_exit_with_20():
      # [FUNCTION DOC]
      # HELPER: task function; workers sleep a very long time, other tasks exit with code twenty.
      if multi_worker_test_base.get_task_type() == 'worker':
        time.sleep(10000)
      sys.exit(20)

    mpr = multi_process_runner.MultiProcessRunner(
        fn_expected_to_exit_with_20,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1))
    mpr.start()

    with self.assertRaisesRegex(
        multi_process_runner.UnexpectedSubprocessExitError,
        'Subprocess chief-0 exited with exit code 20'):
      mpr.join()

  def test_exit_code_is_reported_by_subprocess(self):
    # [FUNCTION DOC]
    # Scenario: one worker exits with code ten. Expected: join raises the unexpected-exit error
    # with a message naming worker-0 and code 10.

    def fn_expected_to_exit_with_10():
      # [FUNCTION DOC]
      # HELPER: task function that exits the process with code ten.
      sys.exit(10)

    mpr = multi_process_runner.MultiProcessRunner(
        fn_expected_to_exit_with_10,
        multi_worker_test_base.create_cluster_spec(num_workers=1))
    mpr.start()

    with self.assertRaisesRegex(
        multi_process_runner.UnexpectedSubprocessExitError,
        'Subprocess worker-0 exited with exit code 10'):
      mpr.join()

  def test_auto_restart(self):
    # [FUNCTION DOC]
    # Scenario: with auto restart on, one worker increments a shared counter and fails on the first
    # run only. Expected: join succeeds and the counter ends at two, showing one restart.

    def fn(counter):
      # [FUNCTION DOC]
      # HELPER: task function that increments the shared counter and raises a value error if this
      # was the first increment.
      counter.value += 1
      if counter.value == 1:
        raise ValueError

    manager = multi_process_runner.manager()
    counter = manager.Value(int, 0)
    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(num_workers=1),
        args=(counter,),
        auto_restart=True)
    mpr.start()
    mpr.join()
    self.assertEqual(counter.value, 2)

  def test_auto_restart_and_timeout(self):
    # [FUNCTION DOC]
    # Scenario: with auto restart on, a worker logs, sleeps a second and always fails; join has a
    # ten second timeout. Expected: join raises a value error (the error wins over the timeout) and
    # the attached output shows the function ran more than once.

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function that logs a line, sleeps one second, then raises a value error.
      logging.info('Running')
      time.sleep(1)
      raise ValueError

    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(num_workers=1),
        auto_restart=True,
        return_output=True)
    mpr.start()
    with self.assertRaises(ValueError) as cm:
      mpr.join(timeout=10)
    self.assertGreater(
        sum(['Running' in msg for msg in cm.exception.mpr_result.stdout]), 1)

  def test_auto_restart_and_chief(self):
    # [FUNCTION DOC]
    # Scenario: auto restart on, chief plus one worker; after one second the chief returns normally
    # and the worker raises a value error. Expected: once the chief has exited with zero, the failed
    # worker is not restarted indefinitely and join (ten second timeout) raises the value error.
    # If the chief has exited with zero exit code, auto restart should stop
    # restarting other tasks even if they fail.

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function that sleeps one second, then raises a value error unless this task is
      # the chief.
      time.sleep(1)
      if multi_worker_test_base.get_task_type() != 'chief':
        raise ValueError

    manager = multi_process_runner.manager()
    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=True, num_workers=1),
        auto_restart=True)
    mpr.start()
    with self.assertRaises(ValueError):
      mpr.join(timeout=10)

  def test_auto_restart_failure_immediate_after_restart(self):
    # [FUNCTION DOC]
    # Scenario: auto restart on, two workers sleeping five seconds; worker 1 is terminated and the
    # test waits until it has a new process id, then terminates worker 0. Expected: join (twenty
    # second timeout) completes without error.
    # Test the case when worker-0 fails immediately after worker-1 restarts.

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function that sleeps five seconds.
      time.sleep(5)

    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=False, num_workers=2),
        auto_restart=True)
    mpr.start()
    pid = mpr.get_process_id('worker', 1)
    mpr.terminate('worker', 1)
    while mpr.get_process_id('worker', 1) == pid:
      time.sleep(0.1)
    mpr.terminate('worker', 0)
    mpr.join(timeout=20)

  def test_auto_restart_terminate(self):
    # [FUNCTION DOC]
    # Scenario: auto restart on, one worker increments a shared counter and sleeps on the first run;
    # after three seconds the worker is terminated by the user. Expected: it is restarted anyway,
    # join completes, and the counter ends at two.
    # Tasks terminated by the user should also be restarted.

    def fn(counter):
      # [FUNCTION DOC]
      # HELPER: task function that increments the shared counter and sleeps a long time if it was
      # the first increment.
      counter.value += 1
      if counter.value == 1:
        time.sleep(100)

    manager = multi_process_runner.manager()
    counter = manager.Value(int, 0)

    mpr = multi_process_runner.MultiProcessRunner(
        fn,
        multi_worker_test_base.create_cluster_spec(
            has_chief=False, num_workers=1),
        args=(counter,),
        auto_restart=True)
    mpr.start()
    time.sleep(3)
    mpr.terminate('worker', 0)
    mpr.join(timeout=20)
    self.assertEqual(counter.value, 2)

  def test_error_reporting_overrides_timeout_reporting(self):
    # [FUNCTION DOC]
    # Scenario: two workers; worker 1 sleeps forever while worker 0 raises an error; join has a
    # twenty second timeout. Expected: the worker's value error is raised rather than the timeout
    # error.

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function; worker 1 sleeps a very long time, others raise a value error.
      if self._worker_idx() == 1:
        time.sleep(10000)
      raise ValueError('Worker 0 errored')

    mpr = multi_process_runner.MultiProcessRunner(
        fn, multi_worker_test_base.create_cluster_spec(num_workers=2))
    mpr.start()

    with self.assertRaisesRegex(
        ValueError,
        'Worker 0 errored'):
      mpr.join(timeout=20)

  def test_process_exists(self):
    # [FUNCTION DOC]
    # Scenario: one long-sleeping worker is started and then terminated. Expected: process_exists
    # is true right after start and eventually becomes false after termination (otherwise the test
    # would time out).

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function that sleeps for a very long time.
      time.sleep(100000)

    mpr = multi_process_runner.MultiProcessRunner(
        fn, multi_worker_test_base.create_cluster_spec(num_workers=1))
    mpr.start()
    self.assertTrue(mpr.process_exists('worker', 0))
    mpr.terminate('worker', 0)
    # Worker 0 should exit at some point, or else the test would time out.
    while mpr.process_exists('worker', 0):
      time.sleep(1)

  def test_timeout_none(self):
    # [FUNCTION DOC]
    # Scenario: a worker sleeps 250 seconds (longer than the default timeout) then raises a value
    # error; join is called with no timeout. Expected: join waits and raises that value error.
    # Skipped in the open-source environment because it is slow.

    if multi_process_runner.is_oss():
      self.skipTest('Intentionally skipping longer test in OSS.')

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function that sleeps 250 seconds then raises a value error.
      time.sleep(250)
      raise ValueError('Worker 0 errored')

    mpr = multi_process_runner.MultiProcessRunner(
        fn, multi_worker_test_base.create_cluster_spec(num_workers=1))

    mpr.start()
    with self.assertRaisesRegex(ValueError, 'Worker 0 errored'):
      mpr.join(timeout=None)


_global_pool = multi_process_runner.MultiProcessPoolRunner(
    multi_worker_test_base.create_cluster_spec(num_workers=2))


class MultiProcessPoolRunnerTest(test.TestCase):
  # [CLASS DOC]
  # Tests for MultiProcessPoolRunner: process reuse across runs, error handling, cluster config,
  # initializers, a module-level shared pool and nested use of a pool from within a child.
  # No shared fixtures beyond the module-level pool.

  def test_same_process_across_runs(self):
    # [FUNCTION DOC]
    # Scenario: a pool on two workers runs the pid-returning function once and then three more
    # times. Expected: each later run returns the same process ids, proving worker reuse.
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec)
    pid = runner.run(fn_that_returns_pid)
    for _ in range(3):
      self.assertAllEqual(runner.run(fn_that_returns_pid), pid)

  def test_exceptions_in_sub_process(self):
    # [FUNCTION DOC]
    # Scenario: run the pid function, then an erroring function, then the pid function again on the
    # same pool. Expected: the error is raised to the caller and the workers survive (same process
    # ids afterwards).
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec)
    pid = runner.run(fn_that_returns_pid)
    with self.assertRaisesRegex(ValueError, 'This is an error.'):
      runner.run(fn_that_errors)
    self.assertAllEqual(runner.run(fn_that_returns_pid), pid)

  def test_tf_config(self):
    # [FUNCTION DOC]
    # Scenario: a pool with a chief and two workers runs a function returning the task type.
    # Expected: results contain two workers and one chief, showing the cluster config is set.
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=True, num_workers=2)
    runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec)
    result = runner.run(fn_that_adds_task_type_in_return_data)

    job_count_dict = {'worker': 2, 'chief': 1}
    for data in result:
      job_count_dict[data] -= 1

    self.assertEqual(job_count_dict['worker'], 0)
    self.assertEqual(job_count_dict['chief'], 0)

  @unittest.expectedFailure
  def test_exception_in_main_process(self):
    # [FUNCTION DOC]
    # Scenario: start a pool, run a function, then raise a value error in the main process without
    # shutting the pool down. Marked as an expected failure, so the raised error is the expected
    # outcome; the point is that the pool must cope with its finalizer not running.
    # When there's an exception in the main process, __del__() is not called.
    # This test is to verify MultiProcessPoolRunner can cope with __del__() not
    # being called.
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=True, num_workers=2)
    runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec)
    runner.run(fn_that_returns_pid)
    raise ValueError('failure')

  def test_initializer(self):
    # [FUNCTION DOC]
    # Scenario: a pool whose initializer sets the module global to one in each worker, then runs a
    # function setting it to two. Expected: each worker returns the old value one, so the result is
    # one for both workers.
    cluster_spec = multi_worker_test_base.create_cluster_spec(num_workers=2)
    runner = multi_process_runner.MultiProcessPoolRunner(
        cluster_spec, initializer=lambda: fn_that_sets_global(1))
    result = runner.run(fn_that_sets_global, args=(2,))
    self.assertAllEqual(result, [1, 1])

  def test_global_pool(self):
    # [FUNCTION DOC]
    # Scenario: run a no-op function on the module-level shared pool. Expected: completes without
    # error.
    _global_pool.run(fn_that_does_nothing)

  def test_nested_pool(self):
    # [FUNCTION DOC]
    # Scenario: run on the shared pool a function that itself runs a no-op on the shared pool (each
    # child uses its own pool instance). Expected: completes without error.

    def fn():
      # [FUNCTION DOC]
      # HELPER: task function that runs a no-op on the module-level pool from inside a child.
      # This runs in sub processes, so they are each using their own
      # MultiProcessPoolRunner.
      _global_pool.run(fn_that_does_nothing)

    _global_pool.run(fn)


@combinations.generate(combinations.combine(required_physical_gpus=2))
class MultiProcessRunnerMultiGPUTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests requiring two physical GPUs, covering runner GPU assignment when GPUs are not shared.
  # No shared fixtures.

  def test_not_share_gpu(self):
    # [FUNCTION DOC]
    # Scenario: with two or four physical GPUs (otherwise skipped), run a chief and a worker with
    # GPU sharing off, first returning the visible-devices variable, then the visible GPU count.
    # Expected: with two GPUs the tasks see 0 and 1 and one GPU each; with four they see 0,2 and 1,3
    # and two GPUs each.
    num_gpus = len(context.context().list_physical_devices('GPU'))
    if num_gpus != 2 and num_gpus != 4:
      self.skipTest('requires 2 or 4 GPUs')
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=True, num_workers=1)

    # Verify that CUDA_VISIBLE_DEVICES are different on each worker.

    def cuda_visible_devices_fn():
      # [FUNCTION DOC]
      # HELPER: task function returning the visible-devices environment variable's value.
      return os.getenv('CUDA_VISIBLE_DEVICES')

    runner = multi_process_runner.MultiProcessRunner(
        cuda_visible_devices_fn, cluster_spec, share_gpu=False)
    runner.start()
    result = runner.join()
    if num_gpus == 2:
      self.assertAllEqual(sorted(result.return_value), ['0', '1'])
    else:
      self.assertAllEqual(sorted(result.return_value), ['0,2', '1,3'])

    # Verify that CUDA_VISIBLE_DEVICES works.

    def num_gpus_fn():
      # [FUNCTION DOC]
      # HELPER: task function returning the number of physical GPUs visible to the process.
      return len(context.context().list_physical_devices('GPU'))

    runner = multi_process_runner.MultiProcessRunner(
        num_gpus_fn, cluster_spec, share_gpu=False)
    runner.start()
    result = runner.join()
    if num_gpus == 2:
      self.assertAllEqual(result.return_value, [1, 1])
    else:
      self.assertAllEqual(result.return_value, [2, 2])


@_REGISTER_DECORATOR(MultiProcessRunnerTest)
def _save_multi_process_runner_test(pickler, obj):
  # [FUNCTION DOC]
  # HELPER: custom dill pickling hook for the runner test class. Inputs: pickler and instance.
  # Makes the instance pickle as a fresh instance built by a reconstruct function. Returns the
  # pickler's save result.
  def reconstruct(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: factory used when unpickling; ignores its arguments and returns a new runner test
    # case instance.
    del args, kwargs
    return MultiProcessRunnerTest()

  return pickler.save_reduce(reconstruct, (), obj=obj)


@_REGISTER_DECORATOR(MultiProcessPoolRunnerTest)
def _save_multi_process_pool_runner_test(pickler, obj):
  # [FUNCTION DOC]
  # HELPER: custom dill pickling hook for the pool runner test class. Inputs: pickler and
  # instance. Makes the instance pickle as a fresh instance built by a reconstruct function.
  # Returns the pickler's save result.
  def reconstruct(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: factory used when unpickling; ignores its arguments and returns a new pool runner
    # test case instance.
    del args, kwargs
    return MultiProcessPoolRunnerTest()

  return pickler.save_reduce(reconstruct, (), obj=obj)


@_REGISTER_DECORATOR(MultiProcessRunnerMultiGPUTest)
def _save_multi_process_runner_multi_gpu_test(pickler, obj):
  # [FUNCTION DOC]
  # HELPER: custom dill pickling hook for the multi-GPU test class. Inputs: pickler and instance.
  # Makes the instance pickle as a fresh instance built by a reconstruct function. Returns the
  # pickler's save result.
  def reconstruct(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: factory used when unpickling; ignores its arguments and returns a new multi-GPU test
    # case instance.
    del args, kwargs
    return MultiProcessRunnerMultiGPUTest()

  return pickler.save_reduce(reconstruct, (), obj=obj)


if __name__ == '__main__':
  multi_process_runner.test_main()
