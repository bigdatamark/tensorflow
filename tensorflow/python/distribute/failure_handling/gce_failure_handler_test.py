# [MODULE DOC]
# Integration tests for the preemption checkpoint handler in its Google Cloud configuration.
# The tests mock platform detection to report a Google Cloud GPU and replace the maintenance
# event watcher with a fake that randomly (or on every poll) reports a termination notice. A
# small training loop with a counter variable is run for several epochs under mirrored, one
# device, local multi-worker mirrored and (multi-process) multi-worker mirrored strategies; the
# worker exits with the cloud restart code when a notice is handled, is restarted, and must
# resume from the saved checkpoint and finish. Also contains helpers to build checkpoint
# managers and wait for processes to exit, and registers a pickling helper for the test class
# when the optional serialization library is installed. Run as a normal test main.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for GCE specifics of PreemptionCheckpointHandler."""
import os
import random
import re
import sys
import threading
import time
import urllib

from absl.testing import parameterized

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import one_device_strategy as one_device_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.failure_handling import failure_handling
from tensorflow.python.distribute.failure_handling import failure_handling_util
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.module import module
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib


try:
  import dill  # pylint:disable=g-import-not-at-top

  _REGISTER_DECORATOR = dill.register
except ImportError:
  _REGISTER_DECORATOR = lambda fn, *_: fn

mock = test.mock


CLUSTER_SIZE = 4
EPOCHS_TO_RUN = 5
STEPS_PER_EPOCH = 6
_PEER_WATCHER_THREAD_PREFIX = 'PeerTerminationWatcher'
_LOCAL_WATCHER_THREAD_PREFIX = 'WorkerTerminationSignalWatcher'
MAX_WAIT_TIME = 30


def _is_oss():
  # [FUNCTION DOC]
  # HELPER: tells tests which remote-call transport to use. Returns true when the program was
  # started through the open-source build tool (its name appears in the first command-line
  # argument), false otherwise.
  """Returns whether the test is run under OSS."""
  return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]


def _make_checkpoint_manager(checkpoint, checkpoint_dir, cluster_resolver):
  # [FUNCTION DOC]
  # HELPER: builds a checkpoint manager the way the handler would. Inputs: a checkpoint object,
  # the checkpoint directory, and a cluster resolver (may be nothing). For no cluster, an empty
  # cluster, or the chief, the manager uses the directory itself; for non-chief workers it uses
  # the handler's per-worker temporary directory. Keeps at most one checkpoint. Returns the
  # manager.

  if not cluster_resolver or not cluster_resolver.cluster_spec().as_dict() or (
      multi_worker_util.is_chief(
          cluster_spec=cluster_resolver.cluster_spec(),
          task_type=cluster_resolver.task_type,
          task_id=cluster_resolver.task_id)):
    return checkpoint_management.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=1)
  else:
    return checkpoint_management.CheckpointManager(
        checkpoint,
        directory=failure_handling._non_chief_checkpoint_dir(
            checkpoint_dir, cluster_resolver.task_id),
        max_to_keep=1)


def raise_if_not_all_exit(grace_period, mpr):
  # [FUNCTION DOC]
  # HELPER: waits for the cluster's worker processes to exit, for tests.
  # KNOWN ISSUE: the final timeout check compares the waiting time with the larger of grace
  # period plus five and forty, which does not match the loop's limit (grace period plus fifteen
  # or thirty), so the error is raised only in some timeout cases and may never be raised.
  # Polls once per second until none of the four worker processes exist, up to the larger of the
  # grace period plus fifteen seconds and thirty seconds. Inputs: the grace period in seconds and
  # the multi-process runner. Returns nothing; raises a runtime error when the equality check
  # described above holds.
  """Wait for all cluster to exit with a time out."""
  waiting_time = 0
  exit_process_count = 0
  # This addition to mitigate the fact that our step time is too short in test
  while exit_process_count != CLUSTER_SIZE and waiting_time < max(
      grace_period + 15, MAX_WAIT_TIME):
    exit_process_count = 0
    for worker_id in range(CLUSTER_SIZE):
      if not mpr.process_exists('worker', worker_id):
        exit_process_count += 1
    waiting_time += 1
    time.sleep(1)

  if waiting_time == max(grace_period + 5, 40):
    raise RuntimeError('Waited long but at least one worker still exist. '
                       'Considering size of our model, this should not'
                       ' happen.')


class GceFailureHandlingTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized integration tests of the preemption checkpoint handler on a mocked Google
  # Cloud GPU. Shared fixture: the worker function that creates the strategy, a trivial model
  # with a counter variable, the handler, and a training loop; tests start it directly or in a
  # multi-process cluster, restart it after a simulated termination and check completion.
  """Integration test for PreemptionCheckpointHandler."""

  def worker_fn(
      self,
      checkpoint_dir,
      cluster_spec,
      input_arg,
      strategy_option,
      maintenance_event=None,
      training_finished=None,
      frequent_send=False,
      training_restarted=None,
      termination_config=failure_handling.TerminationConfig(grace_period=0),
      api_wrapping_train=True,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the restarted-event and finished-event parameters default to nothing but are
    # used unconditionally, so callers must always supply them; the default termination config is
    # one shared object created at definition time.
    # KNOWN ISSUE: the handler for a name-resolution failure reads a message attribute that
    # URL errors do not have, so that branch would itself raise an attribute error.
    # Body executed by each worker (in-process or in a spawned process). Inputs: checkpoint
    # directory, cluster spec, input kind (plain checkpoint or checkpoint manager), strategy
    # name (mirrored, one device, or multi-worker), the maintenance event, finished event,
    # a flag for sending a notice on every poll, the restarted event, a termination config
    # (default: zero grace period) and a flag choosing between wrapping each step in the run
    # method or calling the save-if-preempted method. Mocks the cloud watcher and platform
    # detection, builds the model and handler, and if restarted and unfinished asserts that the
    # expected checkpoints exist (largest number 1, or 2 with a grace period, or equal to total run
    # calls for the other API). Trains for the remaining epochs, marks training finished, asserts
    # the counter equals replicas times epochs times steps, then asserts the watcher threads
    # stop after the destructor is invoked. Returns nothing; may exit with the restart code.
    if strategy_option == 'MS':

      strategy = mirrored_strategy.MirroredStrategy()
    elif strategy_option == 'OneDevice':
      if config.list_physical_devices('GPU'):
        strategy = one_device_lib.OneDeviceStrategy(device='/gpu:0')
      else:
        strategy = one_device_lib.OneDeviceStrategy(device='/cpu:0')
    else:
      strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()

    def mock_termination_watcher_function_gce(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: fake termination watcher replacing the cloud one. Ignores arguments. In normal
      # mode sleeps a second and, if no notice has been issued yet, with roughly one in eleven
      # chance sets the maintenance event and reports a notice. In frequent mode reports a notice
      # on every call until the event is set. Otherwise reports no notice.
      del args, kwargs
      if not frequent_send:
        time.sleep(1)
        if (not maintenance_event.is_set()) and (random.randrange(0, 11) == 5):
          maintenance_event.set()
          logging.info('Termination notice available.')
          return True

      elif frequent_send and not maintenance_event.is_set():
        logging.info('Termination notice available.')
        return True

      return False

    with mock.patch.object(
        failure_handling_util, 'termination_watcher_function_gce',
        mock_termination_watcher_function_gce), mock.patch.object(
            failure_handling_util, 'detect_platform',
            lambda: failure_handling_util.PlatformDevice.GCE_GPU):

      class Model(module.Module):
        # [CLASS DOC]
        # HELPER: minimal module holding a summed, write-synchronized counter variable and a
        # no-argument traced call returning its value, used as the checkpointed model.

        def __init__(self):
          # [FUNCTION DOC]
          # Creates the counter variable starting at zero, synchronized on write with sum
          # aggregation. Returns nothing.
          self.v = variables_lib.Variable(
              0.,
              synchronization=variables_lib.VariableSynchronization.ON_WRITE,
              aggregation=variables_lib.VariableAggregation.SUM)

        @def_function.function(input_signature=[])
        def __call__(self):
          # [FUNCTION DOC]
          # Traced function with no inputs; returns the current value of the counter variable.
          return self.v.read_value()

      with strategy.scope():
        model = Model()
        fh_ckpt = tracking_util.Checkpoint(model=model)

        if input_arg == 'checkpoint':
          checkpoint_or_manager = fh_ckpt
        else:
          checkpoint_or_manager = _make_checkpoint_manager(
              fh_ckpt, checkpoint_dir, strategy.cluster_resolver)
        preemption_handler = (
            failure_handling.PreemptionCheckpointHandler(
                strategy.cluster_resolver, checkpoint_or_manager,
                checkpoint_dir, termination_config))

      def distributed_train_step(current_epoch, current_step):
        # [FUNCTION DOC]
        # HELPER: one training step. Runs a traced step on every replica that adds one to the
        # counter, and logs when the last step of the epoch finishes. Inputs: current epoch and
        # current step numbers (integers). Returns nothing.

        @def_function.function
        def train_step():
          # [FUNCTION DOC]
          # Per-replica traced step: adds one to the counter variable. Returns nothing.
          model.v.assign_add(constant_op.constant(1.))

        strategy.run(train_step)

        if current_step == STEPS_PER_EPOCH - 1:
          logging.info('epoch %d finished', current_epoch)

      logging.info('Start training at %d',
                   preemption_handler.total_run_calls)

      # If the training process has been restarted, verify that the expected
      # number of checkpoints have been written.
      # We also want to check training_finished, because there's a corner case
      # where the signal is sent quite late and training finishes before the
      # grace period ends.
      if training_restarted.is_set() and not training_finished.is_set():
        logging.info(gfile.ListDirectory(checkpoint_dir))
        match_group = [
            re.search(r'.*ckpt-(\d+).index', a_file)
            for a_file in gfile.ListDirectory(checkpoint_dir)
        ]
        checkpoint_index = [
            a_match.group(1) for a_match in match_group if a_match
        ]
        self.assertNotEmpty(checkpoint_index)

        if api_wrapping_train:
          if termination_config.grace_period > 0:
            # Two checkpoints were saved for the extended grace period.
            self.assertEqual(
                max([int(ckpt_index) for ckpt_index in checkpoint_index]), 2)
          else:
            self.assertEqual(
                max([int(ckpt_index) for ckpt_index in checkpoint_index]), 1)

        else:
          # Test if arguments to save_checkpoint_if_preempted are passed
          # successfully.
          self.assertEqual(
              max([int(ckpt_index) for ckpt_index in checkpoint_index]),
              preemption_handler.total_run_calls)

      for epoch in range(preemption_handler.total_run_calls // STEPS_PER_EPOCH,
                         EPOCHS_TO_RUN):

        for step in range(preemption_handler.total_run_calls % STEPS_PER_EPOCH,
                          STEPS_PER_EPOCH):

          # Testing two different APIs to save checkpoint.
          if api_wrapping_train:
            preemption_handler.run(distributed_train_step, epoch, step)

          else:
            preemption_handler.save_checkpoint_if_preempted(
                checkpoint_number=preemption_handler.total_run_calls)
            distributed_train_step(epoch, step)

      logging.info('Training finished.')
      training_finished.set()

      self.assertEqual(
          model.v.numpy(),
          strategy.num_replicas_in_sync * EPOCHS_TO_RUN * STEPS_PER_EPOCH)

      running_threads = test_util.get_running_threads()
      if test_util.has_thread(_PEER_WATCHER_THREAD_PREFIX,
                              running_threads) and test_util.has_thread(
                                  _LOCAL_WATCHER_THREAD_PREFIX,
                                  running_threads):
        try:
          # Explicitly call __del__ since making it None and gc.collect does
          # not invoke __del__ here.
          preemption_handler.__del__()

          time.sleep(2)

          running_threads = test_util.get_running_threads()
          self.assertFalse(
              test_util.has_thread(_LOCAL_WATCHER_THREAD_PREFIX,
                                   running_threads))
          self.assertFalse(
              test_util.has_thread(_PEER_WATCHER_THREAD_PREFIX,
                                   running_threads))

        except urllib.error.URLError as e:
          if 'Temporary failure in name resolution' in e.message:
            # This is caused by a weird flakiness that mock.patch does not
            # correctly patch failure_handling_util.request_compute_metadata, a
            # real request is attempted, and an error is hit in
            # failure_handling_util.request_compute_metadata
            logging.warning('Hit a mock issue.')
            return

  @combinations.generate(
      combinations.combine(
          input_arg=['checkpoint', 'manager'],
          strategy_option=[
              'MS',
              'OneDevice',
              'MWMS_local',
              'MWMS_multi_worker',
          ],
      )
  )
  def test_basic_run(self, input_arg, strategy_option):
    # [FUNCTION DOC]
    # Scenario: basic preemption and restart for every combination of input kind (checkpoint or
    # manager) and strategy. For the multi-process case starts four workers, waits for all to
    # exit, and if training did not finish restarts them and joins; otherwise runs the worker
    # body in-process, catches the system exit and asserts its code is the cloud restart code
    # 143, and if a notice was issued without finishing asserts an exit happened and reruns.
    # Asserts that training finished in the end.
    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt/')

    if strategy_option == 'MWMS_multi_worker':
      has_chief = False
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=CLUSTER_SIZE)
      maintenance_event = multi_process_runner.manager().Event()
      training_finished = multi_process_runner.manager().Event()
      training_restarted = multi_process_runner.manager().Event()

      mpr = multi_process_runner.MultiProcessRunner(
          self.worker_fn,
          cluster_spec,
          args=(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                maintenance_event, training_finished, False,
                training_restarted),
          rpc_layer=rpc_layer,
          return_output=True,
          dependence_on_chief=has_chief)

      logging.info('Cluster starting.')
      mpr.start()

      raise_if_not_all_exit(0, mpr)

      if not training_finished.is_set():
        logging.info('restarting workers')
        training_restarted.set()
        for worker_id in range(CLUSTER_SIZE):
          mpr.start_single_process('worker', worker_id, cluster_spec)
        logging.info('workers restarted')

      mpr.join(timeout=250)
      self.assertTrue(training_finished.is_set())

    else:
      maintenance_event = threading.Event()
      training_finished = threading.Event()
      training_restarted = threading.Event()

      cluster_spec = server_lib.ClusterSpec({})
      caught_exit = False
      try:
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, False,
                       training_restarted)
      except SystemExit as exit_error:
        caught_exit = True
        # We cannot use assertRaise instead, since termination is not always
        # triggered.
        self.assertEqual(exit_error.code, 143)  # pylint:disable=g-assert-in-except

      if maintenance_event.is_set() and not training_finished.is_set():
        self.assertTrue(caught_exit)

        logging.info('restarting workers')
        training_restarted.set()
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, False,
                       training_restarted)

      self.assertTrue(training_finished.is_set())

  @combinations.generate(
      combinations.combine(
          grace_period=[0, 11],
          input_arg=['checkpoint', 'manager'],
          strategy_option=[
              'MS',
              'OneDevice',
              'MWMS_local',
              'MWMS_multi_worker',
          ],
          api_wrapping_train=[True, False]
      ))
  def test_multiple_workers_preempted_consecutively(self, grace_period,
                                                    input_arg, strategy_option,
                                                    api_wrapping_train):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the other tests, there is no final assertion that training finished,
    # and the chief flag is assigned twice.
    # Scenario: notices are sent on every poll (frequent mode) with grace period zero or eleven
    # seconds, both input kinds, all strategies and both save APIs. Multi-process: start,
    # wait for exit, force the event, restart all workers, join and assert finished. In-process:
    # run, catch the system exit asserting code 143, and if a notice was issued assert an exit
    # occurred and rerun the worker once.

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt/')
    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    termination_config = failure_handling.TerminationConfig(
        grace_period=grace_period)

    if strategy_option == 'MWMS_multi_worker':
      has_chief = False
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=CLUSTER_SIZE)
      has_chief = False
      maintenance_event = multi_process_runner.manager().Event()
      training_finished = multi_process_runner.manager().Event()
      training_restarted = multi_process_runner.manager().Event()
      mpr = multi_process_runner.MultiProcessRunner(
          self.worker_fn,
          cluster_spec,
          args=(
              checkpoint_dir,
              cluster_spec,
              input_arg,
              strategy_option,
              maintenance_event,
              training_finished,
              True,
              training_restarted,
              termination_config,
          ),
          kwargs={'api_wrapping_train': api_wrapping_train},
          rpc_layer=rpc_layer,
          return_output=True,
          dependence_on_chief=has_chief,
      )

      logging.info('Cluster starting.')
      mpr.start()

      raise_if_not_all_exit(grace_period, mpr)

      maintenance_event.set()
      logging.info('restarting workers')
      training_restarted.set()
      for worker_id in range(CLUSTER_SIZE):
        mpr.start_single_process('worker', worker_id, cluster_spec)
      logging.info('workers restarted')

      mpr.join(timeout=250)
      self.assertTrue(training_finished.is_set())

    else:
      maintenance_event = threading.Event()
      training_finished = threading.Event()
      training_restarted = threading.Event()

      cluster_spec = server_lib.ClusterSpec({})

      caught_exit = False
      try:
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, True,
                       training_restarted, termination_config)
      except SystemExit as exit_error:
        caught_exit = True
        # We cannot use assertRaise instead, since termination is not always
        # triggered.
        self.assertEqual(exit_error.code, 143)  # pylint:disable=g-assert-in-except

      if maintenance_event.is_set() and not training_finished.is_set():
        self.assertTrue(caught_exit)

        logging.info('restarting workers')
        training_restarted.set()
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, True,
                       training_restarted, termination_config)

  @combinations.generate(
      combinations.combine(
          input_arg=['checkpoint', 'manager'],
          strategy_option=[
              'MS',
              'OneDevice',
              'MWMS_local',
              'MWMS_multi_worker',
          ],
          api_wrapping_train=[True, False]))
  def test_grace_period_continue_training(self, input_arg, strategy_option,
                                          api_wrapping_train):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the multi-process branch is dead code because the strategy name it compares
    # with is not one of the parameterized values (which use a longer name), so every case,
    # including the multi-worker one, runs in-process.
    # Scenario: a grace period of eleven seconds so training continues after the notice and a
    # second checkpoint is taken. In-process: run the worker, catch the system exit asserting
    # code 143, and if a notice was issued assert an exit occurred, rerun the worker and assert
    # training finished. The unreachable branch would start a cluster and restart workers.
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt/')
    grace_period = 11

    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    termination_config = failure_handling.TerminationConfig(
        grace_period=grace_period)

    if strategy_option == 'multi_worker':
      has_chief = False
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=CLUSTER_SIZE)

      checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt/')
      maintenance_event = multi_process_runner.manager().Event()
      training_finished = multi_process_runner.manager().Event()
      training_restarted = multi_process_runner.manager().Event()

      mpr = multi_process_runner.MultiProcessRunner(
          self.worker_fn,
          cluster_spec,
          args=(
              checkpoint_dir,
              cluster_spec,
              input_arg,
              strategy_option,
              maintenance_event,
              training_finished,
              False,
              training_restarted,
              termination_config,
          ),
          kwargs={'api_wrapping_train': api_wrapping_train},
          rpc_layer=rpc_layer,
          return_output=True,
          dependence_on_chief=has_chief,
      )

      logging.info('Cluster starting.')
      mpr.start()

      while (not maintenance_event.is_set()) and (
          not training_finished.is_set()):
        time.sleep(1)

      raise_if_not_all_exit(grace_period, mpr)

      if not training_finished.is_set():
        logging.info('restarting workers')
        training_restarted.set()
        for worker_id in range(CLUSTER_SIZE):
          mpr.start_single_process('worker', worker_id, cluster_spec)
        logging.info('workers restarted')

      mpr.join(timeout=250)

      self.assertTrue(training_finished.is_set())

    else:
      maintenance_event = threading.Event()
      training_finished = threading.Event()
      training_restarted = threading.Event()

      cluster_spec = server_lib.ClusterSpec({})
      caught_exit = False
      try:
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, False,
                       training_restarted, termination_config)
      except SystemExit as exit_error:
        caught_exit = True
        # We cannot use assertRaise instead, since termination is not always
        # triggered.
        self.assertEqual(exit_error.code, 143)  # pylint:disable=g-assert-in-except

      if maintenance_event.is_set() and not training_finished.is_set():
        self.assertTrue(caught_exit)

        logging.info('restarting workers')
        training_restarted.set()
        self.worker_fn(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                       maintenance_event, training_finished, False,
                       training_restarted, termination_config)
        self.assertTrue(training_finished.is_set())


@_REGISTER_DECORATOR(GceFailureHandlingTest)
def _save_test_case(pickler, obj):
  # [FUNCTION DOC]
  # HELPER: custom serialization hook (registered only when the optional library exists; a
  # pass-through otherwise) so the test case can be sent to spawned processes. Reconstructs the
  # test case as a fresh instance instead of copying state. Inputs: the pickler and the object.
  # Returns the pickler's reduce result.
  def reconstruct(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: factory used on deserialization. Ignores all arguments and returns a new instance
    # of the test case class.
    del args, kwargs
    return GceFailureHandlingTest()

  return pickler.save_reduce(reconstruct, (), obj=obj)


if __name__ == '__main__':
  test_util.main()
