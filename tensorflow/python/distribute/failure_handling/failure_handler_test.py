# [MODULE DOC]
# Integration tests for the preemption checkpoint handler (failure_handling). A small training
# loop runs under several distribution strategies (mirrored, one-device, multi-worker local and
# multi-process multi-worker). A termination signal is delivered to a worker or to the current
# process, and the tests verify that checkpoints are written, that workers exit with the expected
# code, and that training resumes after a restart and reaches the expected final variable value.
# Also covers error propagation across workers, grace period behaviour and a user-supplied save
# function. Module level constants set the cluster size, epochs, steps per epoch and wait time.
# Import time: optionally uses the dill library to register a pickling helper for the test class.
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
"""Tests for PreemptionCheckpointHandler."""
import os
import random
import re
import signal
import sys
import threading
import time

from absl.testing import parameterized

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.checkpoint import checkpoint as tracking_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
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
from tensorflow.python.framework import errors_impl
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
EPOCHS_TO_RUN = 8
STEPS_PER_EPOCH = 15
MAX_WAIT_TIME = 40


def _is_oss():
  # [FUNCTION DOC]
  # HELPER: environment check for the tests, used to choose the RPC layer.
  # Takes nothing. Returns true when the first command line argument contains the word bazel
  # (taken to mean running in the open source build), otherwise false.
  """Returns whether the test is run under OSS."""
  return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]


def _make_checkpoint_manager(checkpoint, checkpoint_dir, cluster_resolver):
  # [FUNCTION DOC]
  # HELPER: builds the checkpoint manager passed to the handler when the test uses a manager.
  # Inputs: a checkpoint object, a checkpoint directory, and a cluster resolver (may be empty).
  # Returns a checkpoint manager keeping at most one checkpoint. The chief (or a setup with no
  # cluster) uses the given directory; every other worker uses a per-task directory derived from
  # the given one through a private function of the failure handling module.
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
  # HELPER: waits for the multi-process cluster to terminate after a signal.
  # KNOWN ISSUE: the wait loop uses a limit of the larger of grace period plus 15 and the
  # maximum wait time, but the final check compares the waited time against the larger of grace
  # period plus 5 and 40 using equality. With a grace period of 0 or small values the loop limit
  # and the error threshold are different numbers, and the equality check can fire when the
  # loop ended exactly on the threshold even though all workers exited, or never fire for
  # larger grace periods, so the timeout error is unreliable.
  # Inputs: grace_period (seconds, number) and mpr (the multi-process runner used to ask whether
  # a worker process still exists). Polls once per second, counting workers that no longer
  # exist. Returns nothing; raises a runtime error if the final check decides workers remain.
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


class PreemptionCheckpointTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the preemption checkpoint handler. The shared training routine
  # is the worker function, which builds a strategy, a tiny model with one variable and the
  # handler, then trains for a fixed number of epochs and steps. The tests drive it in threads
  # or in a multi-process cluster, send termination signals and check recovery.
  """Integration test for PreemptionCheckpointHandler."""

  def _maybe_trigger_a_preemption(self, training_started_event,
                                  trigger_it=False):
    # [FUNCTION DOC]
    # HELPER: randomly or deterministically signals that training may be preempted.
    # Inputs: training_started_event, a list of events (or nothing) shared with the signal
    # sender; trigger_it, a flag (default false) forcing the trigger. Does nothing if no events
    # are given. Otherwise, among events not yet set, sets the first one when forced, or with a
    # probability of about two in nine. Returns nothing; side effect is setting an event.
    if not training_started_event:
      return
    clear_events = [
        event for event in training_started_event if not event.is_set()
    ]
    if clear_events:
      if trigger_it:
        logging.info('Set preemption signal')
        clear_events[0].set()
      elif random.randrange(0, 9) > 6:
        clear_events[0].set()

  def worker_fn(self,
                checkpoint_dir,
                cluster_spec,
                strategy_option,
                input_arg='checkpoint',
                training_started_event=None,
                raise_app_error_on_worker=None,
                training_restarted=None,
                training_finished=None,
                termination_config=failure_handling.TerminationConfig(),
                api_wrapping_train=True):
    # [FUNCTION DOC]
    # HELPER: the training routine shared by all tests; run in-process or in worker processes.
    # KNOWN ISSUE: the test_preemption_checkpointing multi-worker branch passes input_arg and
    # strategy_option to this function in swapped order relative to this signature.
    # Inputs: checkpoint_dir (directory for handler checkpoints); cluster_spec (cluster
    # description, empty for single worker); strategy_option (text: MS for mirrored, OneDevice,
    # otherwise a collective all-reduce strategy); input_arg (text: checkpoint to pass a
    # checkpoint object, anything else to pass a checkpoint manager); training_started_event
    # (optional list of events used to trigger preemption); raise_app_error_on_worker (optional
    # task id on which a resource-exhausted error is raised in the train step);
    # training_restarted and training_finished (optional events marking a restart and completion);
    # termination_config (handler termination configuration, default one created once at
    # definition time); api_wrapping_train (true to train through the handler's run method,
    # false to call the save-if-preempted method manually before every step).
    # Builds the strategy, a model with one summed variable, a checkpoint and the handler (with
    # the GCP check patched to false). On a restart it asserts the number of checkpoints written
    # before (2 with a grace period, otherwise 1). Trains epochs times steps adding 1 per step,
    # possibly triggering preemption, then sets the finished event and asserts the variable equals
    # replicas times epochs times steps. Returns nothing.

    if strategy_option == 'MS':
      strategy = mirrored_strategy.MirroredStrategy()
    elif strategy_option == 'OneDevice':
      if config.list_physical_devices('GPU'):
        strategy = one_device_lib.OneDeviceStrategy(device='/gpu:0')
      else:
        strategy = one_device_lib.OneDeviceStrategy(device='/cpu:0')
    else:
      strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()

    class Model(module.Module):
      # [CLASS DOC]
      # Minimal trackable module holding a single float variable (initially zero) that is
      # summed across replicas and synchronized on write. Used as the checkpointed model.

      def __init__(self):
        # [FUNCTION DOC]
        # Creates the single variable, starting at zero, with on-write synchronization and sum
        # aggregation. Takes no arguments and returns nothing.
        self.v = variables_lib.Variable(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_WRITE,
            aggregation=variables_lib.VariableAggregation.SUM)

      @def_function.function(input_signature=[])
      def __call__(self):
        # [FUNCTION DOC]
        # Traced call with no inputs; returns the current value of the variable.
        return self.v.read_value()

    with mock.patch.object(failure_handling_util, 'on_gcp', lambda: False):

      with strategy.scope():
        model = Model()
        # Named it fh_ckpt because it'd be better that the user have their
        # regular checkpoint separate from the checkpoint for
        # PreemptionCheckpointHandler, since we will create CheckpointManager
        # to manage the checkpoint and only one CheckpointManager should be
        # active in a particular directory at a time.
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
        # One training step, also the function handed to the handler's run method. Inputs: the
        # current epoch and step numbers. Runs a replicated step adding 1 to the model variable
        # on every replica, and logs when the last step of an epoch finishes. Returns nothing.

        @def_function.function
        def train_step():
          # [FUNCTION DOC]
          # Per-replica step. If a cluster exists and this task id equals the configured error
          # worker, raises a resource-exhausted error to simulate an application failure.
          # Otherwise adds 1 to the model variable. Returns nothing.
          if cluster_spec and (
              distribute_lib.get_distribution_strategy(
              ).cluster_resolver.task_id == raise_app_error_on_worker):
            raise errors_impl.ResourceExhaustedError(
                node_def=None, op=None, message='Running out of resources')

          model.v.assign_add(constant_op.constant(1.))

        strategy.run(train_step)

        if current_step == STEPS_PER_EPOCH - 1:
          logging.info('epoch %d finished', current_epoch)

      logging.info('Start training at %d',
                   preemption_handler.total_run_calls)

      # If the training process has been restarted, verify that the expected
      # number of checkpoints have been written.
      # we also want to check training_finished, because there's a corner case
      # where the signal is sent quite late and training finishes before the
      # grace period ends.
      if training_restarted and training_restarted.is_set(
      ) and not training_finished.is_set():
        logging.info('training restarted')
        match_group = [
            re.search(r'.*ckpt-(\d+).index', a_file)
            for a_file in gfile.ListDirectory(checkpoint_dir)
        ]
        checkpoint_index = [
            a_match.group(1) for a_match in match_group if a_match
        ]
        if getattr(termination_config, 'grace_period', 0):
          # Two checkpoints were saved for the extended grace period.
          self.assertEqual(int(checkpoint_index[0]), 2)
        else:
          self.assertEqual(int(checkpoint_index[0]), 1)

      for epoch in range(
          preemption_handler.total_run_calls // STEPS_PER_EPOCH,
          EPOCHS_TO_RUN):

        for step in range(
            preemption_handler.total_run_calls % STEPS_PER_EPOCH,
            STEPS_PER_EPOCH):
          if api_wrapping_train:
            preemption_handler.run(distributed_train_step, epoch, step)
          else:
            preemption_handler.save_checkpoint_if_preempted()
            distributed_train_step(epoch, step)
        # Add some randomness to when preemption actually happens. We should
        # trigger it for sure if the training is coming to an end and it hasn't
        # been triggered yet.
        if epoch >= EPOCHS_TO_RUN - 2:
          trigger_it = True
        else:
          trigger_it = False

        self._maybe_trigger_a_preemption(training_started_event, trigger_it)

      training_finished.set()

      logging.info('Training finished.')

      self.assertEqual(
          model.v.numpy(),
          strategy.num_replicas_in_sync * EPOCHS_TO_RUN * STEPS_PER_EPOCH)

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
  def test_preemption_checkpointing(self, input_arg, strategy_option,
                                    api_wrapping_train):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the multi-worker branch the positional arguments are ordered as input
    # argument then strategy option, while the worker function expects strategy option first,
    # so the strategy is effectively not the requested one and a manager is always used.
    # Scenario: preemption signal then restart, for each combination of input kind (checkpoint
    # or manager), strategy (mirrored, one-device, multi-worker local, multi-worker multi-process)
    # and training wrapping. The termination configuration uses an exit function asserting that
    # the preemption-saved-time metric is positive and then exiting with code 42.
    # Multi-worker case: starts a four-worker cluster, waits for training to start, sends a
    # terminate signal to a random worker, waits for all workers to exit, restarts them and
    # joins. Other cases: a thread sends a terminate signal to the current process; the test
    # catches the exit, asserts code 42, and if training did not finish, reruns the worker
    # function as a restart.
    has_chief = False

    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt')
    def exit_fn_checking_metric():
      # [FUNCTION DOC]
      # Termination exit function: asserts that the checkpoint-saved-time metric is greater than
      # zero (a checkpoint was recorded) and then exits the process with code 42.
      self.assertGreater(tracking_util._preemption_checkpoint_saved_time_usecs.get_cell().value(), 0)  # pylint: disable=line-too-long
      sys.exit(42)

    termination_config = failure_handling.TerminationConfig(
        exit_fn=exit_fn_checking_metric)

    if strategy_option == 'MWMS_multi_worker':
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=CLUSTER_SIZE)
      training_started_event = multi_process_runner.manager().Event()
      training_restarted = multi_process_runner.manager().Event()
      training_finished = multi_process_runner.manager().Event()

      mpr = multi_process_runner.MultiProcessRunner(
          self.worker_fn,
          cluster_spec,
          args=(checkpoint_dir, cluster_spec, input_arg, strategy_option,
                [training_started_event], None, training_restarted,
                training_finished, termination_config),
          kwargs={'api_wrapping_train': api_wrapping_train},
          rpc_layer=rpc_layer,
          return_output=True,
          dependence_on_chief=has_chief)

      logging.info('Cluster starting.')
      mpr.start()
      while not training_started_event.is_set():
        time.sleep(1)

      logging.info('sending sigterm')
      killed_worker = random.randrange(0, CLUSTER_SIZE)
      os.kill(mpr.get_process_id('worker', killed_worker), signal.SIGTERM)

      logging.info('sigterm sent')
      raise_if_not_all_exit(0, mpr)

      logging.info('restarting workers')
      training_restarted.set()
      for worker_id in range(CLUSTER_SIZE):
        mpr.start_single_process('worker', worker_id, cluster_spec)
      logging.info('workers restarted')

      mpr.join(timeout=270)

    else:
      cluster_spec = server_lib.ClusterSpec({})

      training_started_event = threading.Event()
      training_restarted = threading.Event()
      training_finished = threading.Event()

      def sending_sigterm(training_started_event):
        # [FUNCTION DOC]
        # HELPER: thread body that simulates preemption. Waits (polling each second) until the
        # given event is set, logs, sets the event again, and sends the terminate signal to the
        # current process. Returns nothing.
        while not training_started_event.is_set():
          time.sleep(1)
        logging.info('sending sigterm')
        training_started_event.set()
        os.kill(os.getpid(), signal.SIGTERM)

      preemption_sender_thread = threading.Thread(
          target=sending_sigterm, args=(training_started_event,))
      preemption_sender_thread.start()

      caught_exit = False
      try:
        self.worker_fn(checkpoint_dir, cluster_spec, strategy_option, input_arg,
                       [training_started_event], None, training_restarted,
                       training_finished)

      except SystemExit as exit_error:
        caught_exit = True
        # We cannot use assertRaise instead, since termination is not always
        # triggered.
        self.assertEqual(exit_error.code, 42)  # pylint: disable=g-assert-in-except

      preemption_sender_thread.join(10)
      if not training_finished.is_set():
        self.assertTrue(caught_exit)

        logging.info('restarting workers')
        training_restarted.set()
        self.worker_fn(checkpoint_dir, cluster_spec, strategy_option, input_arg,
                       [training_started_event], None, training_restarted,
                       training_finished)

  def test_error_propagation(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the random task id is drawn from an inclusive range up to the cluster size, so
    # it can equal the size, which is not a valid task id; then no worker raises and the in-worker
    # assertion fails.
    # Scenario: an application error on one worker reaches all workers. Starts a four-worker
    # cluster where a randomly chosen worker raises a resource-exhausted error in its train step;
    # each worker process asserts that the error is raised with the message about running out of
    # resources. The runner is started and joined with a timeout.
    error_worker = random.randint(0, CLUSTER_SIZE)
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=False, num_workers=CLUSTER_SIZE)
    checkpoint_dir = self.get_temp_dir()

    def assert_raise_error():
      # [FUNCTION DOC]
      # Function run in every worker process: runs the worker function with the multi-worker
      # strategy and the chosen error worker, and asserts a resource-exhausted error is raised
      # whose text mentions running out of resources. Returns nothing.
      # Asserts that an error raised during a training step on one of the worker
      # is caught on all workers.
      with self.assertRaises(errors_impl.ResourceExhaustedError) as error:
        self.worker_fn(
            checkpoint_dir,
            cluster_spec,
            'MWMS_multi_worker',
            raise_app_error_on_worker=error_worker)
      self.assertIn('Running out of resources', str(error.exception))

    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    mpr = multi_process_runner.MultiProcessRunner(
        assert_raise_error,
        cluster_spec,
        rpc_layer=rpc_layer,
        return_output=True,
        dependence_on_chief=False,
    )

    logging.info('Cluster starting.')
    mpr.start()
    mpr.join(timeout=250)

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
  def test_grace_period_continue_training(self, input_arg, strategy_option):
    # [FUNCTION DOC]
    # Scenario: preemption with a grace period, so training continues until the period ends and
    # two checkpoints are saved before exit. Parameterized over input kind and strategy.
    # Multi-worker case: grace period 5 seconds, signal one random worker, wait for all workers to
    # exit, restart them and join. Other cases: grace period 1 second, a thread sends the signal
    # to the process, the exit code 42 is asserted, and training is rerun after a restart if it
    # had not finished. Restart checks live in the worker function.
    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt')

    if strategy_option == 'MWMS_multi_worker':
      grace_period = 5
      termination_config = failure_handling.TerminationConfig(
          grace_period=grace_period)
      has_chief = False
      cluster_spec = multi_worker_test_base.create_cluster_spec(
          has_chief=has_chief,
          num_workers=CLUSTER_SIZE)
      training_started_event = multi_process_runner.manager().Event()
      training_restarted = multi_process_runner.manager().Event()
      training_finished = multi_process_runner.manager().Event()

      mpr = multi_process_runner.MultiProcessRunner(
          self.worker_fn,
          cluster_spec,
          args=(checkpoint_dir, cluster_spec, strategy_option, input_arg,
                [training_started_event], None, training_restarted,
                training_finished, termination_config),
          rpc_layer=rpc_layer,
          return_output=True,
          dependence_on_chief=has_chief)

      logging.info('Cluster starting.')
      mpr.start()
      while not training_started_event.is_set():
        time.sleep(1)

      killed_worker = random.randrange(0, CLUSTER_SIZE)
      logging.info('sending SIGTERM')
      os.kill(mpr.get_process_id('worker', killed_worker), signal.SIGTERM)
      logging.info('SIGTERM sent')

      raise_if_not_all_exit(grace_period, mpr)

      logging.info('restarting workers')
      training_restarted.set()
      for worker_id in range(CLUSTER_SIZE):
        mpr.start_single_process('worker', worker_id, cluster_spec)
      logging.info('workers restarted')

      mpr.join(timeout=250)

    else:
      # This is because single worker trains super fast with regards to the size
      # of "model" here. With a longer grace period, the training just finishes
      # within the grace period so we can't verify the exit behavior.
      grace_period = 1
      termination_config = failure_handling.TerminationConfig(
          grace_period=grace_period)
      cluster_spec = server_lib.ClusterSpec({})

      training_started_event = threading.Event()
      training_restarted = threading.Event()
      training_finished = threading.Event()
      def sending_sigterm(training_started_event):
        # [FUNCTION DOC]
        # HELPER: thread body that simulates preemption. Waits (polling each second) until the
        # given event is set, logs, sets the event again, and sends the terminate signal to the
        # current process. Returns nothing.
        while not training_started_event.is_set():
          time.sleep(1)
        logging.info('sending sigterm')
        training_started_event.set()
        os.kill(os.getpid(), signal.SIGTERM)

      preemption_sender_thread = threading.Thread(
          target=sending_sigterm, args=(training_started_event,))
      preemption_sender_thread.start()

      caught_exit = False
      try:
        self.worker_fn(checkpoint_dir, cluster_spec, strategy_option, input_arg,
                       [training_started_event], None, training_restarted,
                       training_finished, termination_config)

      except SystemExit as exit_error:
        caught_exit = True
        # We cannot use assertRaise instead, since termination is not always
        # triggered.
        self.assertEqual(exit_error.code, 42)  # pylint: disable=g-assert-in-except

      preemption_sender_thread.join(10)
      if not training_finished.is_set():
        self.assertTrue(caught_exit)

        logging.info('restarting workers')
        training_restarted.set()
        self.worker_fn(checkpoint_dir, cluster_spec, strategy_option, input_arg,
                       [training_started_event], None, training_restarted,
                       training_finished, termination_config)

  def test_passed_in_save_fn(self):
    # [FUNCTION DOC]
    # Scenario: the termination configuration supplies a save function that does nothing.
    # Starts a four-worker cluster, signals a random worker, waits for all to exit (grace period
    # of 5 seconds), then lists the checkpoint directory and asserts that no checkpoint index
    # file exists, because the custom save function replaced the default checkpoint saving.
    if _is_oss():
      rpc_layer = 'grpc'
    else:
      rpc_layer = 'grpc+loas'

    checkpoint_dir = os.path.join(self.get_temp_dir(), 'fh_ckpt')
    gfile.MakeDirs(checkpoint_dir)

    save_fn = lambda: print('Do nothing')
    termination_config = failure_handling.TerminationConfig(
        save_fn=save_fn)

    has_chief = False
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=has_chief,
        num_workers=CLUSTER_SIZE)
    training_started_event = multi_process_runner.manager().Event()

    mpr = multi_process_runner.MultiProcessRunner(
        self.worker_fn,
        cluster_spec,
        args=(checkpoint_dir, cluster_spec, 'MWMS_multi_worker', 'checkpoint',
              [training_started_event], None, None, None, termination_config),
        rpc_layer=rpc_layer,
        return_output=True,
        dependence_on_chief=has_chief)

    logging.info('Cluster starting.')
    mpr.start()
    while not training_started_event.is_set():
      time.sleep(1)

    killed_worker = random.randrange(0, CLUSTER_SIZE)
    logging.info('sending SIGTERM')
    os.kill(mpr.get_process_id('worker', killed_worker), signal.SIGTERM)
    logging.info('SIGTERM sent')

    # 5 is the grace period length
    raise_if_not_all_exit(5, mpr)

    match_group = [
        re.search(r'.*ckpt-(\d+).index', a_file)
        for a_file in gfile.ListDirectory(checkpoint_dir)
    ]

    # By default, as tested by other test cases, checkpoint will be saved.
    # This passed in save_fn skips it.
    self.assertEmpty(match_group)


@_REGISTER_DECORATOR(PreemptionCheckpointTest)
def _save_test_case(pickler, obj):
  # [FUNCTION DOC]
  # HELPER: custom pickling hook registered with the dill library (when available) so that test
  # case instances can be sent to worker processes.
  # KNOWN ISSUE: when dill is missing the registration decorator is a pass-through that returns
  # its first argument, so this name is rebound to the test class itself and the hook is unused.
  # Inputs: the pickler and the object to save.
  # Returns the pickler's reduction of the object to a reconstruct function that builds a fresh
  # test case instance.
  def reconstruct(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: ignores all arguments and returns a new instance of the test case class.
    del args, kwargs
    return PreemptionCheckpointTest()

  return pickler.save_reduce(reconstruct, (), obj=obj)


if __name__ == '__main__':
  test_util.main()
