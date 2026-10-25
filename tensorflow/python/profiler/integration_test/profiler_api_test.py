# [MODULE DOC]
# Integration tests for the TensorFlow 2.x profiler API with a small Keras MNIST model under a
# collective all-reduce strategy. Covers no profiling, sampling mode (a profiler server in a
# worker thread plus a client trace request, with various delays) and programmatic mode (start
# and stop around training). Verifies that exactly one profile space file is produced.
# Runs under the multi-process runner test main.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf 2.x profiler."""

import glob
import os
import threading

import portpicker

from tensorflow.python.distribute import collective_all_reduce_strategy as collective_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_client
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.profiler.integration_test import mnist_testing_utils


def _model_setup():
  # [FUNCTION DOC]
  # HELPER: Shared fixture builder for the tests in this module.
  # KNOWN ISSUE: The docstring names the last returned item "mode"; it is actually the Keras model.
  # Turns on device placement logging (global context side effect), then inside a collective
  # all-reduce strategy scope builds a synthetic training dataset (batch size 64, 2 steps) and the
  # MNIST model. Returns the tuple of batch size, steps, training dataset and model.
  """Set up a MNIST Keras model for testing purposes.

  Builds a MNIST Keras model and returns model information.

  Returns:
    A tuple of (batch_size, steps, train_dataset, mode)
  """
  context.set_log_device_placement(True)
  batch_size = 64
  steps = 2
  with collective_strategy.CollectiveAllReduceStrategy().scope():
    # TODO(b/142509827): In rare cases this errors out at C++ level with the
    # "Connect failed" error message.
    train_ds, _ = mnist_testing_utils.mnist_synthetic_dataset(batch_size, steps)
    model = mnist_testing_utils.get_mnist_model((28, 28, 1))
  return batch_size, steps, train_ds, model


def _make_temp_log_dir(test_obj):
  # [FUNCTION DOC]
  # HELPER: Intended log directory provider.
  # KNOWN ISSUE: Not referenced anywhere in this file (dead code); tests call the temp directory
  # method directly. Returns the given test object's temporary directory.
  return test_obj.get_temp_dir()


class ProfilerApiTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Integration test case for profiling a Keras model fit. Fixtures: a worker-start event and a
  # flag telling the training thread when profiling finished, both reset in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the parent setup, creates the event that signals the worker has started
    # its profiler server, and sets the profile-done flag to false.
    super().setUp()
    self.worker_start = threading.Event()
    self.profile_done = False

  def _check_xspace_pb_exist(self, logdir):
    # [FUNCTION DOC]
    # HELPER: Shared assertion for the tests. Input: logdir. Looks for profile space files under
    # the profile plugin run directories and asserts exactly one match exists.
    path = os.path.join(logdir, 'plugins', 'profile', '*', '*.xplane.pb')
    self.assertEqual(1, len(glob.glob(path)),
                     'Expected one path match: ' + path)

  def test_single_worker_no_profiling(self):
    # [FUNCTION DOC]
    # Scenario: build the model fixture and train for 2 epochs without any profiling.
    # Expected: training completes without error; nothing else is asserted.
    """Test single worker without profiling."""

    _, steps, train_ds, model = _model_setup()

    model.fit(x=train_ds, epochs=2, steps_per_epoch=steps)

  def test_single_worker_sampling_mode(self, delay_ms=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The worker thread loops until the profile-done flag is set; if the trace request
    # fails before setting it (exception in the profiling thread), the worker waits forever
    # (though it is joined with a timeout).
    # Scenario: a worker thread starts a profiler server and trains repeatedly; a profiling thread
    # waits for the worker, then requests a 30 ms trace with options (host 2, python 0, device 1
    # and the optional delay_ms, default nothing) and sets the done flag. Threads are joined (the
    # worker with a 120 second timeout). Asserts: exactly one profile space file exists in the
    # log directory. Also reused by the delay variants below.
    """Test single worker sampling mode."""

    def on_worker(port, worker_start):
      # [FUNCTION DOC]
      # HELPER: Worker thread body. Inputs: port for the profiler server and the event to signal.
      # Starts the profiler server, builds the model fixture, signals the event, then keeps
      # training for 2 epochs repeatedly until the profile-done flag is set. Returns nothing.
      logging.info('worker starting server on {}'.format(port))
      profiler.start_server(port)
      _, steps, train_ds, model = _model_setup()
      worker_start.set()
      while True:
        model.fit(x=train_ds, epochs=2, steps_per_epoch=steps)
        if self.profile_done:
          break

    def on_profile(port, logdir, worker_start):
      # [FUNCTION DOC]
      # HELPER: Profiling thread body. Inputs: server port, log directory and the worker-start
      # event. Waits for the worker, then asks the local server for a 30 ms trace with 100
      # attempts and the options built from the outer delay, writing to the log directory, and
      # finally sets the profile-done flag so the worker stops. Exceptions end the thread early.
      # Request for 30 milliseconds of profile.
      duration_ms = 30

      worker_start.wait()
      options = profiler.ProfilerOptions(
          host_tracer_level=2,
          python_tracer_level=0,
          device_tracer_level=1,
          delay_ms=delay_ms,
      )

      profiler_client.trace('localhost:{}'.format(port), logdir, duration_ms,
                            '', 100, options)

      self.profile_done = True

    logdir = self.get_temp_dir()
    port = portpicker.pick_unused_port()
    thread_profiler = threading.Thread(
        target=on_profile, args=(port, logdir, self.worker_start))
    thread_worker = threading.Thread(
        target=on_worker, args=(port, self.worker_start))
    thread_worker.start()
    thread_profiler.start()
    thread_profiler.join()
    thread_worker.join(120)
    self._check_xspace_pb_exist(logdir)

  def test_single_worker_sampling_mode_short_delay(self):
    # [FUNCTION DOC]
    # Runs the sampling-mode scenario with a 1 ms requested start delay. The delayed start is
    # expected to arrive late and a retry then starts immediately; asserts one profile file exists.
    """Test single worker sampling mode with a short delay.

    Expect that requested delayed start time will arrive late, and a subsequent
    retry will issue an immediate start.
    """

    self.test_single_worker_sampling_mode(delay_ms=1)

  def test_single_worker_sampling_mode_long_delay(self):
    # [FUNCTION DOC]
    # Runs the sampling-mode scenario with a 1000 ms requested start delay; asserts one profile
    # file exists in the log directory.
    """Test single worker sampling mode with a long delay."""

    self.test_single_worker_sampling_mode(delay_ms=1000)

  def test_single_worker_programmatic_mode(self):
    # [FUNCTION DOC]
    # Scenario: start profiling with options (host 2, python 0, device 1), train the model for
    # 2 epochs, stop profiling. Asserts: exactly one profile space file exists in the log directory.
    """Test single worker programmatic mode."""
    logdir = self.get_temp_dir()

    options = profiler.ProfilerOptions(
        host_tracer_level=2,
        python_tracer_level=0,
        device_tracer_level=1,
    )
    profiler.start(logdir, options)
    _, steps, train_ds, model = _model_setup()
    model.fit(x=train_ds, epochs=2, steps_per_epoch=steps)
    profiler.stop()
    self._check_xspace_pb_exist(logdir)


if __name__ == '__main__':
  multi_process_runner.test_main()
