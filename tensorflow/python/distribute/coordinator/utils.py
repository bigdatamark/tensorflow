# [MODULE DOC]
# HELPER: small utilities for parameter server training in TF2, used by the coordinator and its
# tests. Provides a function that starts a worker or parameter server process and blocks, and a
# repeating timer thread class. No import time side effects.
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
"""TF2 parameter server training utilities.

Parameter server training in TF2 is currently under development.
"""
import threading
import time

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib


def start_server(cluster_resolver, protocol):
  # [FUNCTION DOC]
  # HELPER: entry point for multi-process tests or users who run one binary for every job.
  # Takes a cluster resolver (whose task type must be worker or ps, and which supplies cluster
  # spec, task type and task id) and a protocol name string. Raises a value error for any other
  # task type. Otherwise creates and starts a TensorFlow server for that task, logs a message,
  # and blocks forever (until the server stops) so the process does not exit. Returns nothing.
  """Start a server and block the process from exiting."""
  # This function is for multi-processing test or users who would like to have
  # every job run the same binary for simplicity.
  if not (cluster_resolver.task_type == 'worker' or
          cluster_resolver.task_type == 'ps'):
    raise ValueError('Unexpected task_type to start a server: {}'.format(
        cluster_resolver.task_type))

  server = server_lib.Server(
      cluster_resolver.cluster_spec().as_cluster_def(),
      job_name=cluster_resolver.task_type,
      task_index=cluster_resolver.task_id,
      protocol=protocol)

  logging.info('TensorFlow server started for job %s, task %d.',
               cluster_resolver.task_type, cluster_resolver.task_id)

  # Blocking the process that starts a server from exiting.
  server.join()


class RepeatedTimer(object):
  # [CLASS DOC]
  # HELPER: calls a function repeatedly at a fixed interval on timer threads, starting
  # immediately on construction. Holds the interval in seconds, the function and its extra
  # positional arguments, the creation time, the current timer thread and a running flag. Each
  # firing schedules the next firing before invoking the function. Call stop to cancel.
  """Threaded Repeated Timer from http://shortn/_3hMZTFr1Iv."""

  def __init__(self, interval, function, *args):
    # [FUNCTION DOC]
    # Records the interval (seconds), the function to call and its extra positional arguments,
    # notes the start time, and immediately starts the timer. Returns nothing.
    self._timer = None
    self.interval = interval
    self.function = function
    self.args = args
    self.start_time = time.time()
    self.is_running = False
    self.start()

  def _get_duration_sec(self):
    # [FUNCTION DOC]
    # HELPER: returns the whole seconds (truncated integer) elapsed since the timer was created.
    return int(time.time() - self.start_time)

  def _run(self):
    # [FUNCTION DOC]
    # HELPER: timer callback. Clears the running flag, restarts the timer for the next interval,
    # then calls the user function with the stored arguments. Exceptions from the function
    # propagate on the timer thread; the next firing is already scheduled. Returns nothing.
    self.is_running = False
    self.start()
    self.function(*self.args)

  def start(self):
    # [FUNCTION DOC]
    # Starts a new one-shot timer for the interval, if not already running, and marks it running.
    # Does nothing when already running. Returns nothing.
    if not self.is_running:
      self._timer = threading.Timer(self.interval, self._run)
      self._timer.start()
      self.is_running = True

  def stop(self):
    # [FUNCTION DOC]
    # Cancels the pending timer, clears the running flag and returns the whole seconds elapsed
    # since creation. Only the most recently created timer is cancelled.
    duration = self._get_duration_sec()
    self._timer.cancel()
    self.is_running = False
    return duration

