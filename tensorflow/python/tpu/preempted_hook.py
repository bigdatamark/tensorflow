# [MODULE DOC]
# Provides a session run hook for preemptible Cloud TPUs. CloudTPUPreemptedHook starts a
# background polling thread (the private _TPUPollingThread) once a session is created when
# running on Google Compute Engine; the thread periodically asks the cluster's cloud TPU client
# whether the TPU node is still recoverable and hard-exits the whole process if it is not.
# Import-time side effects: none beyond imports.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of the SessionRunHook for preemptible Cloud TPUs."""

import logging as _logging
import os
import threading
import time

from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook


class CloudTPUPreemptedHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # Session run hook for preemptible Cloud TPUs. Holds the cluster resolver given at
  # construction and, after a session is created on GCE, a polling thread that watches the TPU
  # node state. Overrides the hook callbacks for session creation and end. Used by passing an
  # instance in the hooks of a monitored session.
  """The SessionRunHook for preemptible Cloud TPUs.

  This is an implementation of SessionRunHook for the pre-emptible Google Cloud
  TPU service. It attempts to close the session if the TPU is preempted, and
  exits the coordinator process if the session cannot be closed.
  """

  def __init__(self, cluster):
    # [FUNCTION DOC]
    # Constructor. Input: cluster, a TPU cluster resolver object that exposes a cloud TPU
    # client. It is only stored for later use. Returns nothing, raises nothing.
    self._cluster = cluster

  def after_create_session(self, session, coord):
    # [FUNCTION DOC]
    # Hook callback run after a session is created. Inputs: session (the new session) and coord
    # (the coordinator, unused). If running on Google Compute Engine, creates a polling thread
    # for the cluster and session and starts it, storing it on the hook. Otherwise does nothing.
    # Returns nothing.
    if tpu_cluster_resolver.is_running_in_gce():
      self._tpu_poller = _TPUPollingThread(self._cluster, session)
      self._tpu_poller.start()

  def end(self, session):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if no polling thread was ever created (not running on GCE, so
    # after_create_session did nothing), the poller attribute does not exist and this raises an
    # attribute error.
    # Hook callback run when the session ends. Input: session (unused). Stops the polling thread
    # and waits for it to finish. Returns nothing.
    self._tpu_poller.stop()


class _TPUPollingThread(threading.Thread):
  # [CLASS DOC]
  # HELPER: background worker used by CloudTPUPreemptedHook.
  # Daemon thread that every 30 seconds checks whether the TPU node is recoverable. State: the
  # cluster, the session, a running flag, a session-closed flag and the polling interval. If the
  # node becomes unrecoverable it logs a warning and terminates the whole process immediately.
  """A thread that polls the state of a TPU node.

  When the node transitions into a TERMINAL state (PREEMPTED, TERMINATED)
  that's considered as not recoverable by the underlying infrastructure,
  it attempts to close the session, and exits the entire process if the
  session.close() stucks.
  """

  def __init__(self, cluster, session):
    # [FUNCTION DOC]
    # Constructor. Inputs: cluster (TPU cluster resolver with a cloud TPU client) and session
    # (the session the thread is guarding). Marks the thread as a daemon, initializes the
    # running and session-closed flags, and sets a 30 second polling interval. Side effect:
    # lowers the log level of the Google API discovery and OAuth client loggers to warning
    # because they are chatty. Returns nothing.
    super(_TPUPollingThread, self).__init__()

    self.daemon = True
    self._running = True
    self._session_closed = False
    self._cluster = cluster
    self._session = session
    self._interval = 30

    # Some of the Google API libraries are quite chatty, so disable them.
    for name in ['googleapiclient.discovery', 'oauth2client.client']:
      _logging.getLogger(name).setLevel(_logging.WARNING)

  def stop(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring of the class says it closes the session, but the session-closed
    # flag set here is never read and no session is closed anywhere in this class.
    # Asks the thread to stop: clears the running flag, sets the session-closed flag, then blocks
    # until the thread finishes (may wait up to one polling interval). Takes no inputs; returns
    # nothing.
    self._running = False
    self._session_closed = True
    self.join()

  def run(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the class docstring claims it tries to close the session first and only exits
    # if closing hangs, but the code never closes the session; it exits the process directly.
    # Thread body. If not running on GCE, logs a warning, clears the running flag and returns.
    # Otherwise loops while running: asks the cluster's cloud TPU client if the node is
    # recoverable; if not, logs a warning with the TPU name and state and terminates the process
    # immediately with exit status 1 (no cleanup handlers); else sleeps for the polling interval.
    # Returns nothing.
    if not tpu_cluster_resolver.is_running_in_gce():
      logging.warning(
          'TPUPollingThread is running in a non-GCE environment, exiting...')
      self._running = False
      return

    while self._running:
      recoverable = self._cluster._cloud_tpu_client.recoverable()  # pylint: disable=protected-access
      if not recoverable:
        logging.warning(
            'TPUPollingThread found TPU %s in state %s',
            self._cluster._tpu, self._cluster._cloud_tpu_client.state())  # pylint: disable=protected-access
        os._exit(1)  # pylint: disable=protected-access
      time.sleep(self._interval)
