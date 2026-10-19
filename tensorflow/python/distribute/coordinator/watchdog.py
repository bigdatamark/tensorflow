# [MODULE DOC]
# Watchdog support for the cluster coordinator. Defines a class that runs a daemon thread which,
# when no closure has finished for a configured number of seconds, logs a warning, calls an
# optional callback and dumps the stack traces of all threads to a file, to help diagnose hangs.
# The timeout can be overridden through an environment variable. Nothing is started at import.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Watchdog that monitors activity of ClusterCoordinator."""

import faulthandler
import os
import sys
import threading
import time
from absl import logging


class WatchDog(object):
  # [CLASS DOC]
  # Hang detector for the cluster coordinator. Holds the timeout in seconds (non-positive means
  # disabled), the time of the last activity, the output file for stack traces, an optional
  # trigger callback, a stopped flag and, when enabled, a daemon thread named for the watchdog.
  # Created by the coordinator; call the done-report method on every finished closure and the
  # stop method when finished.
  """A class to dump stack traces if no activity happens in ClusterCoordinator."""

  def __init__(self, timeout=-1, traceback_file=sys.stdout, on_triggered=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the default traceback file is the standard output stream object captured when
    # the module is imported, so later redirection of standard output is not honored by default.
    # Parameters: timeout (seconds, integer, default minus one meaning disabled), traceback_file
    # (writable file object for stack traces, default standard output) and on_triggered (optional
    # callback with no arguments). If the environment variable for the watchdog timeout holds only
    # digits it overrides the timeout argument. Records the last activity time as now. If the
    # effective timeout is positive, starts a daemon thread running the watchdog loop. Returns
    # nothing.
    if os.environ.get("TF_CLUSTER_COORDINATOR_WATCH_DOG_TIMEOUT",
                      "").isnumeric():
      timeout = int(os.environ["TF_CLUSTER_COORDINATOR_WATCH_DOG_TIMEOUT"])
    self._timeout = timeout
    self._last_activity_time = time.time()
    self._traceback_file = traceback_file
    self._on_triggered = on_triggered
    self._stopped = False
    if timeout > 0:
      self._watchdog_thread = threading.Thread(
          target=self._watchdog_function, name="WatchDog", daemon=True)
      self._watchdog_thread.start()

  def stop(self):
    # [FUNCTION DOC]
    # Asks the watchdog thread to end by setting the stopped flag. Does not wait for the thread;
    # the thread exits after its current sleep. Returns nothing.
    self._stopped = True

  def _watchdog_function(self):
    # [FUNCTION DOC]
    # HELPER: body of the watchdog thread.
    # Logs the start, then loops until stopped: sleeps one tenth of the timeout, and if the time
    # since the last activity is at least the timeout, logs a warning, calls the callback if set,
    # dumps all thread tracebacks to the file, writes an end marker line and resets the last
    # activity time to now (so it can fire again after another timeout). Returns nothing.
    """The watchdog thread."""
    logging.info("Starting watchdog thread with timeout %r", self._timeout)
    while not self._stopped:
      time.sleep(self._timeout / 10.0)
      current_time = time.time()
      if current_time - self._last_activity_time >= self._timeout:
        logging.warning(
            "No activity for ClusterCoordinator for %r seconds. "
            "Dumping stack traces.", self._timeout)
        if self._on_triggered:
          self._on_triggered()
        faulthandler.dump_traceback(file=self._traceback_file)
        self._traceback_file.write("==== End of stack traces ====\n")
        self._last_activity_time = current_time

  def report_closure_done(self):
    # [FUNCTION DOC]
    # Records activity by setting the last activity time to now; does nothing when the watchdog is
    # disabled (timeout not positive). Returns nothing.
    if self._timeout > 0:
      self._last_activity_time = time.time()
