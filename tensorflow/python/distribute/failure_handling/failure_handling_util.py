# [MODULE DOC]
# HELPER: platform-detection and Google Cloud utilities for the preemption handling modules.
# Provides constants (metadata header, restartable exit code, default grace period for Google
# Cloud), exit callbacks for Google Cloud and for TPU strategy, a function that reads Google
# Cloud compute metadata, a termination watcher based on the maintenance-event metadata, a check
# for whether the program runs on Google Cloud, an enumeration of platform and device kinds, and
# a function that detects the current platform and device. No import-time side effects other than
# imports and constant definitions.
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
"""Util of GCE specifics to ingegrate with WorkerPreemptionHandler."""
import enum
import os
import sys
import requests

from six.moves.urllib import request
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging


GCP_METADATA_HEADER = {'Metadata-Flavor': 'Google'}
_GCE_METADATA_URL_ENV_VARIABLE = 'GCE_METADATA_IP'
_RESTARTABLE_EXIT_CODE = 143
GRACE_PERIOD_GCE = 3600


def gce_exit_fn():
  # [FUNCTION DOC]
  # Default exit callback for Google Cloud: terminates the process by raising the system-exit
  # request with the restartable status code 143, so a supervisor can restart the job. Takes no
  # inputs and does not return normally.
  sys.exit(_RESTARTABLE_EXIT_CODE)


def default_tpu_exit_fn():
  # [FUNCTION DOC]
  # KNOWN ISSUE: uses an unrestricted catch-all error handler, so any failure (not only the
  # expected error once workers are down) triggers the exit with status 42.
  # Exit callback for TPU strategy. Logs, then asks the coordination service for a key that is
  # never set, which blocks until the workers are gone and then fails. On any failure logs a
  # restart message and exits the process with status 42. Takes no inputs; returns nothing
  # normally.
  """Default exit function to run after saving checkpoint for TPUStrategy.

  For TPUStrategy, we want the coordinator to exit after workers are down so
  that restarted coordinator would not connect to workers scheduled to be
  preempted. This function achieves so by attempting to get a key-value store
  from coordination service, which will block until workers are done and then
  returns with error. Then we have the coordinator sys.exit(42) to re-schedule
  the job.
  """
  logging.info('Waiting for workers to exit...')
  try:
    context.context().get_config_key_value('BLOCK_TILL_EXIT')
  except:  # pylint: disable=bare-except
    logging.info('Restarting cluster due to preemption.')
    sys.exit(42)


def request_compute_metadata(path):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the request has no timeout, so it can block indefinitely if the metadata
  # server is unreachable; the metadata header is also repeated inline instead of using the
  # module constant.
  # Fetches a value from the Google Cloud instance metadata server. Input: the metadata path
  # (text). The server address comes from an environment variable if set, otherwise the standard
  # internal host name. Sends a request with the metadata header and returns the response body
  # as text (decoded as UTF-8 when received as bytes). Network errors propagate to the caller.
  """Returns GCE VM compute metadata."""
  gce_metadata_endpoint = 'http://' + os.environ.get(
      _GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')
  req = request.Request(
      '%s/computeMetadata/v1/%s' % (gce_metadata_endpoint, path),
      headers={'Metadata-Flavor': 'Google'})
  info = request.urlopen(req).read()
  if isinstance(info, bytes):
    return info.decode('utf-8')
  else:
    return info


def termination_watcher_function_gce():
  # [FUNCTION DOC]
  # Default termination watcher for Google Cloud. Queries the instance maintenance-event
  # metadata and returns true when it says the host will terminate for maintenance, false
  # otherwise. Network errors from the metadata request propagate. Takes no inputs.
  result = request_compute_metadata(
      'instance/maintenance-event') == 'TERMINATE_ON_HOST_MAINTENANCE'
  return result


def on_gcp():
  # [FUNCTION DOC]
  # Detects Google Cloud by requesting the instance host name from the metadata server with a
  # five second timeout. Returns true if the response status is success, false on a success
  # failure status or on any request-level error (connection problem, timeout). Takes no inputs.
  """Detect whether the current running environment is on GCP."""
  gce_metadata_endpoint = 'http://' + os.environ.get(
      _GCE_METADATA_URL_ENV_VARIABLE, 'metadata.google.internal')

  try:
    # Timeout in 5 seconds, in case the test environment has connectivity issue.
    # There is not default timeout, which means it might block forever.
    response = requests.get(
        '%s/computeMetadata/v1/%s' %
        (gce_metadata_endpoint, 'instance/hostname'),
        headers=GCP_METADATA_HEADER,
        timeout=5)
    return response.status_code == 200
  except requests.exceptions.RequestException:
    return False


@enum.unique
class PlatformDevice(enum.Enum):
  # [CLASS DOC]
  # Enumeration (values must be unique) of the platform and accelerator combinations the
  # failure handling code distinguishes: internal CPU, GPU and TPU, Google Cloud GPU, TPU and
  # CPU, and unsupported. Each member's value is a short text label. The unsupported member is
  # never returned by the detection function.
  INTERNAL_CPU = 'internal_CPU'
  INTERNAL_GPU = 'internal_GPU'
  INTERNAL_TPU = 'internal_TPU'
  GCE_GPU = 'GCE_GPU'
  GCE_TPU = 'GCE_TPU'
  GCE_CPU = 'GCE_CPU'
  UNSUPPORTED = 'unsupported'


def detect_platform():
  # [FUNCTION DOC]
  # Determines the platform kind and accelerator. If running on Google Cloud, returns the cloud
  # GPU member if logical GPU devices exist, else cloud TPU if logical TPU devices exist, else
  # cloud CPU. Otherwise returns the internal GPU, TPU or CPU member by the same precedence
  # (GPU first). Queries the metadata server and the eager context's device list. Takes no
  # inputs.
  """Returns the platform and device information."""
  if on_gcp():
    if context.context().list_logical_devices('GPU'):
      return PlatformDevice.GCE_GPU
    elif context.context().list_logical_devices('TPU'):
      return PlatformDevice.GCE_TPU
    else:
      return PlatformDevice.GCE_CPU

  else:
    if context.context().list_logical_devices('GPU'):
      return PlatformDevice.INTERNAL_GPU
    elif context.context().list_logical_devices('TPU'):
      return PlatformDevice.INTERNAL_TPU
    else:
      return PlatformDevice.INTERNAL_CPU
