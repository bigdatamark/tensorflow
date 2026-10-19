# [MODULE DOC]
# HELPER: backend set-up helpers shared by DTensor tests.
# Provides a small configurator class that cleans up the accelerator system after a test, a
# function that restricts a process to a slice of host devices for multi-client tests, a
# function returning the multiprocessing context, and a thin wrapper that runs a test main.
# No import-time side effects beyond imports. Used by the DTensor test utility module.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utility to set up DTensor backend in tests."""

# LINT.IfChange
import multiprocessing
import os

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.python.platform import test as tf_test


class DTensorTestBackendConfigurator:
  # [CLASS DOC]
  # HELPER: per-test backend set-up and tear-down helper used by the DTensor base test class.
  # Holds a reference to the owning test case. Currently only implements tear-down, which shuts
  # down the accelerator system if it was initialized. Created in each test's set-up.
  """Configurate test backends."""

  def __init__(self, test_case: tf_test.TestCase):
    # [FUNCTION DOC]
    # Constructor. Stores the given test case (a TensorFlow test case instance) for later use.
    # Returns nothing and has no other side effects.
    self._test_case = test_case
    # TODO(b/260771689): Refactor common backend set up logic to here.

  def tearDown(self):
    # [FUNCTION DOC]
    # Test tear-down hook. If the DTensor accelerator system is currently initialized, shuts it
    # down; otherwise does nothing. Takes no inputs and returns nothing.
    # Side effect: global accelerator system state is shut down.
    # Only need to explicitly shuts down TPU system in TFRT since in current
    # runtime, the shutdown is done in initialization process.
    if accelerator_util.is_initialized():
      accelerator_util.shutdown_accelerator_system()


def slice_host_devices_for_multiworker(num_clients, client_id, ports):
  # [FUNCTION DOC]
  # HELPER: multi-client test set-up, limits which devices a client process can see.
  # KNOWN ISSUE: when the client count is zero, the function deletes the CUDA and HIP visibility
  # environment variables, which raises a key error if either is not set.
  # Inputs: num_clients (integer number of client processes), client_id (integer index of this
  # client), ports (list of ports, unused). If num_clients is zero, removes the GPU-visibility
  # environment variables so all GPUs are visible. Otherwise sets the GPU-visibility variables
  # and the cloud TPU task id variable to the client id. If the test backend name contains tpu,
  # raises a not-implemented error since multi-client TPU tests are unsupported in open source.
  # Returns nothing; mutates process environment variables.
  """Configure the current process to only use a slice of devices."""
  if num_clients == 0:
    # All GPUs are visible to the client.
    del os.environ['CUDA_VISIBLE_DEVICES']
    del os.environ['HIP_VISIBLE_DEVICES']
  else:
    # Make the client_id-th GPU visible to the client.
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{client_id}'
    os.environ['HIP_VISIBLE_DEVICES'] = f'{client_id}'
    # Make the client_id-th (4x) TPU cores visible to the client.
    os.environ['CLOUD_TPU_TASK_ID'] = f'{client_id}'
    if 'tpu' in DTENSOR_TEST_UTIL_BACKEND.value:
      del ports  # Unused due to lack of implementation.
      # We need to find out if there is a way to slice a CloudTPU host to
      # multiple workers.
      raise NotImplementedError(
          'OSS multi-client tests of TPU is not supported.'
      )


def get_mp_context():
  # [FUNCTION DOC]
  # HELPER: supplies the multiprocessing context for multi-client tests.
  # Returns a multiprocessing context that uses the forkserver start method. No inputs.
  return multiprocessing.get_context('forkserver')


def handle_test_main(main, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: hook for launching a test entry point; in this open-source version it simply calls
  # the given main callback with all remaining positional and keyword arguments and returns
  # nothing (the callback's result is discarded).
  main(*args, **kwargs)


# LINT.ThenChange(test_backend_util.py)
