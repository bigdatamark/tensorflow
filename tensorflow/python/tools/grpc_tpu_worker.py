# [MODULE DOC]
# Executable script that starts a blocking TensorFlow GRPC server on a Cloud TPU worker host.
# It reads instance metadata from the Google Cloud metadata service, sets the TPU-related
# environment variables (before TensorFlow is imported by the app runner), builds a server
# definition for a single local task and joins the server forever. No import-time side effects
# beyond imports; the work happens in the run entry point when executed as a script.
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
# pylint: disable=g-import-not-at-top
"""Python-based TPU Worker GRPC server.

Start a blocking TPU Worker GRPC server.

Usage:
    python3 grpc_tpu_worker.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import requests

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.training import server_lib


def get_metadata(key):
  # [FUNCTION DOC]
  # HELPER: reads Google Cloud instance attributes for the TPU environment setup.
  # Performs an HTTP request to the metadata server for the instance attribute with the given
  # name (a string), sending the required vendor header, and returns the response body as text.
  # Raises whatever the HTTP library raises on connection failure; no status check is made, so an
  # error page would be returned as text.
  return requests.get(
      'http://metadata.google.internal/computeMetadata'
      '/v1/instance/attributes/{}'.format(key),
      headers={
          'Metadata-Flavor': 'Google'
      }).text


def get_host_ip():
  # [FUNCTION DOC]
  # HELPER: looks up this host's address for the TPU hostname override.
  # Takes no inputs. Queries the metadata server for the internal IP of the first network
  # interface and returns it as text. Raises on connection failure; the status is not checked.
  return requests.get(
      'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip',
      headers={
          'Metadata-Flavor': 'Google'
      }).text


def setup_env_vars():
  # [FUNCTION DOC]
  # HELPER: prepares process environment variables required by the TPU runtime.
  # Takes no inputs and returns nothing. Reads the worker number, accelerator type and worker
  # network endpoints from instance metadata, then sets environment variables for log level, task
  # id, device locking, mesh controller address (taken from the first endpoint's address plus port
  # 8476) and port. For accelerator types other than the v4 and v5 families, it also sets the
  # chips-per-host bounds and host bounds from a fixed table of v2/v3 slice sizes; an unknown type
  # in that case raises a key error. Finally sets the hostname override to the host IP.
  # Mutates os.environ and makes network calls.
  """Set environment variables."""
  worker_id = get_metadata('agent-worker-number')
  accelerator_type = get_metadata('accelerator-type')
  worker_network_endpoints = get_metadata('worker-network-endpoints')
  os.environ['TPU_STDERR_LOG_LEVEL'] = '0'
  os.environ['CLOUD_TPU_TASK_ID'] = worker_id
  os.environ['TPU_LOCK_DEVICE'] = 'true'
  os.environ['TPU_MESH_CONTROLLER_ADDRESS'] = (
      worker_network_endpoints.split(',')[0].split(':')[2] + ':8476'
  )
  os.environ['TPU_MESH_CONTROLLER_PORT'] = '8476'

  accelerator_type_to_host_bounds = {
      # v2
      'v2-8': '1,1,1',
      'v2-32': '2,2,1',
      'v2-128': '4,4,1',
      'v2-256': '4,8,1',
      'v2-512': '8,8,1',
      # v3
      'v3-8': '1,1,1',
      'v3-32': '2,2,1',
      'v3-64': '2,4,1',
      'v3-128': '4,4,1',
      'v3-256': '4,8,1',
      'v3-512': '8,8,1',
      'v3-1024': '8,16,1',
      'v3-2048': '16,16,1',

  }

  # If v4 TPU don't set any topology related flags,
  # libtpu will set these values.
  if not (accelerator_type.startswith('v4-') or
          accelerator_type.startswith('v5')):
    os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'
    os.environ['TPU_HOST_BOUNDS'] = accelerator_type_to_host_bounds[
        accelerator_type]

  # Set the hostname override.
  os.environ['TPU_HOSTNAME_OVERRIDE'] = get_host_ip()


def main(unused_args):
  # [FUNCTION DOC]
  # Application main for the worker. The argument (remaining command-line args) is ignored.
  # Builds a server definition using the grpc protocol with one job named tpu_worker containing a
  # single task at localhost on the port from the TPU_TASK_PORT environment variable (default
  # 8470 when unset or empty), creates a TensorFlow server with a default configuration and blocks
  # forever in its join call. Does not return normally.
  # Create Protobuf ServerDef.
  server_def = tensorflow_server_pb2.ServerDef(protocol='grpc')
  job_def = server_def.cluster.job.add()
  job_def.name = 'tpu_worker'
  tpu_task_port = os.getenv('TPU_TASK_PORT')
  if tpu_task_port is None or not tpu_task_port:
    tpu_task_port = '8470'  # If TPU task port is not available, use 8470.
  job_def.tasks[0] = 'localhost:' + tpu_task_port
  server_def.job_name = 'tpu_worker'
  server_def.task_index = 0

  config = config_pb2.ConfigProto()

  # Create GRPC Server instance
  server = server_lib.Server(server_def, config=config)

  # join() is blocking, unlike start()
  server.join()


def run():
  # [FUNCTION DOC]
  # Script entry point. Parses known command-line arguments (none are defined; the rest are kept),
  # sets up TPU environment variables first (they must exist before TensorFlow's app runner is
  # imported), then imports the platform app module lazily and runs main with the program name
  # plus the unparsed arguments. Returns nothing; side effects are environment changes and the
  # blocking server.
  parser = argparse.ArgumentParser()

  _, unparsed = parser.parse_known_args()
  # Must set environment variables before importing tensorflow.
  setup_env_vars()
  from tensorflow.python.platform import app
  app.run(main=main, argv=[sys.argv[0]] + unparsed)


if __name__ == '__main__':
  run()
