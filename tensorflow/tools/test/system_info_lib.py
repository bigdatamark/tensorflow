# [MODULE DOC]
# HELPER: helper library for the TensorFlow test and benchmark tooling (log-gathering scripts).
# Library that collects a description of the machine a test runs on and fills in the
# MachineConfiguration protocol buffer message defined in the test log proto module.
# It gathers CPU details, platform details, memory totals, hostname, the devices visible to
# TensorFlow and GPU device descriptions (via the sibling GPU info library).
# Depends on the third-party cpuinfo and psutil packages, which must be installed separately.
# Importing has no side effects beyond the imports; all work happens when the functions run.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Library for getting system information during TensorFlow tests."""

import glob
import multiprocessing
import platform
import re
import socket

# pylint: disable=g-bad-import-order
# Note: cpuinfo and psutil are not installed for you in the TensorFlow
# OSS tree.  They are installable via pip.
import cpuinfo
import psutil
# pylint: enable=g-bad-import-order

from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile
from tensorflow.tools.test import gpu_info_lib


def gather_machine_configuration():
  # [FUNCTION DOC]
  # Top-level entry point: builds a complete MachineConfiguration message for this machine.
  # Takes no inputs. Fills in CPU info, platform info, the TensorFlow-visible device list,
  # GPU device descriptions (packed into the generic device info field), memory info and the
  # hostname, and returns the populated message.
  # Side effects: enumerates TensorFlow local devices (which initializes devices) and probes
  # GPUs. The order matters: the TensorFlow device listing must run before the direct GPU
  # probing because the latter may load the CUDA runtime directly and confuse the executor.
  """Gather Machine Configuration.  This is the top level fn of this library."""
  config = test_log_pb2.MachineConfiguration()

  config.cpu_info.CopyFrom(gather_cpu_info())
  config.platform_info.CopyFrom(gather_platform_info())

  # gather_available_device_info must come before gather_gpu_devices
  # because the latter may access libcudart directly, which confuses
  # TensorFlow StreamExecutor.
  for d in gather_available_device_info():
    config.available_device_info.add().CopyFrom(d)
  for gpu in gpu_info_lib.gather_gpu_devices():
    config.device_info.add().Pack(gpu)

  config.memory_info.CopyFrom(gather_memory_info())

  config.hostname = gather_hostname()

  return config


def gather_hostname():
  # [FUNCTION DOC]
  # Returns the network host name of this machine as a string. Takes no inputs, no side effects.
  return socket.gethostname()


def gather_memory_info():
  # [FUNCTION DOC]
  # Returns a MemoryInfo message holding total and currently available system memory in bytes,
  # as reported by the psutil package. Takes no inputs; no side effects.
  """Gather memory info."""
  mem_info = test_log_pb2.MemoryInfo()
  vmem = psutil.virtual_memory()
  mem_info.total = vmem.total
  mem_info.available = vmem.available
  return mem_info


def gather_cpu_info():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the core count first set from the multiprocessing module is later overwritten
  # by the count reported by the cpuinfo package, so the first assignment is effectively dead.
  # KNOWN ISSUE: the CPU governor files are opened without ever being closed explicitly, and
  # a missing "hz_advertised_raw" entry from cpuinfo would raise an unhandled error.
  # Returns a CPUInfo message. Takes no inputs. Sets the total core count, then the number of
  # cores this process is allowed to use (parsed from the allowed-CPU bitmask in the process
  # status file on Linux; falls back to the total count when unavailable or on an OpError),
  # then the brand string, core count, advertised clock speed in MHz and the L2 cache size
  # in bytes (when reported in KB) from the cpuinfo package. Finally it reads the frequency
  # scaling governor of every CPU and records it, or the word "mixed" when they differ.
  # Governor read failures reported as OpError are ignored. Assumes all CPUs are identical.
  """Gather CPU Information.  Assumes all CPUs are the same."""
  cpu_info = test_log_pb2.CPUInfo()
  cpu_info.num_cores = multiprocessing.cpu_count()

  # Gather num_cores_allowed
  try:
    with gfile.GFile('/proc/self/status', 'rb') as fh:
      nc = re.search(r'(?m)^Cpus_allowed:\s*(.*)$', fh.read().decode('utf-8'))
    if nc:  # e.g. 'ff' => 8, 'fff' => 12
      cpu_info.num_cores_allowed = (
          bin(int(nc.group(1).replace(',', ''), 16)).count('1'))
  except errors.OpError:
    pass
  finally:
    if cpu_info.num_cores_allowed == 0:
      cpu_info.num_cores_allowed = cpu_info.num_cores

  # Gather the rest
  info = cpuinfo.get_cpu_info()
  cpu_info.cpu_info = info['brand']
  cpu_info.num_cores = info['count']
  cpu_info.mhz_per_cpu = info['hz_advertised_raw'][0] / 1.0e6
  l2_cache_size = re.match(r'(\d+)', str(info.get('l2_cache_size', '')))
  if l2_cache_size:
    # If a value is returned, it's in KB
    cpu_info.cache_size['L2'] = int(l2_cache_size.group(0)) * 1024

  # Try to get the CPU governor
  try:
    cpu_governors = set([
        gfile.GFile(f, 'r').readline().rstrip()
        for f in glob.glob(
            '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')
    ])
    if cpu_governors:
      if len(cpu_governors) > 1:
        cpu_info.cpu_governor = 'mixed'
      else:
        cpu_info.cpu_governor = list(cpu_governors)[0]
  except errors.OpError:
    pass

  return cpu_info


def gather_available_device_info():
  # [FUNCTION DOC]
  # Returns a list of AvailableDeviceInfo messages, one per local device TensorFlow can see,
  # with name, device type, memory limit and physical device description copied over.
  # Takes no inputs. Side effect: asks TensorFlow to list local devices, which may initialize them.
  """Gather list of devices available to TensorFlow.

  Returns:
    A list of test_log_pb2.AvailableDeviceInfo messages.
  """
  device_info_list = []
  devices = device_lib.list_local_devices()

  for d in devices:
    device_info = test_log_pb2.AvailableDeviceInfo()
    device_info.name = d.name
    device_info.type = d.device_type
    device_info.memory_limit = d.memory_limit
    device_info.physical_description = d.physical_device_desc
    device_info_list.append(device_info)

  return device_info_list


def gather_platform_info():
  # [FUNCTION DOC]
  # Returns a PlatformInfo message with the interpreter architecture bits and linkage, machine
  # type, OS release, OS system name and OS version from the standard platform module.
  # Takes no inputs; no side effects.
  """Gather platform info."""
  platform_info = test_log_pb2.PlatformInfo()
  (platform_info.bits, platform_info.linkage) = platform.architecture()
  platform_info.machine = platform.machine()
  platform_info.release = platform.release()
  platform_info.system = platform.system()
  platform_info.version = platform.version()
  return platform_info
