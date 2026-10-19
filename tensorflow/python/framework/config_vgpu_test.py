# [MODULE DOC]
# Tests for virtual GPU configuration through the framework config module: splitting physical
# GPUs into several logical devices with memory limits and optional device ordinals, then
# checking how many logical GPU devices are listed. Tests need a fresh process each (virtual GPU
# initialization limitation), and skip themselves when the required physical GPU count is absent.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests that the system configuration methods work properly."""

from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def reset_eager(fn):
  # [FUNCTION DOC]
  # HELPER: decorator for the tests in this file that restores a clean eager context afterwards.
  # Takes a test function and returns a wrapper that calls it with the same arguments and, even
  # on failure, resets the JIT compiler flags and the global context, re-enables eager execution
  # and asserts a context exists. Side effect: replaces the global runtime context.

  def wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper produced by the reset decorator. Forwards all positional and keyword
    # arguments to the wrapped test and returns its result; in a final step always resets the
    # JIT flags and global context and re-enables eager execution. Raises an assertion error if
    # no context exists afterwards; exceptions from the wrapped test propagate.
    try:
      return fn(*args, **kwargs)
    finally:
      # Reset the context.
      context._reset_jit_compiler_flags()
      context._reset_context()
      ops.enable_eager_execution_internal()
      assert context._context is not None

  return wrapper


class DeviceTest(test.TestCase):
  # [CLASS DOC]
  # Test case for virtual GPU device configuration. Each test must run in its own process, which
  # is enforced by a class-level flag recording whether a test already ran (see setUp).

  # Due to a limitation in TensorFlow virtual GPU initialization, tests in
  # this file must run with a new process per test. This is achieved by
  # setting the number of shards to a large number.
  already_run = False

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture run before each test. Calls the parent setup, then raises a runtime error if a test
    # already ran in this process (the message asks to raise the shard count), otherwise marks
    # that a test has run. Note the flag is set on the instance, not the class, so the guard only
    # trips if the same instance is reused.
    super().setUp()
    if self.already_run:
      raise RuntimeError(
          'Each test in this test suite must run in a separate process. '
          'Increase number of shards used to run this test.')
    self.already_run = True

  @reset_eager
  def testVGpu1P2V(self):
    # [FUNCTION DOC]
    # Scenario: exactly one physical GPU (otherwise skipped). Split it into two logical devices of
    # 100 memory limit with device ordinals 0 and 1, initialize the context, list logical GPUs.
    # Expected: two logical GPU devices.
    gpus = config.list_physical_devices('GPU')
    if len(gpus) != 1:
      self.skipTest('Need 1 GPUs')

    config.set_logical_device_configuration(gpus[0], [
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=0),
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=1)
    ])

    context.ensure_initialized()

    vcpus = config.list_logical_devices('GPU')
    self.assertEqual(len(vcpus), 2)

  @reset_eager
  def testVGpu2P2V1Default(self):
    # [FUNCTION DOC]
    # Scenario: exactly two physical GPUs (otherwise skipped). Split only the first into two
    # logical devices with memory limit 100 and no ordinals; leave the second default. Expected:
    # three logical GPU devices after context initialization.
    gpus = config.list_physical_devices('GPU')
    if len(gpus) != 2:
      self.skipTest('Need 2 GPUs')

    config.set_logical_device_configuration(gpus[0], [
        context.LogicalDeviceConfiguration(memory_limit=100),
        context.LogicalDeviceConfiguration(memory_limit=100)
    ])

    context.ensure_initialized()

    vcpus = config.list_logical_devices('GPU')
    self.assertEqual(len(vcpus), 3)

  @reset_eager
  def testGpu2P2V2V(self):
    # [FUNCTION DOC]
    # Scenario: exactly two physical GPUs (otherwise skipped). Split each into two logical devices
    # with memory limit 100 and explicit ordinals 0 and 1. Expected: four logical GPU devices
    # after context initialization.
    gpus = config.list_physical_devices('GPU')
    if len(gpus) != 2:
      self.skipTest('Need 2 GPUs')

    config.set_logical_device_configuration(gpus[0], [
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=0),
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=1)
    ])

    config.set_logical_device_configuration(gpus[1], [
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=0),
        context.LogicalDeviceConfiguration(
            memory_limit=100, experimental_device_ordinal=1)
    ])

    context.ensure_initialized()

    vcpus = config.list_logical_devices('GPU')
    self.assertEqual(len(vcpus), 4)

  @reset_eager
  def testGpu2P2D2D(self):
    # [FUNCTION DOC]
    # Scenario: exactly two physical GPUs (otherwise skipped). Split each into two logical devices
    # with memory limit 100 and no explicit ordinals. Expected: four logical GPU devices after
    # context initialization.
    gpus = config.list_physical_devices('GPU')
    if len(gpus) != 2:
      self.skipTest('Need 2 GPUs')

    config.set_logical_device_configuration(gpus[0], [
        context.LogicalDeviceConfiguration(memory_limit=100),
        context.LogicalDeviceConfiguration(memory_limit=100)
    ])

    config.set_logical_device_configuration(gpus[1], [
        context.LogicalDeviceConfiguration(memory_limit=100),
        context.LogicalDeviceConfiguration(memory_limit=100)
    ])

    context.ensure_initialized()

    vcpus = config.list_logical_devices('GPU')
    self.assertEqual(len(vcpus), 4)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
