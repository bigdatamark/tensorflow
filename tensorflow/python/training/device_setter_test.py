# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Unit tests for replica_device_setter: verifies device strings assigned to variables,
# initializers and compute ops across cluster description forms, custom job names, CPU
# constraints, explicit device overrides, and the no-merge mode. Tests only; no import-time
# side effects.
"""Tests for device function for replicated training."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.training import server_lib


class DeviceSetterTest(test.TestCase):
  # [CLASS DOC]
  # Test case for replica_device_setter. Shared fixture: a class-level cluster spec with two
  # ps tasks and three worker tasks. Tests run in deprecated graph mode.

  _cluster_spec = server_lib.ClusterSpec({
      "ps": ["ps0:2222", "ps1:2222"],
      "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
  })

  @test_util.run_deprecated_v1
  def testCPUOverride(self):
    # [FUNCTION DOC]
    # Scenario: variables created with and without an inner CPU device scope. Expects the first
    # variable on ps task 0 with CPU, the second on ps task 1, and an add under a CPU scope on
    # the worker CPU.
    with ops.device(
        device_setter.replica_device_setter(cluster=self._cluster_spec)):
      with ops.device("/cpu:0"):
        v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      with ops.device("/cpu:0"):
        a = v + w
      self.assertDeviceEqual("/job:ps/task:0/cpu:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0/cpu:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:worker/cpu:0", a.device)

  @test_util.run_deprecated_v1
  def testResource(self):
    # [FUNCTION DOC]
    # Scenario: a resource variable created under the setter. Expects placement on ps task 0.
    with ops.device(
        device_setter.replica_device_setter(cluster=self._cluster_spec)):
      v = resource_variable_ops.ResourceVariable([1, 2])
      self.assertDeviceEqual("/job:ps/task:0", v.device)

  @test_util.run_deprecated_v1
  def testPS2TasksWithClusterSpecClass(self):
    # [FUNCTION DOC]
    # Scenario: cluster given as a cluster spec object. Expects two variables on ps tasks 0 and
    # 1 (also their initializers) and the add op on the worker job.
    with ops.device(
        device_setter.replica_device_setter(cluster=self._cluster_spec)):
      v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  @test_util.run_deprecated_v1
  def testPS2TasksPinVariableToJob(self):
    # [FUNCTION DOC]
    # Scenario: a variable explicitly pinned to another job stays there, while one pinned to
    # the ps job without a task gets the next task. Expects ps task 0, the moon job, ps task 1,
    # and the add op on the worker job.
    with ops.device(
        device_setter.replica_device_setter(cluster=self._cluster_spec)):
      v = variables.Variable([1, 2])
      with ops.device("/job:moon"):
        w = variables.Variable([2, 1])
        with ops.device("/job:ps"):  # Explicit PS job will get task set.
          x = variables.Variable([0, 1])
      a = v + w + x
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:moon", w.device)
      self.assertDeviceEqual("/job:moon", w.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", x.device)
      self.assertDeviceEqual("/job:ps/task:1", x.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  @test_util.run_deprecated_v1
  def testPS2TasksUseCpuForPS(self):
    # [FUNCTION DOC]
    # Scenario: one ps task whose ps device is a CPU without a job. Expects the variable on
    # that CPU, a variable pinned to another job merged with the CPU, and the add op on the
    # worker job.
    with ops.device(
        device_setter.replica_device_setter(ps_tasks=1, ps_device="/cpu:0")):
      v = variables.Variable([1, 2])
      with ops.device("/job:moon"):
        w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/cpu:0", v.device)
      self.assertDeviceEqual("/cpu:0", v.initializer.device)
      self.assertDeviceEqual("/job:moon/cpu:0", w.device)
      self.assertDeviceEqual("/job:moon/cpu:0", w.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  @test_util.run_deprecated_v1
  def testPS2TasksNoMerging(self):
    # [FUNCTION DOC]
    # Scenario: merging disabled. A plain variable goes to ps task 0, but one with an explicit
    # ps job keeps exactly that device without a task. Add op on the worker job.
    with ops.device(
        device_setter.replica_device_setter(
            cluster=self._cluster_spec, merge_devices=False)):
      v = variables.Variable([1, 2])
      with ops.device("/job:ps"):  # Won't assign task when merge_devices=False.
        w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps", w.device)
      self.assertDeviceEqual("/job:ps", w.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  @test_util.run_deprecated_v1
  def testPS2TasksWithClusterSpecDict(self):
    # [FUNCTION DOC]
    # Scenario: cluster passed as a plain dictionary. Expects round-robin ps placement (tasks 0
    # and 1) and the add op on the worker job.
    with ops.device(
        device_setter.replica_device_setter(cluster=self._cluster_spec.as_dict(
        ))):
      v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  @test_util.run_deprecated_v1
  def testPS2TasksWithClusterDef(self):
    # [FUNCTION DOC]
    # Scenario: cluster passed as a cluster definition message. Expects the same placement as
    # the dictionary case: ps tasks 0 and 1, add op on the worker job.
    with ops.device(
        device_setter.replica_device_setter(
            cluster=self._cluster_spec.as_cluster_def())):
      v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:ps/task:0", v.device)
      self.assertDeviceEqual("/job:ps/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:ps/task:1", w.device)
      self.assertDeviceEqual("/job:ps/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:worker", a.device)

  @test_util.run_deprecated_v1
  def testPS2TasksWithDevice(self):
    # [FUNCTION DOC]
    # Scenario: custom job names, moon as the ps job and sun as the worker job. Expects the
    # variables on moon tasks 0 and 1 and the add op on the sun job.
    cluster_spec = server_lib.ClusterSpec({
        "sun": ["sun0:2222", "sun1:2222", "sun2:2222"],
        "moon": ["moon0:2222", "moon1:2222"]
    })

    with ops.device(
        device_setter.replica_device_setter(
            ps_device="/job:moon",
            worker_device="/job:sun",
            cluster=cluster_spec.as_cluster_def())):
      v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:moon/task:0", v.device)
      self.assertDeviceEqual("/job:moon/task:0", v.initializer.device)
      self.assertDeviceEqual("/job:moon/task:1", w.device)
      self.assertDeviceEqual("/job:moon/task:1", w.initializer.device)
      self.assertDeviceEqual("/job:sun", a.device)

  @test_util.run_deprecated_v1
  def testPS2TasksWithCPUConstraint(self):
    # [FUNCTION DOC]
    # Scenario: ps device includes a CPU constraint on the moon job. Expects variables on moon
    # tasks 0 and 1 with the CPU, and the add op on the sun job.
    cluster_spec = server_lib.ClusterSpec({
        "sun": ["sun0:2222", "sun1:2222", "sun2:2222"],
        "moon": ["moon0:2222", "moon1:2222"]
    })

    with ops.device(
        device_setter.replica_device_setter(
            ps_device="/job:moon/cpu:0",
            worker_device="/job:sun",
            cluster=cluster_spec.as_cluster_def())):
      v = variables.Variable([1, 2])
      w = variables.Variable([2, 1])
      a = v + w
      self.assertDeviceEqual("/job:moon/task:0/cpu:0", v.device)
      self.assertDeviceEqual("/job:moon/task:0/cpu:0", v.initializer.device)
      self.assertDeviceEqual("/job:moon/task:1/cpu:0", w.device)
      self.assertDeviceEqual("/job:moon/task:1/cpu:0", w.initializer.device)
      self.assertDeviceEqual("/job:sun", a.device)


if __name__ == "__main__":
  test.main()
