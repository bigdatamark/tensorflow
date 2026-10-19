# [MODULE DOC]
# Tests for MultiDeviceSaver in functional_saver: saving/restoring resource variables, using
# a local io device, converting to a saver definition, sharding by remote task, multi-device
# saves on virtual CPUs and single-shard save/restore helpers. Runs under the TensorFlow test
# runner, enabling eager execution when run as a script.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Tests for the functional saver."""

import os
import time

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.eager import context
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import server_lib
from tensorflow.python.training.saving import saveable_object_util

LOCALHOST = "/job:localhost/replica:0/task:0/device:CPU:0"


class SaverTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering MultiDeviceSaver. Shared fixture: three virtual CPU devices and checkpoint
  # options that force the io device to the local host.

  def setUp(self):
    # [FUNCTION DOC]
    # Splits the first physical CPU into three logical CPUs and creates checkpoint options whose
    # io device is the local host, stored as local_options.
    super(SaverTest, self).setUp()
    cpus = config.list_physical_devices("CPU")
    # Set 3 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])
    self.local_options = checkpoint_options.CheckpointOptions(
        experimental_io_device=LOCALHOST)

  def _get_tensors_by_task(self, root):
    # [FUNCTION DOC]
    # HELPER: gathers serialized tensors of a trackable root for the sharding test.
    # Builds a trackable saver over the root's object graph, gathers its serialized tensors, and
    # returns a mapping from device task (CPU 0 variant of the tensor's device) to a mapping of
    # checkpoint key to slice spec to tensor.
    serialized_tensors, _, _, _ = (
        checkpoint.TrackableSaver(graph_view.ObjectGraphView(root))
        ._gather_serialized_tensors(None))

    tensors_by_task = {}
    for tensor_dict in serialized_tensors.values():
      for checkpoint_key, maybe_tensor in tensor_dict.items():
        if not isinstance(maybe_tensor, dict):
          maybe_tensor = {"": maybe_tensor}
        for slice_spec, tensor in maybe_tensor.items():
          tensor_task = saveable_object_util.set_cpu0(tensor.device)
          (tensors_by_task
           .setdefault(tensor_task, {})
           .setdefault(checkpoint_key, {})[slice_spec]) = tensor
    return tensors_by_task

  @test_util.run_in_graph_and_eager_modes
  def test_resource_variable(self):
    # [FUNCTION DOC]
    # Saves a resource variable (value 2) with a saver, expects two files at the prefix (index and
    # data), changes the variable to 1, restores and expects 2 again. Then a second saver over
    # another variable (value 3) with the same name restores from the same prefix and must see 2.
    v1 = resource_variable_ops.ResourceVariable(2.)
    self.evaluate(v1.initializer)
    saver = functional_saver.MultiDeviceSaver.from_saveables(
        saveable_object_util.saveable_objects_for_op(v1, "x"))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(saver.save(constant_op.constant(prefix)))
    self.assertEqual(2, len(gfile.Glob(prefix + "*")))
    self.evaluate(v1.assign(1.))
    self.evaluate(saver.restore(prefix))
    self.assertEqual(2., self.evaluate(v1))

    v2 = resource_variable_ops.ResourceVariable(3.)
    self.evaluate(v2.initializer)
    second_saver = functional_saver.MultiDeviceSaver.from_saveables(
        saveable_object_util.saveable_objects_for_op(v2, "x"))
    self.evaluate(second_saver.restore(prefix))
    self.assertEqual(2., self.evaluate(v2))

  @test_util.run_in_graph_and_eager_modes
  def test_resource_variable_use_localhost(self):
    # [FUNCTION DOC]
    # Same scenario as the basic resource variable test but passing the local-host io options to
    # save and restore. Additionally, in graph mode asserts that every save and restore op was
    # placed on the local host device.
    v1 = resource_variable_ops.ResourceVariable(2.)
    self.evaluate(v1.initializer)
    saver = functional_saver.MultiDeviceSaver.from_saveables(
        saveable_object_util.saveable_objects_for_op(v1, "x"))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(saver.save(constant_op.constant(prefix), self.local_options))
    self.assertEqual(2, len(gfile.Glob(prefix + "*")))
    self.evaluate(v1.assign(1.))
    self.evaluate(saver.restore(prefix, self.local_options))
    self.assertEqual(2., self.evaluate(v1))

    v2 = resource_variable_ops.ResourceVariable(3.)
    self.evaluate(v2.initializer)
    second_saver = functional_saver.MultiDeviceSaver.from_saveables(
        saveable_object_util.saveable_objects_for_op(v2, "x"))
    self.evaluate(second_saver.restore(prefix, self.local_options))
    self.assertEqual(2., self.evaluate(v2))

    # In graph mode, verify that the save and restore ops were set to run on
    # localhost.
    if not context.executing_eagerly():
      for op in ops.get_default_graph().get_operations():
        if op.type in ("SaveV2", "RestoreV2"):
          self.assertEqual(LOCALHOST, op.device)

  def test_to_proto(self):
    # [FUNCTION DOC]
    # Builds a saver for a variable, calls to_proto inside a wrapped function, then prunes the
    # wrapped graph into save and restore callables using the tensor/op names in the proto. Saves,
    # alters the variable, restores from the returned path and expects the original value 2; a
    # second saver restoring from that path also sees 2.
    v1 = resource_variable_ops.ResourceVariable(2.)
    saver = functional_saver.MultiDeviceSaver.from_saveables(
        saveable_object_util.saveable_objects_for_op(v1, "x"))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")

    proto_accumulator = []
    wrapped = wrap_function.wrap_function(
        lambda: proto_accumulator.append(saver.to_proto()), signature=())
    self.assertEqual(1, len(proto_accumulator))
    proto = proto_accumulator[0]
    save = wrapped.prune(
        feeds=wrapped.graph.get_tensor_by_name(proto.filename_tensor_name),
        fetches=wrapped.graph.get_tensor_by_name(proto.save_tensor_name))
    restore = wrapped.prune(
        feeds=wrapped.graph.get_tensor_by_name(proto.filename_tensor_name),
        fetches=wrapped.graph.get_operation_by_name(proto.restore_op_name))
    save_path = save(constant_op.constant(prefix))
    v1.assign(1.)
    restore(constant_op.constant(save_path))
    self.assertEqual(2., self.evaluate(v1))

    v2 = resource_variable_ops.ResourceVariable(3.)
    second_saver = functional_saver.MultiDeviceSaver.from_saveables(
        saveable_object_util.saveable_objects_for_op(v2, "x"))
    second_saver.restore(save_path)
    self.assertEqual(2., self.evaluate(v2))

  @test_util.disable_tfrt("b/171765113: server is not supported in TFRT yet.")
  def test_checkpoint_is_sharded_by_task(self):
    # [FUNCTION DOC]
    # Starts three local servers, connects to them as a cluster, places one variable per worker
    # task, and saves them with one saver. Expects four files (three data shards plus index),
    # then overwrites the variables, restores, and expects the original values 0, 1, 2.
    servers = [server_lib.Server.create_local_server() for _ in range(3)]
    cluster_spec = server_lib.ClusterSpec({
        "worker": [s.target[len("grpc://"):] for s in servers]})
    remote.connect_to_cluster(cluster_spec)
    with ops.device("/job:worker/task:0/cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.)
    with ops.device("/job:worker/task:1/cpu:0"):
      v1 = resource_variable_ops.ResourceVariable(1.)
    with ops.device("/job:worker/task:2/cpu:0"):
      v2 = resource_variable_ops.ResourceVariable(2.)

    self.evaluate([v0.initializer, v1.initializer, v2.initializer])
    saver = functional_saver.MultiDeviceSaver.from_saveables(
        list(saveable_object_util.saveable_objects_for_op(v0, "v0")) +
        list(saveable_object_util.saveable_objects_for_op(v1, "v1")) +
        list(saveable_object_util.saveable_objects_for_op(v2, "v2")))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(saver.save(constant_op.constant(prefix)))
    self.assertEqual(4, len(gfile.Glob(prefix + "*")))
    self.evaluate(v0.assign(-1.))
    self.evaluate(v1.assign(-1.))
    self.evaluate(v2.assign(-1.))
    self.evaluate(saver.restore(constant_op.constant(prefix)))
    self.assertEqual(0., self.evaluate(v0))
    self.assertEqual(1., self.evaluate(v1))
    self.assertEqual(2., self.evaluate(v2))

  @test_util.run_in_graph_and_eager_modes
  def test_checkpoint_multi_device_using_localhost(self):
    # [FUNCTION DOC]
    # Places three variables on three virtual CPUs, saves with the local-host io device and
    # expects two files only, then overwrites and restores them and expects values 0, 1, 2. In
    # graph mode asserts save, restore and merge ops all ran on the local host device.
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.)
    with ops.device("cpu:1"):
      v1 = resource_variable_ops.ResourceVariable(1.)
    with ops.device("cpu:2"):
      v2 = resource_variable_ops.ResourceVariable(2.)

    self.evaluate([v0.initializer, v1.initializer, v2.initializer])
    saver = functional_saver.MultiDeviceSaver.from_saveables(
        list(saveable_object_util.saveable_objects_for_op(v0, "v0")) +
        list(saveable_object_util.saveable_objects_for_op(v1, "v1")) +
        list(saveable_object_util.saveable_objects_for_op(v2, "v2")))
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    self.evaluate(saver.save(constant_op.constant(prefix), self.local_options))
    self.assertEqual(2, len(gfile.Glob(prefix + "*")))
    self.evaluate(v0.assign(-1.))
    self.evaluate(v1.assign(-1.))
    self.evaluate(v2.assign(-1.))
    self.evaluate(
        saver.restore(constant_op.constant(prefix), self.local_options))
    self.assertEqual(0., self.evaluate(v0))
    self.assertEqual(1., self.evaluate(v1))
    self.assertEqual(2., self.evaluate(v2))

    # In graph mode, verify that the save and restore ops were set to run on
    # localhost.
    if not context.executing_eagerly():
      for op in ops.get_default_graph().get_operations():
        if op.type in ("SaveV2", "RestoreV2", "MergeV2Checkpoints"):
          self.assertEqual(LOCALHOST, op.device)

  def test_single_task_save_singlehost_multidevice(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: calls functional_saver single-task save and restore helpers, but the module now
    # defines single-shard helpers with different names, so this test would fail with an
    # attribute error if run.
    # Scenario: a module with three variables on three virtual CPUs; gathers tensors by task,
    # and for each device and shard writes the shard to a sharded filename, waits up to five
    # seconds for files to appear, expects a non-empty directory, restores the shard on the
    # device and compares the restored value with the variable's numpy value.
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.)
    with ops.device("cpu:1"):
      v1 = resource_variable_ops.ResourceVariable(1.)
    with ops.device("cpu:2"):
      v2 = resource_variable_ops.ResourceVariable(2.)
    root.v0 = v0
    root.v1 = v1
    root.v2 = v2

    tensors_by_task = self._get_tensors_by_task(root)
    var_names = [
        "v0/.ATTRIBUTES/VARIABLE_VALUE",
        "v1/.ATTRIBUTES/VARIABLE_VALUE",
        "v2/.ATTRIBUTES/VARIABLE_VALUE"
    ]
    vars_numpy = [v0.numpy(), v1.numpy(), v2.numpy()]
    tmp_dir = self.get_temp_dir()

    for device in ["cpu:0", "cpu:1", "cpu:2"]:
      for shard, (_, tensor_slice_dict) in enumerate(
          sorted(tensors_by_task.items())[1:]):
        with ops.device(device):
          shard_prefix = gen_io_ops.sharded_filename(
              os.path.join(tmp_dir, str(shard)), shard, 3)
          functional_saver._single_task_save(
              shard_prefix, tensor_slice_dict)

        start_time = time.time()
        max_save_time = start_time + 5  # seconds
        while not (gfile.ListDirectory(tmp_dir) or time.time() > max_save_time):
          pass  # eager execution is lovely
        self.assertNotEmpty(gfile.ListDirectory(tmp_dir))

        with ops.device(device):
          restored_dict = functional_saver._single_task_restore(
              shard_prefix, tensor_slice_dict)
          self.evaluate(restored_dict)
          self.assertEqual(
              restored_dict[var_names[shard]][""].numpy(),
              vars_numpy[shard])


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
