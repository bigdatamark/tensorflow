# [MODULE DOC]
# Tests for the sharding_util module: hashing of sharding callbacks and the shard validation
# function. Each validation test builds a small module of variables, wraps its tensors as
# shardable tensors per task, runs a purpose-built callback that violates (or respects) one rule,
# and asserts the validation function passes or raises the expected runtime error. Importing has
# no side effects; running as main starts the test runner.

# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for checkpoint sharding structures and utilities."""


from typing import Sequence

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint.sharding import sharding_policies
from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import server_lib
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util


class ShardingUtilTest(test.TestCase):
  # [CLASS DOC]
  # Test case for callback hashing and shard validation. Holds a shared helper that turns a
  # trackable root into shardable tensors grouped per task.

  def _get_shardable_tensors_by_task(self, root):
    # [FUNCTION DOC]
    # HELPER: builds the validation inputs shared by the tests in this class.
    # Gathers serialized tensors of the object graph under root, wraps each tensor or slice in a
    # shardable-tensor record (slice spec whitespace-stripped), and groups the records by device
    # task (cpu index forced to 0). Returns the collection of per-task lists.
    serialized_tensors, _, _, _ = (
        checkpoint.TrackableSaver(graph_view.ObjectGraphView(root))
        ._gather_serialized_tensors(None))

    shardable_tensors_by_task = {}
    for obj, tensor_dict in serialized_tensors.items():
      for checkpoint_key, tensor_slice_dict in tensor_dict.items():
        if not isinstance(tensor_slice_dict, dict):
          # Make sure that maybe_tensor is structured as {slice_spec -> tensor}.
          tensor_slice_dict = {"": tensor_slice_dict}
        for slice_spec, tensor_save_spec in tensor_slice_dict.items():
          if not isinstance(tensor_save_spec, saveable_object.SaveSpec):
            tensor_save_spec = saveable_object.SaveSpec(
                tensor=tensor_save_spec,
                slice_spec=slice_spec,
                name=checkpoint_key,
                dtype=tensor_save_spec.dtype,
                device=tensor_save_spec.device)
          save_spec_tensor = tensor_save_spec.tensor
          device = (device_lib.DeviceSpec.from_string(tensor_save_spec.device)
                    if isinstance(tensor_save_spec.device, str)
                    else tensor_save_spec.device)
          task = device_lib.DeviceSpec.from_string(
              saveable_object_util.set_cpu0(device.to_string()))
          shardable_tensors_by_task.setdefault(task, []).append(
              sharding_util.ShardableTensor(
                  _tensor_save_spec=tensor_save_spec,
                  tensor=save_spec_tensor,
                  dtype=tensor_save_spec.dtype,
                  device=device,
                  name=tensor_save_spec.name,
                  shape=save_spec_tensor.shape,
                  slice_spec=slice_spec.strip(),
                  checkpoint_key=checkpoint_key,
                  trackable=obj))
    return shardable_tensors_by_task.values()

  def test_hash_ShardingCallback(self):
    # [FUNCTION DOC]
    # Checks callback hashing: two instances of a callback with no state hash equally, and two
    # instances of a callback holding a value hash equally for equal values and differently for
    # different values.
    class BlankCallback(sharding_util.ShardingCallback):
      # [CLASS DOC]
      # Test helper: stateless callback with an empty description whose call does nothing.
      @property
      def description(self):
        # [FUNCTION DOC]
        # Returns an empty description text.
        return ""

      def __call__(
          self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
      ) -> Sequence[sharding_util.Shard]:
        # [FUNCTION DOC]
        # Does nothing and returns null; never actually invoked by the test.
        pass

    self.assertEqual(hash(BlankCallback()), hash(BlankCallback()))

    class ValueCallback(sharding_util.ShardingCallback):
      # [CLASS DOC]
      # Test helper: callback holding one stored value, used to show that the hash depends on
      # instance attributes.
      def __init__(self, val):
        # [FUNCTION DOC]
        # Stores val (any hashable value) on the instance.
        self.val = val

      @property
      def description(self):
        # [FUNCTION DOC]
        # Returns the fixed description text for the value callback.
        return "value callback"

      def __call__(
          self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
      ) -> Sequence[sharding_util.Shard]:
        # [FUNCTION DOC]
        # Does nothing and returns null; never actually invoked by the test.
        pass

    self.assertEqual(hash(ValueCallback(1)), hash(ValueCallback(1)))
    self.assertNotEqual(hash(ValueCallback(1)), hash(ValueCallback(2)))

  def test_validate_shards_correct(self):
    # [FUNCTION DOC]
    # Places three variables on three cpu devices, shards them with the by-task policy, runs the
    # validation function (expects no error), then asserts a single shard holds the three variable
    # keys plus the object graph key in that order, with values equal to the variables.
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.0, name="v0")
    with ops.device("cpu:1"):
      v1 = resource_variable_ops.ResourceVariable(1.0, name="v1")
    with ops.device("cpu:2"):
      v2 = resource_variable_ops.ResourceVariable(2.0, name="v2")
    root.v0 = v0
    root.v1 = v1
    root.v2 = v2

    shardable_tensors = self._get_shardable_tensors_by_task(root)
    shardable_tensors_flat = []
    for tensors in shardable_tensors:
      shardable_tensors_flat.extend(tensors)

    sharding_callback = sharding_policies.ShardByTaskPolicy()
    shards = []
    for tensors in shardable_tensors:
      shards.extend(sharding_callback(tensors))

    sharding_util.validate_shards(
        shards, shardable_tensors_flat, sharding_callback.description)

    self.assertEqual(
        [list(shard.keys()) for shard in shards],
        [[
            "v0/.ATTRIBUTES/VARIABLE_VALUE",
            "v1/.ATTRIBUTES/VARIABLE_VALUE",
            "v2/.ATTRIBUTES/VARIABLE_VALUE",
            "_CHECKPOINTABLE_OBJECT_GRAPH"
        ]])

    self.assertEqual(
        shards[0]["v0/.ATTRIBUTES/VARIABLE_VALUE"][""].numpy(),
        v0.numpy())
    self.assertEqual(
        shards[0]["v1/.ATTRIBUTES/VARIABLE_VALUE"][""].numpy(),
        v1.numpy())
    self.assertEqual(
        shards[0]["v2/.ATTRIBUTES/VARIABLE_VALUE"][""].numpy(),
        v2.numpy())

  def test_validate_shards_duplicate_tensor(self):
    # [FUNCTION DOC]
    # Uses a callback that emits the first tensor twice in two shards; asserts validation raises a
    # runtime error about multiple tensors with the same checkpoint key and slice spec.
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.0, name="v0")
    with ops.device("cpu:1"):
      v1 = resource_variable_ops.ResourceVariable(1.0, name="v1")
    root.v0 = v0
    root.v1 = v1

    class DuplicateTensorCallback(sharding_util.ShardingCallback):
      # [CLASS DOC]
      # Test helper: callback that returns the first input tensor twice, in two separate shards.
      @property
      def description(self):
        # [FUNCTION DOC]
        # Returns the fixed description text for this callback.
        return "duplicate tensor callback"

      def __call__(
          self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
      ) -> Sequence[sharding_util.Shard]:
        # [FUNCTION DOC]
        # Takes the first shardable tensor and returns two shards each containing it.
        tensor = shardable_tensors[0].tensor
        checkpoint_key = shardable_tensors[0].checkpoint_key
        slice_spec = shardable_tensors[0].slice_spec
        shards = [
            {checkpoint_key: {slice_spec: tensor}},
            {checkpoint_key: {slice_spec: tensor}}
        ]
        return shards

    shardable_tensors = self._get_shardable_tensors_by_task(root)
    shardable_tensors_flat = []
    for tensors in shardable_tensors:
      shardable_tensors_flat.extend(tensors)

    sharding_callback = DuplicateTensorCallback()
    shards = []
    for tensors in shardable_tensors:
      shards.extend(sharding_callback(tensors))

    with self.assertRaisesRegex(RuntimeError,
                                "multiple tensors with the same checkpoint "
                                "key and slice spec were found"):
      sharding_util.validate_shards(
          shards, shardable_tensors_flat, sharding_callback.description)

  def test_validate_shards_added_tensor(self):
    # [FUNCTION DOC]
    # Uses a callback that returns a shard with a made-up checkpoint key; asserts validation raises
    # a runtime error saying a tensor not originally in the object graph was found.
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.0, name="v0")
    root.v0 = v0

    class AddedTensorCallback(sharding_util.ShardingCallback):
      # [CLASS DOC]
      # Test helper: callback that ignores its input and returns one shard with an unknown key.
      @property
      def description(self):
        # [FUNCTION DOC]
        # Returns the fixed description text for this callback.
        return "added tensor callback"

      def __call__(
          self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
      ) -> Sequence[sharding_util.Shard]:
        # [FUNCTION DOC]
        # Ignores the input and returns a single shard mapping an invented key and empty slice
        # spec to a freshly constructed, empty tensor object.
        checkpoint_key = "ADDED_TENSOR_ABC123"
        slice_spec = ""
        tensor = tensor_lib.Tensor()
        return [{checkpoint_key: {slice_spec: tensor}}]

    shardable_tensors = self._get_shardable_tensors_by_task(root)
    shardable_tensors_flat = []
    for tensors in shardable_tensors:
      shardable_tensors_flat.extend(tensors)

    sharding_callback = AddedTensorCallback()
    shards = []
    for tensors in shardable_tensors:
      shards.extend(sharding_callback(tensors))

    with self.assertRaisesRegex(RuntimeError,
                                "a tensor not originally in the object graph"):
      sharding_util.validate_shards(
          shards, shardable_tensors_flat, sharding_callback.description)

  def test_validate_shards_shape_change(self):
    # [FUNCTION DOC]
    # Uses a callback that transposes the 1x2 variable's tensor; asserts validation raises a
    # runtime error about a tensor with an altered shape.
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable([[0.0, 1.0]], name="v0")
    root.v0 = v0

    class ShapeChangeCallback(sharding_util.ShardingCallback):
      # [CLASS DOC]
      # Test helper: callback that puts each tensor in its own shard, transposing the one for the
      # first variable so its shape changes.
      @property
      def description(self):
        # [FUNCTION DOC]
        # Returns the fixed description text for this callback.
        return "shape change callback"

      def __call__(
          self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
      ) -> Sequence[sharding_util.Shard]:
        # [FUNCTION DOC]
        # Returns one shard per input tensor; the tensor of the first variable is transposed.
        shards = []
        for shardable_tensor in shardable_tensors:
          tensor = shardable_tensor.tensor
          checkpoint_key = shardable_tensor.checkpoint_key
          slice_spec = shardable_tensor.slice_spec
          if checkpoint_key == "v0/.ATTRIBUTES/VARIABLE_VALUE":
            tensor = array_ops.transpose(tensor)
          shards.append({checkpoint_key: {slice_spec: tensor}})
        return shards

    shardable_tensors = self._get_shardable_tensors_by_task(root)
    shardable_tensors_flat = []
    for tensors in shardable_tensors:
      shardable_tensors_flat.extend(tensors)

    sharding_callback = ShapeChangeCallback()
    shards = []
    for tensors in shardable_tensors:
      shards.extend(sharding_callback(tensors))

    with self.assertRaisesRegex(RuntimeError,
                                "a tensor was found with an altered shape"):
      sharding_util.validate_shards(
          shards, shardable_tensors_flat, sharding_callback.description)

  def test_validate_shards_dtype_change(self):
    # [FUNCTION DOC]
    # Uses a callback that casts the variable's tensor to 32-bit integer; asserts validation
    # raises a runtime error about a tensor with an altered dtype.
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.0, name="v0")
    root.v0 = v0

    class DtypeChangeCallback(sharding_util.ShardingCallback):
      # [CLASS DOC]
      # Test helper: callback that puts each tensor in its own shard, casting the one for the
      # first variable to a different dtype.
      @property
      def description(self):
        # [FUNCTION DOC]
        # Returns the fixed description text for this callback.
        return "dtype change callback"

      def __call__(
          self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
      ) -> Sequence[sharding_util.Shard]:
        # [FUNCTION DOC]
        # Returns one shard per input tensor; the first variable's tensor is cast to int32.
        shards = []
        for shardable_tensor in shardable_tensors:
          tensor = shardable_tensor.tensor
          checkpoint_key = shardable_tensor.checkpoint_key
          slice_spec = shardable_tensor.slice_spec
          if checkpoint_key == "v0/.ATTRIBUTES/VARIABLE_VALUE":
            tensor = math_ops.cast(tensor, dtype=dtypes.int32)
          shards.append({checkpoint_key: {slice_spec: tensor}})
        return shards

    shardable_tensors = self._get_shardable_tensors_by_task(root)
    shardable_tensors_flat = []
    for tensors in shardable_tensors:
      shardable_tensors_flat.extend(tensors)

    sharding_callback = DtypeChangeCallback()
    shards = []
    for tensors in shardable_tensors:
      shards.extend(sharding_callback(tensors))

    with self.assertRaisesRegex(RuntimeError,
                                "a tensor was found with an altered dtype"):
      sharding_util.validate_shards(
          shards, shardable_tensors_flat, sharding_callback.description)

  def test_validate_shards_task_change(self):
    # [FUNCTION DOC]
    # Starts two local servers as a worker cluster with a variable on each task. Uses a callback
    # that copies the first variable's tensor onto task 1; asserts validation raises a runtime
    # error about a tensor with an altered task.
    servers = [server_lib.Server.create_local_server() for _ in range(2)]
    cluster_spec = server_lib.ClusterSpec({
        "worker": [s.target[len("grpc://"):] for s in servers]})
    remote.connect_to_cluster(cluster_spec)

    root = module.Module()
    with ops.device("/job:worker/task:0/cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.0, name="v0")
    with ops.device("/job:worker/task:1/cpu:0"):
      v1 = resource_variable_ops.ResourceVariable(0.0, name="v1")
    root.v0 = v0
    root.v1 = v1

    class TaskChangeCallback(sharding_util.ShardingCallback):
      # [CLASS DOC]
      # Test helper: callback that puts each tensor in its own shard, moving the first variable's
      # tensor to a different worker task via an identity operation.
      @property
      def description(self):
        # [FUNCTION DOC]
        # Returns the fixed description text for this callback.
        return "task change callback"

      def __call__(
          self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
      ) -> Sequence[sharding_util.Shard]:
        # [FUNCTION DOC]
        # Returns one shard per input tensor; the first variable's tensor is recomputed on task 1.
        shards = []
        for shardable_tensor in shardable_tensors:
          tensor = shardable_tensor.tensor
          checkpoint_key = shardable_tensor.checkpoint_key
          slice_spec = shardable_tensor.slice_spec
          if checkpoint_key == "v0/.ATTRIBUTES/VARIABLE_VALUE":
            with ops.device("/job:worker/task:1/cpu:0"):
              tensor = array_ops.identity(tensor)
          shards.append({checkpoint_key: {slice_spec: tensor}})
        return shards

    shardable_tensors = self._get_shardable_tensors_by_task(root)
    shardable_tensors_flat = []
    for tensors in shardable_tensors:
      shardable_tensors_flat.extend(tensors)

    sharding_callback = TaskChangeCallback()
    shards = []
    for tensors in shardable_tensors:
      shards.extend(sharding_callback(tensors))

    with self.assertRaisesRegex(RuntimeError,
                                "a tensor was found with an altered task"):
      sharding_util.validate_shards(
          shards, shardable_tensors_flat, sharding_callback.description)

  def test_validate_shards_different_tasks(self):
    # [FUNCTION DOC]
    # Starts two local servers with a variable on each task. Uses a callback that puts all tensors
    # (from both tasks) in one shard; asserts validation raises a runtime error saying tensors with
    # different tasks were found.
    servers = [server_lib.Server.create_local_server() for _ in range(2)]
    cluster_spec = server_lib.ClusterSpec({
        "worker": [s.target[len("grpc://"):] for s in servers]})
    remote.connect_to_cluster(cluster_spec)

    root = module.Module()
    with ops.device("/job:worker/task:0/cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.0, name="v0")
    with ops.device("/job:worker/task:1/cpu:0"):
      v1 = resource_variable_ops.ResourceVariable(0.0, name="v1")
    root.v0 = v0
    root.v1 = v1

    class DifferentTasksCallback(sharding_util.ShardingCallback):
      # [CLASS DOC]
      # Test helper: callback that merges every input tensor into a single shard.
      @property
      def description(self):
        # [FUNCTION DOC]
        # Returns the fixed description text for this callback.
        return "different tasks callback"

      def __call__(
          self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
      ) -> Sequence[sharding_util.Shard]:
        # [FUNCTION DOC]
        # Returns a list with one shard containing all input tensors keyed by checkpoint key and
        # slice spec.
        shard = {}
        for shardable_tensor in shardable_tensors:
          tensor = shardable_tensor.tensor
          checkpoint_key = shardable_tensor.checkpoint_key
          slice_spec = shardable_tensor.slice_spec
          shard.setdefault(checkpoint_key, {})[slice_spec] = tensor
        return [shard]

    shardable_tensors = self._get_shardable_tensors_by_task(root)
    shardable_tensors_flat = []
    for tensors in shardable_tensors:
      shardable_tensors_flat.extend(tensors)

    sharding_callback = DifferentTasksCallback()
    shards = sharding_callback(shardable_tensors_flat)

    with self.assertRaisesRegex(RuntimeError,
                                "tensors with different tasks were found"):
      sharding_util.validate_shards(
          shards, shardable_tensors_flat, sharding_callback.description)

  def test_validate_shards_tensor_removal(self):
    # [FUNCTION DOC]
    # Uses a callback that returns no shards; asserts validation raises a runtime error saying
    # tensors in the object graph were not found in the shards.
    root = module.Module()
    with ops.device("cpu:0"):
      v0 = resource_variable_ops.ResourceVariable(0.0, name="v0")
    root.v0 = v0

    class TensorRemovalCallback(sharding_util.ShardingCallback):
      # [CLASS DOC]
      # Test helper: callback that drops every tensor by returning no shards.
      @property
      def description(self):
        # [FUNCTION DOC]
        # Returns the fixed description text for this callback.
        return "tensor removal callback"

      def __call__(
          self, shardable_tensors: Sequence[sharding_util.ShardableTensor]
      ) -> Sequence[sharding_util.Shard]:
        # [FUNCTION DOC]
        # Ignores the input and returns an empty list of shards.
        return []

    shardable_tensors = self._get_shardable_tensors_by_task(root)
    shardable_tensors_flat = []
    for tensors in shardable_tensors:
      shardable_tensors_flat.extend(tensors)

    sharding_callback = TensorRemovalCallback()
    shards = []
    for tensors in shardable_tensors:
      shards.extend(sharding_callback(tensors))

    with self.assertRaisesRegex(RuntimeError,
                                "tensors in the object graph were not found"):
      sharding_util.validate_shards(
          shards, shardable_tensors_flat, sharding_callback.description)


if __name__ == "__main__":
  test.main()
