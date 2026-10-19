# [MODULE DOC]
# Unit tests for the DTensor utilities used by the experimental DTensor-based distribution
# strategies. Covers two components: the distributed-value wrapper that can only be built from
# a DTensor, and the replica context that reports a fixed replica layout and rejects
# unsupported collective operations. Uses the DTensor test base class to configure a
# two-device mesh on TPU, GPU and CPU. No import-time side effects besides the test main entry.
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
# ==============================================================================
"""Test for DTensor related utilities in tf.distribute."""

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.distribute.experimental import dtensor_util
from tensorflow.python.distribute.experimental import mirrored_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op


class DTensorDistributedValueTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests for the DTensor-backed distributed value wrapper. Fixtures: a one-dimensional mesh
  # named batch with two devices, and a DTensor of shape two packed from two single-element
  # tensors and sharded along the batch dimension. Stored on the test instance.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup. Calls the parent setUp, builds a mesh with a single batch dimension of
    # size two for each of TPU, GPU and CPU, registers it as the test mesh, creates two
    # one-element float tensors (one and two), builds a batch-sharded layout of rank one and
    # packs the tensors into a DTensor. Sets the mesh, batch layout and dtensor attributes.
    super().setUp()
    global_ids = test_util.create_device_ids_array((2,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout.Mesh(['batch'], global_ids, local_ids,
                            test_util.create_device_list((2,), device))
        for device in ['TPU', 'GPU', 'CPU']
    }
    self.mesh = self.configTestMesh(mesh_dict)

    tensor_1 = constant_op.constant([1.0])
    tensor_2 = constant_op.constant([2.0])
    self.batch_layout = layout.Layout.batch_sharded(
        self.mesh, batch_dim='batch', rank=1)
    self.dtensor = d_api.pack([tensor_1, tensor_2], self.batch_layout)

  @parameterized.named_parameters([
      ('py_floats', [1.0, 2.0]),
      ('np_floats', np.array([1.0, 2.0])),
      ('tf_const', lambda: constant_op.constant([1.0, 2.0])),
      ('distribute_value', values_lib.PerReplica([1.0, 2.0])),
  ])
  def test_input_validation(self, inputs):
    # [FUNCTION DOC]
    # Parameterized over a python float list, a numpy array, a lazily built tensor and a
    # per-replica distributed value. If the input is a callable it is called first to produce
    # the real input (so tensors are created inside the test). Constructing the DTensor
    # distributed value from any of these non-DTensor inputs must raise a value error whose
    # message says it can only be built with DTensor.
    if callable(inputs):
      inputs = inputs()

    with self.assertRaisesRegex(ValueError, 'can only be built with DTensor'):
      dtensor_util.DTensorDistributedValue(inputs)

  def test_unpack(self):
    # [FUNCTION DOC]
    # Wraps the fixture DTensor in a distributed value and checks that the underlying DTensor is
    # returned by identity, and that the values property yields two per-replica tensors close to
    # the original one-element tensors (one and two respectively).
    v = dtensor_util.DTensorDistributedValue(self.dtensor)

    self.assertIs(self.dtensor, v.get_dtensor())

    per_replica_result = v.values
    self.assertLen(per_replica_result, 2)
    self.assertAllClose(per_replica_result[0], constant_op.constant([1.0]))
    self.assertAllClose(per_replica_result[1], constant_op.constant([2.0]))

  def test_graph_behavior(self):
    # [FUNCTION DOC]
    # Inside a compiled function, wraps the fixture DTensor in a distributed value and returns
    # it. Asserts that when the value crosses the function boundary it is unwrapped into a plain
    # DTensor, whose contents equal the two-element tensor and which keeps the batch layout.

    @def_function.function
    def run_fn(input_dtensor):
      # [FUNCTION DOC]
      # Nested function compiled by the test. Takes a DTensor and returns a DTensor distributed
      # value wrapping it.
      return dtensor_util.DTensorDistributedValue(input_dtensor)

    result = run_fn(self.dtensor)
    # When it cross the boundary of tf.function, it will be unwrapped and
    # return a dtensor instance directly.
    self.assertTrue(d_api.is_dtensor(result))
    self.assertDTensorEqual(constant_op.constant([1.0, 2.0]),
                            self.batch_layout, result)


class DTensorReplicaContextTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests for the DTensor replica context used by the experimental mirrored strategy. The
  # fixture is a one-dimensional two-device batch mesh stored on the test instance.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup. Calls the parent setUp, then builds a two-device mesh with a single
    # batch dimension for each of TPU, GPU and CPU and registers it as the test mesh, stored in
    # the mesh attribute.
    super().setUp()
    global_ids = test_util.create_device_ids_array((2,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout.Mesh(['batch'], global_ids, local_ids,
                            test_util.create_device_list((2,), device))
        for device in ['TPU', 'GPU', 'CPU']
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def test_unsupported_methods(self):
    # [FUNCTION DOC]
    # Creates the experimental mirrored strategy on the fixture mesh and a DTensor replica
    # context from it. Asserts the context reports two replicas in sync and replica id zero,
    # and that merge call, all-reduce (sum) and all-gather each raise a not-implemented error
    # whose message matches the context's unsupported-operation message.
    strategy = mirrored_strategy.MirroredStrategy(mesh=self.mesh)
    replica_context = dtensor_util.DTensorReplicaContext(strategy)

    expected_error = replica_context._UNSUPPORTED_ERROR_MSG

    self.assertEqual(replica_context.num_replicas_in_sync, 2)

    self.assertEqual(replica_context.replica_id_in_sync_group, 0)

    with self.assertRaisesRegex(NotImplementedError, expected_error):
      replica_context.merge_call(None)

    with self.assertRaisesRegex(NotImplementedError, expected_error):
      replica_context.all_reduce(reduce_util.ReduceOp.SUM, None)

    with self.assertRaisesRegex(NotImplementedError, expected_error):
      replica_context.all_gather([], 0)


if __name__ == '__main__':
  test.main()
