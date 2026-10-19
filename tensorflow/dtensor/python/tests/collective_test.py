# [MODULE DOC]
# Tests for DTensor collective operations (all-reduce, all-gather, all-scatter, all-to-all) on
# small meshes. Defines two test classes: one on a two-device mesh checking reductions and
# relayouts for several dtypes, and one that builds custom eight-device meshes to exercise the
# all-reduce combiner optimization and mixed-precision reduction. Many tests are skipped on
# GPU/TPU depending on dtype support or device count. Importing has no side effects beyond
# defining module constants for the mesh dimension names.
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
"""Tests for DTensor collectives."""

import os

from absl.testing import parameterized
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

# pylint: enable=g-direct-tensorflow-import

Layout = layout_lib.Layout

_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_MESH_DIMS = [_MESH_DIM_X, _MESH_DIM_Y]


class CollectiveTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case for collective behavior on a two-device mesh with dimensions x (size 2) and y
  # (size 1). Setup provides a mesh plus three layouts: fully replicated rank 2, first-dimension
  # sharded rank 2, and a replicated scalar layout. Tests compare DTensor results against plain
  # eager results.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Builds a 2x1 mesh for each of CPU, GPU and TPU, lets the base class choose
    # the one for the current device type, and stores it as the mesh attribute. Also stores the
    # replicated rank-2 layout, the batch-sharded rank-2 layout over the x dimension, and the
    # replicated rank-0 (scalar) layout. Returns nothing.
    super(CollectiveTest, self).setUp()

    global_ids = test_util.create_device_ids_array((2, 1))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh(_MESH_DIMS, global_ids, local_ids,
                                test_util.create_device_list((2, 1), device))
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)
    self.fully_replicated_layout_2d = Layout.replicated(self.mesh, rank=2)
    self.first_dimension_sharded_layout_2d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, 2)
    self.scalar_layout = Layout.replicated(self.mesh, rank=0)

  def testReduceOnBfloat16(self):
    # [FUNCTION DOC]
    # Scenario: sum-reduce a 2x4 bfloat16 tensor that is sharded across its first dimension.
    # Skipped on GPU (no bfloat16 collective reduce) and on TPU unless exactly 2 cores exist.
    # Asserts the DTensor result equals the eager sum and has the replicated scalar layout.
    self.skipForDeviceType(['GPU'],
                           'GPUs do not support bfloat16 collective reduce')
    self.skipForDeviceType(['TPU'],
                           'This test only needs to run on 2 cores.',
                           unless_device_count_equals_to=2)

    a = constant_op.constant(
        np.array([[1, 2, 3, 4], [5.0, 6.0, 7.0, 8.0]]), dtype=dtypes.bfloat16)
    expected_result = math_ops.reduce_sum(a)

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    dtensor_result = math_ops.reduce_sum(sharded_a)

    self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

  def testReduceOnInt32(self):
    # [FUNCTION DOC]
    # Scenario: sum-reduce a 2x4 int32 tensor sharded across its first dimension. Asserts the
    # DTensor result equals the eager sum and carries the replicated scalar layout.
    a = constant_op.constant(
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), dtype=dtypes.int32)

    expected_result = math_ops.reduce_sum(a)

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    dtensor_result = math_ops.reduce_sum(sharded_a)

    self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

  def testReduceOnInt8(self):
    # [FUNCTION DOC]
    # Scenario: sum-reduce a 2x4 int8 tensor sharded across its first dimension. Asserts the
    # DTensor result equals the eager sum and carries the replicated scalar layout.
    a = constant_op.constant(
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), dtype=dtypes.int8
    )

    expected_result = math_ops.reduce_sum(a)

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    dtensor_result = math_ops.reduce_sum(sharded_a)

    self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

  def testTwoReducesWithAssign(self):
    # [FUNCTION DOC]
    # Scenario: inside a compiled function, reduce one sharded input, assign that input to a
    # DTensor variable (initialized from the second sharded input), then reduce the second
    # input and return the product of both sums. Validates the control dependency DTensor adds
    # between the reductions and the assignment (comments admit the per-device graph cannot be
    # inspected, so correctness relies on logs). Runs under the default mesh and asserts the
    # result equals the eager product of sums with a replicated scalar layout. Uses float32.
    # FIXME(b/238384852): The purpose of this test is to validate the control
    # dependency added by DTensor.
    # However, as we have no way of testing the per-device graph
    # produced by the DTensor SPMD expansion, we have to use manual logging
    # to verify if the results are correct.
    # For example, this test is used to check cl/459358521.

    # Uses dtypes.float32 to avoid int32 problem with VarHandleOp on GPUs.
    a = constant_op.constant(
        np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), dtype=dtypes.float32)
    b = constant_op.constant(
        np.array([[11, 12, 13, 4], [15, 16, 17, 18]]), dtype=dtypes.float32)

    expected_result = math_ops.reduce_sum(a) * math_ops.reduce_sum(b)

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    sharded_b = api.relayout(b, self.first_dimension_sharded_layout_2d)
    sharded_v = d_variable.DVariable(sharded_b)

    @polymorphic_function.function
    def func(a, b):
      # [FUNCTION DOC]
      # Compiled helper: sums a, assigns a into the captured DTensor variable, sums b, and
      # returns the product of the two sums. Side effect: overwrites the variable's value.
      a1 = math_ops.reduce_sum(a, name='reducea')
      sharded_v.assign(a)
      b1 = math_ops.reduce_sum(b, name='reduceb')
      return a1 * b1

    with api.default_mesh(self.mesh):
      dtensor_result = func(sharded_a, sharded_b)
    self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_all', math_ops.reduce_all),
      ('_any', math_ops.reduce_any),
  )
  def testReduceOnBool(self, reduction):
    # [FUNCTION DOC]
    # Parameterized test (reduce-all and reduce-any). The reduction argument is the reduction
    # function to apply. Reduces a 2x4 boolean tensor sharded on its first dimension and asserts
    # the DTensor result matches the eager result with a replicated scalar layout. Skipped on GPU
    # (no int32 collective reduce) and on TPU unless exactly 2 cores exist.
    # TODO(b/193531363): Track the work to support int32 reduce.
    self.skipForDeviceType(['GPU'],
                           'GPUs do not support int32 collective reduce')
    self.skipForDeviceType(['TPU'],
                           'This test only needs to run on 2 cores.',
                           unless_device_count_equals_to=2)

    a = constant_op.constant(
        np.array([[True, False, False, True], [False, False, False, True]]),
        dtype=dtypes.bool)
    expected_result = reduction(a)

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    dtensor_result = reduction(sharded_a)

    self.assertDTensorEqual(expected_result, self.scalar_layout, dtensor_result)

  def testAllToAllOnBool(self):
    # [FUNCTION DOC]
    # Scenario: relayout a boolean 2x4 tensor from first-dimension sharded to fully replicated
    # (which gathers shards) and assert it equals the original with the replicated layout.
    # Skipped on GPU and on TPU unless exactly 2 cores exist.
    # TODO(b/193531363): Track the work to support int32 reduce.
    self.skipForDeviceType(['GPU'],
                           'GPUs do not support int32 collective reduce')
    self.skipForDeviceType(['TPU'],
                           'This test only needs to run on 2 cores.',
                           unless_device_count_equals_to=2)

    a = constant_op.constant(
        np.array([[True, False, False, True], [False, False, False, True]]),
        dtype=dtypes.bool)
    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    unsharded_a = api.relayout(sharded_a, self.fully_replicated_layout_2d)

    self.assertDTensorEqual(a, self.fully_replicated_layout_2d, unsharded_a)

  def testAllToAllOnInt32(self):
    # [FUNCTION DOC]
    # Scenario: relayout a 2x2 int32 tensor from sharded to fully replicated and assert it
    # equals the original. The comment notes int32 all-concat is unsupported in some cases (a
    # segfault is mentioned). Skipped on GPU and on TPU unless exactly 2 cores exist.
    # TODO(b/193471732): Tracking the work to do int32 all-concat.
    #
    # Currently, the test will fail with segfault.
    self.skipForDeviceType(['GPU'],
                           'GPUs do not support int32 StridedSliceXXX Ops')
    self.skipForDeviceType(['TPU'],
                           'This test only needs to run on 2 cores.',
                           unless_device_count_equals_to=2)

    a = constant_op.constant(np.array([[1, 2], [3, 4]]), dtype=dtypes.int32)
    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    unsharded_a = api.relayout(sharded_a, self.fully_replicated_layout_2d)

    self.assertDTensorEqual(a, self.fully_replicated_layout_2d, unsharded_a)

  def testCollectiveOpsOnComplex64(self):
    # [FUNCTION DOC]
    # Scenario: shard a 2x2 complex64 tensor (scatter) and then replicate it (gather/reduce).
    # Asserts the replicated result equals the original value with the replicated layout.
    # This functions tests for AllScatter, AllGather, and AllReduce.
    a = constant_op.constant(
        np.array([[1, 2 + 2j], [3 + 1j, 4 + 5j]]), dtype=dtypes.complex64
    )
    # Tests AllScatter
    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    # Tests AllGather / AllReduce
    unsharded_a = api.relayout(sharded_a, self.fully_replicated_layout_2d)

    self.assertDTensorEqual(a, self.fully_replicated_layout_2d, unsharded_a)

  def testCollectiveOpsOnComplex128(self):
    # [FUNCTION DOC]
    # Scenario: relayout a 2x2 complex128 tensor first to batch-sharded and then to inner-sharded
    # over the x dimension (exercising scatter, all-to-all and reduce). Asserts the final layout
    # matches the inner-sharded layout. Skipped on TPU (complex128 unsupported).
    # This function tests for AllScattering, AllReduce, and AllToAll.
    self.skipForDeviceType(['TPU'], 'TPU does not support comolex128')
    expected_layout = Layout.inner_sharded(self.mesh, 'x', rank=2)
    initial_layout = Layout.batch_sharded(self.mesh, 'x', rank=2)

    a = constant_op.constant(
        np.array([[1, 2 + 2j], [3 + 1j, 4 + 5j]]), dtype=dtypes.complex128)
    # Tests AllScatter
    sharded_a_initial = api.relayout(a, initial_layout)
    # Tests AllToAll / AllReduce
    sharded_a = api.relayout(sharded_a_initial, expected_layout)
    api.check_layout(sharded_a, expected_layout)

  def testNoOpAllToAll(self):
    # [FUNCTION DOC]
    # Scenario: relayout a float32 2x4 tensor to sharded and back to fully replicated, and assert
    # the final value and layout equal the original. Skipped on TPU unless exactly 2 cores exist.
    self.skipForDeviceType(['TPU'],
                           'This test only needs to run on 2 cores.',
                           unless_device_count_equals_to=2)

    a = constant_op.constant(
        np.array([[1., 2., 3., 4.], [5.0, 6.0, 7.0, 8.0]]),
        dtype=dtypes.float32)
    expected_result = a

    sharded_a = api.relayout(a, self.first_dimension_sharded_layout_2d)
    dtensor_result = api.relayout(sharded_a, self.fully_replicated_layout_2d)

    self.assertDTensorEqual(expected_result, self.fully_replicated_layout_2d,
                            dtensor_result)

  # Regression test for b/184401449.
  def testDeviceIdTensorOnSplitHost(self):
    # [FUNCTION DOC]
    # Regression test for a bug about device id tensors when local TPU device ids are not
    # sequential. Only runs on TPU with 8 cores. Builds a 2x4 TPU mesh with a non-sequential local
    # id list, registers its TPU core ids on a new DTensor device, then multiplies a
    # batch-sharded id matrix (ids in first column) by a y-sharded matrix of ones inside a
    # compiled function. Asserts the result layout is batch-sharded over x and each local shard
    # holds its own batch id repeated four times.
    # KNOWN ISSUE: the code comments mention a 2x2 mesh while the mesh is 2x4 (comment typo only).
    if not test_util.is_tpu_present():
      self.skipTest('This test only runs on TPUs.')
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = [0, 1, 4, 5, 2, 3, 6, 7]  # not sequentially increasing
    mesh = layout_lib.Mesh(_MESH_DIMS, global_ids, local_ids,
                           test_util.create_device_list((2, 4), 'TPU'),
                           'tpu_mesh')
    # This works because on 2x2, global device IDs are equal to physical TPU
    # core IDs: both are range(8). So local device IDs happen to be usable here.
    # TODO(b/180046115): Add a device.get_tpu_core_ids method and translate
    # device IDs to core IDs before setting the list here.
    device = dtensor_device.DTensorDevice(meshes=[mesh])
    device.set_tpu_core_ids('tpu_mesh', local_ids)
    layout_x = Layout.batch_sharded(mesh, _MESH_DIM_X, 2)
    layout_y = Layout.batch_sharded(mesh, _MESH_DIM_Y, 2)

    # Create a 2x4 batch-sharded d-tensor, with batch IDs in its first column
    # and zeros in other columns.
    replica_ids = constant_op.constant(
        np.array([[0, 0, 0, 0], [1, 0, 0, 0]]), dtype=dtypes.int32
    )
    replica_ids = api.relayout(replica_ids, layout_x)

    # Create a 4x4 y-sharded d-tensor filled with ones.
    ones = constant_op.constant(
        np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]),
        dtype=dtypes.int32,
    )
    ones = api.relayout(ones, layout_y)

    # If `a` has a layout of [x, unsharded], and `b` has a layout of
    # [y, unsharded], the matmul will slice `a` to [x, y], do a local matmul,
    # and all-reduce to produce a final result with a layout of [x, unsharded].
    #
    # Because `a` only has non-zero values in its first column, local devices
    # must be given a correct device ID tensor (as the first argument of every
    # function) to produce correct `begin` values for slicing `a`.
    #
    # Although this function only contains a single op, running it in op-by-op
    # mode doesn't produce the intended effect because the output of
    # math_ops.matmul would have a layout of [y, unsharded] instead of
    # [x, unsharded].
    @polymorphic_function.function
    def func(a, b):
      # [FUNCTION DOC]
      # Compiled helper returning the matrix product of its two inputs, used so the matmul and
      # its all-reduce run inside one function with device id tensors.
      return math_ops.matmul(a, b)

    dtensor_result = func(replica_ids, ones)

    # The correct result is a 2x4 batch-sharded d-tensor, with rows filled with
    # batch IDs.
    expected_result = [
        constant_op.constant(
            [loc[_MESH_DIM_X]] * 4, dtype=dtypes.int32, shape=[1, 4])
        for loc in mesh.local_device_locations()
    ]

    self.assertEqual(api.fetch_layout(dtensor_result), layout_x)
    dtensor_result = [t.numpy() for t in api.unpack(dtensor_result)]
    self.assertAllEqual(expected_result, dtensor_result)

  def testDifferentShapesBetweenCalls(self):
    # [FUNCTION DOC]
    # Scenario: call a compiled function twice with sharded inputs whose number of unique values
    # differs between calls (so output shapes differ). The function takes unique values of the
    # inputs and sums labels per unique index. Asserts the two results equal [4, 6] and
    # [4, 2, 4]. Skipped under the TFRT CPU runtime and on TPU (needs static shapes).
    self.skipForTfrt(
        'b/269333905, TFRT cpu fails due to step_id not propagated.'
    )
    self.skipForDeviceType(
        ['TPU'],
        'Known failure under TPU for legalization requires a static shape.',
    )

    # The error only happens across the batch, where the value of
    # tf.unique are differnet.
    def produce_data(inputs, label):
      # [FUNCTION DOC]
      # HELPER: builds sharded test inputs for testDifferentShapesBetweenCalls.
      # Takes an inputs list and a label list and returns both relaid out as rank-1 tensors
      # sharded over the x dimension of the test mesh.
      inputs = api.relayout(
          inputs, Layout.batch_sharded(self.mesh, _MESH_DIM_X, 1)
      )
      label = api.relayout(
          label, Layout.batch_sharded(self.mesh, _MESH_DIM_X, 1)
      )
      return inputs, label

    @polymorphic_function.function
    def train_fn(inputs, label):
      # [FUNCTION DOC]
      # Compiled helper: finds unique values of inputs (with an index for each element) and
      # returns the label values summed per unique index, so the output length depends on data.
      inputs, indices = array_ops.unique(inputs)
      return math_ops.unsorted_segment_sum(label, indices, len(inputs))

    input1, label1 = produce_data([6, 0, 6, 0], [1, 2, 3, 4])
    input2, label2 = produce_data([2, 1, 2, 0], [1, 2, 3, 4])

    result1 = train_fn(input1, label1)
    result2 = train_fn(input2, label2)
    self.assertAllEqual(
        result1.numpy(),
        [
            4,
            6,
        ],
    )
    self.assertAllEqual(
        result2.numpy(),
        [
            4,
            2,
            4,
        ],
    )


class CollectiveTestWithCustomMesh(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case whose tests each build their own eight-device mesh (one-dimensional or 4x2) to
  # check the all-reduce combiner pass and mixed-precision all-reduce. Most tests require 8 TPU
  # cores when run on TPU. No shared fixtures beyond the base class.

  # Create two independent global AllReduce ops that should get combined.
  def testGlobalAllReduceCombiner(self):
    # [FUNCTION DOC]
    # Scenario: on an 8-device one-dimensional mesh, a compiled function sums two batch-sharded
    # inputs over axis 0 (two independent global all-reduces), squares and adds the results. The
    # reference result is computed eagerly before sharding. Asserts the DTensor result equals it
    # with a replicated rank-1 layout. Intended to exercise combining of the two all-reduces.
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    # Create and use an eight-device mesh just for this test.
    global_ids = test_util.create_device_ids_array((8,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids,
                                test_util.create_device_list((8,), device))
        for device in ('CPU', 'GPU', 'TPU')
    }

    mesh = self.configTestMesh(mesh_dict)
    fully_replicated_layout_1d = Layout.replicated(mesh, rank=1)
    first_dimension_sharded_layout_2d = Layout.batch_sharded(
        mesh, _MESH_DIM_X, 2)

    @polymorphic_function.function
    def func(a, b):
      # [FUNCTION DOC]
      # Compiled helper: sums a and b over axis 0 (two independent all-reduces when sharded),
      # then returns the sum of their element-wise squares.
      a = math_ops.reduce_sum(a, axis=[0])
      b = math_ops.reduce_sum(b, axis=[0])
      # Do something with the results before adding them, to make sure the MLIR
      # pass can handle dependent ops sandwiched between two all-reduce ops.
      return gen_math_ops.square(a) + gen_math_ops.square(b)

    row = constant_op.constant(np.array([[1., 2.0]]), dtype=dtypes.float32)
    a = array_ops.repeat(row, repeats=[8], axis=0)
    b = gen_array_ops.reverse_v2(a, axis=[1])
    expected_result = func(a, b)

    a = api.relayout(a, first_dimension_sharded_layout_2d)
    b = api.relayout(b, first_dimension_sharded_layout_2d)
    dtensor_result = func(a, b)

    self.assertDTensorEqual(expected_result, fully_replicated_layout_1d,
                            dtensor_result)

  # Create two independent global AllReduce ops that should get combined.
  # Create two independent global AllReduce ops with different reductions, that
  # should not get combined

  def testGlobalAllReduceCombinerDifferentReduce(self):
    # [FUNCTION DOC]
    # Same setup as the global combiner test, but the compiled function applies a sum to one
    # input and a mean to the other, so the two all-reduces use different reductions and must not
    # be combined. Asserts the DTensor result matches the eager result with a replicated rank-1
    # layout. Requires 8 TPU cores on TPU.
    # KNOWN ISSUE: the comment block above this method repeats a stale line about ops that "should
    # get combined", contradicting the test's purpose of ops that should not be combined.
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    # Create and use an eight-device mesh just for this test.
    global_ids = test_util.create_device_ids_array((8,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids,
                                test_util.create_device_list((8,), device))
        for device in ('CPU', 'GPU', 'TPU')
    }

    mesh = self.configTestMesh(mesh_dict)
    fully_replicated_layout_1d = Layout.replicated(mesh, rank=1)
    first_dimension_sharded_layout_2d = Layout.batch_sharded(
        mesh, _MESH_DIM_X, 2)

    @polymorphic_function.function
    def func(a, b):
      # [FUNCTION DOC]
      # Compiled helper: sums a over axis 0 and averages b over axis 0 (different reduction
      # kinds), then returns the sum of their element-wise squares.
      a = math_ops.reduce_sum(a, axis=[0])
      b = math_ops.reduce_mean(b, axis=[0])

      # Do something with the results before adding them, to make sure the MLIR
      # pass can handle dependent ops sandwiched between two all-reduce ops.
      return gen_math_ops.square(a) + gen_math_ops.square(b)

    row = constant_op.constant(np.array([[1., 2.0]]), dtype=dtypes.float32)
    a = array_ops.repeat(row, repeats=[8], axis=0)
    b = gen_array_ops.reverse_v2(a, axis=[1])
    expected_result = func(a, b)

    a = api.relayout(a, first_dimension_sharded_layout_2d)
    b = api.relayout(b, first_dimension_sharded_layout_2d)
    dtensor_result = func(a, b)

    self.assertDTensorEqual(expected_result, fully_replicated_layout_1d,
                            dtensor_result)

  # Create two independent subgroup AllReduce ops that should get combined.
  def testSubgroupAllReduceCombiner(self):
    # [FUNCTION DOC]
    # Scenario: on a 4x2 mesh (eight devices) with a fully sharded rank-2 layout, a compiled
    # function sums two inputs over axis 0 (subgroup all-reduces over the x dimension) and adds
    # their squares. Asserts the result equals the eager result under a layout sharded only on y.
    # Requires 8 TPU cores on TPU.
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    # Create and use an eight-device mesh just for this test.
    global_ids = test_util.create_device_ids_array((4, 2))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh(_MESH_DIMS, global_ids, local_ids,
                                test_util.create_device_list((4, 2), device))
        for device in ('CPU', 'GPU', 'TPU')
    }

    mesh = self.configTestMesh(mesh_dict)
    fully_sharded_layout_2d = Layout(_MESH_DIMS, mesh)

    @polymorphic_function.function
    def func(a, b):
      # [FUNCTION DOC]
      # Compiled helper: sums a and b over axis 0 and returns the sum of their element-wise
      # squares.
      a = math_ops.reduce_sum(a, axis=[0])
      b = math_ops.reduce_sum(b, axis=[0])
      # Do something with the results before adding them, to make sure the MLIR
      # pass can handle dependent ops sandwiched between two all-reduce ops.
      return gen_math_ops.square(a) + gen_math_ops.square(b)

    row = constant_op.constant(np.array([[1., 2.0]]), dtype=dtypes.float32)
    a = array_ops.repeat(row, repeats=[8], axis=0)
    b = gen_array_ops.reverse_v2(a, axis=[1])
    expected_result = func(a, b)

    a = api.relayout(a, fully_sharded_layout_2d)
    b = api.relayout(b, fully_sharded_layout_2d)
    dtensor_result = func(a, b)

    self.assertDTensorEqual(expected_result, Layout([_MESH_DIM_Y], mesh),
                            dtensor_result)

  # TODO(b/188605096): also add a MixedPrecisionReduceScatter test
  def testMixedPrecisionAllReduce(self):
    # [FUNCTION DOC]
    # Scenario: enable mixed-precision reduction through two environment variables (enable flag
    # and a bfloat16 maximum group size of 4), build an 8-device mesh, and reduce a bfloat16
    # 8x6 tensor sharded on the first dimension, so the reduction group exceeds the limit.
    # Asserts the result equals the numpy column sums with a replicated rank-1 layout. Skipped on
    # GPU; needs 8 cores on TPU. Side effects: sets both environment variables and, at the end,
    # restores or removes them.
    # KNOWN ISSUE: environment variables are set before the skip checks and cleanup is not in a
    # finally block, so a skip or failing assertion leaves them set for later tests.
    has_enable_dtensor_mixed_precision_reduce = (
        'DTENSOR_ENABLE_MIXED_PRECISION_REDUCE' in os.environ
    )
    has_dtensor_reduce_in_bfloat16_max_group_size = (
        'DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE' in os.environ
    )
    if has_dtensor_reduce_in_bfloat16_max_group_size:
      old_dtensor_reduce_in_bfloat16_max_group_size = os.environ[
          'DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE']
    os.environ['DTENSOR_ENABLE_MIXED_PRECISION_REDUCE'] = ''
    os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE'] = '4'

    self.skipForDeviceType(['GPU'],
                           'GPUs do not support bfloat16 reduce')
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    # Create and use an 8-device mesh just for this test. Mixed-precision
    # AllReduce will be in effect since the reduction will be across 8 devices
    # which is larger than the max group size flag value of 4.
    global_ids = test_util.create_device_ids_array((8,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids,
                                test_util.create_device_list((8,), device))
        for device in ('CPU', 'GPU', 'TPU')
    }

    mesh = self.configTestMesh(mesh_dict)
    replicated_layout_1d = Layout.replicated(mesh, rank=1)
    first_dim_sharded_layout_1d = Layout.batch_sharded(
        mesh, _MESH_DIM_X, rank=2)

    @polymorphic_function.function
    def func(x):
      # [FUNCTION DOC]
      # Compiled helper returning the sum of its input over axis 0.
      return math_ops.reduce_sum(x, axis=0)

    # Reduce across 8 devices.
    inp = constant_op.constant(
        np.arange(48.).reshape((8, 6)), dtype=dtypes.bfloat16)
    expected_result = np.sum(inp, axis=0)

    inp_dtensor = api.relayout(inp, first_dim_sharded_layout_1d)
    dtensor_result = func(inp_dtensor)

    self.assertDTensorEqual(
        expected_result, replicated_layout_1d, dtensor_result)

    if not has_enable_dtensor_mixed_precision_reduce:
      del os.environ['DTENSOR_ENABLE_MIXED_PRECISION_REDUCE']
    if has_dtensor_reduce_in_bfloat16_max_group_size:
      os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE'] = (
          old_dtensor_reduce_in_bfloat16_max_group_size
      )
    else:
      del os.environ['DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE']

  # Create two independent AllReduce ops with indirect dependency, that should
  # not get combined.
  def testAllReduceCombinerWithIndirectDependency(self):
    # [FUNCTION DOC]
    # Scenario: validates the dependency check of the all-reduce combiner. On an 8-device mesh,
    # a compiled function sums a DTensor variable, adds to the variable (in place) a value
    # depending on that sum, then sums the variable again, so the second reduction indirectly
    # depends on the first. Asserts the result equals the same computation done on plain eager
    # tensors, with a replicated scalar layout. Requires 8 TPU cores on TPU.
    # The purpose of this test is to validate the depdency check in AllReduce
    # AllReduce combiner (dtensor_allreduce_combine_optimization). Specifically,
    # the side effects from indirect dependency.
    self.skipForDeviceType(['TPU'],
                           'This test requires 8 TPU cores.',
                           unless_device_count_equals_to=8)

    # Create and use an eight-device mesh just for this test.
    global_ids = test_util.create_device_ids_array((8,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh([_MESH_DIM_X], global_ids, local_ids,
                                test_util.create_device_list((8,), device))
        for device in ('CPU', 'GPU', 'TPU')
    }

    mesh = self.configTestMesh(mesh_dict)
    first_dim_sharded_layout_1d = Layout.batch_sharded(
        mesh, _MESH_DIM_X, rank=1)

    init_value = constant_op.constant(np.ones(32), dtype=dtypes.float32)
    init_value = api.relayout(init_value, first_dim_sharded_layout_1d)

    # 2nd reduction indrectly depend on the 1st reduction
    @polymorphic_function.function
    def func(v):
      # [FUNCTION DOC]
      # Compiled helper taking a variable: sums it, adds the variable plus that sum back into
      # the variable (mutating it), then returns the sum of the updated variable.
      a = math_ops.reduce_sum(v)
      v.assign_add(v+a)
      b = math_ops.reduce_sum(v)
      return b

    v = d_variable.DVariable(init_value)
    dtensor_result = func(v)

    # Replicate the scenario above without using dtensor
    expected_result = constant_op.constant(np.ones(32), dtype=dtypes.float32)
    expected_result += expected_result + math_ops.reduce_sum(expected_result)
    expected_result = math_ops.reduce_sum(expected_result)

    self.assertDTensorEqual(
        expected_result, Layout.replicated(mesh=mesh, rank=0), dtensor_result
    )

if __name__ == '__main__':
  test.main()
