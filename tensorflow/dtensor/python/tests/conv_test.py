# [MODULE DOC]
# Tests for convolution ops (2D and 3D) executed on DTensor meshes. One test class uses a
# 2x2x2 mesh (batch, height, width) to check conv followed by einsum, forward and with gradients,
# in eager and compiled modes with replicated or batch-sharded inputs. A second class checks
# spatially partitioned 2D convolution (TPU only). Module constants define dimension names and
# sizes, and a shared list of parameter cases for the spatial tests. Results are compared with
# plain non-DTensor computations.
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

"""Tests for executing ops needed to implement image model."""

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.platform import test


UNSHARDED = layout_lib.UNSHARDED
Mesh = layout_lib.Mesh
Layout = layout_lib.Layout

BATCH_DIM = 'batch'
DEPTH_DIM = 'depth'
HEIGHT_DIM = 'height'
WIDTH_DIM = 'width'
BATCH_SIZE = 4
DEPTH = 8
HEIGHT = 12
WIDTH = 12
CHANNEL_IN = 1
CHANNEL_OUT = 3


class ConvOpTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case for convolution followed by einsum on a three-dimensional 2x2x2 mesh (batch,
  # height, width). Setup stores the mesh, a replicated rank-2 layout and a batch-sharded rank-2
  # layout used for expected outputs.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Builds a 2x2x2 mesh with dimensions batch, height and width for CPU, GPU and
    # TPU, lets the base class pick the one for the current device, and stores it as mesh plus
    # the replicated and batch-sharded rank-2 layouts. Returns nothing.
    super().setUp()

    global_ids = test_util.create_device_ids_array((2, 2, 2))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {}
    for device in ('CPU', 'GPU', 'TPU'):
      mesh_dict[device] = Mesh(
          [BATCH_DIM, HEIGHT_DIM, WIDTH_DIM],
          global_ids,
          local_ids,
          test_util.create_device_list((2, 2, 2), device),
      )

    self.mesh = self.configTestMesh(mesh_dict)

    self.replicated_2d = Layout.replicated(self.mesh, 2)
    self.batch_sharded_2d = Layout.batch_sharded(self.mesh, BATCH_DIM, 2)

  @parameterized.named_parameters(
      test_util.product(
          *[
              [
                  (
                      'Conv2D',
                      nn_ops.conv2d_v2,
                      (BATCH_SIZE, HEIGHT, WIDTH, CHANNEL_IN),
                      (2, 2, CHANNEL_IN, CHANNEL_OUT),
                      'bhwc,xy->by',
                      [1, 2, 1, 1],
                  ),
                  (
                      'Conv3D',
                      nn_ops.conv3d_v2,
                      (BATCH_SIZE, DEPTH, HEIGHT, WIDTH, CHANNEL_IN),
                      (2, 2, 2, CHANNEL_IN, CHANNEL_OUT),
                      'bdhwc,xy->by',
                      [1, 1, 2, 1, 1],
                  ),
              ],
              [
                  ('Eager', True),
                  ('Graph', False),
              ],
              [
                  ('ReplicatedInput', 'replicated'),
                  ('BatchShardedInput', 'batch_sharded'),
              ],
              [
                  ('ValidPadding', 'VALID'),
                  ('SamePadding', 'SAME'),
              ],
          ]
      )
  )
  def testConvFollowedByEinsum(self, conv_op, input_size, kernel_size,
                               einsum_eq, strides, eager_mode, input_sharding,
                               padding):
    # [FUNCTION DOC]
    # Parameterized over 2D/3D convolution, eager/compiled execution, replicated/batch-sharded
    # input and valid/same padding. Parameters: conv_op (the convolution function), input_size
    # and kernel_size (shapes), einsum_eq (equation applied to the conv output and a 2x2 weight),
    # strides, eager_mode (true to run uncompiled), input_sharding (name of input layout kind) and
    # padding. Computes a golden result on plain random tensors, packs the same values into
    # DTensors (kernel and weight replicated), reruns, and asserts equality and that the output
    # layout is replicated or batch-sharded matching the input.
    x_in = constant_op.constant(
        np.random.random(size=input_size), dtype=dtypes.float32
    )
    kernel_in = constant_op.constant(
        np.random.random(size=kernel_size), dtype=dtypes.float32
    )
    weight = constant_op.constant(
        np.random.random(size=(2, 2)), dtype=dtypes.float32
    )

    def conv_fn(inputs, img_kernel, layer_weights):
      # [FUNCTION DOC]
      # Helper under test: applies the parameterized convolution to inputs and kernel with the
      # given strides and padding, then an einsum with the layer weights, and returns the result.
      output = conv_op(inputs, img_kernel, strides=strides, padding=padding)
      output = special_math_ops.einsum(einsum_eq, output, layer_weights)
      return output

    if not eager_mode:
      conv_fn = polymorphic_function.function(conv_fn)

    golden_result = conv_fn(x_in, kernel_in, weight)

    if input_sharding == 'replicated':
      input_layout = Layout.replicated(self.mesh, len(input_size))
      output_layout = self.replicated_2d
    elif input_sharding == 'batch_sharded':
      input_layout = Layout.batch_sharded(self.mesh, BATCH_DIM, len(input_size))
      output_layout = self.batch_sharded_2d

    kernel_layout = Layout.replicated(self.mesh, len(kernel_size))

    d_x_in = numpy_util.pack_numpy(x_in, input_layout)
    d_kernel_in = numpy_util.pack_numpy(kernel_in, kernel_layout)
    d_weight = numpy_util.pack_numpy(weight, self.replicated_2d)
    d_result = conv_fn(d_x_in, d_kernel_in, d_weight)

    self.assertDTensorEqual(golden_result, output_layout, d_result)

  @parameterized.named_parameters(
      test_util.product(
          *[
              [
                  (
                      'Conv2D',
                      nn_ops.conv2d_v2,
                      (BATCH_SIZE, HEIGHT, WIDTH, CHANNEL_IN),
                      (2, 2, CHANNEL_IN, CHANNEL_OUT),
                      'bhwc,xy->by',
                      [1, 1, 1, 1],
                  ),
                  (
                      'Conv3D',
                      nn_ops.conv3d_v2,
                      (BATCH_SIZE, DEPTH, HEIGHT, WIDTH, CHANNEL_IN),
                      (2, 2, 2, CHANNEL_IN, CHANNEL_OUT),
                      'bdhwc,xy->by',
                      [1, 1, 1, 1, 1],
                  ),
              ],
              [
                  ('ReplicatedInput', 'replicated'),
                  ('BatchShardedInput', 'batch_sharded'),
              ],
              [
                  ('ValidPadding', 'VALID'),
                  ('SamePadding', 'SAME'),
              ],
          ]
      )
  )
  def testConvFollowedByEinsumWithGradient(self, conv_op, input_size,
                                           kernel_size, einsum_eq, strides,
                                           input_sharding, padding):
    # [FUNCTION DOC]
    # Parameterized over 2D/3D convolution, replicated/batch-sharded input and valid/same padding
    # (parameters as in the forward-only test minus the eager flag). Runs a compiled function
    # with a gradient tape computing the conv plus einsum output and gradients with respect to
    # input, kernel and weight, once on plain tensors and once on DTensors. Asserts the output
    # layout follows the input sharding, input gradients are replicated (see the TODO: they
    # should match the input layout), and kernel and weight gradients are replicated.
    x_in = constant_op.constant(
        np.random.random(size=input_size), dtype=dtypes.float32
    )
    kernel_in = constant_op.constant(
        np.random.random(size=kernel_size), dtype=dtypes.float32
    )
    weight = constant_op.constant(
        np.random.random(size=(2, 2)), dtype=dtypes.float32
    )

    @polymorphic_function.function
    def conv_fn(inputs, img_kernel, layer_weights):
      # [FUNCTION DOC]
      # Compiled helper: under a gradient tape watching all three inputs, applies the convolution
      # and the einsum, then returns the output and its gradients with respect to the inputs,
      # kernel and layer weights (in that order).
      with backprop.GradientTape() as tape:
        tape.watch([inputs, img_kernel, layer_weights])
        output = conv_op(inputs, img_kernel, strides=strides, padding=padding)
        output = special_math_ops.einsum(einsum_eq, output, layer_weights)

      inputs_grad, kernel_grad, weight_grad = tape.gradient(
          output, [inputs, img_kernel, layer_weights])
      return output, inputs_grad, kernel_grad, weight_grad

    result, inputs_grad, kernel_grad, weight_grad = conv_fn(
        x_in, kernel_in, weight)

    if input_sharding == 'replicated':
      input_layout = Layout.replicated(self.mesh, len(input_size))
      output_layout = self.replicated_2d
    elif input_sharding == 'batch_sharded':
      input_layout = Layout.batch_sharded(self.mesh, BATCH_DIM, len(input_size))
      output_layout = self.batch_sharded_2d

    kernel_layout = Layout.replicated(self.mesh, len(kernel_size))

    d_x_in = numpy_util.pack_numpy(x_in, input_layout)
    d_kernel_in = numpy_util.pack_numpy(kernel_in, kernel_layout)
    d_weight = numpy_util.pack_numpy(weight, self.replicated_2d)
    d_result, d_inputs_grad, d_kernel_grad, d_weight_grad = conv_fn(
        d_x_in, d_kernel_in, d_weight)

    self.assertDTensorEqual(result, output_layout, d_result)
    # TODO(b/208700444): layout of input grads should match layout of input.
    self.assertDTensorEqual(
        inputs_grad,
        Layout.replicated(self.mesh, len(input_size)),
        d_inputs_grad,
    )
    self.assertDTensorEqual(kernel_grad, kernel_layout, d_kernel_grad)
    self.assertDTensorEqual(weight_grad, self.replicated_2d, d_weight_grad)


SPATIALLY_PARTITIONED_CONV_TEST_CASES = [
    [
        ('Case1', (BATCH_SIZE, 8, 16, CHANNEL_IN), (3, 5, CHANNEL_IN,
                                                    CHANNEL_OUT)),
        ('Case2', (BATCH_SIZE, 8, 128, CHANNEL_IN), (3, 9, CHANNEL_IN,
                                                     CHANNEL_OUT)),
    ],
    [
        ('ValidPadding', 'VALID'),
        ('SamePadding', 'SAME'),
    ],
    [
        ('Batch_1d_2x4', [BATCH_DIM, UNSHARDED, WIDTH_DIM, UNSHARDED], (2, 4)),
        ('2d_2x4', [UNSHARDED, HEIGHT_DIM, WIDTH_DIM, UNSHARDED], (2, 4)),
        ('Batch_2d_2x2x2', [BATCH_DIM, HEIGHT_DIM, WIDTH_DIM,
                            UNSHARDED], (2, 2, 2)),
    ],
]


class SpatiallyPartitionedConvOpTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case for 2D convolutions whose input is partitioned across spatial (height/width)
  # dimensions, which needs collective permute and is therefore TPU-only for now. Meshes are
  # built per test from sharding specs and topology via a private helper.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Calls the parent setup, then skips the test unless a TPU is present
    # (collective permute is not yet supported on CPU). Returns nothing.
    super().setUp()

    # TODO(b/261485237): Enable CPU testing once CollectivePermute is supported
    # on CPU's.
    if not test_util.is_tpu_present():
      self.skipTest('This test only runs on TPUs.')

  def _create_mesh(self, mesh_dims, topology):
    # [FUNCTION DOC]
    # HELPER: builds the mesh for the spatial convolution tests.
    # Takes mesh_dims (list of dimension names) and topology (tuple of dimension sizes). Creates
    # a mesh with that shape for each of CPU, GPU and TPU and returns the one the base class
    # selects for the current device type.
    global_ids = test_util.create_device_ids_array(topology)
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {}
    for device in ('CPU', 'GPU', 'TPU'):
      mesh_dict[device] = Mesh(
          mesh_dims,
          global_ids,
          local_ids,
          test_util.create_device_list(topology, device),
      )

    return self.configTestMesh(mesh_dict)

  @parameterized.named_parameters(
      test_util.product(*SPATIALLY_PARTITIONED_CONV_TEST_CASES))
  def testConv(self, input_shape, kernel_shape, padding, sharding_specs,
               topology):
    # [FUNCTION DOC]
    # Parameterized over input/kernel shapes, valid/same padding, sharding specs and mesh
    # topology. Builds a mesh from the sharded dimension names, runs a stride-1 2D convolution on
    # plain random tensors and on DTensors (input sharded per the specs, kernel replicated), and
    # asserts the DTensor output equals the plain output and keeps the input layout.
    mesh_dims = [spec for spec in sharding_specs if spec != UNSHARDED]
    mesh = self._create_mesh(mesh_dims, topology)

    x_in = constant_op.constant(
        np.random.random(size=input_shape), dtype=dtypes.float32
    )
    kernel_in = constant_op.constant(
        np.random.random(size=kernel_shape), dtype=dtypes.float32
    )

    expected_output = nn_ops.conv2d_v2(
        x_in, kernel_in, strides=[1, 1, 1, 1], padding=padding
    )

    input_layout = Layout(sharding_specs, mesh)
    kernel_layout = Layout.replicated(mesh, 4)

    d_x_in = numpy_util.pack_numpy(x_in, input_layout)
    d_kernel_in = numpy_util.pack_numpy(kernel_in, kernel_layout)
    d_output = nn_ops.conv2d_v2(
        d_x_in, d_kernel_in, strides=[1, 1, 1, 1], padding=padding
    )

    self.assertDTensorEqual(expected_output, input_layout, d_output)

  @parameterized.named_parameters(
      test_util.product(*SPATIALLY_PARTITIONED_CONV_TEST_CASES))
  def testConvWithGradient(self, input_shape, kernel_shape, padding,
                           sharding_specs, topology):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test body always skips immediately (spatially partitioned conv backprop is
    # not implemented), so the code after the skip never runs.
    # Intended scenario (same parameters as the forward test): compute convolution output and
    # gradients for input and kernel in a compiled function on plain and DTensor values, and
    # assert outputs and gradients match, with input gradients using the input layout and kernel
    # gradients replicated.
    # TODO(b/208700444): add support for SPMD expansion of spatially partitioned
    # conv backprop.
    self.skipTest(
        'b/208700444: Spatially partitioned conv backprop not implemented.')

    mesh_dims = [spec for spec in sharding_specs if spec != UNSHARDED]
    mesh = self._create_mesh(mesh_dims, topology)

    x_in = constant_op.constant(
        np.random.random(size=input_shape), dtype=dtypes.float32
    )
    kernel_in = constant_op.constant(
        np.random.random(size=kernel_shape), dtype=dtypes.float32
    )

    @polymorphic_function.function
    def conv_fn(inputs, img_kernel, padding):
      # [FUNCTION DOC]
      # Compiled helper (unreachable because the test skips earlier): runs a stride-1 2D
      # convolution under a gradient tape and returns the output and gradients with respect to
      # the inputs and the kernel. Padding is the padding mode string.
      with backprop.GradientTape() as tape:
        tape.watch([inputs, img_kernel])
        output = nn_ops.conv2d_v2(
            inputs, img_kernel, strides=[1, 1, 1, 1], padding=padding
        )
      inputs_grad, kernel_grad = tape.gradient(output, [inputs, img_kernel])
      return output, inputs_grad, kernel_grad

    expected_output, expected_inputs_grad, expected_kernel_grad = conv_fn(
        x_in, kernel_in, padding)

    input_layout = Layout(sharding_specs, mesh)
    kernel_layout = Layout.replicated(mesh, 4)

    d_x_in = numpy_util.pack_numpy(x_in, input_layout)
    d_kernel_in = numpy_util.pack_numpy(kernel_in, kernel_layout)

    d_output, d_inputs_grad, d_kernel_grad = conv_fn(d_x_in, d_kernel_in,
                                                     padding)

    self.assertDTensorEqual(expected_output, input_layout, d_output)
    self.assertDTensorEqual(expected_inputs_grad, input_layout, d_inputs_grad)
    self.assertDTensorEqual(expected_kernel_grad, kernel_layout, d_kernel_grad)


if __name__ == '__main__':
  test.main()
