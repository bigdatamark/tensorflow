# [MODULE DOC]
# Large test module for DTensor SPMD (single program multiple data) expansion of TensorFlow ops.
# It defines one parameterized test class, DTensorSPMDTest, whose tests build ordinary eager or
# tf.function computations, run them once on plain tensors and once on DTensors laid out on a 2-D
# (x=2, y=4) mesh with various sharding layouts, and compare results and output layouts with the
# test utility assertDTensorEqual. Covered areas include unary/binary/reduction ops, matmul and
# batch matmul, convolution/pooling, relayout/all-to-all, random ops, gather/scatter, control flow,
# variables, and more.
# Also defines module-level mesh dimension names, parameter tables for matmul tests, the Layout,
# Mesh and UNSHARDED aliases, and two small helper functions (select_tol, order_broadcastable_operands).
# Import has no side effects beyond constants; running the file as a script starts the test runner
# through the platform test main (see the bottom of the file).
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
"""Tests for the open source DTensor Python API."""

import os
from unittest import mock
from absl.testing import parameterized
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.dtensor.python.tests import test_util_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as tf_test
from tensorflow.python.util import nest
# pylint: enable=g-direct-tensorflow-import

# Makes a 2-D mesh with dimensions as, X(2) and Y(4).
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_MESH_DIMS = [_MESH_DIM_X, _MESH_DIM_Y]

_MATMUL_IMPLEMENTED = (('_unsharded', 0,
                        0), ('_a_unsharded_b_contracting', 0,
                             1), ('_a_unsharded_b_non_contracting', 0,
                                  2), ('_a_non_contracting_b_unsharded', 1,
                                       0), ('_a_contracting_b_unsharded', 2, 0),
                       ('_a_contracting_b_contracting', 2,
                        1), ('_a_non_contracting_b_contracting', 1,
                             1), ('_a_non_contracting_b_non_contracting', 1, 2),
                       ('_a_contracting_b_non_contracting', 2, 2))
_BATCH_MATMUL_IMPLEMENTED = (('_unsharded', 0,
                              0), ('_a_unsharded_b_contracting', 0,
                                   2), ('_a_unsharded_b_non_contracting', 0,
                                        3), ('_a_batch_b_batch', 1, 1),
                             ('_a_non_contracting_b_unsharded', 2,
                              0), ('_a_contracting_b_unsharded', 3,
                                   0), ('_a_contracting_b_contracting', 3, 2),
                             ('_a_non_contracting_b_contracting', 2,
                              2), ('_a_non_contracting_b_non_contracting', 2,
                                   3), ('_a_contracting_b_non_contracting', 3,
                                        3), ('_a_unsharded_b_batch', 0,
                                             1), ('_a_batch_b_unsharded', 1, 0),
                             ('_a_batch_b_contracting', 1,
                              2), ('_a_batch_b_non_contracting', 1,
                                   3), ('_a_non_contracting_b_batch', 2,
                                        1), ('_a_contracting_b_batch', 3, 1))
_MATMUL_TRANSPOSE = (('', False, False), ('_b_transpose', False, True),
                     ('_a_transpose', True, False), ('_a_transpose_b_transpose',
                                                     True, True))

Layout = layout_lib.Layout
Mesh = layout_lib.Mesh
UNSHARDED = layout_lib.UNSHARDED


def select_tol(op, mesh, default_tol, low_res_tol):
  # [FUNCTION DOC]
  # HELPER: tolerance selection for numeric comparisons in the tests of this module.
  # Inputs: op (the op function under test), mesh (the DTensor mesh), default_tol and low_res_tol
  # (numbers). Returns low_res_tol when op is one of a fixed list of low-precision-on-TPU ops
  # (pow, log_softmax, tanh, acosh, asinh, digamma, igammac, lgamma, log1p, xlog1py, xlogy, zeta,
  # tan, sin, sinh, softplus) and the first local device of the mesh is a TPU; otherwise returns
  # default_tol. No side effects.
  # Lowers the tol for math_ops.pow,
  # nn_ops.log_softmax_v2 and gen_math_ops.tanh due to
  # resolution on TPU
  if (op not in [
      math_ops.pow, nn_ops.log_softmax_v2, gen_math_ops.tanh,
      gen_math_ops.acosh, gen_math_ops.asinh, gen_math_ops.digamma,
      gen_math_ops.igammac, gen_math_ops.lgamma, gen_math_ops.log1p,
      math_ops.xlog1py, gen_math_ops.xlogy, gen_math_ops.zeta, gen_math_ops.tan,
      gen_math_ops.sin, gen_math_ops.sinh, math_ops.softplus
  ]):
    return default_tol

  if 'TPU' in mesh.local_devices()[0]:
    return low_res_tol
  else:
    return default_tol


def order_broadcastable_operands(op, lhs, rhs):
  # [FUNCTION DOC]
  # HELPER: operand ordering for binary op tests with a broadcasting operand.
  # Inputs: op (binary op function), lhs and rhs (tensors; lhs is assumed to be the broadcasting
  # one). Returns a pair of operands: swapped (rhs, lhs) when op is truncate_div or truncate_mod,
  # which only accept a broadcastable right operand; otherwise the unchanged pair. No side effects.
  # Swaps operands lhs and rhs. Assumes lhs is the broadcasting tensor. Due to
  # ops only with right broadcastable operand like gen_math_ops.truncate_div
  if (op in [gen_math_ops.truncate_div, gen_math_ops.truncate_mod]):
    return rhs, lhs
  return lhs, rhs


class DTensorSPMDTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Parameterized test case covering DTensor SPMD expansion of many TensorFlow ops. Extends the
  # DTensor base test case. setUp builds a 2-D mesh (x of size 2, y of size 4) and a set of common
  # layouts (replicated, first-dim sharded, last-dim sharded for ranks 0 to 3) stored as instance
  # attributes. Each test method compares a plain-tensor result with a DTensor result and layout.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture. Calls the base setUp, skips on TPU unless exactly 8 TPU cores exist, then
    # creates a (2, 4) device id array and, for CPU, GPU and TPU, a mesh with dimensions x and y
    # (using the XLA SPMD flag for that device type). Picks the mesh for the current device type via
    # the base-class config helper and stores it as self.mesh. Also stores common layouts:
    # scalar_replicated_layout, replicated/first-dimension-sharded layouts for rank 1, 2 and 3,
    # last-dimension-sharded layouts for ranks 2 and 3, a middle-dimension-sharded 3-D layout, the
    # lists layouts_2d and layouts_3d, and a mapping shardings from 'batch'/'inner' to the
    # corresponding Layout factory. Returns nothing.
    super(DTensorSPMDTest, self).setUp()

    self.skipForDeviceType(['TPU'],
                           'all tests require 8 TPU cores.',
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = dict()
    for device in ('CPU', 'GPU', 'TPU'):
      mesh_dict[device] = Mesh(
          [_MESH_DIM_X, _MESH_DIM_Y],
          global_ids,
          local_ids,
          test_util.create_device_list((2, 4), device),
          use_xla_spmd=test_util.get_use_xla_spmd(device),
      )
    self.mesh = self.configTestMesh(mesh_dict)

    # Creates a bunch of common layouts used by tests later.
    # - 0-d
    self.scalar_replicated_layout = Layout.replicated(self.mesh, rank=0)
    # - 1-d
    self.replicated_layout_1d = Layout.replicated(self.mesh, rank=1)
    self.first_dimension_sharded_layout_1d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=1)
    # - 2-d
    self.replicated_layout_2d = Layout.replicated(self.mesh, rank=2)
    self.first_dimension_sharded_layout = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=2)
    self.last_dimension_sharded_layout = Layout.inner_sharded(
        self.mesh, _MESH_DIM_X, rank=2)

    self.layouts_2d = [
        self.replicated_layout_2d, self.first_dimension_sharded_layout,
        self.last_dimension_sharded_layout
    ]

    # - 3-d
    self.replicated_layout_3d = Layout.replicated(self.mesh, rank=3)
    self.first_dimension_sharded_layout_3d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=3)
    self.middle_dimension_sharded_layout_3d = Layout(
        [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    self.last_dimension_sharded_layout_3d = Layout.inner_sharded(
        self.mesh, _MESH_DIM_X, rank=3)

    self.layouts_3d = [
        self.replicated_layout_3d, self.first_dimension_sharded_layout_3d,
        self.middle_dimension_sharded_layout_3d,
        self.last_dimension_sharded_layout_3d
    ]

    self.shardings = {
        'batch': Layout.batch_sharded,
        'inner': Layout.inner_sharded
    }

  @parameterized.named_parameters(
      ('unsharded_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
      ('x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('unsharded_x', [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('x,y', [_MESH_DIM_X, _MESH_DIM_Y]),
  )
  @mock.patch.dict(
      os.environ, {'DTENSOR_ENABLE_REPLICATED_SPMD_AS_DEFAULT_TF.MOD': '1'}
  )
  def testDefaultReplicatedSpmd(self, shard_specs):
    # [FUNCTION DOC]
    # Parameterized by shard_specs (a list of mesh dimension names or unsharded markers). With an
    # environment variable patched to enable replicated SPMD as default for the Mod op, computes Mod
    # of a random 4x8 tensor (int32 on GPU, else float32) and the scalar 7, then repeats on inputs
    # relayouted to the given sharding (scalar replicated). Asserts the DTensor result equals the
    # plain result and has a fully replicated rank-2 layout.
    if test_util.is_gpu_present():
      dtype = dtypes.int32
    else:
      dtype = dtypes.float32

    x = stateless_random_ops.stateless_random_uniform(
        shape=[4, 8], seed=[0, 1], maxval=7, dtype=dtype
    )
    y = constant_op.constant(7, dtype=dtype)

    expected_result = math_ops.Mod(x=x, y=y)
    expected_layout = Layout.replicated(self.mesh, rank=2)
    dtensor_result = math_ops.Mod(
        x=api.relayout(x, layout=Layout(shard_specs, self.mesh)),
        y=api.relayout(y, layout=Layout([], self.mesh)),
    )

    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.product(
      shard_type=['replicated', 'batch_sharded'], full_matrices=[True, False])
  def testQR(self, shard_type, full_matrices):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the plain-tensor reference always uses full_matrices true even when the DTensor
    # call uses the parameter; the layout choice is also inverted relative to the name (the
    # 'replicated' case uses the batch-sharded layout and vice versa).
    # Parameters: shard_type ('replicated' or 'batch_sharded') and full_matrices (boolean). Computes
    # QR of a random 8x9x10 float32 tensor, relayouts the input to the chosen layout, runs QR
    # on it and asserts both outputs (q and r) equal the reference with that layout.
    np.random.seed(123)
    inputs = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 10).reshape([8, 9, 10]),
        dtype=dtypes.float32)

    expected_result = gen_linalg_ops.qr(
        input=inputs, full_matrices=True, name=None)

    if shard_type == 'replicated':
      layout = self.first_dimension_sharded_layout_3d
    else:
      layout = self.replicated_layout_3d

    inputs = api.relayout(inputs, layout)

    got = gen_linalg_ops.qr(
        input=inputs, full_matrices=full_matrices, name=None)
    self.assertDTensorEqual(expected_result[0], layout, got[0])
    self.assertDTensorEqual(expected_result[1], layout, got[1])

  def testReduceScatter(self,):
    # [FUNCTION DOC]
    # Builds two 128x128 random DTensors (one last-dim sharded, one first-dim sharded) and runs a
    # matmul inside a compiled function, once returning the product as is and once relayouting
    # it to first-dimension sharded (which lets the all-reduce plus scatter combine into a
    # reduce-scatter). Asserts both results are equal with first-dim sharded layout.
    # Generates an AllReduce due to sharding of inner dimensions of Matmul
    # and a Scatter due to the Relayout.  The AllReduce+Scatter can be combined
    # to a single ReduceScatter.
    a, b, c = 128, 128, 128
    seed = [0, 1]
    first_dim_sharded = self.first_dimension_sharded_layout
    second_dim_sharded = self.last_dimension_sharded_layout

    with api.default_mesh(self.mesh):
      m1 = numpy_util.stateless_random_uniform(
          layout=second_dim_sharded, shape=[a, b], seed=seed
      )
      m2 = numpy_util.stateless_random_uniform(
          layout=first_dim_sharded, shape=[b, c], seed=seed
      )

    @polymorphic_function.function
    def func():
      # [FUNCTION DOC]
      # Nested helper: multiplies the captured DTensors m1 and m2 and returns the product.
      m3 = math_ops.matmul(m1, m2)
      return m3

    @polymorphic_function.function
    def scattered_func():
      # [FUNCTION DOC]
      # Nested helper: multiplies captured m1 and m2 and relayouts the product to first-dimension
      # sharded, returning it.
      m3 = math_ops.matmul(m1, m2)
      return api.relayout(m3, self.first_dimension_sharded_layout)

    dtensor_result = func()
    dtensor_scattered_result = scattered_func()

    self.assertDTensorEqual(dtensor_result, self.first_dimension_sharded_layout,
                            dtensor_scattered_result)

  def testReduceScatterLastDimSharded(
      self,
  ):
    # [FUNCTION DOC]
    # Like testReduceScatter but the product is relayouted to the last-dimension sharded layout,
    # a reduce-scatter on a non-leading dimension that needs a transpose. Inputs (128x128) are
    # generated by a compiled random-uniform-plus-relayout function under the default mesh.
    # Asserts the plain product and the relayouted product are equal with last-dim sharded layout.
    # ReduceScatter on non-0th dimension which requires a transpose.
    a, b, c = 128, 128, 128
    seed = [0, 1]
    first_dim_sharded = self.first_dimension_sharded_layout
    second_dim_sharded = self.last_dimension_sharded_layout

    @polymorphic_function.function
    def uniform(shape, seed, layout):
      # [FUNCTION DOC]
      # Nested helper: draws stateless random uniform values of the given shape and seed and
      # relayouts them to the given layout; returns the resulting tensor.
      return api.relayout(
          stateless_random_ops.stateless_random_uniform(shape=shape, seed=seed),
          layout=layout,
      )

    with api.default_mesh(self.mesh):
      m1 = uniform(layout=second_dim_sharded, shape=[a, b], seed=seed)
      m2 = uniform(layout=first_dim_sharded, shape=[b, c], seed=seed)

    @polymorphic_function.function
    def func():
      # [FUNCTION DOC]
      # Nested helper: multiplies the captured DTensors m1 and m2 and returns the product.
      m3 = math_ops.matmul(m1, m2)
      return m3

    @polymorphic_function.function
    def scattered_func():
      # [FUNCTION DOC]
      # Nested helper: multiplies captured m1 and m2 and relayouts the product to last-dimension
      # sharded, returning it.
      m3 = math_ops.matmul(m1, m2)
      return api.relayout(m3, self.last_dimension_sharded_layout)

    dtensor_result = func()
    dtensor_scattered_result = scattered_func()

    self.assertDTensorEqual(
        dtensor_result,
        self.last_dimension_sharded_layout,
        dtensor_scattered_result,
    )

  @parameterized.named_parameters(
      (
          'xu_ux',
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'ux_xu',
          [layout_lib.UNSHARDED, _MESH_DIM_X],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'yu_uy',
          [_MESH_DIM_Y, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
      (
          'uy_yu',
          [layout_lib.UNSHARDED, _MESH_DIM_Y],
          [_MESH_DIM_Y, layout_lib.UNSHARDED],
      ),
  )
  def testAllToAll2D(self, src_spec, tgt_spec):
    # [FUNCTION DOC]
    # Parameters: src_spec and tgt_spec, sharding specs (lists of mesh dimension names or unsharded
    # markers) of rank 2. Packs an 8x8 float32 range tensor into a DTensor with the source
    # layout, relayouts it to the target layout inside a compiled function (an all-to-all exchange)
    # and asserts the values are unchanged and the result has the target layout.
    a = constant_op.constant(
        np.arange(
            8 * 8,
        ).reshape((8, 8)),
        dtype=dtypes.float32,
    )
    sharded_a = numpy_util.pack_numpy(a, layout=Layout(src_spec, self.mesh))

    @polymorphic_function.function
    def func(a):
      # [FUNCTION DOC]
      # Nested helper: relayouts its input to the target layout (from the enclosing test) and
      # returns it.
      return api.relayout(a, Layout(tgt_spec, self.mesh))

    dtensor_result = func(sharded_a)
    self.assertDTensorEqual(a, Layout(tgt_spec, self.mesh), dtensor_result)

  @parameterized.named_parameters(
      (
          'yuu_uuy',
          [_MESH_DIM_Y, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
      (
          'xuu_uux',
          [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'uux_xuu',
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X],
          [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
      ),
      (
          'xuu_uxu',
          [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'uxu_xuu',
          [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
      ),
      (
          'xuy_uxy',
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
          [layout_lib.UNSHARDED, _MESH_DIM_X, _MESH_DIM_Y],
      ),
      (
          'uxy_xuy',
          [layout_lib.UNSHARDED, _MESH_DIM_X, _MESH_DIM_Y],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
      (
          'xyu_uyx',
          [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, _MESH_DIM_Y, _MESH_DIM_X],
      ),
      # Requires additional transpose
      (
          'uxu_uux',
          [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'uux_uxu',
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X],
          [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'xyu_xuy',
          [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
      (
          'xuy_xyu',
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
          [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
      ),
      (
          'yxu_yux',
          [_MESH_DIM_Y, _MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_Y, layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'yux_yxu',
          [_MESH_DIM_Y, layout_lib.UNSHARDED, _MESH_DIM_X],
          [_MESH_DIM_Y, _MESH_DIM_X, layout_lib.UNSHARDED],
      ),
  )
  def testAllToAll3D(self, src_spec, tgt_spec):
    # [FUNCTION DOC]
    # Same as testAllToAll2D but with an 8x8x8 tensor and rank-3 source/target sharding specs,
    # including cases that need an additional transpose. Asserts values are preserved and the
    # result has the target layout.
    a = constant_op.constant(
        np.arange(8 * 8 * 8).reshape((8, 8, 8)), dtype=dtypes.float32
    )
    sharded_a = numpy_util.pack_numpy(a, layout=Layout(src_spec, self.mesh))

    @polymorphic_function.function
    def func(a):
      # [FUNCTION DOC]
      # Nested helper: relayouts its input to the target layout (from the enclosing test) and
      # returns it.
      return api.relayout(a, Layout(tgt_spec, self.mesh))

    dtensor_result = func(sharded_a)

    self.assertDTensorEqual(a, Layout(tgt_spec, self.mesh), dtensor_result)

  def testExpandDimsDifferentInputAndOutputLayouts(self,):
    # [FUNCTION DOC]
    # Expands a replicated random 10x10 tensor with a trailing axis, using a scalar axis and then a
    # list axis, inside compiled functions that relayout the rank-3 result to first-dimension
    # sharded. Asserts both results equal the plain expand_dims result with that layout.
    src_numpy = np.random.uniform(size=[10, 10])
    src = constant_op.constant(src_numpy, dtype=dtypes.float32)

    expected = array_ops.expand_dims_v2(src, axis=-1)

    src = api.relayout(src, self.replicated_layout_2d)

    @polymorphic_function.function
    def expand_dims_fn(src):
      # [FUNCTION DOC]
      # Nested helper: expands the last axis (scalar axis argument) and relayouts the result to
      # first-dimension sharded rank 3; returns it.
      expanded = array_ops.expand_dims_v2(src, axis=-1)
      return api.relayout(expanded, self.first_dimension_sharded_layout_3d)

    dtensor_result = expand_dims_fn(src)
    self.assertDTensorEqual(expected, self.first_dimension_sharded_layout_3d,
                            dtensor_result)

    @polymorphic_function.function
    def expand_dims_list_axis_fn(src):
      # [FUNCTION DOC]
      # Nested helper: like expand_dims_fn but passes the axis as a one-element list.
      expanded = array_ops.expand_dims_v2(src, axis=[-1])
      return api.relayout(expanded, self.first_dimension_sharded_layout_3d)

    dtensor_result_2 = expand_dims_list_axis_fn(src)
    self.assertDTensorEqual(expected, self.first_dimension_sharded_layout_3d,
                            dtensor_result_2)

  def testPackAndUnpackAssertion(self):
    # [FUNCTION DOC]
    # Packs eight different 1x1x1 components (values 1 to 8) into a replicated rank-3 DTensor, i.e.
    # deliberately incompatible components since pack does not validate them. Converting to numpy
    # picks the last component (8), asserted equal. The stricter DTensor equality assertion is
    # expected to raise an assertion error mentioning mismatched value.
    layout = Layout.replicated(self.mesh, rank=3)
    # Due to Perf concerns, `pack` does not check the compatibility of
    # components and layout. Here, we inject a wrong value components.
    with api.default_mesh(self.mesh):
      b = api.pack(
          [constant_op.constant([[[(x + 1) * 1.0]]]) for x in range(8)],
          layout=layout)
      assert b.shape == [1, 1, 1]

    # `to_numpy` assumes all unpacked tensors are compatible with the
    # layout. So, it picks any component to use if that dimension is replicated.
    # In this case, it picks the final one.
    result_dtensor = numpy_util.to_numpy(b)

    self.assertAllEqual(constant_op.constant([[[8.]]]), result_dtensor)

    # assertDTensorEqual does more aggressive check, which respects the layout.
    with self.assertRaisesRegex(AssertionError, 'Mismatched value'):
      self.assertDTensorEqual(constant_op.constant([[[8.]]]), layout, b)

  @parameterized.named_parameters(test_util_ops.UNARY_OPS)
  def testUnaryOpsWithTwoShardedAndOneReplicatedDimension(self, op):
    # [FUNCTION DOC]
    # Parameterized over the unary ops table. Applies op to a 2x4x1 tensor, then to the same data
    # relayouted to sharded on x, sharded on y, unsharded. Asserts equality and layout, using a
    # tolerance chosen by select_tol (1e-4 for low-precision ops on TPU).
    a = constant_op.constant([[[1.], [2.], [3.], [4.]], [[5.], [6.], [7.],
                                                         [8.]]])
    assert a.shape == [2, 4, 1]
    expected_result = op(a)

    layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED], self.mesh)
    a = api.relayout(a, layout)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(expected_result, layout, dtensor_result, tol=tol)

  @parameterized.named_parameters(test_util_ops.UNARY_OPS)
  def testUnaryOpsWithFullyReplicatedInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over the unary ops table. Applies op to a 2x2 tensor copied to the mesh with a
    # replicated layout and asserts the result matches the plain result and is replicated
    # (tolerance via select_tol).
    a = constant_op.constant([[1., 2.], [3., 4.]])
    assert a.shape == [2, 2]
    expected_result = op(a)

    a = api.copy_to_mesh(a, self.replicated_layout_2d)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(
        expected_result, self.replicated_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(test_util_ops.UNARY_OPS)
  def testUnaryOpsWithFullyShardedInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over the unary ops table. Applies op to a 2x4x2 float tensor relayouted to
    # sharded on x, sharded on y, unsharded; asserts the result matches and keeps that layout.
    a = constant_op.constant(
        np.arange(16).reshape((2, 4, 2)), dtype=dtypes.float32)
    expected_result = op(a)

    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                            self.mesh)
    a = api.relayout(a, sharded_layout)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(
        expected_result, sharded_layout, dtensor_result, tol=tol)

  @parameterized.named_parameters(test_util_ops.UNARY_OPS)
  def testUnaryOpsWithBatchShardedInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over the unary ops table. Applies op to a 2x3 float tensor relayouted to the
    # first-dimension sharded layout; asserts the result matches and keeps that layout (tolerance
    # via select_tol with 1e-3 for low-precision ops).
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-3)
    a = constant_op.constant(np.arange(6).reshape((2, 3)), dtype=dtypes.float32)
    expected_result = op(a)

    a = api.relayout(a, self.first_dimension_sharded_layout)
    dtensor_result = op(a)

    self.assertDTensorEqual(
        expected_result,
        self.first_dimension_sharded_layout,
        dtensor_result,
        tol=tol)

  def testInvertOpsWithFullyShardedInputs(self):
    # [FUNCTION DOC]
    # Applies the bitwise invert op (integer inputs only) to a 2x4x2 int32 tensor relayouted to
    # sharded on x, sharded on y, unsharded; asserts value and layout equality.
    # Invert only support int inputs.
    op = lambda x: gen_bitwise_ops.invert(x=x, name='Invert')

    a = constant_op.constant(
        np.arange(16).reshape((2, 4, 2)), dtype=dtypes.int32)
    expected_result = op(a)

    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                            self.mesh)
    a = api.relayout(a, sharded_layout)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(
        expected_result, sharded_layout, dtensor_result, tol=tol)

  @parameterized.named_parameters(('replicated', layout_lib.UNSHARDED),
                                  ('sharded', _MESH_DIM_X))
  def testInvertPermutationOp(self, shard):
    # [FUNCTION DOC]
    # Parameter shard: either unsharded or the x mesh dimension. Skipped on GPU and TPU (the op
    # runs on CPU only). Inverts a fixed permutation vector given as a DTensor relayouted with the
    # given sharding; the result is expected to be replicated and equal to the plain result.
    self.skipForDeviceType(['GPU', 'TPU'],
                           'Invert Permutation runs in CPU only.')
    op_input = constant_op.constant([3, 4, 0, 2, 1, 5])
    expected_result = gen_array_ops.invert_permutation(op_input)
    # We should always expected the output to be replicated as the
    # expander should relayout both inputs and outputs to replicated.
    expected_layout = Layout.replicated(self.mesh, rank=1)

    self.assertDTensorEqual(
        expected_result,
        expected_layout,
        gen_array_ops.invert_permutation(
            api.relayout(op_input, Layout([shard], self.mesh))
        ),
    )

  def testErfcInvOpsWithFullyShardedInputs(self):
    # [FUNCTION DOC]
    # Applies the raw inverse-error-function op (the public erfcinv is avoided as it routes
    # through ndtri) to a 2x4x2 float tensor with values in (0.1, 0.6], relayouted sharded on x
    # and y; asserts value and layout equality.
    # By official doc, math_ops.erfcinv is defined on (0, 2]. In addition,
    # math_ops.erfcinv internally calls ndtri internally. So to test the op for
    # spmd expanding, we call raw op here.
    op = lambda x: gen_math_ops.erfinv(x=x, name='erfinv')

    a = constant_op.constant(
        np.arange(16).reshape((2, 4, 2)) / 30 + 0.1, dtype=dtypes.float32)
    expected_result = op(a)

    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                            self.mesh)
    a = api.relayout(a, sharded_layout)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(
        expected_result, sharded_layout, dtensor_result, tol=tol)

  def testPopulationCountWithFullyShardedInputs(self):
    # [FUNCTION DOC]
    # Applies the population count (set-bit count) op, integer inputs only, to a 2x4x2 int32 tensor
    # relayouted sharded on x and y; asserts value and layout equality.
    # By official doc, gen_bitwise_ops.population_count only supports int
    # inputs.
    op = lambda x: gen_bitwise_ops.population_count(x=x, name='pc')

    a = constant_op.constant(
        np.arange(16).reshape((2, 4, 2)), dtype=dtypes.int32)
    expected_result = op(a)

    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                            self.mesh)
    a = api.relayout(a, sharded_layout)
    dtensor_result = op(a)

    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, 1e-4)
    self.assertDTensorEqual(
        expected_result, sharded_layout, dtensor_result, tol=tol)

  def testIgammacOpsWithFullyShardedInputs(self):
    # [FUNCTION DOC]
    # Separate test for the upper incomplete gamma op (very low precision on TPU) with shape
    # parameter 4 and a 2x4x2 float input relayouted sharded on x and y; asserts equality with a
    # loose tolerance of 1e-2.
    # Igammac has super low precision on TPU. So we test it as a separated unit
    # tests to avoid lower the tol of other tests.
    #
    # In addition, according to wiki link below, for s=4, all values are not
    # inf/nan.
    #
    # https://en.wikipedia.org/wiki/Incomplete_gamma_function
    tol = 1e-2
    op = lambda x: gen_math_ops.igammac(4, x)

    a = constant_op.constant(
        np.arange(16).reshape((2, 4, 2)), dtype=dtypes.float32)
    expected_result = op(a)

    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                            self.mesh)
    a = api.relayout(a, sharded_layout)
    dtensor_result = op(a)

    self.assertDTensorEqual(
        expected_result, sharded_layout, dtensor_result, tol=tol)

  @parameterized.parameters(('replicated',), ('sharded',))
  def testBiasAdd2D(self, shard_type):
    # [FUNCTION DOC]
    # Parameter shard_type: 'replicated' or 'sharded'. Adds a length-2 bias to a 2x2 value with the
    # value replicated or first-dimension sharded and the bias replicated; asserts result and layout
    # match the plain computation.
    value = np.array([[1., 2.], [3., 4.]])
    bias = np.array([0.1, 0.2])
    expected_result = nn_ops.bias_add(value, bias)

    if shard_type == 'replicated':
      layout = self.replicated_layout_2d
    else:
      layout = self.first_dimension_sharded_layout

    value = api.relayout(value, layout)
    bias = api.relayout(bias, self.replicated_layout_1d)
    dtensor_result = nn_ops.bias_add(value, bias)
    self.assertDTensorEqual(expected_result, layout, dtensor_result)

  @parameterized.product(
      shard_type=['replicated', 'batch_sharded'],
      data_format=['N...C', 'NC...'])
  def testBiasAdd4D(self, shard_type, data_format):
    # [FUNCTION DOC]
    # Parameters: shard_type (replicated or batch sharded) and data_format (channels last or first).
    # Bias-adds a length-2 bias to a ones tensor of shape 6x2x4x2 with the value in the chosen layout
    # and a replicated bias; asserts the result equals the plain result with the same layout.
    value = np.ones(shape=(6, 2, 4, 2), dtype=np.float32)
    bias = np.array([0.1, 0.2], dtype=np.float32)
    expected_result = nn_ops.bias_add(value, bias, data_format=data_format)

    if shard_type == 'replicated':
      layout = Layout.replicated(self.mesh, rank=4)
    else:
      layout = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=4)

    value = api.relayout(value, layout)
    bias = api.relayout(bias, self.replicated_layout_1d)

    dtensor_result = nn_ops.bias_add(value, bias, data_format=data_format)
    self.assertDTensorEqual(expected_result, layout, dtensor_result)

  @parameterized.product(
      data_format=['N...C', 'NC...'],
      bias_sharding=['x', 'y', layout_lib.UNSHARDED],
      c_dim_sharding=['x', layout_lib.UNSHARDED])
  def testBiasAddDataFormatTest(self, data_format, bias_sharding,
                                c_dim_sharding):
    # [FUNCTION DOC]
    # Parameters: data_format (channels last or first), bias_sharding (x, y or unsharded) and
    # c_dim_sharding (sharding of the channel dimension: x or unsharded). Builds a 4-D ones input
    # sharded on y for a spatial dimension and c_dim_sharding for the channel dimension, and a
    # length-4 bias with bias_sharding. The expected output layout equals the input layout, except
    # when the channel dimension is unsharded and the bias is not sharded on y: then the channel
    # dimension takes the bias sharding. Asserts result values and layout.
    if data_format == 'N...C':
      c_dim = 3
      input_sharding = [
          layout_lib.UNSHARDED, layout_lib.UNSHARDED, 'y', c_dim_sharding
      ]
      a = np.ones(shape=(1, 1, 4, 4), dtype=np.float32)
      layout = Layout(input_sharding, self.mesh)
    else:
      c_dim = 1
      input_sharding = [
          layout_lib.UNSHARDED, c_dim_sharding, 'y', layout_lib.UNSHARDED
      ]
      a = np.ones(shape=(1, 4, 4, 1), dtype=np.float32)
      layout = Layout(input_sharding, self.mesh)

    bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    expected_result = nn_ops.bias_add(a, bias, data_format=data_format)
    expected_result_sharding = input_sharding
    if c_dim_sharding == layout_lib.UNSHARDED and bias_sharding != 'y':
      expected_result_sharding[c_dim] = bias_sharding

    expected_layout = Layout(expected_result_sharding, self.mesh)
    a = api.relayout(a, layout)
    bias = api.relayout(bias, Layout([bias_sharding], self.mesh))
    result = nn_ops.bias_add(a, bias=bias, data_format=data_format)

    self.assertDTensorEqual(expected_result, expected_layout, result)

  @parameterized.parameters(('replicated',), ('batch_sharded',))
  def testBiasAddGrad2D(self, shard_type):
    # [FUNCTION DOC]
    # Parameter shard_type: replicated or batch_sharded. Computes the bias-add gradient (sum over
    # the batch dimension) of a 2x2 float input in the chosen layout; asserts the 1-D result equals
    # the plain result and is replicated.
    value = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    expected_result = gen_nn_ops.bias_add_grad(out_backprop=value)

    if shard_type == 'replicated':
      layout = self.replicated_layout_2d
    else:
      layout = self.first_dimension_sharded_layout
    expected_layout = self.replicated_layout_1d

    value = api.relayout(value, layout)
    dtensor_result = gen_nn_ops.bias_add_grad(out_backprop=value)
    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.product(
      shard_type=['replicated', 'batch_sharded'], data_format=['NHWC', 'NCHW'])
  def testBiasAddGrad4D(self, shard_type, data_format):
    # [FUNCTION DOC]
    # Parameters: shard_type (replicated or batch sharded) and data_format (NHWC or NCHW). Computes
    # the bias-add gradient of a ones tensor of shape 2x3x4x5 in the chosen rank-4 layout; asserts
    # the result equals the plain result and is a replicated 1-D tensor.
    value = np.ones(shape=(2, 3, 4, 5), dtype=np.float32)
    expected_result = gen_nn_ops.bias_add_grad(
        out_backprop=value, data_format=data_format)

    if shard_type == 'replicated':
      layout = Layout.replicated(self.mesh, rank=4)
    else:
      layout = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=4)
    expected_layout = self.replicated_layout_1d

    value = api.relayout(value, layout)
    dtensor_result = gen_nn_ops.bias_add_grad(
        out_backprop=value, data_format=data_format)
    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(test_util_ops.BINARY_FLOAT_OPS)
  def testBinaryOpsWithFullyReplicatedInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over float binary ops. Applies op to two 2x2 float tensors copied to the mesh
    # replicated; asserts the result equals the plain one and is replicated (tolerance via
    # select_tol, 1e-2 for low-precision ops on TPU).
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    a = constant_op.constant([[1., 2.], [3., 4.]])
    b = constant_op.constant([[10., 20.], [30., 40.]])
    expected_result = op(a, b)

    a = api.copy_to_mesh(a, self.replicated_layout_2d)
    b = api.copy_to_mesh(b, self.replicated_layout_2d)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, self.replicated_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(test_util_ops.BINARY_FLOAT_OPS)
  def testBinaryFloatOpsWithFullyShardedInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over float binary ops. Applies op to two 2x4 float tensors both relayouted
    # sharded on x and y; asserts values and sharded layout (tolerance via select_tol).
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    a = constant_op.constant(np.arange(8).reshape((2, 4)), dtype=dtypes.float32)
    b = constant_op.constant(
        np.arange(8).reshape((2, 4)) + 10.0, dtype=dtypes.float32)
    expected_result = op(a, b)

    sharded_layout_2d = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    a = api.relayout(a, sharded_layout_2d)
    b = api.relayout(b, sharded_layout_2d)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, sharded_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(test_util_ops.BINARY_BOOL_OPS)
  def testBinaryBoolOpsWithFullyShardedInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over boolean binary ops. Applies op to two 2x4 boolean tensors both relayouted
    # sharded on x and y; asserts equality with the plain result and the sharded layout.
    a = array_ops.reshape(
        constant_op.constant(
            [True, False, True, False, True, False, True, False]), [2, 4])
    b = array_ops.reshape(
        constant_op.constant(
            [True, True, True, True, False, False, False, False]), [2, 4])
    expected_result = op(a, b)

    sharded_layout_2d = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    a = api.relayout(a, sharded_layout_2d)
    b = api.relayout(b, sharded_layout_2d)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(expected_result, sharded_layout_2d, dtensor_result)

  @parameterized.named_parameters(test_util_ops.BINARY_INT_OPS)
  def testBinaryIntOpsWithFullyShardedInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over integer binary ops. Uses int64 inputs (int32 for truncate_mod when a GPU
    # is present) of shape 2x4 sharded on x and y; asserts equality with the plain result and layout.
    dtype = dtypes.int64
    if test_util.is_gpu_present() and op is gen_math_ops.truncate_mod:
      dtype = dtypes.int32

    a = constant_op.constant(np.arange(8).reshape((2, 4)), dtype=dtype)
    b = constant_op.constant(np.arange(8).reshape((2, 4)) + 1, dtype=dtype)
    expected_result = op(a, b)

    sharded_layout_2d = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    a = api.relayout(a, sharded_layout_2d)
    b = api.relayout(b, sharded_layout_2d)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(expected_result, sharded_layout_2d, dtensor_result)

  @parameterized.named_parameters(test_util_ops.BINARY_FLOAT_OPS)
  def testBinaryFloatOpsWithBatchShardedInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over float binary ops. Applies op to two 2x2 float tensors both relayouted to
    # the first-dimension sharded layout; asserts values and layout (tolerance via select_tol).
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.]]), dtype=dtypes.float32)
    b = constant_op.constant(
        np.array([[10., 20.], [30., 40.]]), dtype=dtypes.float32)
    expected_result = op(a, b)

    a = api.relayout(a, self.first_dimension_sharded_layout)
    b = api.relayout(b, self.first_dimension_sharded_layout)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result,
        self.first_dimension_sharded_layout,
        dtensor_result,
        tol=tol)

  @parameterized.named_parameters(test_util_ops.BINARY_INT_OPS)
  def testBinaryIntOpsWithBatchShardedInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over integer binary ops. Applies op to two 2x2 integer tensors (int64, or int32
    # for truncate_mod with a GPU) relayouted to first-dimension sharded; asserts values and layout.
    dtype = dtypes.int64
    if test_util.is_gpu_present() and op is gen_math_ops.truncate_mod:
      dtype = dtypes.int32

    a = constant_op.constant(np.array([[1, 2], [3, 4]]), dtype=dtype)
    b = constant_op.constant(np.array([[5, 6], [7, 4]]), dtype=dtype)
    expected_result = op(a, b)

    a = api.relayout(a, self.first_dimension_sharded_layout)
    b = api.relayout(b, self.first_dimension_sharded_layout)
    dtensor_result = op(a, b)

    self.assertDTensorEqual(expected_result,
                            self.first_dimension_sharded_layout, dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.BINARY_FLOAT_OPS_WITH_BROADCASTING_SUPPORT
  )
  def testBinaryFloatOpsWithFullyReplicatedBroadcastableInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over float binary ops that support broadcasting. Applies op to a scalar and a
    # 2x2 tensor, both copied to the mesh fully replicated (only scalar broadcasting is
    # supported); asserts the result equals the plain one and is replicated.
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    # Currently we only support scalar.
    a = constant_op.constant(23.4)
    b = constant_op.constant([[10., 20.], [30., 40.]])
    expected_result = op(a, b)

    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=a.ndim))
    b = api.copy_to_mesh(b, Layout.replicated(self.mesh, rank=b.ndim))
    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, self.replicated_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(
      test_util_ops.BINARY_INT_OPS_WITH_BROADCASTING_SUPPORT
  )
  def testBinaryIntOpsWithFullyReplicatedBroadcastableInputs(self, op):
    # [FUNCTION DOC]
    # Integer counterpart of the float replicated broadcast test: a scalar and a 2x2 int tensor
    # (ordered by order_broadcastable_operands, which swaps them for truncate div/mod) copied to
    # the mesh replicated; asserts equality with the plain result and a replicated layout.
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    # Currently we only support scalar.
    a = constant_op.constant(3)
    b = constant_op.constant([[0, 1], [2, 3]])
    a, b = order_broadcastable_operands(op, a, b)
    expected_result = op(a, b)

    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=a.ndim))
    b = api.copy_to_mesh(b, Layout.replicated(self.mesh, rank=b.ndim))
    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, self.replicated_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(
      test_util_ops.BINARY_FLOAT_OPS_WITH_BROADCASTING_SUPPORT
  )
  def testBinaryOpsWithFullyShardedBroadcastableInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over broadcasting float binary ops. A scalar (replicated) is combined with a
    # 2x4 tensor sharded on x and y; asserts values and that the output keeps the sharded layout.
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    # Currently we only support scalar.
    a = constant_op.constant(23.4)
    b = constant_op.constant(
        10.0 * np.arange(8).reshape((2, 4)), dtype=dtypes.float32)
    expected_result = op(a, b)

    a = api.copy_to_mesh(a, self.scalar_replicated_layout)
    sharded_layout_2d = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    b = api.relayout(b, sharded_layout_2d)

    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, sharded_layout_2d, dtensor_result, tol=tol)

  @parameterized.named_parameters(
      test_util_ops.BINARY_FLOAT_OPS_WITH_BROADCASTING_SUPPORT
  )
  def testBinaryOpsWithBatchShardedBroadcastableInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over broadcasting float binary ops. A replicated scalar is combined with a 2x2
    # tensor in first-dimension sharded layout; asserts values and that layout is kept.
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    # Currently we only support scalar.
    a = constant_op.constant(23.4)
    b = constant_op.constant(
        np.array([[10., 20.], [30., 40.]]), dtype=dtypes.float32)
    expected_result = op(a, b)

    a = api.copy_to_mesh(a, self.scalar_replicated_layout)
    b = api.relayout(b, self.first_dimension_sharded_layout)

    dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result,
        self.first_dimension_sharded_layout,
        dtensor_result,
        tol=tol)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          [
              {
                  'testcase_name': 'Concat',
                  'op': (lambda v: array_ops.concat(values=v, axis=1)),
              },
              {
                  'testcase_name':
                      'ConcatV1',
                  'op':
                      (lambda v: gen_array_ops.concat(concat_dim=1, values=v)),
              },
              {
                  'testcase_name': 'ConcatV2',
                  'op': (lambda v: gen_array_ops.concat_v2(values=v, axis=1)),
              },
          ],
          [
              {
                  'shard_type': 'replicated',
              },
              {
                  'shard_type': 'sharded',
              },
              {
                  'shard_type': 'mixed',
              },
          ],
      ))
  def testConcatOpSPMD(self, op, shard_type):
    # [FUNCTION DOC]
    # Parameters: op (a concat variant joining a list of tensors on axis 1) and shard_type
    # (replicated, sharded or mixed). Concatenates two 2x2 tensors with layouts depending on
    # shard_type (both replicated; both first-dim sharded; or only the second sharded, with the
    # output sharded) under the default mesh; asserts the result equals the plain result and has
    # the expected output layout.
    layout_a = self.replicated_layout_2d
    layout_b = self.replicated_layout_2d
    layout_output = self.replicated_layout_2d

    if shard_type == 'sharded':
      layout_a = self.first_dimension_sharded_layout
      layout_b = self.first_dimension_sharded_layout
      layout_output = self.first_dimension_sharded_layout
    elif shard_type == 'mixed':
      layout_b = self.first_dimension_sharded_layout
      layout_output = self.first_dimension_sharded_layout

    a = constant_op.constant([[1., 2.], [3., 4.]])
    b = constant_op.constant([[1., 2.], [3., 4.]])
    expected_result = op([a, b])

    with api.default_mesh(self.mesh):
      a = api.relayout(a, layout_a)
      b = api.relayout(b, layout_b)
      c = op([a, b])

    self.assertDTensorEqual(expected_result, layout_output, c)

  @parameterized.named_parameters([{
      'testcase_name': 'ConcatV1',
      'op': (lambda v: gen_array_ops.concat(concat_dim=1, values=v))
  }, {
      'testcase_name': 'ConcatV2',
      'op': (lambda v: gen_array_ops.concat_v2(values=v, axis=1))
  }])
  def testConcatOpShardedOnConcatDim(self, op):
    # [FUNCTION DOC]
    # Parameter op: a concat variant (V1 or V2) on axis 1. Concatenates two 2x2x4 tensors where
    # the first is sharded on x (dim 1) and y (dim 2) and the second on x (dim 0), inside a compiled
    # function. Because an input is sharded on the concat dimension, the output layout has the
    # concat dim and dim 0 unsharded and dim 2 sharded on y. Asserts values and that layout.
    a = constant_op.constant(
        np.arange(16).reshape((2, 2, 4)), dtype=dtypes.float32)
    b = constant_op.constant(
        np.arange(16).reshape((2, 2, 4)), dtype=dtypes.float32)
    expected_result = op([a, b])

    a_layout = Layout([layout_lib.UNSHARDED, _MESH_DIM_X, _MESH_DIM_Y],
                      self.mesh)
    b_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
                      self.mesh)
    # If any input is sharded on the concat dim, then the concat dim is
    # replicated in the output. Dim 0 in the output is replicated because of
    # broadcast compatibility, mesh dimension X is already used in dim 1 of
    # input a.
    output_layout = Layout(
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_Y], self.mesh)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)

    @polymorphic_function.function
    def concat_fn(a, b):
      # [FUNCTION DOC]
      # Nested helper: applies the op under test to the list of the two inputs and returns it.
      return op([a, b])

    dtensor_result = concat_fn(a, b)

    self.assertDTensorEqual(expected_result, output_layout, dtensor_result)

  def testPackWithDifferentInputLayouts(self):
    # [FUNCTION DOC]
    # Stacks (pack op, last axis) two 2x2 tensors, one replicated and one first-dim sharded, inside
    # a compiled function whose output is relayouted to first-dim sharded rank 3. Asserts equality
    # with the plain result and that layout.
    a = constant_op.constant([[1., 2.], [3., 4.]])
    b = constant_op.constant([[1., 2.], [3., 4.]])
    expected_result = gen_array_ops.pack(values=[a, b], axis=-1)

    a = api.relayout(a, self.replicated_layout_2d)
    b = api.relayout(b, self.first_dimension_sharded_layout)

    @polymorphic_function.function
    def pack_fn(a, b):
      # [FUNCTION DOC]
      # Nested helper: packs the two inputs along the last axis and relayouts the result to
      # first-dimension sharded rank 3; returns it.
      c = gen_array_ops.pack(values=[a, b], axis=-1)
      return api.relayout(c, self.first_dimension_sharded_layout_3d)

    dtensor_result = pack_fn(a, b)

    self.assertDTensorEqual(expected_result,
                            self.first_dimension_sharded_layout_3d,
                            dtensor_result)

  @parameterized.named_parameters(test_util_ops.REDUCTION_OPS)
  def testReductionOpsWithFullyReplicatedInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over reduction ops. For axes [0], [1], [0, 1] and none, reduces a replicated 2x2
    # tensor under the default mesh; asserts equality with the plain reduction and a replicated
    # layout (1-D when one axis is reduced, scalar otherwise).
    for axis, expected_layout in [([0], self.replicated_layout_1d),
                                  ([1], self.replicated_layout_1d),
                                  ([0, 1], self.scalar_replicated_layout),
                                  (None, self.scalar_replicated_layout)]:
      # Disable the pylint as the cell var is used for this iteration only.
      # pylint: disable=cell-var-from-loop
      reduction_op = lambda x: op(x, axis=axis)
      # pylint: enable=cell-var-from-loop

      a = constant_op.constant([[1., 2.], [3., 4.]])
      expected_result = reduction_op(a)

      a = api.copy_to_mesh(a, self.replicated_layout_2d)
      with api.default_mesh(self.mesh):
        dtensor_result = reduction_op(a)

      self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(test_util_ops.REDUCTION_OPS)
  def testReductionOpsWithBatchParallelInputs(self, op):
    # [FUNCTION DOC]
    # Parameterized over reduction ops. Reduces a 4x2 float tensor sharded on x along the first
    # dimension over axes [0], [1], [0, 1], none. Expected layouts: replicated 1-D for axis 0,
    # x-sharded 1-D for axis 1, replicated scalar for the others. Asserts values and layouts.
    sharded_layout_1d = Layout([_MESH_DIM_X], self.mesh)
    for axis, expected_layout in [
        (
            [0],
            self.replicated_layout_1d,
        ),
        ([1], sharded_layout_1d),
        (
            [0, 1],
            self.scalar_replicated_layout,
        ),
        (
            None,
            self.scalar_replicated_layout,
        ),
    ]:
      # Disable the pylint as the cell var is used for this iteration only.
      # pylint: disable=cell-var-from-loop
      reduction_op = lambda x: op(x, axis=axis)
      # pylint: enable=cell-var-from-loop

      a = constant_op.constant(
          np.array([[1., 2.], [3., 4.], [5.0, 6.0], [7.0, 8.0]]),
          dtype=dtypes.float32)
      expected_result = reduction_op(a)

      a = api.relayout(a, self.first_dimension_sharded_layout)

      with api.default_mesh(self.mesh):
        dtensor_result = reduction_op(a)

        self.assertDTensorEqual(expected_result, expected_layout,
                                dtensor_result)

  def testReduceLogSumExpWithBatchParallelInputs(self):
    # [FUNCTION DOC]
    # Computes log-sum-exp over the last axis of a 4x2 float tensor sharded on x in its first
    # dimension; asserts equality with the plain result and a first-dim sharded 1-D layout.
    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.], [5.0, 6.0], [7.0, 8.0]]),
        dtype=dtypes.float32)
    expected_result = math_ops.reduce_logsumexp(a, axis=-1)

    a = api.relayout(a, self.first_dimension_sharded_layout)

    with api.default_mesh(self.mesh):
      dtensor_result = math_ops.reduce_logsumexp(a, axis=-1)

      self.assertDTensorEqual(expected_result,
                              self.first_dimension_sharded_layout_1d,
                              dtensor_result)

  @parameterized.named_parameters(test_util_ops.REDUCTION_OPS)
  def testReductionOpsWithBatchParallelInputsWithInt64Dtype(self, op):
    # [FUNCTION DOC]
    # Same scenario as the batch-parallel reduction test but with int64 data. Skipped on TPU
    # (int32 only), and on GPU in multi-device mode for reduce_min and reduce_mean (floats only).
    # Asserts values and the expected layouts for each axis choice.
    self.skipForDeviceType(['TPU'], 'reduce on TPU only supports int32')
    # TODO(b/303662238): Replicate soft placement (or implement these kernels).
    if test_util.use_multi_device_mode() and (
        op is math_ops.reduce_min or op is math_ops.reduce_mean
    ):
      self.skipForDeviceType(['GPU'], 'reduce on GPU only supports floats')

    sharded_layout_1d = Layout([_MESH_DIM_X], self.mesh)
    for axis, expected_layout in [
        (
            [0],
            self.replicated_layout_1d,
        ),
        (
            [1],
            sharded_layout_1d,
        ),
        (
            [0, 1],
            self.scalar_replicated_layout,
        ),
        (
            None,
            self.scalar_replicated_layout,
        ),
    ]:
      # Disable the pylint as the cell var is used for this iteration only.
      # pylint: disable=cell-var-from-loop
      reduction_op = lambda x: op(x, axis=axis)
      # pylint: enable=cell-var-from-loop

      a = constant_op.constant(
          np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), dtype=dtypes.int64)
      expected_result = reduction_op(a)

      # pylint: disable=g-long-lambda
      a = api.relayout(a, self.first_dimension_sharded_layout)

      with api.default_mesh(self.mesh):
        dtensor_result = reduction_op(a)

        self.assertDTensorEqual(expected_result, expected_layout,
                                dtensor_result)

  @parameterized.named_parameters(test_util_ops.REDUCTION_OPS)
  def testReductionOpsWithBatchParallelInputsWithInt32(self, op):
    # [FUNCTION DOC]
    # Same scenario as the batch-parallel reduction test but with int32 data. Skipped on GPU (int64
    # only there). Asserts values and the expected layouts for each axis choice.
    self.skipForDeviceType(['GPU'], 'reduce on GPU only supports int64')

    sharded_layout_1d = Layout([_MESH_DIM_X], self.mesh)
    for axis, expected_layout in [
        (
            [0],
            self.replicated_layout_1d,
        ),
        (
            [1],
            sharded_layout_1d,
        ),
        (
            [0, 1],
            self.scalar_replicated_layout,
        ),
        (
            None,
            self.scalar_replicated_layout,
        ),
    ]:
      # Disable the pylint as the cell var is used for this iteration only.
      # pylint: disable=cell-var-from-loop
      reduction_op = lambda x: op(x, axis=axis)
      # pylint: enable=cell-var-from-loop

      a = constant_op.constant(
          np.array([[1, 2], [3, 4], [5, 6], [7, 8]]), dtype=dtypes.int32)
      expected_result = reduction_op(a)

      # pylint: disable=g-long-lambda
      a = api.relayout(a, self.first_dimension_sharded_layout)

      with api.default_mesh(self.mesh):
        dtensor_result = reduction_op(a)

        self.assertDTensorEqual(expected_result, expected_layout,
                                dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          test_util_ops.REDUCTION_OPS,
          [
              {
                  'dtype': dtypes.float32
              },
              {
                  'dtype': dtypes.int32
              },
          ],
      ))
  def testReductionOpsWithReplicatedWithDtypes(self, op, dtype):
    # [FUNCTION DOC]
    # Parameters: reduction op and dtype (float32 or int32). Skipped on GPU (int32 segfault).
    # Reduces axis 0 of a replicated 4x2 tensor; asserts equality and a replicated 1-D layout.
    self.skipForDeviceType(['GPU'], 'b/169353279: int32 caused segfault on GPU')

    axis = [0]
    # Disable the pylint as the cell var is used for this iteration only.
    # pylint: disable=cell-var-from-loop
    reduction_op = lambda x: op(x, axis=axis)
    # pylint: enable=cell-var-from-loop

    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.], [5.0, 6.0], [7.0, 8.0]]), dtype=dtype)
    expected_result = reduction_op(a)
    a = api.relayout(a, self.replicated_layout_2d)

    with api.default_mesh(self.mesh):
      dtensor_result = reduction_op(a)

      self.assertDTensorEqual(expected_result, self.replicated_layout_1d,
                              dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          test_util_ops.REDUCTION_OPS,
          [
              {
                  'dtype': dtypes.float32
              },
              {
                  'dtype': dtypes.int32
              },
          ],
      ))
  def testReductionOpsWithBatchShardingWithDTypes(self, op, dtype):
    # [FUNCTION DOC]
    # Parameters: reduction op and dtype (float32 or int32). Skipped on GPU. Reduces axis 1 of a 4x2
    # tensor sharded on x in dim 0; asserts equality and a first-dim sharded 1-D layout.
    self.skipForDeviceType(['GPU'], 'b/169353279: int32 caused segfault on GPU')

    axis = [1]
    # Disable the pylint as the cell var is used for this iteration only.
    # pylint: disable=cell-var-from-loop
    reduction_op = lambda x: op(x, axis=axis)
    # pylint: enable=cell-var-from-loop

    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.], [5.0, 6.0], [7.0, 8.0]]), dtype=dtype)
    expected_result = reduction_op(a)
    a = api.relayout(a, self.first_dimension_sharded_layout)

    with api.default_mesh(self.mesh):
      dtensor_result = reduction_op(a)

      self.assertDTensorEqual(expected_result,
                              self.first_dimension_sharded_layout_1d,
                              dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          test_util_ops.REDUCTION_OPS,
          [
              {
                  'axis': [0, 1],
                  'dtype': dtypes.float32
              },
              {
                  'axis': [0, 1],
                  'dtype': dtypes.int32
              },
              {
                  'axis': None,
                  'dtype': dtypes.float32
              },
              {
                  'axis': None,
                  'dtype': dtypes.int32
              },
          ],
      ))
  def testReductionOpsWithReplicatedLayoutAndDTypes(self, op, axis, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says replicated layout but the input is relayouted to first-dim sharded.
    # Parameters: reduction op, axis (all axes or none) and dtype (float32 or int32). Skipped on
    # GPU. Reduces over all dimensions of a 4x2 tensor sharded on x; asserts equality with the plain
    # result and a replicated scalar layout.
    self.skipForDeviceType(['GPU'], 'b/169353279: int32 caused segfault on GPU')

    # Disable the pylint as the cell var is used for this iteration only.
    # pylint: disable=cell-var-from-loop
    reduction_op = lambda x: op(x, axis=axis)
    # pylint: enable=cell-var-from-loop

    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.], [5.0, 6.0], [7.0, 8.0]]), dtype=dtype)
    expected_result = reduction_op(a)
    a = api.relayout(a, self.first_dimension_sharded_layout)

    with api.default_mesh(self.mesh):
      dtensor_result = reduction_op(a)

      self.assertDTensorEqual(expected_result, self.scalar_replicated_layout,
                              dtensor_result)

  @parameterized.named_parameters(
      test_util_ops.expand_test_config(
          [
              {
                  'testcase_name': 'FullyReplicatedInputs',
                  'shard_type': 'replicated',
              },
              {
                  'testcase_name': 'BatchShardedInputs',
                  'shard_type': 'batch_sharded',
              },
          ],
          [
              {
                  'axis': -1,
              },
              {
                  'axis': 0,
              },
              {
                  'axis': 1,
              },
          ],
      )
  )
  def testOneHotSPMDWith(self, shard_type, axis):
    # [FUNCTION DOC]
    # Parameters: shard_type (replicated or batch_sharded indices) and axis (-1, 0, 1). Skips every
    # axis other than -1 (known layout propagation issue). Builds one-hot of 2x2 int32 indices with
    # depth 10 and asserts equality with the plain result and a replicated or batch-sharded rank-3
    # output layout (a code branch for axis 0 with batch sharding is unreachable because of the skip).
    if axis != -1:
      self.skipTest('b/177569789: fix this test with layout propagation v2')

    indices = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.int32)
    depth = constant_op.constant(10, dtype=dtypes.int32)
    indices_layout = (
        self.replicated_layout_2d
        if shard_type == 'replicated' else self.first_dimension_sharded_layout)
    output_layout = (
        Layout.replicated(self.mesh, rank=3) if shard_type == 'replicated' else
        Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=3))

    expected_result = array_ops.one_hot(indices, depth, axis=axis)

    indices = api.relayout(indices, indices_layout)
    depth = api.copy_to_mesh(depth, self.scalar_replicated_layout)
    dtensor_result = array_ops.one_hot(indices, depth, axis=axis)
    if axis == 0 and shard_type == 'batch_sharded':
      output_layout = self.middle_dimension_sharded_layout_3d

    self.assertDTensorEqual(expected_result, output_layout, dtensor_result)

  def testOneHotSPMDWithDifferentLayout(self):
    # [FUNCTION DOC]
    # Builds one-hot (depth 10, axis 2) of replicated 2x2 int32 indices inside a compiled function
    # that relayouts the rank-3 result to first-dim sharded. Asserts equality and that layout.
    indices = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.int32)
    depth = constant_op.constant(10, dtype=dtypes.int32)
    expected_result = array_ops.one_hot(indices, depth, axis=2)

    indices = api.relayout(indices, self.replicated_layout_2d)

    depth = api.copy_to_mesh(depth, self.scalar_replicated_layout)

    @polymorphic_function.function
    def one_hot_fn(indices, depth):
      # [FUNCTION DOC]
      # Nested helper: one-hot encodes the indices with the given depth on axis 2 and relayouts the
      # result to first-dimension sharded rank 3; returns it.
      result = array_ops.one_hot(indices, depth, axis=2)
      return api.relayout(result, self.first_dimension_sharded_layout_3d)

    dtensor_result = one_hot_fn(indices, depth)

    self.assertDTensorEqual(expected_result,
                            self.first_dimension_sharded_layout_3d,
                            dtensor_result)

  def testL2LossOpsWithFullyReplicatedInputs(self):
    # [FUNCTION DOC]
    # Computes the L2 loss of a 2x2 tensor copied to the mesh replicated; asserts equality with
    # the plain scalar result and a replicated scalar layout.
    loss_op = gen_nn_ops.l2_loss
    a = constant_op.constant([[1., 2.], [3., 4.]])
    expected_result = loss_op(a)
    expected_layout = self.scalar_replicated_layout

    a = api.copy_to_mesh(a, self.replicated_layout_2d)
    dtensor_result = loss_op(a)

    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  def testL2LossOpsWithFullyShardedInputs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, the input is only first-dimension sharded on x.
    # Computes the L2 loss of a 2x2 tensor relayouted first-dim sharded; asserts equality with the
    # plain result and a replicated scalar layout (the reduction output is replicated).
    loss_op = gen_nn_ops.l2_loss
    a = constant_op.constant([[1., 2.], [3., 4.]])
    expected_result = loss_op(a)
    expected_layout = self.scalar_replicated_layout

    a = api.relayout(a, self.first_dimension_sharded_layout)
    dtensor_result = loss_op(a)
    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(test_util_ops.EXPANSION_OPS)
  def testExpansionOpsReplicatedLayout(self, inputs, op):
    # [FUNCTION DOC]
    # KNOWN ISSUE: always skipped (layout propagation v2 issue), so the body never runs.
    # Parameters: inputs (callable producing the op arguments) and op. Intended scenario: copy all
    # arguments to the mesh replicated, run op under a replicated default output layout and compare
    # with the plain result.
    self.skipTest('b/177569789: fix this test with layout propagation v2')

    global_op_args = inputs()
    expected_result = op(*global_op_args)

    with api.default_mesh(self.mesh):
      dtensor_op_args = inputs()

      def _broadcast_to_replicated(x):
        # [FUNCTION DOC]
        # HELPER: converts an argument to a tensor and copies it to the mesh with a fully replicated
        # layout of the same rank; returns the DTensor.
        x = constant_op.constant(x)
        return api.copy_to_mesh(
            x, Layout.replicated(self.mesh, rank=x.shape.ndims))

      dtensor_op_args = nest.map_structure(_broadcast_to_replicated,
                                           dtensor_op_args)

      with api._dtensor_device()._default_layout(self.replicated_layout_2d):
        dtensor_result = op(*dtensor_op_args)
    self.assertDTensorEqual(expected_result, self.replicated_layout_2d,
                            dtensor_result)

  @parameterized.named_parameters(test_util_ops.EXPANSION_OPS)
  def testExpansionOpsFullySharded(self, inputs, op):
    # [FUNCTION DOC]
    # KNOWN ISSUE: always skipped (layout propagation v2 issue), so the body never runs.
    # Intended scenario: like the replicated variant but with the default output layout sharded on
    # x and y.
    self.skipTest('b/177569789: fix this test with layout propagation v2')

    global_op_args = inputs()
    expected_result = op(*global_op_args)

    with api.default_mesh(self.mesh):
      dtensor_op_args = inputs()

      def _broadcast_to_replicated(x):
        # [FUNCTION DOC]
        # HELPER: converts an argument to a tensor and copies it to the mesh with a fully replicated
        # layout of the same rank; returns the DTensor.
        x = constant_op.constant(x)
        return api.copy_to_mesh(
            x, Layout.replicated(self.mesh, rank=x.shape.ndims))

      dtensor_op_args = nest.map_structure(_broadcast_to_replicated,
                                           dtensor_op_args)
      sharded_layout_2d = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
      with api._dtensor_device()._default_layout(sharded_layout_2d):
        dtensor_result = op(*dtensor_op_args)

    self.assertDTensorEqual(expected_result, sharded_layout_2d, dtensor_result)

  @parameterized.named_parameters(test_util_ops.EXPANSION_OPS)
  def testExpansionOpsBatchSharded(self, inputs, op):
    # [FUNCTION DOC]
    # KNOWN ISSUE: always skipped (layout propagation v2 issue), so the body never runs.
    # Intended scenario: like the replicated variant but with the default output layout sharded
    # on x in the first dimension.
    self.skipTest('b/177569789: fix this test with layout propagation v2')

    global_op_args = inputs()
    expected_result = op(*global_op_args)

    first_d_shard_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED],
                                  self.mesh)

    with api.default_mesh(self.mesh):
      dtensor_op_args = inputs()

      def _broadcast_to_replicated(x):
        # [FUNCTION DOC]
        # HELPER: converts an argument to a tensor and copies it to the mesh with a fully replicated
        # layout of the same rank; returns the DTensor.
        x = constant_op.constant(x)
        return api.copy_to_mesh(
            x, Layout.replicated(self.mesh, rank=x.shape.ndims))

      dtensor_op_args = nest.map_structure(_broadcast_to_replicated,
                                           dtensor_op_args)

      with api._dtensor_device().default_layout(first_d_shard_layout):
        dtensor_result = op(*dtensor_op_args)

    self.assertDTensorEqual(expected_result, first_d_shard_layout,
                            dtensor_result)

  def testSliceOpsWithFullyReplicatedInputs(self):
    # [FUNCTION DOC]
    # Slices a replicated 2x4 tensor (all rows, first two columns) under the default mesh; asserts
    # equality with the plain slice and a replicated layout.
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops.slice(t, [0, 0], [-1, 2])

    a = api.copy_to_mesh(t, self.replicated_layout_2d)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.slice(a, [0, 0], [-1, 2])

    self.assertDTensorEqual(expected_result, self.replicated_layout_2d,
                            dtensor_result)

  @parameterized.named_parameters(('_minus_one_size', -1), ('_pos_size', 2))
  def testSliceOpsWithFullSlicingOnShardedInputs(self, size):
    # [FUNCTION DOC]
    # Parameter size: slice length along dimension 0 (-1 meaning everything, or 2, the full
    # extent). Slices a first-dim sharded 2x4 tensor (first two columns); asserts equality with the
    # plain slice and that the sharded layout is kept.
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops.slice(t, [0, 0], [size, 2])
    sharded_layout = self.first_dimension_sharded_layout

    t = api.relayout(t, sharded_layout)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.slice(t, [0, 0], [size, 2])

    self.assertDTensorEqual(expected_result, sharded_layout, dtensor_result)

  def testSliceOpsWithDynamicBeginFullSlicingOnShardedInputs(self):
    # [FUNCTION DOC]
    # Slices a first-dim sharded 2x4 tensor with a begin vector passed as a (replicated) tensor
    # argument of a compiled function, with fixed size 2x2. Asserts equality with the plain result
    # and the sharded layout.
    tensor = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    begins = constant_op.constant([0, 0], dtype=dtypes.int32)

    @polymorphic_function.function
    def slice_fn(tensor, begins):
      # [FUNCTION DOC]
      # Nested helper: slices the tensor starting at the begins vector with size 2x2; returns it.
      return array_ops.slice(tensor, begins, [2, 2])

    expected_result = slice_fn(tensor, begins)

    sharded_layout = self.first_dimension_sharded_layout

    tensor = api.relayout(tensor, sharded_layout)
    begins = api.relayout(begins, self.replicated_layout_1d)

    dtensor_result = slice_fn(tensor, begins)

    self.assertDTensorEqual(expected_result, sharded_layout, dtensor_result)

  @parameterized.named_parameters(
      (
          'FullyReplicatedInputs',
          {'begin': [0, 0], 'end': [-1, 2], 'strides': [1, 2]},
          [layout_lib.UNSHARDED] * 2,
      ),
      (
          'NewAxisMask',
          {
              'begin': [0, 0, 0, 0],
              'end': [0, 0, 2, 4],
              'strides': [1, 1, 1, 1],
              'new_axis_mask': 3,
          },
          [layout_lib.UNSHARDED] * 2,
          [layout_lib.UNSHARDED] * 4,
      ),
      (
          'ShrinkAxisMask',
          {
              'begin': [0, 0],
              'end': [-1, 2],
              'strides': [1, 1],
              'shrink_axis_mask': 2,
          },
          [layout_lib.UNSHARDED] * 2,
          [layout_lib.UNSHARDED],
      ),
      (
          'EllipsisAxisMask',
          {
              'begin': [0, 0, 0],
              'end': [0, 0, 0],
              'strides': [1, 1, 1],
              'ellipsis_mask': 1,
              'new_axis_mask': 6,
          },
          [layout_lib.UNSHARDED] * 2,
          [layout_lib.UNSHARDED] * 4,
      ),
      (
          'MoreAxis',
          {
              'begin': [0],
              'end': [2],
              'strides': [1],
          },
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'ShardingOnNonSlicedDimension',
          {'begin': [0, 0], 'end': [2, 2], 'strides': [1, 2]},
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'StrideOnShardedDimensionNoRelayout1',
          {'begin': [0, 0], 'end': [2, 4], 'strides': [1, 2]},
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'StrideOnShardedDimensionNoRelayout2',
          {'begin': [0, 1], 'end': [2, 4], 'strides': [1, 2]},
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'StrideOnShardedDimensionNoRelayout3',
          {'begin': [0, 0], 'end': [2, 3], 'strides': [1, 2]},
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'StrideOnShardedDimensionNeedRelayout',
          {'begin': [0, 0], 'end': [-1, 4], 'strides': [1, 3]},
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED] * 2,
      ),
      (
          'DynamicSliceWithBeginEndMask',
          {
              'begin': lambda: array_ops.fill([2], 0),
              'end': [-1, 4],
              'strides': [1, 3],
              'begin_mask': 3,
              'end_mask': 3,
          },
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'DynamicSliceNoMask',
          {
              'begin': lambda: array_ops.fill([2], 0),
              'end': [-1, 4],
              'strides': [1, 3],
              'begin_mask': 0,
              'end_mask': 0,
          },
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED],
      ),
  )
  def testStridedSliceOps(self, args, input_layout, expected_layout=None):
    # [FUNCTION DOC]
    # Parameters: args (mapping of strided-slice arguments such as begin, end, strides and masks;
    # values may be callables evaluated inside the traced function), input_layout (sharding spec
    # list of the input), expected_layout (optional sharding spec of the output; defaults to the
    # input layout). Runs a strided slice of a 2x4 float tensor plainly, then on a DTensor
    # relayouted to input_layout, and asserts equal values and the expected layout.
    input_tensor = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])

    @polymorphic_function.function
    def func(input_tensor):
      # [FUNCTION DOC]
      # Nested helper: evaluates callable argument values, then applies the raw strided slice op to
      # the input with those arguments; returns the slice.
      newargs = {}
      for key, value in args.items():
        newargs[key] = value() if hasattr(value, '__call__') else value

      return gen_array_ops.strided_slice(input=input_tensor, **newargs)

    expected_result = func(input_tensor)

    input_layout = Layout(input_layout, self.mesh)
    if expected_layout is None:
      expected_layout = input_layout
    else:
      expected_layout = Layout(expected_layout, self.mesh)

    dtensor_input_tensor = api.relayout(input_tensor, input_layout)
    dtensor_result = func(dtensor_input_tensor)

    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      (
          'FullyReplicatedInputs',
          {'begin': [0, 0], 'end': [-1, 2], 'strides': [1, 2]},
          [layout_lib.UNSHARDED] * 2,
      ),
      (
          'NewAxisMask',
          {
              'begin': [0, 0, 0, 0],
              'end': [0, 0, 2, 4],
              'strides': [1, 1, 1, 1],
              'new_axis_mask': 3,
          },
          [layout_lib.UNSHARDED] * 2,
          [layout_lib.UNSHARDED] * 4,
      ),
      (
          'ShrinkAxisMask',
          {
              'begin': [0, 0],
              'end': [-1, 2],
              'strides': [1, 1],
              'shrink_axis_mask': 2,
          },
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED],
      ),
      (
          'EllipsisAxisMask',
          {
              'begin': [0, 0, 0],
              'end': [0, 0, 0],
              'strides': [1, 1, 1],
              'ellipsis_mask': 1,
              'new_axis_mask': 6,
          },
          [layout_lib.UNSHARDED] * 2,
          [layout_lib.UNSHARDED] * 4,
      ),
      (
          'MoreAxis',
          {
              'begin': [0],
              'end': [2],
              'strides': [1],
          },
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'ShardingOnNonSlicedDimension',
          {'begin': [0, 0], 'end': [2, 2], 'strides': [1, 2]},
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'StrideOnShardedDimensionNoRelayout1',
          {'begin': [0, 0], 'end': [2, 4], 'strides': [1, 2]},
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'StrideOnShardedDimensionNoRelayout2',
          {'begin': [0, 1], 'end': [2, 4], 'strides': [1, 2]},
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'StrideOnShardedDimensionNoRelayout3',
          {'begin': [0, 0], 'end': [2, 3], 'strides': [1, 2]},
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'StrideOnShardedDimensionNeedRelayout',
          {'begin': [0, 0], 'end': [-1, 4], 'strides': [1, 3]},
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED] * 2,
      ),
      (
          'DynamicSliceWithBeginEndMask',
          {
              'begin': lambda: array_ops.fill([2], 0),
              'end': [-1, 4],
              'strides': [1, 3],
              'begin_mask': 3,
              'end_mask': 3,
          },
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'DynamicSliceNoMask',
          {
              'begin': lambda: array_ops.fill([2], 0),
              'end': [-1, 4],
              'strides': [1, 3],
              'begin_mask': 0,
              'end_mask': 0,
          },
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED],
      ),
  )
  def testStridedSliceGradOps(self, args, expected_layout, value_layout=None):
    # [FUNCTION DOC]
    # Parameters: args (mapping of strided-slice arguments; callable values are evaluated),
    # expected_layout (sharding spec of the gradient output), value_layout (optional sharding spec
    # of the incoming gradient; defaults to expected_layout). Slices a 2x4 tensor to make a
    # gradient, runs the strided-slice gradient op on it plainly and on a DTensor relayouted to
    # value_layout; asserts equal values and the expected layout.
    input_tensor = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    shape = input_tensor.shape.as_list()
    expected_layout = Layout(expected_layout, self.mesh)
    if value_layout is None:
      value_layout = expected_layout
    else:
      value_layout = Layout(value_layout, self.mesh)

    def get_newargs():
      # [FUNCTION DOC]
      # Nested helper: returns a copy of args where callable values are replaced by their results.
      newargs = {}
      for key, value in args.items():
        newargs[key] = value() if hasattr(value, '__call__') else value
      return newargs

    @polymorphic_function.function
    def func(grad):
      # [FUNCTION DOC]
      # Nested helper: applies the raw strided-slice gradient op with the input shape, the
      # evaluated arguments and the given gradient; returns the result.
      return gen_array_ops.strided_slice_grad(
          shape=shape, **get_newargs(), dy=grad
      )

    grad = gen_array_ops.strided_slice(input=input_tensor, **get_newargs())
    expected_result = func(grad)

    dtensor_grad = api.relayout(grad, value_layout)
    dtensor_result = func(dtensor_grad)

    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      (
          'FullyReplicatedInputs',
          {'begin': [0, 0], 'end': [-1, 2], 'strides': [1, 2]},
          [layout_lib.UNSHARDED] * 2,
          [layout_lib.UNSHARDED] * 2,
      ),
      (
          'NewAxisMask',
          {
              'begin': [0, 0, 0, 0],
              'end': [0, 0, 2, 4],
              'strides': [1, 1, 1, 1],
              'new_axis_mask': 3,
          },
          [layout_lib.UNSHARDED] * 2,
          [layout_lib.UNSHARDED] * 4,
      ),
      (
          'ShrinkAxisMask',
          {
              'begin': [0, 0],
              'end': [-1, 2],
              'strides': [1, 1],
              'shrink_axis_mask': 2,
          },
          [layout_lib.UNSHARDED] * 2,
          [layout_lib.UNSHARDED],
      ),
      (
          'EllipsisAxisMask',
          {
              'begin': [0, 0, 0],
              'end': [0, 0, 0],
              'strides': [1, 1, 1],
              'ellipsis_mask': 1,
              'new_axis_mask': 6,
          },
          [layout_lib.UNSHARDED] * 2,
          [layout_lib.UNSHARDED] * 4,
          [layout_lib.UNSHARDED] * 2,
      ),
      (
          'MoreAxis',
          {
              'begin': [0],
              'end': [2],
              'strides': [1],
          },
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'ShardingOnNonSlicedDimension',
          {'begin': [0, 0], 'end': [2, 2], 'strides': [1, 2]},
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'StrideOnShardedDimensionNoRelayout1',
          {'begin': [0, 0], 'end': [2, 4], 'strides': [1, 2]},
          [layout_lib.UNSHARDED, _MESH_DIM_X],
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'StrideOnShardedDimensionNoRelayout2',
          {'begin': [0, 1], 'end': [2, 4], 'strides': [1, 2]},
          [layout_lib.UNSHARDED, _MESH_DIM_X],
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'StrideOnShardedDimensionNoRelayout3',
          {'begin': [0, 0], 'end': [2, 3], 'strides': [1, 2]},
          [layout_lib.UNSHARDED, _MESH_DIM_X],
          [layout_lib.UNSHARDED, _MESH_DIM_X],
      ),
      (
          'StrideOnShardedDimensionNeedRelayout',
          {'begin': [0, 0], 'end': [-1, 4], 'strides': [1, 3]},
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED] * 2,
          [layout_lib.UNSHARDED] * 2,
      ),
      (
          'DynamicSliceWithBeginEndMask',
          {
              'begin': lambda: array_ops.fill([2], 0),
              'end': [-1, 4],
              'strides': [1, 3],
              'begin_mask': 3,
              'end_mask': 3,
          },
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED],
      ),
      (
          'DynamicSliceNoMask',
          {
              'begin': lambda: array_ops.fill([2], 0),
              'end': [-1, 4],
              'strides': [1, 3],
              'begin_mask': 0,
              'end_mask': 0,
          },
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [layout_lib.UNSHARDED, layout_lib.UNSHARDED],
      ),
  )
  def testStridedSliceUpdateOps(self,
                                args,
                                input_layout,
                                value_layout,
                                expected_layout=None):
    # [FUNCTION DOC]
    # Parameters: args (mapping of strided-slice arguments; callable values are evaluated),
    # input_layout and value_layout (sharding specs of the base tensor and of the update values),
    # expected_layout (optional; defaults to input layout). Skipped on TPU (no XLA implementation).
    # Builds update values by slicing a 2x4 tensor and scaling by ten, runs the strided slice update
    # op plainly and on relayouted DTensors; asserts equal values and the expected layout.
    self.skipForDeviceType(['TPU'], 'b/123559667; op has no XLA implementation')
    input_tensor = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])

    def get_newargs():
      # [FUNCTION DOC]
      # Nested helper: returns a copy of args where callable values are replaced by their results.
      newargs = {}
      for key, value in args.items():
        newargs[key] = value() if hasattr(value, '__call__') else value
      return newargs

    value_tensor = (
        gen_array_ops.strided_slice(input=input_tensor, **get_newargs()) * 10.0
    )

    @polymorphic_function.function
    def func(input_tensor, value_tensor):
      # [FUNCTION DOC]
      # Nested helper: applies the raw tensor strided-slice update op, writing the value tensor into
      # the input using the evaluated arguments; returns the updated tensor.
      return gen_array_ops.tensor_strided_slice_update(
          input=input_tensor, value=value_tensor, **get_newargs()
      )

    expected_result = func(input_tensor, value_tensor)

    input_layout = Layout(input_layout, self.mesh)
    value_layout = Layout(value_layout, self.mesh)
    if expected_layout is None:
      expected_layout = input_layout
    else:
      expected_layout = Layout(expected_layout, self.mesh)

    dtensor_input_tensor = api.relayout(input_tensor, input_layout)
    dtensor_value_tensor = api.relayout(value_tensor, value_layout)

    dtensor_result = func(dtensor_input_tensor, dtensor_value_tensor)
    self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  def testBroadcastGradientArgs(self):
    # [FUNCTION DOC]
    # Calls the broadcast-gradient-args op on two equal shape vectors (128, 10) plainly and with
    # both copied to the mesh replicated; asserts both outputs match and are replicated 1-D.
    a = constant_op.constant([128, 10])
    b = constant_op.constant([128, 10])
    ea, eb = gen_array_ops.broadcast_gradient_args(s0=a, s1=b)

    a = api.copy_to_mesh(a, self.replicated_layout_1d)
    b = api.copy_to_mesh(b, self.replicated_layout_1d)
    da, db = gen_array_ops.broadcast_gradient_args(s0=a, s1=b)

    self.assertDTensorEqual(ea, self.replicated_layout_1d, da)
    self.assertDTensorEqual(eb, self.replicated_layout_1d, db)

  def _transpose_shape(self, transpose, shape):
    # [FUNCTION DOC]
    # HELPER: shape preparation for the matmul tests.
    # Inputs: transpose (boolean) and shape (list of integers). If transpose is true, swaps the
    # last two entries of shape in place. Returns the same (possibly mutated) list.
    if transpose:
      shape[-1], shape[-2] = shape[-2:]
    return shape

  def _merge_layouts_for_matmul(self, layout_a, layout_b, transpose_a,
                                transpose_b):
    # [FUNCTION DOC]
    # HELPER: computes the expected output layout of a (batch) matmul for the matmul tests.
    # Inputs: layout_a and layout_b (layouts of the operands on the same mesh) and transpose_a /
    # transpose_b (booleans). Pads the shorter sharding spec with unsharded entries on the left,
    # swaps the last two entries of a transposed operand, takes the batch dimension sharding from b
    # unless b is unsharded there (then from a), appends a's row sharding and b's column sharding,
    # unshards both of those if they use the same mesh dimension, and unshards any batch dimension
    # whose mesh dimension is also used by one of the last two dimensions of either operand.
    # Returns a Layout on the mesh of layout_a. Performs no error checking.
    # This merge does no error checking and assumes that mesh dimensions
    # are compatible and that layout_a and b are on the same mesh.
    # Prepend enough layout_lib.UNSHARDED to give both lists the same size.
    a_sharding_spec = (
        [layout_lib.UNSHARDED] * max(0, layout_b.rank - layout_a.rank) +
        layout_a.sharding_specs)
    b_sharding_spec = (
        [layout_lib.UNSHARDED] * max(0, layout_a.rank - layout_b.rank) +
        layout_b.sharding_specs)
    if transpose_a:
      a_sharding_spec[-1], a_sharding_spec[-2] = a_sharding_spec[-2:]
    if transpose_b:
      b_sharding_spec[-1], b_sharding_spec[-2] = b_sharding_spec[-2:]

    def _get_mesh_dim(i):
      # [FUNCTION DOC]
      # HELPER: for batch dimension i, returns a's sharding when b is unsharded there, else b's.
      if b_sharding_spec[i] == layout_lib.UNSHARDED:
        return a_sharding_spec[i]
      return b_sharding_spec[i]

    final_layout = [_get_mesh_dim(i) for i in range(len(a_sharding_spec) - 2)]
    final_layout.append(a_sharding_spec[-2])
    final_layout.append(b_sharding_spec[-1])
    if final_layout[-2] == final_layout[-1]:
      final_layout[-2] = layout_lib.UNSHARDED
      final_layout[-1] = layout_lib.UNSHARDED
    for i in range(len(final_layout) - 2):
      if (final_layout[i] == a_sharding_spec[-2] or
          final_layout[i] == a_sharding_spec[-1] or
          final_layout[i] == b_sharding_spec[-2] or
          final_layout[i] == b_sharding_spec[-1]):
        final_layout[i] = layout_lib.UNSHARDED

    return Layout(final_layout, layout_a.mesh)

  @parameterized.named_parameters(*test_util.product(_MATMUL_IMPLEMENTED,
                                                     _MATMUL_TRANSPOSE))
  def testMatMul(self, a_layout, b_layout, transpose_a, transpose_b):
    # [FUNCTION DOC]
    # Parameters: a_layout and b_layout (integer indexes 0 to 2 into the list of 2-D layouts:
    # replicated, first-dim sharded, last-dim sharded), transpose_a and transpose_b (booleans).
    # When an operand is transposed, a non-zero layout index is swapped (1 <-> 2) so test names
    # keep meaning. Multiplies random 4x8 and 8x12 float32 matrices (shapes transposed as needed)
    # plainly and as relayouted DTensors; asserts equal values and the merged expected layout.
    # Swap layout 1 and 2, so that test name is correct (contracting and
    # non_contracting dims switch when transposed).
    if transpose_a and a_layout > 0:
      a_layout = 3 - a_layout
    if transpose_b and b_layout > 0:
      b_layout = 3 - b_layout
    a_layout = self.layouts_2d[a_layout]
    b_layout = self.layouts_2d[b_layout]
    a_numpy = np.random.uniform(size=self._transpose_shape(transpose_a, [4, 8]))
    b_numpy = np.random.uniform(
        size=self._transpose_shape(transpose_b, [8, 12]))
    a = constant_op.constant(a_numpy, dtype=dtypes.float32)
    b = constant_op.constant(b_numpy, dtype=dtypes.float32)

    expected = math_ops.matmul(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)
    dtensor_result = math_ops.matmul(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    expected_layout = self._merge_layouts_for_matmul(a_layout, b_layout,
                                                     transpose_a, transpose_b)
    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(*test_util.product(_BATCH_MATMUL_IMPLEMENTED,
                                                     _MATMUL_TRANSPOSE))
  def testBatchMatMul(self, a_layout, b_layout, transpose_a, transpose_b):
    # [FUNCTION DOC]
    # Parameters: a_layout and b_layout (integer indexes 0 to 3 into the list of 3-D layouts),
    # transpose_a and transpose_b (booleans). Index 2 and 3 are swapped when the operand is
    # transposed. Multiplies random 2x4x8 and 2x8x12 float32 tensors (transposed as needed), which
    # emits a batch matmul, plainly and as relayouted DTensors; asserts values and merged layout.
    # Swap layout 2 and 3, so that test name is correct (contracting and
    # non_contracting dims switch when transposed).
    if transpose_a and a_layout > 1:
      a_layout = 5 - a_layout
    if transpose_b and b_layout > 1:
      b_layout = 5 - b_layout
    a_layout = self.layouts_3d[a_layout]
    b_layout = self.layouts_3d[b_layout]
    a_numpy = np.random.uniform(
        size=self._transpose_shape(transpose_a, [2, 4, 8]))
    b_numpy = np.random.uniform(
        size=self._transpose_shape(transpose_b, [2, 8, 12]))
    a = constant_op.constant(a_numpy, dtype=dtypes.float32)  # 2x4x8
    b = constant_op.constant(b_numpy, dtype=dtypes.float32)  # 2x8x12

    # math_ops.matmul should emit a BatchMatMulV2 op here.
    expected = math_ops.matmul(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)
    dtensor_result = math_ops.matmul(
        a, b, transpose_a=transpose_a, transpose_b=transpose_b)
    expected_layout = self._merge_layouts_for_matmul(a_layout, b_layout,
                                                     transpose_a, transpose_b)
    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_a_unsharded_b_unsharded', 0, 0), ('_a_batch_b_unsharded', 1, 0),
      ('_a_non_contracting_b_unsharded', 2, 0),
      ('_a_contracting_b_unsharded', 3, 0),
      ('_a_unsharded_b_non_contracting', 0, 2),
      ('_a_unsharded_b_contracting', 0, 1),
      ('_a_contracting_b_contracting', 3, 1),
      ('_a_contracting_b_non_contracting', 3, 2),
      ('_a_non_contracting_b_non_contracting', 2, 2),
      ('_a_non_contracting_b_contracting', 2, 1),
      ('_a_batch_b_non_contracting', 1, 2), ('_a_batch_b_contracting', 1, 1))
  def testBatchMatMulWithBroadcasting(self, a_layout, b_layout):
    # [FUNCTION DOC]
    # Parameters: a_layout (index into the 3-D layouts) and b_layout (index into the 2-D layouts).
    # Multiplies a random 2x2x4 tensor by a random 4x6 matrix (broadcast batch matmul) plainly and
    # as relayouted DTensors; asserts equal values and the layout from the merge helper.
    a_layout = self.layouts_3d[a_layout]
    b_layout = self.layouts_2d[b_layout]
    a_numpy = np.random.uniform(size=[2, 2, 4])
    b_numpy = np.random.uniform(size=[4, 6])
    a = constant_op.constant(a_numpy, dtype=dtypes.float32)  # 2x2x4
    b = constant_op.constant(b_numpy, dtype=dtypes.float32)  # 4x6

    # math_ops.matmul should emit a BatchMatMulV2 op here.
    expected = math_ops.matmul(a, b)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)
    dtensor_result = math_ops.matmul(a, b)
    expected_layout = self._merge_layouts_for_matmul(a_layout, b_layout, False,
                                                     False)
    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(('_positive_axis_negative_batch', 0, -1),
                                  ('_negative_axis_positive_batch', -2, 0))
  def testGather(self, axis, batch_dims):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the case names mention positive/negative batch while the parameters are an axis
    # and a batch_dims value; the comment about batch size 2 does not match the 4x3 indices.
    # Parameters: axis and batch_dims for gather. Gathers from a replicated 1000x4 table using 4x3
    # random indices sharded on y in the first dimension; asserts equality with the plain gather
    # and a rank-3 layout batch sharded on y.
    params = np.arange(1000 * 4).reshape((1000, 4))
    # "batch" size = 2, num_indices = 3 per example
    indices = np.random.randint(0, 1000, size=4 * 3).reshape((4, 3))
    expected = array_ops.gather_v2(
        params, indices, axis=axis, batch_dims=batch_dims)

    params = api.relayout(params, layout=Layout.replicated(self.mesh, 2))
    indices = api.relayout(
        indices, Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=2)
    )

    dtensor_result = array_ops.gather_v2(
        params, indices, axis=axis, batch_dims=batch_dims)
    expected_layout = Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=3)
    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testResourceGather(self):
    # [FUNCTION DOC]
    # Skipped under XLA SPMD (variables unsupported). Gathers from a distributed variable holding a
    # replicated 1000x4 table using 1000x3 random indices sharded on y; asserts equality with the
    # gather on a normal variable and a rank-3 layout batch sharded on y.
    if self.mesh.use_xla_spmd():
      self.skipTest('Variables not supported yet with DTensor Xla Spmd.')

    params = np.arange(1000 * 4).reshape((1000, 4))
    indices = np.random.randint(0, 1000, size=1000 * 3).reshape((1000, 3))

    expected = array_ops.gather_v2(variables.Variable(params), indices)

    params = api.relayout(params, layout=Layout.replicated(self.mesh, 2))
    indices = api.relayout(
        indices, Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=2)
    )

    dtensor_result = array_ops.gather_v2(d_variable.DVariable(params), indices)
    expected_layout = Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=3)

    self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testResourceGatherRaisesErrorWhenResourceZeroDimSharded(self):
    # [FUNCTION DOC]
    # Skipped under XLA SPMD. Wraps a 1000x4 tensor sharded on y in dimension 0 in a distributed
    # variable and gathers with replicated indices; expects an unknown-error exception saying that
    # sharded 0th dimension for the resource tensor is not supported.
    if self.mesh.use_xla_spmd():
      self.skipTest('Variables not supported yet with DTensor Xla Spmd.')

    sharded_tensor = api.relayout(
        np.arange(1000 * 4).reshape((1000, 4)),
        layout=Layout.batch_sharded(self.mesh, _MESH_DIM_Y, 2),
    )
    # "batch" size = 2, num_indices = 3 per example
    indices = api.copy_to_mesh(
        np.random.randint(0, 1000, size=4 * 3).reshape((4, 3)),
        Layout.replicated(self.mesh, rank=2))

    with self.assertRaisesRegex(
        errors_impl.UnknownError,
        'DTensor does not support sharded 0th dimension for the resource tensor'
    ):
      array_ops.gather_v2(d_variable.DVariable(sharded_tensor), indices)

  def testUnsortedSegmentSum(self):
    # [FUNCTION DOC]
    # Skipped on TPU. Unsorted segment sum with 12 segments over a replicated 12x4 random data tensor
    # and segment ids sharded on y; asserts equality with the plain result and a replicated rank-2
    # output layout.
    self.skipForDeviceType(['TPU'], 'waiting for cl/344197900')
    num_segments = 12
    data = np.random.uniform(size=[num_segments, 4])
    segment_ids = np.random.randint(0, num_segments, size=num_segments)
    expected = gen_math_ops.unsorted_segment_sum(data, segment_ids,
                                                 num_segments)

    data = api.relayout(data, Layout.replicated(self.mesh, 2))
    segment_ids = api.relayout(
        segment_ids, Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1)
    )
    with api.default_mesh(self.mesh):
      dtensor_result = gen_math_ops.unsorted_segment_sum(
          data, segment_ids, num_segments)
      expected_layout = Layout.replicated(self.mesh, 2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testUnsortedSegmentSumWithFullyShardedIndices(self):
    # [FUNCTION DOC]
    # Skipped on TPU. Unsorted segment sum with 8 segments over replicated 2x4x3 data and 2x4
    # segment ids sharded on x and y; asserts equality with the plain result and a replicated
    # rank-2 output layout.
    self.skipForDeviceType(['TPU'], 'waiting for cl/344197900')

    num_segments = 8
    data = np.random.uniform(size=[2, 4, 3])
    segment_ids = np.random.randint(0, num_segments, size=[2, 4])
    expected = gen_math_ops.unsorted_segment_sum(data, segment_ids,
                                                 num_segments)

    data = api.relayout(data, Layout.replicated(self.mesh, 3))
    segment_ids = api.relayout(
        segment_ids, Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    )
    with api.default_mesh(self.mesh):
      dtensor_result = gen_math_ops.unsorted_segment_sum(
          data, segment_ids, num_segments)
      expected_layout = Layout.replicated(self.mesh, 2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_same_rank', [2, 2]),
      ('_adding_one_rank', [2, 2, 1]),
      ('_adding_one_rank_and_broadcasting', [2, 2, 2]),
  )
  def testBroadcastOpsWithFullyReplicatedInputs(self, new_shape):
    # [FUNCTION DOC]
    # Parameter new_shape: target shape list (same rank, or one more dimension, with broadcasting).
    # Broadcasts a replicated 2x1 tensor to new_shape; asserts equality with the plain result and
    # a replicated layout of the target rank.
    op = gen_array_ops.broadcast_to
    a = constant_op.constant([[1.], [3.]])
    assert a.shape == [2, 1]

    expected_result = op(a, new_shape)

    a = api.copy_to_mesh(a, self.replicated_layout_2d)
    dtensor_result = op(a, new_shape)

    self.assertDTensorEqual(expected_result,
                            Layout.replicated(self.mesh, len(new_shape)),
                            dtensor_result)

  def testBooleanMask(self):
    # [FUNCTION DOC]
    # Skipped under XLA SPMD and on TPU. For four input layouts (first-dim sharded, sharded on x and
    # y, last-dim sharded, replicated), applies boolean mask to a 2x4 tensor and an equally laid
    # out mask inside a compiled function. Asserts the 1-D result equals the plain one and carries
    # the expected layout converted to a parted layout (sharded 1-D for the first two, replicated
    # for the others).
    if self.mesh.use_xla_spmd():
      self.skipTest('Boolean mask not supported yet with DTensor Xla Spmd.')
    self.skipForDeviceType(['TPU'], 'int64 XlaAllReduce not supported.')

    for input_layout, expected_output_layout in [
        (
            self.first_dimension_sharded_layout,
            self.first_dimension_sharded_layout_1d,
        ),
        (
            Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh),
            self.first_dimension_sharded_layout_1d,
        ),
        (self.last_dimension_sharded_layout, self.replicated_layout_1d),
        (self.replicated_layout_2d, self.replicated_layout_1d),
    ]:
      tensor = constant_op.constant(np.arange(8).reshape(2, 4))
      mask = constant_op.constant(
          np.array([True, True, False, False, True, False, True, True]).reshape(
              2, 4
          )
      )
      expected = array_ops.boolean_mask(tensor, mask)

      tensor = api.relayout(tensor, input_layout)
      mask = api.relayout(mask, input_layout)

      @polymorphic_function.function
      def boolean_mask_func(t, m):
        # [FUNCTION DOC]
        # Nested helper: applies boolean mask with tensor t and mask m; returns the selected values.
        return array_ops.boolean_mask(t, m)

      result = boolean_mask_func(tensor, mask)
      self.assertDTensorEqual(
          expected, expected_output_layout.to_parted(), result
      )

  def testRawWhere(self):
    # [FUNCTION DOC]
    # Skipped under XLA SPMD. Runs the raw where op on an 8-element boolean vector sharded on x
    # inside a compiled function. With a parted layout the op returns local indices, so the
    # expected index list is 0,1 for the first shard and 0,2,3 for the second. Asserts the result
    # equals that list with the parted version of the first-dim sharded layout.
    if self.mesh.use_xla_spmd():
      self.skipTest('Where op not supported yet with DTensor Xla Spmd.')

    condition = constant_op.constant(
        np.array([True, True, False, False, True, False, True, True])
    )
    condition = api.relayout(condition, self.first_dimension_sharded_layout_1d)

    @polymorphic_function.function
    def func(c):
      # [FUNCTION DOC]
      # Nested helper: returns the raw where op (indices of true entries) of the condition c.
      return gen_array_ops.where(c)

    result = func(condition)
    # With parted layout, the raw where op will output local index instead of
    # global index. So the second half of test expectation ([0], [2], [3]) has
    # an offset of 4.
    expected = constant_op.constant(
        np.array([[0], [1], [0], [2], [3]]), dtype=dtypes.int64
    )
    self.assertDTensorEqual(
        expected, self.first_dimension_sharded_layout.to_parted(), result
    )

  @parameterized.named_parameters([
      {
          'testcase_name': 'FullyReplicatedInputs',
          'op': array_ops.where_v2,
          'shard_type': 'replicated',
      },
      {
          'testcase_name': 'BatchShardedInputs',
          'op': array_ops.where_v2,
          'shard_type': 'batch_sharded',
      },
  ])
  def testWhere(self, op, shard_type):
    # [FUNCTION DOC]
    # Parameters: op (the v2 where select function) and shard_type (replicated or batch_sharded).
    # Selects between two 2x2 float tensors by a 2x2 boolean condition, all in the chosen layout;
    # asserts equality with the plain result and that layout.
    layout = (
        self.replicated_layout_2d
        if shard_type == 'replicated' else self.first_dimension_sharded_layout)

    a = constant_op.constant([[True, False], [False, True]])
    b = constant_op.constant([[10., 20.], [30., 40.]])
    c = constant_op.constant([[50., 60.], [70., 80.]])
    expected_result = op(a, b, c)

    a = api.relayout(a, layout)
    b = api.relayout(b, layout)
    c = api.relayout(c, layout)
    dtensor_result = op(a, b, c)

    self.assertDTensorEqual(expected_result, layout, dtensor_result)

  def testSqueezeOp(self):
    # [FUNCTION DOC]
    # Squeezes a ones tensor of shape 1x2x1 sharded on x in dimension 1 (local shape 1x1x1, so
    # the sharded dimension must not be squeezed) with no axis, axis 0 and axis -1. Asserts
    # equality with plain results and the expected layouts: x-sharded 1-D, x then unsharded,
    # and unsharded then x.
    t = array_ops.ones([1, 2, 1])
    expected_result0 = array_ops.squeeze_v2(t)
    expected_result1 = array_ops.squeeze_v2(t, axis=0)
    expected_result2 = array_ops.squeeze_v2(t, axis=-1)

    # t will have [1,1,1] as locally sharded shape, this covers the case that
    # we should not squeeze the dim that's sharded.
    t = api.relayout(
        t,
        Layout(
            [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED], self.mesh
        ),
    )
    dtensor_result0 = array_ops.squeeze_v2(t)
    dtensor_result1 = array_ops.squeeze_v2(t, axis=0)
    dtensor_result2 = array_ops.squeeze_v2(t, axis=-1)

    self.assertDTensorEqual(expected_result0, Layout([_MESH_DIM_X], self.mesh),
                            dtensor_result0)
    self.assertDTensorEqual(
        expected_result1, Layout([_MESH_DIM_X, layout_lib.UNSHARDED],
                                 self.mesh), dtensor_result1)
    self.assertDTensorEqual(
        expected_result2, Layout([layout_lib.UNSHARDED, _MESH_DIM_X],
                                 self.mesh), dtensor_result2)

  @parameterized.parameters(('replicated',), ('sharded',))
  def testDiagPart(self, shard_type):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shard_type labels are inverted: 'replicated' uses a layout sharded on x and
    # y, while 'sharded' uses the fully replicated layout.
    # Takes the diagonal of a random 16x16 tensor in the chosen layout; asserts equality with
    # the plain result and a replicated 1-D layout.
    x = stateless_random_ops.stateless_random_uniform(
        shape=(16, 16), seed=[0, 1])
    expected = gen_array_ops.diag_part(input=x)

    if shard_type == 'replicated':
      layout = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)
    else:
      layout = Layout.replicated(self.mesh, 2)
    x = api.relayout(x, layout)

    got = gen_array_ops.diag_part(input=x)
    self.assertDTensorEqual(expected, Layout.replicated(self.mesh, 1), got)

  @parameterized.product(
      axis_dim=[-3, -2, -1, 0, 1, 2],
      shard_type=['replicated', 'batch_sharded'],
      reverse=[True, False])
  def testCumSum(self, axis_dim, shard_type, reverse):
    # [FUNCTION DOC]
    # Parameters: axis_dim (-3 to 2), shard_type (replicated or batch_sharded on x) and reverse
    # (boolean). Cumulative sum of a random 16x16x16 tensor along axis_dim. Expected layout is the
    # input layout, except that for batch sharding with axis 0 or -3 the result is replicated
    # (the scanned axis is never sharded). Asserts values and layout.
    input_tensor = stateless_random_ops.stateless_random_uniform(
        shape=(16, 16, 16), seed=[0, 1])
    expected = math_ops.cumsum(x=input_tensor, axis=axis_dim, reverse=reverse)

    if shard_type == 'replicated':
      layout = Layout.replicated(self.mesh, rank=3)
      expected_layout = layout
    else:
      layout = Layout.batch_sharded(self.mesh, batch_dim=_MESH_DIM_X, rank=3)
      # Axis dimension should always be replicated, even on sharding dim.
      if axis_dim in [-3, 0]:
        expected_layout = Layout.replicated(self.mesh, rank=3)
      else:
        expected_layout = layout

    input_tensor = api.relayout(input_tensor, layout)
    got = math_ops.cumsum(x=input_tensor, axis=axis_dim, reverse=reverse)

    self.assertDTensorEqual(expected, expected_layout, got)

  @parameterized.named_parameters(('Sharded', 'sharded'),
                                  ('Replicated', 'replicated'))
  def testStringFormat(self, shard_spec):
    # [FUNCTION DOC]
    # Parameter shard_spec: 'sharded' (batch sharded on x) or 'replicated'. Skipped on TPU, and on
    # GPU in multi-device mode. Formats a random 8x9x9x1 float tensor into a string. Because the
    # result is a string, compares manually: the output layout must be a replicated scalar and
    # every unpacked component must equal the plain result.
    self.skipForDeviceType(['TPU'], 'StringFormat not supported on TPU.')
    # TODO(b/303662238): See whether we can replicate soft placement here.
    if test_util.use_multi_device_mode():
      self.skipForDeviceType(['GPU'], 'StringFormat not supported on GPU.')

    np.random.seed(123)
    inputs = constant_op.constant(
        np.random.normal(0.0, 1.0, 8 * 9 * 9).reshape([8, 9, 9, 1]),
        dtype=dtypes.float32)
    expected_result = gen_string_ops.string_format(inputs=[inputs])

    if shard_spec == 'sharded':
      layout = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=4)
    else:
      layout = Layout.replicated(self.mesh, rank=4)
    inputs = api.relayout(inputs, layout)
    got = gen_string_ops.string_format(inputs=[inputs])

    # Manually compare instead of assertDTensorEqual since outputs are strings.
    self.assertEqual(
        api.fetch_layout(got), Layout.replicated(self.mesh, rank=0))
    for got_tensor in api.unpack(got):
      self.assertEqual(expected_result, got_tensor)

  @parameterized.named_parameters(('Sharded', 'sharded'),
                                  ('Replicated', 'replicated'))
  def testStringFormatOnTPURequiresCopyToMeshToCPU(self, shard_spec):
    # [FUNCTION DOC]
    # Runs only on TPU. Builds a TPU mesh and a CPU mesh of the same shape. Relayouts a random
    # input on the TPU mesh (sharded or replicated), relayouts it to replicated, copies it to the CPU
    # mesh, and formats it to a string there (unsupported on TPU). Asserts the output is a replicated
    # scalar on the CPU mesh and each unpacked component equals the plain result.
    self.skipForDeviceType(['CPU', 'GPU'], 'Testing only for TPU.')

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()

    tpu_mesh = Mesh([_MESH_DIM_X, _MESH_DIM_Y], global_ids, local_ids,
                    test_util.create_device_list((2, 4), 'TPU'))
    cpu_mesh = Mesh([_MESH_DIM_X, _MESH_DIM_Y], global_ids, local_ids,
                    test_util.create_device_list((2, 4), 'CPU'))

    cpu_layout = Layout.replicated(cpu_mesh, rank=4)
    if shard_spec == 'sharded':
      tpu_layout = Layout.batch_sharded(tpu_mesh, _MESH_DIM_X, rank=4)
    else:
      tpu_layout = Layout.replicated(tpu_mesh, rank=4)

    inputs = stateless_random_ops.stateless_random_uniform(
        shape=(8, 9, 9, 1), seed=[0, 1])
    expected_result = gen_string_ops.string_format(inputs=[inputs])

    inputs = api.relayout(inputs, tpu_layout)
    # StringFormat is not supported on TPU, so copy_to_mesh to the CPU.
    # Since we cannot eager copy_to_mesh from an input with non-replicated
    # layout yet, relayout to replicated layout first, and then transfer to CPU.
    inputs = api.copy_to_mesh(
        api.relayout(inputs, Layout.replicated(tpu_mesh, rank=4)), cpu_layout)

    got = gen_string_ops.string_format(inputs=[inputs])
    # Manually compare instead of assertDTensorEqual since outputs are strings.
    self.assertEqual(api.fetch_layout(got), Layout.replicated(cpu_mesh, rank=0))
    for got_tensor in api.unpack(got):
      self.assertEqual(expected_result, got_tensor)

  @parameterized.named_parameters(
      # TODO(feyu): to_hash_bucket and to_hash_bucket_strong are not defined
      # in the tf MLIR dialect.
      ('ShardedFast', gen_string_ops.string_to_hash_bucket_fast, 'sharded'),
      ('ReplicatedFast', gen_string_ops.string_to_hash_bucket_fast,
       'replicated'),
  )
  def testStringToHashBucket(self, to_hash_bucket_fn, shard_spec):
    # [FUNCTION DOC]
    # Parameters: to_hash_bucket_fn (the fast string-to-hash-bucket op) and shard_spec ('sharded'
    # or 'replicated'). Skipped on GPU and TPU. Hashes four strings into 32 buckets with the input
    # in the chosen 1-D layout; asserts equality with the plain result and the same layout.
    self.skipForDeviceType(
        ['GPU', 'TPU'],
        'StringToHashBucket functions not supported on TPU or GPU.')

    inputs = constant_op.constant(['a', 'b', 'c', 'd'], dtype=dtypes.string)
    expected_result = to_hash_bucket_fn(inputs, num_buckets=32)

    if shard_spec == 'sharded':
      layout = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=1)
    else:
      layout = Layout.replicated(self.mesh, rank=1)
    inputs = api.relayout(inputs, layout)
    got = to_hash_bucket_fn(inputs, num_buckets=32)

    self.assertDTensorEqual(expected_result, layout, got)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Replicated',
          'shard_type': 'replicated',
      }, {
          'testcase_name': 'BatchSharded',
          'shard_type': 'batch_sharded',
      })
  def testTensorListReserveSetAndGetRetrievesCorrectTensor(self, shard_type):
    # [FUNCTION DOC]
    # Parameter shard_type: replicated or batch_sharded. Runs only on CPU. In a compiled function,
    # reserves a tensor list of four 4x4 int32 elements, sets item 0 to the input, and reads item 0
    # back. Asserts the retrieved tensor equals the ones input with a replicated rank-2 layout.
    self.skipForDeviceType(['TPU', 'GPU'], 'Testing only for CPU.')

    input_tensor = array_ops.ones([4, 4], dtype=dtypes.int32)

    if shard_type == 'replicated':
      layout = Layout.replicated(self.mesh, rank=2)
    else:
      layout = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=2)

    @polymorphic_function.function
    def f(input_tensor):
      # [FUNCTION DOC]
      # Nested helper: reserves a list of four 4x4 int32 elements, stores the input at index 0 and
      # returns the element at index 0.
      list_handle = gen_list_ops.tensor_list_reserve(
          element_shape=constant_op.constant([4, 4], dtype=dtypes.int32),
          num_elements=constant_op.constant(4, dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      list_handle = gen_list_ops.tensor_list_set_item(
          input_handle=list_handle,
          index=constant_op.constant(0, dtype=dtypes.int32),
          item=input_tensor)
      return gen_list_ops.tensor_list_get_item(
          input_handle=list_handle,
          index=constant_op.constant(0, dtype=dtypes.int32),
          element_shape=constant_op.constant([4, 4], dtype=dtypes.int32),
          element_dtype=dtypes.int32)

    got_tensor = f(api.relayout(input_tensor, layout))
    self.assertDTensorEqual(input_tensor, Layout.replicated(self.mesh, rank=2),
                            got_tensor)

  @parameterized.named_parameters(
      ('x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('unsharded_x', [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('x_y', [_MESH_DIM_X, _MESH_DIM_Y]),
      ('unsharded_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]))
  def testDisableCopyOnRead(self, sharding_specs):
    # [FUNCTION DOC]
    # Parameter sharding_specs: sharding spec list of rank 2. Skipped on TPU. Creates a distributed
    # variable from a ones 4x8 tensor with that layout, calls the disable-copy-on-read op on its
    # handle once eagerly and once through a compiled function, and asserts that the variable's
    # layout is unchanged in both cases.
    self.skipForDeviceType(['TPU'], 'Op not supported on TPUs.')

    def f(d_var):
      # [FUNCTION DOC]
      # Nested helper: disables copy-on-read for the variable's resource handle and returns the
      # variable.
      gen_resource_variable_ops.disable_copy_on_read(resource=d_var.handle)
      return d_var

    layout = Layout(sharding_specs, self.mesh)

    variable = d_variable.DVariable(
        initial_value=api.relayout(
            array_ops.ones([4, 8], dtype=dtypes.float32), layout
        )
    )

    # Eager
    self.assertEqual(api.fetch_layout(f(variable)), layout)

    # Function
    self.assertEqual(
        api.fetch_layout(polymorphic_function.function(f)(variable)), layout)

  def testShardedFilename(self):
    # [FUNCTION DOC]
    # Runs only on CPU (host-only string op). Builds the sharded filename from a base name, shard
    # number 1 and 16 shards, plainly and with all three scalar inputs relayouted replicated.
    # Asserts the output layout is replicated scalar and every unpacked component equals the plain
    # result.
    self.skipForDeviceType(['TPU', 'GPU'],
                           'Strings only for CPUs, this is a host-only op.')

    basename = constant_op.constant('dtensor-file')
    shard = constant_op.constant(1, dtype=dtypes.int32)
    num_shards = constant_op.constant(16, dtype=dtypes.int32)

    layout = Layout.replicated(self.mesh, rank=0)

    expected = gen_io_ops.sharded_filename(
        basename=basename, shard=shard, num_shards=num_shards, name=None)

    result = gen_io_ops.sharded_filename(
        basename=api.relayout(basename, layout),
        shard=api.relayout(shard, layout),
        num_shards=api.relayout(num_shards, layout),
    )

    self.assertEqual(api.fetch_layout(result), layout)
    for result_tensor in api.unpack(result):
      self.assertEqual(expected, result_tensor)

  @parameterized.named_parameters(*test_util.product(
      (('_indices_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
       ('_indices_x', [_MESH_DIM_X, layout_lib.UNSHARDED])),
      (('_updates_unsharded_unsharded',
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
       ('_updates_x_unsharded',
        [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED]),
       ('_updates_unsharded_y',
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_Y]),
       ('_updates_x_y', [layout_lib.UNSHARDED, _MESH_DIM_X, _MESH_DIM_Y]))))
  def testScatterNd(self, indices_spec, updates_spec):
    # [FUNCTION DOC]
    # Parameters: indices_spec (rank-2 sharding spec) and updates_spec (rank-3 sharding spec).
    # Scatters two 4x4 update slabs at indices 0 and 15 into shape 16x4x4, plainly and with
    # relayouted indices and updates. Asserts equal values and that the result has the updates layout.
    indices_layout = Layout(indices_spec, self.mesh)
    updates_layout = Layout(updates_spec, self.mesh)
    indices = constant_op.constant([[0], [15]])
    updates = constant_op.constant([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7],
                                     [8, 8, 8, 8]],
                                    [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7],
                                     [8, 8, 8, 8]]])
    shape = [16, 4, 4]

    expected_result = gen_array_ops.scatter_nd(indices, updates, shape)
    got_result = gen_array_ops.scatter_nd(
        api.relayout(indices, indices_layout),
        api.relayout(updates, updates_layout),
        shape,
    )

    self.assertDTensorEqual(expected_result, updates_layout, got_result)


class DTensorConvSPMDTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests for SPMD expansion of convolution and pooling ops on a 3-D CPU mesh of shape 2x3x3
  # (dimensions b, x, y; 18 devices). Skipped on TPU and when GPU or TPU system devices exist. The
  # mesh and its dimension names are stored on the instance by setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture. Calls the base setUp, skips on TPU and when GPUs or a TPU system are present.
    # Stores the dimension names b, x, y (_mesh_dim_b, _mesh_dim_x, _mesh_dim_y, _dims), builds
    # a 2x3x3 mesh for each device type using CPU devices, selects the mesh for the current device
    # type as self.mesh, stores its size, and resets the logical CPU devices to that size.
    # Returns nothing.
    super(DTensorConvSPMDTest, self).setUp()

    # TODO(b/169436213): Re-enable TPU after figuring out multi-chip story.
    self.skipForDeviceType(['TPU'], 'reserving 4 chips on forge is unreliable')

    if config.list_physical_devices('GPU') or config.list_logical_devices(
        'TPU_SYSTEM'):
      self.skipTest(
          'Skipping as 3D mesh with 18 devices cannot be tested on GPU/TPU.')

    # Builds a 2x3x3 mesh.
    self._mesh_dim_b = 'b'
    self._mesh_dim_x = 'x'
    self._mesh_dim_y = 'y'
    self._dims = [self._mesh_dim_b, self._mesh_dim_x, self._mesh_dim_y]

    global_ids = test_util.create_device_ids_array([2, 3, 3])
    local_ids = np.ravel(global_ids).tolist()

    mesh_dict = {
        device: Mesh(self._dims, global_ids, local_ids,
                     test_util.create_device_list([2, 3, 3], 'CPU'))
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)
    self._num_devices = self.mesh.size
    test_util.reset_logical_devices('CPU', self.mesh.size)

  @parameterized.named_parameters(test_util_ops.PADDINGS)
  def testConv2DWithFullReplicatedInputs(self, padding):
    # [FUNCTION DOC]
    # Parameter padding: padding mode name. Convolves a random 1x9x9x1 input with a fixed 2x2x1x2
    # kernel (stride 1), plainly and with input and kernel copied to the mesh fully replicated.
    # Asserts equal values and a fully replicated rank-4 layout.
    np.random.seed(123)

    x_in = np.random.normal(0.0, 1.0, 9 * 9).reshape([1, 9, 9, 1])
    kernel_in = np.array([
        [[[2, 0.1]], [[3, 0.2]]],
        [[[0, 0.3]], [[1, 0.4]]],
    ])

    x = constant_op.constant(x_in, dtype=dtypes.float32)
    kernel = constant_op.constant(kernel_in, dtype=dtypes.float32)
    expected_result = nn_ops.conv2d_v2(
        x, kernel, strides=[1, 1, 1, 1], padding=padding)

    x = api.copy_to_mesh(x, Layout([layout_lib.UNSHARDED] * 4, self.mesh))
    kernel = api.copy_to_mesh(kernel,
                              Layout([layout_lib.UNSHARDED] * 4, self.mesh))

    got = nn_ops.conv2d_v2(x, kernel, strides=[1, 1, 1, 1], padding=padding)

    self.assertDTensorEqual(expected_result,
                            Layout([layout_lib.UNSHARDED] * 4, self.mesh), got)

  @parameterized.product(shard_type=['replicated', 'batch_sharded'])
  def testConv3DBackpropInput(self, shard_type):
    # [FUNCTION DOC]
    # Parameter shard_type: replicated or batch_sharded (on mesh dimension b). Runs the 3-D
    # convolution input-backprop op (same padding, unit strides) with random 4x4x4x4x4 filter and
    # out_backprop; the input sizes and filter are replicated and out_backprop uses the chosen
    # layout. Asserts equality with the plain result and that layout.
    input_sizes = constant_op.constant([4, 4, 4, 4, 4])
    filter_input = stateless_random_ops.stateless_random_uniform(
        shape=[4, 4, 4, 4, 4], seed=[0, 1])
    out_backprop = stateless_random_ops.stateless_random_uniform(
        shape=[4, 4, 4, 4, 4], seed=[0, 1])
    strides = [1, 1, 1, 1, 1]

    expected_result = gen_nn_ops.conv3d_backprop_input_v2(
        input_sizes=input_sizes,
        filter=filter_input,
        out_backprop=out_backprop,
        strides=strides,
        padding='SAME')

    if shard_type == 'replicated':
      grad_layout = Layout.replicated(self.mesh, rank=5)
    else:
      grad_layout = Layout.batch_sharded(self.mesh, self._mesh_dim_b, rank=5)

    got_result = gen_nn_ops.conv3d_backprop_input_v2(
        input_sizes=api.relayout(
            input_sizes, Layout.replicated(self.mesh, rank=1)
        ),
        filter=api.relayout(filter_input, Layout.replicated(self.mesh, rank=5)),
        out_backprop=api.relayout(out_backprop, grad_layout),
        strides=strides,
        padding='SAME',
    )

    self.assertDTensorEqual(expected_result, grad_layout, got_result)

  @parameterized.named_parameters(test_util_ops.PADDINGS)
  def testConv2DWithBatchShardedInputs(self, padding):
    # [FUNCTION DOC]
    # KNOWN ISSUE: always skipped (conv gradient ops do not know input layouts), so the rest of
    # the body is dead code.
    # Intended scenario: convolve a 2x9x9x1 input sharded on b in the batch dimension with a
    # replicated kernel and compare outputs and input/filter gradients (recorded with a gradient
    # tape) against the plain computation and the expected layouts.
    self.skipTest(
        reason='b/272579753: ensure Conv grad Ops know about input layouts.'
    )
    # Reason to flip same shape policy: The backprop of the nn_ops.conv2d_v2 is
    # simply array_ops.ones_like_v2(conv2d_result). However, as DTensor does not
    # control gradient tape, the tape will not attach the layout from
    # conv2d_result to the ones. In normal computation, the backprop pass has
    # more information to pass from the final scalar loss to the conv2d, so
    # this is not a problem.
    # But this well-design unit tests, without same shape policy, it will get a
    # different layout for the inputs' grad.
    np.random.seed(123)

    x_in = np.random.normal(0.0, 1.0, 2 * 9 * 9).reshape([2, 9, 9, 1])
    kernel_in = np.array([
        [[[2, 0.1]], [[3, 0.2]]],
        [[[0, 0.3]], [[1, 0.4]]],
    ])

    x = constant_op.constant(x_in, dtype=dtypes.float32)
    kernel = constant_op.constant(kernel_in, dtype=dtypes.float32)
    with backprop.GradientTape() as tape:
      tape.watch([x, kernel])
      expected_result = nn_ops.conv2d_v2(
          x, kernel, strides=[1, 1, 1, 1], padding=padding)
    expected_input_gradient, expected_filter_gradient = tape.gradient(
        expected_result, [x, kernel])

    x = api.relayout(
        x, Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh)
    )
    kernel = api.relayout(kernel, Layout([layout_lib.UNSHARDED] * 4, self.mesh))
    # Explicitly open the scope as ops generated from tape could be broadcasted
    # to replicated by default.
    with api.default_mesh(self.mesh):
      with backprop.GradientTape() as tape:
        tape.watch([x, kernel])
        got = nn_ops.conv2d_v2(x, kernel, strides=[1, 1, 1, 1], padding=padding)
      got_input_gradient, got_filter_filter = tape.gradient(got, [x, kernel])

    self.assertDTensorEqual(
        expected_result,
        Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh), got)
    self.assertDTensorEqual(
        expected_input_gradient,
        Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh),
        got_input_gradient)
    self.assertDTensorEqual(expected_filter_gradient,
                            Layout([layout_lib.UNSHARDED] * 4, self.mesh),
                            got_filter_filter)

  @parameterized.named_parameters(test_util_ops.PADDINGS)
  def testMaxPoolWithBatchShardedInputs(self, padding):
    # [FUNCTION DOC]
    # Parameter padding: padding mode name. Max-pools a random 2-image input (window 3x4, strides
    # 2x3, sized so windows tile evenly) plainly and with the input sharded on mesh dimension b in
    # the batch dimension. Asserts equal values and the batch-sharded rank-4 layout.
    np.random.seed(123)
    row_window_size = 3
    col_window_size = 4
    window_size = [1, row_window_size, col_window_size, 1]
    stride_size = [1, row_window_size - 1, col_window_size - 1, 1]

    num_rows = (row_window_size - 1) * 5 + 1
    num_cols = (col_window_size - 1) * 7 + 1
    x_in = np.random.normal(0.0, 1.0, 2 * num_rows * num_cols * 3).reshape(
        [2, num_rows, num_cols, 3])

    inputs = constant_op.constant(x_in, dtype=dtypes.float32)
    expected_result = nn_ops.max_pool_v2(inputs, window_size, stride_size,
                                         padding)

    x = api.relayout(
        inputs, Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh)
    )

    got = nn_ops.max_pool_v2(x, window_size, stride_size, padding)

    self.assertDTensorEqual(
        expected_result,
        Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh), got)

  @parameterized.named_parameters(test_util_ops.PADDINGS)
  def testMaxPoolGradWithBatchShardedInputs(self, padding):
    # [FUNCTION DOC]
    # Parameter padding: padding mode name. Same setup as the max pool test, but records the
    # pooling under a gradient tape and compares the gradient with respect to the input, plainly and
    # for the batch-sharded DTensor under the default mesh. Asserts equal values and the batch-sharded
    # rank-4 layout for the gradient.
    np.random.seed(123)
    row_window_size = 3
    col_window_size = 4
    window_size = [1, row_window_size, col_window_size, 1]
    stride_size = [1, row_window_size - 1, col_window_size - 1, 1]

    num_rows = (row_window_size - 1) * 5 + 1
    num_cols = (col_window_size - 1) * 7 + 1
    x_in = np.random.normal(0.0, 1.0, 2 * num_rows * num_cols * 3).reshape(
        [2, num_rows, num_cols, 3])
    inputs = constant_op.constant(x_in, dtype=dtypes.float32)
    with backprop.GradientTape() as tape:
      tape.watch([inputs])
      expected_result = nn_ops.max_pool_v2(inputs, window_size, stride_size,
                                           padding)
    expected_grad = tape.gradient(expected_result, [inputs])

    x = api.relayout(
        inputs, Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh)
    )

    with api.default_mesh(self.mesh):
      with backprop.GradientTape() as tape:
        tape.watch([x])
        dtensor_result = nn_ops.max_pool_v2(x, window_size, stride_size,
                                            padding)
      dtensor_grad = tape.gradient(dtensor_result, [x])

    self.assertDTensorEqual(
        expected_grad[0],
        Layout([self._dims[0]] + [layout_lib.UNSHARDED] * 3, self.mesh),
        dtensor_grad[0])


class DTensorFFTSPMDTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests for SPMD expansion of FFT ops on a small 1x1x2 mesh (dimensions b, x, y) with CPU devices.
  # The mesh is stored on the instance by setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture. Calls the base setUp, stores dimension names b, x, y, builds a 1x1x2 mesh for
    # each device type from CPU devices and selects the one for the current device type as
    # self.mesh. (The comment in the code mentions 2x3x3 but the actual mesh shape is 1x1x2.)
    # Returns nothing.
    super().setUp()

    # Builds a 2x3x3 mesh.
    self._mesh_dim_b = 'b'
    self._mesh_dim_x = 'x'
    self._mesh_dim_y = 'y'
    self._dims = [self._mesh_dim_b, self._mesh_dim_x, self._mesh_dim_y]

    global_ids = test_util.create_device_ids_array([1, 1, 2])
    local_ids = np.ravel(global_ids).tolist()

    mesh_dict = {
        device: Mesh(self._dims, global_ids, local_ids,
                     test_util.create_device_list([1, 1, 2], 'CPU'))
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
  )
  def testFFT(self, input_layout_specs, expected_layout_specs, input_datatype):
    # [FUNCTION DOC]
    # Parameters: input_layout_specs (sharding spec list for the rank-3 input on mesh dims b, x, y),
    # expected_layout_specs (sharding spec of the output), input_datatype (complex64 or complex128).
    # Builds a random complex 2x4x8 tensor, copies it to the mesh with the input layout, applies the
    # 1-D FFT op (last axis) and asserts equality with the numpy FFT (cast to single precision for
    # complex64) and the expected output layout (last axis unsharded).
    a = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    b = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    x = constant_op.constant(
        a + b * 1j,
        dtype=input_datatype,
    )
    expected_result = np.fft.fft(x)
    if input_datatype == dtypes.complex64:
      expected_result = expected_result.astype(np.complex64)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    result = gen_spectral_ops.fft(x)
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
  )
  def testIFFT(self, input_layout_specs, expected_layout_specs, input_datatype):
    # [FUNCTION DOC]
    # Same scenario as testFFT but applies the inverse 1-D FFT op and compares with numpy's inverse
    # FFT; asserts values and the expected output layout.
    a = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    b = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    x = constant_op.constant(
        a + b * 1j,
        dtype=input_datatype,
    )
    expected_result = np.fft.ifft(x)
    if input_datatype == dtypes.complex64:
      expected_result = expected_result.astype(np.complex64)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    result = gen_spectral_ops.ifft(x)
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'x'],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'x'],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'x'],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'x'],
          input_datatype=dtypes.complex128,
      ),
  )
  def testFFT2(self, input_layout_specs, expected_layout_specs, input_datatype):
    # [FUNCTION DOC]
    # Same scenario as testFFT but applies the 2-D FFT op (last two axes) and compares with numpy's
    # 2-D FFT; asserts values and the expected output layout (the transformed axes become
    # unsharded or swapped as specified in the cases).
    a = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    b = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    x = constant_op.constant(
        a + b * 1j,
        dtype=input_datatype,
    )
    expected_result = np.fft.fft2(x)
    if input_datatype == dtypes.complex64:
      expected_result = expected_result.astype(np.complex64)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    result = gen_spectral_ops.fft2d(x)
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
  )
  def testIFFT2(
      self, input_layout_specs, expected_layout_specs, input_datatype
  ):
    # [FUNCTION DOC]
    # Same scenario as testFFT2 but applies the inverse 2-D FFT op and compares with numpy's inverse
    # 2-D FFT; asserts values and the expected output layout.
    a = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    b = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    x = constant_op.constant(
        a + b * 1j,
        dtype=input_datatype,
    )
    expected_result = np.fft.ifft2(x)
    if input_datatype == dtypes.complex64:
      expected_result = expected_result.astype(np.complex64)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    result = gen_spectral_ops.ifft2d(x)
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'x'],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'x'],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'y'],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'y'],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'x'],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'x'],
          input_datatype=dtypes.complex128,
      ),
  )
  def testFFT3(self, input_layout_specs, expected_layout_specs, input_datatype):
    # [FUNCTION DOC]
    # Same scenario as testFFT but applies the 3-D FFT op (all three axes) and compares with numpy's
    # n-dimensional FFT; asserts values and the expected output layout.
    a = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    b = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    x = constant_op.constant(
        a + b * 1j,
        dtype=input_datatype,
    )
    expected_result = np.fft.fftn(x)
    if input_datatype == dtypes.complex64:
      expected_result = expected_result.astype(np.complex64)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    result = gen_spectral_ops.fft3d(x)
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['x', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['x', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['x', 'b', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['x', 'b', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
      ),
  )
  def testIFFT3(
      self, input_layout_specs, expected_layout_specs, input_datatype
  ):
    # [FUNCTION DOC]
    # Same scenario as testFFT3 but applies the inverse 3-D FFT op and compares with numpy's inverse
    # n-dimensional FFT; asserts values and the expected output layout.
    a = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    b = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    x = constant_op.constant(
        a + b * 1j,
        dtype=input_datatype,
    )
    expected_result = np.fft.ifftn(x)
    if input_datatype == dtypes.complex64:
      expected_result = expected_result.astype(np.complex64)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    result = gen_spectral_ops.ifft3d(x)
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='unsharded_float32',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='unsharded_float64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
      dict(
          testcase_name='fullySharded_float32',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='fullySharded_float64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
      dict(
          testcase_name='partiallySharded1_float32',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='partiallySharded1_float64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
      dict(
          testcase_name='partiallySharded2_float32',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='partiallySharded2_float64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
  )

  # Each input dimension for RFFT must have length of at least fft_length[i].
  def testRFFT(
      self,
      input_layout_specs,
      expected_layout_specs,
      input_datatype,
      complex_type,
  ):
    # [FUNCTION DOC]
    # Parameters: input and expected sharding spec lists, input_datatype (float32 or float64) and
    # complex_type (matching numpy complex type for the output). Builds a random real 2x4x10
    # tensor, copies it to the mesh, applies the real FFT op with length 10 on the last axis and
    # asserts equality with numpy's real FFT (cast to complex64 for float32) and the expected layout.
    x = constant_op.constant(
        np.random.normal(0, 10, 80).reshape([2, 4, 10]),
        dtype=input_datatype,
    )
    expected_result = np.fft.rfft(x, n=10)
    if input_datatype == dtypes.float32:
      expected_result = expected_result.astype(np.complex64)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    length = constant_op.constant(
        [10],
        dtype=dtypes.int32,
    )
    result = gen_spectral_ops.rfft(x, fft_length=length, Tcomplex=complex_type)
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'x'],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'x'],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'x'],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', layout_lib.UNSHARDED, 'x'],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
  )
  def testRFFT2(
      self,
      input_layout_specs,
      expected_layout_specs,
      input_datatype,
      complex_type,
  ):
    # [FUNCTION DOC]
    # Parameters as in testRFFT. Builds a random real 2x4x12 tensor, copies it to the mesh, applies
    # the 2-D real FFT op with fft length (4, 10) (which crops the last axis) and asserts equality
    # with numpy's 2-D real FFT with that size and the expected output layout.
    x = constant_op.constant(
        np.random.normal(0, 10, 96).reshape([2, 4, 12]),
        dtype=input_datatype,
    )
    expected_result = np.fft.rfft2(x, s=[4, 10])
    if input_datatype == dtypes.float32:
      expected_result = expected_result.astype(np.complex64)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    length = constant_op.constant(
        [4, 10],
        dtype=dtypes.int32,
    )
    result = gen_spectral_ops.rfft2d(
        x, fft_length=length, Tcomplex=complex_type
    )
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'x'],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'x'],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'y'],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'y'],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'x'],
          input_datatype=dtypes.float32,
          complex_type=np.complex64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=[layout_lib.UNSHARDED, 'b', 'x'],
          input_datatype=dtypes.float64,
          complex_type=np.complex128,
      ),
  )
  def testRFFT3(
      self,
      input_layout_specs,
      expected_layout_specs,
      input_datatype,
      complex_type,
  ):
    # [FUNCTION DOC]
    # Parameters as in testRFFT. Builds a random real 2x4x10 tensor, copies it to the mesh, applies
    # the 3-D real FFT op with fft length (2, 4, 10) and asserts equality with numpy's n-dimensional
    # real FFT and the expected output layout.
    x = constant_op.constant(
        np.random.normal(0, 10, 80).reshape([2, 4, 10]),
        dtype=input_datatype,
    )
    expected_result = np.fft.rfftn(x, s=[2, 4, 10])
    if input_datatype == dtypes.float32:
      expected_result = expected_result.astype(np.complex64)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    length = constant_op.constant(
        [2, 4, 10],
        dtype=dtypes.int32,
    )
    result = gen_spectral_ops.rfft3d(
        x, fft_length=length, Tcomplex=complex_type
    )
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
  )

  # Each input dimension for IRFFT must have length of at least fft_length[i].
  # For the inner-most input dimension, it should >= fft_shape[i] / 2 + 1.
  def testIRFFT(
      self, input_layout_specs, expected_layout_specs, input_datatype, real_type
  ):
    # [FUNCTION DOC]
    # Parameters: input and expected sharding spec lists, input_datatype (complex64 or complex128)
    # and real_type (output real type). Builds a random complex 2x4x10 tensor, copies it to the
    # mesh, applies the inverse real FFT op with output length 8 on the last axis and asserts
    # equality with numpy's inverse real FFT (cast to float32 for complex64) and the expected layout.
    a = np.random.normal(0, 10, 80).reshape([2, 4, 10])
    b = np.random.normal(0, 10, 80).reshape([2, 4, 10])
    x = constant_op.constant(
        a + b * 1j,
        dtype=input_datatype,
    )
    expected_result = np.fft.irfft(x, n=8)
    if input_datatype == dtypes.complex64:
      expected_result = expected_result.astype(np.float32)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    length = constant_op.constant(
        [8],
        dtype=dtypes.int32,
    )
    result = gen_spectral_ops.irfft(x, fft_length=length, Treal=real_type)
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
  )
  def testIRFFT2(
      self, input_layout_specs, expected_layout_specs, input_datatype, real_type
  ):
    # [FUNCTION DOC]
    # Parameters as in testIRFFT. Random complex 2x4x8 input, copied to the mesh; applies the 2-D
    # inverse real FFT with length (4, 8); asserts equality with numpy's 2-D inverse real FFT and
    # the expected output layout.
    a = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    b = np.random.normal(0, 10, 64).reshape([2, 4, 8])
    x = constant_op.constant(
        a + b * 1j,
        dtype=input_datatype,
    )
    expected_result = np.fft.irfft2(x, s=[4, 8])
    if input_datatype == dtypes.complex64:
      expected_result = expected_result.astype(np.float32)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    length = constant_op.constant(
        [4, 8],
        dtype=dtypes.int32,
    )
    result = gen_spectral_ops.irfft2d(x, fft_length=length, Treal=real_type)
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='_unsharded_complex64',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_unsharded_complex128',
          input_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          expected_layout_specs=[
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
              layout_lib.UNSHARDED,
          ],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
      dict(
          testcase_name='_fullySharded_complex64',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['x', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_fullySharded_compelx128',
          input_layout_specs=['b', 'x', 'y'],
          expected_layout_specs=['x', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
      dict(
          testcase_name='_partiallySharded1_complex64',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_partiallySharded1_complex128',
          input_layout_specs=['b', layout_lib.UNSHARDED, 'y'],
          expected_layout_specs=['b', 'y', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
      dict(
          testcase_name='_partiallySharded2_complex64',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['x', 'b', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex64,
          real_type=dtypes.float32,
      ),
      dict(
          testcase_name='_partiallySharded2_complex128',
          input_layout_specs=['b', 'x', layout_lib.UNSHARDED],
          expected_layout_specs=['x', 'b', layout_lib.UNSHARDED],
          input_datatype=dtypes.complex128,
          real_type=dtypes.float64,
      ),
  )
  def testIRFFT3(
      self, input_layout_specs, expected_layout_specs, input_datatype, real_type
  ):
    # [FUNCTION DOC]
    # Parameters as in testIRFFT. Random complex 2x4x6 input, copied to the mesh; applies the 3-D
    # inverse real FFT with length (2, 4, 8); asserts equality with numpy's n-dimensional inverse
    # real FFT and the expected output layout.
    a = np.random.normal(0, 10, 48).reshape([2, 4, 6])
    b = np.random.normal(0, 10, 48).reshape([2, 4, 6])
    x = constant_op.constant(
        a + b * 1j,
        dtype=input_datatype,
    )
    expected_result = np.fft.irfftn(x, s=[2, 4, 8])
    if input_datatype == dtypes.complex64:
      expected_result = expected_result.astype(np.float32)
    x = api.copy_to_mesh(x, Layout(input_layout_specs, self.mesh))
    length = constant_op.constant(
        [2, 4, 8],
        dtype=dtypes.int32,
    )
    result = gen_spectral_ops.irfft3d(x, fft_length=length, Treal=real_type)
    self.assertDTensorEqual(
        expected_result, Layout(expected_layout_specs, self.mesh), result
    )


class DTensorLayoutPropSPMDTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests for layout propagation through ops during SPMD expansion on the same 2x4 (x, y) mesh as
  # the main test class: checks which output layouts DTensor infers for ops such as reductions,
  # slices, reshapes, einsum, gradients, random ops, resource variables and control flow. The mesh
  # is stored on the instance by setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture. Calls the base setUp, skips on TPU unless exactly 8 cores exist, creates a
    # (2, 4) mesh with dimensions x and y for each device type and stores the one for the current
    # device type as self.mesh. Returns nothing.
    super(DTensorLayoutPropSPMDTest, self).setUp()

    self.skipForDeviceType(['TPU'],
                           'all tests require 8 TPU cores.',
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: Mesh([_MESH_DIM_X, _MESH_DIM_Y], global_ids, local_ids,
                     test_util.create_device_list((2, 4), device))
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)

    self.scalar_replicated_layout = Layout.replicated(self.mesh, rank=0)

    self.replicated_layout_1d = Layout.replicated(self.mesh, rank=1)
    self.first_dimension_sharded_layout_1d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=1)

    self.replicated_layout_2d = Layout.replicated(self.mesh, rank=2)
    self.first_dimension_sharded_layout_2d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=2)
    self.last_dimension_sharded_layout_2d = Layout.inner_sharded(
        self.mesh, _MESH_DIM_X, rank=2)

    self.replicated_layout_3d = Layout.replicated(self.mesh, rank=3)
    self.first_dimension_sharded_layout_3d = Layout.batch_sharded(
        self.mesh, _MESH_DIM_X, rank=3)
    self.middle_dimension_sharded_layout_3d = Layout(
        [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    self.last_dimension_sharded_layout_3d = Layout.inner_sharded(
        self.mesh, _MESH_DIM_X, rank=3)

    # Make a list so that we can index layouts by sharding dimension and then
    # by rank.
    self.layouts = [
        [
            None, self.first_dimension_sharded_layout_1d,
            self.first_dimension_sharded_layout_2d,
            self.first_dimension_sharded_layout_3d
        ],
        [
            None, None, self.last_dimension_sharded_layout_2d,
            self.middle_dimension_sharded_layout_3d
        ],
        [None, None, None, self.last_dimension_sharded_layout_3d],
        # Keep this at the end so a sharding dim of -1 corresponds to
        # replicated.
        [
            None, self.replicated_layout_1d, self.replicated_layout_2d,
            self.replicated_layout_3d
        ],
    ]

  @parameterized.named_parameters(
      ('_collapse_all_dims', [2, 4, 3], 0, [24], 0),
      ('_collapse_partial_dims', [2, 4, 3], 0, [8, 3], 0),
      ('_collapse_partial_dims_with_ones', [2, 4, 1], 0, [8, 1], 0),
      ('_collapse_partial_dims_non_batch', [2, 4, 3], 1, [2, 12], 1),
      ('_collapse_partial_dims_non_batch_with_ones', [1, 4, 3], 1, [1, 12], 1),
      ('_uncollapse_all_dims', [24], 0, [2, 4, 3], 0),
      ('_uncollapse_partial_dims', [4, 4], 0, [2, 2, 4], 0),
      ('_uncollapse_partial_dims_non_batch', [2, 12], 1, [2, 4, 3], 1),
      ('_expand_dims', [2, 2], 0, [2, 1, 2], 0),
      ('_squeeze', [2, 1, 2], 0, [2, 2], 0),
  )
  def testReshape(self, src_shape, src_sharding_dim, target_shape,
                  target_sharding_dim):
    # [FUNCTION DOC]
    # Parameters: src_shape and target_shape (integer lists), src_sharding_dim and
    # target_sharding_dim (index into the table of layouts by sharded dimension; -1 means
    # replicated). Reshapes a random float tensor of src_shape whose layout is looked up by sharding
    # dimension and rank; asserts equality with the plain reshape and the looked-up target layout.
    src_numpy = np.random.uniform(size=src_shape)
    src = constant_op.constant(src_numpy, dtype=dtypes.float32)

    expected = array_ops.reshape(src, target_shape)

    src = api.relayout(src, self.layouts[src_sharding_dim][len(src_shape)])
    dtensor_result = array_ops.reshape(src, target_shape)
    self.assertDTensorEqual(
        expected, self.layouts[target_sharding_dim][len(target_shape)],
        dtensor_result)

  def testReshapeWithAllConcatOutputLayout(self):
    # [FUNCTION DOC]
    # Reshapes a 2x4x3 tensor sharded on x in the middle dimension to 2x12 under a requested default
    # output layout that is fully replicated; asserts equality and that replicated layout.
    src_shape = [2, 4, 3]
    target_shape = [2, 12]
    src_layout = Layout(
        [layout_lib.UNSHARDED, _MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    target_layout = Layout.replicated(self.mesh, rank=2)

    src_numpy = np.random.uniform(size=src_shape)
    src = constant_op.constant(src_numpy, dtype=dtypes.float32)

    expected = array_ops.reshape(src, target_shape)

    src = api.relayout(src, src_layout)
    with api._dtensor_device()._default_layout(target_layout):
      dtensor_result = array_ops.reshape(src, target_shape)
    self.assertDTensorEqual(expected, target_layout, dtensor_result)

  def testReshapeWithSplitOnOutputLayout(self):
    # [FUNCTION DOC]
    # Reshapes a replicated 2x4x3 tensor to 2x12 under a requested default output layout that is
    # sharded on x in the second dimension; asserts equality and that layout.
    src_shape = [2, 4, 3]
    target_shape = [2, 12]
    src_layout = Layout.replicated(self.mesh, rank=3)
    target_layout = Layout([layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh)

    src_numpy = np.random.uniform(size=src_shape)
    src = constant_op.constant(src_numpy, dtype=dtypes.float32)

    expected = array_ops.reshape(src, target_shape)

    src = api.relayout(src, src_layout)
    with api._dtensor_device()._default_layout(target_layout):
      dtensor_result = array_ops.reshape(src, target_shape)
    self.assertDTensorEqual(expected, target_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_shard_on_first_output_dim', [4], [2], -1, 0),
      ('_shard_on_first_input_dim', [4, 4], [1, 2], 0, 0),
  )
  def testTile(self, src_shape, multiples, src_sharding_dim,
               target_sharding_dim):
    # [FUNCTION DOC]
    # Parameters: src_shape, multiples (tile counts), src_sharding_dim and target_sharding_dim
    # (index into the layouts table). Tiles a random tensor whose layout comes from the source
    # sharding dimension, under a requested default output layout from the target sharding
    # dimension (same rank); asserts equality with the plain tile and that target layout.
    src_numpy = np.random.uniform(size=src_shape)
    src = constant_op.constant(src_numpy, dtype=dtypes.float32)

    expected = gen_array_ops.tile(src, multiples)

    src = api.relayout(src, self.layouts[src_sharding_dim][len(src_shape)])
    with api._dtensor_device()._default_layout(
        self.layouts[target_sharding_dim][len(src_shape)]):
      dtensor_result = gen_array_ops.tile(src, multiples)
    self.assertDTensorEqual(expected,
                            self.layouts[target_sharding_dim][len(src_shape)],
                            dtensor_result)

  @parameterized.named_parameters(
      ('_unsharded_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
      ('_x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('_unsharded_x', [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('_y_unsharded', [_MESH_DIM_Y, layout_lib.UNSHARDED]),
      ('_unsharded_y', [layout_lib.UNSHARDED, _MESH_DIM_Y]),
      ('_x_y', [_MESH_DIM_X, _MESH_DIM_Y]),
      ('_y_x', [_MESH_DIM_Y, _MESH_DIM_X]))
  def testConst(self, sharding):
    # [FUNCTION DOC]
    # Parameter sharding: rank-2 sharding spec list. A constant op cannot run eagerly on the DTensor
    # device, so a compiled function takes an ignored zeros DTensor argument (to trigger DTensor
    # execution) and creates a 4x12 float constant under a requested default layout. Asserts the
    # result equals the plain constant with that layout.
    src_numpy = np.random.uniform(size=[4, 12])
    zero_numpy = np.zeros_like(src_numpy)
    expected = constant_op.constant(src_numpy, dtype=dtypes.float32)  # 4x12

    layout = Layout(sharding, self.mesh)
    zeros = api.relayout(zero_numpy, layout)

    # We can't execute const on dtensor device eagerly, so we wrap it in a
    # function and pass a dtensor (which we ignore) to the function in order to
    # trigger dtensor execution.
    @polymorphic_function.function
    def const_test(_):
      # [FUNCTION DOC]
      # Nested helper: ignores its argument and returns a float32 constant of the full 4x12 array
      # created under the requested default layout.
      with api._dtensor_device()._default_layout(layout):
        return constant_op.constant(src_numpy, dtype=dtypes.float32)  # 4x12

    dtensor_result = const_test(zeros)

    self.assertDTensorEqual(expected, layout, dtensor_result)

  @parameterized.named_parameters(
      ('_unsharded_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
      ('_x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('_unsharded_x', [layout_lib.UNSHARDED, _MESH_DIM_X]),
      ('_y_unsharded', [_MESH_DIM_Y, layout_lib.UNSHARDED]),
      ('_unsharded_y', [layout_lib.UNSHARDED, _MESH_DIM_Y]),
      ('_x_y', [_MESH_DIM_X, _MESH_DIM_Y]),
      ('_y_x', [_MESH_DIM_Y, _MESH_DIM_X]))
  def testConstScalar(self, sharding):
    # [FUNCTION DOC]
    # Like testConst but the constant is a single scalar value expanded to shape 4x12 by the
    # constant op. Asserts the result equals the plain constant with the requested layout.
    src_numpy = np.random.uniform()
    zero_numpy = np.zeros(shape=[4, 12])
    # Note: The diff between this test and the one above is `src_numpy` is
    # single value here; while the one above is full shape numpy array.
    expected = constant_op.constant(
        src_numpy, shape=[4, 12], dtype=dtypes.float32)  # 4x12

    layout = Layout(sharding, self.mesh)
    zeros = api.relayout(zero_numpy, layout)

    # We can't execute const on dtensor device eagerly, so we wrap it in a
    # function and pass a dtensor (which we ignore) to the function in order to
    # trigger dtensor execution.
    @polymorphic_function.function
    def const_test(_):
      # [FUNCTION DOC]
      # Nested helper: ignores its argument and returns a float32 constant of shape 4x12 filled
      # from the scalar value, created under the requested default layout.
      with api._dtensor_device()._default_layout(layout):
        return constant_op.constant(
            src_numpy, shape=[4, 12], dtype=dtypes.float32)  # 4x12

    dtensor_result = const_test(zeros)

    self.assertDTensorEqual(expected, layout, dtensor_result)

  def testRandomOpByOp(self):
    # [FUNCTION DOC]
    # Generates a 2x2 int64 stateless random full-int tensor with seed (1, 2), first on the mesh's
    # device type without DTensor, then with the seed copied to the mesh replicated. Asserts equal
    # values with a replicated rank-2 layout (equality only holds for the same device type and
    # fully replicated outputs).
    with ops.device_v2(self.mesh.device_type()):
      seed = constant_op.constant([1, 2])
      expected = gen_stateless_random_ops.stateless_random_uniform_full_int(
          shape=[2, 2], seed=seed, dtype=dtypes.int64)

    seed = api.copy_to_mesh(seed, Layout.replicated(rank=1, mesh=self.mesh))
    dtensor_result = gen_stateless_random_ops.stateless_random_uniform_full_int(
        shape=[2, 2], seed=seed, dtype=dtypes.int64)
    # Note that we only expect the same result (a) for the same device since
    # this determines the algorithm, and (b) for fully-replicated output layouts
    # since device_id hashing does not reproduce exactly the single-machine
    # numbers, only their distribution.
    self.assertDTensorEqual(expected, Layout.replicated(rank=2, mesh=self.mesh),
                            dtensor_result)

  def testRange(self):
    # [FUNCTION DOC]
    # Computes range from 0 to 3 under a requested unsharded default layout; asserts equality with
    # the plain range and that 1-D unsharded layout.
    start = 0
    limit = 3
    expected = math_ops.range(start, limit)

    layout = Layout([layout_lib.UNSHARDED], self.mesh)

    with api._dtensor_device()._default_layout(layout):
      dtensor_result = math_ops.range(start, limit)

    self.assertDTensorEqual(expected, layout, dtensor_result)

  def testBroadcastTo(self):
    # [FUNCTION DOC]
    # Broadcasts a replicated 3-element vector to shape 3x3 under the default mesh; asserts
    # equality with the plain result and a replicated rank-2 layout.
    inputs = constant_op.constant([1, 2, 3])
    shape = [3, 3]
    expected = gen_array_ops.broadcast_to(inputs, shape)

    inputs = api.copy_to_mesh(inputs, Layout.replicated(self.mesh, rank=1))
    with api.default_mesh(self.mesh):
      dtensor_result = gen_array_ops.broadcast_to(inputs, shape)

    self.assertDTensorEqual(expected, Layout.replicated(self.mesh, rank=2),
                            dtensor_result)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Replicated',
          'sharding': [layout_lib.UNSHARDED, layout_lib.UNSHARDED],
      },
      {
          'testcase_name': 'BatchSharded',
          'sharding': [_MESH_DIM_X, layout_lib.UNSHARDED],
      },
      {
          'testcase_name': 'FullySharded',
          'sharding': [_MESH_DIM_X, _MESH_DIM_Y],
      },
  )
  def testTanhGrad(self, sharding):
    # [FUNCTION DOC]
    # Parameter sharding: rank-2 sharding spec (replicated, batch, or fully sharded). Computes the
    # gradient of tanh of a 2x4 float tensor with a gradient tape, plainly and for the relayouted
    # DTensor under the default mesh. Asserts the gradients are equal with the input layout, with a
    # looser tolerance of 1e-4 when running on TPU.
    inputs = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]],
                                  dtype=dtypes.float32)

    with backprop.GradientTape() as tape:
      tape.watch([inputs])
      expected_result = gen_math_ops.tanh(inputs)
    expected_grad = tape.gradient(expected_result, [inputs])

    layout = Layout(sharding, self.mesh)
    inputs = api.relayout(inputs.numpy(), layout)
    with api.default_mesh(self.mesh):
      with backprop.GradientTape() as tape:
        tape.watch([inputs])
        dtensor_result = gen_math_ops.tanh(inputs)
      dtensor_grad = tape.gradient(dtensor_result, [inputs])
    # df2x2 lowers the tanh preceision to 1e-4.
    self.assertDTensorEqual(
        expected_grad[0],
        layout,
        dtensor_grad[0],
        tol=1e-4
        if 'TPU' in self.mesh.local_devices()[0] else test_util.DEFAULT_TOL)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Replicated',
          'shard_type': 'replicated',
      }, {
          'testcase_name': 'BatchSharded',
          'shard_type': 'batch_sharded',
      })
  def testIdentityN(self, shard_type):
    # [FUNCTION DOC]
    # Parameter shard_type: replicated or batch_sharded. Passes two 2x2 float tensors (copied to
    # the mesh replicated, or relayouted batch sharded) through the identity-n op; asserts each
    # output equals its input with the same layout.
    layout = (
        Layout.replicated(self.mesh, rank=2) if shard_type == 'replicated' else
        Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=2))

    a = constant_op.constant([[10., 20.], [30., 40.]])
    b = constant_op.constant([[50., 60.], [70., 80.]])
    expected_c, expected_d = gen_array_ops.identity_n([a, b])

    if shard_type == 'replicated':
      a = api.copy_to_mesh(a, layout)
      b = api.copy_to_mesh(b, layout)
    else:
      a = api.relayout(a, layout)
      b = api.relayout(b, layout)
    dtensor_c, dtensor_d = gen_array_ops.identity_n([a, b])

    self.assertDTensorEqual(expected_c, layout, dtensor_c)
    self.assertDTensorEqual(expected_d, layout, dtensor_d)

  @parameterized.named_parameters(
      {
          'testcase_name': 'Replicated',
          'sharding': [layout_lib.UNSHARDED, layout_lib.UNSHARDED]
      },
      {
          'testcase_name': 'BatchSharded',
          'sharding': [_MESH_DIM_X, layout_lib.UNSHARDED],
      },
      {
          'testcase_name': 'FullySharded',
          'sharding': [_MESH_DIM_X, _MESH_DIM_Y],
      },
  )
  def testArgMax(self, sharding):
    # [FUNCTION DOC]
    # Parameter sharding: rank-2 sharding spec. For each axis 0 and 1, takes argmax (int32 output)
    # of a 2x4 float tensor in that layout; asserts equality with the plain result and a 1-D output
    # layout that keeps the sharding of the non-reduced dimension.
    for axis in [0, 1]:
      inputs = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]],
                                    dtype=dtypes.float32)
      expect_result = math_ops.argmax_v2(
          inputs, axis=axis, output_type=dtypes.int32)

      input_layout = Layout(sharding, self.mesh)
      inputs = api.relayout(inputs.numpy(), input_layout)

      output_layout = Layout([sharding[1 - axis]], self.mesh)
      dtensor_result = math_ops.argmax_v2(
          inputs, axis=axis, output_type=dtypes.int32)
      self.assertDTensorEqual(expect_result, output_layout, dtensor_result)

  @parameterized.named_parameters(('PositiveAxis', 0), ('NegativeAxis', -2))
  def testSplitOpsWithFullyReplicatedInputs(self, split_axis):
    # [FUNCTION DOC]
    # Parameter split_axis: 0 or -2. Splits a replicated random 2x4 tensor into two equal parts on
    # that axis; asserts both results are lists of two tensors equal to the plain split and
    # replicated.
    t = random_ops.random_uniform([2, 4])
    expected_result = array_ops.split(t, 2, axis=split_axis)
    t = api.copy_to_mesh(t, self.replicated_layout_2d)
    dtensor_result = array_ops.split(t, 2, axis=split_axis)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 2)
    self.assertLen(dtensor_result, 2)
    self.assertDTensorEqual(expected_result[0], self.replicated_layout_2d,
                            dtensor_result[0])
    self.assertDTensorEqual(expected_result[1], self.replicated_layout_2d,
                            dtensor_result[1])

  def testSplitOpsWithNonSplitAxisSharded(self):
    # [FUNCTION DOC]
    # Splits a 2x4 tensor sharded on x in dimension 0 into two parts along axis 1; asserts two
    # results equal to the plain split, each keeping the first-dim sharded layout.
    t = random_ops.random_uniform([2, 4])
    expected_result = array_ops.split(t, 2, axis=1)
    t = api.relayout(t, self.first_dimension_sharded_layout_2d)
    dtensor_result = array_ops.split(t, 2, axis=1)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 2)
    self.assertLen(dtensor_result, 2)
    self.assertDTensorEqual(expected_result[0],
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result[0])
    self.assertDTensorEqual(expected_result[1],
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result[1])

  def testSplitOpsWithSplitAxisShardedRaisesError(self):
    # [FUNCTION DOC]
    # Splits a 2x4 tensor sharded on x in the last dimension along that sharded axis; expects an
    # unknown-error exception since splitting over a sharded dimension is unsupported.
    t = random_ops.random_uniform([2, 4])
    t = api.relayout(t, self.last_dimension_sharded_layout_2d)
    with self.assertRaises(errors_impl.UnknownError):
      # Spliting over sharded dimension is not yet supported.
      _ = array_ops.split(t, 2, axis=1)

  @parameterized.named_parameters(('PositiveAxis', 1), ('NegativeAxis', -1))
  def testSplitVOpsWithFullyReplicatedInputs(self, split_axis):
    # [FUNCTION DOC]
    # Parameter split_axis: 1 or -1. Splits a replicated random 4x5 tensor with explicit sizes
    # 1, 3, 1 on that axis; asserts three results equal to the plain split and replicated.
    t = random_ops.random_uniform([4, 5])
    expected_result = array_ops.split(t, [1, 3, 1], axis=split_axis)
    t = api.copy_to_mesh(t, self.replicated_layout_2d)
    dtensor_result = array_ops.split(t, [1, 3, 1], axis=split_axis)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 3)
    self.assertLen(dtensor_result, 3)
    self.assertDTensorEqual(expected_result[0], self.replicated_layout_2d,
                            dtensor_result[0])
    self.assertDTensorEqual(expected_result[1], self.replicated_layout_2d,
                            dtensor_result[1])
    self.assertDTensorEqual(expected_result[2], self.replicated_layout_2d,
                            dtensor_result[2])

  def testSplitVOpsWithNonSplitAxisSharded(self):
    # [FUNCTION DOC]
    # Splits a 4x5 tensor sharded on x in dimension 0 with sizes 1, 3, 1 along axis 1; asserts three
    # results equal to the plain split, each keeping the first-dim sharded layout.
    t = random_ops.random_uniform([4, 5])
    expected_result = array_ops.split(t, [1, 3, 1], axis=1)
    t = api.relayout(t, self.first_dimension_sharded_layout_2d)
    dtensor_result = array_ops.split(t, [1, 3, 1], axis=1)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 3)
    self.assertLen(dtensor_result, 3)
    self.assertDTensorEqual(expected_result[0],
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result[0])
    self.assertDTensorEqual(expected_result[1],
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result[1])
    self.assertDTensorEqual(expected_result[2],
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result[2])

  def testSplitVOpsWithSplitAxisShardedRaisesError(self):
    # [FUNCTION DOC]
    # Splits a 2x4 tensor sharded on x in the last dimension with sizes 1, 1, 2 along that sharded
    # axis; expects an unknown-error exception (unsupported). Note the code calls the same split
    # function as the non-V variant, passing a list of sizes.
    t = random_ops.random_uniform([2, 4])
    t = api.relayout(t, self.last_dimension_sharded_layout_2d)
    with self.assertRaises(errors_impl.UnknownError):
      # Spliting over sharded dimension is not yet supported.
      _ = array_ops.split(t, [1, 1, 2], axis=1)

  def testUnpackWithFullyReplicatedInputs(self):
    # [FUNCTION DOC]
    # Unstacks a replicated 2x4 tensor along axis 0; asserts two results, each equal to the plain
    # one and replicated 1-D.
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops_stack.unstack(t, axis=0)
    t = api.copy_to_mesh(t, self.replicated_layout_2d)
    dtensor_result = array_ops_stack.unstack(t, axis=0)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 2)
    self.assertLen(dtensor_result, 2)
    self.assertDTensorEqual(expected_result[0], self.replicated_layout_1d,
                            dtensor_result[0])
    self.assertDTensorEqual(expected_result[1], self.replicated_layout_1d,
                            dtensor_result[1])

  def testUnpackWithShardedInput(self):
    # [FUNCTION DOC]
    # Unstacks a 2x4 tensor sharded on x in dimension 1 along axis 1; asserts four results, each
    # equal to the plain one and replicated 1-D.
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops_stack.unstack(t, axis=1)
    t = api.relayout(t, Layout([layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh))
    dtensor_result = array_ops_stack.unstack(t, axis=1)
    self.assertIsInstance(expected_result, list)
    self.assertIsInstance(dtensor_result, list)
    self.assertLen(expected_result, 4)
    self.assertLen(dtensor_result, 4)
    for i in range(4):
      self.assertDTensorEqual(expected_result[i], self.replicated_layout_1d,
                              dtensor_result[i])

  @parameterized.named_parameters(
      (
          '_unshard_input_batch_sharded',
          'bsd,dnh->bsnh',
          [8, 128, 128],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
          [128, 16, 64],
          [layout_lib.UNSHARDED, _MESH_DIM_Y, layout_lib.UNSHARDED],
          [
              _MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y,
              layout_lib.UNSHARDED
          ],
      ),
      (
          '_unshard_input_all_reduce_output',
          'bfi,bfd->di',
          [8, 128, 256],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
          [8, 128, 128],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
          [layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
      (
          '_contracting_dim_sharded_in_output',
          'bfnh,nhd->bfd',
          [8, 128, 16, 8],
          [
              _MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y,
              layout_lib.UNSHARDED
          ],
          [16, 8, 128],
          [_MESH_DIM_Y, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
          [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y],
      ),
  )
  def testEinsum(self, equation, a_shape, a_layout, b_shape, b_layout,
                 output_layout):
    # [FUNCTION DOC]
    # Parameters: equation (einsum subscript string), shapes and sharding spec lists of the two
    # operands, and the desired output sharding spec. Computes einsum on random float32 operands
    # plainly, and on relayouted DTensors inside a compiled function that relayouts the result to
    # the output layout. Asserts equal values and the output layout.
    a_numpy = np.random.uniform(size=a_shape)
    b_numpy = np.random.uniform(size=b_shape)
    a = constant_op.constant(a_numpy, dtype=dtypes.float32)
    b = constant_op.constant(b_numpy, dtype=dtypes.float32)

    expected = special_math_ops.einsum(equation, a, b)

    a_layout = Layout(a_layout, self.mesh)
    b_layout = Layout(b_layout, self.mesh)
    output_layout = Layout(output_layout, self.mesh)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)

    @polymorphic_function.function
    def einsum_fn(x, y):
      # [FUNCTION DOC]
      # Nested helper: computes einsum of the two inputs with the enclosing equation and relayouts
      # the result to the requested output layout; returns it.
      result = special_math_ops.einsum(equation, x, y)
      return api.relayout(result, output_layout)

    dtensor_result = einsum_fn(a, b)
    self.assertDTensorEqual(expected, output_layout, dtensor_result)

  def testAddV2DifferentSharding(self):
    # [FUNCTION DOC]
    # Adds two random 16x8 tensors, one sharded on y and one on x in dimension 0, in a compiled
    # function; asserts equality with the plain sum and that the output takes the first
    # operand's layout (sharded on y).
    a_numpy = np.random.uniform(size=[16, 8])
    b_numpy = np.random.uniform(size=[16, 8])
    a = constant_op.constant(a_numpy, dtype=dtypes.float32)
    b = constant_op.constant(b_numpy, dtype=dtypes.float32)

    expected = math_ops.add(a, b)

    a_layout = Layout([_MESH_DIM_Y, layout_lib.UNSHARDED], self.mesh)
    b_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)

    a = api.relayout(a, a_layout)
    b = api.relayout(b, b_layout)

    @polymorphic_function.function
    def add_fn(x, y):
      # [FUNCTION DOC]
      # Nested helper: returns the element-wise sum of the two inputs.
      return math_ops.add(x, y)

    dtensor_result = add_fn(a, b)
    self.assertDTensorEqual(expected, a_layout, dtensor_result)

  @parameterized.named_parameters(test_util_ops.BINARY_FLOAT_OPS)
  def testBinaryOpsWithArbitrarySharding(self, op):
    # [FUNCTION DOC]
    # Parameterized over float binary ops. Applies op to two 2x2 tensors, one sharded on x in dim 0
    # and one on x in dim 1, under a requested default output layout sharded on dim 1; asserts
    # equality with the plain result and that layout (tolerance via select_tol).
    tol = select_tol(op, self.mesh, test_util.DEFAULT_TOL, low_res_tol=1e-2)
    a = constant_op.constant(
        np.array([[1., 2.], [3., 4.]]), dtype=dtypes.float32)
    b = constant_op.constant(
        np.array([[10., 20.], [30., 40.]]), dtype=dtypes.float32)
    expected_result = op(a, b)

    layout_x_n = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    layout_n_x = Layout([layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh)

    a = api.relayout(a, layout_x_n)
    b = api.relayout(b, layout_n_x)

    with api._dtensor_device()._default_layout(layout_n_x):
      dtensor_result = op(a, b)

    self.assertDTensorEqual(
        expected_result, layout_n_x, dtensor_result, tol=tol)

  def testUnsortedSegmentSum(self):
    # [FUNCTION DOC]
    # Skipped on GPU. Unsorted segment sum with 12 segments over replicated 12x4 data and int32
    # segment ids sharded on y; asserts equality with the plain result and a replicated rank-2
    # layout.
    self.skipForDeviceType(['GPU'], 'reduce on GPU only supports int64')
    num_segments = 12
    data = np.random.uniform(size=[num_segments, 4])
    segment_ids = np.random.randint(
        0, num_segments, size=num_segments, dtype=np.int32)
    expected = gen_math_ops.unsorted_segment_sum(data, segment_ids,
                                                 num_segments)

    data = api.relayout(data, Layout.replicated(self.mesh, 2))
    segment_ids = api.relayout(
        segment_ids, Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1)
    )
    with api.default_mesh(self.mesh):
      dtensor_result = gen_math_ops.unsorted_segment_sum(
          data, segment_ids, num_segments)
      expected_layout = Layout.replicated(self.mesh, 2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testUnsortedSegmentSumWithShardingData(self):
    # [FUNCTION DOC]
    # Skipped on GPU. Unsorted segment sum with 12 segments where the float32 data is sharded on x
    # in the last dimension and ids are replicated, under a requested replicated default layout;
    # asserts equality with the plain result and a replicated rank-2 layout.
    self.skipForDeviceType(['GPU'], 'reduce on GPU only supports int64')
    num_segments = 12
    data = np.random.uniform(size=[num_segments, 4]).astype(np.float32)
    segment_ids = np.random.randint(
        0, num_segments, size=num_segments, dtype=np.int32)
    expected = gen_math_ops.unsorted_segment_sum(data, segment_ids,
                                                 num_segments)

    data = api.relayout(
        data, Layout([layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh)
    )
    segment_ids = api.relayout(segment_ids, Layout.replicated(self.mesh, 1))

    with api._dtensor_device()._default_layout(Layout.replicated(self.mesh, 2)):
      with api.default_mesh(self.mesh):
        dtensor_result = gen_math_ops.unsorted_segment_sum(
            data, segment_ids, num_segments)
        expected_layout = Layout.replicated(self.mesh, 2)
        self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParams(self):
    # [FUNCTION DOC]
    # Gathers rows (axis 0) of a 1000x4 float table sharded on x in dim 0 with replicated 4x4 int32
    # indices under the default mesh; asserts equality with the plain gather and a fully
    # replicated rank-3 layout.
    params = np.arange(1000 * 4).reshape((1000, 4)).astype(np.float32)
    # "batch" size = 2, num_indices = 4 per example
    indices = np.random.randint(
        0, 1000, size=4 * 4).reshape((4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, axis=0)

    params = api.relayout(
        params, layout=Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    )
    indices = api.relayout(
        indices, Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    )

    expected_layout = Layout(
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
        self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(params, indices, axis=0)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_float32', np.float32),
      ('_int32', np.int32))
  @mock.patch.dict(os.environ,
                   {'LOWER_DTENSOR_GATHER_TO_COLLECTIVE_GATHER_V2': '1'})
  def testGatherIndicesShardingParams(self, data_type):
    # [FUNCTION DOC]
    # Parameter data_type: numpy float32 or int32 for the table. An environment variable is patched
    # to lower the gather to a collective gather. Gathers axis 0 of a 1000x4 table sharded on x in
    # dim 0 using 4x4 indices also sharded on x in dim 0; asserts equality with the plain gather
    # and a rank-3 layout sharded on x in dim 0.
    params = np.arange(1000 * 4).reshape((1000, 4)).astype(data_type)
    # "batch" size = 2, num_indices = 4 per example
    indices = np.random.randint(
        0, 1000, size=4 * 4).reshape((4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, axis=0)

    params = api.relayout(
        params, layout=Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    )
    indices = api.relayout(
        indices, Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    )

    expected_layout = Layout(
        [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(params, indices, axis=0)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsAxisIs1(self):
    # [FUNCTION DOC]
    # Gathers along axis 1 of a 1000x32 table sharded on x in dim 0 with replicated 4x4 indices;
    # asserts equality with the plain gather and a rank-3 layout sharded on x in dim 0.
    params = np.arange(1000 * 32).reshape((1000, 32)).astype(np.float32)
    # "batch" size = 2, num_indices = 4 per example
    indices = np.random.randint(
        0, 32, size=4 * 4).reshape((4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, axis=1)

    params = api.relayout(
        params, layout=Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)
    )
    indices = api.relayout(
        indices, Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    )

    expected_layout = Layout(
        [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(params, indices, axis=1)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingIndices(self):
    # [FUNCTION DOC]
    # Gathers axis 0 of a replicated 1000x4 table with replicated 4x4 indices under a requested
    # default output layout with the middle dimension sharded on y; asserts equality with the plain
    # gather and that layout.
    params = np.arange(1000 * 4).reshape((1000, 4))
    # "batch" size = 2, num_indices = 4 per example
    indices = np.random.randint(0, 1000, size=4 * 4).reshape((4, 4))
    expected = array_ops.gather_v2(params, indices, axis=0)

    params = api.relayout(params, layout=Layout.replicated(self.mesh, 2))
    indices = api.relayout(
        indices, Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    )
    expected_layout = Layout(
        [layout_lib.UNSHARDED, _MESH_DIM_Y, layout_lib.UNSHARDED], self.mesh)
    with api.default_mesh(self.mesh):
      with api._dtensor_device()._default_layout(expected_layout):
        dtensor_result = array_ops.gather_v2(params, indices, axis=0)
        self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsWithBatchDim(self):
    # [FUNCTION DOC]
    # Gathers with one batch dimension (axis 1) from a 128x1000x2 table sharded on x in the batch
    # dimension with replicated 128x4x4 indices; asserts equality with the plain gather and a
    # rank-4 layout sharded on x in dim 0.
    params = np.arange(128 * 1000 * 2).reshape(
        (128, 1000, 2)).astype(np.float32)
    indices = np.random.randint(
        0, 1000, size=128 * 4 * 4).reshape((128, 4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=1)

    params = api.relayout(
        params,
        layout=Layout(
            [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh
        ),
    )
    indices = api.relayout(
        indices,
        Layout(
            [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
            self.mesh,
        ),
    )

    expected_layout = Layout([
        _MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED,
        layout_lib.UNSHARDED
    ], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=1)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsWithBatchDimAxisIs2(self):
    # [FUNCTION DOC]
    # Gathers with one batch dimension along axis 2 from a 128x1000x32 table sharded on x (batch)
    # and y (dimension 1) with replicated 128x4x4 indices; asserts equality with the plain gather
    # and a rank-4 result layout sharded on x then y, with the other two dimensions unsharded.
    params = np.arange(128 * 1000 * 32).reshape(
        (128, 1000, 32)).astype(np.float32)
    indices = np.random.randint(
        0, 32, size=128 * 4 * 4).reshape((128, 4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=2)

    params = api.relayout(
        params,
        layout=Layout(
            [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED], self.mesh
        ),
    )
    indices = api.relayout(
        indices,
        Layout(
            [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
            self.mesh,
        ),
    )

    expected_layout = Layout(
        [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
        self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsWithBatchDimAxisIs2ShardingAfterAxis(self):
    # [FUNCTION DOC]
    # Gathers with one batch dimension along axis 2 from a 128x5x32x64 table sharded on x (batch)
    # and y (last dimension, after the gather axis) with replicated 128x4x4 indices; asserts
    # equality with the plain gather and a rank-5 layout sharded on x in dim 0 and y in the last dim.
    params = np.arange(128 * 5 * 32 * 64).reshape(
        (128, 5, 32, 64)).astype(np.float32)
    indices = np.random.randint(
        0, 32, size=128 * 4 * 4).reshape((128, 4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=2)

    params = api.relayout(
        params,
        layout=Layout(
            [
                _MESH_DIM_X,
                layout_lib.UNSHARDED,
                layout_lib.UNSHARDED,
                _MESH_DIM_Y,
            ],
            self.mesh,
        ),
    )
    indices = api.relayout(
        indices,
        Layout(
            [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
            self.mesh,
        ),
    )

    expected_layout = Layout([
        _MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED,
        layout_lib.UNSHARDED, _MESH_DIM_Y
    ], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsWithBatchDimAxisIs2ShardingIndices(self):
    # [FUNCTION DOC]
    # Gathers with one batch dimension along axis 2 from a 128x5x32x64 table sharded on x (batch)
    # with 128x4x4 indices sharded on y in the middle dimension; asserts equality with the plain
    # gather and a rank-5 layout sharded on x in dim 0 and y in dim 2.
    params = np.arange(128 * 5 * 32 * 64).reshape(
        (128, 5, 32, 64)).astype(np.float32)
    indices = np.random.randint(
        0, 32, size=128 * 4 * 4).reshape((128, 4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=2)

    params = api.relayout(
        params,
        layout=Layout(
            [
                _MESH_DIM_X,
                layout_lib.UNSHARDED,
                layout_lib.UNSHARDED,
                layout_lib.UNSHARDED,
            ],
            self.mesh,
        ),
    )
    indices = api.relayout(
        indices,
        Layout(
            [layout_lib.UNSHARDED, _MESH_DIM_Y, layout_lib.UNSHARDED], self.mesh
        ),
    )

    expected_layout = Layout([
        _MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y, layout_lib.UNSHARDED,
        layout_lib.UNSHARDED
    ], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=2)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherParamsShardingAfterAxisWithBatchDim(self):
    # [FUNCTION DOC]
    # Gathers with one batch dimension along axis 1 from a 128x1000x2 table sharded on x in the
    # last dimension (after the gather axis) with replicated 128x4 indices; asserts equality with the
    # plain gather and a rank-3 layout sharded on x in the last dimension.
    params = np.arange(128 * 1000 * 2).reshape(
        (128, 1000, 2)).astype(np.float32)
    indices = np.random.randint(
        0, 1000, size=128 * 4).reshape((128, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=1)

    params = api.relayout(
        params,
        layout=Layout(
            [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh
        ),
    )
    indices = api.relayout(
        indices, Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    )

    expected_layout = Layout(
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, _MESH_DIM_X], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=1)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testGatherShardingParamsIndicesWithBatchDim(self):
    # [FUNCTION DOC]
    # Gathers with one batch dimension along axis 1 from a 128x1000x2 table and 128x4x4 indices,
    # both sharded on x in the batch dimension; asserts equality with the plain gather and a rank-4
    # layout sharded on x in dim 0.
    params = np.arange(128 * 1000 * 2).reshape(
        (128, 1000, 2)).astype(np.float32)
    indices = np.random.randint(
        0, 1000, size=128 * 4 * 4).reshape((128, 4, 4)).astype(np.int32)
    expected = array_ops.gather_v2(params, indices, batch_dims=1, axis=1)

    params = api.relayout(
        params,
        layout=Layout(
            [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh
        ),
    )
    indices = api.relayout(
        indices,
        Layout(
            [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh
        ),
    )

    expected_layout = Layout([
        _MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED,
        layout_lib.UNSHARDED
    ], self.mesh)
    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.gather_v2(
          params, indices, batch_dims=1, axis=1)
      self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testTranspose(self):
    # [FUNCTION DOC]
    # Transposes a 5x4x6 float tensor laid out as unsharded, y, x with each of the six
    # permutations of its three axes under the default mesh; asserts equality with the plain
    # transpose and that the output layout is the input sharding permuted the same way.
    original_a = np.arange(5 * 4 * 6).reshape((5, 4, 6)).astype(np.float32)

    original_layout = Layout([layout_lib.UNSHARDED, _MESH_DIM_Y, _MESH_DIM_X],
                             self.mesh)

    # paris of (perm, expected_layout)
    combinations = [
        ([2, 0, 1], [_MESH_DIM_X, layout_lib.UNSHARDED, _MESH_DIM_Y]),
        ([2, 1, 0], [_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED]),
        ([1, 2, 0], [_MESH_DIM_Y, _MESH_DIM_X, layout_lib.UNSHARDED]),
        ([1, 0, 2], [_MESH_DIM_Y, layout_lib.UNSHARDED, _MESH_DIM_X]),
        ([0, 1, 2], [layout_lib.UNSHARDED, _MESH_DIM_Y, _MESH_DIM_X]),
        ([0, 2, 1], [layout_lib.UNSHARDED, _MESH_DIM_X, _MESH_DIM_Y]),
    ]

    for (perm, expected_spec) in combinations:
      a = original_a
      expected = array_ops.transpose_v2(a, perm)

      a = api.relayout(a, original_layout)
      expected_layout = Layout(expected_spec, self.mesh)
      with api.default_mesh(self.mesh):
        dtensor_result = array_ops.transpose_v2(a, perm)
        self.assertDTensorEqual(expected, expected_layout, dtensor_result)

  def testSliceOpsWithNonFullSlicingOnShardedInputs(self):
    # [FUNCTION DOC]
    # Slices one row and two columns out of a 2x4 tensor sharded on x in dim 0 under the default
    # mesh; since the sliced dimension is not fully covered, asserts equality with the plain slice
    # and a fully unsharded rank-2 output layout.
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops.slice(t, [0, 0], [1, 2])
    sharded_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)

    t = api.relayout(t, sharded_layout)
    expected_layout = Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED],
                             self.mesh)

    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.slice(t, [0, 0], [1, 2])
      self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  def testSliceOpWithNonFullSlicingOnShardedInputsAndFullSlicingOnAnother(self):
    # [FUNCTION DOC]
    # Slices one row and all four columns of a 2x4 tensor sharded on x and y; asserts equality with
    # the plain slice and an output layout with dimension 0 unsharded (partially sliced) and
    # dimension 1 still sharded on y (fully sliced).
    t = constant_op.constant([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    expected_result = array_ops.slice(t, [0, 0], [1, 4])
    sharded_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y], self.mesh)

    t = api.relayout(t, sharded_layout)
    expected_layout = Layout([layout_lib.UNSHARDED, _MESH_DIM_Y], self.mesh)

    with api.default_mesh(self.mesh):
      dtensor_result = array_ops.slice(t, [0, 0], [1, 4])
      self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  def testSliceOpsWithNonFullSlicingOnShardedInputsWithShardedOutputs(self):
    # [FUNCTION DOC]
    # Slices the top-left 2x2 block of a 4x4 tensor sharded on x in dim 0 inside a compiled function
    # that relayouts the slice back to sharded on x in dim 0; asserts equality with the plain
    # slice and that layout.
    t = constant_op.constant([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [11., 12., 13., 14.],
        [15., 16., 17., 18.],
    ])
    expected_result = array_ops.slice(t, [0, 0], [2, 2])
    sharded_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)

    t = api.relayout(t, sharded_layout)
    expected_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)

    with api.default_mesh(self.mesh):

      @polymorphic_function.function
      def op_fn(x):
        # [FUNCTION DOC]
        # Nested helper: slices the top-left 2x2 block of x and relayouts it to the expected layout
        # (sharded on x in dim 0); returns it.
        y = array_ops.slice(x, [0, 0], [2, 2])
        return api.relayout(y, expected_layout)

      dtensor_result = op_fn(t)
      self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  def testSliceOpsWithFullyReplicatedInputsWithShardedOutputs(self):
    # [FUNCTION DOC]
    # Slices the top-left 2x2 block of a replicated 4x4 tensor inside a compiled function that
    # relayouts the slice to sharded on x in dim 0; asserts equality with the plain slice and that
    # layout.
    t = constant_op.constant([
        [1., 2., 3., 4.],
        [5., 6., 7., 8.],
        [11., 12., 13., 14.],
        [15., 16., 17., 18.],
    ])
    expected_result = array_ops.slice(t, [0, 0], [2, 2])
    operand_layout = Layout([layout_lib.UNSHARDED, layout_lib.UNSHARDED],
                            self.mesh)

    t = api.relayout(t, operand_layout)
    expected_layout = Layout([_MESH_DIM_X, layout_lib.UNSHARDED], self.mesh)

    with api.default_mesh(self.mesh):

      @polymorphic_function.function
      def op_fn(x):
        # [FUNCTION DOC]
        # Nested helper: slices the top-left 2x2 block of x and relayouts it to the expected layout
        # (sharded on x in dim 0); returns it.
        y = array_ops.slice(x, [0, 0], [2, 2])
        return api.relayout(y, expected_layout)

      dtensor_result = op_fn(t)
      self.assertDTensorEqual(expected_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      test_util.product(
          [('_tensor_unsharded_updates_unsharded', -1, -1, 2),
           ('_tensor_first_updates_unsharded', 0, -1, 2),
           ('_tensor_second_updates_unsharded', 1, -1, 2),
           ('_tensor_unsharded_updates_first', -1, 0, 2),
           ('_tensor_first_updates_first', 0, 0, 2),
           ('_tensor_second_updates_first', 1, 0, 2),
           ('_tensor_unsharded_updates_second', -1, 1, 2),
           ('_tensor_first_updates_second', 0, 1, 2),
           ('_tensor_second_updates_second', 1, 1, 2),
           ('_tensor_second_updates_second_rank_three_indices', 1, 1, 3)], [(
               'update',
               gen_array_ops.tensor_scatter_update,
           ), (
               'add',
               gen_array_ops.tensor_scatter_add,
           )]))
  def testTensorScatterUpdate(self, tensor_dimension, updates_dimension,
                              indices_rank, op_type):
    # [FUNCTION DOC]
    # Parameters: tensor_dimension and updates_dimension (index into the layouts table: sharded
    # dimension, -1 for replicated), indices_rank (2 or 3) and op_type (tensor scatter update or
    # add). Builds a random 4x2 tensor, random updates, and two distinct random indices in range;
    # applies op_type plainly and with the tensor and updates relayouted and replicated indices.
    # Asserts equal values and that the output is sharded in the second dimension if either input
    # is, otherwise replicated.
    tensor_layout = self.layouts[tensor_dimension][2]
    updates_layout = self.layouts[updates_dimension][indices_rank]

    # Tensor is shape [4, 2], indices is shape [..., 2, 1], updates is shape
    # [..., 2, 2]
    #
    # Entries in indices should be unique and integers in the range [0, 4).
    # Tensor and updates are float.

    tensor_numpy = np.random.uniform(size=[4, 2])
    padding_axes = [1] * (indices_rank - 2)
    updates_numpy = np.random.uniform(size=padding_axes + [2, 2])
    indices_numpy_flat = np.array(
        [np.random.randint(0, 4),
         np.random.randint(0, 3)])
    if indices_numpy_flat[0] == indices_numpy_flat[1]:
      indices_numpy_flat[1] += 1
    indices_numpy = indices_numpy_flat.reshape(padding_axes + [2, 1])

    tensor = constant_op.constant(tensor_numpy, dtype=dtypes.float32)
    updates = constant_op.constant(updates_numpy, dtype=dtypes.float32)
    indices = constant_op.constant(indices_numpy, dtype=dtypes.int32)

    golden_result = op_type(tensor=tensor, updates=updates, indices=indices)

    tensor = api.relayout(tensor, tensor_layout)
    updates = api.relayout(updates, updates_layout)
    indices = api.relayout(
        indices, Layout.replicated(tensor_layout.mesh, indices_rank)
    )

    dtensor_result = op_type(tensor=tensor, updates=updates, indices=indices)

    # If either of the inputs are sharded in the non-index dimension, then
    # the output is as well, otherwise it is replicated.
    if tensor_dimension == 1 or updates_dimension == 1:
      expected_layout = self.layouts[1][2]
    else:
      expected_layout = self.layouts[-1][2]

    self.assertDTensorEqual(golden_result, expected_layout, dtensor_result)

  @parameterized.named_parameters(
      ('_params_unsharded_indices_unsharded', -1, -1),
      ('_params_first_indices_unsharded', 0, -1),
      ('_params_third_indices_unsharded', 2, -1),
      ('_params_unsharded_indices_first', -1, 0),
      ('_params_first_indices_first', 0, 0),
      ('_params_third_indices_first', 2, 0),
      ('_params_unsharded_indices_second', -1, 1),
      ('_params_first_indices_second', 0, 1),
      ('_params_third_indices_second', 2, 1),
  )
  def testGatherNd(self, params_dimension, indices_dimension):
    # [FUNCTION DOC]
    # Parameters: params_dimension and indices_dimension (index into the layouts table; -1 means
    # replicated). Skipped on GPU. Gathers slices from a random 6x4x4 tensor using random 2x2
    # indices (result 2x4). Asserts equal values and the expected layout: first-dim sharded when
    # params is not sharded in the last dim and indices is sharded in the first dim; middle/last
    # sharded when params is sharded in the last dim; otherwise replicated.
    self.skipForDeviceType(['GPU'], 'b/179387248 cases with AllConcat crash')
    params_layout = self.layouts[params_dimension][3]
    indices_layout = self.layouts[indices_dimension][2]

    # Params will have shape [6, 4, 4] and indices will have shape [2, 2]
    # this will result in a tensor with final shape [2, 4].

    params_numpy = np.random.uniform(size=[6, 4, 4])
    indices_numpy = np.array(
        [[np.random.randint(0, 6),
          np.random.randint(0, 4)],
         [np.random.randint(0, 6),
          np.random.randint(0, 4)]])

    params = constant_op.constant(params_numpy, dtype=dtypes.float32)
    indices = constant_op.constant(indices_numpy, dtype=dtypes.int32)

    golden_result = gen_array_ops.gather_nd(params=params, indices=indices)

    params = api.relayout(params, params_layout)
    indices = api.relayout(indices, indices_layout)

    dtensor_result = gen_array_ops.gather_nd(params=params, indices=indices)

    if params_dimension < 2 and indices_dimension == 0:
      # if params isn't sharded in the last dimension, then sharding of indices
      # the first dimension gives a first dimension sharding of the output
      expected_layout = self.layouts[0][2]
    elif params_dimension == 2:
      # if params is sharded in the last dimension, then sharding of indices is
      # ignored as they are both sharded on the same dimension.
      expected_layout = self.layouts[1][2]
    else:
      expected_layout = self.layouts[-1][2]

    self.assertDTensorEqual(golden_result, expected_layout, dtensor_result)

  def testGatherNdUnshardedInputShardedOutput(self):
    # [FUNCTION DOC]
    # Gather-nd with replicated 6x4x4 params and replicated random 2x2 indices inside a compiled
    # function that relayouts the 2x4 result to first-dim sharded; asserts equality with the plain
    # result and that layout.
    # Params will have shape [6, 4, 4] and indices will have shape [2, 2]
    # this will result in a tensor with final shape [2, 4].

    params_numpy = np.random.uniform(size=[6, 4, 4])
    indices_numpy = np.array(
        [[np.random.randint(0, 6),
          np.random.randint(0, 4)],
         [np.random.randint(0, 6),
          np.random.randint(0, 4)]])

    params = constant_op.constant(params_numpy, dtype=dtypes.float32)
    indices = constant_op.constant(indices_numpy, dtype=dtypes.int32)

    golden_result = gen_array_ops.gather_nd(params=params, indices=indices)

    params = api.relayout(params, self.replicated_layout_3d)
    indices = api.relayout(indices, self.replicated_layout_2d)

    @polymorphic_function.function
    def gather_with_relayout(params, indices):
      # [FUNCTION DOC]
      # Nested helper: gather-nd of params at indices, relayouted to first-dim sharded rank 2;
      # returns it.
      result = gen_array_ops.gather_nd(params=params, indices=indices)
      return api.relayout(result, self.first_dimension_sharded_layout_2d)

    dtensor_result = gather_with_relayout(params, indices)

    self.assertDTensorEqual(golden_result,
                            self.first_dimension_sharded_layout_2d,
                            dtensor_result)

  @parameterized.named_parameters(
      ('_unsharded_unsharded', [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
      ('_x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
      ('_x_y', [_MESH_DIM_X, _MESH_DIM_Y]))
  def testTopK(self, sharding):
    # [FUNCTION DOC]
    # Parameter sharding: rank-2 sharding spec. Takes top-2 values and indices along the last axis
    # of a random 4x16 float tensor in that layout; asserts both outputs equal the plain ones and
    # have layout with dim 0 sharding kept and the last dim unsharded.
    inputs_layout = Layout(sharding, self.mesh)

    inputs = constant_op.constant(
        np.random.uniform(size=[4, 16]), dtype=dtypes.float32)
    topk = constant_op.constant(2, dtype=dtypes.int32)

    expected_topk_values, expected_topk_indices = nn_ops.top_k(inputs, k=topk)

    inputs = api.relayout(inputs, inputs_layout)

    dtensor_topk_values, dtensor_topk_indices = nn_ops.top_k(inputs, k=topk)

    expected_sharding = [sharding[0], layout_lib.UNSHARDED]
    expected_layout = Layout(expected_sharding, self.mesh)

    self.assertDTensorEqual(expected_topk_values, expected_layout,
                            dtensor_topk_values)
    self.assertDTensorEqual(expected_topk_indices, expected_layout,
                            dtensor_topk_indices)

  @parameterized.named_parameters(*test_util.product(
      (('_targets_unsharded', [layout_lib.UNSHARDED]),
       ('_targets_x', [_MESH_DIM_X]), ('_targets_y', [_MESH_DIM_Y])),
      (('_predictions_unsharded_unsharded',
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED]),
       ('_predictions_x_unsharded', [_MESH_DIM_X, layout_lib.UNSHARDED]),
       ('_predictions_unsharded_y', [layout_lib.UNSHARDED, _MESH_DIM_Y]),
       ('_predictions_x_y', [_MESH_DIM_X, _MESH_DIM_Y]))))
  def testInTopK(self, targets_sharding, predictions_sharding):
    # [FUNCTION DOC]
    # Parameters: targets_sharding (rank-1 sharding spec) and predictions_sharding (rank-2 spec).
    # Evaluates in-top-k (k=2) of four class targets against a 4x4 predictions matrix (int64 integer
    # type on GPU, int32 otherwise) with relayouted inputs. Asserts equal values and a 1-D output
    # layout using the sharding of the predictions if sharded, else the targets' sharding, else
    # unsharded.

    # TODO(b/193471732): changed back to dtypes.int32 once it is fixed.
    int_dtype = dtypes.int64 if self.mesh.device_type(
    ) == 'GPU' else dtypes.int32

    targets_layout = Layout(targets_sharding, self.mesh)
    predictions_layout = Layout(predictions_sharding, self.mesh)

    targets = constant_op.constant([2, 2, 1, 0], dtype=int_dtype)
    predictions = constant_op.constant([[0.1, 0.3, 0.2, 0.4],
                                        [0.1, 0.2, 0.3, 0.4],
                                        [0.3, 0.4, 0.1, 0.2],
                                        [0.1, 0.3, 0.4, 0.2]])
    topk = constant_op.constant(2, dtype=int_dtype)

    expected_output = nn_ops.in_top_k_v2(targets, predictions, k=topk)

    targets = api.relayout(targets, targets_layout)
    predictions = api.relayout(predictions, predictions_layout)

    dtensor_output = nn_ops.in_top_k_v2(targets, predictions, k=topk)

    expected_sharding = [layout_lib.UNSHARDED]
    # Select the more sharded layout
    if targets_sharding[0] != layout_lib.UNSHARDED:
      expected_sharding[0] = targets_sharding[0]
    if predictions_sharding[0] != layout_lib.UNSHARDED:
      expected_sharding[0] = predictions_sharding[0]

    expected_layout = Layout(expected_sharding, self.mesh)
    self.assertDTensorEqual(expected_output, expected_layout, dtensor_output)


class DTensorRelayoutTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests for the relayout API on a 2x4 (x, y) mesh, in eager and in compiled (graph) execution,
  # covering relayouts that gather (all-concat) a sharded dimension and that slice (split) an
  # unsharded dimension. The mesh is stored by setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture. Calls the base setUp, skips on TPU unless exactly 8 cores exist, builds a (2, 4)
    # mesh with dimensions x and y (using the XLA SPMD flag per device type) for each device type,
    # stores the one for the current device type as self.mesh and ensures the eager context is
    # initialized. Returns nothing.
    super(DTensorRelayoutTest, self).setUp()

    self.skipForDeviceType(['TPU'],
                           'all tests require 8 TPU cores.',
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: Mesh([_MESH_DIM_X, _MESH_DIM_Y], global_ids, local_ids,
                     test_util.create_device_list((2, 4), device),
                     use_xla_spmd=test_util.get_use_xla_spmd(device))
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)
    context.ensure_initialized()

  def testRelayoutEagerAllConcat(self):
    # [FUNCTION DOC]
    # Applies relu to a 2x4x1 tensor sharded on x and y, then eagerly relayouts the result to
    # sharded on x only (gathering the second dimension); asserts equality with the plain relu and
    # the final layout.
    op = gen_nn_ops.relu

    a = constant_op.constant([[[1.], [-2.], [3.], [-4.]],
                              [[5.], [-6.], [-7.], [8.]]])
    assert a.shape == [2, 4, 1]
    expected_result = op(a)

    init_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                         self.mesh)
    a = api.relayout(a, init_layout)
    dtensor_output = op(a)

    final_layout = Layout(
        [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    # eager relayout
    dtensor_result = api.relayout(dtensor_output, final_layout)
    self.assertDTensorEqual(expected_result, final_layout, dtensor_result)

  def testRelayoutEagerSlice(self):
    # [FUNCTION DOC]
    # Applies relu to a replicated 2x4x1 tensor, then eagerly relayouts the result to sharded on x
    # and y (splitting replicated data); asserts equality with the plain relu and the final layout.
    op = gen_nn_ops.relu

    a = constant_op.constant([[[1.], [-2.], [3.], [-4.]],
                              [[5.], [-6.], [-7.], [8.]]])
    assert a.shape == [2, 4, 1]
    expected_result = op(a)

    init_layout = Layout(
        [layout_lib.UNSHARDED, layout_lib.UNSHARDED, layout_lib.UNSHARDED],
        self.mesh)
    a = api.relayout(a, init_layout)
    dtensor_output = op(a)

    final_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                          self.mesh)
    # eager relayout
    dtensor_result = api.relayout(dtensor_output, final_layout)
    self.assertDTensorEqual(expected_result, final_layout, dtensor_result)

  def testRelayoutGraphAllConcat(self):
    # [FUNCTION DOC]
    # Same as testRelayoutEagerAllConcat but the relu and relayout (x,y sharded to x sharded) run
    # inside a compiled function; asserts equality with the plain relu and the final layout.
    op = gen_nn_ops.relu

    a = constant_op.constant([[[1.], [-2.], [3.], [-4.]],
                              [[5.], [-6.], [-7.], [8.]]])
    assert a.shape == [2, 4, 1]
    expected_result = op(a)

    init_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                         self.mesh)
    a = api.relayout(a, init_layout)

    final_layout = Layout(
        [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)

    @polymorphic_function.function
    def wrap_fn(x):
      # [FUNCTION DOC]
      # Nested helper: applies relu to x and relayouts the result to the final layout; returns it.
      dtensor_output = op(x)
      return api.relayout(dtensor_output, final_layout)

    dtensor_result = wrap_fn(a)
    self.assertDTensorEqual(expected_result, final_layout, dtensor_result)

  def testRelayoutGraphSlice(self):
    # [FUNCTION DOC]
    # Same as testRelayoutEagerSlice but starting from a tensor sharded on x only and relayouting
    # to sharded on x and y inside a compiled function; asserts equality and the final layout.
    op = gen_nn_ops.relu

    a = constant_op.constant([[[1.], [-2.], [3.], [-4.]],
                              [[5.], [-6.], [-7.], [8.]]])
    assert a.shape == [2, 4, 1]
    expected_result = op(a)

    init_layout = Layout(
        [_MESH_DIM_X, layout_lib.UNSHARDED, layout_lib.UNSHARDED], self.mesh)
    a = api.relayout(a, init_layout)

    final_layout = Layout([_MESH_DIM_X, _MESH_DIM_Y, layout_lib.UNSHARDED],
                          self.mesh)

    @polymorphic_function.function
    def wrap_fn(x):
      # [FUNCTION DOC]
      # Nested helper: applies relu to x and relayouts the result to the final layout; returns it.
      dtensor_output = op(x)
      return api.relayout(dtensor_output, final_layout)

    dtensor_result = wrap_fn(a)
    self.assertDTensorEqual(expected_result, final_layout, dtensor_result)


if __name__ == '__main__':
  tf_test.main()
