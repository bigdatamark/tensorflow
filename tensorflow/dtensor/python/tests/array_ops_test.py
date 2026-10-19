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
# [MODULE DOC]
# Test module for DTensor support in array creation ops: fill, zeros, ones, zeros-like and
# ones-like with layouts, in eager and graph modes. Runs the TensorFlow test main when executed.
"""Tests for DTensor support in array_ops."""

import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import combinations
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

Layout = layout_lib.Layout

_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_MESH_DIMS = [_MESH_DIM_X, _MESH_DIM_Y]


class ArrayOpsTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests DTensor layout support in array creation ops (fill, zeros, ones and their like
  # variants). Fixture: a 2x1 mesh with dimensions x and y registered for CPU, GPU and TPU.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: after the base setup, creates 2x1 meshes with dimensions x and y for CPU, GPU
    # and TPU device types and registers them, storing the active one as the test mesh.
    super().setUp()

    global_ids = test_util.create_device_ids_array((2, 1))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: layout_lib.Mesh(
            _MESH_DIMS,
            global_ids,
            local_ids,
            test_util.create_device_list((2, 1), device),
        )
        for device in ('CPU', 'GPU', 'TPU')
    }
    self.mesh = self.configTestMesh(mesh_dict)

  @combinations.generate(
      combinations.combine(is_graph=[False, True], size=[32, 4096])
  )
  def testTwoFills(self, is_graph, size):
    # [FUNCTION DOC]
    # Parameterized over graph or eager mode and size. Fills two vectors with zero, one with an
    # x batch-sharded layout and one with y, in a single function under the default mesh;
    # asserts both equal a plain zeros tensor and carry their requested layouts.
    layout_x = Layout.batch_sharded(self.mesh, _MESH_DIM_X, rank=1)
    layout_y = Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1)

    def fn():
      # [FUNCTION DOC]
      # Returns a pair of zero-filled vectors, the first with the x layout, the second with y.
      return (
          array_ops.fill([size], 0.0, layout=layout_x),
          array_ops.fill([size], 0.0, layout=layout_y),
      )

    if is_graph:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.mesh):
      dtensor_x, dtensor_y = fn()
    tensor = array_ops.zeros([size], layout=None)

    self.assertDTensorEqual(tensor, layout_x, dtensor_x)
    self.assertDTensorEqual(tensor, layout_y, dtensor_y)

  @combinations.generate(
      combinations.combine(
          is_graph=[False, True],
          size=[32, 4096],
          nullary_op=[array_ops.zeros, array_ops.ones],
      )
  )
  def testNullaryOp(self, is_graph, size, nullary_op):
    # [FUNCTION DOC]
    # Parameterized over graph or eager mode, size and the zeros / ones ops. Builds a plain
    # tensor and a DTensor with a y batch-sharded layout under the default mesh; asserts they
    # are equal in value and the DTensor has that layout.
    layout_y = Layout.batch_sharded(self.mesh, _MESH_DIM_Y, rank=1)

    tensor = nullary_op([size], layout=None)

    def fn():
      # [FUNCTION DOC]
      # Calls the nullary op for the given size with the y batch-sharded layout.
      return nullary_op([size], layout=layout_y)

    if is_graph:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.mesh):
      dtensor = fn()

    self.assertDTensorEqual(tensor, layout_y, dtensor)

  @combinations.generate(
      combinations.combine(
          is_graph=[False, True],
          size=[32, 4096],
          nullary_op=[array_ops.zeros_like_v2, array_ops.ones_like_v2],
      )
  )
  def testNullaryLikeOpWithLayout(self, is_graph, size, nullary_op):
    # [FUNCTION DOC]
    # Parameterized over graph or eager mode, size and the zeros-like / ones-like ops. Applies
    # the like-op to a DTensor sharded on x while requesting a layout sharded on y, under the
    # default mesh, and asserts the result equals the plain like-result with the y layout.
    layout_x = Layout.batch_sharded(self.mesh, batch_dim=_MESH_DIM_X, rank=1)
    layout_y = Layout.batch_sharded(self.mesh, batch_dim=_MESH_DIM_Y, rank=1)

    tensor = array_ops.zeros([size], layout=None)
    tensor_like = nullary_op(tensor, layout=None)
    dtensor = array_ops.zeros([size], layout=layout_x)
    self.assertDTensorEqual(tensor, layout_x, dtensor)

    def fn(layout):
      # [FUNCTION DOC]
      # Applies the like-op to the sharded DTensor using the given layout.
      return nullary_op(dtensor, layout=layout)

    if is_graph:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.mesh):
      dtensor_like = fn(layout_y)

    self.assertDTensorEqual(tensor_like, layout_y, dtensor_like)

  @combinations.generate(
      combinations.combine(
          is_graph=[True],
          size=[32, 4096],
          nullary_op=[array_ops.zeros_like_v2, array_ops.ones_like_v2],
      )
  )
  def testNullaryLikeOpWithoutLayoutEager(self, is_graph, size, nullary_op):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says Eager, but the parameterization only uses is_graph true, so the
    # function is always wrapped in a graph function; the Graph variant below uses false (eager).
    # Parameterized over size and the zeros-like / ones-like ops. Creates a plain tensor and a
    # DTensor sharded on x, applies the like-op to the DTensor with no layout under the default
    # mesh, and asserts the result equals the plain like-result with a fully replicated layout.
    layout_x = Layout.batch_sharded(self.mesh, batch_dim=_MESH_DIM_X, rank=1)
    layout_replicated = Layout.replicated(self.mesh, rank=1)

    tensor = array_ops.zeros([size], layout=None)
    tensor_like = nullary_op(tensor, layout=None)
    dtensor = array_ops.zeros([size], layout=layout_x)
    self.assertDTensorEqual(tensor, layout_x, dtensor)

    def fn(layout):
      # [FUNCTION DOC]
      # Applies the like-op to the sharded DTensor using the given layout (may be nothing).
      return nullary_op(dtensor, layout=layout)

    if is_graph:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.mesh):
      dtensor_like = fn(None)

    self.assertDTensorEqual(tensor_like, layout_replicated, dtensor_like)

  @combinations.generate(
      combinations.combine(
          is_graph=[False],
          size=[32, 4096],
          nullary_op=[array_ops.zeros_like_v2, array_ops.ones_like_v2],
      )
  )
  def testNullaryLikeOpWithoutLayoutGraph(self, is_graph, size, nullary_op):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says Graph, but the parameterization only uses is_graph false, so the
    # function runs eagerly; the Eager variant above is the graph one.
    # Parameterized over size and the zeros-like / ones-like ops. Applies the like-op with no
    # layout to a DTensor sharded on x under the default mesh and asserts the result keeps the
    # input's x batch-sharded layout (unlike the other variant, which gets a replicated layout).
    layout_x = Layout.batch_sharded(self.mesh, batch_dim=_MESH_DIM_X, rank=1)

    tensor = array_ops.zeros([size], layout=None)
    tensor_like = nullary_op(tensor, layout=None)
    dtensor = array_ops.zeros([size], layout=layout_x)
    self.assertDTensorEqual(tensor, layout_x, dtensor)

    def fn(layout):
      # [FUNCTION DOC]
      # Applies the like-op to the sharded DTensor using the given layout (may be nothing).
      return nullary_op(dtensor, layout=layout)

    if is_graph:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.mesh):
      dtensor_like = fn(None)

    self.assertDTensorEqual(tensor_like, layout_x, dtensor_like)


if __name__ == '__main__':
  test.main()
