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
# Test module for the DTensor device caches used for compiled functions and eager ops. It checks
# cache hits, misses and memory growth through the device statistics. Runs the TensorFlow test
# main when executed directly.
"""Tests DTensor device cache for compiled function computation."""

import gc
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_stateless_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

# Convenient constants to use for tests.
_BATCH_DIM = "batch"
_MESH_DIM_X = "x"

# Shorter notation.
Layout = layout_lib.Layout
Mesh = layout_lib.Mesh


def diff_dicts(dict1, dict2):
  # [FUNCTION DOC]
  # HELPER: computes per-key differences between two statistics mappings for the cache tests.
  # Takes two mappings from names to numbers; returns a mapping over the union of keys with the
  # first value minus the second, treating a missing key as zero. No side effects.
  keys = set(dict1.keys()) | set(dict2.keys())
  return {key: dict1.get(key, 0) - dict2.get(key, 0) for key in keys}


class DTensorDeviceCacheTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests the DTensor device's caches (function manager, kernel cache, device cache, eager pure
  # optimization) by reading the device statistics before and after operations and checking hit,
  # miss and size changes. Fixture: a one-dimensional mesh of two devices for CPU, GPU and TPU.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: after base setup, builds a mesh named batch with two devices for CPU, GPU and
    # TPU, registers it and stores the active one as the test mesh.
    super(DTensorDeviceCacheTest, self).setUp()
    device_ids = test_util.create_device_ids_array((2,))
    local_device_ids = np.ravel(device_ids).tolist()
    mesh_dict = {
        device: Mesh(
            [_BATCH_DIM],
            device_ids,
            local_device_ids,
            test_util.create_device_list((2,), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def testBasic(self):
    # [FUNCTION DOC]
    # Calls two compiled functions on replicated DTensors that differ in layout, shape and dtype
    # and asserts each call returns the correct value (no cache confusion): inputs differing in
    # layout (rank), shape, dtype, and a second function with a different name.

    @polymorphic_function.function
    def func0(a):
      # [FUNCTION DOC]
      # Returns the input plus one.
      return a + 1

    @polymorphic_function.function
    def func1(a):
      # [FUNCTION DOC]
      # Returns the input plus two.
      return a + 2

    c0 = api.copy_to_mesh(
        constant_op.constant(1.0), Layout.replicated(self.mesh, rank=0)
    )
    c1 = api.copy_to_mesh(
        constant_op.constant([2.0, 3.0]), Layout.replicated(self.mesh, rank=1)
    )
    c2 = api.copy_to_mesh(
        constant_op.constant([4.0]), Layout.replicated(self.mesh, rank=1)
    )
    c3 = api.copy_to_mesh(
        constant_op.constant(1, dtype=dtypes.int32),
        Layout.replicated(self.mesh, rank=0),
    )

    # c0 and c1 have different layouts. c1 and c2 have different shapes.
    # c0 and c3 have different dtypes.
    self.assertAllEqual(func0(c0), 2.0)
    self.assertAllEqual(func0(c1), [3.0, 4.0])
    self.assertAllEqual(func0(c2), [5.0])
    self.assertAllEqual(func0(c3), 2)

    # func0 and func1 have different names.
    self.assertAllEqual(func1(c0), 3.0)

  def testFunctionInputConstantFoldingCacheHits(self):
    # [FUNCTION DOC]
    # Calls a compiled add function twice with the same two replicated scalar inputs and checks
    # the result is 38 both times. From the device statistics, asserts the function manager had
    # exactly one miss (first call) and one hit (second call).

    @polymorphic_function.function
    def add(a, b):
      # [FUNCTION DOC]
      # Returns the sum of the two inputs.
      return a + b

    c0 = api.copy_to_mesh(
        constant_op.constant(17.0), Layout.replicated(self.mesh, rank=0)
    )
    c1 = api.copy_to_mesh(
        constant_op.constant(21.0), Layout.replicated(self.mesh, rank=0)
    )

    stats1 = api._dtensor_device()._get_stats()
    self.assertAllEqual(add(c0, c1), 38.0)
    self.assertAllEqual(add(c0, c1), 38.0)

    # First call should miss and second should hit.
    stats2 = api._dtensor_device()._get_stats()
    diff = {key: stats2[key] - stats1[key] for key in stats1.keys()}
    self.assertEqual(diff["function_manager.miss"], 1)
    self.assertEqual(diff["function_manager.hit"], 1)

  def testFunctionInputConstantFoldingCacheMiss(self):
    # [FUNCTION DOC]
    # Calls a compiled add function four times with different scalar input combinations and
    # checks sums. Because input values are folded as constants, a call with a changed first or
    # second input misses the cache while an exact repeat hits. Asserts three misses and one hit
    # in the function manager statistics.

    @polymorphic_function.function
    def add(a, b):
      # [FUNCTION DOC]
      # Returns the sum of the two inputs.
      return a + b

    c0 = api.copy_to_mesh(
        constant_op.constant(17.0), Layout.replicated(self.mesh, rank=0)
    )
    c1 = api.copy_to_mesh(
        constant_op.constant(21.0), Layout.replicated(self.mesh, rank=0)
    )
    c2 = api.copy_to_mesh(
        constant_op.constant(0.0), Layout.replicated(self.mesh, rank=0)
    )

    stats1 = api._dtensor_device()._get_stats()
    # First call should log a cache miss.
    self.assertAllEqual(add(c0, c1), 38.0)

    # Second call should also log a cache miss since second constant changed.
    self.assertAllEqual(add(c0, c2), 17.0)

    # Third call should not log a cache miss since the same input as the prev.
    self.assertAllEqual(add(c0, c2), 17.0)

    # Fourth call should log a cache miss since first input changed.
    self.assertAllEqual(add(c1, c2), 21.0)

    stats2 = api._dtensor_device()._get_stats()
    diff = {key: stats2[key] - stats1[key] for key in stats1.keys()}
    self.assertEqual(diff["function_manager.miss"], 3)
    self.assertEqual(diff["function_manager.hit"], 1)

  def testCacheWithRNG(self):
    # [FUNCTION DOC]
    # Under a replicated default layout, draws stateless normal random values with the same
    # shape and seed twice, a different shape, and a different seed. Asserts equal draws for the
    # same shape and seed, matching replicated layouts, different shapes differ, and different
    # seeds give different values (the cache does not confuse them).
    with api._dtensor_device()._default_layout(
        Layout.replicated(self.mesh, rank=1)):
      v0 = gen_stateless_random_ops.stateless_random_normal(
          shape=[1], seed=[1, 2]
      )

    with api._dtensor_device()._default_layout(
        Layout.replicated(self.mesh, rank=1)):
      v1 = gen_stateless_random_ops.stateless_random_normal(
          shape=[1], seed=[1, 2]
      )
      v2 = gen_stateless_random_ops.stateless_random_normal(
          shape=[2], seed=[1, 2]
      )
      v3 = gen_stateless_random_ops.stateless_random_normal(
          shape=[1], seed=[3, 4]
      )

    # v0 and v1 have same layouts.
    self.assertAllEqual(v0, v1)
    api.check_layout(v0, Layout.replicated(self.mesh, rank=1))
    api.check_layout(v1, Layout.replicated(self.mesh, rank=1))
    # v1 and v2 have different shapes.
    self.assertNotEqual(v1.shape, v2.shape)
    # v1 and v3 have different seeds.
    self.assertNotEqual(v1.numpy(), v3.numpy())

  def testCacheWithVariable(self):
    # [FUNCTION DOC]
    # Creates DTensor variables from a replicated scalar, a replicated vector and a
    # batch-sharded packed tensor. Asserts reading the variables returns the original values and
    # that unpacking the sharded variable returns its per-device pieces.
    c0 = api.copy_to_mesh(
        constant_op.constant(1.0), Layout.replicated(self.mesh, rank=0)
    )
    c1 = api.copy_to_mesh(
        constant_op.constant([2.0, 3.0]), Layout.replicated(self.mesh, rank=1)
    )
    a = constant_op.constant([4.0])
    b = constant_op.constant([5.0])
    c2 = api.pack(
        [a, b], layout=Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=1)
    )

    v0 = d_variable.DVariable(c0)
    v1 = d_variable.DVariable(c1)
    v2 = d_variable.DVariable(c2)

    self.assertAllEqual(v0.read_value(), 1.0)
    self.assertAllEqual(v1.read_value(), [2.0, 3.0])
    unpacked_tensor = api.unpack(v2.read_value())
    self.assertAllClose([4.0], unpacked_tensor[0])
    self.assertAllClose([5.0], unpacked_tensor[1])

  @combinations.generate(
      combinations.combine(size=[16, 40], same_value=[True, False])
  )
  def testManyFunctions(self, size, same_value):
    # [FUNCTION DOC]
    # Parameterized over input size and whether all ten inputs are the same value. Defines ten
    # compiled sum functions in a loop (one per input, same name), runs each, then deletes the
    # function and forces garbage collection. Asserts the function manager, kernel cache and
    # device cache sizes are unchanged from before, i.e. the caches do not leak.
    r = range(100)

    values = [np.reshape(r[i : i + size], (4, size // 4)) for i in range(10)]
    c_layout = Layout.replicated(self.mesh, rank=2)
    values = [constant_op.constant(v, dtype=dtypes.float32) for v in values]
    c0 = [api.copy_to_mesh(v, c_layout) for v in values]

    c0 = [c0[0 if same_value else i] for i in range(10)]
    e0 = [values[0 if same_value else i] for i in range(10)]
    stats1 = api._dtensor_device()._get_stats()

    for i in range(10):
      # Use a special to ensure no conflicts with otherwise used names.
      @polymorphic_function.function
      def fn_31415926(c):
        # [FUNCTION DOC]
        # Returns the sum of all elements of the input.
        return math_ops.reduce_sum(c)

      self.assertAllEqual(fn_31415926(c0[i]).numpy(), np.sum(e0[i]))

    del fn_31415926
    gc.collect()

    stats2 = api._dtensor_device()._get_stats()
    diff = diff_dicts(stats2, stats1)
    self.assertEqual(diff["function_manager.size"], 0)
    self.assertEqual(diff["kernel_cache.size"], 0)
    self.assertEqual(diff["device_cache.size"], 0)

  @combinations.generate(
      combinations.combine(size=[16, 40], same_value=[True, False])
  )
  def testManyEagerOps(self, size, same_value):
    # [FUNCTION DOC]
    # Parameterized over size and same_value. Runs an identity eager op on ten replicated inputs
    # and compares cache statistics before and after. Expectations depend on device type: on
    # CPU/GPU the op runs locally (zero cache growth, ten eager-pure hits); on TPU the function
    # manager and kernel cache grow by amounts that depend on same_value and size (a known TPU
    # issue). Kernel and device cache sizes are only checked when TFRT is not enabled.
    # KNOWN ISSUE: the non-same-value branches for size at least 20 and below 20 assert exactly
    # the same things (duplicated code), and the comment mentions leaks to be fixed.
    if self.mesh.device_type() != "TPU":
      # For the CPU/GPU mesh, we have a shortcut that doesn't go through the
      # MLIR, but run the eager op locally and broadcast to all the devices.
      expected_cache_diff = 0
      expected_kernel_cache = 0
      expected_device_cache = 0
      expected_eager_pure_hit = 10
    else:
      # TODO(b/287529295): Remove this branch after the TPU issue is fixed.
      expected_device_cache = 0
      expected_eager_pure_hit = 0
      if same_value:
        expected_cache_diff = 1
        expected_kernel_cache = 2
      else:
        if size >= 20:
          expected_cache_diff = 1
          expected_kernel_cache = 2
        else:
          expected_cache_diff = 2
          expected_kernel_cache = 4

    r = range(100)
    c_layout = Layout.replicated(self.mesh, rank=2)
    values = [np.reshape(r[i : i + size], (4, size // 4)) for i in range(10)]
    values = [constant_op.constant(v, dtype=dtypes.float32) for v in values]
    c0 = [api.copy_to_mesh(v, c_layout) for v in values]

    c0 = [c0[0 if same_value else i] for i in range(10)]
    e0 = [values[0 if same_value else i] for i in range(10)]

    stats1 = api._dtensor_device()._get_stats()

    for i in range(10):
      self.assertAllEqual(array_ops.identity(c0[i]).numpy(), e0[i])

    gc.collect()

    stats2 = api._dtensor_device()._get_stats()
    diff = diff_dicts(stats2, stats1)

    if same_value:
      self.assertEqual(diff["function_manager.size"], expected_cache_diff)
      self.assertEqual(
          diff["eager_pure_optimization.hit"], expected_eager_pure_hit
      )
      # TFRT doesn't use eager cache.
      if not test_util.is_tfrt_enabled():
        self.assertEqual(diff["kernel_cache.size"], expected_kernel_cache)
        self.assertEqual(diff["device_cache.size"], expected_device_cache)
    else:
      # FIXME(feyu): Update these when the leaks are fixed.
      if size >= 20:
        self.assertEqual(diff["function_manager.size"], expected_cache_diff)
        self.assertEqual(
            diff["eager_pure_optimization.hit"], expected_eager_pure_hit
        )
        # TFRT doesn't use eager cache.
        if not test_util.is_tfrt_enabled():
          self.assertEqual(diff["kernel_cache.size"], expected_kernel_cache)
          self.assertEqual(diff["device_cache.size"], expected_device_cache)
      else:
        self.assertEqual(diff["function_manager.size"], expected_cache_diff)
        self.assertEqual(
            diff["eager_pure_optimization.hit"], expected_eager_pure_hit
        )
        # TFRT doesn't use eager cache.
        if not test_util.is_tfrt_enabled():
          self.assertEqual(diff["kernel_cache.size"], expected_kernel_cache)
          self.assertEqual(diff["device_cache.size"], expected_device_cache)

  def testManyEagerOpsVaryInput(self):
    # [FUNCTION DOC]
    # Copies a rank-10 constant to a replicated rank-10 layout and reduces it by sum along each
    # of the ten axes in turn, asserting each result equals the plain numeric sum along that axis.
    c_layout = Layout.replicated(self.mesh, rank=10)

    c0 = constant_op.constant(
        [[[[[[[[[[0, 1, 2, 3], [4, 5, 6, 7]]]]]]]]]], dtype=dtypes.float32
    )
    e0 = c0.numpy()
    c0 = api.copy_to_mesh(c0, c_layout)

    for ax in range(10):
      self.assertAllEqual(
          math_ops.reduce_sum(c0, axis=ax).numpy(), np.sum(e0, axis=ax)
      )


if __name__ == "__main__":
  test.main()
