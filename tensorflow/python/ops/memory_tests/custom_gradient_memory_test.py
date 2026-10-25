# [MODULE DOC]
# Memory tests for the recompute-gradient decorator of the custom gradient module. They compare
# peak accelerator memory (measured through device memory statistics, or through compiled XLA
# buffer assignment sizes) of a deep chain of matrix multiplications with and without gradient
# recomputation, expecting recomputation to use clearly less memory while giving the same
# gradient. Requires an accelerator for the non-XLA test (skipped on CPU and TPU); the XLA test is
# unconditionally skipped. Imports an XLA protobuf module at import time.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Memory tests for tensorflow.ops.custom_gradient."""

import functools
import unittest

from absl.testing import parameterized
from xla.service import hlo_pb2
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class RecomputeGradMemoryTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case checking that wrapping a function with the recompute-gradient
  # decorator lowers peak memory during gradient computation, in eager and graph modes (non-XLA)
  # and under XLA compilation. Provides helper methods to pick the device type and to build a
  # gradient function from a given function.

  def _get_device_type(self):
    # [FUNCTION DOC]
    # HELPER: Picks the device for the memory tests.
    # Returns the string GPU if a physical GPU exists, else TPU if a physical TPU exists, else
    # CPU. Takes no arguments and has no side effects.
    for accelerator in ["GPU", "TPU"]:
      if config.list_physical_devices(accelerator):
        return accelerator
    return "CPU"

  def _grad(self, test_func, argnums=0):
    # [FUNCTION DOC]
    # HELPER: Builds the gradient function used by both memory tests.
    # Takes a function and the index (default zero) of the argument to differentiate against.
    # Returns a new function that, given positional parameters, records them on a gradient tape,
    # calls test_func, and returns the gradient of the output with respect to the selected
    # parameter.

    def _f(*params):
      # [FUNCTION DOC]
      # HELPER: Nested gradient function returned by the gradient builder above.
      # Watches all given parameters on a gradient tape, evaluates the wrapped function on them
      # and returns the gradient of its output with respect to the parameter chosen by the
      # enclosing argument index.
      with backprop.GradientTape() as tape:
        tape.watch(params)
        output = test_func(*params)
      return tape.gradient(output, params[argnums])

    return _f

  @parameterized.named_parameters(
      (f"_{mode}", mode) for mode in ["eager", "graph"])
  @test_util.run_v2_only
  def testRecomputeGradNonXla(self, mode):
    # [FUNCTION DOC]
    # Test run twice, with mode eager and mode graph (v2 behavior only). Skipped on TPU (needs
    # XLA) and on CPU (memory info unsupported). On an accelerator, resets memory statistics,
    # makes a 500 by 500 half-precision ones matrix, and defines a function applying five matrix
    # squarings, wrapped in an outer function applying it five times and returning one element.
    # Computes gradients with and without the recompute-gradient decorator (eagerly or inside a
    # compiled function), recording peak memory above the baseline. Asserts: without recompute the
    # peak is at least the size of 25 matrices; with recompute it is at least ten matrices and
    # less than fifteen matrices; and that the two gradient results are numerically close.
    device_type = self._get_device_type()
    device_name = f"{device_type}:0"

    if device_type == "TPU":
      self.skipTest("XLA is required for TPU.")

    if device_type == "CPU":
      self.skipTest("b/185371422: get_memory_info does't support CPU yet.")

    config.reset_memory_stats(device_name)
    base_memory = config.get_memory_info(device_name)["current"]
    n = 500
    with ops.device(device_name):
      a = array_ops.ones((n, n), dtype=dtypes.float16)

    def f(x):
      # [FUNCTION DOC]
      # HELPER: Test workload. Multiplies the matrix by itself five times in a row and returns
      # the result.
      for _ in range(5):
        x = math_ops.matmul(x, x)
      return x

    def g(f, x):
      # [FUNCTION DOC]
      # HELPER: Test workload driver. Applies the given function f five times in sequence to x
      # and returns the first element of the result (a scalar), giving a scalar to differentiate.
      for _ in range(5):
        x = f(x)
      return x[0][0]

    def run(test_func):
      # [FUNCTION DOC]
      # HELPER: Runs the gradient of test_func at the matrix on the chosen device, directly in
      # eager mode or through a compiled function in graph mode (according to the enclosing mode
      # parameter). Returns the gradient.
      with ops.device(device_name):
        if mode == "eager":
          return self._grad(test_func)(a)
        else:
          return def_function.function(self._grad(test_func))(a)

    f_no_recompute = functools.partial(g, f)
    f_recompute = functools.partial(g, custom_gradient.recompute_grad(f))

    # The result is not saved so the base memory will stay the same.
    run(f_no_recompute)
    peak_memory_no_recompute = (
        config.get_memory_info(device_name)["peak"] - base_memory)

    config.reset_memory_stats(device_name)
    run(f_recompute)
    peak_memory_recompute = (
        config.get_memory_info(device_name)["peak"] - base_memory)

    # 2 * n * n (size of `a`) * 5 (loop of f) * 5 (loop of g)
    self.assertGreaterEqual(peak_memory_no_recompute, 2 * n * n * 5 * 5)
    # 2 * n * n (size of `a`) * (5 (loop of g) + 5 (recompute in f))
    self.assertGreaterEqual(peak_memory_recompute, 2 * n * n * 5 * 2)
    # peak_memory_recompute should be less than peak_memory_no_recompute.
    self.assertLess(peak_memory_recompute, 2 * n * n * 5 * 3)

    res_no_recompute = run(f_no_recompute)
    res_recompute = run(f_recompute)
    self.assertAllClose(res_no_recompute, res_recompute)

  @unittest.skip("b/335476600")
  @test_util.run_v2_only
  def testRecomputeGradXla(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: This test is unconditionally skipped (reason references a bug id), so the
    # XLA path is currently not exercised.
    # Scenario: same chain of five-by-five matrix squarings as the non-XLA test, on a 500 by 500
    # matrix (bfloat16 on TPU, float32 otherwise), but memory is measured statically from the
    # optimised XLA program's buffer allocations. Initialises the TPU system first if running on
    # TPU. Asserts that the total allocation without recompute is at least 25 matrices, with
    # recompute at least ten and below fifteen matrices, and that results with and without
    # recompute are numerically close.
    device_type = self._get_device_type()
    device_name = f"{device_type}:0"
    # Necessary for TFRT tests.
    if device_type == "TPU":
      tpu_cluster_resolver.initialize_tpu_system()

    n = 500
    with ops.device(device_name):
      # XLA:TPU converts f32 matmuls to bf16, and XLA:CPU converts bf16/f16
      # matmuls to f32 after cl/461262189.  Use a type that doesn't get
      # converted.
      if device_type == "TPU":
        dtype = dtypes.bfloat16
        elem_size = 2
      else:
        dtype = dtypes.float32
        elem_size = 4
      a = array_ops.zeros((n, n), dtype=dtype)  # elem_size * n * n bytes

    def f(x):
      # [FUNCTION DOC]
      # HELPER: Test workload. Multiplies the matrix by itself five times (matrix products are
      # not fused by the compiler) and returns the result.
      for _ in range(5):
        # matmul can not be fused by XLA.
        x = math_ops.matmul(x, x)
      return x

    def g(f, x):
      # [FUNCTION DOC]
      # HELPER: Test workload driver. Applies f five times in sequence to x and returns the
      # first element of the result (a scalar).
      for _ in range(5):
        x = f(x)
      return x[0][0]

    def get_peak_memory(test_func):
      # [FUNCTION DOC]
      # HELPER: Measures memory of a function under XLA. Compiles the gradient of test_func
      # with just-in-time compilation, fetches the serialized optimised program proto for the
      # device, parses it, and returns the sum of sizes of all buffer allocations in bytes.
      test_func = def_function.function(self._grad(test_func), jit_compile=True)
      # The hlo_proto contains statically allocated memory info of HLO values.
      hlo_proto_serialized = test_func.experimental_get_compiler_ir(a)(
          stage="optimized_hlo_proto_serialized", device_name=device_name)

      hlo_proto = hlo_pb2.HloProto.FromString(hlo_proto_serialized)
      allocations = hlo_proto.buffer_assignment.buffer_allocations
      return sum(getattr(allocation, "size") for allocation in allocations)

    f_no_recompute = functools.partial(g, f)
    f_recompute = functools.partial(g, custom_gradient.recompute_grad(f))

    peak_memory_no_recompute = get_peak_memory(f_no_recompute)
    peak_memory_recompute = get_peak_memory(f_recompute)

    # elem_size * n * n (size of `a`) * 5 (loop of g) * 5 (loop of f)
    self.assertGreaterEqual(peak_memory_no_recompute, elem_size * n * n * 5 * 5)
    # elem_size * n * n (size of `a`) * (5 (loop of g) + 5 (recompute in f))
    self.assertGreaterEqual(peak_memory_recompute, elem_size * n * n * 5 * 2)
    # peak_memory_recompute should be less than peak_memory_no_recompute.
    self.assertLess(peak_memory_recompute, elem_size * n * n * 5 * 3)

    with ops.device(device_name):
      res_recompute = f_recompute(a)
      res_no_recompute = f_no_recompute(a)

    self.assertAllClose(res_recompute, res_no_recompute)


if __name__ == "__main__":
  googletest.main()
