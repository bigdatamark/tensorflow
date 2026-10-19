# [MODULE DOC]
# Tests and micro-benchmarks for the runtime mode in which a single eager op is wrapped in a call
# op (a small function) before being executed. Defines a micro-benchmark class that times matrix
# multiplication, bitwise-and and random-normal on CPU and GPU for several sizes, a functional test
# class that exercises many op kinds under the "eager op as function" mode (default attribute
# values, list-typed inputs with differing arity, dataset map, critical sections), and an internals
# test class that checks, through metric counters, that simple graphs run synchronously on a
# single-threaded executor. No import-time side effects beyond constants for device names; the
# test runner is started when the file is executed as a script.
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
"""Tests for wrapping an eager op in a call op at runtime."""
import time

from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import benchmarks_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import critical_section_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_inspect


def run_benchmark(func, num_iters, unused_execution_mode):
  # [FUNCTION DOC]
  # HELPER: timing callback handed to the benchmark base class's report runner.
  # Calls the given zero-argument callable once as a warm-up, then calls it num_iters more times
  # and returns the wall-clock seconds taken by the timed iterations only. The third argument
  # (execution mode) is ignored. Raises whatever the callable raises; no other side effects.
  # warm up
  func()
  start = time.time()
  for _ in range(num_iters):
    func()
  end = time.time()
  return end - start


CPU = "/device:CPU:0"
GPU = "/device:GPU:0"


# TODO(srbs): Why can't we use absl parameterized here?
@test_util.with_eager_op_as_function
class MicroBenchmarks(benchmarks_test_base.MicroBenchmarksBase):
  # [CLASS DOC]
  # Micro-benchmarks of single eager ops (matmul, bitwise and, random normal) run with the
  # eager-op-as-function mode enabled by the class decorator. Holds pre-built random 2x2, 100x100
  # and 1000x1000 float and int32 matrices as instance attributes; each public benchmark method
  # picks one matrix and a device (CPU or GPU) and reports mean microseconds per call.

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Calls the base class constructor and creates random uniform matrices of sizes
    # 2x2, 100x100 and 1000x1000, each in float and (values below 5) 32-bit integer flavors,
    # stored as instance attributes for use by the benchmark methods. Takes no arguments.
    super().__init__()
    self._m_2_by_2 = random_ops.random_uniform((2, 2))
    self._m_2_by_2_int32 = random_ops.random_uniform((2, 2),
                                                     maxval=5,
                                                     dtype=dtypes.int32)
    self._m_100_by_100 = random_ops.random_uniform((100, 100))
    self._m_100_by_100_int32 = random_ops.random_uniform((100, 100),
                                                         maxval=5,
                                                         dtype=dtypes.int32)
    self._m_1000_by_1000 = random_ops.random_uniform((1000, 1000))
    self._m_1000_by_1000_int32 = random_ops.random_uniform((1000, 1000),
                                                           maxval=5,
                                                           dtype=dtypes.int32)

  def _get_benchmark_name(self):
    # [FUNCTION DOC]
    # HELPER: derives the reported benchmark name for the benchmark base class.
    # Walks the call stack from the outermost frame inward looking for a frame whose "self" is a
    # benchmark instance and takes that frame's function name, skipping frames named as a
    # decorator wrapper. Appends a suffix when the TFRT runtime is enabled and another when the
    # eager-op-as-function mode is enabled. Returns the name string. Raises a value error if no
    # benchmark frame is found. Reads global context state; no side effects.
    """Copied from benchmarks_test.py."""
    stack = tf_inspect.stack()
    name = None
    for frame in stack[::-1]:
      f_locals = frame[0].f_locals
      f_self = f_locals.get("self", None)
      if isinstance(f_self, test.Benchmark):
        name = frame[3]  # Get the method name
        # This is a hack to get around the fact that some methods might have a
        # disable_tfrt decorator around them. In that case a function called
        # 'decorated' wraps the real called function underneath and so we
        # peek one deeper into the stack to get the real name.
        if name == "decorated":
          continue
        else:
          break
    if name is None:
      raise ValueError("Unable to determine calling Benchmark function.")
    if context.is_tfrt_enabled():
      name = name + "_tfrt"
    if context.run_eager_op_as_function_enabled():
      name = name + "_eager_op_as_function"
    return name

  def _run(self, func, num_iters):
    # [FUNCTION DOC]
    # HELPER: runs one benchmark. Passes the timing function, the callable and the iteration count
    # to the base class report runner, requesting the mean time in microseconds per iteration.
    # Returns nothing; the result is recorded by the benchmark reporting machinery.
    self.run_report(run_benchmark, func, num_iters, report_mean_us=True)

  def _benchmark_matmul(self, mat, device):
    # [FUNCTION DOC]
    # HELPER: shared body of the matmul benchmarks. Inputs: a square matrix and a device name.
    # Returns immediately (does nothing) when the GPU is requested but none is present. Otherwise
    # places work on the device, copies the matrix to the GPU if needed, and benchmarks multiplying
    # the matrix by itself for 5000 iterations. Returns nothing.
    if device == GPU and not context.num_gpus():
      return
    with context.device(device):
      if device == GPU:
        mat = mat.gpu()
      func = lambda: math_ops.matmul(mat, mat)
      self._run(func, num_iters=5000)

  def _benchmark_bitwise_and(self, mat, device):
    # [FUNCTION DOC]
    # HELPER: shared body of the bitwise-and benchmarks. Inputs: an integer matrix and a device
    # name. Skips silently if a GPU is requested but unavailable. Otherwise runs on the device
    # (copying the matrix to the GPU when needed) and benchmarks the element-wise bitwise and of
    # the matrix with itself for 5000 iterations. Returns nothing.
    if device == GPU and not context.num_gpus():
      return
    with context.device(device):
      if device == GPU:
        mat = mat.gpu()
      func = lambda: bitwise_ops.bitwise_and(mat, mat)
      self._run(func, num_iters=5000)

  def _benchmark_random_normal(self, device):
    # [FUNCTION DOC]
    # HELPER: shared body of the random-normal benchmark. Input: a device name. Skips silently if
    # a GPU is requested but unavailable. Otherwise, on that device, benchmarks a function that
    # builds a small integer constant, adds it to itself to form a shape and draws a random normal
    # tensor of that shape (the shape lives on the host, exercising redundant transfers), for 5000
    # iterations. Returns nothing.
    if device == GPU and not context.num_gpus():
      return
    with context.device(device):

      def func():
        # [FUNCTION DOC]
        # HELPER: the timed workload of the random-normal benchmark. Creates a one-element integer
        # constant, doubles it by adding it to itself, and uses the result as the shape for a
        # random normal tensor. Takes no arguments, returns nothing, result discarded.
        mat = constant_op.constant([3], dtypes.int32)
        s = mat + mat
        random_ops.random_normal(shape=s)

      self._run(func, num_iters=5000)

  # This only needs to be tested on GPU where redundant data transfers can occur
  def benchmark_random_normal_GPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: random-normal with a computed shape on the GPU (no-op without a GPU).
    self._benchmark_random_normal(GPU)

  def benchmark_tf_matmul_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: matrix multiply of the 2x2 float matrix on the CPU.
    self._benchmark_matmul(self._m_2_by_2, CPU)

  def benchmark_tf_bitwise_and_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: bitwise and of the 2x2 integer matrix on the CPU.
    self._benchmark_bitwise_and(self._m_2_by_2_int32, CPU)

  def benchmark_tf_matmul_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: matrix multiply of the 2x2 float matrix on the GPU (no-op without GPU).
    self._benchmark_matmul(self._m_2_by_2, GPU)

  def benchmark_tf_bitwise_and_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: bitwise and of the 2x2 integer matrix on the GPU (no-op without GPU).
    self._benchmark_bitwise_and(self._m_2_by_2_int32, GPU)

  def benchmark_tf_matmul_100_by_100_CPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: matrix multiply of the 100x100 float matrix on the CPU.
    self._benchmark_matmul(self._m_100_by_100, CPU)

  def benchmark_tf_bitwise_and_100_by_100_CPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: bitwise and of the 100x100 integer matrix on the CPU.
    self._benchmark_bitwise_and(self._m_100_by_100_int32, CPU)

  def benchmark_tf_matmul_100_by_100_GPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: matrix multiply of the 100x100 float matrix on the GPU (no-op w/o GPU).
    self._benchmark_matmul(self._m_100_by_100, GPU)

  def benchmark_tf_bitwise_and_100_by_100_GPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: bitwise and of the 100x100 integer matrix on the GPU (no-op w/o GPU).
    self._benchmark_bitwise_and(self._m_100_by_100_int32, GPU)

  def benchmark_tf_matmul_1000_by_1000_CPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: matrix multiply of the 1000x1000 float matrix on the CPU.
    self._benchmark_matmul(self._m_1000_by_1000, CPU)

  def benchmark_tf_bitwise_and_1000_by_1000_CPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: bitwise and of the 1000x1000 integer matrix on the CPU.
    self._benchmark_bitwise_and(self._m_1000_by_1000_int32, CPU)

  def benchmark_tf_matmul_1000_by_1000_GPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: matrix multiply of 1000x1000 float matrix on the GPU (no-op w/o GPU).
    self._benchmark_matmul(self._m_1000_by_1000, GPU)

  def benchmark_tf_bitwise_and_1000_by_1000_GPU(self):
    # [FUNCTION DOC]
    # Benchmark entry point: bitwise and of 1000x1000 integer matrix on the GPU (no-op w/o GPU).
    self._benchmark_bitwise_and(self._m_1000_by_1000_int32, GPU)


@test_util.with_eager_op_as_function
class RunEagerOpAsFunctionTest(test.TestCase):
  # [CLASS DOC]
  # Functional smoke tests run with the eager-op-as-function mode enabled by the class decorator.
  # Each test just executes some ops (no explicit assertions) and passes if no error is raised.
  # Shared fixture: a random 2x2 float matrix created in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the base setUp and stores a random uniform 2x2 float matrix as an
    # instance attribute used by the tests below.
    super().setUp()
    self._m_2_by_2 = random_ops.random_uniform((2, 2))

  def testDefaultAttrValues(self):
    # [FUNCTION DOC]
    # Scenario: map an identity function over a ragged tensor containing one row with one value,
    # with a ragged-tensor output type of rank one. Checks that ops relying on default attribute
    # values execute under the wrapped-call mode without error; nothing is asserted.
    ragged_map_ops.map_fn(
        fn=lambda x: x,
        elems=ragged_factory_ops.constant([[7]]),
        dtype=ragged_tensor.RaggedTensorType(dtype=dtypes.int32, ragged_rank=1))

  def testArrayFill(self):
    # [FUNCTION DOC]
    # Scenario: fill a shape of one 64-bit integer element with the value one. Passes if the fill
    # op (whose shape input lives on the host) runs without error; nothing is asserted.
    array_ops.fill(
        constant_op.constant([2], dtype=dtypes.int64), constant_op.constant(1))

  def testDatasetMap(self):
    # [FUNCTION DOC]
    # Scenario: build a range dataset of two elements and map the square function over it. With a
    # GPU present this checks that the wrapped call ops are placed on the CPU (device chosen from
    # the unwrapped op). Passes if construction succeeds; nothing is asserted.
    # When a GPU is available, this would test that the wrapped call ops are
    # placed on the CPU (i.e. the device is selected using the unwrapped op).
    dataset_ops.Dataset.range(2).map(math_ops.square)

  def testPrefetchToDevice(self):
    # [FUNCTION DOC]
    # Scenario: apply the prefetch-to-device transformation targeting the first GPU on a range
    # dataset of ten elements. Skipped when no GPU is available. Passes if no error is raised.
    if not context.num_gpus():
      self.skipTest("No GPU available")
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(prefetching_ops.prefetch_to_device("/gpu:0"))

  def testMatmul(self):
    # [FUNCTION DOC]
    # Scenario: multiply the fixture matrix by itself; passes if no error is raised.
    math_ops.matmul(self._m_2_by_2, self._m_2_by_2)

  def testMixedTypeListInputFastPath(self):
    # [FUNCTION DOC]
    # Scenario: identity-n (an op with a list of differently typed inputs) on two tensors, taking
    # the fast execution path. Passes if no error is raised.
    array_ops.identity_n([self._m_2_by_2, self._m_2_by_2])

  def testMixedTypeListInputEagerFallback(self):
    # [FUNCTION DOC]
    # Scenario: identity-n on a list of plain Python numbers, which forces the slower fallback
    # path that converts inputs first. Passes if no error is raised.
    array_ops.identity_n([1, 1])

  def testMixedTypeListInputFastPathDifferentArity(self):
    # [FUNCTION DOC]
    # Scenario: call identity-n with two tensors and then with three tensors on the fast path.
    # Verifies the cached function definition key includes the number of arguments, so the second
    # call does not reuse the two-argument function. Passes if no error is raised.
    # This tests that the FunctionDef cache key contains the number of args.
    array_ops.identity_n([self._m_2_by_2, self._m_2_by_2])
    array_ops.identity_n([self._m_2_by_2, self._m_2_by_2, self._m_2_by_2])

  def testMixedTypeListInputEagerFallbackDifferentArity(self):
    # [FUNCTION DOC]
    # Scenario: identity-n on two and then three Python numbers (fallback path). Passes if no
    # error is raised for either arity.
    array_ops.identity_n([1, 1])
    array_ops.identity_n([1, 1, 1])

  def testSingleTypeListFastPath(self):
    # [FUNCTION DOC]
    # Scenario: concatenate two tensors along the last axis (single-typed list input, fast path).
    # Passes if no error is raised.
    array_ops.concat([self._m_2_by_2, self._m_2_by_2], axis=-1)

  def testSingleTypeListEagerFallback(self):
    # [FUNCTION DOC]
    # Scenario: concatenate two nested Python lists along the last axis, forcing the conversion
    # fallback path. Passes if no error is raised.
    array_ops.concat([[1], [2]], axis=-1)

  def testSingleTypeListFastPathDifferentArity(self):
    # [FUNCTION DOC]
    # Scenario: concatenate two tensors and then three tensors on the fast path, checking that the
    # differing list length is handled (distinct cached functions). Passes if no error is raised.
    array_ops.concat([self._m_2_by_2, self._m_2_by_2], axis=-1)
    array_ops.concat([self._m_2_by_2, self._m_2_by_2, self._m_2_by_2], axis=-1)

  def testSingleTypeListEagerFallbackDifferentArity(self):
    # [FUNCTION DOC]
    # Scenario: concatenate two and then three Python lists along the last axis on the fallback
    # path. Passes if no error is raised for either arity.
    array_ops.concat([[1], [2]], axis=-1)
    array_ops.concat([[1], [2], [3]], axis=-1)

  def testCreateCriticalSection(self):
    # [FUNCTION DOC]
    # Scenario: create a critical section with a shared name and execute a function returning a
    # constant inside it. Passes if no error is raised (resource creation under wrapped calls).
    cs = critical_section_ops.CriticalSection(shared_name="cs")
    cs.execute(lambda: 1.0)


class RunEagerOpAsFunctionInternalsTest(test.TestCase):
  # [CLASS DOC]
  # Tests that look at executor internals through metric counter deltas to confirm that a simple
  # wrapped op runs on the single-threaded executor and synchronously. Each test enables the
  # eager-op-as-function mode with a decorator.

  @test_util.enable_eager_op_as_function
  def testSimpleGraphExecutesSynchronously(self):
    # [FUNCTION DOC]
    # CPU-only scenario (skipped when GPUs exist, since the graph must be unpartitioned). Creates
    # counter deltas for the default and single-threaded function executors, async and sync
    # partitioned-runner calls, and the "safe for sync" subgraph summary; runs a fill op; then
    # asserts the default executor was not used, the single-threaded executor was, no async run
    # occurred, at least one sync run occurred, and the subgraph was classified safe for sync.
    if context.num_gpus():
      self.skipTest("CPU-only test (requires unpartitioned graph).")

    default_executor = test_util.TestDelta("flr_executor", "default")
    single_threaded = test_util.TestDelta("flr_executor", "single_threaded")
    run_async = test_util.TestDelta("pflr_runsync", "async")
    run_sync = test_util.TestDelta("pflr_runsync", "sync")
    safe = test_util.TestDelta("subgraph_async_summary", "safe_for_sync")

    array_ops.fill([2], constant_op.constant(7, dtype=dtypes.int64))

    assert default_executor.Get() == 0
    assert single_threaded.Get() > 0
    assert run_async.Get() == 0
    assert run_sync.Get() > 0
    assert safe.Get() > 0

  @test_util.enable_eager_op_as_function
  def testSendRecvPartitionedGraphExecutesSynchronously(self):
    # [FUNCTION DOC]
    # GPU-only scenario (skipped without a GPU, since a partitioned graph is needed). Same setup as
    # the CPU variant but tracks the "send only" and "recv only" subgraph summaries. After running
    # a fill op it asserts: default executor unused, single-threaded executor used, no async run,
    # at least one sync run, and at least one send-only and one recv-only subgraph.
    if not context.num_gpus():
      self.skipTest("GPU-only test (requires partitioned graph).")

    default_executor = test_util.TestDelta("flr_executor", "default")
    single_threaded = test_util.TestDelta("flr_executor", "single_threaded")
    run_async = test_util.TestDelta("pflr_runsync", "async")
    run_sync = test_util.TestDelta("pflr_runsync", "sync")
    send_only = test_util.TestDelta("subgraph_async_summary", "send_only")
    recv_only = test_util.TestDelta("subgraph_async_summary", "recv_only")

    array_ops.fill([2], constant_op.constant(7, dtype=dtypes.int64))

    assert default_executor.Get() == 0
    assert single_threaded.Get() > 0
    assert run_async.Get() == 0
    assert run_sync.Get() > 0
    assert send_only.Get() > 0
    assert recv_only.Get() > 0

if __name__ == "__main__":
  test.main()
