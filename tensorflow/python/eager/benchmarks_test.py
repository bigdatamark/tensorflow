# [MODULE DOC]
# Micro-benchmarks for low-level eager execution primitives in TensorFlow. Defines one benchmark
# class, MicroBenchmarks, built on the shared benchmark base class, plus two module level helpers: a
# wrapper that calls the C fast-path executor for matrix multiplication directly, and a timing loop
# used by every benchmark. The benchmarks cover creating tensors and constants, tensor addition,
# indexing, multiply, convolution, matmul (eager, via generated ops, via the raw C entry points, and
# via traced functions with and without signatures, in sync and async execution modes), gradient
# tape and forward-mode accumulator overhead, reductions, tensordot, zeros, random numbers, dropout,
# transpose, variable reads, scan, type conversion, nest utilities, boolean mask, range loops,
# resource-variable reads in conditionals and sparse/ragged embedding lookups. Many benchmarks come
# in CPU and GPU flavors; GPU ones are expected to run only on machines with GPUs. Run directly as a
# test binary with a benchmark filter flag to select benchmarks; each benchmark reports wall time
# per iteration through the benchmark framework. Import-time side effects: defines module constants
# for the CPU and GPU device names and a global scratch value used by constant-creation benchmarks.
# Importing the module does not run benchmarks.
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
r"""Benchmarks for low-level eager execution primitives.

To run CPU benchmarks:
  bazel run -c opt benchmarks_test -- --benchmark_filter=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" benchmarks_test -- \
    --benchmark_filter=.

To run a subset of benchmarks using --benchmarks flag.
--benchmarks: the list of benchmarks to run. The specified value is interpreted
as a regular expression and any benchmark whose name contains a partial match
to the regular expression is executed.
e.g. --benchmark_filter=".*matmul*." will run all matmul related benchmarks.

"""
import time

import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.compat import compat as forward_compat
from tensorflow.python.eager import backprop  # pylint: disable=unused-import
from tensorflow.python.eager import benchmarks_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import def_function
from tensorflow.python.eager import forwardprop
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect

CPU = "/device:CPU:0"
GPU = "/device:GPU:0"
GLOBAL_TEST_VALUE = None


def c_tfe_py_fastpath_execute(a,
                              b,
                              transpose_a=False,
                              transpose_b=False,
                              name=None):
  # [FUNCTION DOC]
  # HELPER: benchmark glue that calls the C fast-path executor for the matrix multiplication op.
  # Runs a MatMul operation by calling the native fast-path execute entry point directly, bypassing
  # the Python op wrappers. Inputs: two tensors a and b, flags transpose_a and transpose_b (default
  # false), and an optional op name. Returns the resulting tensor. Asserts that eager execution is
  # enabled (the prototype has no graph-construction path), so it raises an assertion error
  # otherwise. If the native call fails with a non-OK status, the error message is extended with the
  # op name when one was given and the status is converted into the matching Python exception,
  # raised without chaining the original.
  ctx = context.context()
  assert ctx.executing_eagerly(
  ), "The prototype doesn't contain C code for graph construction"
  try:
    return pywrap_tfe.TFE_Py_FastPathExecute(ctx, "MatMul", name, a, b,
                                             "transpose_a", transpose_a,
                                             "transpose_b", transpose_b)
  except core._NotOkStatusException as e:
    if name is not None:
      e.message += " name: " + name
    raise core._status_to_exception(e) from None


def run_benchmark(func, num_iters, execution_mode=None):
  # [FUNCTION DOC]
  # HELPER: timing loop passed to the benchmark base class report method.
  # Times a zero-argument callable. Inputs: func (callable), num_iters (number of timed calls), and
  # an optional execution_mode (sync, async, or none for the current default). Inside that execution
  # mode it calls func once to warm up (draining the async executor afterwards when the mode is
  # async), then calls it num_iters times, drains the async executor again if needed, and returns
  # the elapsed wall-clock seconds for the timed loop only (warmup excluded).
  ctx = context.context()
  with context.execution_mode(execution_mode):
    # call func to warm up
    func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    start = time.time()
    for _ in range(num_iters):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    end = time.time()

    return end - start


class MicroBenchmarks(benchmarks_test_base.MicroBenchmarksBase):
  # [CLASS DOC]
  # Collection of micro-benchmarks for eager execution overheads, derived from the shared benchmark
  # base. Instances precompute random input tensors (a vector of two values, 2 by 2 and 100 by 784
  # matrices, an 8x28x28x3 image batch, a 1x3x3x1 filter, and a 10000 by 16 embedding table) and
  # iteration counts for 2 by 2 and 100 by 784 matmuls. Methods named benchmark_* (or benchmark
  # followed by a capitalized word) are discovered and run by the benchmark framework; methods
  # starting with an underscore are parameterized helpers shared by several benchmarks. Results are
  # reported through the base class.
  def __init__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent class constructor; harmless only if the base needs no
    #   setup.
    # Constructor with no arguments. Creates random uniform tensors used as shared inputs: a vector
    # of two elements (multiply benchmarks), a 2 by 2 and a 100 by 784 matrix (matmul benchmarks),
    # iteration counts of 30000 for both matmul sizes, an 8x28x28x3 input and a 1x3x3x1 filter
    # (convolution benchmarks), and a 10000 by 16 table (embedding benchmarks). Tensors are created
    # on the default device.
    # used for multiply benchmarks
    self._m_2 = random_ops.random_uniform([2])

    # used for matmul benchmarks
    self._m_2_by_2 = random_ops.random_uniform((2, 2))
    self._m_100_by_784 = random_ops.random_uniform((100, 784))

    self._num_iters_2_by_2 = 30000
    self._num_iters_100_by_784 = 30000

    # used for conv2d benchmarks
    self._m_8_28_28_3 = random_ops.random_uniform((8, 28, 28, 3))
    self._m_1_3_3_1 = random_ops.random_uniform((1, 3, 3, 1))

    # used for embedding benchmarks
    self._m_10000_by_16 = random_ops.random_uniform((10000, 16))

  def _get_benchmark_name(self):
    # [FUNCTION DOC]
    # HELPER: supplies the benchmark name to the base class report method.
    # KNOWN ISSUE: the base class run_report calls this; it relies on walking the call stack and on
    #   the
    # frame method named decorated being a decorator wrapper, which is a fragile hack. Determines
    # the name of the benchmark method being run by scanning the call stack from the outermost frame
    # inward for a frame whose self object is a benchmark instance, taking that frame's function
    # name, and skipping frames named decorated (wrappers added by decorators) so the real method
    # name is used. Appends a suffix when the runtime-for-TFRT mode is enabled. Takes no arguments,
    # returns the name text, and raises a value error if no benchmark frame is found.
    """Mostly copied from benchmark.py _get_name()."""
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
    return name

  def _run(self, func, num_iters, execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared entry for all benchmarks to time a callable and report it.
    # Runs the timing loop on func for num_iters iterations in the optional execution_mode and
    # reports the result via the base class report method. Returns nothing.
    self.run_report(run_benchmark, func, num_iters, execution_mode)

  def benchmark_create_np_array(self):
    # [FUNCTION DOC]
    # Benchmark: cost of creating a one-element numpy array of floats, 30000 iterations. Baseline
    # for tensor creation.
    func = lambda: np.array([3.0])
    self._run(func, 30000)

  def _benchmark_create_tensor(self, value, dtype, device):
    # [FUNCTION DOC]
    # HELPER: shared body for the tensor-creation benchmarks.
    # Benchmarks building an eager tensor object from a value (Python list or numpy array), with a
    # given dtype enum and device name. When the device is the GPU, performs a warmup creation
    # first. Times 30000 creations. Returns nothing.
    """Benchmark overheads of creating a Tensor object."""
    if device == GPU:
      # Warmup the GPU
      ops.EagerTensor(value, device=device)

    def func():
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      ops.EagerTensor(value, device=device, dtype=dtype)

    self._run(func, 30000)

  def _benchmark_create_constant(self, value, dtype, cached=True):
    # [FUNCTION DOC]
    # HELPER: shared body for constant-creation benchmarks.
    # Sets the module-level scratch value to value, then times creating a constant of the given
    # dtype. When cached is true (default) the same value is used each time; otherwise the scratch
    # value is incremented before each creation so no cache can be hit. Runs on the GPU if any
    # exist, otherwise the CPU, with 1000 warmup calls and 3000 timed iterations. Mutates the
    # module-level scratch value.
    global GLOBAL_TEST_VALUE
    GLOBAL_TEST_VALUE = value

    def cached_func():
      # [FUNCTION DOC]
      # Creates or converts the same value on every call, allowing caches to be used.
      constant_op.constant(value, dtype=dtype)

    def uncached_func():
      # [FUNCTION DOC]
      # Increments the module-level scratch value and uses the new value, defeating caches. Mutates
      # module state.
      global GLOBAL_TEST_VALUE
      GLOBAL_TEST_VALUE += 1
      constant_op.constant(GLOBAL_TEST_VALUE, dtype=dtype)

    func = cached_func if cached else uncached_func

    with ops.device("GPU:0" if context.num_gpus() else "CPU:0"):
      for _ in range(1000):
        func()  # Warmup.
      self._run(func, 3000)

  def benchmark_create_float_constant(self):
    # [FUNCTION DOC]
    # Benchmark: creating a float constant from a repeated Python value (cache-friendly), default
    # dtype.
    self._benchmark_create_constant(42.0, dtype=None)

  def benchmark_create_float_constant_uncached(self):
    # [FUNCTION DOC]
    # Benchmark: creating a float constant whose value changes every call, so no constant cache can
    # be used.
    self._benchmark_create_constant(42.0, dtype=None, cached=False)

  def benchmark_create_int32_constant(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 32-bit integer constant from a repeated value. Skipped (returns
    # immediately) when GPUs exist, since int32 constants always live on the host.
    if context.num_gpus():
      return  # int32 constants are always allocated on CPU.

    self._benchmark_create_constant(42, dtype=dtypes.int32)

  def benchmark_create_int32_constant_uncached(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 32-bit integer constant whose value changes each call. Skipped when GPUs
    # exist.
    if context.num_gpus():
      return  # int32 constants are always allocated on CPU.

    self._benchmark_create_constant(42, dtype=dtypes.int32, cached=False)

  def _benchmark_add(self, a, b):
    # [FUNCTION DOC]
    # HELPER: shared body for addition benchmarks using the add op function.
    # Times calling the add op on operands a and b and wrapping the result in a memory view (forcing
    # the result to be materialized as a buffer). Runs on the GPU if available, else CPU, 1000
    # warmup calls then 30000 timed iterations. Returns nothing.

    def func():
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      return memoryview(math_ops.add_v2(a, b))

    with ops.device("GPU:0" if context.num_gpus() else "CPU:0"):
      for _ in range(1000):
        func()  # Warmup.
      self._run(func, 30000)

  def _benchmark_add_operator_overload(self, a, b):
    # [FUNCTION DOC]
    # HELPER: shared body for addition benchmarks using the operator overload.
    # Same as the plain add helper but uses the language's addition operator on the operands, to
    # measure the overloaded-operator path. Wraps the result in a memory view; 1000 warmup calls,
    # 30000 timed calls.

    def func():
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      return memoryview(a + b)

    with ops.device("GPU:0" if context.num_gpus() else "CPU:0"):
      for _ in range(1000):
        func()  # Warmup.
      self._run(func, 30000)

  def benchmark_add_float_scalars(self):
    # [FUNCTION DOC]
    # Benchmark: adding two plain Python float scalars through the add op.
    self._benchmark_add(42.0, 24.0)

  def benchmark_add_int32_scalars(self):
    # [FUNCTION DOC]
    # Benchmark: adding two plain Python integer scalars through the add op.
    self._benchmark_add(42, 24)

  def benchmark_add_float_scalar_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: adding two float scalar tensors through the add op.
    tensor_a = constant_op.constant(42.0)
    tensor_b = constant_op.constant(24.0)
    self._benchmark_add(tensor_a, tensor_b)

  def benchmark_add_float_scalar_tensor_overloaded_operator(self):
    # [FUNCTION DOC]
    # Benchmark: adding two float scalar tensors through the overloaded addition operator.
    tensor_a = constant_op.constant(42.0)
    tensor_b = constant_op.constant(24.0)
    self._benchmark_add_operator_overload(tensor_a, tensor_b)

  def benchmark_add_int32_scalar_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: adding two integer scalar tensors through the add op.
    tensor_a = constant_op.constant(42)
    tensor_b = constant_op.constant(24)
    self._benchmark_add(tensor_a, tensor_b)

  def benchmark_add_float_dense_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: adding two 2 by 2 float tensors through the add op.
    tensor_a = constant_op.constant([[42.0, 42.0], [42.0, 42.0]])
    tensor_b = constant_op.constant([[24.0, 24.0], [24.0, 24.0]])
    self._benchmark_add(tensor_a, tensor_b)

  def benchmark_add_int32_dense_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: adding two 2 by 2 integer tensors through the add op.
    tensor_a = constant_op.constant([[42, 42], [42, 42]])
    tensor_b = constant_op.constant([[24, 24], [24, 24]])
    self._benchmark_add(tensor_a, tensor_b)

  def benchmark_create_float_tensor_from_list_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a float32 eager tensor from a nested Python list on the CPU.
    self._benchmark_create_tensor([[3.0]], dtypes.float32.as_datatype_enum, CPU)

  def benchmark_create_float_tensor_from_np_array_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a float32 eager tensor from a numpy array on the CPU.
    self._benchmark_create_tensor(
        np.array([[3.0]], dtype=np.float32), dtypes.float32.as_datatype_enum,
        CPU)

  def benchmark_create_int32_tensor_from_list_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating an int32 eager tensor from a nested Python list on the CPU.
    self._benchmark_create_tensor([[3]], dtypes.int32.as_datatype_enum, CPU)

  def benchmark_create_int32_tensor_from_np_array_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating an int32 eager tensor from a numpy array on the CPU.
    self._benchmark_create_tensor(
        np.array([[3]], dtype=np.int32), dtypes.int32.as_datatype_enum, CPU)

  def benchmark_create_float_tensor_from_list_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a float32 eager tensor from a nested list on the GPU. Returns immediately
    # if no GPU exists.
    if not context.num_gpus():
      return
    self._benchmark_create_tensor([[3.0]], dtypes.float32.as_datatype_enum, GPU)

  def benchmark_create_float_tensor_from_np_array_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a float32 eager tensor from a numpy array on the GPU. Skipped if no GPU
    # exists.
    if not context.num_gpus():
      return
    self._benchmark_create_tensor(
        np.array([[3.0]], dtype=np.float32), dtypes.float32.as_datatype_enum,
        GPU)

  def benchmark_create_int32_tensor_from_list_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating an int32 eager tensor from a list targeting the GPU (int32 data stays in
    # host memory). Skipped if no GPU exists.
    # int32's are kept on host memory even when executing on GPU.
    if not context.num_gpus():
      return
    self._benchmark_create_tensor([[3]], dtypes.int32.as_datatype_enum, GPU)

  def benchmark_create_int32_tensor_from_np_array_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating an int32 eager tensor from a numpy array targeting the GPU. Skipped if no
    # GPU exists.
    # int32's are kept on host memory even when executing on GPU.
    if not context.num_gpus():
      return
    self._benchmark_create_tensor(
        np.array([[3]], dtype=np.int32), dtypes.int32.as_datatype_enum, GPU)

  def benchmark_index_tensor_with_literal(self):
    # [FUNCTION DOC]
    # Benchmark: building a one-element constant and indexing it with a literal integer.
    func = lambda: constant_op.constant([3.0])[0]
    self._run(func, 30000)

  def benchmark_index_tensor_with_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: building a one-element constant and indexing it with a scalar tensor index.
    func = lambda idx=constant_op.constant(0): constant_op.constant([3.0])[idx]
    self._run(func, 30000)

  def benchmark_index_tensor_with_np_array(self):
    # [FUNCTION DOC]
    # Benchmark: building a one-element constant and indexing it with a zero-dimensional numpy
    # array.
    func = lambda idx=np.array(0): constant_op.constant([3.0])[idx]
    self._run(func, 30000)

  def _benchmark_np_multiply(self, m, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for the numpy multiply benchmark.
    # Copies tensor m to host as a numpy array and times elementwise self-multiplication with
    # num_iters iterations.
    a = m.cpu().numpy()
    func = lambda: a * a
    self._run(func, num_iters)

  def _benchmark_tf_multiply(self, m, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for tensor multiply benchmarks.
    # Times elementwise multiplication of tensor m with itself using the overloaded operator,
    # num_iters times.
    func = lambda: m * m
    self._run(func, num_iters)

  def _benchmark_tf_conv2d(self, m1, m2, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for convolution benchmarks.
    # Times a 2D convolution of input m1 with filter m2, unit strides and valid padding, num_iters
    # times.
    func = lambda: nn_ops.conv2d(m1, m2, strides=[1, 1, 1, 1], padding="VALID")
    self._run(func, num_iters)

  def _benchmark_tf_multiply_op(self, m, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for multiply benchmarks that call the multiply op function.
    # Times calling the multiply op on m and m for num_iters iterations.
    func = lambda: math_ops.multiply(m, m)
    self._run(func, num_iters)

  def benchmark_np_multiply(self):
    # [FUNCTION DOC]
    # Benchmark: numpy elementwise multiply of a two-element vector (30000 iterations), a baseline
    # for tensor multiply.
    self._benchmark_np_multiply(self._m_2, 30000)

  def benchmark_tf_multiply_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: tensor multiply with the overloaded operator on a two-element vector pinned to the
    # CPU.
    with context.device(CPU):
      m = self._m_2.cpu()
      self._benchmark_tf_multiply(m, 30000)

  def benchmark_tf_multiply_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: tensor multiply with the overloaded operator on a vector copied to the GPU. Skipped
    # if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2.gpu()
      self._benchmark_tf_multiply(m, 30000)

  def benchmark_tf_multiply_op_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: multiply op function on a two-element vector pinned to the CPU.
    with context.device(CPU):
      m = self._m_2.cpu()
      self._benchmark_tf_multiply_op(m, 30000)

  def benchmark_tf_multiply_op_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: multiply op function on a vector copied to the GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2.gpu()
      self._benchmark_tf_multiply_op(m, 30000)

  def benchmark_tf_conv2d_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: 2D convolution of an 8x28x28x3 batch with a 1x3x3x1 filter on the CPU.
    with context.device(CPU):
      m1 = self._m_8_28_28_3.cpu()
      m2 = self._m_1_3_3_1.cpu()
      self._benchmark_tf_conv2d(m1, m2, 30000)

  def benchmark_tf_conv2d_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: 2D convolution of an 8x28x28x3 batch with a 1x3x3x1 filter on the GPU. Skipped if
    # no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m1 = self._m_8_28_28_3.gpu()
      m2 = self._m_1_3_3_1.gpu()
      self._benchmark_tf_conv2d(m1, m2, 30000)

  def benchmark_tf_identity(self):
    # [FUNCTION DOC]
    # Benchmark: generated identity op on a tensor (fast path), 30000 iterations.
    m = self._m_2
    self._run(lambda: gen_array_ops.identity(m), 30000)

  def benchmark_slowpath_tf_identity(self):
    # [FUNCTION DOC]
    # Benchmark: generated identity op on a plain Python integer, which forces the slower conversion
    # path.
    self._run(lambda: gen_array_ops.identity(1), 30000)

  def benchmark_tfe_py_execute_identity(self):
    # [FUNCTION DOC]
    # Benchmark: calling the native execute entry point directly for the identity op with a
    # precomputed attribute list and the context handle.
    m = self._m_2
    ctx_handle = context.context()._handle
    attrs = ("T", self._m_2.dtype.as_datatype_enum)
    inputs = [m]

    def f():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark.
      pywrap_tfe.TFE_Py_Execute(ctx_handle, None, "Identity", inputs, attrs, 1)

    self._run(f, 30000)

  def benchmark_tf_gradient_function_identity(self):
    # [FUNCTION DOC]
    # Benchmark: creating and calling a gradients function of the identity op on a CPU tensor each
    # iteration.
    with context.device(CPU):
      m = gen_array_ops.identity(self._m_2)
      self._run(
          lambda: backprop.gradients_function(gen_array_ops.identity, [0])(m),
          30000)

  def benchmark_tf_gradient_forward_identity(self):
    # [FUNCTION DOC]
    # Benchmark: identity op while a gradient tape is recording and watching the input.
    with backprop.GradientTape() as tape:
      m = self._m_2
      tape.watch(m)
      self._run(lambda: gen_array_ops.identity(m), 30000)

  def benchmark_tf_gradient_tape_push_pop(self):
    # [FUNCTION DOC]
    # Benchmark: overhead of entering and leaving an empty gradient tape.

    def f():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark.
      with backprop.GradientTape():
        pass

    self._run(f, 30000)

  def benchmark_tf_gradient_function_no_op(self):
    # [FUNCTION DOC]
    # Benchmark: creating and calling a gradients function of a pass-through function on a CPU
    # tensor.
    with context.device(CPU):
      m = gen_array_ops.identity(self._m_2)
      self._run(lambda: backprop.gradients_function(lambda x: x, [0])(m), 30000)

  def _benchmark_np_matmul(self, m, transpose_b, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for numpy matmul benchmarks.
    # Moves m to host numpy, optionally transposes the second operand according to transpose_b, and
    # times the dot product num_iters times.
    a = m.cpu().numpy()
    b = a.T if transpose_b else a
    func = lambda: np.dot(a, b)
    self._run(func, num_iters)

  def _benchmark_tf_matmul(self,
                           m,
                           transpose_b,
                           num_iters,
                           execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body for the tensor matmul benchmarks.
    # Times the matmul op on m with itself, optionally transposing the second operand, for num_iters
    # iterations in the optional execution_mode.
    func = lambda: math_ops.matmul(m, m, transpose_b=transpose_b)
    self._run(func, num_iters, execution_mode=execution_mode)

  def _benchmark_gen_math_ops_matmul(self, m, transpose_b, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for benchmarks calling the generated matrix multiplication op.
    # Times the generated mat_mul op on m with itself (transpose_b optional) for num_iters
    # iterations.

    def func():
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      gen_math_ops.mat_mul(m, m, transpose_b=transpose_b)

    self._run(func, num_iters)

  def _benchmark_tfe_py_fastpath_execute_matmul(self, m, transpose_b,
                                                num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for benchmarks of the native fast-path executor.
    # Times the module-level fast-path wrapper for matmul of m with itself, num_iters times.

    def func():
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      c_tfe_py_fastpath_execute(m, m, transpose_b=transpose_b)

    self._run(func, num_iters)

  def _benchmark_tfe_py_execute_matmul(self, m, transpose_b, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for benchmarks of the native execute entry point.
    # Builds the attribute list for matmul (transpose flags and dtype) and gets the context handle
    # and current device name, then times calling the native execute function for MatMul with m as
    # both inputs.
    inputs = [m, m]
    # pylint: disable=protected-access
    ctx_handle = context.context()._handle
    # pylint: enable=protected-access
    device = context.context().device_name
    attrs = ("transpose_a", False, "transpose_b", transpose_b, "T",
             m.dtype.as_datatype_enum)

    def func():
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      pywrap_tfe.TFE_Py_Execute(ctx_handle, device, "MatMul", inputs, attrs, 1)

    self._run(func, num_iters)

  def _benchmark_defun_matmul(self,
                              m,
                              transpose_b,
                              num_iters,
                              execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body for benchmarks of a traced matmul function.
    # Wraps the matmul op into a traced function, then times calling it with m, m and the
    # transpose_b keyword for num_iters iterations in the optional execution mode. The warmup in the
    # timing loop triggers tracing.
    f = def_function.function(math_ops.matmul)
    func = lambda: f(m, m, transpose_b=transpose_b)
    self._run(func, num_iters, execution_mode=execution_mode)

  def _benchmark_defun_matmul_with_signature(self,
                                             m,
                                             num_iters,
                                             execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body for traced matmul with an explicit input signature.
    # Defines a traced function with a fixed 2 by 2 float32 input signature that multiplies its
    # argument by itself, then times calling it with m for num_iters iterations in the optional
    # execution mode.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([2, 2], dtypes.float32)])
    def defun_matmul(m):
      # [FUNCTION DOC]
      # Traced function that multiplies its input matrix by itself.
      return math_ops.matmul(m, m)

    func = lambda: defun_matmul(m)
    self._run(func, num_iters, execution_mode=execution_mode)

  def _benchmark_defun_matmul_relaxed_shape(self,
                                            m,
                                            num_iters,
                                            execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body for traced matmul with retracing-reduction enabled.
    # Defines a traced function that reduces retracing, calls it once with a 3 by 3 tensor so it is
    # generalized, then times calling it with m for num_iters iterations in the optional execution
    # mode.

    @def_function.function(reduce_retracing=True)
    def defun_matmul(m):
      # [FUNCTION DOC]
      # Traced function that multiplies its input matrix by itself.
      return math_ops.matmul(m, m)

    m_3_by_3 = random_ops.random_uniform((3, 3))
    defun_matmul(m_3_by_3)
    func = lambda: defun_matmul(m)
    self._run(func, num_iters, execution_mode=execution_mode)

  def _benchmark_defun_args_matmul(self, m, num_iters, execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body for traced matmul with tensor arguments.
    # Defines a traced function of one argument that multiplies it by itself and times calling it
    # with m.

    @def_function.function
    def defun_matmul(m):
      # [FUNCTION DOC]
      # Traced function that multiplies its input matrix by itself.
      return math_ops.matmul(m, m)

    func = lambda: defun_matmul(m)
    self._run(func, num_iters, execution_mode=execution_mode)

  def _benchmark_nested_defun_matmul(self, m, transpose_b, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for a traced function that calls another traced function.
    # Builds an inner traced matmul and an outer traced function that multiplies the inner result
    # with a third operand; warms up 1000 calls, then times calling outer with m for the three
    # operands.
    inner = def_function.function(math_ops.matmul)

    @def_function.function
    def outer(a, b, c, transpose_b):
      # [FUNCTION DOC]
      # Traced outer function; calls the inner traced matmul and multiplies its result by a third
      # operand.
      return math_ops.matmul(inner(a, b, transpose_b=transpose_b), c)

    func = lambda: outer(m, m, m, transpose_b=transpose_b)
    # Warmup before benchmark
    for _ in range(1000):
      func()
    self._run(func, num_iters)

  def _benchmark_defun_matmul_forward_backward(self,
                                               m,
                                               transpose_b,
                                               num_iters,
                                               execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body for traced matmul plus gradient computation.
    # Wraps matmul in a traced function; each timed call records it under a gradient tape that
    # watches m and computes the gradient of the output with respect to m, discarding it. Optional
    # execution_mode.
    f = def_function.function(math_ops.matmul)

    def func():
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      with backprop.GradientTape() as gt:
        gt.watch(m)
        y = f(m, m, transpose_b=transpose_b)
      _ = gt.gradient(y, m)

    self._run(func, num_iters, execution_mode=execution_mode)

  def _benchmark_read_variable(self, m, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for variable read benchmarks.
    # Times calling the variable's value read method, num_iters times.
    self._run(m.value, num_iters)

  def _benchmark_matmul_read_variable(self, m, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for matmul of a variable.
    # Runs the generated matmul helper on variable m (without transpose), which includes reading the
    # variable.
    self._benchmark_gen_math_ops_matmul(
        m, transpose_b=False, num_iters=num_iters)

  def _benchmark_matmul_read_variable_with_tape(self, m, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for matmul of a variable under a tape.
    # Same as the matmul-of-variable helper but inside a gradient tape that watches the variable.
    with backprop.GradientTape() as tape:
      tape.watch(m)
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=False, num_iters=num_iters)

  def _benchmark_read_variable_with_tape(self, m, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for variable reads under a tape.
    # Times reading the variable's value inside a gradient tape that watches it.
    with backprop.GradientTape() as tape:
      tape.watch(m)
      self._run(m.value, num_iters)

  # Benchmarks for A^2, A of dimension 2 by 2.
  def benchmark_np_matmul_2_by_2(self):
    # [FUNCTION DOC]
    # Benchmark: numpy matrix multiply of the 2 by 2 matrix with itself (no transpose).
    self._benchmark_np_matmul(
        self._m_2_by_2, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_matmul_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: tensor matmul of a 2 by 2 matrix on the CPU, synchronous execution.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_tf_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_matmul_2_by_2_CPU_async(self):
    # [FUNCTION DOC]
    # Benchmark: tensor matmul of a 2 by 2 matrix on the CPU in asynchronous execution mode.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_tf_matmul(
          m,
          transpose_b=False,
          num_iters=self._num_iters_2_by_2,
          execution_mode=context.ASYNC)

  def benchmark_gen_math_ops_matmul_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: generated matmul op on a 2 by 2 matrix on the CPU.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tfe_py_fastpath_execute_matmul_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: native fast-path executor matmul on a 2 by 2 matrix on the CPU.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_tfe_py_fastpath_execute_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tfe_py_execute_matmul_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: native execute entry point for matmul on a 2 by 2 matrix on the CPU.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_tfe_py_execute_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul function on a 2 by 2 matrix on the CPU.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_2_by_2_with_signature_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul with a fixed input signature on a 2 by 2 matrix on the CPU.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_matmul_with_signature(
          m, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_2_by_2_relaxed_shape_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul with relaxed shapes on a 2 by 2 matrix on the CPU.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_matmul_relaxed_shape(
          m, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_args_matmul_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul taking the tensor as an argument, 2 by 2 on the CPU.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_args_matmul(m, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_2_by_2_CPU_async(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul of a 2 by 2 matrix on the CPU in asynchronous mode.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_matmul(
          m,
          transpose_b=False,
          num_iters=self._num_iters_2_by_2,
          execution_mode=context.ASYNC)

  def _benchmark_matmul_forward_backward_2_by_2_CPU(self, run_eager=False):
    # [FUNCTION DOC]
    # HELPER: shared body for forward-and-backward matmul benchmarks.
    # Sets the global run-functions-eagerly flag to run_eager (default false), times the
    # forward-backward helper on a 2 by 2 CPU matrix, then resets the flag to false. Side effect:
    # toggles global function execution mode.
    def_function.run_functions_eagerly(run_eager)
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_matmul_forward_backward(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)
    def_function.run_functions_eagerly(False)

  def _benchmark_matmul_forward_backward_2_by_2_CPU_async(
      self, run_eager=False):
    # [FUNCTION DOC]
    # HELPER: shared body for the asynchronous forward-and-backward matmul benchmarks.
    # KNOWN ISSUE: unlike the synchronous variant, it never resets the run-functions-eagerly flag to
    #   false after
    # the benchmark, so passing run_eager true leaves eager mode switched on for later benchmarks.
    # Sets the global run-functions-eagerly flag to run_eager (default false), then times the
    # forward-backward helper on a 2 by 2 CPU matrix in asynchronous execution mode.
    def_function.run_functions_eagerly(run_eager)
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_defun_matmul_forward_backward(
          m,
          transpose_b=False,
          num_iters=self._num_iters_2_by_2,
          execution_mode=context.ASYNC)

  def benchmark_defun_matmul_forward_backward_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul with gradient, 2 by 2 on the CPU, synchronous, functions run as
    # graphs.
    self._benchmark_matmul_forward_backward_2_by_2_CPU(False)

  def benchmark_defun_matmul_forward_backward_2_by_2_CPU_async(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul with gradient, 2 by 2 on the CPU, asynchronous, functions run as
    # graphs.
    self._benchmark_matmul_forward_backward_2_by_2_CPU_async(False)

  def benchmark_defun_eager_matmul_forward_backward_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: matmul with gradient, 2 by 2 on the CPU, synchronous, with functions forced to run
    # eagerly.
    self._benchmark_matmul_forward_backward_2_by_2_CPU(True)

  def benchmark_defun_eager_matmul_forward_backward_2_by_2_CPU_async(self):
    # [FUNCTION DOC]
    # Benchmark: matmul with gradient, 2 by 2 on the CPU, asynchronous, with functions forced to run
    # eagerly (leaves the eager flag set, see helper).
    self._benchmark_matmul_forward_backward_2_by_2_CPU_async(True)

  def benchmark_tf_matmul_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: tensor matmul of a 2 by 2 matrix on the GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_tf_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_matmul_2_by_2_GPU_async(self):
    # [FUNCTION DOC]
    # Benchmark: tensor matmul of a 2 by 2 matrix on the GPU, asynchronous. Skipped if no GPU
    # exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_tf_matmul(
          m,
          transpose_b=False,
          num_iters=self._num_iters_2_by_2,
          execution_mode=context.ASYNC)

  def benchmark_gen_math_ops_matmul_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: generated matmul op on a 2 by 2 matrix on the GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_tfe_py_execute_matmul_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: native execute entry point for matmul, 2 by 2 on the GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_tfe_py_execute_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul function, 2 by 2 on the GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_defun_matmul(
          m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_2_by_2_with_signature_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul with fixed signature, 2 by 2 on the GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_defun_matmul_with_signature(
          m, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_2_by_2_relaxed_shape_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul with relaxed shapes, 2 by 2 on the GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_defun_matmul_relaxed_shape(
          m, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_args_matmul_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul taking an argument, 2 by 2 on the GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_defun_args_matmul(m, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_matmul_2_by_2_GPU_async(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul, 2 by 2 on the GPU, asynchronous. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_defun_matmul(
          m,
          transpose_b=False,
          num_iters=self._num_iters_2_by_2,
          execution_mode=context.ASYNC)

  def benchmark_nested_defun_matmul_2_by_2(self):
    # [FUNCTION DOC]
    # Benchmark: nested traced matmul functions on a 2 by 2 CPU matrix.
    m = self._m_2_by_2.cpu()
    self._benchmark_nested_defun_matmul(
        m, transpose_b=False, num_iters=self._num_iters_2_by_2)

  # Benchmarks for AA.T, A of dimension 100 by 784.
  def benchmark_np_matmul_100_by_784(self):
    # [FUNCTION DOC]
    # Benchmark: numpy product of a 100 by 784 matrix with its transpose.
    self._benchmark_np_matmul(
        self._m_100_by_784,
        transpose_b=True,
        num_iters=self._num_iters_100_by_784)

  def benchmark_tf_matmul_100_by_784_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: tensor matmul of a 100 by 784 matrix with its transpose on the CPU.
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_tf_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_tf_matmul_100_by_784_CPU_async(self):
    # [FUNCTION DOC]
    # Benchmark: same as the CPU tensor matmul on 100 by 784 but in asynchronous mode.
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_tf_matmul(
          m,
          transpose_b=True,
          num_iters=self._num_iters_100_by_784,
          execution_mode=context.ASYNC)

  def benchmark_gen_math_ops_matmul_100_by_784_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: generated matmul op, 100 by 784 times its transpose, CPU.
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_tfe_py_fastpath_execute_matmul_100_by_784_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: native fast-path executor matmul, 100 by 784 times its transpose, CPU.
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_tfe_py_fastpath_execute_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_tfe_py_execute_matmul_100_by_784_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: native execute entry point for matmul, 100 by 784 times its transpose, CPU.
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_tfe_py_execute_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_defun_matmul_100_by_784_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul function, 100 by 784 times its transpose, CPU.
    with context.device(CPU):
      m = self._m_100_by_784.cpu()
      self._benchmark_defun_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_tf_matmul_100_by_784_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: tensor matmul, 100 by 784 times its transpose, GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_100_by_784.gpu()
      self._benchmark_tf_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_tf_matmul_100_by_784_GPU_async(self):
    # [FUNCTION DOC]
    # Benchmark: tensor matmul, 100 by 784 times its transpose, GPU, asynchronous. Skipped if no GPU
    # exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_100_by_784.gpu()
      self._benchmark_tf_matmul(
          m,
          transpose_b=True,
          num_iters=self._num_iters_100_by_784,
          execution_mode=context.ASYNC)

  def benchmark_gen_math_ops_matmul_100_by_784_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: generated matmul op, 100 by 784 times its transpose, GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_100_by_784.gpu()
      self._benchmark_gen_math_ops_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_tfe_py_execute_matmul_100_by_784_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: native execute entry point for matmul, 100 by 784, GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_100_by_784.gpu()
      self._benchmark_tfe_py_execute_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def benchmark_defun_matmul_100_by_784_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: traced matmul function, 100 by 784 times its transpose, GPU. Skipped if no GPU
    # exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = self._m_100_by_784.gpu()
      self._benchmark_defun_matmul(
          m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  @test_util.disable_tfrt(
      "b/169371527: Support inserting transfer op in lowering.")
  def benchmark_nested_defun_matmul_100_by_784_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: has no guard for machines without a GPU, unlike the other GPU benchmarks; it
    #   would fail there.
    # Benchmark: nested traced matmul functions on the 100 by 784 matrix copied to the GPU. Disabled
    # under the alternative runtime because of a known transfer-op limitation.
    m = self._m_100_by_784.gpu()
    self._benchmark_nested_defun_matmul(
        m, transpose_b=True, num_iters=self._num_iters_100_by_784)

  def _benchmark_forwardprop_matmul_CPU(self, shape):
    # [FUNCTION DOC]
    # HELPER: shared body for forward-mode differentiation benchmarks.
    # For a matrix shape, creates a random input and tangent on the CPU, then times a call that
    # computes matmul with transpose inside a forward accumulator and returns the result with its
    # Jacobian-vector product. 100 warmup calls and 3000 timed iterations.
    with ops.device(CPU):
      m = random_ops.random_uniform(shape).cpu()
      tangent = random_ops.random_uniform(shape).cpu()

      def func():
        # [FUNCTION DOC]
        # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
        # the operation under test and may return its result.
        with forwardprop.ForwardAccumulator(m, tangent) as acc:
          result = math_ops.matmul(m, m, transpose_b=True)
        return result, acc.jvp(result)

      # Warmup before benchmark
      for _ in range(100):
        func()
      self._run(func, 3000)

  def _benchmark_forwardprop_in_defun_matmul_CPU(self, shape):
    # [FUNCTION DOC]
    # HELPER: shared body for forward-mode differentiation inside a traced function.
    # Same as the forward-mode helper but the accumulator and matmul are inside a traced function.
    # 100 warmups, 3000 iterations.
    with ops.device(CPU):

      @def_function.function
      def compiled_function(x, tangent):
        # [FUNCTION DOC]
        # Traced function that computes matmul under a forward accumulator and returns the result
        # with its tangent.
        with forwardprop.ForwardAccumulator(x, tangent) as acc:
          result = math_ops.matmul(x, x, transpose_b=True)
        return result, acc.jvp(result)

      m = random_ops.random_uniform(shape).cpu()
      tangent = random_ops.random_uniform(shape).cpu()
      func = lambda: compiled_function(m, tangent)

      # Warmup before benchmark
      for _ in range(100):
        func()
      self._run(func, 3000)

  def _benchmark_forwardprop_in_defun_of_defun_matmul_CPU(self, shape):
    # [FUNCTION DOC]
    # HELPER: shared body for forward-mode differentiation inside a traced function that calls a
    #   traced matmul.
    # The accumulator lives in an outer traced function that calls an inner traced matmul. 100
    # warmups, 3000 iterations.
    with ops.device(CPU):
      matmul = def_function.function(math_ops.matmul)

      @def_function.function()
      def compiled_function(x, tangent):
        # [FUNCTION DOC]
        # Traced function that computes matmul under a forward accumulator and returns the result
        # with its tangent.
        with forwardprop.ForwardAccumulator(x, tangent) as acc:
          result = matmul(x, x, transpose_b=True)
        return result, acc.jvp(result)

      m = random_ops.random_uniform(shape).cpu()
      tangent = random_ops.random_uniform(shape).cpu()
      func = lambda: compiled_function(m, tangent)

      # Warmup before benchmark
      for _ in range(100):
        func()
      self._run(func, 3000)

  def _benchmark_forwardprop_of_defun_matmul_CPU(self, shape):
    # [FUNCTION DOC]
    # HELPER: shared body for forward-mode differentiation of a traced matmul.
    # The accumulator is applied eagerly around a traced matmul function. 100 warmups, 3000
    # iterations.
    with ops.device(CPU):
      m = random_ops.random_uniform(shape).cpu()
      tangent = random_ops.random_uniform(shape).cpu()
      matmul = def_function.function(math_ops.matmul)

      def func():
        # [FUNCTION DOC]
        # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
        # the operation under test and may return its result.
        with forwardprop.ForwardAccumulator(m, tangent) as acc:
          result = matmul(m, m, transpose_b=True)
        return result, acc.jvp(result)

      # Warmup before benchmark
      for _ in range(100):
        func()
      self._run(func, 3000)

  def benchmark_forwardprop_matmul_256_by_2096_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: eager forward-mode differentiation of matmul on a 256 by 2096 input.
    self._benchmark_forwardprop_matmul_CPU(shape=(256, 2096))

  def benchmark_forwardprop_in_defun_matmul_256_by_2096_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: forward-mode differentiation inside a traced function, 256 by 2096 input.
    self._benchmark_forwardprop_in_defun_matmul_CPU(shape=(256, 2096))

  def benchmark_forwardprop_in_defun_of_defun_matmul_256_by_2096_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: forward-mode differentiation inside a traced function calling a traced matmul, 256
    # by 2096 input.
    self._benchmark_forwardprop_in_defun_of_defun_matmul_CPU(shape=(256, 2096))

  def benchmark_forwardprop_of_defun_matmul_256_by_2096_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: forward-mode differentiation around a traced matmul, 256 by 2096 input.
    self._benchmark_forwardprop_of_defun_matmul_CPU(shape=(256, 2096))

  def benchmark_forwardprop_matmul_100_by_784_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: eager forward-mode differentiation of matmul on a 100 by 784 input.
    self._benchmark_forwardprop_matmul_CPU(shape=(100, 784))

  def benchmark_forwardprop_in_defun_matmul_100_by_784_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: forward-mode differentiation inside a traced function, 100 by 784 input.
    self._benchmark_forwardprop_in_defun_matmul_CPU(shape=(100, 784))

  def benchmark_forwardprop_in_defun_of_defun_matmul_100_by_784_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: forward-mode differentiation inside a traced function calling a traced matmul, 100
    # by 784 input.
    self._benchmark_forwardprop_in_defun_of_defun_matmul_CPU(shape=(100, 784))

  def benchmark_forwardprop_of_defun_matmul_100_by_784_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: forward-mode differentiation around a traced matmul, 100 by 784 input.
    self._benchmark_forwardprop_of_defun_matmul_CPU(shape=(100, 784))

  def _benchmark_tf_reduce_logsumexp(self,
                                     device=CPU,
                                     execution_mode=None,
                                     defunc=False,
                                     xla_compile=False):
    # [FUNCTION DOC]
    # HELPER: shared body for the log-sum-exp reduction benchmarks.
    # Parameters: device (default CPU), execution_mode (optional), defunc (wrap in a traced function
    # when true), xla_compile (enables just-in-time compilation of the traced function). Reduces a 2
    # by 2 constant for 3000 iterations.
    with context.device(device):
      x = constant_op.constant([[1, 0.], [0., 0.]])
      if defunc:
        reduce_func = def_function.function(
            math_ops.reduce_logsumexp, jit_compile=xla_compile)
        func = lambda: reduce_func(x)
      else:
        func = lambda: math_ops.reduce_logsumexp(x)
      self._run(func, 3000, execution_mode=execution_mode)

  def benchmark_tf_reduce_logsumexp_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: eager log-sum-exp reduction on the CPU.
    self._benchmark_tf_reduce_logsumexp()

  def benchmark_tf_reduce_logsumexp_CPU_async(self):
    # [FUNCTION DOC]
    # Benchmark: eager log-sum-exp reduction on the CPU, asynchronous.
    self._benchmark_tf_reduce_logsumexp(execution_mode=context.ASYNC)

  def benchmark_tf_reduce_logsumexp_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability, unlike other GPU benchmarks.
    # Benchmark: eager log-sum-exp reduction on the GPU.
    self._benchmark_tf_reduce_logsumexp(device=GPU)

  def benchmark_tf_reduce_logsumexp_GPU_async(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability, unlike other GPU benchmarks.
    # Benchmark: eager log-sum-exp reduction on the GPU, asynchronous.
    self._benchmark_tf_reduce_logsumexp(
        device=GPU, execution_mode=context.ASYNC)

  @test_util.disable_tfrt(
      "b/169371527: Support inserting transfer op in lowering.")
  def benchmark_tf_reduce_logsumexp_CPU_defunc(self):
    # [FUNCTION DOC]
    # Benchmark: log-sum-exp reduction in a traced function on the CPU. Disabled under the
    # alternative runtime.
    self._benchmark_tf_reduce_logsumexp(defunc=True)

  @test_util.disable_tfrt(
      "b/169371527: Support inserting transfer op in lowering.")
  def benchmark_tf_reduce_logsumexp_CPU_async_defun(self):
    # [FUNCTION DOC]
    # Benchmark: traced log-sum-exp on the CPU, asynchronous. Disabled under the alternative
    # runtime.
    self._benchmark_tf_reduce_logsumexp(
        execution_mode=context.ASYNC, defunc=True)

  def benchmark_tf_reduce_logsumexp_GPU_defun(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: traced log-sum-exp reduction on the GPU.
    self._benchmark_tf_reduce_logsumexp(device=GPU, defunc=True)

  def benchmark_tf_reduce_logsumexp_GPU_async_defun(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: traced log-sum-exp reduction on the GPU, asynchronous.
    self._benchmark_tf_reduce_logsumexp(
        device=GPU, execution_mode=context.ASYNC, defunc=True)

  def benchmark_tf_reduce_logsumexp_GPU_defun_compile(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: traced log-sum-exp on the GPU with just-in-time compilation enabled.
    self._benchmark_tf_reduce_logsumexp(
        device=GPU, defunc=True, xla_compile=True)

  def benchmark_tf_reduce_logsumexp_GPU_async_defun_compile(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: traced, compiled log-sum-exp on the GPU, asynchronous.
    self._benchmark_tf_reduce_logsumexp(
        device=GPU, execution_mode=context.ASYNC, defunc=True, xla_compile=True)

  def _benchmark_tf_tensordot(self, device=CPU, execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body for tensordot benchmarks.
    # Contracts two 2 by 2 ones matrices over the second axis of the first and the first axis of the
    # second, on the given device (default CPU) in the optional execution mode, 30000 iterations.
    with context.device(device):
      a = array_ops.ones((2, 2))
      b = array_ops.ones((2, 2))
      func = lambda: math_ops.tensordot(a, b, [[1], [0]])
      self._run(func, 30000, execution_mode=execution_mode)

  def benchmark_tf_tensordot_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: tensordot of two 2 by 2 matrices on the CPU.
    self._benchmark_tf_tensordot()

  def benchmark_tf_tensordot_CPU_async(self):
    # [FUNCTION DOC]
    # Benchmark: tensordot of two 2 by 2 matrices on the CPU, asynchronous.
    self._benchmark_tf_tensordot(execution_mode=context.ASYNC)

  def benchmark_tf_tensordot_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: tensordot of two 2 by 2 matrices on the GPU.
    self._benchmark_tf_tensordot(device=GPU)

  def benchmark_tf_tensordot_GPU_async(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: tensordot of two 2 by 2 matrices on the GPU, asynchronous.
    self._benchmark_tf_tensordot(device=GPU, execution_mode=context.ASYNC)

  def _benchmark_tf_zeros(self, shape, dtype, device=CPU):
    # [FUNCTION DOC]
    # HELPER: shared body for zeros benchmarks.
    # Times creating a zeros tensor of the given shape and dtype on the given device (default CPU),
    # 3000 iterations.
    with context.device(device):
      func = lambda: array_ops.zeros(shape, dtype)
      self._run(func, 3000)

  def benchmark_tf_zeros_2_by_2_float32_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 2 by 2 float32 zeros tensor on the CPU.
    self._benchmark_tf_zeros((2, 2), dtypes.float32)

  def benchmark_tf_zeros_2_by_2_bool_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 2 by 2 bool zeros tensor on the CPU.
    self._benchmark_tf_zeros((2, 2), dtypes.bool)

  def benchmark_tf_zeros_2_by_2_string_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 2 by 2 string zeros tensor on the CPU.
    self._benchmark_tf_zeros((2, 2), dtypes.string)

  def benchmark_tf_zeros_2_by_2_float32_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability, unlike other GPU benchmarks.
    # Benchmark: creating a 2 by 2 float32 zeros tensor on the GPU.
    self._benchmark_tf_zeros((2, 2), dtypes.float32, device=GPU)

  def benchmark_tf_zeros_2_by_2_bool_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability, unlike other GPU benchmarks.
    # Benchmark: creating a 2 by 2 bool zeros tensor on the GPU.
    self._benchmark_tf_zeros((2, 2), dtypes.bool, device=GPU)

  def benchmark_tf_zeros_30_by_30_float32_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 30 by 30 float32 zeros tensor on the CPU.
    self._benchmark_tf_zeros((30, 30), dtypes.float32)

  def benchmark_tf_zeros_30_by_30_bool_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 30 by 30 bool zeros tensor on the CPU.
    self._benchmark_tf_zeros((30, 30), dtypes.bool)

  def benchmark_tf_zeros_30_by_30_string_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 30 by 30 string zeros tensor on the CPU.
    self._benchmark_tf_zeros((30, 30), dtypes.string)

  def benchmark_tf_zeros_30_by_30_float32_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability, unlike other GPU benchmarks.
    # Benchmark: creating a 30 by 30 float32 zeros tensor on the GPU.
    self._benchmark_tf_zeros((30, 30), dtypes.float32, device=GPU)

  def benchmark_tf_zeros_30_by_30_bool_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability, unlike other GPU benchmarks.
    # Benchmark: creating a 30 by 30 bool zeros tensor on the GPU.
    self._benchmark_tf_zeros((30, 30), dtypes.bool, device=GPU)

  def benchmark_tf_zeros_100_by_100_float32_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 100 by 100 float32 zeros tensor on the CPU.
    self._benchmark_tf_zeros((100, 100), dtypes.float32)

  def benchmark_tf_zeros_100_by_100_bool_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 100 by 100 bool zeros tensor on the CPU.
    self._benchmark_tf_zeros((100, 100), dtypes.bool)

  def benchmark_tf_zeros_100_by_100_string_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: creating a 100 by 100 string zeros tensor on the CPU.
    self._benchmark_tf_zeros((100, 100), dtypes.string)

  def benchmark_tf_zeros_100_by_100_float32_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability, unlike other GPU benchmarks.
    # Benchmark: creating a 100 by 100 float32 zeros tensor on the GPU.
    self._benchmark_tf_zeros((100, 100), dtypes.float32, device=GPU)

  def benchmark_tf_zeros_100_by_100_bool_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability, unlike other GPU benchmarks.
    # Benchmark: creating a 100 by 100 bool zeros tensor on the GPU.
    self._benchmark_tf_zeros((100, 100), dtypes.bool, device=GPU)

  def _benchmark_tf_zeros_like(self, m, device=CPU):
    # [FUNCTION DOC]
    # HELPER: shared body for zeros-like benchmarks.
    # Times creating a zeros tensor shaped like m (tensor or variable) on the given device (default
    # CPU), 3000 iterations.
    with context.device(device):
      func = lambda: array_ops.zeros_like(m)
      self._run(func, 3000)

  def benchmark_tf_zeros_like_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: zeros-like of a 2 by 2 tensor on the CPU.
    self._benchmark_tf_zeros_like(self._m_2_by_2)

  def benchmark_tf_zeros_like_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: zeros-like of a 2 by 2 tensor on the GPU.
    self._benchmark_tf_zeros_like(self._m_2_by_2, device=GPU)

  def benchmark_tf_zeros_like_variable_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: zeros-like of a resource variable on the CPU.
    m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
    self._benchmark_tf_zeros_like(m)

  def benchmark_tf_zeros_like_variable_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: zeros-like of a resource variable on the GPU.
    m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
    self._benchmark_tf_zeros_like(m, device=GPU)

  def _benchmark_tf_random_uniform_2_by_2(self,
                                          shape=(2, 2),
                                          dtype=dtypes.int32,
                                          device=CPU):
    # [FUNCTION DOC]
    # HELPER: shared body for random-uniform benchmarks.
    # Times drawing a random uniform tensor of the given shape (default 2 by 2) and dtype (default
    # int32) with maximum value 3 on the given device (default CPU), using the standard 2 by 2
    # iteration count.
    with context.device(device):

      def func():
        # [FUNCTION DOC]
        # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
        # the operation under test and may return its result.
        return random_ops.random_uniform(shape, maxval=3, dtype=dtype)

      self._run(func, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_random_uniform_2_by_2_integer_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: random uniform integers, 2 by 2, CPU.
    self._benchmark_tf_random_uniform_2_by_2()

  def benchmark_tf_random_uniform_2_by_2_integer_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: random uniform integers, 2 by 2, GPU.
    self._benchmark_tf_random_uniform_2_by_2(device=GPU)

  def benchmark_tf_random_uniform_2_by_2_float_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: random uniform floats, 2 by 2, CPU.
    self._benchmark_tf_random_uniform_2_by_2(dtype=dtypes.float32)

  def benchmark_tf_random_uniform_2_by_2_float_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: random uniform floats, 2 by 2, GPU.
    self._benchmark_tf_random_uniform_2_by_2(dtype=dtypes.float32, device=GPU)

  def benchmark_tf_random_uniform_2_by_2_default_setting_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: random uniform with default dtype and range, 2 by 2, CPU.
    with context.device(CPU):
      func = lambda: random_ops.random_uniform((2, 2))
      self._run(func, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_random_uniform_2_by_2_default_setting_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: random uniform with default settings, 2 by 2, GPU.
    with context.device(GPU):
      func = lambda: random_ops.random_uniform((2, 2))
      self._run(func, num_iters=self._num_iters_2_by_2)

  def _benchmark_tf_dropout_2_by_2(self,
                                   rate=0.5,
                                   is_rate_tensor=True,
                                   noise_shape=None,
                                   device=CPU):
    # [FUNCTION DOC]
    # HELPER: shared body for dropout benchmarks.
    # Parameters: rate (default 0.5), is_rate_tensor (default true: convert rate to a float32
    # constant first), noise_shape (optional), device (default CPU). Times dropout on the 2 by 2
    # input with the standard iteration count.
    if is_rate_tensor:
      rate = constant_op.constant(rate, dtype=dtypes.float32)
    with context.device(device):

      def func():
        # [FUNCTION DOC]
        # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
        # the operation under test and may return its result.
        return nn_ops.dropout(
            self._m_2_by_2, rate=rate, noise_shape=noise_shape)

      self._run(func, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_dropout_scalar_rate_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: dropout with a Python scalar rate, CPU.
    self._benchmark_tf_dropout_2_by_2(is_rate_tensor=False)

  def benchmark_tf_dropout_scalar_rate_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: dropout with a Python scalar rate, GPU.
    self._benchmark_tf_dropout_2_by_2(is_rate_tensor=False, device=GPU)

  def benchmark_tf_dropout_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: dropout with a tensor rate, CPU.
    self._benchmark_tf_dropout_2_by_2()

  def benchmark_tf_dropout_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: dropout with a tensor rate, GPU.
    self._benchmark_tf_dropout_2_by_2(device=GPU)

  def benchmark_tf_dropout_scalar_rate_2_by_2_CPU_rate_0(self):
    # [FUNCTION DOC]
    # Benchmark: dropout with scalar rate zero, CPU.
    self._benchmark_tf_dropout_2_by_2(rate=0, is_rate_tensor=False)

  def benchmark_tf_dropout_scalar_rate_2_by_2_GPU_rate_0(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: dropout with scalar rate zero, GPU.
    self._benchmark_tf_dropout_2_by_2(
        rate=0.0, is_rate_tensor=False, device=GPU)

  def benchmark_tf_dropout_2_by_2_CPU_rate_0(self):
    # [FUNCTION DOC]
    # Benchmark: dropout with a tensor rate of zero, CPU.
    self._benchmark_tf_dropout_2_by_2(rate=0.0)

  def benchmark_tf_dropout_2_by_2_GPU_rate_0(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: dropout with a tensor rate of zero, GPU.
    self._benchmark_tf_dropout_2_by_2(rate=0, device=GPU)

  def _benchmark_transpose(self,
                           m,
                           num_iters,
                           perm=None,
                           conjugate=False,
                           execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: shared body for transpose benchmarks.
    # Times transposing m with optional permutation perm and conjugate flag (default false) for
    # num_iters iterations in the optional execution mode.
    func = lambda: array_ops.transpose(m, perm, conjugate)
    self._run(func, num_iters, execution_mode=execution_mode)

  def benchmark_tf_transpose_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: transpose of a 2 by 2 tensor on the CPU.
    with context.device(CPU):
      m = self._m_2_by_2.cpu()
      self._benchmark_transpose(m, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_transpose_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: transpose of a 2 by 2 tensor copied to the GPU.
    with context.device(GPU):
      m = self._m_2_by_2.gpu()
      self._benchmark_transpose(m, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_transpose_variable_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: transpose of a 2 by 2 resource variable on the CPU.
    with context.device(CPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
      self._benchmark_transpose(m, num_iters=self._num_iters_2_by_2)

  def benchmark_tf_transpose_variable_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: transpose of a 2 by 2 resource variable under the GPU device.
    with context.device(GPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
      self._benchmark_transpose(m, num_iters=self._num_iters_2_by_2)

  def benchmark_defun_without_signature(self):
    # [FUNCTION DOC]
    # Benchmark: calling a traced function with eight positional scalar tensors and no input
    # signature (cache lookup cost).

    def func(t1, t2, t3, t4, t5, t6, t7, t8):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      del t1, t2, t3, t4, t5, t6, t7, t8
      return None

    defined = def_function.function(func)
    t = constant_op.constant(0.0)
    cache_computation = lambda: defined(t, t, t, t, t, t, t, t)
    self._run(cache_computation, 30000)

  def benchmark_defun_without_signature_and_with_kwargs(self):
    # [FUNCTION DOC]
    # Benchmark: calling a traced function with eight keyword arguments and no input signature.

    def func(t1, t2, t3, t4, t5, t6, t7, t8):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      del t1, t2, t3, t4, t5, t6, t7, t8
      return None

    defined = def_function.function(func)
    t = constant_op.constant(0.0)

    def cache_computation():
      # [FUNCTION DOC]
      # Calls the traced function with eight keyword arguments or positional values for cache lookup
      # measurement.
      return defined(t1=t, t2=t, t3=t, t4=t, t5=t, t6=t, t7=t, t8=t)

    self._run(cache_computation, 30000)

  def benchmark_defun_with_signature(self):
    # [FUNCTION DOC]
    # Benchmark: calling a traced function with eight positional scalars and a fixed input
    # signature.

    def func(t1, t2, t3, t4, t5, t6, t7, t8):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      del t1, t2, t3, t4, t5, t6, t7, t8
      return None

    defined = def_function.function(
        func, input_signature=[tensor_spec.TensorSpec([], dtypes.float32)] * 8)
    t = constant_op.constant(0.0)
    signature_computation = lambda: defined(t, t, t, t, t, t, t, t)
    self._run(signature_computation, 30000)

  def benchmark_defun_with_signature_and_kwargs(self):
    # [FUNCTION DOC]
    # Benchmark: calling a traced function with eight keyword scalars and a fixed input signature.

    def func(t1, t2, t3, t4, t5, t6, t7, t8):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      del t1, t2, t3, t4, t5, t6, t7, t8
      return None

    defined = def_function.function(
        func, input_signature=[tensor_spec.TensorSpec([], dtypes.float32)] * 8)
    t = constant_op.constant(0.0)

    def signature_computation():
      # [FUNCTION DOC]
      # Calls the traced function with eight scalar arguments and signature.
      return defined(t1=t, t2=t, t3=t, t4=t, t5=t, t6=t, t7=t, t8=t)

    self._run(signature_computation, 30000)

  def benchmark_matmul_read_variable_op_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: matmul of a 2 by 2 resource variable on the CPU (includes variable read).
    with context.device(CPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
      self._benchmark_matmul_read_variable(m, num_iters=self._num_iters_2_by_2)

  def benchmark_matmul_read_variable_op_with_tape_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: matmul of a 2 by 2 variable on the CPU while a tape watches it.
    with context.device(CPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
      self._benchmark_matmul_read_variable_with_tape(
          m, num_iters=self._num_iters_2_by_2)

  def benchmark_read_variable_op_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: reading a 2 by 2 resource variable on the CPU.
    with context.device(CPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
      self._benchmark_read_variable(m, num_iters=self._num_iters_2_by_2)

  def benchmark_read_variable_op_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: reading a 2 by 2 resource variable placed on the GPU. Skipped if no GPU exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2.gpu())
      self._benchmark_read_variable(m, num_iters=self._num_iters_2_by_2)

  def benchmark_read_variable_op_with_tape_2_by_2_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: reading a 2 by 2 variable on the CPU while a tape watches it.
    with context.device(CPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2)
      self._benchmark_read_variable_with_tape(
          m, num_iters=self._num_iters_2_by_2)

  def benchmark_read_variable_op_with_tape_2_by_2_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: reading a 2 by 2 variable on the GPU while a tape watches it. Skipped if no GPU
    # exists.
    if not context.num_gpus():
      return
    with context.device(GPU):
      m = resource_variable_ops.ResourceVariable(self._m_2_by_2.gpu())
      self._benchmark_read_variable_with_tape(
          m, num_iters=self._num_iters_2_by_2)

  def benchmarkScan(self):
    # [FUNCTION DOC]
    # Benchmark: eager scan (running sum) over a range of 1600 elements with one parallel iteration,
    # 100 iterations.
    elems = math_ops.range(1600)

    def scan():
      # [FUNCTION DOC]
      # Callable that performs a running sum over the captured range of elements with one parallel
      # iteration.
      return functional_ops.scan(
          lambda a, x: a + x, elems, parallel_iterations=1)

    self._run(scan, 100)

  @test_util.disable_tfrt("tf.While not supported RTFB tensor. b/169374895")
  def benchmarkScanDefun(self):
    # [FUNCTION DOC]
    # Benchmark: scan over 1600 elements inside a traced function, 100 iterations. Disabled under
    # the alternative runtime.
    elems = math_ops.range(1600)

    @def_function.function
    def scan():
      # [FUNCTION DOC]
      # Callable that performs a running sum over the captured range of elements with one parallel
      # iteration.
      return functional_ops.scan(
          lambda a, x: a + x, elems, parallel_iterations=1)

    self._run(scan, 100)

  def benchmark_fastpath_conversion_type_inference(self):
    # [FUNCTION DOC]
    # Benchmark: adding a float32 constant and a Python integer so the fast path must infer the
    # type, 10000 iterations.
    c = constant_op.constant(1., dtype=dtypes.float32)

    def fn():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark; performs the
      # operation under test.
      return gen_math_ops.add(c, 1)

    self._run(fn, 10000)

  def benchmark_convert_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: converting an existing tensor to a tensor (identity conversion), 10000 iterations.
    value = ops.convert_to_tensor(42)

    def fn():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark; performs the
      # operation under test.
      return ops.convert_to_tensor(value)

    self._run(fn, 10000)

  def _benchmark_convert_constant(self, value, cached):
    # [FUNCTION DOC]
    # HELPER: shared body for conversion benchmarks.
    # Sets the module-level scratch value to value, then times converting to a tensor. When cached
    # is true the same value is converted every time; otherwise the scratch value is incremented
    # before each conversion. 10000 iterations. Mutates module state.
    global GLOBAL_TEST_VALUE
    GLOBAL_TEST_VALUE = value

    def cached_func():
      # [FUNCTION DOC]
      # Creates or converts the same value on every call, allowing caches to be used.
      ops.convert_to_tensor(value)

    def uncached_func():
      # [FUNCTION DOC]
      # Increments the module-level scratch value and uses the new value, defeating caches. Mutates
      # module state.
      global GLOBAL_TEST_VALUE
      GLOBAL_TEST_VALUE += 1
      ops.convert_to_tensor(GLOBAL_TEST_VALUE)

    func = cached_func if cached else uncached_func

    self._run(func, 10000)

  def benchmark_convert_python_int(self):
    # [FUNCTION DOC]
    # Benchmark: converting a repeated Python integer to a tensor.
    self._benchmark_convert_constant(42, cached=True)

  def benchmark_convert_python_int_uncached(self):
    # [FUNCTION DOC]
    # Benchmark: converting a changing Python integer to a tensor.
    self._benchmark_convert_constant(42, cached=False)

  def benchmark_convert_python_float(self):
    # [FUNCTION DOC]
    # Benchmark: converting a repeated Python float to a tensor.
    self._benchmark_convert_constant(42.0, cached=True)

  def benchmark_convert_python_float_uncached(self):
    # [FUNCTION DOC]
    # Benchmark: converting a changing Python float to a tensor.
    self._benchmark_convert_constant(42.0, cached=False)

  def benchmark_convert_numpy_int(self):
    # [FUNCTION DOC]
    # Benchmark: converting a repeated numpy integer to a tensor.
    self._benchmark_convert_constant(np.array(42), cached=True)

  def benchmark_convert_numpy_int_uncached(self):
    # [FUNCTION DOC]
    # Benchmark: converting a numpy integer scalar to a tensor, uncached path.
    self._benchmark_convert_constant(np.array(42), cached=False)

  def benchmark_convert_numpy_float(self):
    # [FUNCTION DOC]
    # Benchmark: converting a repeated numpy float to a tensor.
    self._benchmark_convert_constant(np.array(42.0), cached=True)

  def benchmark_convert_numpy_float_uncached(self):
    # [FUNCTION DOC]
    # Benchmark: converting a numpy float scalar to a tensor, uncached path.
    self._benchmark_convert_constant(np.array(42.0), cached=False)

  def benchmark_convert_3x_list_to_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: converting a three-element Python list to a tensor.
    xs = [1, 2, 3]
    self._run(lambda: ops.convert_to_tensor(xs), 1000)

  def benchmark_convert_3x_array_to_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: converting a three-element int32 numpy array to a tensor.
    xs = np.array([1, 2, 3], dtype=np.int32)
    self._run(lambda: ops.convert_to_tensor(xs), 1000)

  def benchmark_constant_40x2_list_to_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: building a constant from a 40 by 2 nested list.
    xs = [[0] * 2] * 40
    self._run(lambda: constant_op.constant(xs), 1000)

  def benchmark_constant_40x2_array_to_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: building a constant from a 40 by 2 int32 numpy array.
    xs = np.array([[0] * 2] * 40, dtype=np.int32)
    self._run(lambda: constant_op.constant(xs), 1000)

  def benchmark_constant_40x_list_of_2x_arrays_to_tensor(self):
    # [FUNCTION DOC]
    # Benchmark: building a constant from a list of forty two-element numpy arrays.
    xs = [np.array([0] * 2, dtype=np.int32)] * 40
    self._run(lambda: constant_op.constant(xs), 1000)

  def benchmark_constant_20x20x20_double_list_to_float32_tensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the data is wrapped in an extra outer list and has 21 innermost values, so its
    #   shape is 1 by 20 by 20 by 21, not 20 by 20 by 20 as the name suggests.
    # Benchmark: building a float32 constant from a nested list of doubles.
    xs = [[[np.linspace(0, 1, 21).tolist()] * 20] * 20]
    self._run(lambda: constant_op.constant(xs, dtype=dtypes.float32), 10000)

  def benchmark_constant_20x20x20_double_list_to_float64_tensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: same shape mismatch as the float32 variant: actual shape 1 by 20 by 20 by 21.
    # Benchmark: building a float64 constant from a nested list of doubles.
    xs = [[[np.linspace(0, 1, 21).tolist()] * 20] * 20]
    self._run(lambda: constant_op.constant(xs, dtype=dtypes.float64), 10000)

  def benchmark_list_of_zeros_to_np_array(self):
    # [FUNCTION DOC]
    # Benchmark: converting a list of 1000 eager zeros tensors (1000 elements each) to one numpy
    # array via per-tensor numpy conversion.
    values = []
    for _ in range(1000):
      values.append(array_ops.zeros(shape=(1000,)))
    self._run(lambda: np.array([x.numpy() for x in values]), 1000)

  def benchmark_function_trace(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the timed callable returns a lazy generator that is never consumed, so no
    #   function is traced and almost nothing is measured.
    # Benchmark intended to measure creating and invoking a traced function for each of 1000 inputs.

    def func(x):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      return x

    self._run(lambda: (def_function.function(func)(x) for x in range(1000)),
              30000)

  def _benchmarkFunctionWithResourceInputs(self, num_resources, num_iters):
    # [FUNCTION DOC]
    # HELPER: shared body for benchmarks of functions with many resource inputs.
    # Creates num_resources resource variables on the CPU and times calling a traced function that
    # sums them, num_iters times.

    @def_function.function
    def add_all(*args):
      # [FUNCTION DOC]
      # Traced function that sums all the resource variables passed in.
      return math_ops.add_n(*args)

    with context.device(CPU):
      resources = []
      for _ in range(num_resources):
        resources.append(resource_variable_ops.ResourceVariable(self._m_2))
      self._run(lambda: add_all(resources), num_iters)

  def benchmarkFunctionWithFiveResourceInputs(self):
    # [FUNCTION DOC]
    # Benchmark: traced function taking five resource variables (1000 iterations).
    self._benchmarkFunctionWithResourceInputs(5, 1000)

  def benchmarkFunctionWithFiveHundredResourceInputs(self):
    # [FUNCTION DOC]
    # Benchmark: traced function taking five hundred resource variables (100 iterations).
    self._benchmarkFunctionWithResourceInputs(500, 100)

  def _benchmarkResourceReadsInCondInInnerFunc(self, var_count):
    # [FUNCTION DOC]
    # HELPER: shared body for benchmarks of resource reads in a conditional of a nested traced
    #   function.
    # Creates var_count variables. The timed function defines new nested traced functions each call,
    # so graph-building is re-done every time; the inner one uses a conditional whose true branch
    # sums all variables. Runs 10 iterations on the CPU.
    rvars = []
    for _ in range(var_count):
      rvars.append(resource_variable_ops.ResourceVariable(1.0))

    # Note: We want to benchmark the graph building time so we intentionally
    # add this outer function so that the tf.function gets retraced every time.
    def benchmark_fn():
      # [FUNCTION DOC]
      # Callable timed by the benchmark; defines and calls a nested traced function each time so it
      # is retraced.

      @def_function.function
      def fn_with_many_reads():
        # [FUNCTION DOC]
        # Traced outer function that calls the inner traced function.

        @def_function.function
        def fn_with_many_reads_inner():
          # [FUNCTION DOC]
          # Traced inner function that returns a conditional on a constant true selecting between
          # summing the variables and zero.

          def then_branch():
            # [FUNCTION DOC]
            # True branch of the conditional: sums all captured variables.
            return math_ops.add_n(rvars)

          def else_branch():
            # [FUNCTION DOC]
            # False branch of the conditional: returns zero.
            return 0.

          return cond.cond(
              constant_op.constant(True), then_branch, else_branch)

        return fn_with_many_reads_inner()

      return fn_with_many_reads()

    with context.device(CPU):
      self._run(benchmark_fn, 10)

  def benchmarkTenThousandResourceReadsInCondInInnerFunc(self):
    # [FUNCTION DOC]
    # Benchmark: resource reads in conditional with 10000 variables.
    self._benchmarkResourceReadsInCondInInnerFunc(10000)

  def benchmarkHundredResourceReadsInCondInInnerFunc(self):
    # [FUNCTION DOC]
    # Benchmark: resource reads in conditional with 100 variables.
    self._benchmarkResourceReadsInCondInInnerFunc(100)

  def benchmarkTenResourceReadsInCondInInnerFunc(self):
    # [FUNCTION DOC]
    # Benchmark: resource reads in conditional with 10 variables.
    self._benchmarkResourceReadsInCondInInnerFunc(10)

  def benchmark_tf_name_scope(self):
    # [FUNCTION DOC]
    # Benchmark: entering and leaving a name scope, 10000 iterations.

    def fn():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark; performs the
      # operation under test.
      with ops.name_scope_v2("name"):
        pass

    self._run(fn, 10000)

  def benchmark_tf_nest_map_structure(self):
    # [FUNCTION DOC]
    # Benchmark: mapping an identity function over a small nested structure.
    nested = {"a": [1, 2, 3], "b": (4, 5, 6)}

    def fn():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark; performs the
      # operation under test.
      nest.map_structure(lambda x: x, nested)

    self._run(fn, 10000)

  def benchmark_tf_nest_pack_sequence_as(self):
    # [FUNCTION DOC]
    # Benchmark: packing a flat list back into a nested structure.
    nested = {"a": [1, 2, 3], "b": (4, 5, 6)}
    flat = nest.flatten(nested)

    def fn():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark; performs the
      # operation under test.
      nest.pack_sequence_as(nested, flat)

    self._run(fn, 10000)

  def benchmark_tf_nest_flatten_none(self):
    # [FUNCTION DOC]
    # Benchmark: flattening a null value, 100000 iterations.

    def fn():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark; performs the
      # operation under test.
      nest.flatten(None)

    self._run(fn, 100000)

  def benchmark_tf_nest_flatten(self):
    # [FUNCTION DOC]
    # Benchmark: flattening a small nested structure, 100000 iterations.
    nested = {"a": [1, 2, 3], "b": (4, 5, 6)}

    def fn():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark; performs the
      # operation under test.
      nest.flatten(nested)

    self._run(fn, 100000)

  def benchmark_tf_flatten_dict_items(self):
    # [FUNCTION DOC]
    # Benchmark: flattening a mapping whose keys and values are nested structures.
    nested = {(4, 5, (6, 8)): ("a", "b", ("c", "d"))}

    def fn():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark; performs the
      # operation under test.
      nest.flatten_dict_items(nested)

    self._run(fn, 100000)

  def benchmark_tf_nn_convolution_overhead(self):
    # [FUNCTION DOC]
    # Benchmark: overhead of the convolution wrapper on 1 by 1 by 1 by 1 inputs.
    inputs = array_ops.ones((1, 1, 1, 1))
    filters = array_ops.ones((1, 1, 1, 1))

    def fn():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark; performs the
      # operation under test.
      nn_ops.convolution_v2(inputs, filters)

    self._run(fn, 10000)

  def benchmark_tf_tensor_shape_creation_overhead(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses a private tensor method for the shape tuple.
    # Benchmark: constructing a shape object from a tuple, which happens the first time a tensor
    # shape is requested.
    # A `TensorShape` is created the first time `EagerTensor.shape` is
    # called, which puts `TensorShape.__init__` on the hotpath. The
    # `TensorShape` is created from `EagerTensor._shape_tuple`.

    x = array_ops.ones((1, 1))
    shape_tuple = x._shape_tuple()

    def fn():
      # [FUNCTION DOC]
      # Zero-argument callable run in the timing loop by the enclosing benchmark; performs the
      # operation under test.
      tensor_shape.TensorShape(shape_tuple)

    self._run(fn, 100000)

  def _boolean_mask_input(self):
    # [FUNCTION DOC]
    # HELPER: builds inputs for the boolean mask benchmarks.
    # Returns a pair of a 3000 by 3000 tensor of ones and a same-shape boolean tensor filled with
    # true.
    n = 3000
    return (array_ops.ones([n, n]), array_ops.fill([n, n], True))

  def _boolean_mask_fn(self, input_tensor, mask):
    # [FUNCTION DOC]
    # HELPER: operation under test for the boolean mask benchmarks.
    # Applies boolean masking to input_tensor with mask and returns the result.
    return array_ops.boolean_mask(input_tensor, mask)

  def benchmark_tf_boolean_mask_eager(self):
    # [FUNCTION DOC]
    # Benchmark: boolean mask run eagerly on a large all-true mask.
    input_tensor, mask = self._boolean_mask_input()

    self._run(lambda: self._boolean_mask_fn(input_tensor, mask), 10000)

  def benchmark_tf_boolean_mask_graph(self):
    # [FUNCTION DOC]
    # Benchmark: boolean mask wrapped in a traced function.
    input_tensor, mask = self._boolean_mask_input()
    compiled_fn = def_function.function(self._boolean_mask_fn)

    self._run(lambda: compiled_fn(input_tensor, mask), 10000)

  def _benchmark_tf_range_var(self,
                              limit=100,
                              dtype=dtypes.int32,
                              range_dtype=dtypes.int32,
                              device=CPU,
                              num_iters=1000):
    # [FUNCTION DOC]
    # HELPER: shared body for range-loop-with-variable benchmarks.
    # Traces a function that loops over a range of the given limit, dtype range_dtype, adding one to
    # a variable each step. Variable and limit are built on the CPU; the function is called once to
    # trace, then timed on the device for num_iters.

    def func(v, lim):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      for _ in math_ops.range(lim, dtype=range_dtype):
        v.assign_add(constant_op.constant(1, dtype=dtype))
      return v

    compiled_func = def_function.function(func)

    with context.device(CPU):
      m = resource_variable_ops.ResourceVariable(
          constant_op.constant(1, dtype=dtype), dtype=dtype)
      limit_t = constant_op.constant(limit, dtype=dtype)

    with context.device(device):
      compiled_func(m, limit_t)
      self._run(lambda: compiled_func(m, limit_t), num_iters=num_iters)

  def benchmark_tf_range_var_int32_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: range loop updating an int32 variable, CPU.
    self._benchmark_tf_range_var()

  def benchmark_tf_range_var_int64_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: range loop updating an int64 variable with int64 range, CPU.
    self._benchmark_tf_range_var(dtype=dtypes.int64, range_dtype=dtypes.int64)

  def benchmark_tf_range_var_int32_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: range loop updating an int32 variable, GPU.
    self._benchmark_tf_range_var(device=GPU)

  def benchmark_tf_range_var_int64_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: range loop updating an int64 variable, GPU.
    self._benchmark_tf_range_var(
        dtype=dtypes.int64, range_dtype=dtypes.int64, device=GPU)

  def _benchmark_tf_range_const(self,
                                limit=100,
                                dtype=dtypes.int32,
                                range_dtype=dtypes.int32,
                                device=CPU,
                                num_iters=1000):
    # [FUNCTION DOC]
    # HELPER: shared body for range-loop-with-constant benchmarks.
    # Traces a function that loops over a range of dtype range_dtype adding one to a constant.
    # Inputs on CPU, called once to trace, then timed on the device.

    def func(c, lim):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      for _ in math_ops.range(lim, dtype=range_dtype):
        c += 1
      return c

    compiled_func = def_function.function(func)

    with context.device(CPU):
      input_c = constant_op.constant(1, dtype=dtype)
      limit_t = constant_op.constant(limit, dtype=dtype)

    with context.device(device):
      compiled_func(input_c, limit_t)
      self._run(lambda: compiled_func(input_c, limit_t), num_iters=num_iters)

  # int32 constant, int32 range, CPU
  def benchmark_tf_range_const_int32_int32_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: range loop with a constant accumulator; constant type, range type and device are
    # given by the benchmark name.
    self._benchmark_tf_range_const()

  # int32 constant, int64 range, CPU
  def benchmark_tf_range_const_int32_int64_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: range loop with a constant accumulator; constant type, range type and device are
    # given by the benchmark name.
    self._benchmark_tf_range_const(range_dtype=dtypes.int64)

  # int64 constant, int32 range, CPU
  def benchmark_tf_range_const_int64_int32_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: range loop with a constant accumulator; constant type, range type and device are
    # given by the benchmark name.
    self._benchmark_tf_range_const(dtype=dtypes.int64)

  # int64 constant, int64 range, CPU
  def benchmark_tf_range_const_int64_int64_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: range loop with a constant accumulator; constant type, range type and device are
    # given by the benchmark name.
    self._benchmark_tf_range_const(dtype=dtypes.int64, range_dtype=dtypes.int64)

  # int32 constant, int32 range, GPU
  def benchmark_tf_range_const_int32_int32_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: range loop with a constant accumulator; constant type, range type and device are
    # given by the benchmark name.
    self._benchmark_tf_range_const(device=GPU)

  # int32 constant, int64 range, GPU
  def benchmark_tf_range_const_int32_int64_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: range loop with a constant accumulator; constant type, range type and device are
    # given by the benchmark name.
    self._benchmark_tf_range_const(range_dtype=dtypes.int64, device=GPU)

  # int64 constant, int32 range, GPU
  def benchmark_tf_range_const_int64_int32_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: range loop with a constant accumulator; constant type, range type and device are
    # given by the benchmark name.
    self._benchmark_tf_range_const(dtype=dtypes.int64, device=GPU)

  # int64 constant, int64 range, GPU
  def benchmark_tf_range_const_int64_int64_GPU(self):
    # [FUNCTION DOC]
    # Benchmark: range loop with a constant accumulator; constant type, range type and device are
    # given by the benchmark name.
    self._benchmark_tf_range_const(
        dtype=dtypes.int64, range_dtype=dtypes.int64, device=GPU)

  def _benchmark_tf_range_return(self,
                                 limit=100000,
                                 dtype=dtypes.int32,
                                 device=CPU,
                                 num_iters=100000):
    # [FUNCTION DOC]
    # HELPER: shared body for range-returning benchmarks.
    # Traces a function that returns a range of the given limit (default 100000) and dtype, then
    # times calling it on the device (default CPU) for num_iters.

    def func(lim):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      return math_ops.range(lim, dtype=dtype)

    compiled_func = def_function.function(func)

    with context.device(device):
      limit_t = constant_op.constant(limit, dtype=dtype)
      compiled_func(limit_t)
      self._run(lambda: compiled_func(limit_t), num_iters=num_iters)

  def benchmark_tf_range_return_int32_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: returning an int32 range, CPU.
    self._benchmark_tf_range_return()

  def benchmark_tf_range_return_int64_CPU(self):
    # [FUNCTION DOC]
    # Benchmark: returning an int64 range, CPU.
    self._benchmark_tf_range_return(dtype=dtypes.int64)

  def benchmark_tf_range_return_int32_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: returning an int32 range, GPU.
    self._benchmark_tf_range_return(device=GPU)

  def benchmark_tf_range_return_int64_GPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: returning an int64 range, GPU.
    self._benchmark_tf_range_return(dtype=dtypes.int64, device=GPU)

  def _benchmark_embedding_lookup_sparse_with_sparse_input(
      self, allow_fast_lookup=True, batch_size=32000, device=GPU
  ):
    # [FUNCTION DOC]
    # HELPER: shared body for sparse-embedding-lookup benchmarks with sparse input.
    # Builds random ids and row ids on the device (default GPU), makes a sparse input through a
    # ragged tensor, and times embedding lookup with the allow_fast_lookup flag. 2000 iterations.
    def func(sp_ids):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      return embedding_ops.embedding_lookup_sparse(
          self._m_10000_by_16, sp_ids, None, allow_fast_lookup=allow_fast_lookup
      )

    with context.device(device):
      values = random_ops.random_uniform(
          shape=(batch_size,), minval=1, maxval=10000, dtype=dtypes.int64
      )
      value_rowids = ops.EagerTensor(np.arange(batch_size), device=device)

      ragged_input = ragged_tensor.RaggedTensor.from_value_rowids(
          values, value_rowids
      )
      sparse_input = ragged_input.to_sparse()
      func(sparse_input)
      self._run(lambda: func(sparse_input), num_iters=2000)

  def benchmark_tf_embedding_lookup_sparse_with_sparse_input_sparse_grads(self):
    # [FUNCTION DOC]
    # Benchmark: sparse-input embedding lookup with fast lookup off.
    self._benchmark_embedding_lookup_sparse_with_sparse_input(
        allow_fast_lookup=False
    )

  def benchmark_tf_embedding_lookup_sparse_with_sparse_input_dense_grads(self):
    # [FUNCTION DOC]
    # Benchmark: sparse-input embedding lookup with fast lookup on.
    self._benchmark_embedding_lookup_sparse_with_sparse_input(
        allow_fast_lookup=True
    )

  def _benchmark_embedding_lookup_sparse_with_ragged_input(
      self, allow_fast_lookup=True, batch_size=32000, device=GPU
  ):
    # [FUNCTION DOC]
    # HELPER: shared body for sparse-embedding-lookup benchmarks with ragged input.
    # Same as the sparse-input helper but passes the ragged tensor directly. 2000 iterations.
    def func(sp_ids):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      return embedding_ops.embedding_lookup_sparse(
          self._m_10000_by_16, sp_ids, None, allow_fast_lookup=allow_fast_lookup
      )

    with context.device(device):
      values = random_ops.random_uniform(
          shape=(batch_size,), minval=1, maxval=10000, dtype=dtypes.int64
      )
      value_rowids = ops.EagerTensor(np.arange(batch_size), device=device)

      ragged_input = ragged_tensor.RaggedTensor.from_value_rowids(
          values, value_rowids
      )
      func(ragged_input)
      self._run(lambda: func(ragged_input), num_iters=2000)

  def benchmark_embedding_lookup_sparse_with_ragged_input_sparse_grads(self):
    # [FUNCTION DOC]
    # Benchmark: ragged-input embedding lookup with fast lookup off.
    self._benchmark_embedding_lookup_sparse_with_ragged_input(
        allow_fast_lookup=False
    )

  def benchmark_embedding_lookup_sparse_with_ragged_input_dense_grads(self):
    # [FUNCTION DOC]
    # Benchmark: ragged-input embedding lookup with fast lookup on.
    self._benchmark_embedding_lookup_sparse_with_ragged_input(
        allow_fast_lookup=True
    )

  def _RandomIdsAndWeights(self, batch_size, vocab_size, max_val_per_entry):
    # [FUNCTION DOC]
    # HELPER: builds random sparse ids and weights for the embedding gradient benchmark.
    # Given batch_size, vocab_size and max_val_per_entry, draws per-row value counts, random ids and
    # weights, builds the index list and returns a pair of sparse tensors (ids as int32, weights as
    # float32) with shape batch_size by max_val_per_entry. Uses numpy randomness.
    vals_per_batch_entry = np.random.randint(
        1, max_val_per_entry, size=batch_size
    )
    num_vals = np.sum(vals_per_batch_entry)

    ids = np.random.randint(vocab_size, size=num_vals)
    weights = 1 + np.random.rand(num_vals)

    indices = []
    for batch_entry, num_val in enumerate(vals_per_batch_entry):
      for val_index in range(num_val):
        indices.append([batch_entry, val_index])

    shape = [batch_size, max_val_per_entry]

    sp_ids = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(ids, dtypes.int32),
        constant_op.constant(shape, dtypes.int64),
    )
    sp_weights = sparse_tensor.SparseTensor(
        constant_op.constant(indices, dtypes.int64),
        constant_op.constant(weights, dtypes.float32),
        constant_op.constant(shape, dtypes.int64),
    )
    return sp_ids, sp_weights

  def _benchmark_embedding_lookup_sparse_with_gradient(
      self, params, batch_size, max_val_per_entry, device
  ):
    # [FUNCTION DOC]
    # HELPER: shared body for embedding lookup with gradient.
    # Builds random sparse ids for params, then times a function that computes the lookup under a
    # gradient tape inside a forward-compatibility horizon and returns the gradient of params. 2000
    # iterations on the device.
    def func(sp_ids):
      # [FUNCTION DOC]
      # Zero-argument callable executed in the timing loop by the enclosing benchmark; it performs
      # the operation under test and may return its result.
      with forward_compat.forward_compatibility_horizon(2023, 9, 26):
        with gradients.GradientTape() as g:
          y = embedding_ops.embedding_lookup_sparse(params, sp_ids, None)
        params_grad = g.gradient(y, params)
        return params_grad

    vocab_size = params.get_shape()[0]
    with context.device(device):
      sp_ids, _ = self._RandomIdsAndWeights(
          batch_size, vocab_size, max_val_per_entry
      )
      func(sp_ids)
      self._run(lambda: func(sp_ids), num_iters=2000)

  def benchmark_embedding_lookup_sparse_with_gradient(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no check for GPU availability.
    # Benchmark: sparse embedding lookup gradient for a 1048576 by 16 variable on the GPU, batch
    # size 32768, up to 64 values per row.
    params = random_ops.random_uniform((1024 * 1024, 16))
    params = params.gpu()
    params = resource_variable_ops.ResourceVariable(params)
    self._benchmark_embedding_lookup_sparse_with_gradient(
        params, batch_size=32768, max_val_per_entry=64, device=GPU
    )


if __name__ == "__main__":
  test.main()
