# [MODULE DOC]
# Benchmark module comparing the experimental map-defun operation (applying a traced function over
# the leading dimension of a tensor) against the standard map-function utility, for several input
# sizes. Not a library module; run directly it starts the benchmark runner.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Benchmarks for MapDefunOp."""

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import map_defun
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops


class MapDefunBenchmark(benchmark_base.DatasetBenchmarkBase):
  # [CLASS DOC]
  # Benchmark suite comparing map-defun with map-fn on an identity function. Stateless; uses the
  # dataset benchmark base class for op benchmarking and reporting.
  """Benchmarks for MapDefunOp."""

  def _run(self, op, name, num_iters, benchmark_id):
    # [FUNCTION DOC]
    # HELPER: runs one op benchmark and reports it. op is the graph operation or tensor to time,
    # name (text) the report name, num_iters (int) the iteration count, benchmark_id (int) used in
    # the model name. Benchmarks the op with warmup, adds a tiny delta to the wall time to avoid
    # division by zero, and reports iterations, wall time and extras (examples per second as the
    # inverse of wall time, model name, iteration count as parameter). Returns nothing.

    wall_time = self.run_op_benchmark(op=op, iters=num_iters, warmup=True)
    zero_division_delta = 1e-100
    wall_time = wall_time + zero_division_delta
    self.report_benchmark(
        name=name,
        iters=num_iters,
        wall_time=wall_time,
        extras={
            "examples_per_sec": 1 / float(wall_time),
            "model_name": "map_defun.benchmark.%d" % benchmark_id,
            "parameters": "%d" % num_iters,
        })

  def benchmark_defun_vs_map_fn(self):
    # [FUNCTION DOC]
    # Benchmark: for input sizes 10, 100, 1000 and 10000, applies an identity function over a
    # range of 10000 integers once with map-defun (traced function with an integer scalar
    # signature) and once with map-fn, running each for 10000 divided by input size iterations
    # and reporting them under "with_defun" and "without_defun" names (ids 1 and 2). Returns nothing.
    # KNOWN ISSUE: the input size only changes the iteration count; both ops always process the
    # full 10000-element range.
    """Benchmarks to compare the performance of MapDefun vs tf.map_fn."""

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec([], dtypes.int32)])
    def defun(x):
      # [FUNCTION DOC]
      # HELPER: traced identity function (integer scalar input) passed to map-defun; returns x.
      return array_ops.identity(x)

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: identity function passed to map-fn; returns x.
      return array_ops.identity(x)

    base = math_ops.range(10000)
    for input_size in [10, 100, 1000, 10000]:
      num_iters = 10000 // input_size
      map_defun_op = map_defun.map_defun(defun, [base], [dtypes.int32], [()])
      map_fn_op = map_fn.map_fn(fn, base)

      self._run(
          op=map_defun_op,
          name="with_defun_size_%d" % input_size,
          num_iters=num_iters,
          benchmark_id=1)
      self._run(
          op=map_fn_op,
          name="without_defun_size_%d" % input_size,
          num_iters=num_iters,
          benchmark_id=2)


if __name__ == "__main__":
  benchmark_base.test.main()
