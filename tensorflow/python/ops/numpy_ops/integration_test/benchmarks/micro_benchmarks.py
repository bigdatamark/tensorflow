# [MODULE DOC]
# Micro benchmarks comparing plain numpy with tf.experimental.numpy: MLP inference (numpy,
# tf-numpy eager, tf-numpy inside a compiled function) and per-op timings (count_nonzero, log,
# exp, tanh, matmul) over several input sizes, reported through the TensorFlow benchmark
# framework. Uses the sibling numpy_mlp and tf_numpy_mlp modules for the models.
# Import-time side effects: defines the command-line flags repeat and number. When run as a
# script it lowers log verbosity, enables v2 behavior and starts the test runner.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Micro benchmark.

bazel run -c opt --config=cuda \
  //third_party/tensorflow/python/ops/numpy_ops/integration_test/benchmarks:micro_benchmarks -- \
  --number=100 --repeat=100 \
  --benchmark_filter=.
"""
import gc
import time

from absl import flags
from absl import logging

import numpy as np  # pylint: disable=unused-import
import tensorflow.compat.v2 as tf

from tensorflow.python.ops.numpy_ops.integration_test.benchmarks import numpy_mlp
from tensorflow.python.ops.numpy_ops.integration_test.benchmarks import tf_numpy_mlp

FLAGS = flags.FLAGS

# Used instead of "import tensorflow(dot)experimental.numpy as tfnp" due to
# copybara issues.
tfnp = tf.experimental.numpy

flags.DEFINE_integer('repeat', 100, '#Measurements per benchmark.')
flags.DEFINE_integer('number', 100, '#Runs per a measure.')


class MicroBenchmarks(tf.test.Benchmark):
  # [CLASS DOC]
  # Benchmark suite (TensorFlow Benchmark subclass) whose methods named benchmark_* are
  # discovered by the benchmark runner. Holds no state of its own; helper methods time callables
  # and print comparison tables between numpy and tf-numpy.
  """Main micro benchmark class."""

  def _benchmark_and_report(
      self,
      name,
      fn,
      repeat=None,
      number=None):
    # [FUNCTION DOC]
    # HELPER: timing engine for all benchmarks in this class.
    # Inputs: a benchmark name, a callable taking no arguments, and optional repeat and number
    # counts (defaulting at call time to the command-line flags, because flags are not parsed at
    # import time). Calls fn once as warmup, then repeat times measures the wall time of number
    # consecutive calls with garbage collection disabled during each measurement (collected
    # afterwards). Reports total wall time with the fastest per-call time in microseconds as an
    # extra through the benchmark framework, and returns that fastest time in microseconds.
    """Run fn repeat * number times, report time, and return fastest time."""
    # Can't make these default above since the flags may not have been parsed
    # at module import time.
    repeat = repeat or int(FLAGS.repeat)
    number = number or int(FLAGS.number)

    # Warmup
    fn()

    times = []
    for _ in range(repeat):
      gc.disable()
      start = time.time()
      for _ in range(number):
        fn()
      times.append(time.time() - start)
      gc.enable()
      gc.collect()

    # Regular benchmark to report numbers.
    fastest_time_us = min(times) * 1e6 / number
    total_time = sum(times)
    self.report_benchmark(name=name,
                          wall_time=total_time,
                          extras={'fastest_time_us': fastest_time_us})

    return fastest_time_us

  def benchmark_tf_np_mlp_inference_batch_1_cpu(self):
    # [FUNCTION DOC]
    # Benchmark: on the CPU device builds the tf-numpy MLP and a one-row float32 input, then times
    # eager inference. Reports through the benchmark framework; returns nothing.
    with tf.device('/CPU:0'):
      model = tf_numpy_mlp.MLP()
      x = tfnp.ones(shape=(1, 10)).astype(np.float32)
      self._benchmark_and_report(self._get_name(), lambda: model.inference(x))

  def benchmark_tf_np_tf_function_mlp_inference_batch_1_cpu(self):
    # [FUNCTION DOC]
    # Benchmark: same as the eager tf-numpy MLP benchmark but the inference call is wrapped in a
    # compiled tf function before timing. Runs on the CPU device; returns nothing.
    with tf.device('/CPU:0'):
      model = tf_numpy_mlp.MLP()
      x = tfnp.ones(shape=(1, 10)).astype(np.float32)
      self._benchmark_and_report(
          self._get_name(), tf.function(lambda: model.inference(x)))

  def benchmark_numpy_mlp_inference_batch_1_cpu(self):
    # [FUNCTION DOC]
    # Benchmark: baseline using the pure numpy MLP with a random one-row float32 input. Reports
    # timing through the benchmark framework; returns nothing.
    model = numpy_mlp.MLP()
    x = np.random.uniform(size=(1, 10)).astype(np.float32, copy=False)
    self._benchmark_and_report(self._get_name(), lambda: model.inference(x))

  def _benchmark_np_and_tf_np(self, name, op, args, repeat=None):  # pylint: disable=redefined-builtin
    # [FUNCTION DOC]
    # HELPER: runs one op in both libraries for the benchmarks below.
    # Inputs: a benchmark name prefix, an op name (looked up by name in both numpy and
    # tf-numpy), a tuple of arguments, and an optional repeat count. Times the numpy function and
    # then the tf-numpy function on the CPU device, with report names suffixed numpy and tfnp_cpu.
    # Asserts that both functions exist. Returns the pair (numpy fastest time, tf-numpy fastest
    # time) in microseconds.
    fn = getattr(np, op)
    assert fn is not None

    np_time = self._benchmark_and_report(
        '{}_numpy'.format(name), lambda: fn(*args), repeat=repeat)

    fn = getattr(tfnp, op)
    assert fn is not None

    with tf.device('CPU:0'):
      tf_time = self._benchmark_and_report(
          '{}_tfnp_cpu'.format(name), lambda: fn(*args), repeat=repeat)

    return np_time, tf_time

  def _print_times(self, op, sizes, times):
    # [FUNCTION DOC]
    # HELPER: console reporting for the per-op benchmarks.
    # Inputs: op name, a list of input sizes and a parallel list of (numpy time, tf-numpy time)
    # pairs in microseconds. Prints a titled table with one row per size. Returns nothing.
    # For easy reporting.
    print('For np.{}:'.format(op))
    print('{:<15}  {:>11}  {:>11}'.format('Size', 'NP time', 'TF NP Time'))
    for size, (np_time, tf_time) in zip(sizes, times):
      print('{:<15} {:>10.5}us {:>10.5}us'.format(
          str(size), np_time, tf_time))
    print()

  def _benchmark_np_and_tf_np_unary(self, op):
    # [FUNCTION DOC]
    # HELPER: shared body of the unary-op benchmarks.
    # Input is an op name taking one array. For vector sizes of one hundred, ten thousand and one
    # million random float32 elements (repeat count from the flag for the first two and ten for
    # the last), times numpy against tf-numpy and finally prints the comparison table. Returns
    # nothing.
    sizes = [(100,), (10000,), (1000000,)]
    repeats = [FLAGS.repeat] * 2 + [10]
    times = []
    for size, repeat in zip(sizes, repeats):
      x = np.random.uniform(size=size).astype(np.float32, copy=False)
      name = '{}_{}'.format(self._get_name(), size)
      times.append(self._benchmark_np_and_tf_np(name, op, (x,), repeat))
    self._print_times(op, sizes, times)

  def benchmark_count_nonzero(self):
    # [FUNCTION DOC]
    # Benchmark: compares numpy and tf-numpy count_nonzero on vectors of three sizes.
    self._benchmark_np_and_tf_np_unary('count_nonzero')

  def benchmark_log(self):
    # [FUNCTION DOC]
    # Benchmark: compares numpy and tf-numpy log on vectors of three sizes.
    self._benchmark_np_and_tf_np_unary('log')

  def benchmark_exp(self):
    # [FUNCTION DOC]
    # Benchmark: compares numpy and tf-numpy exp on vectors of three sizes.
    self._benchmark_np_and_tf_np_unary('exp')

  def benchmark_tanh(self):
    # [FUNCTION DOC]
    # Benchmark: compares numpy and tf-numpy tanh on vectors of three sizes.
    self._benchmark_np_and_tf_np_unary('tanh')

  def benchmark_matmul(self):
    # [FUNCTION DOC]
    # Benchmark: compares numpy and tf-numpy matrix multiplication of a random float32 square
    # matrix with itself for sizes 2, 10, 100, 200 and 1000 per side. Repeat counts are lowered
    # for the largest sizes because they are slow. Prints the comparison table; returns nothing.
    sizes = [(2, 2), (10, 10), (100, 100), (200, 200), (1000, 1000)]
    # Override repeat flag since this can be very slow.
    repeats = [FLAGS.repeat] * 3 + [50, 10]
    times = []
    for size, repeat in zip(sizes, repeats):
      x = np.random.uniform(size=size).astype(np.float32, copy=False)
      name = '{}_{}'.format(self._get_name(), size)
      times.append(
          self._benchmark_np_and_tf_np(name, 'matmul', (x, x), repeat=repeat))

    self._print_times('matmul', sizes, times)


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.enable_v2_behavior()
  tf.test.main()
