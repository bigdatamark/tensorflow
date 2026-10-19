# [MODULE DOC]
# Meta benchmarks comparing ways of measuring dataset iteration speed (native-only iteration,
# session run, session callable, eager) on fast and slow datasets. Does not use the shared base.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test utilities for tf.data benchmarking functionality."""
import timeit

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.platform import test


class MetaBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark suite that compares ways of measuring dataset iteration speed: native-only
  # iteration via skip, session run, session callable, and eager iteration, each on a fast and a
  # slow (1 ms sleep) dataset. Instance state: num_reps (repetitions) and iters (iterations per
  # repetition), set by the setup methods. Each benchmark is its own method so it can be profiled.
  """Benchmark that compares various ways of running tf.data benchmarks."""

  # Note that each of these benchmarks is a separate method so that we can
  # run them independently and collect a performance profile.

  def setup_fast_dataset(self):
    # [FUNCTION DOC]
    # HELPER: fixture for the fast-dataset benchmarks.
    # Sets num_reps to 15 and iters to 100000 on the instance, and returns a range dataset of
    # about one hundred million elements with default optimizations disabled.
    self.num_reps = 15
    self.iters = 100000
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    return dataset_ops.Dataset.range(10000**2).with_options(options)

  def benchmark_fast_dataset_with_only_cpp_iterations(self):
    # [FUNCTION DOC]
    # Benchmarks the fast dataset iterated entirely in native code.
    dataset = self.setup_fast_dataset()
    self.run_benchmark_with_only_cpp_iterations(dataset)

  def benchmark_fast_dataset_with_session_run(self):
    # [FUNCTION DOC]
    # Benchmarks the fast dataset fetched one element at a time with session run calls.
    dataset = self.setup_fast_dataset()
    self.run_benchmark_with_session_run(dataset)

  def benchmark_fast_dataset_with_session_callable(self):
    # [FUNCTION DOC]
    # Benchmarks the fast dataset fetched one element at a time through a prebuilt session callable.
    dataset = self.setup_fast_dataset()
    self.run_benchmark_with_session_run(dataset, make_callable=True)

  def benchmark_fast_dataset_in_eager(self):
    # [FUNCTION DOC]
    # Within an eager mode scope, benchmarks the fast dataset iterated with a Python iterator.
    with context.eager_mode():
      dataset = self.setup_fast_dataset()
      self.run_benchmark_in_eager(dataset)

  def setup_slow_dataset(self):
    # [FUNCTION DOC]
    # HELPER: fixture for the slow-dataset benchmarks.
    # Builds the fast dataset (setting repetitions and iterations), reduces iters to 1000, and
    # returns the dataset with a one-millisecond sleep per element.
    dataset = self.setup_fast_dataset()
    self.iters = 1000
    # sleep for 1e-3s per iteration
    return dataset.apply(testing.sleep(1000))

  def benchmark_slow_dataset_with_only_cpp_iterations(self):
    # [FUNCTION DOC]
    # Benchmarks the slow dataset iterated entirely in native code.
    dataset = self.setup_slow_dataset()
    self.run_benchmark_with_only_cpp_iterations(dataset)

  def benchmark_slow_dataset_with_session_run(self):
    # [FUNCTION DOC]
    # Benchmarks the slow dataset fetched one element at a time with session run calls.
    dataset = self.setup_slow_dataset()
    self.run_benchmark_with_session_run(dataset)

  def benchmark_slow_dataset_with_session_callable(self):
    # [FUNCTION DOC]
    # Benchmarks the slow dataset fetched one element at a time through a prebuilt session callable.
    dataset = self.setup_slow_dataset()
    self.run_benchmark_with_session_run(dataset, make_callable=True)

  def benchmark_slow_dataset_in_eager(self):
    # [FUNCTION DOC]
    # Within an eager mode scope, benchmarks the slow dataset iterated with a Python iterator.
    with context.eager_mode():
      dataset = self.setup_slow_dataset()
      self.run_benchmark_in_eager(dataset)

  def report(self, deltas):
    # [FUNCTION DOC]
    # HELPER: shared reporting for the benchmarks in this class.
    # Takes a list of total elapsed seconds, one per repetition (each covering iters iterations).
    # Converts to per-iteration times, discards the first five as warm-up, computes median, mean,
    # min and max, and reports the median as wall time with extras holding iterations per second,
    # those statistics and the number of retained repetitions. Returns nothing.
    # Each `delta` is the time taken for `self.iters` iterations. Divide by the
    # number of iterations here to get per-element iteration time.
    deltas = np.array(deltas) / self.iters
    # Discard the first 5 results from "warming up" the session.
    deltas = deltas[5:]

    median = np.median(deltas)
    mean = np.mean(deltas)
    min_val = np.min(deltas)
    max_val = np.max(deltas)
    extras = {
        "iters_per_second": 1 / median,
        "median": median,
        "mean": mean,
        "min": min_val,
        "max": max_val,
        "num_reps": self.num_reps - 5,
    }
    self.report_benchmark(wall_time=median, iters=self.iters, extras=extras)

  def run_benchmark_in_eager(self, dataset):
    # [FUNCTION DOC]
    # HELPER: eager measurement. For each repetition makes a new iterator over dataset and times
    # iters calls fetching the next element, then reports the collected timings. Returns nothing.
    deltas = []
    for _ in range(self.num_reps):
      iterator = iter(dataset)
      deltas.append(timeit.timeit(lambda: next(iterator), number=self.iters))  # pylint: disable=cell-var-from-loop

    self.report(deltas)

  def run_benchmark_with_session_run(self, dataset, make_callable=False):
    # [FUNCTION DOC]
    # HELPER: graph measurement. Creates an initializable iterator over dataset and, in one
    # session, for each repetition initializes it and times iters fetches of the next element
    # op, either through a session callable (when make_callable is true, default false) or plain
    # session run calls on the op (avoiding copying the tensor). Reports the timings. Returns
    # nothing.
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()

    with session.Session() as sess:
      deltas = []
      for _ in range(self.num_reps):
        if make_callable:
          get_next_element = sess.make_callable(next_element)
        else:
          # Note: session.run(next_element.op) is more performant than
          # session.run(next_element) because we avoid the cost of copying the
          # tensor from C++ to python.
          get_next_element = lambda: sess.run(next_element.op)

        sess.run(iterator.initializer)
        deltas.append(timeit.timeit(get_next_element, number=self.iters))
    self.report(deltas)

  def run_benchmark_with_only_cpp_iterations(self, dataset):
    # [FUNCTION DOC]
    # HELPER: graph measurement with native iteration. Skips iters minus one elements of dataset so
    # the iteration happens in native code, then for each repetition initializes the iterator and
    # times a single session run fetching one element. Reports the timings. Returns nothing.
    """Benchmarks the dataset with the iterations performed in C++."""
    # NOTE: We use `dataset.skip()` to perform the iterations in C++, avoiding
    # the overhead of multiple `session.run()` calls. Note that this relies on
    # the underlying implementation of `skip`: if it is optimized in the future,
    # we will have to change this code.
    dataset = dataset.skip(self.iters - 1)
    iterator = dataset_ops.make_initializable_iterator(dataset)
    next_element = iterator.get_next()

    with session.Session() as sess:
      deltas = []
      for _ in range(self.num_reps):
        sess.run(iterator.initializer)
        deltas.append(
            timeit.timeit(lambda: sess.run(next_element.op), number=1))
    self.report(deltas)


if __name__ == "__main__":
  test.main()
