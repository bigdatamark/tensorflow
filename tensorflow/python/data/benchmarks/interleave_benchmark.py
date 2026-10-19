# [MODULE DOC]
# Benchmarks comparing non-parallel, experimental parallel and core parallel interleave using fake
# datasets with configurable delays. Defines constants naming the interleave versions and a helper
# that builds the fake dataset factory.
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
"""Benchmarks for `tf.data.Dataset.interleave()`."""

from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.ops import dataset_ops

NON_PARALLEL = "non_parallel"
EXPERIMENTAL_PARALLEL = "experimental_parallel"
CORE_PARALLEL = "core_parallel"


def _make_fake_dataset_fn(initial_delay_us, remainder_delay_us):
  # [FUNCTION DOC]
  # HELPER: builds input functions for the interleave benchmarks.
  # Returns a factory function (taking one ignored argument) that creates a hundred-element range
  # dataset emulating remote storage. Delays are in microseconds; a zero initial delay means no
  # special first-element delay, and zero delays add no sleeping. Otherwise an empty dataset with
  # the initial delay is concatenated before the 100-element dataset with the remainder delay.
  # KNOWN ISSUE: the docstring says it returns a dataset, but it returns a factory function.
  """Returns a dataset that emulates a remote storage data source.

  Returns a dataset factory which creates a dataset with 100 elements that
  emulates the performance characteristic of a file-based dataset stored in a
  remote storage. In particular, the first element will take an order of
  magnitude longer to produce than the remaining elements (100ms vs. 1ms).

  Args:
    initial_delay_us: How long to wait before producing the first element.
    remainder_delay_us: How long to wait before producing subsequent elements.
  """

  def fake_dataset_fn(unused):
    # [FUNCTION DOC]
    # HELPER: dataset factory returned by the outer function. Ignores its argument and returns the
    # fake delayed dataset described by the enclosing delays.
    """Returns a function that creates a dataset with the specified delays."""
    del unused

    def make_dataset(time_us, num_elements):
      # [FUNCTION DOC]
      # HELPER: builds a range dataset of num_elements and, if time_us (microseconds) is positive,
      # adds a sleep of that duration per element. Returns the dataset.
      dataset = dataset_ops.Dataset.range(num_elements)
      if time_us > 0:
        dataset = dataset.apply(testing.sleep(time_us))
      return dataset

    if not initial_delay_us:
      return make_dataset(remainder_delay_us, 100)

    return make_dataset(initial_delay_us,
                        0).concatenate(make_dataset(remainder_delay_us, 100))

  return fake_dataset_fn


class ParallelInterleaveBenchmark(benchmark_base.DatasetBenchmarkBase):
  # [CLASS DOC]
  # Benchmark suite comparing non-parallel, experimental parallel and core parallel interleave
  # across remote-file simulation, fast input, single cycle, single parallel call and long cycle.
  """Benchmarks for `tf.data.experimental.parallel_interleave()`."""

  def apply_interleave(self, interleave_version, dataset, interleave_fn,
                       cycle_length, num_parallel_calls):
    # [FUNCTION DOC]
    # HELPER: applies the selected interleave variant to dataset. interleave_version is one of
    # the three version name constants; interleave_fn maps an element to a dataset; cycle_length
    # is the cycle size. For core parallel, a missing or zero num_parallel_calls defaults to the
    # cycle length; it is ignored by the other versions. Returns the new dataset. Raises a value
    # error for an unknown version.
    # KNOWN ISSUE: the error message concatenates the version with a string, so a non-string
    # version would raise a type error instead.
    if interleave_version == NON_PARALLEL:
      return dataset.interleave(interleave_fn, cycle_length=cycle_length)
    elif interleave_version == EXPERIMENTAL_PARALLEL:
      return dataset.apply(
          interleave_ops.parallel_interleave(
              interleave_fn, cycle_length=cycle_length))
    elif interleave_version == CORE_PARALLEL:
      if not num_parallel_calls:
        num_parallel_calls = cycle_length
      return dataset.interleave(
          interleave_fn,
          cycle_length=cycle_length,
          num_parallel_calls=num_parallel_calls)
    else:
      raise ValueError("Unknown version: " + interleave_version)

  def make_dataset(self,
                   interleave_version,
                   initial_delay,
                   remainder_delay,
                   cycle_length,
                   num_parallel_calls=None):
    # [FUNCTION DOC]
    # HELPER: builds the benchmark input. Starts from an endlessly repeated single element, builds
    # the fake delayed dataset factory from the two delays (microseconds), and returns the result
    # of applying the chosen interleave version with cycle_length and optional num_parallel_calls.
    dataset = dataset_ops.Dataset.range(1).repeat()
    interleave_fn = _make_fake_dataset_fn(initial_delay, remainder_delay)
    return self.apply_interleave(
        interleave_version=interleave_version,
        dataset=dataset,
        interleave_fn=interleave_fn,
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls)

  def _benchmark(self,
                 interleave_version,
                 num_elements,
                 benchmark_id,
                 benchmark_label,
                 initial_delay_us=0,
                 remainder_delay_us=0,
                 cycle_length=10,
                 iters=100,
                 num_parallel_calls=None,
                 name=None):
    # [FUNCTION DOC]
    # HELPER: shared body of the interleave benchmarks. Builds the dataset for interleave_version
    # with the given delays (default zero), cycle_length (default 10) and num_parallel_calls, then
    # runs and reports it for num_elements elements and iters iterations (default 100) with warm-up.
    # benchmark_label and benchmark_id form the model name; name is the benchmark name. Returns
    # nothing.
    dataset = self.make_dataset(
        interleave_version=interleave_version,
        initial_delay=initial_delay_us,
        remainder_delay=remainder_delay_us,
        cycle_length=cycle_length,
        num_parallel_calls=num_parallel_calls)

    self.run_and_report_benchmark(
        dataset=dataset,
        num_elements=num_elements,
        iters=iters,
        warmup=True,
        extras={
            "model_name":
                "interleave.benchmark.%s.%d" % (benchmark_label, benchmark_id),
            "parameters":
                "%d.%d.%d.%s" %
                (num_elements, cycle_length, iters, str(num_parallel_calls)),
        },
        name=name)

  def benchmark_remote_file_simulation(self):
    # [FUNCTION DOC]
    # Benchmarks the experimental and core parallel versions with a 100 ms first-element delay and
    # a 1 ms per-element delay, for 5000 elements.
    for i, version in enumerate([EXPERIMENTAL_PARALLEL, CORE_PARALLEL]):
      self._benchmark(
          interleave_version=version,
          initial_delay_us=100 * 1000,
          remainder_delay_us=1000,
          num_elements=5000,
          name="remote_file_simulation_" + version,
          benchmark_id=i,
          benchmark_label="remote_file")

  def benchmark_fast_input(self):
    # [FUNCTION DOC]
    # Benchmarks the experimental and core parallel versions with no delays, for 200,000 elements.
    for i, version in enumerate([EXPERIMENTAL_PARALLEL, CORE_PARALLEL]):
      self._benchmark(
          interleave_version=version,
          num_elements=200000,
          name="fast_input_" + version,
          benchmark_id=i,
          benchmark_label="fast_input")

  # Measure the overhead of parallel interleaves compared to non-parallel
  # interleave.
  def benchmark_single_cycle(self):
    # [FUNCTION DOC]
    # Benchmarks all three versions with a cycle length of one and 200,000 elements, to measure
    # the overhead of parallel interleave relative to non-parallel interleave.
    for i, version in enumerate(
        [NON_PARALLEL, EXPERIMENTAL_PARALLEL, CORE_PARALLEL]):
      self._benchmark(
          interleave_version=version,
          cycle_length=1,
          num_elements=200000,
          name="single_cycle_" + version,
          benchmark_id=i,
          benchmark_label="single_cycle")

  # Compare with a more reasonable cycle length. Experimental interleave
  # cannot be compared here because it sets num_parallel_calls = cycle_length.
  def benchmark_single_parallel_call(self):
    # [FUNCTION DOC]
    # Benchmarks only the core parallel version with one parallel call and the default cycle
    # length for 200,000 elements.
    self._benchmark(
        interleave_version=CORE_PARALLEL,
        num_elements=200000,
        num_parallel_calls=1,
        name="single_parallel_call_" + CORE_PARALLEL,
        benchmark_id=1,
        benchmark_label="single_parallel_call")

  def benchmark_long_cycle(self):
    # [FUNCTION DOC]
    # Benchmarks the experimental and core parallel versions with a cycle length of 1000 and
    # 100,000 elements.
    for i, version in enumerate([EXPERIMENTAL_PARALLEL, CORE_PARALLEL]):
      self._benchmark(
          interleave_version=version,
          cycle_length=1000,
          num_elements=100000,
          name="long_cycle_" + version,
          benchmark_id=i,
          benchmark_label="long_cycle")


if __name__ == "__main__":
  benchmark_base.test.main()
