# [MODULE DOC]
# Tests for the benchmark framework in the platform benchmark module. Defines two sample
# benchmark classes (one for checking registry and selective running, one for checking report
# output including an op benchmark) and a test case that verifies registration, name-based
# running, and the serialized benchmark entries written when the reporter environment variable
# is set. Import-time side effect: defining the benchmark classes registers them in the global
# benchmark registry; module-level flags record which benchmarks ran.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.framework.importer."""
import json
import os
import random

import numpy as np

from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test

# Used by SomeRandomBenchmark class below.
_ran_somebenchmark_1 = [False]
_ran_somebenchmark_2 = [False]
_ran_somebenchmark_but_shouldnt = [False]


class SomeRandomBenchmark(test.Benchmark):
  # [CLASS DOC]
  # HELPER: sample benchmark fixture for the registry and selective-run tests.
  # Benchmark subclass with two real benchmark methods and two methods that must never be run
  # (one underscore-prefixed, one not starting with the benchmark prefix). Each method sets a
  # module-level flag so tests can see which ones ran. Defining it registers it in the registry.
  """This Benchmark should automatically be registered in the registry."""

  def _dontRunThisBenchmark(self):
    # [FUNCTION DOC]
    # HELPER: decoy method; the runner must skip it because of its underscore prefix.
    # Sets the "ran but shouldn't" flag if invoked. Takes no inputs, returns nothing.
    _ran_somebenchmark_but_shouldnt[0] = True

  def notBenchmarkMethod(self):
    # [FUNCTION DOC]
    # HELPER: decoy method; the runner must skip it because its name lacks the benchmark prefix.
    # Sets the "ran but shouldn't" flag if invoked. Takes no inputs, returns nothing.
    _ran_somebenchmark_but_shouldnt[0] = True

  def benchmark1(self):
    # [FUNCTION DOC]
    # Sample benchmark: sets the first "ran" flag so tests can observe it was executed.
    # Takes no inputs, returns nothing.
    _ran_somebenchmark_1[0] = True

  def benchmark2(self):
    # [FUNCTION DOC]
    # Sample benchmark: sets the second "ran" flag so tests can observe it was executed.
    # Takes no inputs, returns nothing.
    _ran_somebenchmark_2[0] = True


class TestReportingBenchmark(test.Benchmark):
  # [CLASS DOC]
  # HELPER: sample benchmark fixture for the report-writing test.
  # Benchmark subclass whose methods emit reports: one with only an iteration count, one with a
  # custom name and extra values, and one that benchmarks a simple addition op in a session.
  # Registered automatically in the benchmark registry when defined.
  """This benchmark (maybe) reports some stuff."""

  def benchmarkReport1(self):
    # [FUNCTION DOC]
    # Reports a benchmark with an iteration count of one and default name. Writes a report file
    # only if the reporter environment variable is set. Returns nothing.
    self.report_benchmark(iters=1)

  def benchmarkReport2(self):
    # [FUNCTION DOC]
    # Reports a benchmark with two iterations, the custom name "custom_benchmark_name", and extras
    # holding one numeric and one string value. Writes a report file only if the reporter
    # environment variable is set. Returns nothing.
    self.report_benchmark(
        iters=2,
        name="custom_benchmark_name",
        extras={"number_key": 3,
                "other_key": "string"})

  def benchmark_times_an_op(self):
    # [FUNCTION DOC]
    # Benchmarks adding a five-element float placeholder to itself in a new session configured
    # with the benchmark config, using at least 1000 iterations, storing a trace, under the name
    # "op_benchmark". Feeds a range of numbers. Returns the benchmark result mapping (including
    # the chrome-format trace in its extras); also writes a report if the reporter variable is set.
    input_size = 5
    with session.Session(config=benchmark.benchmark_config()) as sess:
      a = array_ops.placeholder(dtype=dtypes.float32, shape=(input_size))
      a_plus_a = a + a
      return self.run_op_benchmark(
          sess,
          a_plus_a,
          feed_dict={a: np.arange(input_size)},
          min_iters=1000,
          store_trace=True,
          name="op_benchmark")


class BenchmarkTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the benchmark framework: the global registry contents, running benchmarks
  # selected by name pattern, and the report files written when the reporter environment variable
  # is set. Uses the sample benchmark classes defined in this module as fixtures.

  def testGlobalBenchmarkRegistry(self):
    # [FUNCTION DOC]
    # Scenario: inspect the global benchmark registry. Expects exactly two registered benchmarks,
    # the sample random benchmark class and the reporting benchmark class.
    registry = list(benchmark.GLOBAL_BENCHMARK_REGISTRY)
    self.assertEqual(len(registry), 2)
    self.assertTrue(SomeRandomBenchmark in registry)
    self.assertTrue(TestReportingBenchmark in registry)

  def testRunSomeRandomBenchmark(self):
    # [FUNCTION DOC]
    # Scenario: running benchmarks by name pattern. Checks no flag is set initially; running an
    # unrelated pattern raises a value error and runs nothing; running the pattern for the random
    # benchmark (with the reporter variable removed) runs both real benchmarks but not the decoys;
    # after resetting flags, a pattern ending in "1" runs only the first benchmark.
    # Side effect: removes the reporter environment variable if present.
    # Validate that SomeBenchmark has not run yet
    self.assertFalse(_ran_somebenchmark_1[0])
    self.assertFalse(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

    # Run other benchmarks, but this wont run the one we care about
    with self.assertRaises(ValueError):
      benchmark._run_benchmarks("unrelated")

    # Validate that SomeBenchmark has not run yet
    self.assertFalse(_ran_somebenchmark_1[0])
    self.assertFalse(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

    # Run all the benchmarks, avoid generating any reports
    if benchmark.TEST_REPORTER_TEST_ENV in os.environ:
      del os.environ[benchmark.TEST_REPORTER_TEST_ENV]
    benchmark._run_benchmarks("SomeRandom")

    # Validate that SomeRandomBenchmark ran correctly
    self.assertTrue(_ran_somebenchmark_1[0])
    self.assertTrue(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

    _ran_somebenchmark_1[0] = False
    _ran_somebenchmark_2[0] = False
    _ran_somebenchmark_but_shouldnt[0] = False

    # Test running a specific method of SomeRandomBenchmark
    if benchmark.TEST_REPORTER_TEST_ENV in os.environ:
      del os.environ[benchmark.TEST_REPORTER_TEST_ENV]
    benchmark._run_benchmarks("SomeRandom.*1$")

    self.assertTrue(_ran_somebenchmark_1[0])
    self.assertFalse(_ran_somebenchmark_2[0])
    self.assertFalse(_ran_somebenchmark_but_shouldnt[0])

  def testReportingBenchmark(self):
    # [FUNCTION DOC]
    # Scenario: benchmark reports written to files. Creates the temp directory and a random file
    # prefix. Without the reporter environment variable, running a benchmark writes nothing. With
    # it set to the prefix, runs the three reporting benchmarks and checks three report files
    # exist. Reads each back (nested helper) and compares: the first two to expected entries
    # (name, iterations, extras); the third by name, iteration count of 1000, positive wall time,
    # a trace in chrome format containing trace events, and positive allocator peak byte extras.
    # Cleans up by recursively deleting the temp directory. Modifies the process environment.
    tempdir = test.get_temp_dir()
    try:
      gfile.MakeDirs(tempdir)
    except OSError as e:
      # It's OK if the directory already exists.
      if " exists:" not in str(e):
        raise e

    prefix = os.path.join(tempdir,
                          "reporting_bench_%016x_" % random.getrandbits(64))
    expected_output_file = "%s%s" % (prefix,
                                     "TestReportingBenchmark.benchmarkReport1")
    expected_output_file_2 = "%s%s" % (
        prefix, "TestReportingBenchmark.custom_benchmark_name")
    expected_output_file_3 = "%s%s" % (prefix,
                                       "TestReportingBenchmark.op_benchmark")
    try:
      self.assertFalse(gfile.Exists(expected_output_file))
      # Run benchmark but without env, shouldn't write anything
      if benchmark.TEST_REPORTER_TEST_ENV in os.environ:
        del os.environ[benchmark.TEST_REPORTER_TEST_ENV]
      reporting = TestReportingBenchmark()
      reporting.benchmarkReport1()  # This should run without writing anything
      self.assertFalse(gfile.Exists(expected_output_file))

      # Runbenchmark with env, should write
      os.environ[benchmark.TEST_REPORTER_TEST_ENV] = prefix

      reporting = TestReportingBenchmark()
      reporting.benchmarkReport1()  # This should write
      reporting.benchmarkReport2()  # This should write
      benchmark_values3 = reporting.benchmark_times_an_op()  # This should write

      # Check the files were written
      self.assertTrue(gfile.Exists(expected_output_file))
      self.assertTrue(gfile.Exists(expected_output_file_2))
      self.assertTrue(gfile.Exists(expected_output_file_3))

      # Check the contents are correct
      expected_1 = test_log_pb2.BenchmarkEntry()
      expected_1.name = "TestReportingBenchmark.benchmarkReport1"
      expected_1.iters = 1

      expected_2 = test_log_pb2.BenchmarkEntry()
      expected_2.name = "TestReportingBenchmark.custom_benchmark_name"
      expected_2.iters = 2
      expected_2.extras["number_key"].double_value = 3
      expected_2.extras["other_key"].string_value = "string"

      expected_3 = test_log_pb2.BenchmarkEntry()
      expected_3.name = "TestReportingBenchmark.op_benchmark"
      expected_3.iters = 1000

      def read_benchmark_entry(f):
        # [FUNCTION DOC]
        # HELPER: report reader for testReportingBenchmark.
        # Input: path of a report file. Reads it, parses a serialized entries message, asserts it
        # holds exactly one entry, and returns that entry.
        s = gfile.GFile(f, "rb").read()
        entries = test_log_pb2.BenchmarkEntries.FromString(s)
        self.assertEqual(1, len(entries.entry))
        return entries.entry[0]

      read_benchmark_1 = read_benchmark_entry(expected_output_file)
      self.assertProtoEquals(expected_1, read_benchmark_1)

      read_benchmark_2 = read_benchmark_entry(expected_output_file_2)
      self.assertProtoEquals(expected_2, read_benchmark_2)

      read_benchmark_3 = read_benchmark_entry(expected_output_file_3)
      self.assertEqual(expected_3.name, read_benchmark_3.name)
      self.assertEqual(expected_3.iters, read_benchmark_3.iters)
      self.assertGreater(read_benchmark_3.wall_time, 0)

      # Trace is not stored in benchmark entry. Instead we get it from
      # return value of `run_op_benchmark` call.
      full_trace = benchmark_values3["extras"]["full_trace_chrome_format"]
      json_trace = json.loads(full_trace)

      self.assertTrue(isinstance(json_trace, dict))
      self.assertTrue("traceEvents" in json_trace.keys())
      allocator_keys = [k for k in read_benchmark_3.extras.keys()
                        if k.startswith("allocator_maximum_num_bytes_")]
      self.assertGreater(len(allocator_keys), 0)
      for k in allocator_keys:
        self.assertGreater(read_benchmark_3.extras[k].double_value, 0)

    finally:
      gfile.DeleteRecursively(tempdir)


if __name__ == "__main__":
  test.main()
