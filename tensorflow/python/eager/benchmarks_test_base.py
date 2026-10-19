# [MODULE DOC]
# HELPER: helper module providing the shared base class for eager micro-benchmarks.
# Defines MicroBenchmarksBase, a benchmark base class that times a callable, reports wall time and
# throughput, and optionally re-runs the callable under the profiler with and without Python
# tracing. Import-time side effects: defines two command-line flags, a boolean flag to enable
# profiler runs (default off) and a string flag giving the directory where profiler data is stored
# (default a temp directory). Used by the eager benchmark module in this package.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Benchmark base to run and report benchmark results."""

import os
import uuid

from tensorflow.python.eager import test
from tensorflow.python.platform import flags
from tensorflow.python.profiler import profiler_v2 as profiler

flags.DEFINE_bool("xprof", False, "Run and report benchmarks with xprof on")
flags.DEFINE_string("logdir", "/tmp/xprof/", "Directory to store xprof data")


class MicroBenchmarksBase(test.Benchmark):
  # [CLASS DOC]
  # HELPER: base class for the eager micro-benchmark classes.
  # Derives from the framework benchmark class. Provides a method that times a callable and reports
  # results, and a method that repeats the measurement under the profiler. Subclasses must provide a
  # method returning the benchmark name, which the reporting method calls.
  """Run and report benchmark results.

  The first run is without any profilng.
  Second run is with xprof and python trace. Third run is with xprof without
  python trace. Note: xprof runs are with fewer iterations.
  """

  def run_with_xprof(self, enable_python_trace, run_benchmark, func,
                     num_iters_xprof, execution_mode, suid):
    # [FUNCTION DOC]
    # HELPER: profiler-enabled measurement used by run_report.
    # Runs the timing function under a profiler session. Inputs: enable_python_trace (true to use
    # Python tracing level 1, false for level 0), run_benchmark (timing function taking a callable,
    # iteration count and execution mode and returning total seconds), func (callable to time),
    # num_iters_xprof (iterations), execution_mode, and suid (unique id text). The profiler output
    # directory is the log directory flag joined with suid, with a suffix for Python tracing runs.
    # Returns a pair: the directory path and the microseconds per iteration rounded to three
    # decimals. Side effect: writes profiler data to disk.
    if enable_python_trace:
      options = profiler.ProfilerOptions(python_tracer_level=1)
      logdir = os.path.join(flags.FLAGS.logdir, suid + "_with_python")
    else:
      options = profiler.ProfilerOptions(python_tracer_level=0)
      logdir = os.path.join(flags.FLAGS.logdir, suid)
    with profiler.Profile(logdir, options):
      total_time = run_benchmark(func, num_iters_xprof, execution_mode)
    us_per_example = float("{0:.3f}".format(total_time * 1e6 / num_iters_xprof))
    return logdir, us_per_example

  def run_report(self, run_benchmark, func, num_iters, execution_mode=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the reported wall time is in microseconds per iteration, whereas the framework
    #   typically expects seconds.
    # Times a callable and reports the result through the benchmark framework. Inputs: run_benchmark
    # (timing function), func (callable), num_iters, optional execution_mode. Computes mean
    # microseconds per iteration and examples per second, and records them as extra values. If the
    # profiler flag is on, it generates a unique id and re-runs the measurement twice with at most
    # 100 iterations, first with Python tracing and then without, recording the profiler directories
    # and per-iteration times in the extras. Finally asks the subclass for the benchmark name and
    # reports the benchmark with iteration count, wall time per iteration and the extras. Returns
    # nothing.
    """Run and report benchmark results."""
    total_time = run_benchmark(func, num_iters, execution_mode)
    mean_us = total_time * 1e6 / num_iters
    extras = {
        "examples_per_sec": float("{0:.3f}".format(num_iters / total_time)),
        "us_per_example": float("{0:.3f}".format(total_time * 1e6 / num_iters))
    }

    if flags.FLAGS.xprof:
      suid = str(uuid.uuid4())
      # Re-run with xprof and python trace.
      num_iters_xprof = min(100, num_iters)
      xprof_link, us_per_example = self.run_with_xprof(True, run_benchmark,
                                                       func, num_iters_xprof,
                                                       execution_mode, suid)
      extras["xprof link with python trace"] = xprof_link
      extras["us_per_example with xprof and python"] = us_per_example

      # Re-run with xprof but no python trace.
      xprof_link, us_per_example = self.run_with_xprof(False, run_benchmark,
                                                       func, num_iters_xprof,
                                                       execution_mode, suid)
      extras["xprof link"] = xprof_link
      extras["us_per_example with xprof"] = us_per_example

    benchmark_name = self._get_benchmark_name()
    self.report_benchmark(
        iters=num_iters, wall_time=mean_us, extras=extras, name=benchmark_name)
