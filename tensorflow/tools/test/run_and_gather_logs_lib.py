# [MODULE DOC]
# HELPER: library behind the run_and_gather_logs command-line tool.
# Runs a bazel test executable as a subprocess with an environment variable pointing at a log
# file prefix, then collects the benchmark entry files the test wrote, and packages them with git
# commit, GPU and machine configuration into a test results record. Exposes a missing-logs
# exception, get_git_commit_sha, process_test_logs, process_benchmarks and run_and_gather_logs.
# No import-time side effects.
# KNOWN ISSUE: the module docstring ("getting system information") is copied from another module
# and does not describe this one.
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
"""Library for getting system information during TensorFlow tests."""

import os
import re
import shlex
import subprocess
import tempfile
import time

from tensorflow.core.util import test_log_pb2
from tensorflow.python.platform import gfile
from tensorflow.tools.test import gpu_info_lib
from tensorflow.tools.test import system_info_lib


class MissingLogsError(Exception):
  # [CLASS DOC]
  # Exception raised by run_and_gather_logs when the test ran successfully but produced no
  # benchmark log files. Plain subclass of the base exception class with no extra state.
  pass


def get_git_commit_sha():
  # [FUNCTION DOC]
  # Takes no inputs. Returns the value of the GIT_COMMIT environment variable (set by CI build
  # agents), or nothing/null if it is unset. Only reads the environment.
  """Get git commit SHA for this build.

  Attempt to get the SHA from environment variable GIT_COMMIT, which should
  be available on Jenkins build agents.

  Returns:
    SHA hash of the git commit used for the build, if available
  """

  return os.getenv("GIT_COMMIT")


def process_test_logs(name, test_name, test_args, benchmark_type,
                      start_time, run_time, log_files):
  # [FUNCTION DOC]
  # Builds a test results record. Inputs: name (benchmark identifier), test_name (bazel target),
  # test_args (arguments the target ran with), benchmark_type (name of a benchmark-type enum
  # value, case-insensitive), start_time (epoch seconds), run_time (wall seconds) and log_files
  # (paths of benchmark log files). Sets those fields, the git commit if available, the parsed
  # benchmark entries, the arguments and the machine configuration. Returns the results record.
  # Raises a value error for an unknown benchmark type, and propagates parse errors from
  # process_benchmarks. Side effects: reads files and inspects the machine.
  # KNOWN ISSUE: the docstring says test_args is a string, but run_and_gather_logs passes the
  # already split list, which is what extending the argument list expects.
  """Gather test information and put it in a TestResults proto.

  Args:
    name: Benchmark target identifier.
    test_name: A unique bazel target, e.g. "//path/to:test"
    test_args: A string containing all arguments to run the target with.
    benchmark_type: A string representing the BenchmarkType enum; the
      benchmark type for this target.
    start_time: Test starting time (epoch)
    run_time:   Wall time that the test ran for
    log_files:  Paths to the log files

  Returns:
    A TestResults proto
  """

  results = test_log_pb2.TestResults()
  results.name = name
  results.target = test_name
  results.start_time = start_time
  results.run_time = run_time
  results.benchmark_type = test_log_pb2.TestResults.BenchmarkType.Value(
      benchmark_type.upper())

  # Gather source code information
  git_sha = get_git_commit_sha()
  if git_sha:
    results.commit_id.hash = git_sha

  results.entries.CopyFrom(process_benchmarks(log_files))
  results.run_configuration.argument.extend(test_args)
  results.machine_configuration.CopyFrom(
      system_info_lib.gather_machine_configuration())
  return results


def process_benchmarks(log_files):
  # [FUNCTION DOC]
  # HELPER: merges benchmark log files for process_test_logs.
  # Input: list of paths to binary serialized benchmark-entries files. Reads each file and merges
  # it into one benchmark-entries message. Returns the merged message. Raises a generic exception
  # naming the file if the content cannot be fully parsed.
  benchmarks = test_log_pb2.BenchmarkEntries()
  for f in log_files:
    content = gfile.GFile(f, "rb").read()
    if benchmarks.MergeFromString(content) != len(content):
      raise Exception("Failed parsing benchmark entry from %s" % f)
  return benchmarks


def run_and_gather_logs(name,
                        test_name,
                        test_args,
                        benchmark_type,
                        skip_processing_logs=False):
  # [FUNCTION DOC]
  # Runs a bazel test binary and gathers its benchmark logs. Inputs: name (benchmark identifier),
  # test_name (unique bazel target such as a path plus colon plus target), test_args (one string
  # of arguments, split shell-style), benchmark_type (benchmark-type enum name) and optional
  # skip_processing_logs (default false). Validates the target (must start with two slashes, be
  # a single concrete target, no wildcards). Locates the executable under the bazel output
  # directory or the current directory (also trying a python3 suffixed variant). Appends a GPU
  # short name (for a few Tesla/NVIDIA models) to the name, makes a temp directory, sets an
  # environment variable telling the test where to write reports, runs the test and, unless
  # skipping, collects log files and builds the results.
  # Returns a pair: the results record (null if skipped) and the adjusted benchmark name.
  # Raises value error for bad target or missing executable, called-process error if the test
  # fails, and a missing-logs error if no logs were written. Side effects: modifies the process
  # environment, creates and finally deletes the temp directory.
  # KNOWN ISSUE: the docstring calls the second return value "mangled_test_name" in one place and
  # "test_adjusted_name" in another; the code returns the unmangled adjusted name.
  """Run the bazel test given by test_name.  Gather and return the logs.

  Args:
    name: Benchmark target identifier.
    test_name: A unique bazel target, e.g. "//path/to:test"
    test_args: A string containing all arguments to run the target with.
    benchmark_type: A string representing the BenchmarkType enum; the
      benchmark type for this target.
    skip_processing_logs: Whether to skip processing test results from log
      files.

  Returns:
    A tuple (test_results, mangled_test_name), where
    test_results: A test_log_pb2.TestResults proto, or None if log processing
      is skipped.
    test_adjusted_name: Unique benchmark name that consists of
      benchmark name optionally followed by GPU type.

  Raises:
    ValueError: If the test_name is not a valid target.
    subprocess.CalledProcessError: If the target itself fails.
    IOError: If there are problems gathering test log output from the test.
    MissingLogsError: If we couldn't find benchmark logs.
  """
  if not (test_name and test_name.startswith("//") and ".." not in test_name and
          not test_name.endswith(":") and not test_name.endswith(":all") and
          not test_name.endswith("...") and len(test_name.split(":")) == 2):
    raise ValueError("Expected test_name parameter with a unique test, e.g.: "
                     "--test_name=//path/to:test")
  test_executable = test_name.rstrip().strip("/").replace(":", "/")

  if gfile.Exists(os.path.join("bazel-bin", test_executable)):
    # Running in standalone mode from core of the repository
    test_executable = os.path.join("bazel-bin", test_executable)
  else:
    # Hopefully running in sandboxed mode
    test_executable = os.path.join(".", test_executable)

  test_adjusted_name = name
  gpu_config = gpu_info_lib.gather_gpu_devices()
  if gpu_config:
    gpu_name = gpu_config[0].model
    gpu_short_name_match = re.search(
        r"(Tesla|NVIDIA) (K40|K80|P100|V100|A100)", gpu_name
    )
    if gpu_short_name_match:
      gpu_short_name = gpu_short_name_match.group(0)
      test_adjusted_name = name + "|" + gpu_short_name.replace(" ", "_")

  temp_directory = tempfile.mkdtemp(prefix="run_and_gather_logs")
  mangled_test_name = (
      test_adjusted_name.strip("/").replace("|",
                                            "_").replace("/",
                                                         "_").replace(":", "_"))
  test_file_prefix = os.path.join(temp_directory, mangled_test_name)
  test_file_prefix = "%s." % test_file_prefix

  try:
    if not gfile.Exists(test_executable):
      test_executable_py3 = test_executable + ".python3"
      if not gfile.Exists(test_executable_py3):
        raise ValueError("Executable does not exist: %s" % test_executable)
      test_executable = test_executable_py3
    test_args = shlex.split(test_args)

    # This key is defined in tf/core/util/reporter.h as
    # TestReporter::kTestReporterEnv.
    os.environ["TEST_REPORT_FILE_PREFIX"] = test_file_prefix
    start_time = time.time()
    subprocess.check_call([test_executable] + test_args)
    if skip_processing_logs:
      return None, test_adjusted_name
    run_time = time.time() - start_time
    log_files = gfile.Glob("{}*".format(test_file_prefix))
    if not log_files:
      raise MissingLogsError("No log files found at %s." % test_file_prefix)

    return (process_test_logs(
        test_adjusted_name,
        test_name=test_name,
        test_args=test_args,
        benchmark_type=benchmark_type,
        start_time=int(start_time),
        run_time=run_time,
        log_files=log_files), test_adjusted_name)

  finally:
    try:
      gfile.DeleteRecursively(temp_directory)
    except OSError:
      pass
