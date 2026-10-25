# [MODULE DOC]
# Benchmark infrastructure for TensorFlow. Defines the abstract Benchmark base class (and its
# TensorFlow-specific subclass exported as the public test benchmark class), a metaclass that
# registers every concrete subclass in a global registry, a metaclass for generating
# parameterized benchmarks, a function that writes benchmark results (either printing them or
# serializing them to a file named by an environment variable), a ConfigProto factory, and the
# entry point that runs registered benchmarks selected by a regular expression from the command
# line. Import-time side effects: creates the global benchmark registry (an empty set) and
# registers a few public symbols through the export decorator. Not a helper module; it is the
# public benchmarking API surfaced by the test package.

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

"""Utilities to run benchmarks."""
import math
import numbers
import os
import re
import sys
import time
import types

from absl import app

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.util import test_log_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


# When a subclass of the Benchmark class is created, it is added to
# the registry automatically
GLOBAL_BENCHMARK_REGISTRY = set()

# Environment variable that determines whether benchmarks are written.
# See also tensorflow/core/util/reporter.h TestReporter::kTestReporterEnv.
TEST_REPORTER_TEST_ENV = "TEST_REPORT_FILE_PREFIX"

# Environment variable that lets the TensorFlow runtime allocate a new
# threadpool for each benchmark.
OVERRIDE_GLOBAL_THREADPOOL = "TF_OVERRIDE_GLOBAL_THREADPOOL"


def _rename_function(f, arg_num, name):
  # [FUNCTION DOC]
  # HELPER: used by the parameterized benchmark metaclass so generated methods show real names.
  # Builds a new function object from an existing function by replacing the code object's
  # argument count and name. Inputs: f is the function to copy, arg_num is the new positional
  # argument count to record, name is the new function name. Returns the new function, which
  # shares the original's globals, defaults and closure. No side effects on f.
  """Rename the given function's name appears in the stack trace."""
  func_code = f.__code__
  new_code = func_code.replace(co_argcount=arg_num, co_name=name)
  return types.FunctionType(new_code, f.__globals__, name, f.__defaults__,
                            f.__closure__)


def _global_report_benchmark(
    name, iters=None, cpu_time=None, wall_time=None,
    throughput=None, extras=None, metrics=None):
  # [FUNCTION DOC]
  # HELPER: internal reporting backend used by the Benchmark report method.
  # KNOWN ISSUE: the docstring says IOError is the only extra exception, but a TypeError is also
  # raised when metrics is not a list or a metric lacks a name or value field; and the logging
  # call formats iters with an integer directive so a non-integer iters would fail formatting.
  # Logs the benchmark, then builds a benchmark-entries protocol buffer with one entry holding
  # the name and whichever of iterations, cpu time, wall time and throughput were given (nothing
  # for those that are null). Extras (a mapping) are stored as numbers when numeric, otherwise as
  # strings. Metrics (a list of mappings with required name and value, optional min_value and
  # max_value) are appended as metric entries. Raises a type error if extras is not a mapping, if
  # metrics is not a list, or if a metric lacks name or value. If the report-file-prefix
  # environment variable is unset, prints the proto text and returns nothing. Otherwise writes the
  # serialized proto to a file named prefix plus the benchmark name with slashes replaced by
  # double underscores, raising an IO error if that file already exists. Side effects: logging,
  # stdout printing or file creation.
  """Method for recording a benchmark directly.

  Args:
    name: The BenchmarkEntry name.
    iters: (optional) How many iterations were run
    cpu_time: (optional) Total cpu time in seconds
    wall_time: (optional) Total wall time in seconds
    throughput: (optional) Throughput (in MB/s)
    extras: (optional) Dict mapping string keys to additional benchmark info.
    metrics: (optional) A list of dict representing metrics generated by the
      benchmark. Each dict should contain keys 'name' and'value'. A dict
      can optionally contain keys 'min_value' and 'max_value'.

  Raises:
    TypeError: if extras is not a dict.
    IOError: if the benchmark output file already exists.
  """
  logging.info("Benchmark [%s] iters: %d, wall_time: %g, cpu_time: %g,"
               "throughput: %g, extras: %s, metrics: %s", name,
               iters if iters is not None else -1,
               wall_time if wall_time is not None else -1,
               cpu_time if cpu_time is not None else -1,
               throughput if throughput is not None else -1,
               str(extras) if extras else "None",
               str(metrics) if metrics else "None")

  entries = test_log_pb2.BenchmarkEntries()
  entry = entries.entry.add()
  entry.name = name
  if iters is not None:
    entry.iters = iters
  if cpu_time is not None:
    entry.cpu_time = cpu_time
  if wall_time is not None:
    entry.wall_time = wall_time
  if throughput is not None:
    entry.throughput = throughput
  if extras is not None:
    if not isinstance(extras, dict):
      raise TypeError("extras must be a dict")
    for (k, v) in extras.items():
      if isinstance(v, numbers.Number):
        entry.extras[k].double_value = v
      else:
        entry.extras[k].string_value = str(v)
  if metrics is not None:
    if not isinstance(metrics, list):
      raise TypeError("metrics must be a list")
    for metric in metrics:
      if "name" not in metric:
        raise TypeError("metric must has a 'name' field")
      if "value" not in metric:
        raise TypeError("metric must has a 'value' field")

      metric_entry = entry.metrics.add()
      metric_entry.name = metric["name"]
      metric_entry.value = metric["value"]
      if "min_value" in metric:
        metric_entry.min_value.value = metric["min_value"]
      if "max_value" in metric:
        metric_entry.max_value.value = metric["max_value"]

  test_env = os.environ.get(TEST_REPORTER_TEST_ENV, None)
  if test_env is None:
    # Reporting was not requested, just print the proto
    print(str(entries))
    return

  serialized_entry = entries.SerializeToString()

  mangled_name = name.replace("/", "__")
  output_path = "%s%s" % (test_env, mangled_name)
  if gfile.Exists(output_path):
    raise IOError("File already exists: %s" % output_path)
  with gfile.GFile(output_path, "wb") as out:
    out.write(serialized_entry)


class _BenchmarkRegistrar(type):
  # [CLASS DOC]
  # HELPER: metaclass machinery behind the Benchmark base class.
  # Metaclass that registers every newly created non-abstract benchmark class in the module-level
  # global benchmark registry. Holds no state itself; side effect is on the registry at class
  # creation time. Used as the metaclass of Benchmark and extended by the parameterized variant.
  """The Benchmark class registrar.  Used by abstract Benchmark class."""

  def __new__(mcs, clsname, base, attrs):
    # [FUNCTION DOC]
    # Creates the class normally, then adds it to the global benchmark registry unless it is the
    # abstract base (decided by the class's abstractness check). Inputs: metaclass, class name,
    # tuple of base classes, mapping of class attributes. Returns the new class.
    newclass = type.__new__(mcs, clsname, base, attrs)
    if not newclass.is_abstract():
      GLOBAL_BENCHMARK_REGISTRY.add(newclass)
    return newclass


@tf_export("__internal__.test.ParameterizedBenchmark", v1=[])
class ParameterizedBenchmark(_BenchmarkRegistrar):
  # [CLASS DOC]
  # Metaclass that expands each benchmark method (name starting with "benchmark") into one method
  # per entry of the class attribute listing benchmark parameters. Each entry is a sequence whose
  # first item is a case-name suffix and the rest are extra arguments passed to the original
  # method. Inherits the registration behavior of the registrar metaclass. Exported publicly as
  # an internal test symbol. The original methods are removed from the class.
  """Metaclass to generate parameterized benchmarks.

  Use this class as a metaclass and override the `_benchmark_parameters` to
  generate multiple benchmark test cases. For example:

  class FooBenchmark(metaclass=tf.test.ParameterizedBenchmark,
                     tf.test.Benchmark):
    # The `_benchmark_parameters` is expected to be a list with test cases.
    # Each of the test case is a tuple, with the first time to be test case
    # name, followed by any number of the parameters needed for the test case.
    _benchmark_parameters = [
      ('case_1', Foo, 1, 'one'),
      ('case_2', Bar, 2, 'two'),
    ]

    def benchmark_test(self, target_class, int_param, string_param):
      # benchmark test body

  The example above will generate two benchmark test cases:
  "benchmark_test__case_1" and "benchmark_test__case_2".
  """

  def __new__(mcs, clsname, base, attrs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the class docstring example lists the metaclass before the Benchmark base in a
    # way that is not valid syntax; also the check for duplicate names raises a bare generic
    # exception rather than a specific error type.
    # Creates a class whose benchmark methods are expanded per parameter set. Inputs: metaclass,
    # class name, bases, attribute mapping, which must contain the parameter list attribute
    # (missing it raises a key error). For each attribute starting with "benchmark", the original
    # is deleted and, for every parameter entry, a new method named original plus two underscores
    # plus the case suffix is added, wrapping the original with the remaining parameters and
    # renamed so stack inspection finds the new name. Raises a generic exception if a generated
    # name already exists. Returns the class built by the parent metaclass (and thus registered).
    param_config_list = attrs["_benchmark_parameters"]

    def create_benchmark_function(original_benchmark, params):
      # [FUNCTION DOC]
      # HELPER: closure factory so each generated benchmark binds its own parameters.
      # Returns a function taking an instance that calls the original benchmark with the instance
      # followed by the given parameters.
      return lambda self: original_benchmark(self, *params)

    for name in attrs.copy().keys():
      if not name.startswith("benchmark"):
        continue

      original_benchmark = attrs[name]
      del attrs[name]

      for param_config in param_config_list:
        test_name_suffix = param_config[0]
        params = param_config[1:]
        benchmark_name = name + "__" + test_name_suffix
        if benchmark_name in attrs:
          raise Exception(
              "Benchmark named {} already defined.".format(benchmark_name))

        benchmark = create_benchmark_function(original_benchmark, params)
        # Renaming is important because `report_benchmark` function looks up the
        # function name in the stack trace.
        attrs[benchmark_name] = _rename_function(benchmark, 1, benchmark_name)

    return super().__new__(mcs, clsname, base, attrs)


class Benchmark(metaclass=_BenchmarkRegistrar):
  # [CLASS DOC]
  # Abstract base for benchmarks. Its metaclass auto-registers each concrete subclass in the
  # global registry. Subclasses define methods whose names start with "benchmark"; these are
  # invoked by the benchmark runner, and call the report method to emit results. Stateless itself.
  """Abstract class that provides helper functions for running benchmarks.

  Any class subclassing this one is immediately registered in the global
  benchmark registry.

  Only methods whose names start with the word "benchmark" will be run during
  benchmarking.
  """

  @classmethod
  def is_abstract(cls):
    # [FUNCTION DOC]
    # Class method telling the registrar whether the class is the abstract base itself. Returns
    # true when the class's method resolution order has at most two entries (the metaclass-built
    # base and its parent), meaning it is Benchmark; false for any subclass.
    # mro: (_BenchmarkRegistrar, Benchmark) means this is Benchmark
    return len(cls.mro()) <= 2

  def _get_name(self, overwrite_name=None):
    # [FUNCTION DOC]
    # HELPER: computes the benchmark entry name for the report method.
    # Walks the current call stack from outermost to innermost looking for the first frame whose
    # local "self" is a Benchmark instance; that frame's function name is the method name. The
    # result is the class name of that instance, a dot, then overwrite_name if given (optional)
    # else the method name. Raises a value error when no Benchmark instance is on the stack.
    """Returns full name of class and method calling report_benchmark."""

    # Find the caller method (outermost Benchmark class)
    stack = tf_inspect.stack()
    calling_class = None
    name = None
    for frame in stack[::-1]:
      f_locals = frame[0].f_locals
      f_self = f_locals.get("self", None)
      if isinstance(f_self, Benchmark):
        calling_class = f_self  # Get the outermost stack Benchmark call
        name = frame[3]  # Get the method name
        break
    if calling_class is None:
      raise ValueError("Unable to determine calling Benchmark class.")

    # Use the method name, or overwrite_name is provided.
    name = overwrite_name or name
    # Prefix the name with the class name.
    class_name = type(calling_class).__name__
    name = "%s.%s" % (class_name, name)
    return name

  def report_benchmark(
      self,
      iters=None,
      cpu_time=None,
      wall_time=None,
      throughput=None,
      extras=None,
      name=None,
      metrics=None):
    # [FUNCTION DOC]
    # Public method benchmarks call to record a result. Determines the entry name from the call
    # stack (or the optional name override), then forwards iterations, cpu time, wall time,
    # throughput, extras mapping, and metrics list to the module-level reporting function, which
    # prints or writes the result. All parameters are optional. Returns nothing. May raise a
    # value error if no benchmark method is on the stack, and the reporting function's errors.
    """Report a benchmark.

    Args:
      iters: (optional) How many iterations were run
      cpu_time: (optional) Median or mean cpu time in seconds.
      wall_time: (optional) Median or mean wall time in seconds.
      throughput: (optional) Throughput (in MB/s)
      extras: (optional) Dict mapping string keys to additional benchmark info.
        Values may be either floats or values that are convertible to strings.
      name: (optional) Override the BenchmarkEntry name with `name`.
        Otherwise it is inferred from the top-level method name.
      metrics: (optional) A list of dict, where each dict has the keys below
        name (required), string, metric name
        value (required), double, metric value
        min_value (optional), double, minimum acceptable metric value
        max_value (optional), double, maximum acceptable metric value
    """
    name = self._get_name(overwrite_name=name)
    _global_report_benchmark(
        name=name, iters=iters, cpu_time=cpu_time, wall_time=wall_time,
        throughput=throughput, extras=extras, metrics=metrics)


@tf_export("test.benchmark_config")
def benchmark_config():
  # [FUNCTION DOC]
  # Builds and returns a new session configuration protocol buffer with the graph optimizer's
  # dependency optimization turned off, so benchmarked ops are not pruned away. Takes no inputs
  # and has no side effects.
  """Returns a tf.compat.v1.ConfigProto for disabling the dependency optimizer.

    Returns:
      A TensorFlow ConfigProto object.
  """
  config = config_pb2.ConfigProto()
  config.graph_options.rewrite_options.dependency_optimization = (
      rewriter_config_pb2.RewriterConfig.OFF)
  return config


@tf_export("test.Benchmark")
class TensorFlowBenchmark(Benchmark):
  # [CLASS DOC]
  # Public benchmark base class for TensorFlow code. Adds running an op or tensor in a session
  # with timing and reporting, plus a helper to evaluate tensors. Instances set an environment
  # variable so the runtime uses a fresh thread pool per benchmark. Subclasses are registered
  # like any Benchmark subclass; this class itself is treated as abstract.
  """Abstract class that provides helpers for TensorFlow benchmarks."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Side effect: sets the process-wide environment variable asking the runtime to
    # allocate a new thread pool per benchmark to the value one, then calls the parent
    # constructor. No inputs.
    # Allow TensorFlow runtime to allocate a new threadpool with different
    # number of threads for each new benchmark.
    os.environ[OVERRIDE_GLOBAL_THREADPOOL] = "1"
    super().__init__()

  @classmethod
  def is_abstract(cls):
    # [FUNCTION DOC]
    # Class method override: true only when the class's method resolution order has at most three
    # entries, i.e. it is TensorFlowBenchmark itself, so that it is not registered; false for
    # subclasses.
    # mro: (_BenchmarkRegistrar, Benchmark, TensorFlowBenchmark) means
    # this is TensorFlowBenchmark.
    return len(cls.mro()) <= 3

  def run_op_benchmark(self,
                       sess,
                       op_or_tensor,
                       feed_dict=None,
                       burn_iters=2,
                       min_iters=10,
                       store_trace=False,
                       store_memory_usage=True,
                       name=None,
                       extras=None,
                       mbs=0):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the returned mapping may include the key
    # "full_chrome_trace_format", but the code stores the trace under "full_trace_chrome_format".
    # KNOWN ISSUE: throughput is computed as mbs divided by the median time, which raises a
    # zero-division error when the median is exactly zero, and gives a negative value when
    # min_iters is zero (median becomes minus one). The default mbs of zero yields zero throughput.
    # Times an op or tensor in a session. Runs burn_iters untimed warm-up runs, then min_iters
    # timed runs (wall-clock deltas). If store_trace or store_memory_usage is true, performs one
    # extra fully-traced run; the trace is kept (Chrome format string) only when store_trace, and
    # per-allocator maximum byte counts are added to extras when store_memory_usage. The median
    # time is reported via the report method with iterations, wall time, extras, name and
    # throughput. Afterwards the mean and standard deviation of times (minus one for empty or, for
    # deviation, single samples) and the trace are merged into the extras and returned
    # along with the reported values as a mapping. Inputs: sess (session), op_or_tensor, optional
    # feed_dict, burn_iters, min_iters, store_trace, store_memory_usage, optional name, optional
    # extras mapping (mutated in place, since memory info is added to it), mbs megabytes moved.
    """Run an op or tensor in the given session.  Report the results.

    Args:
      sess: `Session` object to use for timing.
      op_or_tensor: `Operation` or `Tensor` to benchmark.
      feed_dict: A `dict` of values to feed for each op iteration (see the
        `feed_dict` parameter of `Session.run`).
      burn_iters: Number of burn-in iterations to run.
      min_iters: Minimum number of iterations to use for timing.
      store_trace: Boolean, whether to run an extra untimed iteration and
        store the trace of iteration in returned extras.
        The trace will be stored as a string in Google Chrome trace format
        in the extras field "full_trace_chrome_format". Note that trace
        will not be stored in test_log_pb2.TestResults proto.
      store_memory_usage: Boolean, whether to run an extra untimed iteration,
        calculate memory usage, and store that in extras fields.
      name: (optional) Override the BenchmarkEntry name with `name`.
        Otherwise it is inferred from the top-level method name.
      extras: (optional) Dict mapping string keys to additional benchmark info.
        Values may be either floats or values that are convertible to strings.
      mbs: (optional) The number of megabytes moved by this op, used to
        calculate the ops throughput.

    Returns:
      A `dict` containing the key-value pairs that were passed to
      `report_benchmark`. If `store_trace` option is used, then
      `full_chrome_trace_format` will be included in return dictionary even
      though it is not passed to `report_benchmark` with `extras`.
    """
    for _ in range(burn_iters):
      sess.run(op_or_tensor, feed_dict=feed_dict)

    deltas = [None] * min_iters

    for i in range(min_iters):
      start_time = time.time()
      sess.run(op_or_tensor, feed_dict=feed_dict)
      end_time = time.time()
      delta = end_time - start_time
      deltas[i] = delta

    extras = extras if extras is not None else {}
    unreported_extras = {}
    if store_trace or store_memory_usage:
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      sess.run(op_or_tensor, feed_dict=feed_dict,
               options=run_options, run_metadata=run_metadata)
      tl = timeline.Timeline(run_metadata.step_stats)

      if store_trace:
        unreported_extras["full_trace_chrome_format"] = (
            tl.generate_chrome_trace_format())

      if store_memory_usage:
        step_stats_analysis = tl.analyze_step_stats(show_memory=True)
        allocator_maximums = step_stats_analysis.allocator_maximums
        for k, v in allocator_maximums.items():
          extras["allocator_maximum_num_bytes_%s" % k] = v.num_bytes

    def _median(x):
      # [FUNCTION DOC]
      # HELPER: statistics for run_op_benchmark.
      # Returns the median of a list of numbers (average of the two middle elements for an even
      # count), or minus one for an empty list.
      if not x:
        return -1
      s = sorted(x)
      l = len(x)
      lm1 = l - 1
      return (s[l//2] + s[lm1//2]) / 2.0

    def _mean_and_stdev(x):
      # [FUNCTION DOC]
      # HELPER: statistics for run_op_benchmark.
      # Returns a pair of the mean and sample standard deviation (n minus one denominator) of a
      # list of numbers. Returns minus one for both when empty, and minus one for the deviation
      # when there is a single element.
      if not x:
        return -1, -1
      l = len(x)
      mean = sum(x) / l
      if l == 1:
        return mean, -1
      variance = sum([(e - mean) * (e - mean) for e in x]) / (l - 1)
      return mean, math.sqrt(variance)

    median_delta = _median(deltas)

    benchmark_values = {
        "iters": min_iters,
        "wall_time": median_delta,
        "extras": extras,
        "name": name,
        "throughput": mbs / median_delta
    }
    self.report_benchmark(**benchmark_values)

    mean_delta, stdev_delta = _mean_and_stdev(deltas)
    unreported_extras["wall_time_mean"] = mean_delta
    unreported_extras["wall_time_stdev"] = stdev_delta
    benchmark_values["extras"].update(unreported_extras)
    return benchmark_values

  def evaluate(self, tensors):
    # [FUNCTION DOC]
    # Evaluates a tensor or nested list or tuple of tensors using the default session if one is
    # set, otherwise a cached session obtained from the test-case mixin (so the subclass must
    # provide a cached-session method, else an attribute error occurs). Returns the numpy values.
    """Evaluates tensors and returns numpy values.

    Args:
      tensors: A Tensor or a nested list/tuple of Tensors.

    Returns:
      tensors numpy values.
    """
    sess = ops.get_default_session() or self.cached_session()
    return sess.run(tensors)


def _run_benchmarks(regex):
  # [FUNCTION DOC]
  # HELPER: internal driver behind the benchmarks main entry point.
  # KNOWN ISSUE: the docstring says the regex is matched against benchmark class names, but it is
  # matched against the full dotted module, class and method name. The registry is a set, so the
  # run order is not deterministic.
  # Iterates over a snapshot of the global registry; for every attribute whose name starts with
  # "benchmark" and is callable, builds the full name and, if the regex is the literal word "all"
  # or a search match, lazily instantiates the class once and calls the bound method. Raises a
  # value error if nothing matched. Side effects: runs benchmarks (which may print or write
  # result files).
  """Run benchmarks that match regex `regex`.

  This function goes through the global benchmark registry, and matches
  benchmark class and method names of the form
  `module.name.BenchmarkClass.benchmarkMethod` to the given regex.
  If a method matches, it is run.

  Args:
    regex: The string regular expression to match Benchmark classes against.

  Raises:
    ValueError: If no benchmarks were selected by the input regex.
  """
  registry = list(GLOBAL_BENCHMARK_REGISTRY)

  selected_benchmarks = []
  # Match benchmarks in registry against regex
  for benchmark in registry:
    benchmark_name = "%s.%s" % (benchmark.__module__, benchmark.__name__)
    attrs = dir(benchmark)
    # Don't instantiate the benchmark class unless necessary
    benchmark_instance = None

    for attr in attrs:
      if not attr.startswith("benchmark"):
        continue
      candidate_benchmark_fn = getattr(benchmark, attr)
      if not callable(candidate_benchmark_fn):
        continue
      full_benchmark_name = "%s.%s" % (benchmark_name, attr)
      if regex == "all" or re.search(regex, full_benchmark_name):
        selected_benchmarks.append(full_benchmark_name)
        # Instantiate the class if it hasn't been instantiated
        benchmark_instance = benchmark_instance or benchmark()
        # Get the method tied to the class
        instance_benchmark_fn = getattr(benchmark_instance, attr)
        # Call the instance method
        instance_benchmark_fn()

  if not selected_benchmarks:
    raise ValueError("No benchmarks matched the pattern: '{}'".format(regex))


def benchmarks_main(true_main, argv=None):
  # [FUNCTION DOC]
  # Entry point. Looks in argv (defaults to the process arguments) for an argument starting with
  # the benchmark-filter flag (single or double dash, with an equals sign). If found, removes it
  # from argv (mutating the list), takes the text after the first equals sign as a regular
  # expression, and runs the matching benchmarks through the application runner. Otherwise
  # calls true_main, a callable taking no arguments. Returns nothing; the app runner normally
  # exits the process. Inputs: true_main callable, optional argv list.
  """Run benchmarks as declared in argv.

  Args:
    true_main: True main function to run if benchmarks are not requested.
    argv: the command line arguments (if None, uses sys.argv).
  """
  if argv is None:
    argv = sys.argv
  found_arg = [
      arg
      for arg in argv
      if arg.startswith("--benchmark_filter=")
      or arg.startswith("-benchmark_filter=")
  ]
  if found_arg:
    # Remove --benchmark_filter arg from sys.argv
    argv.remove(found_arg[0])

    regex = found_arg[0].split("=")[1]
    app.run(lambda _: _run_benchmarks(regex), argv=argv)
  else:
    true_main()
