# [MODULE DOC]
# HELPER: internal metrics support for the parameter server coordinator.
# Lazily creates a fixed mapping of monitoring metrics (timing samplers for function tracing,
# closure execution, remote value fetch and server definition update; gauges for queued and
# in-flight closures; a counter for recoverable worker failures) and exposes helpers to time a
# block, set an integer gauge, increment the counter, and summarize a metric. A module level flag
# turns metric collection on or off. Metrics are registered only when first needed, not at import.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Metrics collecting utilities for single client training."""

import time

from tensorflow.python.eager import monitoring
from tensorflow.python.util import tf_contextlib

enable_metrics = True
_METRICS_MAPPING = {}


def _init():
  # [FUNCTION DOC]
  # HELPER: lazily builds the metrics mapping used by the other helpers.
  # Defines exponential bucket boundaries for the samplers, creates the samplers, gauges and
  # counter under the parameter server coordinator metric path, and stores them in the module
  # level mapping under short names (function tracing, closure execution, remote value fetch,
  # server definition update, queued closures, in-flight closures, worker failures). Takes no
  # input and returns nothing. Registering the same metric names twice would be an error in the
  # monitoring library, so it must be called once (callers check that the mapping is empty).
  """Initialize the metrics mapping."""
  global _METRICS_MAPPING

  # Define the boundaries for bucketing times of distribution (Sampler) metrics.

  # Closure execution: range from 0.1s to 10000s, i.e. [(0.1, 1), (1, 10), ...]
  execution_time_buckets = monitoring.ExponentialBuckets(
      scale=0.1, growth_factor=10, bucket_count=6)
  # Tracing: same range as execution
  tracing_time_buckets = execution_time_buckets
  # Remote value fetch: range from 0.001s (i.e. 1ms) to 1000s
  fetch_time_buckets = monitoring.ExponentialBuckets(
      scale=0.001, growth_factor=10, bucket_count=7)
  # Server def update: range from 1s to 10000s
  server_update_time_buckets = monitoring.ExponentialBuckets(
      scale=1, growth_factor=10, bucket_count=5)

  function_tracing_sampler = monitoring.Sampler(
      '/tensorflow/api/ps_strategy/coordinator/function_tracing',
      tracing_time_buckets,
      'Sampler to track the time (in seconds) for tracing functions.')

  closure_execution_sampler = monitoring.Sampler(
      '/tensorflow/api/ps_strategy/coordinator/closure_execution',
      execution_time_buckets,
      'Sampler to track the time (in seconds) for executing closures.')

  remote_value_fetch_sampler = monitoring.Sampler(
      '/tensorflow/api/ps_strategy/coordinator/remote_value_fetch',
      fetch_time_buckets,
      'Sampler to track the time (in seconds) for fetching remote_value.')

  server_def_update_sampler = monitoring.Sampler(
      '/tensorflow/api/ps_strategy/coordinator/server_def_update',
      server_update_time_buckets,
      'Sample to track the time (in seconds) for updating the server def upon '
      'worker recovery.')

  queued_closure_gauge = monitoring.IntGauge(
      '/tensorflow/api/ps_strategy/coordinator/queued_closures',
      'Track how many closures are in the coordinator queue pending execution.')

  inflight_closure_gauge = monitoring.IntGauge(
      '/tensorflow/api/ps_strategy/coordinator/inflight_closures',
      'Track how many closures are currently being processed by workers.')

  worker_failure_counter = monitoring.Counter(
      '/tensorflow/api/ps_strategy/coordinator/recoverable_worker_failure_count',
      'Track how many recoverable worker failures have been encountered.')

  _METRICS_MAPPING = {
      'function_tracing': function_tracing_sampler,
      'closure_execution': closure_execution_sampler,
      'remote_value_fetch': remote_value_fetch_sampler,
      'server_def_update': server_def_update_sampler,
      'queued_closures': queued_closure_gauge,
      'inflight_closures': inflight_closure_gauge,
      'worker_failures': worker_failure_counter,
  }


@tf_contextlib.contextmanager
def monitored_timer(metric_name, state_tracker=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: if the wrapped block raises an exception, nothing after the yield runs, so no
  # duration is recorded for failed blocks.
  # Context manager that measures the wall clock time of the wrapped block and adds it to the
  # named sampler metric. Parameters: metric_name (one of the sampler names in the mapping), and
  # optional state_tracker, a callback returning a comparable state; when given, the duration is
  # recorded only if the state after the block differs from the state before it. Does nothing but
  # run the block when metrics are disabled. Initializes the mapping on first use. Raises a
  # missing-key error for an unknown metric name when recording.
  """Monitor the execution time and collect it into the specified metric."""
  if not enable_metrics:
    yield
  else:
    if not _METRICS_MAPPING:
      _init()
    start_time = time.time()
    start_state = state_tracker() if state_tracker else None
    yield
    duration_sec = time.time() - start_time
    # If a state_checker is provided, record the metric only if the end state is
    # different from the start state.
    if state_tracker is None or state_tracker() != start_state:
      metric = _METRICS_MAPPING[metric_name]
      metric.get_cell().add(duration_sec)


def monitor_int(metric_name, value):
  # [FUNCTION DOC]
  # Sets the named integer gauge metric to value (integer). Does nothing if metrics are disabled;
  # initializes the mapping on first use. Raises a missing-key error for an unknown metric name.
  # Returns nothing.
  if not enable_metrics:
    return
  else:
    if not _METRICS_MAPPING:
      _init()
    metric = _METRICS_MAPPING[metric_name]
    metric.get_cell().set(value)


def monitor_increment_counter(metric_name):
  # [FUNCTION DOC]
  # Increments the named counter metric by one. Does nothing if metrics are disabled; initializes
  # the mapping on first use. An unknown name raises a missing-key error, and a name of a
  # non-counter metric would fail inside the monitoring library. Returns nothing.
  if not enable_metrics:
    return
  else:
    if not _METRICS_MAPPING:
      _init()
    metric = _METRICS_MAPPING[metric_name]
    metric.get_cell().increase_by(1)


def _get_metric_histogram(histogram_proto):
  # [FUNCTION DOC]
  # HELPER: converts a sampler result for get_metric_summary.
  # Takes a histogram message and returns a mapping with the minimum, maximum, count and sum, and
  # a mapping from (lower bound, upper bound) pairs to bucket values, where an extra lower limit
  # of zero is prepended because the metrics are durations. Mutates the bucket limit list of the
  # given message (inserts a zero at the front).
  """Convert a histogram proto into a dict.

  Args:
    histogram_proto: a proto containing a Sampler metric's result histogram.

  Returns:
    A dict containing summary statistics and the raw histogram values.
  """
  ret = dict()
  ret['min'] = histogram_proto.min
  ret['max'] = histogram_proto.max
  ret['num'] = histogram_proto.num
  ret['sum'] = histogram_proto.sum

  bucket_limits = histogram_proto.bucket_limit
  bucket_vals = histogram_proto.bucket
  ret['histogram'] = {}
  # Add lower limit as 0, since all these metrics are durations
  bucket_limits.insert(0, 0)
  for lb, ub, val in zip(bucket_limits[:-1], bucket_limits[1:], bucket_vals):
    ret['histogram'][(lb, ub)] = val
  return ret


def get_metric_summary(metric_name):
  # [FUNCTION DOC]
  # KNOWN ISSUE: does not initialize the metrics mapping, so it raises a missing-key error if
  # called before any metric helper has run.
  # Returns the current value of the named metric: a summary mapping for sampler metrics (see the
  # histogram helper), or the raw integer value for gauges and counters.
  """Get summary for the specified metric."""
  metric = _METRICS_MAPPING[metric_name]
  result = metric.get_cell().value()
  if isinstance(metric, monitoring.Sampler):
    result = _get_metric_histogram(result)
  return result
