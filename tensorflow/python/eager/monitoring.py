# [MODULE DOC]
# Python wrappers for the TensorFlow runtime monitoring (metrics) API. Exposes counters, integer,
# string and boolean gauges, samplers (histograms) with exponential bucketing, and a timer
# context manager / decorator that adds elapsed microseconds to a counter cell. Each metric wraps
# a native handle created through the eager C extension, selected by the number of labels via
# tables of create/delete/get-cell methods. Import time side effects: builds those method tables
# and registers the boolean gauge as an internal export.
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
"""TensorFlow monitoring APIs."""

import collections
import functools
import time

from tensorflow.core.framework import summary_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import c_api_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

_MetricMethod = collections.namedtuple('MetricMethod', 'create delete get_cell')
_counter_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewCounter0,
        delete=pywrap_tfe.TFE_MonitoringDeleteCounter0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellCounter0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewCounter1,
        delete=pywrap_tfe.TFE_MonitoringDeleteCounter1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellCounter1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewCounter2,
        delete=pywrap_tfe.TFE_MonitoringDeleteCounter2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellCounter2),
]
_int_gauge_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewIntGauge0,
        delete=pywrap_tfe.TFE_MonitoringDeleteIntGauge0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellIntGauge0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewIntGauge1,
        delete=pywrap_tfe.TFE_MonitoringDeleteIntGauge1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellIntGauge1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewIntGauge2,
        delete=pywrap_tfe.TFE_MonitoringDeleteIntGauge2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellIntGauge2),
]
_string_gauge_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge0,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge1,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge2,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge2),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge3,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge3,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge3),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewStringGauge4,
        delete=pywrap_tfe.TFE_MonitoringDeleteStringGauge4,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellStringGauge4),
]
_bool_gauge_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewBoolGauge0,
        delete=pywrap_tfe.TFE_MonitoringDeleteBoolGauge0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellBoolGauge0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewBoolGauge1,
        delete=pywrap_tfe.TFE_MonitoringDeleteBoolGauge1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellBoolGauge1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewBoolGauge2,
        delete=pywrap_tfe.TFE_MonitoringDeleteBoolGauge2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellBoolGauge2),
]
_sampler_methods = [
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewSampler0,
        delete=pywrap_tfe.TFE_MonitoringDeleteSampler0,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellSampler0),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewSampler1,
        delete=pywrap_tfe.TFE_MonitoringDeleteSampler1,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellSampler1),
    _MetricMethod(
        create=pywrap_tfe.TFE_MonitoringNewSampler2,
        delete=pywrap_tfe.TFE_MonitoringDeleteSampler2,
        get_cell=pywrap_tfe.TFE_MonitoringGetCellSampler2),
]


class Metric(object):
  # [CLASS DOC]
  # Base class of all metrics. Holds the native metric handle, the metric type name, the table of
  # native methods (indexed by label count) and the label count. Subclasses supply the method
  # table and wrap returned cells. Uses slots, so no other attributes can be added.
  """The base class of metric."""

  __slots__ = ["_metric", "_metric_name", "_metric_methods", "_label_length"]

  def __init__(self, metric_name, metric_methods, label_length, *args):
    # [FUNCTION DOC]
    # Creates the native metric. metric_name: type name used in messages. metric_methods: list of
    # method records indexed by label count. label_length: number of labels. Extra positional
    # arguments are passed to the native create method. Raises a value error if the label count is
    # not smaller than the number of supported variants. Stores the native handle.
    """Creates a new metric.

    Args:
      metric_name: name of the metric class.
      metric_methods: list of swig metric methods.
      label_length: length of label args.
      *args: the arguments to call create method.
    """
    self._metric_name = metric_name
    self._metric_methods = metric_methods
    self._label_length = label_length

    if label_length >= len(self._metric_methods):
      raise ValueError('Cannot create {} metric with label >= {}'.format(
          self._metric_name, len(self._metric_methods)))

    self._metric = self._metric_methods[self._label_length].create(*args)

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: deletes the native metric handle using the delete method for this label count.
    # Silently returns if the object was only partly initialized (missing attributes). Returns
    # nothing.
    try:
      deleter = self._metric_methods[self._label_length].delete
      metric = self._metric
    except AttributeError:
      return

    if deleter is not None:
      deleter(metric)

  def get_cell(self, *labels):
    # [FUNCTION DOC]
    # Returns the raw native cell for the given label values. Raises a value error if the number
    # of labels differs from the number the metric was created with.
    """Retrieves the cell."""
    if len(labels) != self._label_length:
      raise ValueError('The {} expects taking {} labels'.format(
          self._metric_name, self._label_length))
    return self._metric_methods[self._label_length].get_cell(
        self._metric, *labels)


class CounterCell(object):
  # [CLASS DOC]
  # One value of a Counter (for one label combination); wraps a native counter cell handle.
  """CounterCell stores each value of a Counter."""

  __slots__ = ["_cell"]

  def __init__(self, cell):
    # [FUNCTION DOC]
    # Stores the native cell handle. cell: native counter cell pointer. Returns nothing.
    """Creates a new CounterCell.

    Args:
      cell: A c pointer of TFE_MonitoringCounterCell.
    """
    self._cell = cell

  def increase_by(self, value):
    # [FUNCTION DOC]
    # Atomically adds value (a non-negative number) to the counter cell through the native layer.
    # Returns nothing; the native layer decides how negative values are handled.
    """Atomically increments the value.

    Args:
      value: non-negative value.
    """
    pywrap_tfe.TFE_MonitoringCounterCellIncrementBy(self._cell, value)

  def value(self):
    # [FUNCTION DOC]
    # Returns the current integer value of the counter cell from the native layer.
    """Retrieves the current value."""
    return pywrap_tfe.TFE_MonitoringCounterCellValue(self._cell)


class Counter(Metric):
  # [CLASS DOC]
  # Cumulative integer metric with optional labels; each label combination is a cell that can only
  # be incremented. Backed by the counter method table (0 to 2 labels).
  """A stateful class for updating a cumulative integer metric.

  This class encapsulates a set of values (or a single value for a label-less
  metric). Each value is identified by a tuple of labels. The class allows the
  user to increment each value.
  """

  __slots__ = []

  def __init__(self, name, description, *labels):
    # [FUNCTION DOC]
    # Creates a counter named name with a description and zero or more label names (at most 2,
    # otherwise a value error is raised by the base class). Registers a native counter.
    """Creates a new Counter.

    Args:
      name: name of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
    super(Counter, self).__init__('Counter', _counter_methods, len(labels),
                                  name, description, *labels)

  def get_cell(self, *labels):
    # [FUNCTION DOC]
    # Returns a counter cell wrapper for the given label values (count must match the label
    # count, else a value error is raised).
    """Retrieves the cell."""
    return CounterCell(super(Counter, self).get_cell(*labels))


class IntGaugeCell(object):
  # [CLASS DOC]
  # One integer value of an IntGauge for one label combination; wraps a native cell handle.
  """A single integer value stored in an `IntGauge`."""

  __slots__ = ["_cell"]

  def __init__(self, cell):
    # [FUNCTION DOC]
    # Stores the native cell handle. cell: native integer gauge cell pointer.
    """Creates a new IntGaugeCell.

    Args:
      cell: A c pointer of TFE_MonitoringIntGaugeCell.
    """
    self._cell = cell

  def set(self, value):
    # [FUNCTION DOC]
    # Atomically sets the gauge cell to the given integer. Returns nothing.
    """Atomically set the value.

    Args:
      value: integer value.
    """
    pywrap_tfe.TFE_MonitoringIntGaugeCellSet(self._cell, value)

  def value(self):
    # [FUNCTION DOC]
    # Returns the current integer value of the gauge cell.
    """Retrieves the current value."""
    return pywrap_tfe.TFE_MonitoringIntGaugeCellValue(self._cell)


class IntGauge(Metric):
  # [CLASS DOC]
  # Gauge-like integer metric with optional labels (0 to 2); each label combination is a cell
  # whose value can be set.
  """A stateful class for updating a gauge-like integer metric.

  This class encapsulates a set of integer values (or a single value for a
  label-less metric). Each value is identified by a tuple of labels. The class
  allows the user to set each value.
  """

  __slots__ = []

  def __init__(self, name, description, *labels):
    # [FUNCTION DOC]
    # Creates an integer gauge with the given name, description and label names (at most 2, else
    # a value error is raised by the base class).
    """Creates a new IntGauge.

    Args:
      name: name of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
    super(IntGauge, self).__init__('IntGauge', _int_gauge_methods, len(labels),
                                   name, description, *labels)

  def get_cell(self, *labels):
    # [FUNCTION DOC]
    # Returns an integer gauge cell wrapper for the given label values (count must match).
    """Retrieves the cell."""
    return IntGaugeCell(super(IntGauge, self).get_cell(*labels))


class StringGaugeCell(object):
  # [CLASS DOC]
  # One string value of a StringGauge for one label combination; wraps a native cell handle.
  """A single string value stored in an `StringGauge`."""

  __slots__ = ["_cell"]

  def __init__(self, cell):
    # [FUNCTION DOC]
    # Stores the native cell handle. cell: native string gauge cell pointer.
    """Creates a new StringGaugeCell.

    Args:
      cell: A c pointer of TFE_MonitoringStringGaugeCell.
    """
    self._cell = cell

  def set(self, value):
    # [FUNCTION DOC]
    # Atomically sets the gauge cell to the given string. Returns nothing.
    """Atomically set the value.

    Args:
      value: string value.
    """
    pywrap_tfe.TFE_MonitoringStringGaugeCellSet(self._cell, value)

  def value(self):
    # [FUNCTION DOC]
    # Returns the current string value: the native layer writes it into a temporary buffer, which
    # is read and decoded as UTF-8 text.
    """Retrieves the current value."""
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_MonitoringStringGaugeCellValue(self._cell, buffer_)
      value = pywrap_tf_session.TF_GetBuffer(buffer_).decode('utf-8')
    return value


class StringGauge(Metric):
  # [CLASS DOC]
  # Gauge-like string metric with optional labels (0 to 4); each label combination is a cell
  # whose value can be set.
  """A stateful class for updating a gauge-like string metric.

  This class encapsulates a set of string values (or a single value for a
  label-less metric). Each value is identified by a tuple of labels. The class
  allows the user to set each value.
  """

  __slots__ = []

  def __init__(self, name, description, *labels):
    # [FUNCTION DOC]
    # Creates a string gauge with the given name, description and label names (at most 4, else a
    # value error is raised by the base class).
    """Creates a new StringGauge.

    Args:
      name: name of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
    super(StringGauge, self).__init__('StringGauge', _string_gauge_methods,
                                      len(labels), name, description, *labels)

  def get_cell(self, *labels):
    # [FUNCTION DOC]
    # Returns a string gauge cell wrapper for the given label values (count must match).
    """Retrieves the cell."""
    return StringGaugeCell(super(StringGauge, self).get_cell(*labels))


class BoolGaugeCell(object):
  # [CLASS DOC]
  # One boolean value of a BoolGauge for one label combination; wraps a native cell handle.
  """A single boolean value stored in an `BoolGauge`."""

  __slots__ = ["_cell"]

  def __init__(self, cell):
    # [FUNCTION DOC]
    # Stores the native cell handle. cell: native boolean gauge cell pointer.
    """Creates a new BoolGaugeCell.

    Args:
      cell: A c pointer of TFE_MonitoringBoolGaugeCell.
    """
    self._cell = cell

  def set(self, value):
    # [FUNCTION DOC]
    # Atomically sets the gauge cell to the given boolean. Returns nothing.
    """Atomically set the value.

    Args:
      value: bool value.
    """
    pywrap_tfe.TFE_MonitoringBoolGaugeCellSet(self._cell, value)

  def value(self):
    # [FUNCTION DOC]
    # Returns the current boolean value of the gauge cell.
    """Retrieves the current value."""
    return pywrap_tfe.TFE_MonitoringBoolGaugeCellValue(self._cell)


@tf_export("__internal__.monitoring.BoolGauge", v1=[])
class BoolGauge(Metric):
  # [CLASS DOC]
  # Gauge-like boolean metric with optional labels (0 to 2); exported as an internal API.
  """A stateful class for updating a gauge-like bool metric.

  This class encapsulates a set of boolean values (or a single value for a
  label-less metric). Each value is identified by a tuple of labels. The class
  allows the user to set each value.
  """

  __slots__ = []

  def __init__(self, name, description, *labels):
    # [FUNCTION DOC]
    # Creates a boolean gauge with the given name, description and label names (at most 2, else a
    # value error is raised by the base class).
    """Creates a new BoolGauge.

    Args:
      name: name of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
    super(BoolGauge, self).__init__('BoolGauge', _bool_gauge_methods,
                                    len(labels), name, description, *labels)

  def get_cell(self, *labels):
    # [FUNCTION DOC]
    # Returns a boolean gauge cell wrapper for the given label values (count must match).
    """Retrieves the cell."""
    return BoolGaugeCell(super(BoolGauge, self).get_cell(*labels))


class SamplerCell(object):
  # [CLASS DOC]
  # One histogram of a Sampler for one label combination; wraps a native cell handle.
  """SamplerCell stores each value of a Sampler."""

  __slots__ = ["_cell"]

  def __init__(self, cell):
    # [FUNCTION DOC]
    # Stores the native cell handle. cell: native sampler cell pointer.
    """Creates a new SamplerCell.

    Args:
      cell: A c pointer of TFE_MonitoringSamplerCell.
    """
    self._cell = cell

  def add(self, value):
    # [FUNCTION DOC]
    # Atomically records one sample (a float) in the histogram. Returns nothing.
    """Atomically add a sample.

    Args:
      value: float value.
    """
    pywrap_tfe.TFE_MonitoringSamplerCellAdd(self._cell, value)

  def value(self):
    # [FUNCTION DOC]
    # Reads the serialized histogram from the native layer into a temporary buffer and parses it
    # into a histogram protocol message, which is returned.
    """Retrieves the current distribution of samples.

    Returns:
      A HistogramProto describing the distribution of samples.
    """
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_MonitoringSamplerCellValue(self._cell, buffer_)
      proto_data = pywrap_tf_session.TF_GetBuffer(buffer_)
    histogram_proto = summary_pb2.HistogramProto()
    histogram_proto.ParseFromString(compat.as_bytes(proto_data))
    return histogram_proto


class Buckets(object):
  # [CLASS DOC]
  # Base class for bucketing strategies; owns a native buckets handle (public attribute buckets)
  # which is deleted when the object is finalized.
  """Bucketing strategies for the samplers."""

  __slots__ = ["buckets"]

  def __init__(self, buckets):
    # [FUNCTION DOC]
    # Stores the native buckets handle. buckets: native buckets pointer. Returns nothing.
    """Creates a new Buckets.

    Args:
      buckets: A c pointer of TFE_MonitoringBuckets.
    """
    self.buckets = buckets

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: releases the native buckets handle. Returns nothing.
    pywrap_tfe.TFE_MonitoringDeleteBuckets(self.buckets)


class ExponentialBuckets(Buckets):
  # [CLASS DOC]
  # Exponentially growing bucket boundaries, created natively from scale, growth factor and count.
  """Exponential bucketing strategy.

  Sets up buckets of the form:
      [-DBL_MAX, ..., scale * growth^i,
       scale * growth_factor^(i + 1), ..., DBL_MAX].
  """

  __slots__ = []

  def __init__(self, scale, growth_factor, bucket_count):
    # [FUNCTION DOC]
    # Creates native exponential buckets from scale (float), growth_factor (float) and
    # bucket_count (integer), and hands the handle to the base class.
    """Creates a new exponential Buckets.

    Args:
      scale: float
      growth_factor: float
      bucket_count: integer
    """
    super(ExponentialBuckets, self).__init__(
        pywrap_tfe.TFE_MonitoringNewExponentialBuckets(scale, growth_factor,
                                                       bucket_count))


class Sampler(Metric):
  # [CLASS DOC]
  # Cumulative histogram metric with optional labels (0 to 2) and a bucketing strategy; each label
  # combination is a cell to which samples are added.
  """A stateful class for updating a cumulative histogram metric.

  This class encapsulates a set of histograms (or a single histogram for a
  label-less metric) configured with a list of increasing bucket boundaries.
  Each histogram is identified by a tuple of labels. The class allows the
  user to add a sample to each histogram value.
  """

  __slots__ = []

  def __init__(self, name, buckets, description, *labels):
    # [FUNCTION DOC]
    # Creates a sampler with the given name, a buckets object (its native handle is used), a
    # description and label names (at most 2, else a value error is raised by the base class).
    """Creates a new Sampler.

    Args:
      name: name of the new metric.
      buckets: bucketing strategy of the new metric.
      description: description of the new metric.
      *labels: The label list of the new metric.
    """
    super(Sampler, self).__init__('Sampler', _sampler_methods, len(labels),
                                  name, buckets.buckets, description, *labels)

  def get_cell(self, *labels):
    # [FUNCTION DOC]
    # Returns a sampler cell wrapper for the given label values (count must match).
    """Retrieves the cell."""
    return SamplerCell(super(Sampler, self).get_cell(*labels))


# Keeping track of current MonitoredTimer sections to prevent repetitive
# counting.
MonitoredTimerSections = []


class MonitoredTimer(object):
  # [CLASS DOC]
  # Context manager measuring wall time of a block and adding the elapsed microseconds to a
  # counter cell on exit. Optionally tracks named sections in the module-level list of active
  # sections so nested sections with the same name are not counted twice.
  """A context manager to measure the walltime and increment a Counter cell."""

  __slots__ = [
      "cell",
      "t",
      "monitored_section_name",
      "_counting",
      "_avoid_repetitive_counting",
  ]

  def __init__(
      self, cell, monitored_section_name=None, avoid_repetitive_counting=False
  ):
    # [FUNCTION DOC]
    # Stores the counter cell, the optional section name (default none) and the flag to avoid
    # repetitive counting (default false); counting starts enabled. Returns nothing.
    """Creates a new MonitoredTimer.

    Args:
      cell: the cell associated with the time metric that will be inremented.
      monitored_section_name: name of action being monitored here.
      avoid_repetitive_counting: when set to True, if already in a monitored
        timer section with the same monitored_section_name, skip counting.
    """
    self.cell = cell
    self.monitored_section_name = monitored_section_name
    self._avoid_repetitive_counting = avoid_repetitive_counting
    self._counting = True

  def __enter__(self):
    # [FUNCTION DOC]
    # Starts timing. If repetitive counting is to be avoided and a section with this name is
    # already active, disables counting for this timer and returns itself without timing.
    # Otherwise records the start time, registers the section name in the module-level active
    # list (if a name was given) and returns itself.
    if (
        self._avoid_repetitive_counting
        and self.monitored_section_name
        and self.monitored_section_name in MonitoredTimerSections
    ):
      self._counting = False
      return self

    self.t = time.time()
    if self.monitored_section_name:
      MonitoredTimerSections.append(self.monitored_section_name)

    return self

  def __exit__(self, exception_type, exception_value, traceback):
    # [FUNCTION DOC]
    # Stops timing. If counting is enabled, adds the elapsed time in whole microseconds to the
    # counter cell and removes the section name from the active list. Exception arguments are
    # ignored and exceptions are not suppressed (returns nothing).
    del exception_type, exception_value, traceback
    if self._counting:
      micro_seconds = (time.time() - self.t) * 1000000
      self.cell.increase_by(int(micro_seconds))
      if self.monitored_section_name:
        MonitoredTimerSections.remove(self.monitored_section_name)


def monitored_timer(cell):
  # [FUNCTION DOC]
  # Returns a decorator that wraps a function so each call runs inside a timer context that adds
  # the elapsed microseconds to the given counter cell (no section name is used).
  """A function decorator for adding MonitoredTimer support.

  Args:
    cell: the cell associated with the time metric that will be inremented.
  Returns:
    A decorator that measure the function runtime and increment the specified
    counter cell.
  """

  def actual_decorator(func):
    # [FUNCTION DOC]
    # HELPER: inner decorator built by the timer decorator factory.
    # Wraps func, preserving its metadata, and returns the wrapper.

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: wrapper for the timing decorator.
      # Calls the wrapped function with the same arguments inside a timer and returns its result.
      with MonitoredTimer(cell):
        return func(*args, **kwargs)

    return wrapper

  return actual_decorator
