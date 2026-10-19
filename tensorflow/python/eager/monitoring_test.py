# [MODULE DOC]
# Unit tests for the monitoring wrappers: counters, integer/string/boolean gauges, samplers with
# exponential buckets, the timer context manager (including section tracking and repetitive
# counting avoidance) and the timing decorator. Runs the test main entry point when executed
# directly.
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
"""Tests for monitoring."""

import time

from tensorflow.python.eager import monitoring
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util


class MonitoringTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering all metric kinds and timer utilities of the monitoring module. No shared
  # fixtures; each test creates its own metrics.

  def test_counter(self):
    # [FUNCTION DOC]
    # Creates a label-less counter, increments its cell by 1 then 5, and asserts the value is 1
    # then 6.
    counter = monitoring.Counter('test/counter', 'test counter')
    counter.get_cell().increase_by(1)
    self.assertEqual(counter.get_cell().value(), 1)
    counter.get_cell().increase_by(5)
    self.assertEqual(counter.get_cell().value(), 6)

  def test_multiple_counters(self):
    # [FUNCTION DOC]
    # Creates counters with one and with two labels, increments labeled cells, and asserts the
    # cell values (1 and 5) are returned for the matching label values.
    counter1 = monitoring.Counter('test/counter1', 'test counter', 'label1')
    counter1.get_cell('foo').increase_by(1)
    self.assertEqual(counter1.get_cell('foo').value(), 1)
    counter2 = monitoring.Counter('test/counter2', 'test counter', 'label1',
                                  'label2')
    counter2.get_cell('foo', 'bar').increase_by(5)
    self.assertEqual(counter2.get_cell('foo', 'bar').value(), 5)

  def test_same_counter(self):
    # [FUNCTION DOC]
    # Creates two counters with the same name; the test only expects no error to be raised (no
    # explicit assertions).
    counter1 = monitoring.Counter('test/same_counter', 'test counter')  # pylint: disable=unused-variable
    counter2 = monitoring.Counter('test/same_counter', 'test counter')  # pylint: disable=unused-variable

  def test_int_gauge(self):
    # [FUNCTION DOC]
    # Sets and reads back integer gauge values (label-less: 1 then 5; one label: 2) and asserts
    # each read returns the last value set.
    gauge = monitoring.IntGauge('test/gauge', 'test gauge')
    gauge.get_cell().set(1)
    self.assertEqual(gauge.get_cell().value(), 1)
    gauge.get_cell().set(5)
    self.assertEqual(gauge.get_cell().value(), 5)

    gauge1 = monitoring.IntGauge('test/gauge1', 'test gauge1', 'label1')
    gauge1.get_cell('foo').set(2)
    self.assertEqual(gauge1.get_cell('foo').value(), 2)

  def test_string_gauge(self):
    # [FUNCTION DOC]
    # Sets and reads back string gauge values, label-less and with one label, asserting each read
    # returns the last string set.
    gauge = monitoring.StringGauge('test/gauge', 'test gauge')
    gauge.get_cell().set('left')
    self.assertEqual(gauge.get_cell().value(), 'left')
    gauge.get_cell().set('right')
    self.assertEqual(gauge.get_cell().value(), 'right')

    gauge1 = monitoring.StringGauge('test/gauge1', 'test gauge1', 'label1')
    gauge1.get_cell('foo').set('start')
    self.assertEqual(gauge1.get_cell('foo').value(), 'start')

  def test_bool_gauge(self):
    # [FUNCTION DOC]
    # Sets and reads back boolean gauge values, label-less (true then false) and with one label
    # (true), asserting truthiness of each read.
    gauge = monitoring.BoolGauge('test/gauge', 'test gauge')
    gauge.get_cell().set(True)
    self.assertTrue(gauge.get_cell().value())
    gauge.get_cell().set(False)
    self.assertFalse(gauge.get_cell().value())

    gauge1 = monitoring.BoolGauge('test/gauge1', 'test gauge1', 'label1')
    gauge1.get_cell('foo').set(True)
    self.assertTrue(gauge1.get_cell('foo').value())

  def test_sampler(self):
    # [FUNCTION DOC]
    # Creates exponential buckets and samplers (label-less and one label), adds samples, and
    # asserts the histogram minimum, maximum, count and sum for the sampled cells (for example
    # samples 1 and 5 give min 1, count 2, sum 6; samples 2 and 4 under one label give max 4,
    # count 2, sum 6).
    buckets = monitoring.ExponentialBuckets(1.0, 2.0, 2)
    sampler = monitoring.Sampler('test/sampler', buckets, 'test sampler')
    sampler.get_cell().add(1.0)
    sampler.get_cell().add(5.0)
    histogram_proto = sampler.get_cell().value()
    self.assertEqual(histogram_proto.min, 1.0)
    self.assertEqual(histogram_proto.num, 2.0)
    self.assertEqual(histogram_proto.sum, 6.0)

    sampler1 = monitoring.Sampler('test/sampler1', buckets, 'test sampler',
                                  'label1')
    sampler1.get_cell('foo').add(2.0)
    sampler1.get_cell('foo').add(4.0)
    sampler1.get_cell('bar').add(8.0)
    histogram_proto1 = sampler1.get_cell('foo').value()
    self.assertEqual(histogram_proto1.max, 4.0)
    self.assertEqual(histogram_proto1.num, 2.0)
    self.assertEqual(histogram_proto1.sum, 6.0)

  def test_context_manager(self):
    # [FUNCTION DOC]
    # Nests two timers (outer 'long' cell, inner 'short' cell) with short sleeps and asserts the
    # outer cell accumulated more time than the inner one.
    counter = monitoring.Counter('test/ctxmgr', 'test context manager', 'slot')
    with monitoring.MonitoredTimer(counter.get_cell('long')):
      time.sleep(0.01)
      with monitoring.MonitoredTimer(counter.get_cell('short')):
        time.sleep(0.01)
    self.assertGreater(
        counter.get_cell('long').value(), counter.get_cell('short').value()
    )

  def test_monitored_timer_tracker(self):
    # [FUNCTION DOC]
    # Nests named timers and asserts the module-level active section list contains the outer
    # names while inside, and no longer contains the inner name after it exits. Then asserts the
    # outer cell exceeds the never-incremented 'short' cell and the second counter is positive.
    counter = monitoring.Counter('test/ctxmgr', 'test context manager', 'slot')
    counter2 = monitoring.Counter('test/ctxmgr2', 'slot')
    with monitoring.MonitoredTimer(counter.get_cell('long'), 'counter'):
      time.sleep(0.01)
      self.assertIn('counter', monitoring.MonitoredTimerSections)
      with monitoring.MonitoredTimer(counter2.get_cell(), 'counter2'):
        time.sleep(0.01)
        self.assertIn('counter', monitoring.MonitoredTimerSections)
        self.assertIn('counter2', monitoring.MonitoredTimerSections)
        with monitoring.MonitoredTimer(counter.get_cell('long'), 'counter'):
          time.sleep(0.01)
      self.assertNotIn('counter2', monitoring.MonitoredTimerSections)
    self.assertGreater(
        counter.get_cell('long').value(), counter.get_cell('short').value()
    )
    self.assertGreater(counter2.get_cell().value(), 0)

  def test_repetitive_monitored_timer(self):
    # [FUNCTION DOC]
    # Nests two timers with the same section name and repetitive-counting avoidance on. Inside the
    # outer block, asserts the counter is still zero after the inner timer (not counted); after
    # the outer block, asserts it is positive. Sleeps about two seconds in total.
    counter = monitoring.Counter('test/ctxmgr', 'test context manager')
    with monitoring.MonitoredTimer(
        counter.get_cell(),
        monitored_section_name='action1',
        avoid_repetitive_counting=True,
    ):
      time.sleep(1)
      with monitoring.MonitoredTimer(
          counter.get_cell(),
          monitored_section_name='action1',
          avoid_repetitive_counting=True,
      ):
        time.sleep(1)

      # The inner section is not timed.
      self.assertEqual(counter.get_cell().value(), 0)

    self.assertGreater(counter.get_cell().value(), 0)

  def test_function_decorator(self):
    # [FUNCTION DOC]
    # Decorates a sleeping function with the timing decorator, calls it for 1 millisecond, and
    # asserts the counter exceeds 1000 microseconds.
    counter = monitoring.Counter('test/funcdecorator', 'test func decorator')

    @monitoring.monitored_timer(counter.get_cell())
    def timed_function(seconds):
      # [FUNCTION DOC]
      # Sleeps for the given number of seconds; the decorator times it.
      time.sleep(seconds)

    timed_function(0.001)
    self.assertGreater(counter.get_cell().value(), 1000)


if __name__ == '__main__':
  test.main()
