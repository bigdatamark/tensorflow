# [MODULE DOC]
# Unit tests for the profiling data structures: checks that the aggregate profile computes
# correct total op time, total execution time, distinct node count and execution count when built
# from one record and when more records (different node, same node, same node on a different
# device) are added. Test module; no import-time side effects.
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
"""Unit tests for the basic data structures and algorithms for profiling."""

from tensorflow.core.framework import step_stats_pb2
from tensorflow.python.debug.lib import profiling
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class AggregateProfile(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the aggregate profile class (the name shadows the production class's name but
  # this is a test class). Shared fixture: four profile records built in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Builds four execution-statistics messages and profile records: record one is an add node on
    # cpu (op time 2, exec time 4), record two a multiply node on cpu (op time 3, exec time 17),
    # record three the same add node as one on cpu (op time 2, exec time 4), and record four the
    # same add node on gpu (op time 2, exec time 4). Stores them as attributes for the tests.
    node_1 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        op_start_rel_micros=3,
        op_end_rel_micros=5,
        all_end_rel_micros=4)
    self.profile_datum_1 = profiling.ProfileDatum(
        "cpu:0", node_1, "/foo/bar.py", 10, "func1", "Add")

    node_2 = step_stats_pb2.NodeExecStats(
        node_name="Mul/456",
        op_start_rel_micros=13,
        op_end_rel_micros=16,
        all_end_rel_micros=17)
    self.profile_datum_2 = profiling.ProfileDatum(
        "cpu:0", node_2, "/foo/bar.py", 11, "func1", "Mul")

    node_3 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        op_start_rel_micros=103,
        op_end_rel_micros=105,
        all_end_rel_micros=4)
    self.profile_datum_3 = profiling.ProfileDatum(
        "cpu:0", node_3, "/foo/bar.py", 12, "func1", "Add")

    node_4 = step_stats_pb2.NodeExecStats(
        node_name="Add/123",
        op_start_rel_micros=203,
        op_end_rel_micros=205,
        all_end_rel_micros=4)
    self.profile_datum_4 = profiling.ProfileDatum(
        "gpu:0", node_4, "/foo/bar.py", 13, "func1", "Add")

  def testAggregateProfileConstructorWorks(self):
    # [FUNCTION DOC]
    # Scenario: aggregate created from record one only. Asserts total op time 2, total execution
    # time 4, one distinct node and one execution.
    aggregate_data = profiling.AggregateProfile(self.profile_datum_1)

    self.assertEqual(2, aggregate_data.total_op_time)
    self.assertEqual(4, aggregate_data.total_exec_time)
    self.assertEqual(1, aggregate_data.node_count)
    self.assertEqual(1, aggregate_data.node_exec_count)

  def testAddToAggregateProfileWithDifferentNodeWorks(self):
    # [FUNCTION DOC]
    # Scenario: record one aggregated, then record two (different node) added. Asserts totals of
    # op time 5 and execution time 21, two distinct nodes and two executions.
    aggregate_data = profiling.AggregateProfile(self.profile_datum_1)
    aggregate_data.add(self.profile_datum_2)

    self.assertEqual(5, aggregate_data.total_op_time)
    self.assertEqual(21, aggregate_data.total_exec_time)
    self.assertEqual(2, aggregate_data.node_count)
    self.assertEqual(2, aggregate_data.node_exec_count)

  def testAddToAggregateProfileWithSameNodeWorks(self):
    # [FUNCTION DOC]
    # Scenario: records one and two aggregated, then record three (same node and device as one)
    # added. Asserts op time 7, execution time 25, two distinct nodes and three executions.
    aggregate_data = profiling.AggregateProfile(self.profile_datum_1)
    aggregate_data.add(self.profile_datum_2)
    aggregate_data.add(self.profile_datum_3)

    self.assertEqual(7, aggregate_data.total_op_time)
    self.assertEqual(25, aggregate_data.total_exec_time)
    self.assertEqual(2, aggregate_data.node_count)
    self.assertEqual(3, aggregate_data.node_exec_count)

  def testAddToAggregateProfileWithDifferentDeviceSameNodeWorks(self):
    # [FUNCTION DOC]
    # Scenario: record one aggregated, then record four (same node name on another device) added.
    # Asserts op time 4, execution time 8, two distinct nodes (device counts as part of the key)
    # and two executions.
    aggregate_data = profiling.AggregateProfile(self.profile_datum_1)
    aggregate_data.add(self.profile_datum_4)

    self.assertEqual(4, aggregate_data.total_op_time)
    self.assertEqual(8, aggregate_data.total_exec_time)
    self.assertEqual(2, aggregate_data.node_count)
    self.assertEqual(2, aggregate_data.node_exec_count)


if __name__ == "__main__":
  googletest.main()
