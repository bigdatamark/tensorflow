# [MODULE DOC]
# Unit test for the benchmark reporting facility. Defines one test class that is both a test case
# and a TensorFlow benchmark, and checks that a reported benchmark is serialized to a file with
# the expected contents. Runs the test main when executed as a script. Not a helper module.

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
"""Test for the tf.test.benchmark."""

import os
from google.protobuf import json_format
from tensorflow.core.util import test_log_pb2
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

class BenchmarkTest(test.TestCase, benchmark.TensorFlowBenchmark):
  # [CLASS DOC]
  # Test case covering the benchmark report method. Inherits from both the TensorFlow test case
  # and the TensorFlow benchmark base class, so it is also registered as a benchmark class
  # (it defines no benchmark-named methods, so nothing runs as a benchmark).

  def testReportBenchmark(self):
    # [FUNCTION DOC]
    # Scenario: sets the report-file-prefix environment variable to a temp directory (side effect:
    # modifies the process environment and is never restored), deletes any stale output file, and
    # reports a benchmark with 2000 iterations, wall time 1000, an explicit name and two metrics
    # (one with only a minimum, one with minimum and maximum). Reads back the file the report
    # produced, parses it as a benchmark-entries proto, converts to a dictionary and deletes the
    # file. Asserts the iteration count is 2000 and that the whole entry equals the expected
    # dictionary: class-qualified name, iterations as a string, wall time, zero cpu time and
    # throughput, empty extras, and both metrics with their bounds.
    output_dir = self.get_temp_dir() + os.path.sep
    os.environ['TEST_REPORT_FILE_PREFIX'] = output_dir
    proto_file_path = os.path.join(output_dir,
                                   'BenchmarkTest.testReportBenchmark')
    if os.path.exists(proto_file_path):
      os.remove(proto_file_path)

    self.report_benchmark(
        iters=2000,
        wall_time=1000,
        name='testReportBenchmark',
        metrics=[{'name': 'metric_name_1', 'value': 0, 'min_value': 1},
                 {'name': 'metric_name_2', 'value': 90, 'min_value': 0,
                  'max_value': 95}])

    with open(proto_file_path, 'rb') as f:
      benchmark_entries = test_log_pb2.BenchmarkEntries()
      benchmark_entries.ParseFromString(f.read())

      actual_result = json_format.MessageToDict(
          benchmark_entries, preserving_proto_field_name=True,
          always_print_fields_with_no_presence=True)['entry'][0]
    os.remove(proto_file_path)

    expected_result = {
        'name': 'BenchmarkTest.testReportBenchmark',
        # google.protobuf.json_format.MessageToDict() will convert
        # int64 field to string.
        'iters': '2000',
        'wall_time': 1000,
        'cpu_time': 0,
        'throughput': 0,
        'extras': {},
        'metrics': [
            {
                'name': 'metric_name_1',
                'value': 0,
                'min_value': 1
            },
            {
                'name': 'metric_name_2',
                'value': 90,
                'min_value': 0,
                'max_value': 95
            }
        ]
    }

    self.assertEqual(2000, benchmark_entries.entry[0].iters)
    self.assertDictEqual(expected_result, actual_result)

if __name__ == '__main__':
  test.main()
