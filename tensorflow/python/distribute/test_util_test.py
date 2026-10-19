# [MODULE DOC]
# Tests for the tf.distribute test utilities: the gather helper (single and nested values) across
# several strategies, creation of logical CPU devices, and the topological-sort based
# sequential-execution assertion. Uses the module's own test main as entry point.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for test utilities."""

import sys

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
        ] + strategy_combinations.strategies_minus_tpu,
        mode=['eager']))
class GatherTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for the test-utility gather helper, run in eager mode over multi-worker
  # mirrored CPU/GPU strategies plus all non-TPU strategies. Each test receives a strategy.

  def testOne(self, strategy):
    # [FUNCTION DOC]
    # Scenario: a traced function returns a scalar float one on every replica. The strategy runs
    # it and the helper gathers the result. Expected: evaluating the gathered tensor equals a
    # list of ones, one per replica in the sync group.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function returning a scalar float one.
      return array_ops.ones((), dtypes.float32)

    results = test_util.gather(strategy, strategy.run(f))
    self.assertAllEqual(
        self.evaluate(results), [1.] * strategy.num_replicas_in_sync)

  def testNest(self, strategy):
    # [FUNCTION DOC]
    # Scenario: a traced function returns a mapping with a scalar one under one key and a list of
    # a scalar zero and a scalar one under another key. After running on the strategy and
    # gathering, each leaf is checked: the first key yields ones per replica, and the list
    # entries yield zeros and ones per replica respectively (the nested structure is preserved).

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function returning a nested mapping: a scalar one under one key and a list of a
      # scalar zero and a scalar one under the other key.
      return {
          'foo':
              array_ops.ones((), dtypes.float32),
          'bar': [
              array_ops.zeros((), dtypes.float32),
              array_ops.ones((), dtypes.float32),
          ]
      }

    results = test_util.gather(strategy, strategy.run(f))
    self.assertAllEqual(
        self.evaluate(results['foo']), [1.] * strategy.num_replicas_in_sync)
    self.assertAllEqual(
        self.evaluate(results['bar'][0]), [0.] * strategy.num_replicas_in_sync)
    self.assertAllEqual(
        self.evaluate(results['bar'][1]), [1.] * strategy.num_replicas_in_sync)


class LogicalDevicesTest(test.TestCase):
  # [CLASS DOC]
  # Tests for creating logical devices through the test-utility helper. No shared fixtures.

  def testLogicalCPUs(self):
    # [FUNCTION DOC]
    # Scenario: skipped on Python 3.8 with a GPU present (known segmentation fault). Otherwise
    # resets the eager context, requests at least three logical CPUs, then checks that the first
    # physical CPU has exactly three logical device configurations. Side effect: resets global
    # eager context and device configuration.
    # TODO(b/273484131): Causing segmentation fault.
    if (test.is_gpu_available() and sys.version_info.major == 3 and
        sys.version_info.minor == 8):
      self.skipTest('Causing segmentation fault in Python 3.8 / GPU')
    context._reset_context()
    test_util.set_logical_devices_to_at_least('CPU', 3)
    cpu_device = config.list_physical_devices('CPU')[0]
    self.assertLen(config.get_logical_device_configuration(cpu_device), 3)


class AssertSequentailExecutionTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: class name is misspelled (Sequentail); harmless.
  # Tests for the topological sort and sequential-execution assertion helpers.

  def test1(self):
    # [FUNCTION DOC]
    # Scenario: builds a traced function with named identity operations a, c, d, e, f where d
    # depends on a and c, e has a control dependency on an operation derived from a, and f
    # depends on c and e. Computes the topological order of all graph operations. Expected:
    # sequential execution holds for the pairs (a, d) and for (e, a, f), while an assertion error
    # is raised for (a, c), (f, a, c) and (d, e, a, c), since no dependency chain links them.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function defining named operations a through f with data and control dependencies
      # (e has a control dependency on a value derived from a); returns the outputs of d and f.
      a = array_ops.identity(1., name='a')
      b = a + 1
      c = array_ops.identity(2., name='c')
      d = array_ops.identity(a + c, name='d')
      with ops.control_dependencies([b]):
        e = array_ops.identity(3., name='e')
      f = array_ops.identity(c + e, name='f')
      return d, f

    graph = f.get_concrete_function().graph
    order = test_util.topological_sort_operations(graph.get_operations())
    a = graph.get_operation_by_name('a')
    c = graph.get_operation_by_name('c')
    d = graph.get_operation_by_name('d')
    e = graph.get_operation_by_name('e')
    f = graph.get_operation_by_name('f')
    test_util.assert_sequential_execution(order, [a, d])
    test_util.assert_sequential_execution(order, [e, a, f])
    with self.assertRaises(AssertionError):
      test_util.assert_sequential_execution(order, [a, c])
    with self.assertRaises(AssertionError):
      test_util.assert_sequential_execution(order, [f, a, c])
    with self.assertRaises(AssertionError):
      test_util.assert_sequential_execution(order, [d, e, a, c])


if __name__ == '__main__':
  test_util.main()
