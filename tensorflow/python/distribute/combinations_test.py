# [MODULE DOC]
# Unit tests for the distribute test-combinations module. They cover cluster parameter
# injection (defaults, from a named distribution, absent strategy, multiple strategies),
# multi-worker execution in subprocesses (including expected-failure checks proving tests
# really run in workers), environment propagation to workers, running a function both as
# a compiled function and eagerly, reloading the module with an empty argument vector, and
# GPU sharing among workers. Run through the standard distribute test entry point.
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
"""Tests for tensorflow.python.distribute.combinations."""

import importlib
import os
import sys
import unittest

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.cluster_resolver import tfconfig_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations as framework_combinations
from tensorflow.python.platform import test


class ClusterCombinationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the cluster combination: how cluster parameters (chief flag, worker count) are
  # derived from named distributions, defaults, and conflicts. Uses the framework-level
  # generator directly so the cluster parameters reach the test method.
  # For this test we need to use `framework.test_combinations` because our
  # `generate` eats the cluster parameters.
  #
  # Note that we don't have a standalone combination for ClusterParameters, so
  # we should use GPUCombination which contains it.

  @framework_combinations.generate(  # pylint: disable=redundant-keyword-arg
      framework_combinations.combine(distribution=[
          combinations.NamedDistribution(
              "HasClusterParams", lambda: None, has_chief=True, num_workers=2),
      ]),
      test_combinations=(combinations.ClusterCombination(),))
  def testClusterParams(self, distribution, has_chief, num_workers):
    # [FUNCTION DOC]
    # Scenario: a distribution declares a chief and two workers. Expects the injected
    # cluster parameters to report a chief and two workers.
    self.assertTrue(has_chief)
    self.assertEqual(num_workers, 2)

  @framework_combinations.generate(  # pylint: disable=redundant-keyword-arg
      framework_combinations.combine(distribution=[
          combinations.NamedDistribution("NoClusterParams", lambda: None),
      ]),
      test_combinations=(combinations.ClusterCombination(),))
  def testClusterParamsHasDefault(self, distribution, has_chief, num_workers):
    # [FUNCTION DOC]
    # Scenario: a distribution with no cluster settings. Expects defaults: no chief and
    # one worker.
    self.assertFalse(has_chief)
    self.assertEqual(num_workers, 1)

  @framework_combinations.generate(  # pylint: disable=redundant-keyword-arg
      framework_combinations.combine(v=1),
      test_combinations=(combinations.ClusterCombination(),))
  def testClusterParamsNoStrategy(self, v, has_chief, num_workers):
    # [FUNCTION DOC]
    # Scenario: the combination has no distribution at all (just a plain value). Expects
    # the cluster parameters to still be provided with defaults: no chief, one worker.
    self.assertFalse(has_chief)
    self.assertEqual(num_workers, 1)

  @framework_combinations.generate(  # pylint: disable=redundant-keyword-arg
      framework_combinations.combine(distribution=[
          combinations.NamedDistribution(
              "WithClusterParams", lambda: None, has_chief=True, num_workers=2),
          combinations.NamedDistribution("WithoutClusterParams", lambda: None),
      ]),
      test_combinations=(combinations.ClusterCombination(),))
  def testClusterParamsAreOptional(self, distribution):
    # [FUNCTION DOC]
    # Scenario: two distributions (one with cluster settings, one without) across
    # combinations, and the test does not request cluster parameters. Passes as long as
    # the combination machinery raises no error; nothing is asserted.
    # If combinations library doesn't raise an exception, the test is passed.
    pass

  @framework_combinations.generate(  # pylint: disable=redundant-keyword-arg
      framework_combinations.combine(
          ds1=combinations.NamedDistribution(
              "Strategy1", lambda: None, has_chief=True, num_workers=0),
          ds2=combinations.NamedDistribution(
              "Strategy2", lambda: None, has_chief=False, num_workers=1),
          ds3=combinations.NamedDistribution(
              "Strategy3", lambda: None, has_chief=True, num_workers=0),
      ),
      test_combinations=(combinations.ClusterCombination(),))
  def testMultipleDistributionSingleWorker(self, ds1, ds2, ds3):
    # [FUNCTION DOC]
    # Scenario: three distributions in one combination whose total worker counts are each
    # at most one (chief counted). Multiple distributions are allowed in that case, so
    # passing means no error is raised; nothing is asserted.
    # If combinations library doesn't raise an exception, the test is passed.
    pass

  @combinations.generate(combinations.combine(num_workers=2,))
  def testUseWithoutStrategy(self):
    # [FUNCTION DOC]
    # Scenario: the distributed generator with two workers and no strategy. The test runs
    # in worker subprocesses; asserts the cluster configuration environment variable is not
    # the empty string (it is unset or set) as an approximation that cluster config exists.
    # There's no perfect way to check if the test runs in a subprocess. We
    # approximate by checking the presence of TF_CONFIG, which is normally not
    # set to the main process.
    self.assertNotEqual(os.getenv("TF_CONFIG"), "")


@combinations.generate(combinations.combine(num_workers=2))
class ClusterCombinationTestEnvTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that the shared test environment set in the main process reaches two worker
  # processes and cannot be modified there. Whole class is generated with two workers.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture run in both main and worker processes: calls the parent setup and, in the
    # main process only, sets the data-service dispatcher address on the environment to a
    # local host name.
    # Note that test case fixtures are executed in both the main process and
    # worker processes.
    super().setUp()
    if combinations.in_main_process():
      combinations.env().tf_data_service_dispatcher = "localhost"

  def testTfDataServiceDispatcher(self):
    # [FUNCTION DOC]
    # Asserts the dispatcher address set in setup is visible in the process running the
    # test (worker, where it was propagated).
    self.assertEqual(combinations.env().tf_data_service_dispatcher, "localhost")

  def testUpdateEnvInWorker(self):
    # [FUNCTION DOC]
    # Asserts that assigning to the environment from a worker process raises a value error.
    with self.assertRaises(ValueError):
      combinations.env().tf_data_service_dispatcher = "localhost"


# unittest.expectedFailure doesn't work with parameterized test methods, so we
# have to decorate the class instead.
@unittest.expectedFailure
class ClusterParametersShouldFailTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Expected-failure test class: combining two multi-worker distributions in one
  # combination must make the combination machinery raise, so the class as a whole is
  # expected to fail.

  @framework_combinations.generate(  # pylint: disable=redundant-keyword-arg
      framework_combinations.combine(
          ds1=combinations.NamedDistribution(
              "Strategy1", lambda: None, has_chief=True, num_workers=2),
          ds2=combinations.NamedDistribution(
              "Strategy2", lambda: None, has_chief=True, num_workers=2),
      ),
      test_combinations=(combinations.ClusterCombination(),))
  def testMultipleDistributionMultiWorker(self, ds1, ds2):
    # [FUNCTION DOC]
    # Scenario: two distributions each with a chief and two workers. The library should
    # raise (hence the expected failure on the class); the body does nothing.
    # combinations library should raise an exception.
    pass


# Tests that we *actually* run the test method in multiple workers instead of
# just passing silently. More importantly, it verifies that the test can fail.
# Note that unittest.expectedFailure doesn't work with parameterized test
# methods, so we have to decorate the class instead.
@unittest.expectedFailure
class CombinationsExpectedFailureTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Expected-failure test class proving multi-worker test methods really run in worker
  # processes and that failures there propagate to the main process.

  @combinations.generate(
      combinations.combine(distribution=[
          combinations.NamedDistribution(
              "OneChiefOneWorker", lambda: None, has_chief=True, num_workers=1),
          combinations.NamedDistribution(
              "TwoWorkers", lambda: None, has_chief=False, num_workers=2),
      ]))
  def testMultiWorkerCanFail(self, distribution):
    # [FUNCTION DOC]
    # Scenario: two distributions (chief plus one worker; two workers). Builds a cluster
    # resolver from the cluster configuration and asserts its task id is nothing. In worker
    # processes the task id is set, so the assertion fails by design, which makes the
    # expected-failure class pass.
    resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    # This should fail.
    self.assertIsNone(resolver.task_id)


# Tests that we *actually* run the test method in multiple workers instead of
# just passing silently. More importantly, it verifies that the test can fail.
# Note that unittest.expectedFailure doesn't work with parameterized test
# methods, so we have to decorate the class instead.
@unittest.expectedFailure
@combinations.generate(
    combinations.combine(distribution=[
        combinations.NamedDistribution(
            "OneChiefOneWorker", lambda: None, has_chief=True, num_workers=1),
        combinations.NamedDistribution(
            "TwoWorkers", lambda: None, has_chief=False, num_workers=2),
    ]))
class CombinationsOnClassMultiWorkerExpectedFailureTest(test.TestCase,
                                                        parameterized.TestCase):
  # [CLASS DOC]
  # Same as the method-level expected-failure check but the generator decorates the whole
  # class, verifying class-level multi-worker execution and failure propagation.

  def test(self, distribution):
    # [FUNCTION DOC]
    # Builds a cluster resolver from the cluster configuration and asserts its task id is
    # nothing; fails by design inside worker processes, making the expected failure hold.
    resolver = tfconfig_cluster_resolver.TFConfigClusterResolver()
    # This should fail.
    self.assertIsNone(resolver.task_id)


class TfFunctionTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the named objects that let a test run a function both as a compiled function and
  # eagerly.

  @combinations.generate(
      combinations.combine(
          tf_function_1=combinations.tf_function,
          tf_function_2=combinations.no_tf_function,
          mode="eager",
      ))
  def testFunc(self, tf_function_1, tf_function_2):
    # [FUNCTION DOC]
    # Scenario (eager mode): one function is wrapped with the compiled-function option and
    # another with the pass-through option. Asserts the first runs in graph (non-eager)
    # mode and the second runs eagerly, then calls both.

    @tf_function_1
    def foo():
      # [FUNCTION DOC]
      # Inner function wrapped by the first option; asserts that it is not executing
      # eagerly.
      self.assertFalse(context.executing_eagerly())

    @tf_function_2
    def bar():
      # [FUNCTION DOC]
      # Inner function wrapped by the second option; asserts that it is executing eagerly.
      self.assertTrue(context.executing_eagerly())

    foo()
    bar()


class ModuleInitializingTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that the combinations module can be (re)loaded under unusual process settings.

  def testSysArgvClearedIsFine(self):
    # [FUNCTION DOC]
    # Saves the program argument list, empties it, reloads the combinations module (which
    # reads it at import time), then restores the saved list. Passes if reloading raises no
    # error. Note the restore is not in a cleanup block, so a failed reload leaves the
    # argument list empty.
    original_argv = list(sys.argv)
    sys.argv.clear()
    importlib.reload(combinations)
    sys.argv = original_argv


class ShareGPUTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests whether the two workers share all physical GPUs (default and explicit) or split
  # them evenly. Skipped unless the machine has exactly two or four GPUs.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the parent setup and, in the main process, skips the test unless the
    # recorded physical GPU count is two or four.
    super().setUp()
    if combinations.in_main_process():
      num_gpus = combinations.env().total_phsyical_gpus
      if num_gpus != 2 and num_gpus != 4:
        self.skipTest("requires 2 or 4 GPUs")

  # Test cases are annotated with required_gpus only for them to run in gpu
  # targets, otherwise they will be skipped.

  @combinations.generate(
      combinations.combine(num_workers=2, required_gpus=1, share_gpu=True))
  def testShareGPU(self):
    # [FUNCTION DOC]
    # Two workers with GPU sharing explicitly enabled: asserts each worker sees all
    # physical GPUs.
    self.assertLen(context.context().list_physical_devices("GPU"),
                   combinations.env().total_phsyical_gpus)

  @combinations.generate(combinations.combine(num_workers=2, required_gpus=1))
  def testShareGPUByDefault(self):
    # [FUNCTION DOC]
    # Two workers with default sharing: asserts each worker sees all physical GPUs.
    self.assertLen(context.context().list_physical_devices("GPU"),
                   combinations.env().total_phsyical_gpus)

  @combinations.generate(
      combinations.combine(num_workers=2, required_gpus=1, share_gpu=False))
  def testNotShareGPU(self):
    # [FUNCTION DOC]
    # Two workers with GPU sharing disabled: asserts each worker sees half of the physical
    # GPUs (compared against the total divided by two).
    self.assertLen(context.context().list_physical_devices("GPU"),
                   combinations.env().total_phsyical_gpus / 2)


if __name__ == "__main__":
  test_util.main()
