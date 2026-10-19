# [MODULE DOC]
# Tests for the parameter-server value wrappers (AggregatingVariable and CachingVariable).
# Contains a shared helper that checks asynchronous checkpoint save and restore, and a test
# class running on central storage strategy with two GPUs in graph and eager modes.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the distributed values library."""

import os

from absl.testing import parameterized

from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_lib


def async_checkpoint_test_helper(test_case, x):
  # [FUNCTION DOC]
  # HELPER: shared check for the tests in this file.
  # Verifies asynchronous checkpointing of a variable. Assigns 123, saves with the async option,
  # changes to 234, restores and expects 123. Then repeats with 345 and 456 to exercise the case
  # where a CPU copy already exists. Inputs: test_case (test instance used for evaluation, temp
  # directory and assertions) and x (variable to checkpoint). Writes checkpoint files to the
  # temporary directory. Returns nothing.
  # First assign an initial value 123 and save it to checkpoint.
  test_case.evaluate(x.assign(123.0))
  checkpoint = trackable_utils.Checkpoint(x=x)
  ckpt_options = checkpoint_options.CheckpointOptions(
      experimental_enable_async_checkpoint=True)
  prefix = os.path.join(test_case.get_temp_dir(), "ckpt")
  save_path = checkpoint.save(prefix, options=ckpt_options)

  # Then we modify the value to 234, restore from checkpoint, and see that the
  # value goes back to 123.
  test_case.evaluate(x.assign(234.0))
  test_case.assertNotAllClose(123.0, x.read_value())
  checkpoint.restore(save_path).assert_consumed().run_restore_ops()
  test_case.assertEqual(test_case.evaluate(x), 123.0)

  # Another round of saving/restoring to ensure that the logic of
  # _copy_trackable_to_cpu works when a copy is already created in object_map.
  test_case.evaluate(x.assign(345.0))
  save_path = checkpoint.save(prefix, options=ckpt_options)
  test_case.evaluate(x.assign(456.0))
  test_case.assertNotAllClose(345.0, x.read_value())
  checkpoint.restore(save_path).assert_consumed().run_restore_ops()
  test_case.assertEqual(test_case.evaluate(x), 345.0)


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.central_storage_strategy_with_two_gpus
        ],
        mode=["graph", "eager"]))
class AggregatingVariableTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests AggregatingVariable and CachingVariable behavior under central storage strategy with two
  # GPUs (graph and eager modes); the strategy is supplied to each test as an argument.

  def testAssignOutOfScope(self, distribution):
    # [FUNCTION DOC]
    # Creates a variable in the strategy scope, expects it to be an aggregating variable, assigns 3
    # outside the scope and expects both the wrapper and the wrapped variable to read 3.
    with distribution.scope():
      aggregating = variables_lib.Variable(1.)
    self.assertIsInstance(aggregating, ps_values.AggregatingVariable)
    self.evaluate(aggregating.assign(3.))
    self.assertEqual(self.evaluate(aggregating.read_value()), 3.)
    self.assertEqual(self.evaluate(aggregating._v.read_value()), 3.)

  def testAssignAdd(self, distribution):
    # [FUNCTION DOC]
    # Creates a mean-aggregated variable with value 1, runs a function adding 2 on each replica
    # through the strategy, and expects the single local result to be 3.
    with distribution.scope():
      v = variable_v1.VariableV1(
          1, aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def assign():
      # [FUNCTION DOC]
      # Function run per replica: adds 2 to the variable and returns the result.
      return v.assign_add(2)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(assign)))
    self.assertAllEqual([3], per_replica_results)

  def testAsyncCheckpointAggregatingVariable(self, distribution):
    # [FUNCTION DOC]
    # Creates a variable in strategy scope, expects an aggregating variable, initializes it and runs
    # the shared async checkpoint round-trip check.
    with self.test_session():
      with distribution.scope():
        x = variables_lib.Variable(1.)
      self.assertIsInstance(x, ps_values.AggregatingVariable)
      self.evaluate(x.initializer)

      async_checkpoint_test_helper(self, x)

  def testAsyncCheckpointCachingVariable(self, distribution):
    # [FUNCTION DOC]
    # Ignores the strategy. Wraps a plain variable in a caching variable, initializes it and runs
    # the shared async checkpoint round-trip check.
    del distribution
    with self.test_session():
      v = variables_lib.Variable(1.)
      x = ps_values.CachingVariable(v)
      self.assertIsInstance(x, ps_values.CachingVariable)
      self.evaluate(x.initializer)

      async_checkpoint_test_helper(self, x)

if __name__ == "__main__":
  test.main()
