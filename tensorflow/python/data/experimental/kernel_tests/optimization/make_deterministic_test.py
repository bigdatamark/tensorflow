# [MODULE DOC]
# Tests for the MakeDeterministic tf.data graph optimization. Under deterministic-ops mode, datasets
# containing stateful functions must be rewritten so parallel map/interleave/batch/prefetch run
# sequentially (ordered, repeatable output), while stateless pipelines keep their parallel ops.
# Exposes one test class and a standard test main entry point (enables v2 behavior).
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the `MakeDeterministic` optimization."""
import os
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import readers as reader_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class MakeDeterministicTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the MakeDeterministic optimization of tf.data. Each test builds a dataset
  # inside a deterministic-ops context and checks that output order/values are deterministic and
  # that expected transformation kernels (via assert_next) appear in the optimized graph.

  def _set_seed(self):
    # [FUNCTION DOC]
    # HELPER: seeds the random generator for the determinism tests.
    # Sets the global random seed to 1 so that graph-mode dataset ops that copy the seed do not
    # raise under determinism. Takes no arguments, returns nothing; mutates global seed state.
    # Set the seed, since in graph mode some non-random dataset ops call
    # tf.compat.v1.get_seed to copy the seed to a Defun. Calling get_seed raises
    # an error with determinism if no seed is set.
    # TODO(reedwm): Ensure such dataset ops do not raise an error when no seed
    # is set.
    random_seed.set_random_seed(1)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              use_function=[False, True], use_legacy_interleave=[False, True])))
  def test_stateful_ops_interleave(self, use_function, use_legacy_interleave):
    # [FUNCTION DOC]
    # Scenario: a stateful map function (increments a variable, returns element and variable value)
    # is used inside an interleave over range(5) with cycle length 5. Parameters: use_function
    # wraps the functions as tf functions; use_legacy_interleave picks the experimental
    # parallel_interleave versus the standard parallel interleave. Under deterministic ops the
    # interleave must run sequentially: expected output is five zeros paired with 1..5 followed by
    # five ones paired with 6..10.
    with test_util.deterministic_ops():

      v = variables.Variable(0.)

      def map_fn(x):
        # [FUNCTION DOC]
        # Stateful map body: adds one to the captured variable and returns the input element
        # together with the variable's current value.
        v.assign_add(1.)
        return (x, v.read_value())

      def interleave_fn(x):
        # [FUNCTION DOC]
        # Interleave body: ignores its input and returns a range(2) dataset mapped by map_fn.
        del x
        return dataset_ops.Dataset.range(2).map(map_fn)

      if use_function:
        map_fn = def_function.function(map_fn)
        interleave_fn = def_function.function(interleave_fn)

      dataset = dataset_ops.Dataset.range(5)
      if use_legacy_interleave:
        dataset = dataset.apply(
            interleave_ops.parallel_interleave(interleave_fn, cycle_length=5))
      else:
        dataset = dataset.interleave(
            interleave_fn, cycle_length=5, num_parallel_calls=3)
      self.evaluate(variables.global_variables_initializer())
      expected_output = list(zip([0] * 5 + [1] * 5, range(1, 11)))
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              use_function=[False, True])))
  def test_stateful_ops_map(self, use_function):
    # [FUNCTION DOC]
    # Scenario: a stateful map (variable increment) over range(5) requested with 5 parallel calls.
    # use_function optionally wraps the map function as a tf function. Deterministic ops force
    # sequential execution, so the output must be the pairs (i, i+1) in order.
    with test_util.deterministic_ops():

      v = variables.Variable(0.)

      def map_fn(x):
        # [FUNCTION DOC]
        # Stateful map body: increments the captured variable and returns the element with the
        # variable's new value.
        v.assign_add(1.)
        return (x, v.read_value())

      if use_function:
        map_fn = def_function.function(map_fn)

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.map(map_fn, num_parallel_calls=5)
      self.evaluate(variables.global_variables_initializer())
      expected_output = list(zip(range(0, 5), range(1, 6)))
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def test_stateful_ops_map_with_random_ops(self):
    # [FUNCTION DOC]
    # Scenario: a parallel map whose function adds a seeded random integer (0 or 1) to each
    # element. The graph is expected to contain a plain Map followed by a ParallelMap (checked with
    # assert_next). Each of the five outputs must be either i or i+1. Takes no parameters.
    with test_util.deterministic_ops():

      def map_fn(x):
        # [FUNCTION DOC]
        # Map body: returns the input plus a random int64 in the range zero to two (exclusive)
        # drawn with a fixed op seed.
        return x + random_ops.random_uniform(
            (), 0, 2, dtype=dtypes.int64, seed=1)

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.apply(testing.assert_next(["Map", "ParallelMap"]))
      dataset = dataset.map(map_fn, num_parallel_calls=5)
      get_next = self.getNext(dataset, requires_initialization=True)
      for i in range(5):
        self.assertIn(self.evaluate(get_next()), [i, i + 1])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(use_function=[False, True])))
  def test_stateful_ops_map_ignore_input(self, use_function):
    # [FUNCTION DOC]
    # Scenario: a parallel map over range(5) whose function ignores its input, mutates a variable
    # and returns the constant one. use_function optionally wraps it as a tf function. The test
    # asserts the output is five ones, i.e. the stateful map is still evaluated correctly.
    with test_util.deterministic_ops():

      v = variables.Variable(0.)

      def map_fn(x):
        # [FUNCTION DOC]
        # Map body: discards the input, increments the captured variable and returns constant one.
        del x
        v.assign_add(1.)
        return math_ops.constant_op.constant(1.)

      if use_function:
        map_fn = def_function.function(map_fn)

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.map(map_fn, num_parallel_calls=5)
      self.evaluate(variables.global_variables_initializer())
      expected_output = [1.] * 5
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(use_function=[False, True])))
  def test_stateful_ops_batch(self, use_function):
    # [FUNCTION DOC]
    # Scenario: a map reading a variable (stateful read) followed by a parallel batch of size 2
    # with 2 parallel calls. The graph is expected to have a Batch op next (assert_next). Output
    # must be batches [0,1], [2,3], [4] paired with zeros. use_function wraps the map function.
    with test_util.deterministic_ops():

      v = variables.Variable(0.)

      def map_fn(x):
        # [FUNCTION DOC]
        # Map body: returns the element together with the current variable value (no mutation).
        return (x, v.read_value())

      if use_function:
        map_fn = def_function.function(map_fn)

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.map(map_fn)
      dataset = dataset.apply(testing.assert_next(["Batch"]))
      dataset = dataset.batch(2, num_parallel_calls=2)
      self.evaluate(variables.global_variables_initializer())
      expected_output = [
          (np.array([0, 1]), np.array([0, 0])),
          (np.array([2, 3]), np.array([0, 0])),
          (np.array([4]), np.array([0])),
      ]
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              use_function=[False, True],
              use_legacy_map_and_batch=[False, True])))
  def test_stateful_ops_map_and_batch(self, use_function,
                                      use_legacy_map_and_batch):
    # [FUNCTION DOC]
    # Scenario: a stateful map (variable increment) with 5 parallel calls combined with batch size
    # 2, either via the legacy experimental map_and_batch or separate map and batch.
    # use_function wraps the map function. Deterministic ops force sequential execution, so the
    # batches are [0,1]/[1,2], [2,3]/[3,4], [4]/[5] in order.
    with test_util.deterministic_ops():

      v = variables.Variable(0.)

      def map_fn(x):
        # [FUNCTION DOC]
        # Stateful map body: increments the captured variable and returns element and new value.
        v.assign_add(1.)
        return (x, v.read_value())

      if use_function:
        map_fn = def_function.function(map_fn)

      dataset = dataset_ops.Dataset.range(5)
      if use_legacy_map_and_batch:
        dataset = dataset.apply(batching.map_and_batch(map_fn, 2,
                                                       num_parallel_calls=5))
      else:
        dataset = dataset.map(map_fn, num_parallel_calls=5)
        dataset = dataset.batch(2)
      self.evaluate(variables.global_variables_initializer())
      expected_output = [
          (np.array([0, 1]), np.array([1, 2])),
          (np.array([2, 3]), np.array([3, 4])),
          (np.array([4]), np.array([5])),
      ]
      self.assertDatasetProduces(
          dataset,
          expected_output=expected_output,
          requires_initialization=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              use_function=[False, True], use_legacy_interleave=[False, True])))
  def test_no_stateful_ops_interleave(self, use_function,
                                      use_legacy_interleave):
    # [FUNCTION DOC]
    # Scenario: a stateless interleave (each input yields range(2)) with cycle length 5 under
    # deterministic ops, after seeding. The parallel interleave kernel (legacy V2 or the standard
    # one) must be kept (assert_next), and output must be five zeros then five ones.
    self._set_seed()
    with test_util.deterministic_ops():

      def interleave_fn(x):
        # [FUNCTION DOC]
        # Interleave body: ignores its input and returns a range(2) dataset (stateless).
        del x
        return dataset_ops.Dataset.range(2)

      if use_function:
        interleave_fn = def_function.function(interleave_fn)

      dataset = dataset_ops.Dataset.range(5)
      if use_legacy_interleave:
        dataset = dataset.apply(
            testing.assert_next(["LegacyParallelInterleaveV2"]))
        dataset = dataset.apply(
            interleave_ops.parallel_interleave(interleave_fn, cycle_length=5))
      else:
        dataset = dataset.apply(testing.assert_next(["ParallelInterleave"]))
        dataset = dataset.interleave(
            interleave_fn, cycle_length=5, num_parallel_calls=3)
      self.evaluate(variables.global_variables_initializer())
      self.assertDatasetProduces(dataset, expected_output=[0] * 5 + [1] * 5)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(use_function=[False, True])))
  def test_no_stateful_ops_map(self, use_function):
    # [FUNCTION DOC]
    # Scenario: a stateless map (add one) with 5 parallel calls under deterministic ops. The
    # ParallelMap kernel must be kept (assert_next) and output must be 1..5. use_function
    # optionally wraps the map function as a tf function.
    self._set_seed()
    with test_util.deterministic_ops():
      def map_fn(x):
        # [FUNCTION DOC]
        # Stateless map body: returns the input plus one.
        return x + 1

      if use_function:
        map_fn = def_function.function(map_fn)

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.apply(testing.assert_next(["ParallelMap"]))
      dataset = dataset.map(map_fn, num_parallel_calls=5)
      self.evaluate(variables.global_variables_initializer())
      expected_output = range(1, 6)
      self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              use_function=[False, True], use_control_flow=[False, True])))
  def test_text_line_dataset(self, use_function, use_control_flow):
    # [FUNCTION DOC]
    # Scenario: three temp text files (numbers 1-3, 4-6, 7-9) are read through a parallel
    # interleave of text line datasets, cycle length 3, under deterministic ops. The interleave
    # function optionally contains control flow (a conditional, one branch never taken) and is
    # optionally wrapped as a tf function. Expects a ParallelInterleave kernel (assert_next) and
    # round-robin output order "1","4","7","2","5","8","3","6","9".
    self._set_seed()
    with test_util.deterministic_ops():

      def write_nums_to_file(filename, numbers):
        # [FUNCTION DOC]
        # HELPER: creates test input files.
        # Writes the numbers, one per line, to a file of the given name in the test temp
        # directory and returns the full path of that file.
        path = os.path.join(self.get_temp_dir(), filename)
        with open(path, "w") as f:
          f.write("\n".join(str(n) for n in numbers))
        return path

      f1 = write_nums_to_file("f1", (1, 2, 3))
      f2 = write_nums_to_file("f2", (4, 5, 6))
      f3 = write_nums_to_file("f3", (7, 8, 9))

      if use_control_flow:
        def interleave_fn(filename):
          # [FUNCTION DOC]
          # Interleave body with control flow: a conditional on whether the filename equals a
          # fixed string picks between two text line datasets; the true branch (filename plus
          # suffix) is never taken in practice, the false branch reads the given file.
          # Test function that uses control flow. The True branch is never taken
          concat = string_ops.string_join([filename, "abc"])
          return cond.cond(
              math_ops.equal(filename, "abc"),
              lambda: reader_ops.TextLineDataset(concat),
              lambda: reader_ops.TextLineDataset(filename))
      else:
        def interleave_fn(filename):
          # [FUNCTION DOC]
          # Interleave body without control flow: returns a text line dataset for the file name.
          return reader_ops.TextLineDataset(filename)

      if use_function:
        interleave_fn = def_function.function(interleave_fn)

      dataset = dataset_ops.Dataset.from_tensor_slices([f1, f2, f3])
      dataset = dataset.apply(testing.assert_next(["ParallelInterleave"]))
      dataset = dataset.interleave(
          interleave_fn, cycle_length=3, num_parallel_calls=3)

      self.assertDatasetProduces(
          dataset,
          expected_output=["1", "4", "7", "2", "5", "8", "3", "6", "9"])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              local_determinism=[None, True, False],
              global_determinism=[True, False])))
  def test_deterministic_attribute(self, local_determinism, global_determinism):
    # [FUNCTION DOC]
    # Scenario: a parallel map (2 calls) over range(100) whose function delays element zero
    # through a non-stateful py_func sleep. The per-map deterministic argument (none/true/false)
    # and the global options deterministic setting (true/false) are varied; under deterministic ops
    # the output must always be exactly the ordered range(100).
    self._set_seed()
    with test_util.deterministic_ops():

      def sleep(x):
        # [FUNCTION DOC]
        # Sleeps for 0.1 seconds and returns its input unchanged (used to perturb ordering).
        time.sleep(0.1)
        return x

      def map_function(x):
        # [FUNCTION DOC]
        # Map body: for element zero it calls the sleeping py_func (non-stateful) to delay it;
        # all other elements pass through unchanged.
        if math_ops.equal(x, 0):
          return script_ops.py_func(sleep, [x], x.dtype, stateful=False)
        else:
          return x

      dataset = dataset_ops.Dataset.range(100)
      dataset = dataset.map(
          map_function, num_parallel_calls=2, deterministic=local_determinism)
      opts = options_lib.Options()
      opts.deterministic = global_determinism
      dataset = dataset.with_options(opts)

      self.assertDatasetProduces(dataset, expected_output=range(100))

  @combinations.generate(test_base.default_test_combinations())
  def test_rewrite_prefetch(self):
    # [FUNCTION DOC]
    # Scenario: map that stores each element in a variable, followed by prefetch(5), under
    # deterministic ops. Prefetch must be rewritten to not run ahead: the variable stays -1 before
    # the first fetch, equals the last fetched element after each get-next (0 then 1).
    with test_util.deterministic_ops():
      v = variables.Variable(-1, dtype=dtypes.int64)

      def map_fn(x):
        # [FUNCTION DOC]
        # Map body: assigns the element to the captured variable and returns it.
        v.assign(x)
        return x

      dataset = dataset_ops.Dataset.range(5)
      dataset = dataset.map(map_fn)
      dataset = dataset.prefetch(5)
      self.evaluate(variables.global_variables_initializer())
      get_next = self.getNext(dataset, requires_initialization=True)
      self.assertEqual(self.evaluate(v), -1)
      self.assertEqual(self.evaluate(get_next()), 0)
      time.sleep(0.01)
      self.assertEqual(self.evaluate(v), 0)
      self.assertEqual(self.evaluate(get_next()), 1)
      time.sleep(0.01)
      self.assertEqual(self.evaluate(v), 1)

  @combinations.generate(test_base.default_test_combinations())
  def test_no_determinism(self):
    # [FUNCTION DOC]
    # Scenario: op determinism is explicitly disabled (global config side effect, not restored),
    # and a parallel interleave with a stateful function (assigns a variable) is built. The
    # ParallelInterleave kernel must be kept (assert_next) and output is five zeros then five ones.
    config.disable_op_determinism()
    v = variables.Variable(0.)

    def interleave_fn(x):
      # [FUNCTION DOC]
      # Interleave body: ignores its input, assigns one to the captured variable (stateful) and
      # returns a range(2) dataset.
      del x
      v.assign(1.)
      return dataset_ops.Dataset.range(2)

    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.apply(testing.assert_next(["ParallelInterleave"]))
    dataset = dataset.interleave(
        interleave_fn, cycle_length=5, num_parallel_calls=3)
    self.evaluate(variables.global_variables_initializer())
    expected_output = [0] * 5 + [1] * 5
    self.assertDatasetProduces(
        dataset, expected_output=expected_output, requires_initialization=True)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
