# [MODULE DOC]
# Tests for the MapFusion tf.data optimization, which merges consecutive map transformations into
# one (parallel) map. Covers combinations of scalar and tuple map chains with different parallelism
# and determinism settings, long chains, mixing sequential and parallel maps, control inputs,
# stateful ops (errors must still surface) and captured inputs (no fusion). Contains a module-level
# combination generator, one test class and a standard test main entry point.
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
"""Tests for the `MapFusion` optimization."""
import functools

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _test_combinations():
  # [FUNCTION DOC]
  # HELPER: builds the parameter combinations for the map fusion test.
  # Takes no arguments. Creates named chains of two or three scalar map functions (all ordered
  # choices of identity, increment and increment-and-square) and two chains of tuple-producing
  # functions. Returns a list of parameter dictionaries each holding a named object wrapping the
  # chain of functions.
  cases = []

  identity = lambda x: x
  increment = lambda x: x + 1

  def increment_and_square(x):
    # [FUNCTION DOC]
    # Returns the square of the input plus one.
    y = x + 1
    return y * y

  functions = [identity, increment, increment_and_square]

  for i, x in enumerate(functions):
    for j, y in enumerate(functions):
      cases.append(("Scalar{}{}".format(i, j), [x, y]))
      for k, z in enumerate(functions):
        cases.append(("Scalar{}{}{}".format(i, j, k), [x, y, z]))

  with_42 = lambda x: (x, 42)
  swap = lambda x, y: (y, x)

  cases.append(("Tuple1", [with_42, swap]))
  cases.append(("Tuple2", [with_42, swap, swap]))

  def reduce_fn(x, y):
    # [FUNCTION DOC]
    # HELPER: reducer used to fold named chains into one combination list.
    # Takes the accumulated list x and a case y (name, functions); returns x extended with a
    # combination holding the functions wrapped in a named object.
    name, functions = y
    return x + combinations.combine(
        functions=combinations.NamedObject(name, functions)
    )

  return functools.reduce(reduce_fn, cases, [])


class MapFusionTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the map fusion graph optimization of tf.data.

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          _test_combinations(),
          combinations.combine(
              num_parallel_calls=[None, 2, dataset_ops.AUTOTUNE]
          ),
          combinations.combine(deterministic=[None, True, False]),
      )
  )
  def testMapFusion(self, functions, num_parallel_calls, deterministic):
    # [FUNCTION DOC]
    # Scenario: range(5) passed through a chain of maps (the given functions) with the given
    # parallelism (none, 2 or autotune) and determinism (none, true, false), then cached, with
    # only map_fusion enabled. assert_next expects a single Map (sequential), a single ParallelMap
    # (autotune) or two ParallelMap (fixed parallelism, not fused) before the cache. Output must
    # equal the functions applied in order (tuples unpacked as multiple arguments); order is
    # ignored when parallel and deterministic is false.
    dataset = dataset_ops.Dataset.range(5)
    if num_parallel_calls is None:
      dataset = dataset.apply(testing.assert_next(["Map", "MemoryCacheImpl"]))
    elif num_parallel_calls in [dataset_ops.AUTOTUNE]:
      # TODO(b/148614504): Support fusion of parallel maps with
      # non-AUTOTUNE value.
      dataset = dataset.apply(
          testing.assert_next(["ParallelMap", "MemoryCacheImpl"])
      )
    else:
      dataset = dataset.apply(
          testing.assert_next(["ParallelMap", "ParallelMap"])
      )

    for function in functions:
      dataset = dataset.map(
          function,
          num_parallel_calls=num_parallel_calls,
          deterministic=deterministic,
      )

    dataset = dataset.cache()
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_fusion = True
    dataset = dataset.with_options(options)
    expected_output = []
    for x in range(5):
      r = x
      for function in functions:
        if isinstance(r, tuple):
          r = function(*r)  # Pass tuple as multiple arguments.
        else:
          r = function(r)
      expected_output.append(r)

    nondeterministic_ordering = (
        num_parallel_calls is not None and deterministic is False  # pylint: disable=g-bool-id-comparison
    )
    self.assertDatasetProduces(
        dataset,
        expected_output=expected_output,
        assert_items_equal=nondeterministic_ordering,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testMapFusionLongMapChain(self):
    # [FUNCTION DOC]
    # Scenario: fifty successive autotuned parallel maps each doubling the value, then cache, with
    # map_fusion enabled. Expects a single ParallelMap before the cache and output equal to each
    # of 0..4 multiplied by two to the fiftieth power (order ignored).
    n = 5
    dataset = dataset_ops.Dataset.range(n)
    dataset = dataset.apply(
        testing.assert_next(["ParallelMap", "MemoryCacheImpl"])
    )

    k = 50
    for _ in range(k):
      dataset = dataset.map(
          lambda x: 2 * x,
          num_parallel_calls=dataset_ops.AUTOTUNE,
      )

    dataset = dataset.cache()
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_fusion = True
    dataset = dataset.with_options(options)

    self.assertDatasetProduces(
        dataset,
        expected_output=[x * 2**k for x in range(n)],
        assert_items_equal=True,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testMixedMapAndParallelMap(self):
    # [FUNCTION DOC]
    # Scenario: three doubling maps, the middle one sequential and the others autotuned parallel,
    # with map_parallelization and map_fusion enabled. The sequential map should first be
    # parallelized, so all three fuse into one ParallelMap before the cache. Output is each of
    # 0..4 times eight (order ignored).
    n = 5
    dataset = dataset_ops.Dataset.range(n)
    dataset = dataset.apply(
        testing.assert_next(["ParallelMap", "MemoryCacheImpl"])
    )

    # These 3 maps should be merged because the non-parallel Map should be
    # parallelized by `map_parallelization` before `map_fusion` is applied.
    dataset = dataset.map(lambda x: 2 * x,
                          num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.map(lambda x: 2 * x)
    dataset = dataset.map(lambda x: 2 * x,
                          num_parallel_calls=dataset_ops.AUTOTUNE)

    dataset = dataset.cache()
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_fusion = True
    dataset = dataset.with_options(options)

    self.assertDatasetProduces(
        dataset,
        expected_output=[x * 2**3 for x in range(n)],
        assert_items_equal=True,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testControlInputs(self):
    # [FUNCTION DOC]
    # Scenario: two autotuned parallel maps whose function has a control dependency on a type
    # assertion and doubles the value; map_fusion enabled. Expects one fused ParallelMap before the
    # cache and output of each of 0..4 times four (order ignored).
    def f(x):
      # [FUNCTION DOC]
      # Map body: asserts under a control dependency that the input is 64-bit integer, then
      # returns twice the input.
      with ops.control_dependencies([check_ops.assert_type(x, dtypes.int64)]):
        return 2 * x

    n = 5
    dataset = dataset_ops.Dataset.range(n)
    dataset = dataset.apply(
        testing.assert_next(["ParallelMap", "MemoryCacheImpl"])
    )
    dataset = dataset.map(f, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.map(f, num_parallel_calls=dataset_ops.AUTOTUNE)

    dataset = dataset.cache()
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_fusion = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(
        dataset,
        expected_output=[x * 4 for x in range(n)],
        assert_items_equal=True,
    )

  @combinations.generate(test_base.default_test_combinations())
  def testStatefulness(self):
    # [FUNCTION DOC]
    # Scenario: three autotuned parallel maps (identity, a function asserting the input is
    # negative, identity) are fused (single ParallelMap expected before the cache). Fetching the
    # first element must raise an invalid-argument error containing "assertion failed", showing
    # the stateful assertion is preserved by fusion.
    def f(x):
      # [FUNCTION DOC]
      # Map body: returns the input guarded by an assertion that it is negative (fails for the
      # non-negative range values).
      return control_flow_ops.with_dependencies(
          [check_ops.assert_negative(x)], x
      )

    dataset = dataset_ops.Dataset.range(5)
    dataset = dataset.apply(
        testing.assert_next(["ParallelMap", "MemoryCacheImpl"])
    )
    dataset = dataset.map(lambda x: x, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.map(f, num_parallel_calls=dataset_ops.AUTOTUNE)
    dataset = dataset.map(lambda x: x, num_parallel_calls=dataset_ops.AUTOTUNE)

    dataset = dataset.cache()
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_fusion = True
    dataset = dataset.with_options(options)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "assertion failed"
    ):
      self.evaluate(self.getNext(dataset)())

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_parallel_calls=[None, 2, dataset_ops.AUTOTUNE]
          ),
      )
  )
  def testCapturedInputs(self, num_parallel_calls):
    # [FUNCTION DOC]
    # Scenario: range(1) mapped by a function returning a captured tensor (3 times 4) followed by
    # an identity map, with parallelism none, 2 or autotune. Functions with captured inputs are not
    # fused, so two Map (sequential) or two ParallelMap ops are expected; output is the tensor.
    a = constant_op.constant(3, dtype=dtypes.int64)
    b = constant_op.constant(4, dtype=dtypes.int64)
    some_tensor = math_ops.mul(a, b)

    dataset = dataset_ops.Dataset.range(1)
    # We currently do not support functions with captured inputs.
    if num_parallel_calls in [2, dataset_ops.AUTOTUNE]:
      dataset = dataset.apply(
          testing.assert_next(["ParallelMap", "ParallelMap"])
      )
    else:
      dataset = dataset.apply(testing.assert_next(["Map", "Map"]))

    dataset = dataset.map(
        lambda x: some_tensor, num_parallel_calls=num_parallel_calls
    ).map(lambda x: x, num_parallel_calls=num_parallel_calls)

    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_fusion = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=[some_tensor])


if __name__ == "__main__":
  test.main()
