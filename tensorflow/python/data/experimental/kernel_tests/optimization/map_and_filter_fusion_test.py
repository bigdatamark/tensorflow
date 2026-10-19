# [MODULE DOC]
# Tests for the MapAndFilterFusion tf.data optimization (a map followed by a filter is fused into
# a Map, Filter, Map sequence), over many scalar and tuple map/predicate combinations, plus a case
# with captured inputs where fusion is not supported. Contains a module-level generator of the
# parameter combinations, one test class and a standard test main entry point.
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
"""Tests for the `MapAndFilterFusion` optimization."""
import functools

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _test_combinations():
  # [FUNCTION DOC]
  # HELPER: builds the parameter combinations for the map-and-filter fusion test.
  # Takes no arguments. Builds the cross product of four scalar map functions with four scalar
  # predicates, and of two tuple-producing map functions with two two-argument predicates, each
  # case named by indices. Returns a list of parameter dictionaries with the map function and a
  # named predicate object, as produced by the combinations library.
  cases = []

  identity = lambda x: x
  increment = lambda x: x + 1
  minus_five = lambda x: x - 5

  def increment_and_square(x):
    # [FUNCTION DOC]
    # Returns the square of the input plus one.
    y = x + 1
    return y * y

  functions = [identity, increment, minus_five, increment_and_square]

  take_all = lambda x: constant_op.constant(True)
  is_zero = lambda x: math_ops.equal(x, 0)
  is_odd = lambda x: math_ops.equal(x % 2, 0)
  greater = lambda x: math_ops.greater(x + 5, 0)
  predicates = [take_all, is_zero, is_odd, greater]

  for i, function in enumerate(functions):
    for j, predicate in enumerate(predicates):
      cases.append((function, "Scalar{}{}".format(i, j), predicate))

  replicate = lambda x: (x, x)
  with_two = lambda x: (x, 2)
  functions = [replicate, with_two]
  take_all = lambda x, y: constant_op.constant(True)
  is_zero = lambda x, y: math_ops.equal(x * math_ops.cast(y, dtypes.int64), 0)
  predicates = [take_all, is_zero]

  for i, function in enumerate(functions):
    for j, predicate in enumerate(predicates):
      cases.append((function, "Tuple{}{}".format(i, j), predicate))

  def reduce_fn(x, y):
    # [FUNCTION DOC]
    # HELPER: reducer used to fold cases into one combination list.
    # Takes the accumulated list x and a case y (function, name, predicate); returns x extended
    # with the combination built from the function and the named predicate.
    function, name, predicate = y
    return x + combinations.combine(
        function=function,
        predicate=combinations.NamedObject(name, predicate))

  return functools.reduce(reduce_fn, cases, [])


class MapAndFilterFusionTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the map-and-filter fusion optimization of tf.data.

  def _testDataset(self, dataset, function, predicate):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the fusion tests.
    # Computes the expected output by applying the map function and then the predicate to each
    # of 0..9 in plain evaluation (tuples are unpacked into multiple arguments), keeping elements
    # whose predicate is true, and asserts the given dataset produces exactly that list.
    expected_output = []
    for x in range(10):
      r = function(x)
      if isinstance(r, tuple):
        b = predicate(*r)  # Pass tuple as multiple arguments.
      else:
        b = predicate(r)
      if self.evaluate(b):
        expected_output.append(r)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_combinations()))
  def testMapAndFilterFusion(self, function, predicate):
    # [FUNCTION DOC]
    # Scenario: range(10) mapped by the given function then filtered by the predicate, with only
    # map_and_filter_fusion enabled. assert_next requires the sequence Map, Filter, Map in the
    # optimized graph; output is verified by the shared helper.
    dataset = dataset_ops.Dataset.range(10).apply(
        testing.assert_next(["Map", "Filter",
                             "Map"])).map(function).filter(predicate)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_and_filter_fusion = True
    dataset = dataset.with_options(options)
    self._testDataset(dataset, function, predicate)

  @combinations.generate(test_base.default_test_combinations())
  def testCapturedInputs(self):
    # [FUNCTION DOC]
    # Scenario: the filter predicate captures an outer tensor (3 times 4), so fusion is not
    # supported; assert_next expects the unfused Map then Filter. Elements are squares of 0..9
    # kept when less than 12; output checked by the shared helper.
    a = constant_op.constant(3, dtype=dtypes.int64)
    b = constant_op.constant(4, dtype=dtypes.int64)
    some_tensor = math_ops.mul(a, b)
    function = lambda x: x * x

    def predicate(y):
      # [FUNCTION DOC]
      # Predicate: true when the input cast to 64-bit integer is less than the captured tensor.
      return math_ops.less(math_ops.cast(y, dtypes.int64), some_tensor)

    # We currently do not support functions with captured inputs.
    dataset = dataset_ops.Dataset.range(10).apply(
        testing.assert_next(["Map", "Filter"])).map(function).filter(predicate)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.map_and_filter_fusion = True
    dataset = dataset.with_options(options)
    self._testDataset(dataset, function, predicate)


if __name__ == "__main__":
  test.main()
