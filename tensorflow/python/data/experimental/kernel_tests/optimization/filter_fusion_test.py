# [MODULE DOC]
# Tests for the filter fusion graph optimization, which merges consecutive filter transformations
# into one. Builds parameterized cases of mapping functions and chains of predicates, and checks
# that fused datasets produce the same output as applying the predicates manually; also checks that
# predicates with captured inputs are not fused. No import-time side effects besides imports.
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
"""Tests for the `FilterFusion` optimization."""
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
  # HELPER: generates the parameter combinations for the filter fusion test. Takes no input.
  # Builds cases of scalar predicates (always true, equals zero, greater than minus five) in
  # chains of two and three, plus two tuple-element cases, each paired with a mapping function.
  # Returns a list of combination mappings with function and predicates (named for test names).
  cases = []

  take_all = lambda x: constant_op.constant(True)
  is_zero = lambda x: math_ops.equal(x, 0)
  greater = lambda x: math_ops.greater(x + 5, 0)
  predicates = [take_all, is_zero, greater]
  for i, x in enumerate(predicates):
    for j, y in enumerate(predicates):
      cases.append((lambda x: x, "Scalar{}{}".format(i, j), [x, y]))
      for k, z in enumerate(predicates):
        cases.append((lambda x: x, "Scalar{}{}{}".format(i, j, k), [x, y, z]))

  take_all = lambda x, y: constant_op.constant(True)
  is_zero = lambda x, y: math_ops.equal(x * math_ops.cast(y, dtypes.int64), 0)

  cases.append((lambda x: (x, x), "Tuple1", [take_all, take_all]))
  cases.append((lambda x: (x, 2), "Tuple2", [take_all, is_zero]))

  def reduce_fn(x, y):
    # [FUNCTION DOC]
    # HELPER: reduction callback; takes the accumulated list of combinations and one case (mapping
    # function, name, predicates) and returns the list extended with the combination for the case.
    function, name, predicates = y
    return x + combinations.combine(
        function=function,
        predicates=combinations.NamedObject(name, predicates))

  return functools.reduce(reduce_fn, cases, [])


class FilterFusionTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the filter fusion optimization.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_combinations()))
  def testFilterFusion(self, function, predicates):
    # [FUNCTION DOC]
    # Parameterized by a mapping function and a list of predicates. Builds a range of 5 mapped by
    # the function, applies each predicate as a filter, caches, enables only filter fusion, and
    # asserts via an assert-next check that the pipeline has a map, a single filter and cache.
    # Computes the expected elements by applying the predicates in Python and asserts they match.
    dataset = dataset_ops.Dataset.range(5).apply(
        testing.assert_next(["Map", "Filter", "MemoryCacheImpl"])).map(function)
    for predicate in predicates:
      dataset = dataset.filter(predicate)

    dataset = dataset.cache()
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.filter_fusion = True
    dataset = dataset.with_options(options)
    expected_output = []
    for x in range(5):
      r = function(x)
      filtered = False
      for predicate in predicates:
        if isinstance(r, tuple):
          b = predicate(*r)  # Pass tuple as multiple arguments.
        else:
          b = predicate(r)
        if not self.evaluate(b):
          filtered = True
          break

      if not filtered:
        expected_output.append(r)
    self.assertDatasetProduces(dataset, expected_output=expected_output)

  @combinations.generate(test_base.default_test_combinations())
  def testCapturedInputs(self):
    # [FUNCTION DOC]
    # Applies a filter whose predicate captures an outer tensor followed by a trivial filter, with
    # fusion enabled; asserts the two filters stay separate (not fused, as captured inputs are
    # unsupported) and that all of 0..9 are produced.
    a = constant_op.constant(3, dtype=dtypes.int64)
    b = constant_op.constant(4, dtype=dtypes.int64)
    some_tensor = math_ops.mul(a, b)

    def predicate(y):
      # [FUNCTION DOC]
      # Predicate callback: returns whether the element, cast to 64-bit integer, is less than the
      # captured product 12.
      return math_ops.less(math_ops.cast(y, dtypes.int64), some_tensor)

    # We currently do not support functions with captured inputs.
    dataset = dataset_ops.Dataset.range(10).apply(
        testing.assert_next(["Filter", "Filter"
                            ])).filter(predicate).filter(lambda x: True)
    options = options_lib.Options()
    options.experimental_optimization.apply_default_optimizations = False
    options.experimental_optimization.filter_fusion = True
    dataset = dataset.with_options(options)
    self.assertDatasetProduces(dataset, expected_output=range(10))


if __name__ == "__main__":
  test.main()
