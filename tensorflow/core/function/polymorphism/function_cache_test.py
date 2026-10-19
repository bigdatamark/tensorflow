# [MODULE DOC]
# Unit tests and micro-benchmarks for FunctionCache. Defines mock trace types (generic, integer,
# subtype-of-two, supertype-with-three, shape) and a mock function record, plus helpers to build
# single-parameter function types. Tests cover insert/lookup, clear, delete, most-specific-match
# selection and order independence; benchmarks time lookups with 50 keys. Run via the test main.
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
"""Tests for function_cache."""

import dataclasses
import itertools
import timeit
from typing import Any, Optional

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.core.function.polymorphism import function_type
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.types import trace


class MockGenericType(trace.TraceType):
  # [CLASS DOC]
  # HELPER: mock trace type used by the function cache tests.
  # Wraps an arbitrary object. It is only a subtype of an equal type, has no common supertype,
  # cannot build placeholders, and compares and hashes by the wrapped object.

  def __init__(self, obj):
    # [FUNCTION DOC]
    # Stores the wrapped object. Input: obj, any hashable value. Returns nothing.
    self._object = obj

  def is_subtype_of(self, other):
    # [FUNCTION DOC]
    # Returns true only when this type equals the other type.
    return self == other

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # Always returns nothing: no common supertype exists. The others argument is ignored.
    return None

  def placeholder_value(self, placeholder_context=None):
    # [FUNCTION DOC]
    # Not supported: always raises a not-implemented error.
    raise NotImplementedError

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality: defers (returns the not-implemented marker) if other is not a trace type; true
    # only if other is a generic mock type wrapping an equal object.
    if not isinstance(other, trace.TraceType):
      return NotImplemented

    return isinstance(other, MockGenericType) and self._object == other._object

  def __hash__(self):
    # [FUNCTION DOC]
    # Hashes the wrapped object.
    return hash(self._object)


class MockIntGenericType(MockGenericType):
  # [CLASS DOC]
  # HELPER: mock trace type whose common supertype is itself when all wrapped objects match.

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # Returns a new instance of this type if every other type wraps an object equal to this one's;
    # otherwise returns nothing.
    if all([self._object == other._object for other in others]):
      return MockIntGenericType(self._object)
    else:
      return None


class MockSubtypeOf2(MockGenericType):
  # [CLASS DOC]
  # HELPER: mock trace type that is a subtype of any generic mock type wrapping the number two.

  def is_subtype_of(self, other):
    # [FUNCTION DOC]
    # Returns false if other is not a generic mock type; otherwise true exactly when the other
    # wraps the number two.
    if not isinstance(other, MockGenericType):
      return False

    return other._object == 2


class MockSupertypes2With3(MockGenericType):
  # [CLASS DOC]
  # HELPER: mock trace type used to test generalization: a type wrapping two generalizes to one
  # wrapping three.

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # If this wraps the number two and the first other wraps an integer, returns a new instance
    # wrapping three; otherwise returns nothing.
    if self._object == 2 and isinstance(others[0]._object, int):
      return MockSupertypes2With3(3)
    else:
      return None


class MockShape(trace.TraceType):
  # [CLASS DOC]
  # HELPER: mock trace type representing a tensor shape as a tuple of optional dimension sizes,
  # where an unknown dimension (nothing) matches anything in subtype checks.

  def __init__(self, *shape: Optional[int]):
    # [FUNCTION DOC]
    # Stores the dimensions given as positional arguments (integers or nothing for unknown).
    self.shape = shape

  def is_subtype_of(self, other: "MockShape") -> bool:
    # [FUNCTION DOC]
    # Returns true if the ranks match and every dimension of the other shape is either unknown
    # or equal to the corresponding dimension of this one; false otherwise.
    if len(self.shape) != len(other.shape):
      return False

    if any(o is not None and s != o for s, o in zip(self.shape, other.shape)):
      return False

    return True

  def most_specific_common_supertype(self, _):
    # [FUNCTION DOC]
    # Not supported: always raises a not-implemented error.
    raise NotImplementedError

  def __str__(self):
    # [FUNCTION DOC]
    # Text form of the shape tuple.
    return str(self.shape)

  def __repr__(self):
    # [FUNCTION DOC]
    # Same text as the string form.
    return str(self)

  def __hash__(self) -> int:
    # [FUNCTION DOC]
    # Hashes the shape tuple.
    return hash(self.shape)

  def __eq__(self, other: "MockShape") -> bool:
    # [FUNCTION DOC]
    # True when the other has an equal shape tuple. Assumes other has a shape attribute.
    return self.shape == other.shape

  def placeholder_value(self, placeholder_context):
    # [FUNCTION DOC]
    # Not supported: always raises a not-implemented error.
    raise NotImplementedError


def make_single_param_type(type_constraint):
  # [FUNCTION DOC]
  # HELPER: builds test function types.
  # Input: type_constraint, a trace type. Returns a function type with a single positional-only
  # parameter named x, without a default, constrained by that type.
  return function_type.FunctionType(
      [
          function_type.Parameter(
              "x",
              function_type.Parameter.POSITIONAL_ONLY,
              False,
              type_constraint,
          )
      ]
  )


@dataclasses.dataclass(frozen=True)
class MockFunction:
  # [CLASS DOC]
  # HELPER: immutable stand-in for a traced function in cache tests. Holds a function type and a
  # label string used by tests to identify which entry a lookup returned.
  function_type: Any
  test_string: str


def make_type(value):
  # [FUNCTION DOC]
  # HELPER: builds test function types from Python values.
  # Input: value, any traceable value. Traces it into a trace type with a fresh tracing context
  # and returns a single-parameter function type constrained by it.
  typing_context = trace_type.InternalTracingContext()
  value_type = trace_type.from_value(value, typing_context)
  f_type = make_single_param_type(value_type)
  return f_type


class FunctionCacheTest(test.TestCase):
  # [CLASS DOC]
  # Tests for FunctionCache using mock functions and mock trace types. No shared fixtures.

  def testConcreteFunctionDictRetainsInsertedKeys(self):
    # [FUNCTION DOC]
    # Adds two functions for the types of 1 and 2 to an empty cache and checks lookups return
    # them by label, while the type of 3 (and 1 before adding) finds nothing.
    cache = function_cache.FunctionCache()

    f_type_1 = make_type(1)
    self.assertIsNone(cache.lookup(f_type_1))

    f_type_2 = make_type(2)
    f_type_3 = make_type(3)

    cache.add(MockFunction(f_type_1, "test_1"))
    cache.add(MockFunction(f_type_2, "test_2"))

    self.assertEqual(cache.lookup(f_type_1).test_string, "test_1")
    self.assertEqual(cache.lookup(f_type_2).test_string, "test_2")
    self.assertIsNone(cache.lookup(f_type_3))

  def testClearRemovesAllConcreteFunctions(self):
    # [FUNCTION DOC]
    # Adds two functions, verifies lookups, clears the cache, then checks every lookup (including
    # the previously added types) returns nothing.
    cache = function_cache.FunctionCache()

    f_type_1 = make_type(1)
    f_type_2 = make_type(2)
    f_type_3 = make_type(3)

    cache.add(MockFunction(f_type_1, "test_1"))
    cache.add(MockFunction(f_type_2, "test_2"))

    self.assertEqual(cache.lookup(f_type_1).test_string, "test_1")
    self.assertEqual(cache.lookup(f_type_2).test_string, "test_2")
    self.assertIsNone(cache.lookup(f_type_3))

    cache.clear()

    self.assertIsNone(cache.lookup(f_type_1))
    self.assertIsNone(cache.lookup(f_type_2))
    self.assertIsNone(cache.lookup(f_type_3))

  def testDeleteRemovesConcreteFunctions(self):
    # [FUNCTION DOC]
    # Adds and deletes a function by its exact type, checking lookup then finds nothing. Then
    # adds a function for a subtype-of-two type, checks that a type wrapping three also resolves
    # to it by subtype dispatch, deletes it, and checks both lookups return nothing.
    cache = function_cache.FunctionCache()
    f_type_1 = make_type(1)
    cache.add(MockFunction(f_type_1, "test_1"))
    self.assertEqual(cache.lookup(f_type_1).test_string, "test_1")
    cache.delete(f_type_1)
    self.assertIsNone(cache.lookup(f_type_1))

    f_type_2 = make_single_param_type(MockSubtypeOf2(2))
    cache.add(MockFunction(f_type_2, "test_2"))
    self.assertEqual(cache.lookup(f_type_2).test_string, "test_2")

    f_type_3 = make_single_param_type(MockSubtypeOf2(3))
    self.assertEqual(cache.lookup(f_type_3).test_string, "test_2")

    cache.delete(f_type_2)
    self.assertIsNone(cache.lookup(f_type_2))
    self.assertIsNone(cache.lookup(f_type_3))

  def testMostSpecificFunctionCacheKeyIsLookedUp(self):
    # [FUNCTION DOC]
    # Adds functions for shapes (1,2,unknown) and (1,2,3) in one context, then looks up
    # (1,2,3). Expects the more specific entry, labeled b.
    ctx = function_cache.FunctionContext(0)
    cache = function_cache.FunctionCache()
    cache.add(
        MockFunction(make_single_param_type(MockShape(1, 2, None)), "a"), ctx
    )
    cache.add(
        MockFunction(make_single_param_type(MockShape(1, 2, 3)), "b"), ctx
    )

    self.assertEqual(
        cache.lookup(
            make_single_param_type(MockShape(1, 2, 3)), ctx
        ).test_string,
        "b",
    )

  def testFirstMostSpecificFunctionCacheKeyIsLookedUp(self):
    # [FUNCTION DOC]
    # Adds functions for shapes (1,2,unknown) and (1,unknown,3), both equally specific for the
    # lookup (1,2,3). Expects the first added entry, labeled a, to be returned.
    ctx = function_cache.FunctionContext(0)
    cache = function_cache.FunctionCache()
    cache.add(
        MockFunction(make_single_param_type(MockShape(1, 2, None)), "a"), ctx
    )
    cache.add(
        MockFunction(
            make_single_param_type(MockShape(1, None, 3)),
            "b",
        ),
        ctx,
    )

    self.assertEqual(
        cache.lookup(
            make_single_param_type(MockShape(1, 2, 3)), ctx
        ).test_string,
        "a",
    )

  def testMostSpecificFunctionCacheKeyIsOrderAgnostic(self):
    # [FUNCTION DOC]
    # Builds four functions with progressively less specific shapes (labels a to d). For every
    # insertion order, builds a fresh cache and checks four lookups resolve to a, b, c and d
    # respectively, showing the result does not depend on insertion order.
    ctx = function_cache.FunctionContext(0)
    keys = [
        (MockFunction(make_single_param_type(MockShape(1, 1, 1)), "a"), ctx),
        (MockFunction(make_single_param_type(MockShape(1, None, 1)), "b"), ctx),
        (
            MockFunction(make_single_param_type(MockShape(None, None, 1)), "c"),
            ctx,
        ),
        (
            MockFunction(
                make_single_param_type(MockShape(None, None, None)), "d"
            ),
            ctx,
        ),
    ]

    for permutation in itertools.permutations(keys):
      cache = function_cache.FunctionCache()
      cache.add(
          permutation[0][0],
          permutation[0][1],
      )
      cache.add(
          permutation[1][0],
          permutation[1][1],
      )
      cache.add(
          permutation[2][0],
          permutation[2][1],
      )
      cache.add(
          permutation[3][0],
          permutation[3][1],
      )

      self.assertEqual(
          cache.lookup(
              make_single_param_type(MockShape(1, 1, 1)), ctx
          ).test_string,
          "a",
      )
      self.assertEqual(
          cache.lookup(
              make_single_param_type(MockShape(1, 2, 1)), ctx
          ).test_string,
          "b",
      )
      self.assertEqual(
          cache.lookup(
              make_single_param_type(MockShape(2, 2, 1)), ctx
          ).test_string,
          "c",
      )
      self.assertEqual(
          cache.lookup(
              make_single_param_type(MockShape(2, 2, 2)), ctx
          ).test_string,
          "d",
      )


class FunctionCacheBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmarks for FunctionCache lookup latency with about 50 stored keys, reported through the
  # benchmark reporting facility.

  def benchmarkCacheHit50thKeyMiss(self):
    # [FUNCTION DOC]
    # Fills the cache with 49 keys built from zero tensors, then times 10000 lookups of the 50th
    # key that is absent. The same lookup is timed twice (labeled subtyping and equality) and
    # the timings and their ratio are reported.
    # KNOWN ISSUE: both timed lambdas do the same lookup, so the subtype/equality split is
    # meaningless.
    # If there are 50 keys and we get a new key that the cache has no concrete
    # functions for.

    cache = function_cache.FunctionCache()
    args_per_call = 5
    num_total_checks = 50

    keys = []
    for i in range(num_total_checks):
      args = []
      for j in range(args_per_call):
        args.append(array_ops.zeros([i, j]))
      keys.append(make_type(args))

    for key in keys[:-1]:
      cache.add(MockFunction(key, "testing"))

    iterations = 10000
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(keys[-1]),
        number=iterations,
    )
    equality_time = timeit.timeit(
        lambda: cache.lookup(keys[-1]),
        number=iterations,
    )

    self.report_benchmark(
        name="cache_hit_50th_f_type_miss",
        iters=iterations,
        wall_time=subtyping_time + equality_time,
        metrics=[
            {
                "name": "cache_hit_50th_f_type_miss_subtype_avg_ms",
                "value": subtyping_time / iterations * 1000,
            },
            {
                "name": "cache_hit_50th_f_type_miss_equality_avg_ms",
                "value": equality_time / iterations * 1000,
            },
            {
                "name": (
                    "cache_hit_50th_f_type_miss_subtype_over_equality_ratio"
                ),
                "value": subtyping_time / equality_time,
            },
        ],
    )

  def benchmarkCacheHit50thKeyEqual(self):
    # [FUNCTION DOC]
    # Fills the cache with 50 keys and times 10000 lookups of the last key, which is present.
    # Timed twice with the same lookup and reported as subtype/equality timings and their ratio.
    # KNOWN ISSUE: both timed lambdas are identical, so the two measurements are not distinct.
    # If there are 50 keys and we get a new key that is equal to a key that is
    # in the cache.

    cache = function_cache.FunctionCache()
    args_per_call = 5
    num_total_checks = 50

    keys = []
    for i in range(num_total_checks):
      args = []
      for j in range(args_per_call):
        args.append(array_ops.zeros([i, j]))
      keys.append(make_type(args))

    for key in keys:
      cache.add(MockFunction(key, "testing"))

    iterations = 10000
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(keys[-1]),
        number=iterations,
    )
    equality_time = timeit.timeit(
        lambda: cache.lookup(keys[-1]),
        number=iterations,
    )

    self.report_benchmark(
        name="cache_hit_50th_f_type_equal",
        iters=iterations,
        wall_time=subtyping_time + equality_time,
        metrics=[
            {
                "name": "cache_hit_50th_f_type_equal_subtype_avg_ms",
                "value": subtyping_time / iterations * 1000,
            },
            {
                "name": "cache_hit_50th_f_type_equal_equality_avg_ms",
                "value": equality_time / iterations * 1000,
            },
            {
                "name": "cache_hit_50th_f_type_subtype_over_equality_ratio",
                "value": subtyping_time / equality_time,
            },
        ],
    )

  def benchmarkCacheHit50thKeyKnownSubtype(self):
    # [FUNCTION DOC]
    # Fills the cache with 49 keys plus a subtype-of-two function, performs a lookup of a type
    # wrapping three first so the cache memorizes it, then times 10000 lookups of a related type
    # and reports the total and average time in milliseconds.
    # If there are 50 keys and we get a key that has a subtype in cache and
    # the cache has observed the key before (to memorize the subtype).

    cache = function_cache.FunctionCache()
    args_per_call = 5
    num_total_checks = 50

    keys = []
    for i in range(num_total_checks - 1):
      args = []
      for j in range(args_per_call):
        args.append(array_ops.zeros([i, j]))
      keys.append(make_type(args))

    for key in keys:
      cache.add(MockFunction(key, "testing"))
    cache.add(
        MockFunction(make_single_param_type(MockSubtypeOf2(2)), "testing"),
    )
    cache.lookup(make_single_param_type(MockSubtypeOf2(3)))

    iterations = 10000
    lookup_key = make_single_param_type(MockSubtypeOf2(2))
    subtyping_time = timeit.timeit(
        lambda: cache.lookup(lookup_key), number=iterations
    )

    self.report_benchmark(
        name="cache_hit_50th_f_type_known_subtype",
        iters=iterations,
        wall_time=subtyping_time,
        metrics=[{
            "name": "cache_hit_50th_f_type_known_subtype_avg_ms",
            "value": subtyping_time / iterations * 1000,
        }],
    )

  def benchmarkCacheHit50thKeyUnknownSubtype(self):
    # [FUNCTION DOC]
    # Times 10000 single lookups, each preceded by a setup that clears the cache and re-adds 49
    # keys plus a function for a subtype type, so the cache has never seen the looked-up type.
    # Reports total and average time in milliseconds.
    # If there are 50 keys and we get a key that has a subtype in cache but
    # the cache has never observed the key before (no memory for the subtype).

    cache = function_cache.FunctionCache()
    args_per_call = 5
    num_total_checks = 50

    keys = []
    for i in range(num_total_checks - 1):
      args = []
      for j in range(args_per_call):
        args.append(array_ops.zeros([i, j]))
      keys.append(make_type(args))

    def setup():
      # [FUNCTION DOC]
      # HELPER: per-iteration setup for the timing loop. Clears the cache and re-adds the keys
      # and the extra subtype function. Returns nothing.
      cache.clear()
      for key in keys:
        cache.add(
            MockFunction(key, "testing"),
        )
      cache.add(
          MockFunction(make_single_param_type(MockSubtypeOf2(3)), "testing"),
      )

    iterations = 10000
    lookup_key = make_single_param_type(MockSubtypeOf2(2))
    subtyping_time = sum(
        timeit.repeat(
            stmt=lambda: cache.lookup(lookup_key),
            setup=setup,
            repeat=iterations,
            number=1,
        )
    )

    self.report_benchmark(
        name="cache_hit_50th_f_type_unknown_subtype",
        iters=iterations,
        wall_time=subtyping_time,
        metrics=[{
            "name": "cache_hit_50th_f_type_unknown_subtype_avg_ms",
            "value": subtyping_time / iterations * 1000,
        }],
    )


if __name__ == "__main__":
  test.main()
