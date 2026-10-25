# [MODULE DOC]
# KNOWN ISSUE: the module still contains Python 2 era branches (six.PY2) and many tests marked
# disabled with a decorator, so part of the file is dead code.
# Large parameterized test suite for the TensorFlow NumPy API (tnp, imported through the test
# wrapper module) checked against reference results from real NumPy (onp). It was adapted from the
# JAX NumPy test suite. At import time it parses absl flags, defines lists of shapes and dtypes
# (scalar, empty, non-empty, float, complex, integer, bool), and builds tables of operation
# records (one-to-one ops, compound ops, bitwise ops, reducers, reducers without dtype, argmin and
# argmax, operator overloads, right-hand operator overloads) that drive generated test cases.
# Two test classes are defined: one covering most NumPy functions (compared with NumPy and also
# compiled through a function-tracing jit helper) and one for gradient tests, most of which are
# disabled. Some tests are wrapped so they only run when an additional-tests flag is set.
# Running the file as a script starts the absl test runner. Nothing is registered globally except
# flag parsing; it is a test module, not a helper.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
import collections
import functools
from functools import partial
import itertools
import operator
import unittest
from unittest import SkipTest
import warnings

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp
import six

from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.ops.numpy_ops.tests.config import config
from tensorflow.python.ops.numpy_ops.tests.config import FLAGS
import tensorflow.python.ops.numpy_ops.tests.extensions as nje
import tensorflow.python.ops.numpy_ops.tests.np_wrapper as tnp
import tensorflow.python.ops.numpy_ops.tests.test_util as jtu
from tensorflow.python.util import nest
from tensorflow.python.util.numpy_compat import np_where

config.parse_flags_with_absl()


nonempty_nonscalar_array_shapes = [(4,), (3, 4), (3, 1), (1, 4), (2, 1, 4), (2, 3, 4)]
nonempty_array_shapes = [()] + nonempty_nonscalar_array_shapes
empty_array_shapes = [(0,), (0, 4), (3, 0),]

scalar_shapes = [jtu.NUMPY_SCALAR_SHAPE, jtu.PYTHON_SCALAR_SHAPE]
array_shapes = nonempty_array_shapes + empty_array_shapes
nonzerodim_shapes = nonempty_nonscalar_array_shapes + empty_array_shapes
nonempty_shapes = scalar_shapes + nonempty_array_shapes
all_shapes =  scalar_shapes + array_shapes

# TODO(wangpeng): float_dtypes = [tnp.bfloat16, onp.float16, onp.float32,
#                                 onp.float64]
float_dtypes = [onp.float16, onp.float32, onp.float64]
complex_dtypes = [onp.complex64, onp.complex128]
int_dtypes = [onp.int32, onp.int64]
unsigned_dtypes = [onp.uint32, onp.uint64]
bool_dtypes = [onp.bool_]
default_dtypes = float_dtypes + int_dtypes
inexact_dtypes = float_dtypes + complex_dtypes
number_dtypes = float_dtypes + complex_dtypes + int_dtypes
all_dtypes = number_dtypes + bool_dtypes


python_scalar_dtypes = [tnp.bool_, tnp.int_, tnp.float64, tnp.complex128]
# pylint: disable=unnecessary-lambda,g-long-lambda,expression-not-assigned

def _valid_dtypes_for_shape(shape, dtypes):
  # [FUNCTION DOC]
  # HELPER: builds the dtype lists used when generating parameterized test cases.
  # Takes a shape (possibly the special marker for a Python scalar) and a list of dtypes. For the
  # Python scalar marker only dtypes that exist as Python scalar types (bool, int, float64,
  # complex128) are kept; for every other shape the list is returned unchanged. Returns a list.
  # Not all (shape, dtype) pairs are valid. In particular, Python scalars only
  # have one type in each category (float, bool, etc.)
  if shape is jtu.PYTHON_SCALAR_SHAPE:
    return [t for t in dtypes if t in python_scalar_dtypes]
  return dtypes

def _shape_and_dtypes(shapes, dtypes):
  # [FUNCTION DOC]
  # HELPER: generator of (shape, dtype) pairs for parameterized test case generation.
  # For each shape in the given list, yields a pair with every dtype that is valid for that shape
  # (see the valid-dtypes helper). Takes a list of shapes and a list of dtypes; yields tuples.
  for shape in shapes:
    for dtype in _valid_dtypes_for_shape(shape, dtypes):
      yield (shape, dtype)

OpRecord = collections.namedtuple(
  "OpRecord",
  ["name", "nargs", "dtypes", "shapes", "rng_factory", "diff_modes",
   "test_name", "check_dtypes", "tolerance", "inexact",
   "check_incomplete_shape"])

def op_record(name, nargs, dtypes, shapes, rng_factory, diff_modes,
              test_name=None, check_dtypes=True, tolerance=None, inexact=False,
              check_incomplete_shape=True):
  # [FUNCTION DOC]
  # HELPER: constructor for the operation records that drive generated test cases.
  # Builds a named tuple describing one NumPy function to test: its name, number of arguments,
  # allowed dtypes, allowed shapes, a random-data generator factory, differentiation modes, a test
  # name (defaults to the function name when not given), whether dtypes should be checked,
  # optional tolerance override, whether the result is inexact (floating) and whether to check
  # incomplete static shapes. Returns the record; no side effects.
  test_name = test_name or name

  return OpRecord(name, nargs, dtypes, shapes, rng_factory, diff_modes,
                  test_name, check_dtypes, tolerance, inexact,
                  check_incomplete_shape)


def minus(a, b):
  # [FUNCTION DOC]
  # HELPER: list difference used when building dtype lists for operation records.
  # Returns a new list holding the elements of the first list that do not appear in the second
  # list, preserving order. Inputs are not modified.
  return [x for x in a if x not in b]


JAX_ONE_TO_ONE_OP_RECORDS = [
    op_record("abs", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("add", 2, all_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("ceil", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("conj", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal, []),
    op_record("exp", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("fabs", 1, float_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("float_power", 2, inexact_dtypes, all_shapes,
              partial(jtu.rand_default, scale=1), ["rev"],
              tolerance={
                  # TODO(wangpeng): tnp.bfloat16: 1e-2,
                  onp.float32: 1e-3,
                  onp.float64: 1e-12, onp.complex64: 2e-4,
                  onp.complex128: 1e-12}, check_dtypes=False),
    op_record("floor", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("greater", 2, minus(all_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_equal, []),
    op_record("greater_equal", 2, minus(all_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_equal, []),
    op_record("less", 2, minus(all_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_equal, []),
    op_record("less_equal", 2, minus(all_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_equal, []),
    op_record("log", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("logical_and", 2, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("logical_not", 1, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("logical_or", 2, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("logical_xor", 2, all_dtypes, all_shapes, jtu.rand_bool, []),
    op_record("maximum", 2, minus(all_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_inf, []),
    op_record("minimum", 2, minus(all_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_inf, []),
    op_record("multiply", 2, all_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("negative", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("nextafter", 2, [f for f in float_dtypes
                               if f not in (tnp.bfloat16, onp.float16)],
              all_shapes, jtu.rand_default, ["rev"], inexact=True, tolerance=0),
    op_record("not_equal", 2, all_dtypes, all_shapes, jtu.rand_some_equal, ["rev"]),
    op_record("array_equal", 2, number_dtypes, all_shapes, jtu.rand_some_equal, ["rev"]),
    op_record("reciprocal", 1, inexact_dtypes, all_shapes, jtu.rand_default, []),
    op_record("subtract", 2, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("signbit", 1, default_dtypes + bool_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, ["rev"]),
    op_record("sin", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("cos", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("tan", 1, number_dtypes, all_shapes,
              partial(jtu.rand_uniform, -1.5, 1.5), ["rev"],
              tolerance={onp.complex64: 3e-5, onp.complex128: 4e-14},
              inexact=True),
    # TODO(wangpeng): Add float16 support
    op_record("sinh", 1, minus(number_dtypes, [onp.float16]), all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("cosh", 1, minus(number_dtypes, [onp.float16]), all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    # TODO(b/142975473): on CPU, tanh for complex128 is only accurate to
    # ~float32 precision.
    # TODO(b/143135720): on GPU, tanh has only ~float32 precision.
    op_record("tanh", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              tolerance={onp.float64: 1e-7, onp.complex128: 1e-7},
              inexact=True),
    op_record("arcsin", 1, minus(float_dtypes, [onp.float16]), all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arccos", 1, minus(float_dtypes, [onp.float16]), all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arctan", 1, minus(float_dtypes, [onp.float16]), all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arctan2", 2, minus(float_dtypes, [onp.float16]), all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
    op_record("arcsinh", 1, minus(number_dtypes, [onp.float16]), all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("arccosh", 1, minus(number_dtypes, [onp.float16]), all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("arctanh", 1, minus(number_dtypes, [onp.float16]), all_shapes, jtu.rand_small, ["rev"],
              inexact=True),
]

JAX_COMPOUND_OP_RECORDS = [
    # angle has inconsistent 32/64-bit return types across numpy versions.
    op_record("angle", 1, number_dtypes, all_shapes, jtu.rand_default, [],
              check_dtypes=False, inexact=True),
    op_record("atleast_1d", 1, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("atleast_2d", 1, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("atleast_3d", 1, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("cbrt", 1, default_dtypes, all_shapes, jtu.rand_default, ["rev"],
              inexact=True),
    op_record("conjugate", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("deg2rad", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("divide", 2, number_dtypes, all_shapes, jtu.rand_nonzero, ["rev"],
              inexact=six.PY3),
    op_record("divmod", 2, minus(int_dtypes + float_dtypes, [onp.float16]),
              all_shapes, jtu.rand_nonzero, []),
    op_record("exp2", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"],
              tolerance={
                  # TODO(wangpeng): tnp.bfloat16: 2e-2,
                  onp.float16: 1e-2}, inexact=True),
    # TODO(b/142975473): on CPU, expm1 for float64 is only accurate to ~float32
    # precision.
    op_record("expm1", 1, number_dtypes, all_shapes, jtu.rand_positive, [],
              test_name="expm1_large", tolerance={onp.float64: 1e-8}, inexact=True),
    op_record("expm1", 1, number_dtypes, all_shapes, jtu.rand_small_positive,
              [], tolerance={onp.float64: 1e-8}, inexact=True),
    op_record("fix", 1, float_dtypes, all_shapes, jtu.rand_default, []),
    op_record("floor_divide", 2, minus(number_dtypes, complex_dtypes),
              all_shapes, jtu.rand_nonzero, ["rev"]),
    op_record("heaviside", 2, default_dtypes, all_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("hypot", 2, default_dtypes, all_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("kron", 2, number_dtypes, nonempty_shapes, jtu.rand_default, [],
              check_incomplete_shape=False),
    op_record("outer", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("imag", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("iscomplex", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("isfinite", 1, minus(inexact_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_inf_and_nan, []),
    op_record("isinf", 1, minus(inexact_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_inf_and_nan, []),
    op_record("isnan", 1, minus(inexact_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_inf_and_nan, []),
    op_record("isneginf", 1, float_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isposinf", 1, float_dtypes, all_shapes, jtu.rand_some_inf_and_nan, []),
    op_record("isreal", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("isrealobj", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("log2", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("log10", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("log1p", 1, number_dtypes, all_shapes, jtu.rand_positive, [],
              test_name="log1p_large", tolerance={onp.float64: 1e-12},
              inexact=True),
    op_record("log1p", 1, number_dtypes, all_shapes, jtu.rand_small_positive, [],
              tolerance={onp.float64: 1e-12}, inexact=True),
    op_record("logaddexp", 2, float_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, ["rev"],
              tolerance={onp.float64: 1e-12}, inexact=True),
    op_record("logaddexp2", 2, float_dtypes, all_shapes,
              jtu.rand_some_inf_and_nan, ["rev"],
              tolerance={onp.float16: 1e-2}, inexact=True),
    op_record("polyval", 2, number_dtypes, nonempty_nonscalar_array_shapes,
              jtu.rand_default, [], check_dtypes=False,
              tolerance={onp.float16: 1e-2, onp.float64: 1e-12},
              check_incomplete_shape=False),
    op_record("positive", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("power", 2, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              tolerance={onp.complex128: 1e-14}),
    op_record("rad2deg", 1, float_dtypes, all_shapes, jtu.rand_default, [],
              tolerance={onp.float64: 5e-6}),
    op_record("ravel", 1, all_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("real", 1, number_dtypes, all_shapes, jtu.rand_some_inf, []),
    op_record("remainder", 2, minus(default_dtypes, [onp.float16]), all_shapes,
              jtu.rand_nonzero, [], tolerance={onp.float16: 1e-2}),
    op_record("mod", 2, minus(default_dtypes, [onp.float16]), all_shapes,
              jtu.rand_nonzero, []),
    op_record("sinc", 1, [t for t in number_dtypes if t != tnp.bfloat16],
              all_shapes, jtu.rand_default, ["rev"],
              tolerance={onp.complex64: 1e-5}, inexact=True,
              check_dtypes=False),
    op_record("square", 1, number_dtypes, all_shapes, jtu.rand_default, ["rev"]),
    op_record("sqrt", 1, number_dtypes, all_shapes, jtu.rand_positive, ["rev"],
              inexact=True),
    op_record("transpose", 1, all_dtypes, all_shapes, jtu.rand_default, ["rev"],
              check_dtypes=False),
    op_record("true_divide", 2, all_dtypes, all_shapes, jtu.rand_nonzero,
              ["rev"], inexact=True),
    op_record("diff", 1, number_dtypes, nonzerodim_shapes, jtu.rand_default,
              ["rev"], check_incomplete_shape=False),
]

JAX_BITWISE_OP_RECORDS = [
    op_record("bitwise_and", 2, int_dtypes, all_shapes,
              jtu.rand_default, []),
    op_record("bitwise_not", 1, int_dtypes, all_shapes,
              jtu.rand_default, []),
    op_record("bitwise_or", 2, int_dtypes, all_shapes,
              jtu.rand_default, []),
    op_record("bitwise_xor", 2, int_dtypes, all_shapes,
              jtu.rand_default, []),
]

JAX_REDUCER_RECORDS = [
    op_record("mean", 1, number_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("prod", 1, all_dtypes, all_shapes, jtu.rand_small_positive, []),
    op_record("sum", 1, all_dtypes, all_shapes, jtu.rand_default, []),
    op_record("nanmean", 1, minus(inexact_dtypes, complex_dtypes),
              nonempty_shapes, jtu.rand_some_nan, [], inexact=True),
    op_record("nanprod", 1, minus(inexact_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_nan, []),
    op_record("nansum", 1, minus(number_dtypes, complex_dtypes), all_shapes,
              jtu.rand_some_nan, []),
]

JAX_REDUCER_NO_DTYPE_RECORDS = [
    op_record("all", 1, all_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("any", 1, all_dtypes, all_shapes, jtu.rand_some_zero, []),
    op_record("max", 1, minus(all_dtypes, complex_dtypes), nonempty_shapes,
              jtu.rand_default, []),
    op_record("min", 1, minus(all_dtypes, complex_dtypes), nonempty_shapes,
              jtu.rand_default, []),
    op_record("var", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
    op_record("std", 1, all_dtypes, nonempty_shapes, jtu.rand_default, [],
              inexact=True),
]

JAX_ARGMINMAX_RECORDS = [
    op_record("argmin", 1, minus(all_dtypes, complex_dtypes), nonempty_shapes,
              jtu.rand_some_equal, []),
    op_record("argmax", 1, minus(all_dtypes, complex_dtypes), nonempty_shapes,
              jtu.rand_some_equal, []),
]

JAX_OPERATOR_OVERLOADS = [
    op_record("__add__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__sub__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__mul__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__eq__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__ne__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__lt__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__gt__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__ge__", 2, default_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__pos__", 1, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__neg__", 1, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__pow__", 2, inexact_dtypes, all_shapes, jtu.rand_positive, [],
              tolerance={onp.float32: 2e-4, onp.complex64: 2e-4, onp.complex128: 1e-14}),
    op_record("__mod__", 2, minus(default_dtypes, [onp.float16]), all_shapes, jtu.rand_nonzero, [],
              tolerance={onp.float16: 1e-1}),
    op_record("__floordiv__", 2, default_dtypes, all_shapes, jtu.rand_nonzero, []),
    op_record("__truediv__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, [],
              inexact=True),
    op_record("__abs__", 1, number_dtypes, all_shapes, jtu.rand_default, []),
    # TODO(mattjj): __invert__ fails on bool dtypes because ~True == -2
    op_record("__invert__", 1, int_dtypes, all_shapes, jtu.rand_default, []),
    # TODO(mattjj): investigate these failures
    # op_record("__or__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__and__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    # op_record("__xor__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__divmod__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, []),
    # TODO(mattjj): lshift, rshift
]

JAX_RIGHT_OPERATOR_OVERLOADS = [
    op_record("__radd__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__rsub__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__rmul__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    op_record("__rpow__", 2, inexact_dtypes, all_shapes, jtu.rand_positive, [],
              tolerance={onp.float32: 2e-4, onp.complex64: 1e-3}),
    op_record("__rmod__", 2, minus(default_dtypes, [onp.float16]), all_shapes, jtu.rand_nonzero, [],
              tolerance={onp.float16: 1e-1}),
    op_record("__rfloordiv__", 2, default_dtypes, all_shapes, jtu.rand_nonzero, []),
    op_record("__rtruediv__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, [],
              inexact=True),
    # op_record("__ror__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__rand__", 2, number_dtypes, all_shapes, jtu.rand_default, []),
    # op_record("__rxor__", 2, number_dtypes, all_shapes, jtu.rand_bool, []),
    # op_record("__rdivmod__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, []),
]

numpy_version = tuple(map(int, onp.version.version.split('.')))
if numpy_version >= (1, 15):
  JAX_COMPOUND_OP_RECORDS += [
      op_record("isclose", 2, [t for t in all_dtypes if t != tnp.bfloat16],
                all_shapes, jtu.rand_small_positive, []),
      op_record("gcd", 2, int_dtypes, all_shapes, jtu.rand_default, []),
      op_record("lcm", 2, int_dtypes, all_shapes, jtu.rand_default, []),
  ]
  JAX_REDUCER_NO_DTYPE_RECORDS += [
      op_record("ptp", 1, minus(number_dtypes, complex_dtypes), nonempty_shapes,
                jtu.rand_default, []),
  ]

if six.PY2:
  JAX_OPERATOR_OVERLOADS += [
    op_record("__div__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, []),
  ]
  JAX_RIGHT_OPERATOR_OVERLOADS += [
    op_record("__rdiv__", 2, number_dtypes, all_shapes, jtu.rand_nonzero, []),
  ]


CombosWithReplacement = itertools.combinations_with_replacement


def _dtypes_are_compatible_for_bitwise_ops(args):
  # [FUNCTION DOC]
  # HELPER: filter for generating bitwise-operation test cases.
  # Takes a sequence of dtypes (one or two). Returns true when there is at most one dtype. For two
  # dtypes, bools are treated as 32-bit integers, the narrower is ordered first, and the pair is
  # rejected for unsigned 32-bit with unsigned 64-bit; otherwise it is accepted when both have the
  # same signedness, both are 32 bits wide, or a 32-bit type is paired with a signed 64-bit type.
  # Returns a boolean; no side effects.
  if len(args) <= 1:
    return True
  is_signed = lambda dtype: tnp.issubdtype(dtype, onp.signedinteger)
  width = lambda dtype: tnp.iinfo(dtype).bits
  x, y = args
  # `tnp.iinfo(dtype).bits` can't be called on bools, so we convert bools to
  # ints.
  if x == tnp.bool_:
    x = tnp.int32
  if y == tnp.bool_:
    y = tnp.int32
  if width(x) > width(y):
    x, y = y, x
  if x == tnp.uint32 and y == tnp.uint64:
    return False
  # The following condition seems a little ad hoc, but seems to capture what
  # numpy actually implements.
  return (
      is_signed(x) == is_signed(y)
      or (width(x) == 32 and width(y) == 32)
      or (width(x) == 32 and width(y) == 64 and is_signed(y)))


def _shapes_are_broadcast_compatible(shapes):
  # [FUNCTION DOC]
  # HELPER: filter for generating test cases whose shapes must broadcast together.
  # Takes a list of shapes. Adds arrays of zeros of those shapes one by one; returns false as soon
  # as NumPy raises a value error for incompatible shapes, otherwise true.
  accumulator = onp.zeros([])
  for shape in shapes:
    try:
      accumulator = accumulator + onp.zeros(shape)
    except ValueError:
      return False
  return True

def _shapes_are_equal_length(shapes):
  # [FUNCTION DOC]
  # HELPER: filter for generating test cases whose shapes must have the same rank.
  # Takes a non-empty list of shapes; returns true when every shape has the same number of
  # dimensions as the first one.
  # KNOWN ISSUE: the name says nothing about the first-shape comparison, and it raises an index
  # error on an empty list of shapes.
  return all(len(shape) == len(shapes[0]) for shape in shapes[1:])


# pylint: disable=g-doc-return-or-yield
def _promote_like_lnp(fun, inexact=False):
  # [FUNCTION DOC]
  # HELPER: wraps NumPy reference implementations so they follow TensorFlow NumPy type promotion.
  # Takes a function and a flag saying whether results should be at least floating point. Returns
  # a wrapper that flattens the positional arguments, computes the TensorFlow NumPy result dtype
  # (using float64 when inexact is requested and no argument is already inexact), converts all
  # arguments to NumPy arrays of that dtype and then calls the wrapped function.
  """Decorator that promotes the arguments of `fun` to `tnp.result_type(*args)`.

  tnp and onp have different type promotion semantics; this decorator allows
  tests make an onp reference implementation act more like an tnp
  implementation.
  """
  def wrapper(*args, **kw):
    # [FUNCTION DOC]
    # HELPER: inner function of the promotion decorator.
    # Promotes all arguments to the common dtype as described for the outer function, then calls
    # the wrapped function with the converted arguments and the original keyword arguments, and
    # returns its result.
    flat_args = nest.flatten(args)
    if inexact and not any(
        tnp.issubdtype(tnp.result_type(x).as_numpy_dtype, tnp.inexact)
        for x in flat_args):
      dtype = tnp.result_type(tnp.float64, *flat_args)
    else:
      dtype = tnp.result_type(*flat_args)
    dtype = dtype.as_numpy_dtype
    args = nest.map_structure(lambda a: onp.asarray(a, dtype), args)
    return fun(*args, **kw)
  return wrapper
# pylint: enable=g-doc-return-or-yield


def new_test(f):
  # [FUNCTION DOC]
  # HELPER: decorator for tests of newly added functionality.
  # Takes a test method and returns a wrapper that skips the test (reports it skipped with a
  # message) unless the additional-tests command line flag is true; otherwise it runs the method.

  def wrapper(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: inner function of the new-test decorator.
    # If the additional-tests flag is false, marks the test case as skipped; otherwise calls the
    # wrapped test with the same arguments. Returns nothing.
    if not FLAGS.tf_numpy_additional_tests:
      self.skipTest("Newly added test is disabled, since flag is False.")
    else:
      f(self, *args, **kwargs)

  return wrapper


def named_parameters(ls):
  # [FUNCTION DOC]
  # HELPER: replacement for the absl named-parameters decorator that tolerates empty lists.
  # Takes a list, tuple or chained iterator of test case dictionaries. If it is empty, returns a
  # decorator that replaces the test with one that just skips. A chained iterator is peeked at
  # first element (consumed and re-chained) to detect emptiness. Otherwise delegates to the absl
  # named-parameters decorator with the cases.
  """A version that allows an empty param list."""
  def noop(_):
    # [FUNCTION DOC]
    # HELPER: decorator returned by the named-parameters helper for empty case lists.
    # Ignores the decorated test and returns a replacement test that skips itself.
    def wrapper(self, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: replacement test body; skips with the message that the parameter list is empty.
      self.skipTest("Empty parameter list")
    return wrapper
  if isinstance(ls, (list, tuple)) and not ls:
    return noop
  if isinstance(ls, itertools.chain):
    try:
      first = next(ls)
    except StopIteration:
      return noop
    else:
      ls = itertools.chain([first], ls)
  return parameterized.named_parameters(ls)


# TODO(wangpeng): Enable all disabled tests in this class
class LaxBackedNumpyTests(jtu.TestCase):
  # [CLASS DOC]
  # Main test case class (derived from the shared test base class in the test utilities) that
  # checks the TensorFlow NumPy functions against real NumPy. Holds no instance state; every test
  # is a generated or hand-written method. Most methods are parameterized from the module-level
  # operation record tables, draw random inputs with a given generator, compare against NumPy
  # (checking values and optionally dtypes) and then compile the function and check again,
  # including static shape inference where requested. Some tests are disabled by a decorator.
  """Tests for LAX-backed Numpy implementation."""

  def _GetArgsMaker(self, rng, shapes, dtypes, onp_arrays=True):
    # [FUNCTION DOC]
    # HELPER: builds the input generator callback for the comparison helpers.
    # Takes a random-data generator (callable of shape and dtype), parallel lists of shapes and
    # dtypes, and a flag. Returns a zero-argument function which on every call creates one random
    # array per shape/dtype pair (a missing dtype means float64); the arrays are NumPy arrays when
    # the flag is true, otherwise they are converted to TensorFlow NumPy arrays.
    def f():
      # [FUNCTION DOC]
      # HELPER: the generator callback returned by the args-maker builder; takes no arguments and
      # returns a list of freshly generated arrays (NumPy or TensorFlow NumPy per the outer flag).
      out = [rng(shape, dtype or tnp.float64)
             for shape, dtype in zip(shapes, dtypes)]
      return out if onp_arrays else [tnp.asarray(a) for a in out]
    return f

  @named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.test_name, shapes,
                                                      dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes,
         "onp_op": getattr(onp, rec.name), "lnp_op": getattr(tnp, rec.name),
         "check_dtypes": rec.check_dtypes, "tolerance": rec.tolerance,
         "inexact": rec.inexact,
         "check_incomplete_shape": rec.check_incomplete_shape}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          CombosWithReplacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes)))
      for rec in itertools.chain(JAX_ONE_TO_ONE_OP_RECORDS,
                                 JAX_COMPOUND_OP_RECORDS)))
  @unittest.skipIf(onp.__version__ >= onp.lib.NumpyVersion('2.0.0'),
                   'tf numpy is implemented to be numpy 1.x compatible')
  def testOp(self, onp_op, lnp_op, rng_factory, shapes, dtypes, check_dtypes,
             tolerance, inexact, check_incomplete_shape):
    # [FUNCTION DOC]
    # Parameterized test over the one-to-one and compound operation records (skipped on NumPy 2 or
    # newer). Inputs: the NumPy and TensorFlow NumPy implementations, random-generator factory,
    # shapes and dtypes of the arguments, dtype checking flag, optional tolerance, an inexact flag
    # and a flag for incomplete-shape checking. The kron case with two (2,3,4) inputs is skipped.
    # Generates random tensors, checks the TensorFlow result against NumPy (with NumPy promotion
    # adjusted to TensorFlow rules) within a tolerance combined from dtype and record, then
    # compiles and checks again; power on integer dtypes is excluded from the XLA checks.
    # TODO(b/147769803): Remove this skipping
    if lnp_op.__name__ == "kron" and shapes == ((2, 3, 4), (2, 3, 4)):
      self.skipTest("Case disabled because of b/147769803")
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, onp_arrays=False)
    tol = max(jtu.tolerance(dtype, tolerance) for dtype in dtypes)
    tol = functools.reduce(jtu.join_tolerance,
                           [tolerance, tol, jtu.default_tolerance()])
    self._CheckAgainstNumpy(_promote_like_lnp(onp_op, inexact), lnp_op,
                            args_maker, check_dtypes=check_dtypes, tol=tol)
    # tf.math.pow doesn't support int32/int64 on XLA (b/169191476).
    check_xla = not (lnp_op.__name__ == "power" and set(dtypes).intersection(
        (onp.int32, onp.int64)))
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=check_dtypes,
                          atol=tol, rtol=tol,
                          check_incomplete_shape=check_incomplete_shape,
                          check_experimental_compile=check_xla,
                          check_xla_forced_compile=check_xla)

  @named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.test_name, shapes,
                                                      dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes, "name": rec.name,
         "tol": rec.tolerance}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          CombosWithReplacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes)))
      for rec in JAX_OPERATOR_OVERLOADS))
  def testOperatorOverload(self, name, rng_factory, shapes, dtypes, tol):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variables for scalar arguments and empty shapes are computed but unused;
    # the dtype check is always on (the original condition is commented out).
    # Parameterized test of Python operator overloads (add, sub, pow, comparisons, unary ops...)
    # on TensorFlow NumPy arrays. Inputs: operator name, generator factory, shapes, dtypes and a
    # tolerance. Builds a function applying the named operator, then compiles it and compares
    # compiled and eager results with dtype checking. Returns nothing.
    rng = rng_factory()
    # onp and tnp arrays have different type promotion rules; force the use of
    # tnp arrays.
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, onp_arrays=False)
    fun = lambda *xs: getattr(operator, name.strip('_'))(*xs)
    scalar_arg = (jtu.PYTHON_SCALAR_SHAPE in shapes or
                  jtu.NUMPY_SCALAR_SHAPE in shapes or
                  () in shapes)
    empty_shape = any(isinstance(s, tuple) and 0 in s for s in shapes)
    self._CompileAndCheck(
      fun, args_maker, check_dtypes=True, #not scalar_arg and not empty_shape,
      atol=tol, rtol=tol)

  @named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(rec.test_name, shapes,
                                                      dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes, "name": rec.name,
         "op_tolerance": rec.tolerance}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          CombosWithReplacement(rec.shapes, rec.nargs))
        for dtypes in itertools.product(
          *(_valid_dtypes_for_shape(s, rec.dtypes) for s in shapes)))
      for rec in JAX_RIGHT_OPERATOR_OVERLOADS))
  def testRightOperatorOverload(self, name, rng_factory, shapes, dtypes,
                                op_tolerance):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the scalar and empty shape variables are computed but never used, and the
    # dtype check condition is commented out (always checked).
    # Parameterized test of reflected operator methods (radd, rsub, rpow, rmod, ...). Skipped when
    # the second operand is the Python scalar marker. Calls the reflected method on the second
    # argument with the first, using a tolerance from the record and dtypes, then runs the
    # compile-and-check helper. Returns nothing.
    if shapes[1] is jtu.PYTHON_SCALAR_SHAPE:
      raise SkipTest()  # TODO(mattjj): clean up
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, shapes, dtypes, onp_arrays=False)
    fun = lambda fst, snd: getattr(snd, name)(fst)
    tol = max(jtu.tolerance(dtype, op_tolerance) for dtype in dtypes)
    scalar_arg = (jtu.PYTHON_SCALAR_SHAPE in shapes or
                  jtu.NUMPY_SCALAR_SHAPE in shapes or
                  () in shapes)
    empty_shape = any(isinstance(s, tuple) and 0 in s for s in shapes)
    self._CompileAndCheck(
      fun, args_maker, check_dtypes=True, # not scalar_arg and not empty_shape,
      atol=tol, rtol=tol)

  @named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.test_name, shapes, dtypes),
         "rng_factory": rec.rng_factory, "shapes": shapes, "dtypes": dtypes,
         "onp_op": getattr(onp, rec.name), "lnp_op": getattr(tnp, rec.name)}
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          CombosWithReplacement(rec.shapes, rec.nargs))
        for dtypes in filter(
          _dtypes_are_compatible_for_bitwise_ops,
          CombosWithReplacement(rec.dtypes, rec.nargs)))
      for rec in JAX_BITWISE_OP_RECORDS))
  def testBitwiseOp(self, onp_op, lnp_op, rng_factory, shapes, dtypes):
    # [FUNCTION DOC]
    # Parameterized test of bitwise functions on integer dtypes. Compares against NumPy with
    # dtype checking. For bitwise not with a Python scalar it stops after that comparison because
    # compiled dtype choice may differ. Otherwise compiles and checks, with dtype checking turned
    # off when any argument is a scalar (NumPy value-dependent promotion). Returns nothing.
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, shapes, dtypes)
    has_python_scalar = jtu.PYTHON_SCALAR_SHAPE in shapes
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    if onp_op == onp.bitwise_not and has_python_scalar:
      # For bitwise_not with a Python `int`, nje.jit may choose a different
      # dtype for the `int` from onp's choice, which may result in a different
      # result value, so we skip _CompileAndCheck.
      return
    # Numpy does value-dependent dtype promotion on Python/numpy/array scalars
    # which `jit` can't do (when np.result_type is called inside `jit`, tensor
    # values are not available), so we skip dtype check in this case.
    check_dtypes = not(set(shapes) & set([jtu.NUMPY_SCALAR_SHAPE,
                                          jtu.PYTHON_SCALAR_SHAPE, ()]))
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=check_dtypes)

  @named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": "{}_inshape={}_axis={}_dtype={}_keepdims={}".format(
            rec.test_name.capitalize(),
            jtu.format_shape_dtype_string(shape, dtype), axis,
            "None" if out_dtype is None else onp.dtype(out_dtype).name, keepdims),
         "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype, "out_dtype": out_dtype,
         "onp_op": getattr(onp, rec.name), "lnp_op": getattr(tnp, rec.name),
         "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
        for shape in rec.shapes for dtype in rec.dtypes
        for out_dtype in [None] + rec.dtypes
        for axis in set(range(-len(shape), len(shape))) | set([None])
        for keepdims in [False, True])
    for rec in JAX_REDUCER_RECORDS))
  def testReducer(self, onp_op, lnp_op, rng_factory, shape, dtype, out_dtype,
                  axis, keepdims, inexact):
    # [FUNCTION DOC]
    # Parameterized test of reductions that take an output dtype (mean, prod, sum, nan variants).
    # Inputs: operation pair, generator factory, input shape and dtype, output dtype (may be
    # nothing), axis (may be nothing for all axes), keepdims flag and inexact flag. Compares the
    # TensorFlow reduction against NumPy (bfloat16 upcast to float32 for NumPy) and then compiles
    # and checks, with a tolerance taken from the larger of input and output dtype tolerances.
    rng = rng_factory()
    def onp_fun(x):
      # [FUNCTION DOC]
      # HELPER: NumPy reference reduction for the reducer test; casts bfloat16 input to float32,
      # maps a bfloat16 output dtype to float32 and applies the reduction with axis, dtype and
      # keepdims. Returns the reduced array.
      x_cast = x if dtype != tnp.bfloat16 else x.astype(onp.float32)
      t = out_dtype if out_dtype != tnp.bfloat16 else onp.float32
      return onp_op(x_cast, axis, dtype=t, keepdims=keepdims)
    onp_fun = _promote_like_lnp(onp_fun, inexact)
    lnp_fun = lambda x: lnp_op(x, axis, dtype=out_dtype, keepdims=keepdims)
    args_maker = lambda: [rng(shape, dtype)]
    tol_spec = {onp.float16: 1e-2, onp.float32: 1e-3, onp.complex64: 1e-3,
                onp.float64: 1e-5, onp.complex128: 1e-5}
    tol = jtu.tolerance(dtype, tol_spec)
    tol = max(tol, jtu.tolerance(out_dtype, tol_spec)) if out_dtype else tol
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker,
                            check_dtypes=tnp.bfloat16 not in (dtype, out_dtype),
                            tol=tol)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True, atol=tol,
                          rtol=tol)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "{}_inshape={}_axis={}_keepdims={}".format(
          rec.test_name.capitalize(),
          jtu.format_shape_dtype_string(shape, dtype), axis, keepdims),
       "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
       "onp_op": getattr(onp, rec.name), "lnp_op": getattr(tnp, rec.name),
       "axis": axis, "keepdims": keepdims, "inexact": rec.inexact}
      for rec in JAX_REDUCER_NO_DTYPE_RECORDS
      for shape in rec.shapes for dtype in rec.dtypes
      for axis in set(range(-len(shape), len(shape))) | set([None])
      for keepdims in [False, True]))
  def testReducerNoDtype(self, onp_op, lnp_op, rng_factory, shape, dtype, axis,
                         keepdims, inexact):
    # [FUNCTION DOC]
    # Parameterized test of reductions without an output dtype argument (all, any, max, min, var,
    # std, ptp). Compares against NumPy with promotion adjusted, then compiles and checks, both
    # with dtype checking, for the given shape, dtype, axis and keepdims. Returns nothing.
    rng = rng_factory()
    onp_fun = lambda x: onp_op(x, axis, keepdims=keepdims)
    onp_fun = _promote_like_lnp(onp_fun, inexact)
    lnp_fun = lambda x: lnp_op(x, axis, keepdims=keepdims)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "shape": shape, "dtype": dtype, "axis": axis}
      for shape in all_shapes for dtype in all_dtypes
      for axis in set(range(-len(shape), len(shape))) | set([None])))
  def testCountNonzero(self, shape, dtype, axis):
    # [FUNCTION DOC]
    # Test of counting non-zero elements along an axis (or over everything) for random data with
    # many zeros. Compares with NumPy without dtype checking, then compiles and checks with dtype
    # checking. Returns nothing.
    rng = jtu.rand_some_zero()
    onp_fun = lambda x: onp.count_nonzero(x, axis)
    lnp_fun = lambda x: tnp.count_nonzero(x, axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=False)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes for dtype in all_dtypes))
  def testNonzero(self, shape, dtype):
    # [FUNCTION DOC]
    # Test of the nonzero function for random data with many zeros. Compares against NumPy (with
    # the input made at least 1-D) without dtype checking, then compiles and checks while turning
    # off shape evaluation, unknown-rank and XLA checks, since the output shape is data dependent
    # and XLA has no kernel for it. Returns nothing.
    rng = jtu.rand_some_zero()
    onp_fun = lambda x: onp.nonzero(onp.atleast_1d(x))  # pylint: disable=unnecessary-lambda
    lnp_fun = lambda x: tnp.nonzero(x)  # pylint: disable=unnecessary-lambda
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=False)
    # The shapes of `nonzero`'s results are value-dependent, so `eval_on_shapes`
    # won't return concrete shapes.
    # Also, `nonzero` requires a known rank.
    # Turns off XLA check because there are no XLA kernels for `Where`, which
    # XLA can't support because it's output shape is dynamic.
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_eval_on_shapes=False,
        check_incomplete_shape=True, check_unknown_rank=False,
        check_experimental_compile=False, check_xla_forced_compile=False)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "{}_inshape={}_axis={}".format(
          rec.test_name.capitalize(),
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng_factory": rec.rng_factory, "shape": shape, "dtype": dtype,
       "onp_op": getattr(onp, rec.name), "lnp_op": getattr(tnp, rec.name),
       "axis": axis}
      for rec in JAX_ARGMINMAX_RECORDS
      for shape, dtype in _shape_and_dtypes(rec.shapes, rec.dtypes)
      for axis in range(-len(shape), len(shape))))
  def testArgMinMax(self, onp_op, lnp_op, rng_factory, shape, dtype, axis):
    # [FUNCTION DOC]
    # Parameterized test of argmin and argmax along an axis. Skipped for complex128 on GPU.
    # Compares against NumPy (results cast to the default integer type) with dtype checking, then
    # compiles and checks with incomplete-shape checking. Returns nothing.
    rng = rng_factory()
    if dtype == onp.complex128 and jtu.device_under_test() == "gpu":
      raise unittest.SkipTest("complex128 reductions not supported on GPU")

    def onp_fun(array_to_reduce):
      # [FUNCTION DOC]
      # HELPER: NumPy reference; applies the argmin/argmax along the axis and casts the indices to
      # the default integer type. Returns an array.
      return onp_op(array_to_reduce, axis).astype(tnp.int_)

    def lnp_fun(array_to_reduce):
      # [FUNCTION DOC]
      # HELPER: TensorFlow NumPy counterpart; applies the argmin/argmax along the axis and returns
      # the indices.
      return lnp_op(array_to_reduce, axis)

    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype),
          axes),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "axes": axes, "rng_factory": rng_factory}
      for rng_factory in [jtu.rand_default]
      for lhs_shape, rhs_shape, axes in [
          [(2,), (2,), (-1, -1, -1, None)], # scalar output
          [(2, 4), (2, 4), (-1, -1, -1, 0)], # 2D vectors
          [(3, 4), (3, 4), (-1, -1, -1, 0)], # 3D vectors
          [(3, 4), (3, 6, 5, 4), (-1, -1, -1, 0)], # broadcasting
          [(4, 3), (3, 6, 5, 4), (1, 0, -1, None)], # different axes
          [(6, 1, 3), (5, 3), (-1, -1, -1, None)], # more broadcasting
          [(6, 1, 2), (5, 3), (-1, -1, -1, None)], # mixed 2D and 3D vectors
          [(10, 5, 2, 8), (1, 5, 1, 3), (-2, -1, -3, None)], # axes/broadcasting
          [(4, 5, 2), (4, 5, 2), (-1, -1, 0, None)], # axisc should do nothing
          [(4, 5, 2), (4, 5, 2), (-1, -1, -1, None)] # same as before
      ]
      for lhs_dtype, rhs_dtype in CombosWithReplacement(
          minus(number_dtypes, complex_dtypes), 2)))
  def testCross(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of the vector cross product for non-complex dtypes, including
    # broadcasting and different axes. The axes parameter holds four values: axis of first
    # input, axis of second input, axis of output and a common axis (may be nothing). Compares
    # against NumPy (bfloat16 upcast, result cast to promoted dtype) with a tolerance derived from
    # both dtypes, then compiles and checks with incomplete-shape checking. Returns nothing.
    rng = rng_factory()
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    axisa, axisb, axisc, axis = axes
    lnp_fun = lambda a, b: tnp.cross(a, b, axisa, axisb, axisc, axis)
    def onp_fun(a, b):
      # [FUNCTION DOC]
      # HELPER: NumPy reference cross product; upcasts bfloat16 inputs to float32, computes the
      # cross product on the given axes and casts to the promoted dtype of the two input dtypes.
      a = a.astype(onp.float32) if lhs_dtype == tnp.bfloat16 else a
      b = b.astype(onp.float32) if rhs_dtype == tnp.bfloat16 else b
      out = onp.cross(a, b, axisa, axisb, axisc, axis)
      return out.astype(tnp.promote_types(lhs_dtype, rhs_dtype))
    tol_spec = {
        # TODO(wangpeng): dtypes.bfloat16: 3e-1,
        onp.float16: 0.15}
    tol = max(jtu.tolerance(lhs_dtype, tol_spec),
              jtu.tolerance(rhs_dtype, tol_spec))
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True,
                            tol=tol)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True, atol=tol,
                          rtol=tol, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          name,
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "rng_factory": rng_factory}
      for rng_factory in [jtu.rand_default]
      for name, lhs_shape, rhs_shape in [
          ("matrix-scalar", (3, 3), ()),
          ("scalar-matrix", (), (3, 3)),
          ("matrix-vector", (4, 5), (5,)),
          ("vector-matrix", (6,), (6, 4)),
          ("matrix-matrix", (3, 4), (4, 5)),
          ("tensor-vector", (4, 3, 2), (2,)),
          ("vector-tensor", (2,), (3, 2, 4)),
          ("tensor-matrix", (4, 3, 2), (2, 5)),
          ("matrix-tensor", (5, 2), (3, 2, 4)),
          ("tensor-tensor", (2, 3, 4), (5, 4, 1))]
      for lhs_dtype, rhs_dtype in CombosWithReplacement(number_dtypes, 2)))
  def testDot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of the dot product across scalar, vector, matrix and tensor shape
    # combinations and numeric dtypes. Compares against NumPy (bfloat16 upcast, result cast to the
    # NumPy result type) with dtype checking and looser tolerance on TPU. Then compiles and checks
    # with dtype checking disabled when a scalar operand is involved (value-dependent promotion)
    # and XLA checks disabled for integer dtypes (no integer matmul kernels). Returns nothing.
    rng = rng_factory()
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {onp.float16: 1e-2, onp.float32: 1e-5, onp.float64: 1e-14,
           onp.complex128: 1e-14}
    if jtu.device_under_test() == "tpu":
      tol[onp.float32] = tol[onp.complex64] = 2e-1
    def onp_dot(x, y):
      # [FUNCTION DOC]
      # HELPER: NumPy reference for the dot test; upcasts bfloat16 inputs to float32, computes the
      # dot product and forces the result to the NumPy promoted type of the inputs (ignoring
      # cases where the dot result dtype differs from it).
      x = x.astype(onp.float32) if lhs_dtype == tnp.bfloat16 else x
      y = y.astype(onp.float32) if rhs_dtype == tnp.bfloat16 else y
      # `onp.dot(x, y).dtype` sometimes differs from `onp.result_type(x, y)`
      # (e.g. when x is float64[] and y is complex64[3,3], or when x is
      # float16[3,3] and y is int64[]). We ignore this corner case and pretend
      # that they agree.
      return onp.dot(x, y).astype(onp.result_type(x, y))
    self._CheckAgainstNumpy(onp_dot, tnp.dot, args_maker,
                            check_dtypes=True, tol=tol)
    # We disable dtype check in the following cases because `np.dot` does
    # value-dependent type promotion in those cases.
    check_dtypes = () not in (lhs_shape, rhs_shape)
    # XLA lacks int32/int64 MatMul kernels (b/168657656).
    check_xla = not set((lhs_dtype, rhs_dtype)).intersection(
        (onp.int32, onp.int64))
    self._CompileAndCheck(tnp.dot, args_maker, check_dtypes=check_dtypes,
                          atol=tol, rtol=tol, check_incomplete_shape=True,
                          check_experimental_compile=check_xla,
                          check_xla_forced_compile=check_xla)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          name,
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "rng_factory": rng_factory}
      for rng_factory in [jtu.rand_default]
      for name, lhs_shape, rhs_shape in [
          ("vector-vector", (3,), (3,)),
          ("matrix-vector", (3, 3), (3,)),
          ("vector-matrix", (3,), (3, 3)),
          ("matrix-matrix", (3, 3), (3, 3)),
          ("vector-tensor", (3,), (5, 3, 2)),
          ("tensor-vector", (5, 3, 2), (2,)),
          ("matrix-tensor", (5, 2), (3, 2, 4)),
          ("tensor-matrix", (5, 2, 3), (3, 2)),
          ("tensor-tensor", (5, 3, 4), (5, 4, 1)),
          ("tensor-tensor-broadcast", (3, 1, 3, 4), (5, 4, 1))]
      for lhs_dtype, rhs_dtype in CombosWithReplacement(number_dtypes, 2)))
  def testMatmul(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of matrix multiplication across vector, matrix, tensor and broadcast
    # shape combinations. Checks three forms together: the matmul function, the matmul method and
    # the reflected matmul method, against NumPy with dtype checking and a looser tolerance on
    # TPU. Then compiles and checks, with XLA checks disabled for integer dtypes. Returns nothing.
    rng = rng_factory()
    def onp_fun(x, y):
      # [FUNCTION DOC]
      # HELPER: NumPy reference returning a tuple of three results (matmul function, matmul
      # method, reflected matmul method) each cast to the promoted dtype of the two inputs.
      dtype = tnp.promote_types(lhs_dtype, rhs_dtype)
      return (onp.matmul(x, y).astype(dtype),
              onp.array(x).__matmul__(y).astype(dtype),
              onp.array(y).__rmatmul__(x).astype(dtype))
    def lnp_fun(x, y):
      # [FUNCTION DOC]
      # HELPER: TensorFlow NumPy counterpart returning the same three-result tuple (function,
      # method and reflected method forms of matrix multiplication).
      return (tnp.matmul(x, y),
              tnp.array(x).__matmul__(y),
              tnp.array(y).__rmatmul__(x))
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {onp.float16: 1e-2, onp.float32: 2e-2, onp.float64: 1e-12,
           onp.complex128: 1e-12}
    if jtu.device_under_test() == "tpu":
      tol[onp.float32] = tol[onp.complex64] = 4e-2
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker,
                            check_dtypes=True, tol=tol)
    # XLA lacks int32/int64 MatMul kernels (b/168657656).
    check_xla = not set((lhs_dtype, rhs_dtype)).intersection(
        (onp.int32, onp.int64))
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True, atol=tol,
                          rtol=tol, check_incomplete_shape=True,
                          check_experimental_compile=check_xla,
                          check_xla_forced_compile=check_xla)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          name,
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "rng_factory": rng_factory}
      for rng_factory in [jtu.rand_default]
      for name, lhs_shape, rhs_shape in [
          ("vector-vector", (3,), (3,)),
          ("vector-matrix", (9,), (3, 3)),
          ("matrix-matrix", (3, 3), (3, 3)),
          ("tensor-vector", (5, 3, 2), (30,))]
      for lhs_dtype, rhs_dtype in CombosWithReplacement(number_dtypes, 2)))
  @new_test
  def testVDot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of the flattened vector dot product (only runs when the additional-tests
    # flag is on). Compares against NumPy with dtype checking, then compiles and checks, with XLA
    # checks disabled for integer dtypes. Returns nothing.
    rng = rng_factory()
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    tol = {onp.float16: 1e-2, onp.float32: 2e-2, onp.float64: 1e-12,
           onp.complex128: 1e-12}
    self._CheckAgainstNumpy(onp.vdot, tnp.vdot, args_maker,
                            check_dtypes=True, tol=tol)
    # XLA lacks int32/int64 MatMul kernels (b/168657656).
    check_xla = not set((lhs_dtype, rhs_dtype)).intersection(
        (onp.int32, onp.int64))
    self._CompileAndCheck(tnp.vdot, args_maker, check_dtypes=True, atol=tol,
                          rtol=tol, check_incomplete_shape=True,
                          check_experimental_compile=check_xla,
                          check_xla_forced_compile=check_xla)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype),
          axes),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "axes": axes, "rng_factory": rng_factory}
      for rng_factory in [jtu.rand_default]
      for lhs_shape, rhs_shape, axes in [
          [(2, 3, 4), (5, 6, 7), 0],  # from issue #740
          [(2, 3, 4), (3, 4, 5, 6), 2],
          [(2, 3, 4), (5, 4, 3, 6), [1, 2]],
          [(2, 3, 4), (5, 4, 3, 6), [[1, 2], [2, 1]]],
          [(1, 2, 3, 4), (4, 5, 3, 6), [[2, 3], [2, 0]]],
      ]
      for lhs_dtype, rhs_dtype in CombosWithReplacement(number_dtypes, 2)))
  def testTensordot(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, axes, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of tensor contraction over a given number of axes or explicit axis
    # lists. Compares against NumPy (bfloat16 upcast, result cast to promoted dtype) with
    # dtype checking and a looser tolerance on TPU, then compiles and checks with a tolerance
    # recomputed from the dtypes and XLA checks disabled for integer dtypes. Returns nothing.
    rng = rng_factory()
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    lnp_fun = lambda a, b: tnp.tensordot(a, b, axes)
    def onp_fun(a, b):
      # [FUNCTION DOC]
      # HELPER: NumPy reference contraction; upcasts bfloat16 inputs, contracts over the axes and
      # casts the result to the promoted dtype of the two input dtypes.
      a = a if lhs_dtype != tnp.bfloat16 else a.astype(onp.float32)
      b = b if rhs_dtype != tnp.bfloat16 else b.astype(onp.float32)
      dtype = tnp.promote_types(lhs_dtype, rhs_dtype)
      return onp.tensordot(a, b, axes).astype(dtype)
    tol = {onp.float16: 1e-1, onp.float32: 1e-3, onp.float64: 1e-12,
           onp.complex64: 1e-3, onp.complex128: 1e-12}
    if jtu.device_under_test() == "tpu":
      tol[onp.float32] = tol[onp.complex64] = 2e-1
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True,
                            tol=tol)
    # XLA lacks int32/int64 MatMul kernels (b/168657656).
    check_xla = not set((lhs_dtype, rhs_dtype)).intersection(
        (onp.int32, onp.int64))

    tol = {onp.float64: 1e-14, onp.float16: 0.04, onp.complex128: 6e-15}
    tol = max(jtu.tolerance(lhs_dtype, tol), jtu.tolerance(rhs_dtype, tol))
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True,
                          check_incomplete_shape=True,
                          check_experimental_compile=check_xla,
                          check_xla_forced_compile=check_xla,
                          atol = tol,
                          rtol = tol)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_{}".format(
          jtu.format_shape_dtype_string(lhs_shape, lhs_dtype),
          jtu.format_shape_dtype_string(rhs_shape, rhs_dtype)),
       "lhs_shape": lhs_shape, "lhs_dtype": lhs_dtype,
       "rhs_shape": rhs_shape, "rhs_dtype": rhs_dtype,
       "rng_factory": jtu.rand_default}
      # TODO(phawkins): support integer dtypes too.
      for lhs_shape, lhs_dtype in _shape_and_dtypes(all_shapes, inexact_dtypes)
      for rhs_shape, rhs_dtype in _shape_and_dtypes(all_shapes, inexact_dtypes)
      if len(jtu._dims_of_shape(lhs_shape)) == 0
      or len(jtu._dims_of_shape(rhs_shape)) == 0
      or lhs_shape[-1] == rhs_shape[-1]))
  def testInner(self, lhs_shape, lhs_dtype, rhs_shape, rhs_dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of the inner product for inexact dtypes where either operand is a scalar
    # or the last dimensions match. Compares against NumPy without dtype checking (known float32
    # versus float64 disagreements), then compiles and checks, also without dtype checking, using
    # a tolerance from both dtypes (looser on TPU). Returns nothing.
    rng = rng_factory()
    args_maker = lambda: [rng(lhs_shape, lhs_dtype), rng(rhs_shape, rhs_dtype)]
    def onp_fun(lhs, rhs):
      # [FUNCTION DOC]
      # HELPER: NumPy reference inner product; upcasts bfloat16 inputs to float32 and casts the
      # result to the promoted dtype of the two input dtypes.
      lhs = lhs if lhs_dtype != tnp.bfloat16 else lhs.astype(onp.float32)
      rhs = rhs if rhs_dtype != tnp.bfloat16 else rhs.astype(onp.float32)
      dtype = tnp.promote_types(lhs_dtype, rhs_dtype)
      return onp.inner(lhs, rhs).astype(dtype)
    lnp_fun = lambda lhs, rhs: tnp.inner(lhs, rhs)  # pylint: disable=unnecessary-lambda
    tol_spec = {onp.float16: 1e-2, onp.float32: 1e-5, onp.float64: 2e-6}
    if jtu.device_under_test() == "tpu":
      tol_spec[onp.float32] = tol_spec[onp.complex64] = 2e-1
    tol = max(jtu.tolerance(lhs_dtype, tol_spec),
              jtu.tolerance(rhs_dtype, tol_spec))
    # TODO(phawkins): there are float32/float64 disagreements for some inputs.
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=False, atol=tol,
                          rtol=tol, check_incomplete_shape=True)

  @named_parameters(
      jtu.cases_from_list(
          {
              "testcase_name": "_{}_amin={}_amax={}".format(
                  jtu.format_shape_dtype_string(shape, dtype), a_min, a_max
              ),
              "shape": shape,
              "dtype": dtype,
              "a_min": a_min,
              "a_max": a_max,
              "rng_factory": jtu.rand_default,
          }
          for shape in all_shapes
          for dtype in minus(number_dtypes, complex_dtypes)
          for a_min, a_max in [
              (-1, None),
              (None, 1),
              (-onp.ones(1), None),
              (None, onp.ones(1)),
          ]
          + (
              []
              if onp.__version__ >= onp.lib.NumpyVersion("2.0.0")
              else [(-1, 1), (-onp.ones(1), onp.ones(1))]
          )
      )
  )
  def testClipStaticBounds(self, shape, dtype, a_min, a_max, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of the clip function with lower and upper bounds that are scalars,
    # arrays or missing (bound combinations depend on the installed NumPy version). Compares
    # against NumPy and then compiles and checks, both with incomplete-shape checking; dtype
    # checking is disabled for 32-bit scalar inputs because of a NumPy promotion inconsistency.
    rng = rng_factory()
    onp_fun = lambda x: onp.clip(x, a_min=a_min, a_max=a_max)
    lnp_fun = lambda x: tnp.clip(x, a_min=a_min, a_max=a_max)
    args_maker = lambda: [rng(shape, dtype)]
    tol_spec = {onp.float64: 2e-7}
    tol = jtu.tolerance(dtype, tol_spec)
    is_x32_scalar = (dtype in [onp.int32, onp.float32] and
                     shape in [jtu.NUMPY_SCALAR_SHAPE, ()])
    # Turns check_dtypes off if is_x32_scalar is True because there is
    # a weird promotion inconsistency in numpy:
    # ```
    # print(np.result_type(np.ones([], np.int32), 1))
    # print(np.result_type(np.ones([1], np.int32), 1))
    # print(np.result_type(np.int32(1), 1))
    # print(np.result_type(np.int32, 1))
    # print(np.result_type(np.ones([], np.float32), 1))
    # print(np.result_type(np.ones([1], np.float32), 1))
    # print(np.result_type(np.float32(1), 1))
    # print(np.result_type(np.float32, 1))
    # ```
    # >>>
    # int64
    # int32
    # int64
    # int32
    # float64
    # float32
    # float64
    # float32
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker,
                            check_dtypes=not is_x32_scalar, tol=tol)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=not is_x32_scalar,
                          atol=tol, rtol=tol, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_amin={}_amax={}".format(
          jtu.format_shape_dtype_string(shape, dtype), a_min, a_max),
       "shape": shape, "dtype": dtype, "a_min": a_min, "a_max": a_max,
       "rng_factory": jtu.rand_default}
      for shape in array_shapes + [jtu.NUMPY_SCALAR_SHAPE]
      for dtype in minus(number_dtypes, complex_dtypes)
      for a_min, a_max in [(-1, None), (None, 1), (-1, 1),
                           (-onp.ones(1), None),
                           (None, onp.ones(1)),
                           (-onp.ones(1), onp.ones(1))]))
  @new_test
  def testClipAsMethodStaticBounds(
      self, shape, dtype, a_min, a_max, rng_factory):
    # [FUNCTION DOC]
    # Same scenario as the clip function test, but calls clip as an array method on a TensorFlow
    # NumPy array (only runs when the additional-tests flag is on). Compares against NumPy and
    # compiles and checks, with dtype checking off for 32-bit scalar inputs.
    rng = rng_factory()
    onp_fun = lambda x: onp.clip(x, a_min=a_min, a_max=a_max)
    lnp_fun = lambda x: tnp.asarray(x).clip(a_min=a_min, a_max=a_max)
    args_maker = lambda: [rng(shape, dtype)]
    tol_spec = {onp.float64: 2e-7}
    tol = jtu.tolerance(dtype, tol_spec)
    is_x32_scalar = (dtype in [onp.int32, onp.float32] and
                     shape in [jtu.NUMPY_SCALAR_SHAPE, ()])
    # Turns check_dtypes off if is_x32_scalar is True because there is
    # a weird promotion inconsistency in numpy:
    # ```
    # print(np.result_type(np.ones([], np.int32), 1))
    # print(np.result_type(np.ones([1], np.int32), 1))
    # print(np.result_type(np.int32(1), 1))
    # print(np.result_type(np.int32, 1))
    # print(np.result_type(np.ones([], np.float32), 1))
    # print(np.result_type(np.ones([1], np.float32), 1))
    # print(np.result_type(np.float32(1), 1))
    # print(np.result_type(np.float32, 1))
    # ```
    # >>>
    # int64
    # int32
    # int64
    # int32
    # float64
    # float32
    # float64
    # float32
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker,
                            check_dtypes=not is_x32_scalar, tol=tol)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=not is_x32_scalar,
                          atol=tol, rtol=tol, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_decimals={}".format(
          jtu.format_shape_dtype_string(shape, dtype), decimals),
       "shape": shape, "dtype": dtype, "decimals": decimals,
       "rng_factory": jtu.rand_default}
      for shape, dtype in _shape_and_dtypes(
          all_shapes, minus(number_dtypes, complex_dtypes))
      for decimals in [0, 1, -2]))
  def testRoundStaticDecimals(self, shape, dtype, decimals, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of rounding to a given number of decimals (0, 1 or -2) for non-complex
    # dtypes. Integer dtypes with negative decimals are skipped as unimplemented. Compares against
    # NumPy (dtype check off for Python scalars) and then compiles and checks. Returns nothing.
    rng = rng_factory()
    if tnp.issubdtype(dtype, onp.integer) and decimals < 0:
      self.skipTest("Integer rounding with decimals < 0 not implemented")
    onp_fun = lambda x: onp.round(x, decimals=decimals)
    lnp_fun = lambda x: tnp.round(x, decimals=decimals)
    args_maker = lambda: [rng(shape, dtype)]
    tol = {
        # TODO(b/154768983): tnp.bfloat16: 5e-2,
        onp.float16: 1e-2}
    check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker,
                            check_dtypes=check_dtypes, tol=tol)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=check_dtypes,
                          atol=tol, rtol=tol, check_incomplete_shape=True)

  def testOperatorRound(self):
    # [FUNCTION DOC]
    # Test that the built-in rounding operator works on TensorFlow NumPy float32 scalars and
    # arrays. Asserts closeness to the NumPy results for rounding to one decimal, two decimals
    # (with dtype checks) and with no digits argument (dtype check off).
    self.assertAllClose(round(onp.float32(7.532), 1),
                        round(tnp.float32(7.5), 1), check_dtypes=True)
    self.assertAllClose(round(onp.float32(1.234), 2),
                        round(tnp.float32(1.234), 2), check_dtypes=True)
    self.assertAllClose(round(onp.float32(1.234)),
                        round(tnp.float32(1.234)), check_dtypes=False)
    self.assertAllClose(
        round(onp.float32(7.532), 1),
        round(tnp.array(7.5, tnp.float32), 1),
        check_dtypes=True,
    )
    self.assertAllClose(
        round(onp.float32(1.234), 2),
        round(tnp.array(1.234, tnp.float32), 2),
        check_dtypes=True,
    )
    self.assertAllClose(round(onp.float32(1.234)),
                        round(tnp.array(1.234, tnp.float32)),
                        check_dtypes=False)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_mode={}_rpadwidth={}_rconstantvalues={}".format(
          jtu.format_shape_dtype_string(shape, dtype), mode, pad_width_rank,
          constant_values_rank),
       "shape": shape, "dtype": dtype, "mode": mode,
       "pad_width_rank": pad_width_rank,
       "constant_values_rank": constant_values_rank,
       "rng_factory": jtu.rand_default,
       "irng_factory": partial(jtu.rand_int, 3)}
      for mode, constant_values_rank, shapes in [
        ('constant', 0, all_shapes),
        ('constant', 1, all_shapes),
        ('constant', 2, all_shapes),
        ('symmetric', None, nonempty_shapes),
        ('reflect', None, nonempty_shapes),
        ('wrap', None, nonempty_shapes),
      ]
      for shape, dtype in _shape_and_dtypes(shapes, all_dtypes)
      for pad_width_rank in range(3)))
  @jtu.disable
  def testPad(self, shape, dtype, mode, pad_width_rank, constant_values_rank,
              rng_factory, irng_factory):
    # [FUNCTION DOC]
    # Disabled parameterized test of array padding in constant, symmetric, reflect and wrap modes.
    # Generates random integer pad widths of a given rank and optional constant values, compares
    # against NumPy and compiles and checks. Currently never runs because of the disable decorator.
    rng = rng_factory()
    irng = irng_factory()
    pad_width = irng([len(shape), 2][2 - pad_width_rank:], onp.int32)
    def onp_fun(x, kwargs):
      # [FUNCTION DOC]
      # HELPER: NumPy reference padding; returns the input unchanged if the pad width is empty,
      # otherwise pads with the chosen mode and extra keyword arguments (constant values).
      if pad_width.size == 0:
        return x
      return onp.pad(x, pad_width, mode=mode, **kwargs)
    def lnp_fun(x, kwargs):
      # [FUNCTION DOC]
      # HELPER: TensorFlow NumPy padding with the chosen mode and extra keyword arguments.
      return tnp.pad(x, pad_width, mode=mode, **kwargs)

    def args_maker():
      # [FUNCTION DOC]
      # HELPER: builds the inputs; returns a pair of a random array and a mapping holding the
      # constant values (random array of rank-dependent shape) when a constant-values rank is set.
      kwargs = {}
      if constant_values_rank:
        kwargs["constant_values"] = rng(
          [len(shape), 2][2 - constant_values_rank:], dtype)
      return rng(shape, dtype), kwargs

    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape=[{}]_reps={}".format(
          jtu.format_shape_dtype_string(shape, dtype), reps),
       "shape": shape, "dtype": dtype, "reps": reps,
       "rng_factory": jtu.rand_default}
      for reps in [(), (2,), (3, 4), (2, 3, 4)]
      for shape, dtype in _shape_and_dtypes(all_shapes, default_dtypes)
      ))
  def testTile(self, shape, dtype, reps, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of tiling an array by a tuple of repetition counts. Compares against
    # NumPy (dtype check off for Python scalars) with a small float64 tolerance, then compiles
    # and checks with dtype checking. Returns nothing.
    rng = rng_factory()
    onp_fun = lambda arg: onp.tile(arg, reps)
    lnp_fun = lambda arg: tnp.tile(arg, reps)
    args_maker = lambda: [rng(shape, dtype)]
    tol_spec = {onp.float64: 2e-7}
    tol = jtu.tolerance(dtype, tol_spec)
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker,
                            check_dtypes=shape is not jtu.PYTHON_SCALAR_SHAPE,
                            tol=tol)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True, atol=tol,
                          rtol=tol)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_axis={}_baseshape=[{}]_dtypes=[{}]".format(
          axis, ",".join(str(d) for d in base_shape),
          ",".join(onp.dtype(dtype).name for dtype in arg_dtypes)),
       "axis": axis, "base_shape": base_shape, "arg_dtypes": arg_dtypes,
       "rng_factory": jtu.rand_default}
      for num_arrs in [3]
      for arg_dtypes in CombosWithReplacement(default_dtypes, num_arrs)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for axis in range(-len(base_shape)+1, len(base_shape))))
  def testConcatenate(self, axis, base_shape, arg_dtypes, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of concatenating three arrays of differing dtypes along an axis. The
    # array shapes share the base shape but vary the size along the axis (cycling 3, 1, 4).
    # Compares against NumPy (result cast to the promoted dtype of all inputs) with dtype
    # checking, then compiles and checks. Returns nothing.
    rng = rng_factory()
    wrapped_axis = axis % len(base_shape)
    shapes = [base_shape[:wrapped_axis] + (size,) + base_shape[wrapped_axis+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), arg_dtypes)]
    def onp_fun(*args):
      # [FUNCTION DOC]
      # HELPER: NumPy reference; concatenates all arguments along the axis and casts the result
      # to the reduced promotion of all argument dtypes.
      # TODO(nareshmodi): enable once bfloat16 has better support
      # args = [x if x.dtype != bfloat16 else x.astype(onp.float32)
      #         for x in args]
      dtype = functools.reduce(tnp.promote_types, arg_dtypes)
      return onp.concatenate(args, axis=axis).astype(dtype)
    lnp_fun = lambda *args: tnp.concatenate(args, axis=axis)

    def args_maker():
      # [FUNCTION DOC]
      # HELPER: returns a list with one random array for each (shape, dtype) pair.
      return [rng(shape, dtype) for shape, dtype in zip(shapes, arg_dtypes)]

    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_axis={}_baseshape=[{}]_dtypes=[{}]".format(
          axis, ",".join(str(d) for d in base_shape),
          ",".join(onp.dtype(dtype).name for dtype in arg_dtypes)),
       "axis": axis, "base_shape": base_shape, "arg_dtypes": arg_dtypes,
       "rng_factory": jtu.rand_default}
      for arg_dtypes in CombosWithReplacement(default_dtypes, 2)
      for base_shape in [(4,), (3, 4), (2, 3, 4)]
      for axis in range(-len(base_shape)+1, len(base_shape))))
  def testAppend(self, axis, base_shape, arg_dtypes, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of appending values to an array along an axis, for two arrays of
    # (possibly) different dtypes with differing sizes along the axis. Compares against NumPy
    # (bfloat16 upcast, result cast to the promoted dtype) with dtype checking, then compiles and
    # checks with incomplete-shape checking. Returns nothing.
    rng = rng_factory()
    wrapped_axis = axis % len(base_shape)
    shapes = [base_shape[:wrapped_axis] + (size,) + base_shape[wrapped_axis+1:]
              for size, _ in zip(itertools.cycle([3, 1, 4]), arg_dtypes)]
    def onp_fun(arr, values):
      # [FUNCTION DOC]
      # HELPER: NumPy reference append; upcasts bfloat16 inputs to float32, appends along the
      # axis and casts to the promoted dtype of the two argument dtypes.
      arr = arr.astype(onp.float32) if tnp.bfloat16 == arr.dtype else arr
      values = (
          values.astype(onp.float32)
          if tnp.bfloat16 == values.dtype else values)
      out = onp.append(arr, values, axis=axis)
      return out.astype(tnp.promote_types(*arg_dtypes))
    lnp_fun = lambda arr, values: tnp.append(arr, values, axis=axis)

    def args_maker():
      # [FUNCTION DOC]
      # HELPER: returns a list with one random array for each (shape, dtype) pair.
      return [rng(shape, dtype) for shape, dtype in zip(shapes, arg_dtypes)]

    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape=[{}]_axis={}_repeats={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, repeats),
       "axis": axis, "shape": shape, "dtype": dtype, "repeats": repeats,
       "rng_factory": jtu.rand_default}
      for repeats in [0, 1, 2]
      for shape, dtype in _shape_and_dtypes(all_shapes, default_dtypes)
      for axis in [None] + list(range(-len(shape), len(shape)))))
  def testRepeat(self, axis, shape, dtype, repeats, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of repeating elements a fixed number of times (0, 1 or 2) along an axis
    # or over the flattened array. Compares against NumPy with promotion adjusted and dtype
    # checking, then compiles and checks with incomplete-shape checking off. Returns nothing.
    rng = rng_factory()
    onp_fun = lambda arg: onp.repeat(arg, repeats=repeats, axis=axis)
    onp_fun = _promote_like_lnp(onp_fun)
    lnp_fun = lambda arg: tnp.repeat(arg, repeats=repeats, axis=axis)

    args_maker = lambda: [rng(shape, dtype)]

    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=False)

  def testIssue1233(self):
    # [FUNCTION DOC]
    # Regression test for repeat with scalar and array repeat counts, adapted from the NumPy test
    # suite. Uses a 1-D array flattened (axis nothing) with counts given as a scalar, lists and
    # TensorFlow NumPy arrays (including a count of zero), then a 2-by-3 matrix along axis 0 and
    # axis 1. Each case must equal NumPy, and the compiled version is checked with XLA checks off
    # (no XLA kernel for the Where operation used by repeat).
    '''
    Following numpy test suite from `test_repeat` at https://github.com/numpy/numpy/blob/master/numpy/core/tests/test_multiarray.py
    '''
    # pylint: disable=bad-whitespace
    tol = 1e-5

    def test_single(m, args_maker, repeats, axis):
      # [FUNCTION DOC]
      # HELPER: checks one repeat case. Computes repeat with TensorFlow NumPy and NumPy, asserts
      # the results are close with dtype checking, then compiles a repeat function over the same
      # arguments and checks it with incomplete-shape and XLA checks turned off.
      lax_ans = tnp.repeat(m, repeats, axis)
      numpy_ans = onp.repeat(m, repeats, axis)

      self.assertAllClose(lax_ans, numpy_ans, check_dtypes=True, rtol=tol, atol=tol)

      lnp_fun = lambda arg: tnp.repeat(arg, repeats=repeats, axis=axis)
      # Turns off XLA check because there are no XLA kernels for `Where` used by
      # tf.repeat (b/169192730).
      self._CompileAndCheck(
          lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=False,
          check_experimental_compile=False, check_xla_forced_compile=False)

    m = tnp.array([1,2,3,4,5,6])
    args_maker = lambda: [m]

    for repeats in [
        2,
        [1, 3, 2, 1, 1, 2],
        [1, 3, 0, 1, 1, 2],
        [2],
        tnp.array([1, 3, 2, 1, 1, 2]),
        tnp.array([2]),
    ]:
      test_single(m, args_maker, repeats, None)

    m_rect = m.reshape((2,3))
    args_maker = lambda: [m_rect]

    for repeats in [2, [2,1], [2], tnp.array([2,1]), tnp.array([2])]:
      test_single(m_rect, args_maker, repeats, axis=0)

    for repeats in [2, [1,3,2], [2], tnp.array([1,3,2]), tnp.array([2])]:
      test_single(m_rect, args_maker, repeats, axis=1)
    # pylint: enable=bad-whitespace

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "op={}_shape=[{}]_axis={}_out_dtype={}".format(
          op, jtu.format_shape_dtype_string(shape, dtype), axis, out_dtype),
       "axis": axis, "shape": shape, "dtype": dtype, "out_dtype": out_dtype,
       "rng_factory": jtu.rand_default, "lnp_op": getattr(tnp, op),
       "onp_op": getattr(onp, op)}
      for op in ["cumsum", "cumprod"]
      for dtype in default_dtypes
      for out_dtype in default_dtypes
      for shape in all_shapes
      for axis in [None] + list(range(-len(shape), len(shape)))))
  def testCumSumProd(self, axis, shape, dtype, out_dtype, onp_op, lnp_op, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of cumulative sum and cumulative product with an output dtype, over axes
    # (or flattened). Compares against NumPy with dtype checking and a tolerance from both dtypes,
    # then compiles and checks; XLA is disabled for int64 output (no kernels) and relative
    # tolerance is loosened for float16 output.
    rng = rng_factory()
    onp_fun = lambda arg: onp_op(arg, axis=axis, dtype=out_dtype)
    lnp_fun = lambda arg: lnp_op(arg, axis=axis, dtype=out_dtype)

    args_maker = lambda: [rng(shape, dtype)]

    tol = max(jtu.tolerance(dtype), jtu.tolerance(out_dtype))
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True,
                            tol=tol)
    # XLA lacks int64 Cumsum/Cumprod kernels (b/168841378).
    check_xla = out_dtype != onp.int64
    rtol = None
    if out_dtype == onp.float16:
      rtol = 2e-3
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, rtol=rtol,
        check_incomplete_shape=True,
        check_experimental_compile=check_xla,
        check_xla_forced_compile=check_xla)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_dtype={}_m={}_n={}_k={}".format(
          onp.dtype(dtype).name, m, n, k),
       "m": m, "n": n, "k": k, "dtype": dtype, "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for n in [0, 4]
      for m in [None, 0, 1, 3, 4]
      for k in list(range(-4, 4))))
  def testTri(self, m, n, k, dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of building a lower-triangular matrix of ones with n rows, optional m
    # columns, diagonal offset k and dtype. Takes no array inputs; compares the result against
    # NumPy with dtype checking, then compiles and checks. Returns nothing.
    rng = rng_factory()
    onp_fun = lambda: onp.tri(n, M=m, k=k, dtype=dtype)
    lnp_fun = lambda: tnp.tri(n, M=m, k=k, dtype=dtype)
    args_maker = lambda: []
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_shape={}_k={}".format(
          op, jtu.format_shape_dtype_string(shape, dtype), k),
       "dtype": dtype, "shape": shape, "op": op, "k": k,
       "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for op in ["tril", "triu"]
      for k in list(range(-3, 3))))
  def testTriLU(self, dtype, shape, op, k, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of extracting the lower (tril) or upper (triu) triangle of arrays with at
    # least two dimensions for diagonal offsets from -3 to 2. Compares against NumPy with dtype
    # checking and compiles and checks with incomplete-shape checking off (not supported yet).
    rng = rng_factory()
    onp_fun = lambda arg: getattr(onp, op)(arg, k=k)
    lnp_fun = lambda arg: getattr(tnp, op)(arg, k=k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    # Incomplete shape support is not implemented at the moment.
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=False)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_ndim={}_n={}".format(ndim, n),
       "ndim": ndim, "n": n}
      for ndim in [0, 1, 4]
      for n in [0, 1, 7]))
  def testDiagIndices(self, ndim, n):
    # [FUNCTION DOC]
    # Parameterized test of the diagonal index generator for sizes 0, 1, 7 and dimensions 0, 1, 4.
    # Asserts that the TensorFlow NumPy result equals the NumPy result using NumPy's equality
    # assertion helper. Returns nothing.
    onp.testing.assert_equal(onp.diag_indices(n, ndim),
                             tnp.diag_indices(n, ndim))

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_k={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k),
       "dtype": dtype, "shape": shape, "k": k, "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) in (1, 2)]
      for k in list(range(-4, 4))))
  def testDiag(self, shape, dtype, k, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of the diag function (extract a diagonal from a matrix or build a matrix
    # from a vector) with offsets from -4 to 3. Compares against NumPy and compiles and checks,
    # with dtype and incomplete-shape checking.
    rng = rng_factory()
    onp_fun = lambda arg: onp.diag(arg, k)
    lnp_fun = lambda arg: tnp.diag(arg, k)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_offset={}_axis1={}_axis2={}".format(
          jtu.format_shape_dtype_string(shape, dtype), offset, axis1, axis2),
       "dtype": dtype, "shape": shape, "offset": offset, "axis1": axis1,
       "axis2": axis2, "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for axis1 in range(-len(shape), len(shape))
      for axis2 in [a for a in range(-len(shape), len(shape))
                    if a % len(shape) != axis1 % len(shape)]
      for offset in list(range(-4, 4))))
  def testDiagonal(self, shape, dtype, offset, axis1, axis2, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of extracting diagonals from arrays of rank two or more for different
    # offsets and distinct pairs of axes (negative axes included). Compares against NumPy and
    # compiles and checks, with dtype and incomplete-shape checking.
    rng = rng_factory()
    onp_fun = lambda arg: onp.diagonal(arg, offset, axis1, axis2)
    lnp_fun = lambda arg: tnp.diagonal(arg, offset, axis1, axis2)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_n={}".format(onp.dtype(dtype).name, n),
       "dtype": dtype, "n": n}
      for dtype in default_dtypes
      for n in list(range(4))))
  def testIdentity(self, n, dtype):
    # [FUNCTION DOC]
    # Parameterized test of building an n-by-n identity matrix of a given dtype (n from 0 to 3).
    # Takes no array inputs; compares against NumPy and compiles and checks.
    onp_fun = lambda: onp.identity(n, dtype)
    lnp_fun = lambda: tnp.identity(n, dtype)
    args_maker = lambda: []
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_dtype_{}_offset={}_axis1={}_axis2={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          out_dtype, offset, axis1, axis2),
       "dtype": dtype, "out_dtype": out_dtype, "shape": shape, "offset": offset,
       "axis1": axis1, "axis2": axis2, "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for out_dtype in [None] + number_dtypes
      for shape in [shape for shape in all_shapes if len(shape) >= 2]
      for axis1 in range(-len(shape), len(shape))
      for axis2 in range(-len(shape), len(shape))
      if (axis1 % len(shape)) != (axis2 % len(shape))
      for offset in list(range(-4, 4))))
  def testTrace(self, shape, dtype, out_dtype, offset, axis1, axis2, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of the matrix trace (sum along a diagonal) for arrays of rank two or
    # more, various offsets, axis pairs and optional output dtypes. Compares against NumPy and
    # compiles and checks, with dtype and incomplete-shape checking.
    rng = rng_factory()
    def onp_fun(arg):
      # [FUNCTION DOC]
      # HELPER: NumPy reference trace. For a bfloat16 output dtype it computes in float32 and
      # casts the result to bfloat16; otherwise it passes the output dtype through.
      if out_dtype == tnp.bfloat16:
        return onp.trace(arg, offset, axis1, axis2, onp.float32).astype(
            tnp.bfloat16
        )
      else:
        return onp.trace(arg, offset, axis1, axis2, out_dtype)
    lnp_fun = lambda arg: tnp.trace(arg, offset, axis1, axis2, out_dtype)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_test_name_suffix("", [shape] * len(dtypes), dtypes), axis),
       "shape": shape, "axis": axis, "dtypes": dtypes, "rng_factory": rng_factory}
      for dtypes in [
        [onp.float32],
        [onp.float32, onp.float32],
        [onp.float32, onp.int32, onp.float32],
        [onp.float32, onp.int64, onp.float32],
        [onp.float32, onp.int32, onp.float64],
      ]
      for shape in [(), (2,), (3, 4), (1, 100)]
      for axis in range(-len(shape), len(shape) + 1)
      for rng_factory in [jtu.rand_default]))
  def testStack(self, shape, axis, dtypes, rng_factory):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the arguments of the comparison helper are passed in swapped order (the
    # TensorFlow function is given where the NumPy reference is expected and vice versa), so the
    # roles of reference and tested function are reversed; the check is symmetric so it still
    # works. Parameterized test of stacking arrays of mixed dtypes along a new axis; checks the
    # result with dtype checking, then compiles and checks with incomplete-shape checking.
    rng = rng_factory()
    args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]
    onp_fun = _promote_like_lnp(partial(onp.stack, axis=axis))
    lnp_fun = partial(tnp.stack, axis=axis)
    self._CheckAgainstNumpy(lnp_fun, onp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_{}".format(
          op, jtu.format_test_name_suffix("", [shape] * len(dtypes), dtypes)),
       "shape": shape, "op": op, "dtypes": dtypes, "rng_factory": rng_factory}
      for op in ["hstack", "vstack", "dstack"]
      for dtypes in [
        [onp.float32],
        [onp.float32, onp.float32],
        [onp.float32, onp.int32, onp.float32],
        [onp.float32, onp.int64, onp.float32],
        [onp.float32, onp.int32, onp.float64],
      ]
      for shape in [(), (2,), (3, 4), (1, 100), (2, 3, 4)]
      for rng_factory in [jtu.rand_default]))
  def testHVDStack(self, shape, op, dtypes, rng_factory):
    # [FUNCTION DOC]
    # KNOWN ISSUE: same swapped reference/tested argument order as the stack test; the TensorFlow
    # function is passed as the reference position. Parameterized test of horizontal, vertical and
    # depth stacking of arrays with mixed dtypes. Compares the two implementations with dtype
    # checking, then compiles and checks with incomplete-shape checking.
    rng = rng_factory()
    args_maker = lambda: [[rng(shape, dtype) for dtype in dtypes]]
    onp_fun = _promote_like_lnp(getattr(onp, op))
    lnp_fun = getattr(tnp, op)
    self._CheckAgainstNumpy(lnp_fun, onp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outdtype={}".format(
          jtu.format_shape_dtype_string(shape, fill_value_dtype),
          onp.dtype(out_dtype).name if out_dtype else "None"),
       "shape": shape, "fill_value_dtype": fill_value_dtype,
       "out_dtype": out_dtype, "rng_factory": jtu.rand_default}
      for shape in array_shapes + [3, onp.array(7, dtype=onp.int32)]
      for fill_value_dtype in default_dtypes
      for out_dtype in [None] + default_dtypes))
  def testFull(self, shape, fill_value_dtype, out_dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of creating an array of a given shape filled with a scalar fill value,
    # with optional output dtype; shape may be a tuple, an integer or a NumPy integer scalar.
    # The fill value is the only runtime input. Compares against NumPy and compiles and checks.
    rng = rng_factory()
    onp_fun = lambda fill_value: onp.full(shape, fill_value, dtype=out_dtype)
    lnp_fun = lambda fill_value: tnp.full(shape, fill_value, dtype=out_dtype)
    args_maker = lambda: [rng((), fill_value_dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(
    jtu.cases_from_list(
      {"testcase_name": ("_op={}_shape={}_dtype={}").format(op, shape, dtype),
       "onp_op": getattr(onp, op), "lnp_op": getattr(tnp, op),
       "shape": shape, "dtype": dtype}
      for op in ["zeros", "ones"]
      for shape in [2, (), (2,), (3, 0), onp.array((4, 5, 6), dtype=onp.int32),
                    onp.array(4, dtype=onp.int32)]
      for dtype in all_dtypes))
  def testZerosOnes(self, onp_op, lnp_op, shape, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the random generator created at the start is never used.
    # Parameterized test of the zeros and ones constructors for various shape forms (integer,
    # tuple, empty, NumPy arrays) and all dtypes. Binds the shape and dtype to both functions,
    # compares against NumPy with dtype checking and compiles and checks.
    rng = jtu.rand_default()
    def args_maker():
      # [FUNCTION DOC]
      # HELPER: returns an empty list because the tested functions take no array inputs.
      return []
    onp_op = partial(onp_op, shape, dtype)
    lnp_op = partial(lnp_op, shape, dtype)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_op, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_filldtype={}_outdtype={}".format(
          jtu.format_shape_dtype_string(shape, in_dtype),
          onp.dtype(fill_value_dtype).name,
          onp.dtype(out_dtype).name),
       "shape": shape, "in_dtype": in_dtype,
       "fill_value_dtype": fill_value_dtype, "out_dtype": out_dtype,
       "rng_factory": jtu.rand_default}
      for shape in array_shapes
      for in_dtype in default_dtypes
      for fill_value_dtype in default_dtypes
      for out_dtype in default_dtypes))
  def testFullLike(self, shape, in_dtype, fill_value_dtype, out_dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of creating a filled array shaped like an input array, with a scalar fill
    # value and an explicit output dtype. Inputs are a random array and a random scalar. Compares
    # against NumPy with dtype checking and compiles and checks.
    rng = rng_factory()
    onp_fun = lambda x, fill_value: onp.full_like(
        x, fill_value, dtype=out_dtype
    )
    lnp_fun = lambda x, fill_value: tnp.full_like(
        x, fill_value, dtype=out_dtype
    )
    args_maker = lambda: [rng(shape, in_dtype), rng((), fill_value_dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_{}sections".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, num_sections),
       "shape": shape, "num_sections": num_sections, "axis": axis,
       "dtype": dtype, "rng_factory": jtu.rand_default}
      for shape, axis, num_sections in [
          ((3,), 0, 3), ((12,), 0, 3), ((12, 4), 0, 4), ((12, 4), 1, 2),
          ((2, 3, 4), -1, 2), ((2, 3, 4), -2, 3)]
      for dtype in default_dtypes))
  def testSplitStaticInt(self, shape, num_sections, axis, dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of splitting an array into an equal number of sections along an axis.
    # Compares the list of pieces against NumPy and compiles and checks, with dtype and
    # incomplete-shape checking.
    rng = rng_factory()
    onp_fun = lambda x: onp.split(x, num_sections, axis=axis)
    lnp_fun = lambda x: tnp.split(x, num_sections, axis=axis)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}_{}sections".format(
          jtu.format_shape_dtype_string(shape, dtype), axis, num_sections),
       "shape": shape, "num_sections": num_sections, "axis": axis,
       "dtype": dtype, "rng_factory": jtu.rand_default}
      for shape, axis, num_sections in [
          ((12, 4), 0, 4), ((12, 4), 1, 2),
          ((2, 3, 4), 2, 2), ((4, 3, 4), 0, 2)]
      for dtype in default_dtypes))
  def testHVDSplit(self, shape, num_sections, axis, dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of vertical, horizontal and depth splitting (chosen from the axis value
    # 0, 1 or 2) into equal sections. Compares against NumPy and compiles and checks.
    rng = rng_factory()
    def fn(module, axis):
      # [FUNCTION DOC]
      # HELPER: picks the split function from a module: vertical split for axis 0, horizontal
      # split for axis 1, depth split otherwise (asserting that the axis is 2).
      # Raises an assertion error for any other axis.
      if axis == 0:
        return module.vsplit
      elif axis == 1:
        return module.hsplit
      else:
        assert axis == 2
        return module.dsplit

    onp_fun = lambda x: fn(onp, axis)(x, num_sections)
    lnp_fun = lambda x: fn(tnp, axis)(x, num_sections)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}_order={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype),
          order),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "order": order, "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for order in ["C", "F"]
      for arg_shape, out_shape in [
          (jtu.NUMPY_SCALAR_SHAPE, (1, 1, 1)),
          ((), (1, 1, 1)),
          ((7, 0), (0, 42, 101)),
          ((3, 4), 12),
          ((3, 4), (12,)),
          ((3, 4), -1),
          ((2, 1, 4), (-1,)),
          ((2, 2, 4), (2, 8))
      ]))
  def testReshape(self, arg_shape, out_shape, dtype, order, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of reshaping with C or Fortran order for scalar, empty and regular
    # inputs and shapes given as integer, tuple or containing -1. Compares against NumPy and
    # compiles and checks with dtype and incomplete-shape checking.
    rng = rng_factory()
    onp_fun = lambda x: onp.reshape(x, out_shape, order=order)
    lnp_fun = lambda x: tnp.reshape(x, out_shape, order=order)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_outshape={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype),
          jtu.format_shape_dtype_string(out_shape, dtype)),
       "arg_shape": arg_shape, "out_shape": out_shape, "dtype": dtype,
       "rng_factory": jtu.rand_default}
      for dtype in default_dtypes
      for arg_shape, out_shape in [
          ((7, 0), (0, 42, 101)),
          ((2, 1, 4), (-1,)),
          ((2, 2, 4), (2, 8))
      ]))
  def testReshapeMethod(self, arg_shape, out_shape, dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of the array reshape method with the dimensions given as separate
    # arguments, compared with NumPy's reshape function. Compiles and checks as well, with dtype
    # and incomplete-shape checking.
    rng = rng_factory()
    onp_fun = lambda x: onp.reshape(x, out_shape)
    lnp_fun = lambda x: x.reshape(*out_shape)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_expanddim={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), dim),
       "arg_shape": arg_shape, "dtype": dtype, "dim": dim,
       "rng_factory": jtu.rand_default}
      for arg_shape in [(), (3,), (3, 4)]
      for dtype in default_dtypes
      for dim in range(-len(arg_shape)+1, len(arg_shape))))
  def testExpandDimsStaticDim(self, arg_shape, dtype, dim, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of inserting a size-one axis at a static position. Compares against
    # NumPy and compiles and checks with dtype and incomplete-shape checking.
    rng = rng_factory()
    onp_fun = lambda x: onp.expand_dims(x, dim)
    lnp_fun = lambda x: tnp.expand_dims(x, dim)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_axes=({},{})".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), ax1, ax2),
       "arg_shape": arg_shape, "dtype": dtype, "ax1": ax1, "ax2": ax2,
       "rng_factory": jtu.rand_default}
      for arg_shape, ax1, ax2 in [
          ((3, 4), 0, 1), ((3, 4), 1, 0), ((3, 4, 5), 1, 2),
          ((3, 4, 5), -1, -2), ((3, 4, 5), 0, 1)]
      for dtype in default_dtypes))
  def testSwapAxesStaticAxes(self, arg_shape, dtype, ax1, ax2, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of swapping two axes (including negative axes). Compares against NumPy
    # and compiles and checks with dtype and incomplete-shape checking.
    rng = rng_factory()
    onp_fun = lambda x: onp.swapaxes(x, ax1, ax2)
    lnp_fun = lambda x: tnp.swapaxes(x, ax1, ax2)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axes=({},{})".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), source, destination),
       "arg_shape": arg_shape, "dtype": dtype, "source": source,
       "destination": destination, "rng_factory": jtu.rand_default}
      for arg_shape, source, destination in [
          (tuple(range(6)), (0, 2), (3, 5)),
          (tuple(range(6)), (0, 2), (-1, -3)),
          (tuple(range(6)), (-6, -4),(3, 5)),
          (tuple(range(6)), (-6, -4), (-1, -3)),
          (tuple(range(6)), 0, 4),
          (tuple(range(6)), -6, -2),
          (tuple(range(6)), tuple(range(6)), tuple(range(6))),
          (tuple(range(6)), tuple(range(6)), tuple(reversed(range(6)))),
          (tuple(range(6)), (), ()),
      ] for dtype in default_dtypes))
  @new_test
  def testMoveaxisStaticAxes(self, arg_shape, dtype, source, destination,
                             rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of moving axes of a six-dimensional array from source positions to
    # destination positions (single or sequences, positive or negative, also empty). Only runs
    # when the additional-tests flag is on. Compares against NumPy and compiles and checks.
    rng = rng_factory()
    onp_fun = lambda x: onp.moveaxis(x, source, destination)
    lnp_fun = lambda x: tnp.moveaxis(x, source, destination)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_inshape={}_axis={}".format(
          jtu.format_shape_dtype_string(arg_shape, dtype), ax),
       "arg_shape": arg_shape, "dtype": dtype, "ax": ax,
       "rng_factory": jtu.rand_default}
      for arg_shape, ax in [
          ((3, 1), None),
          ((3, 1), 1),
          ((1, 3, 1), (0, 2)),
          ((1, 4, 1), (0,))]
      for dtype in default_dtypes))
  def testSqueeze(self, arg_shape, dtype, ax, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of removing size-one axes, either all (axis nothing) or specific axes
    # given as an integer or tuple. Compares against NumPy and compiles and checks.
    rng = rng_factory()
    onp_fun = lambda x: onp.squeeze(x, ax)
    lnp_fun = lambda x: tnp.squeeze(x, ax)
    args_maker = lambda: [rng(arg_shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_axis={}_weights={}_returned={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          axis,
          (None if weights_shape is None else jtu.format_shape_dtype_string(weights_shape, dtype)),
          returned),
       "rng_factory": jtu.rand_default, "shape": shape, "dtype": dtype, "axis": axis,
       "weights_shape": weights_shape, "returned": returned}
      for shape, dtype in _shape_and_dtypes(nonempty_shapes, number_dtypes)
      for axis in set(range(-len(shape), len(shape))) | set([None])
      # `weights_shape` is either `None`, same as the averaged axis, or same as
      # that of the input
      for weights_shape in ([None, shape] if axis is None or len(shape) == 1
                            else [None, (shape[axis],), shape])
      for returned in [False, True]))
  def testAverage(self, shape, dtype, axis, weights_shape, returned, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of the weighted average, with optional weights (same shape as input or
    # one-dimensional along the axis) and optional return of the weight sum. Compares against
    # NumPy with promotion adjusted to at least floating point and a dtype-dependent tolerance;
    # a zero-division error in the comparison causes the test to be skipped. Then compiles and
    # checks with incomplete-shape checking.
    rng = rng_factory()
    if weights_shape is None:
      onp_fun = lambda x: onp.average(x, axis, returned=returned)
      lnp_fun = lambda x: tnp.average(x, axis, returned=returned)
      args_maker = lambda: [rng(shape, dtype)]
    else:
      onp_fun = lambda x, weights: onp.average(x, axis, weights, returned)
      lnp_fun = lambda x, weights: tnp.average(x, axis, weights, returned)
      args_maker = lambda: [rng(shape, dtype), rng(weights_shape, dtype)]
    onp_fun = _promote_like_lnp(onp_fun, inexact=True)
    tol = {
        # TODO(b/154768983): tnp.bfloat16: 1e-1,
        onp.float16: 1e-1, onp.float32: 1e-3, onp.float64: 2e-7,
        onp.complex64: 1e-3, onp.complex128: 1e-10,
    }
    check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE
    try:
      self._CheckAgainstNumpy(
          onp_fun, lnp_fun, args_maker, check_dtypes=check_dtypes, tol=tol)
    except ZeroDivisionError:
      self.skipTest("don't support checking for ZeroDivisionError")
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=check_dtypes,
                          rtol=tol, atol=tol, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_arg{}_ndmin={}".format(i, ndmin),
       "arg": arg, "ndmin": ndmin, "dtype": dtype}
      for i, (arg, dtype) in enumerate([
          ([True, False, True], tnp.bool_),
          (3., tnp.float64),
          ([1, 2, 3], tnp.int_),
          ([1., 2., 3.], tnp.float64),
          ([[1, 2], [3, 4], [5, 6]], tnp.int_),
          ([[1, 2.], [3, 4], [5, 6]], tnp.float64),
          ([[1., 2j], [3., 4.], [5., 6.]], tnp.complex128),
          ([[3, onp.array(2, dtype=tnp.float64), 1],
           onp.arange(3., dtype=tnp.float64)], tnp.float64),  # pylint: disable=bad-continuation
      ])
      for ndmin in [None, onp.ndim(arg), onp.ndim(arg) + 1, onp.ndim(arg) + 2]))
  def testArray(self, arg, ndmin, dtype):
    # [FUNCTION DOC]
    # Parameterized test of array construction from Python values (booleans, floats, ints, nested
    # lists, complex numbers, mixed NumPy elements) with optional minimum number of dimensions.
    # Compares the constructed array with NumPy using the canonical dtype and then compiles and
    # checks with the argument treated as static.
    args_maker = lambda: [arg]
    dtype = tnp.canonicalize_dtype(dtype)
    if ndmin is not None:
      onp_fun = partial(onp.array, ndmin=ndmin, dtype=dtype)
      lnp_fun = partial(tnp.array, ndmin=ndmin)
    else:
      onp_fun = partial(onp.array, dtype=dtype)
      lnp_fun = tnp.array
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True,
                          check_incomplete_shape=True, static_argnums=[0])

  def testIssue121(self):
    # [FUNCTION DOC]
    # Regression test: asserts that NumPy's scalar check reports false for a TensorFlow NumPy
    # array created from the integer 3 (it must not be treated as a scalar).
    assert not onp.isscalar(tnp.array(3))

  @jtu.disable
  def testArrayMethod(self):
    # [FUNCTION DOC]
    # Disabled test that an object exposing the NumPy array conversion protocol is converted by
    # the array constructor. Defines a local class returning the float 3 and asserts the result
    # equals 3. Never runs because of the disable decorator.
    class arraylike(object):
      # [CLASS DOC]
      # Local fake array-like class used only by the array-method test. Declares a float32 dtype
      # attribute and implements the array conversion protocol, always returning the float 3.
      dtype = onp.float32

      def __array__(self, dtype=None):
        # [FUNCTION DOC]
        # Conversion protocol hook; ignores the requested dtype and returns the float value 3.
        return 3.0

    a = arraylike()
    ans = tnp.array(a)
    assert ans == 3.0

  @jtu.skip_on_devices("tpu")  # TODO(b/32368900): TPUs don't support uint8 yet.
  @jtu.disable
  def testMemoryView(self):
    # [FUNCTION DOC]
    # Disabled test (also skipped on TPU) that building an array from a byte buffer containing the
    # byte 0x2A gives an unsigned 8-bit array holding 42. Never runs because of the disable
    # decorator.
    ans = tnp.array(bytearray(b"\x2a"))
    self.assertAllClose(
        ans, onp.array([0x2A], dtype=onp.uint8), check_dtypes=True
    )

  def testAllClose(self):
    # [FUNCTION DOC]
    # Test of the allclose function inside a jit-compiled function. Draws a seeded random matrix
    # and vector; a helper checks that every pair of corresponding elements of two lists are close
    # (tolerance 1e-3). Asserts true for identical inputs both eagerly and compiled, and false
    # when the second vector is doubled.
    rng = onp.random.RandomState(0)
    x = rng.randn(2, 2)
    y = rng.randn(2)

    def same(list1, list2):
      # [FUNCTION DOC]
      # HELPER: returns a scalar true value if every element pair from the two lists is close
      # within an absolute and relative tolerance of 1e-3.
      allclose = functools.partial(tnp.allclose, atol=1e-3, rtol=1e-3)
      elements_close = list(map(allclose, list1, list2))
      return tnp.all(tnp.array(elements_close))

    csame = nje.jit(same)

    a1 = same((x, y), (x, y))
    a2 = csame((x, y), (x, y))
    a3 = csame((x, y), (x, 2 * y))

    self.assertTrue(a1)
    self.assertTrue(a2)
    self.assertFalse(a3)

  @jtu.skip_on_devices("tpu")  # TODO(mattjj): investigate this failure
  @jtu.disable
  def testOnesBroadcastingConstantHandler(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead code. The test is disabled and immediately skips; the remaining body
    # refers to a name for a jit API that is not imported in this module and to compiler
    # internals, so it could not run anyway.
    # Intended scenario: a constant array of ones with zero strides should be materialized with a
    # broadcast op when added to a traced value.
    # TODO(mattjj): update this test for jax3
    self.skipTest("test needs jax3 update")

    def fun(x):
      # [FUNCTION DOC]
      # KNOWN ISSUE: unreachable because the enclosing test skips first. Intended to add a
      # constant ones array to the input while patching the builder's broadcast hook to record
      # that it was called, asserting that it was, and returning the sum.
      ones = tnp.ones((3, 4))
      assert isinstance(ones, onp.ndarray) and ones.strides == (0, 0)

      # To check that the constant handler generates a Broadcast for stride-zero
      # arrays, we monkey-patch the client instance.
      # TODO(mattjj): once we have better HLO dumping and inspecting facilities,
      # we can check the HLO more directly.
      c = x._node.c
      Broadcast = c.Broadcast  # pylint: disable=invalid-name
      was_called = []
      c.Broadcast = lambda *args: was_called.append(True) or Broadcast(*args)
      out = x + ones  # the ndarray constant handler should call Broadcast here
      assert was_called, "Broadcast was not called."

      return out

    fun = api.jit(fun)
    out_val = fun(tnp.ones(4))
    self.assertAllClose(out_val, onp.full((3, 4), 2.), check_dtypes=False)

  def testZeroStridesConstantHandler(self):
    # [FUNCTION DOC]
    # Test that a NumPy constant with zero strides (created by broadcasting a seeded random array
    # to shape 3x2x3x4x5x6) can be captured by a jit-compiled function. The compiled function
    # multiplies a scalar by the constant; asserts the result is close to 3 times the constant.
    raw_const = onp.random.RandomState(0).randn(1, 2, 1, 1, 5, 1)
    const = onp.broadcast_to(raw_const, (3, 2, 3, 4, 5, 6))

    def fun(x):
      # [FUNCTION DOC]
      # HELPER: multiplies the input by the captured broadcast constant and returns the product.
      return x * const

    fun = nje.jit(fun)
    out_val = fun(3.)
    self.assertAllClose(out_val, 3. * const, check_dtypes=False)

  def testIsInstanceNdarrayDuringTracing(self):
    # [FUNCTION DOC]
    # Test that inside a jit-compiled function the traced argument is an instance of the
    # TensorFlow NumPy array class. Calls the compiled function with a NumPy array of three ones;
    # the assertion lives inside the compiled function.
    arr = onp.ones(3)

    @nje.jit
    def f(x):
      # [FUNCTION DOC]
      # HELPER: asserts the argument is a TensorFlow NumPy array while tracing and returns the sum
      # of its elements.
      self.assertIsInstance(x, tnp.ndarray)
      return tnp.sum(x)

    f(arr)

  @jtu.disable
  def testNonArrayErrorMessage(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled; also uses a jit API name that is not imported in this module, so it
    # would fail with a name error if enabled.
    # Intended scenario: passing a Python list to add or dot should raise a type error, both
    # eagerly and under jit.
    x = [1., 2.]
    y = onp.array([3., 4.])

    def g(x, y):
      # [FUNCTION DOC]
      # HELPER: adds the two inputs with the TensorFlow NumPy add function.
      return tnp.add(x, y)

    def f(x, y):
      # [FUNCTION DOC]
      # HELPER: computes the dot product of the two inputs with TensorFlow NumPy.
      return tnp.dot(x, y)

    self.assertRaises(TypeError, lambda: g(x, y))
    self.assertRaises(TypeError, lambda: f(x, y))
    self.assertRaises(TypeError, lambda: api.jit(g)(x, y))
    self.assertRaises(TypeError, lambda: api.jit(f)(x, y))

  @jtu.disable
  def testAbstractionErrorMessage(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled; uses a jit API name that is not imported in this module.
    # Intended scenario: under jit, using a traced value as a loop count or in a Python
    # conditional should raise a type error.

    @api.jit
    def f(x, n):
      # [FUNCTION DOC]
      # HELPER: squares the input repeatedly n times (loop bound is a traced value) and returns it.
      for _ in range(n):
        x = x * x
      return x

    self.assertRaises(TypeError, lambda: f(3., 3))

    @api.jit
    def g(x):
      # [FUNCTION DOC]
      # HELPER: returns twice the input if it is positive, otherwise the input plus two (a Python
      # conditional on a traced value).
      if x > 0.:
        return x * 2
      else:
        return x + 2

    self.assertRaises(TypeError, lambda: g(3.))

  @jtu.disable
  def testTracingPrimitiveWithNoTranslationErrorMessage(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled and skips immediately; the rest is dead code that uses an unimported
    # jit name and a private TensorFlow NumPy helper that may not exist.
    # Intended scenario: a not-implemented function works eagerly but raises a not-implemented
    # error when traced.
    # TODO(mattjj): update this for jax3
    self.skipTest("test needs jax3 update")
    foo = tnp._not_implemented(lambda x: x)

    # No error if there's no tracing.
    foo(onp.arange(3))

    cfoo = api.jit(foo)
    self.assertRaises(NotImplementedError, lambda: cfoo(onp.arange(3)))

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype), axis),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "axis": axis}
      for shape in [(3,), (2, 3)]
      for dtype in default_dtypes
      for axis in list(range(-len(shape), len(shape))) + [None]  # Test negative axes
      for rng_factory in [jtu.rand_default]))
  def testFlip(self, shape, dtype, axis, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of reversing an array along one axis (negative axes and all-axes case
    # included). Compares against NumPy and compiles and checks with dtype and incomplete-shape
    # checking.
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    lnp_op = lambda x: tnp.flip(x, axis)
    onp_op = lambda x: onp.flip(x, axis)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_op, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype}
      for shape in [(3,), (2, 3), (3, 2, 4)]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testFlipud(self, shape, dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of flipping an array up-down (reversing the first axis). Compares against
    # NumPy and compiles and checks with dtype and incomplete-shape checking.
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    lnp_op = lambda x: tnp.flipud(x)
    onp_op = lambda x: onp.flipud(x)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_op, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype}
      for shape in [(3, 2), (2, 3), (3, 2, 4)]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testFliplr(self, shape, dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of flipping an array left-right (reversing the second axis). Compares
    # against NumPy and compiles and checks with dtype and incomplete-shape checking.
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    lnp_op = lambda x: tnp.fliplr(x)
    onp_op = lambda x: onp.fliplr(x)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_op, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_k={}_axes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k, axes),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "k": k, "axes": axes}
      for shape, axes in [
          [(2, 3), (0, 1)],
          [(2, 3), (1, 0)],
          [(4, 3, 2), (0, 2)],
          [(4, 3, 2), (2, 1)],
      ]
      for k in range(-3, 4)
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testRot90(self, shape, dtype, k, axes, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of rotating an array by 90 degrees k times (k from -3 to 3) in the plane
    # given by a pair of axes. Compares against NumPy and compiles and checks.
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    lnp_op = lambda x: tnp.rot90(x, k, axes)
    onp_op = lambda x: onp.rot90(x, k, axes)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_op, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_k={}_axes={}".format(
          jtu.format_shape_dtype_string(shape, dtype), k, axes),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "k": k,
       "axes": axes}
      for shape, axes in [
          [(2, 3), (-2, -1)],
          [(2, 3), (-2, 1)],
          [(4, 3, 2), (-1, -2)],
          [(4, 3, 2), (2, -2)],
      ]
      for k in range(-3, 4)
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  @new_test
  # These tests are only added as a separate test from testRot90 since we would
  # like to measure coverage directly against the existing baseline. Once we
  # stop measuring that, we can combine this test with the above.
  def testRot90Additional(self, shape, dtype, k, axes, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of 90-degree rotation with negative axis specifications (only runs when
    # the additional-tests flag is on). Kept separate from the main rotation test to measure
    # coverage against a baseline. Compares against NumPy and compiles and checks.
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    lnp_op = lambda x: tnp.rot90(x, k, axes)
    onp_op = lambda x: onp.rot90(x, k, axes)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_op, args_maker, check_dtypes=True, check_incomplete_shape=True)

  # TODO(mattjj): test infix operator overrides

  def testRavel(self):
    # [FUNCTION DOC]
    # Test of the array ravel method on a seeded random 3x4 float32 matrix: only the compile and
    # check helper is used (dtype and incomplete-shape checking). Returns nothing.
    rng = onp.random.RandomState(0)
    args_maker = lambda: [rng.randn(3, 4).astype("float32")]
    self._CompileAndCheck(lambda x: x.ravel(), args_maker, check_dtypes=True,
                          check_incomplete_shape=True)

  def testAstype(self):
    # [FUNCTION DOC]
    # Test of casting a float32 matrix to int32 through the array astype method. The same method
    # call is used as both the reference and the tested function (it is applied to NumPy and to
    # TensorFlow NumPy inputs by the comparison helper), then compiled and checked.
    rng = onp.random.RandomState(0)
    args_maker = lambda: [rng.randn(3, 4).astype("float32")]
    op = lambda x: x.astype(tnp.int32)
    self._CheckAgainstNumpy(op, op, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        op, args_maker, check_dtypes=True, check_incomplete_shape=True)

  # TODO(mattjj): test other ndarray-like method overrides

  def testOnpMean(self):
    # [FUNCTION DOC]
    # Regression test (JAX issue 125): NumPy's mean applied directly to a TensorFlow NumPy 3x3
    # identity matrix plus zero should give one third; asserts closeness without dtype checking.
    # from https://github.com/google/jax/issues/125
    x = tnp.add(tnp.eye(3, dtype=tnp.float64), 0.)
    ans = onp.mean(x)
    self.assertAllClose(ans, onp.array(1./3), check_dtypes=False)

  @jtu.disable
  def testArangeOnFloats(self):
    # [FUNCTION DOC]
    # Disabled regression test (JAX issue 145): arange with float start, stop and step should
    # match NumPy's float64 result. Never runs because of the disable decorator.
    # from https://github.com/google/jax/issues/145
    expected = onp.arange(0.0, 1.0, 0.1, dtype=tnp.float64)
    ans = tnp.arange(0.0, 1.0, 0.1)
    self.assertAllClose(expected, ans, check_dtypes=True)

  def testSortManually(self):
    # [FUNCTION DOC]
    # Hand-written tests of sorting (chosen to avoid ties): a 1-D array, and a 2-D array with no
    # axis argument, an explicit nothing axis and axis 0. Each case is compared with NumPy for
    # both eager and jit-compiled sorting.

    def _test(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: runs sort eagerly and through jit (second positional argument static) and compares
      # both with NumPy's sort, asserting closeness with dtype checking.

      raw_ans = tnp.sort(*args, **kwargs)
      fn_ans = nje.jit(tnp.sort, static_argnums=(1,))(*args, **kwargs)
      expected = onp.sort(*args, **kwargs)

      self.assertAllClose(expected, raw_ans, check_dtypes=True)
      self.assertAllClose(expected, fn_ans, check_dtypes=True)

    # manual tests for sort are nice because we don't have to worry about ties.
    # lax.sort is tested combinatorially.
    _test(onp.array([16, 15, 23, 42, 8, 4]))
    _test(onp.array([[1, 4], [3, 1]]), None)
    _test(onp.array([[1, 4], [3, 1]]))
    _test(onp.array([[1, 4], [3, 1]]), 0)

  def testArgsortManually(self):
    # [FUNCTION DOC]
    # Hand-written tests of argsort on a 1-D array and a 2-D array with axis 0, axis 1, nothing
    # and the default axis. Each case is compared with NumPy for eager and jit-compiled calls.

    def _test(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: runs argsort eagerly and through jit (second positional argument static) and
      # compares both with NumPy's argsort, asserting closeness with dtype checking.

      raw_ans = tnp.argsort(*args, **kwargs)
      fn_ans = nje.jit(tnp.argsort, static_argnums=(1,))(*args, **kwargs)
      expected = onp.argsort(*args, **kwargs)

      self.assertAllClose(expected, raw_ans, check_dtypes=True)
      self.assertAllClose(expected, fn_ans, check_dtypes=True)

    _test(onp.array([16, 15, 23, 42, 8, 4]))
    _test(onp.array([[16, 15, 23], [42, 8, 4]]), 0)
    _test(onp.array([[16, 15, 23], [42, 8, 4]]), 1)
    _test(onp.array([[16, 15, 23], [42, 8, 4]]), None)
    _test(onp.array([[16, 15, 23], [42, 8, 4]]))

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_shifts={}_axis={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          shifts, axis),
       "rng_factory": rng_factory, "shape": shape, "dtype": dtype, "shifts": shifts,
       "axis": axis}
      for dtype in all_dtypes
      for shape in [(3, 4), (3, 4, 5), (7, 4, 0)]
      for shifts, axis in [
        (3, None),
        (1, 1),
        ((3,), (0,)),
        ((-2,), (-2,)),
        ((1, 2), (0, -1))
      ]
      for rng_factory in [jtu.rand_default]))
  def testRoll(self, shape, dtype, shifts, axis, rng_factory):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comparison helper receives the TensorFlow function in the reference
    # position and NumPy in the tested position (swapped); the check is symmetric.
    # Parameterized test of rolling elements by shifts along axes (shift passed as a runtime
    # array input). Compares the two implementations with dtype checking and compiles and checks.
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), onp.array(shifts)]
    lnp_op = partial(tnp.roll, axis=axis)
    onp_op = partial(onp.roll, axis=axis)
    self._CheckAgainstNumpy(lnp_op, onp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_op, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_index={}_axis={}_mode={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          jtu.format_shape_dtype_string(index_shape, index_dtype),
          axis, mode),
       "rng_factory": rng_factory, "rng_indices_factory": rng_indices_factory,
       "shape": shape, "index_shape": index_shape, "dtype": dtype,
       "index_dtype": index_dtype, "axis": axis, "mode": mode}
      for shape in [(3,), (3, 4), (3, 4, 5)]
      for index_shape in scalar_shapes + [(3,), (2, 1, 3)]
      for axis in itertools.chain(range(-len(shape), len(shape)), [None])
      for dtype in all_dtypes
      for index_dtype in int_dtypes
      for mode in ['wrap', 'clip']
      for rng_factory in [jtu.rand_default]
      for rng_indices_factory in [partial(jtu.rand_int, -5, 5)]))
  def testTake(self, shape, dtype, index_shape, index_dtype, axis, mode,
               rng_factory, rng_indices_factory):
    # [FUNCTION DOC]
    # KNOWN ISSUE: reference and tested functions are passed to the comparison helper in swapped
    # order (symmetric check). Parameterized test of taking elements by index with wrap or clip
    # modes along an axis (or flattened). Random data and random integer indices in -5..5 are
    # used; compares the implementations with dtype checking, then compiles and checks.
    def args_maker():
      # [FUNCTION DOC]
      # HELPER: returns a pair of a random data array and a random integer index array; the
      # random generators are created after this function is defined but before it is called.
      x = rng(shape, dtype)
      i = rng_indices(index_shape, index_dtype)
      return x, i

    rng = rng_factory()
    rng_indices = rng_indices_factory()
    lnp_op = lambda x, i: tnp.take(x, i, axis=axis, mode=mode)
    onp_op = lambda x, i: onp.take(x, i, axis=axis, mode=mode)
    self._CheckAgainstNumpy(lnp_op, onp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_op, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_ishape={}_axis={}".format(
          jtu.format_shape_dtype_string(x_shape, dtype), i_shape, axis),
       "rng_factory": rng_factory, "x_shape": x_shape, "i_shape": i_shape, "dtype": dtype,
       "axis": axis}
      for x_shape, i_shape in filter(
        _shapes_are_equal_length,
        filter(_shapes_are_broadcast_compatible,
               CombosWithReplacement(nonempty_nonscalar_array_shapes, 2)))
      for axis in itertools.chain(range(len(x_shape)), [-1], [None])
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testTakeAlongAxis(self, x_shape, i_shape, dtype, axis, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of taking values along an axis using an index array of equal rank. The
    # index shape is triple the data size along the axis (or flat when the axis is nothing) so it
    # does not just broadcast, and indices include valid negative values. The NumPy comparison is
    # done only if the installed NumPy has the function; the compile-and-check always runs.
    rng = rng_factory()
    i_shape = onp.array(i_shape)
    if axis is None:
      i_shape = [onp.prod(i_shape, dtype=onp.int64)]
    else:
      # Test the case where the size of the axis doesn't necessarily broadcast.
      i_shape[axis] *= 3
      i_shape = list(i_shape)
    def args_maker():
      # [FUNCTION DOC]
      # HELPER: returns a random data array and a random int32 index array mapped into the valid
      # range from minus (n minus 1) to n minus 1, where n is the size along the axis (or total).
      x = rng(x_shape, dtype)
      n = onp.prod(x_shape, dtype=onp.int32) if axis is None else x_shape[axis]
      i = rng(i_shape, onp.int32) % (2 * n - 1) - (n - 1)
      return x, i

    lnp_op = lambda x, i: tnp.take_along_axis(x, i, axis=axis)

    if hasattr(onp, "take_along_axis"):
      onp_op = lambda x, i: onp.take_along_axis(x, i, axis=axis)
      self._CheckAgainstNumpy(lnp_op, onp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(lnp_op, args_maker, check_dtypes=True,
                          check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_n={}_increasing={}".format(
          jtu.format_shape_dtype_string([shape], dtype),
          n, increasing),
       "dtype": dtype, "shape": shape, "n": n, "increasing": increasing,
       "rng_factory": jtu.rand_default}
      for dtype in inexact_dtypes
      for shape in [0, 5]
      for n in [2, 4]
      for increasing in [False, True]))
  def testVander(self, shape, dtype, n, increasing, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of building a Vandermonde matrix from a vector of inexact dtype with n
    # columns in increasing or decreasing powers. Dtype checking is off because NumPy returns
    # float64 for all float inputs. Compares against NumPy and compiles and checks.
    rng = rng_factory()
    def onp_fun(arg):
      # [FUNCTION DOC]
      # HELPER: NumPy reference Vandermonde matrix; upcasts bfloat16 input to float32 first.
      arg = arg.astype(onp.float32) if dtype == tnp.bfloat16 else arg
      return onp.vander(arg, N=n, increasing=increasing)
    lnp_fun = lambda arg: tnp.vander(arg, N=n, increasing=increasing)
    args_maker = lambda: [rng([shape], dtype)]
    # np.vander seems to return float64 for all floating types. We could obey
    # those semantics, but they seem like a bug.
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=False,
                            tol={onp.float32: 1e-3})
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=False, check_incomplete_shape=True,
        rtol={onp.complex128: 2e-15})

  @named_parameters(jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix("nan_to_num", [shape],
                                                      [dtype]),
         "rng_factory": jtu.rand_some_inf_and_nan, "shape": shape,
         "dtype": dtype}
        for shape in all_shapes
        for dtype in inexact_dtypes))
  @jtu.disable
  def testNanToNum(self, rng_factory, shape, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled, and refers to a dtype-utility name that is not imported in this
    # module, so it would fail with a name error if enabled.
    # Intended scenario: replacing NaN and infinity values in inexact arrays, compared with NumPy
    # (with bfloat16 handled by hand) and compiled.
    rng = rng_factory()
    dtype = onp.dtype(dtypes.canonicalize_dtype(dtype)).type
    def onp_fun(x):
      # [FUNCTION DOC]
      # HELPER: NumPy reference for the replacement; for bfloat16 replaces NaN by zero and
      # infinities by the largest/smallest finite values by hand, otherwise uses NumPy's function
      # and casts to the dtype.
      if dtype == tnp.bfloat16:
        x = onp.where(onp.isnan(x), dtype(0), x)
        x = onp.where(onp.isposinf(x), tnp.finfo(dtype).max, x)
        x = onp.where(onp.isneginf(x), tnp.finfo(dtype).min, x)
        return x
      else:
        return onp.nan_to_num(x).astype(dtype)

    args_maker = lambda: [rng(shape, dtype)]
    check_dtypes = shape is not jtu.PYTHON_SCALAR_SHAPE
    self._CheckAgainstNumpy(onp_fun, tnp.nan_to_num, args_maker,
                            check_dtypes=check_dtypes)
    self._CompileAndCheck(tnp.nan_to_num, args_maker,
                          check_dtypes=check_dtypes)

  @named_parameters(jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix("ix_", shapes, dtypes),
         "rng_factory": jtu.rand_default, "shapes": shapes, "dtypes": dtypes}
        for shapes, dtypes in (
          ((), ()),
          (((7,),), (onp.int32,)),
          (((3,), (4,)), (onp.int32, onp.int32)),
          (((3,), (1,), (4,)), (onp.int32, onp.int32, onp.int32)),
        )))
  def testIx_(self, rng_factory, shapes, dtypes):
    # [FUNCTION DOC]
    # Parameterized test of building an open mesh (index grid) from several 1-D integer
    # sequences, including the empty case. Compares against NumPy and compiles and checks.
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype)
                          for shape, dtype in zip(shapes, dtypes)]
    self._CheckAgainstNumpy(onp.ix_, tnp.ix_, args_maker,
                            check_dtypes=True)
    self._CompileAndCheck(
        tnp.ix_, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(jtu.cases_from_list(
        {"testcase_name":
           "_op={}_a_shape={}_q_shape={}_axis={}_keepdims={}".format(
             op,
             jtu.format_shape_dtype_string(a_shape, a_dtype),
             jtu.format_shape_dtype_string(q_shape, q_dtype),
             axis, keepdims),
         "a_rng": jtu.rand_default(), "q_rng": q_rng, "op": op,
         "a_shape": a_shape, "a_dtype": a_dtype,
         "q_shape": q_shape, "q_dtype": q_dtype, "axis": axis,
         "keepdims": keepdims}
        for (op, q_rng) in (
          ("percentile", jtu.rand_uniform(low=0., high=100.)),
          ("quantile", jtu.rand_uniform(low=0., high=1.)),
          ("median", jtu.rand_uniform(low=0., high=1.)),
        )
        for a_dtype in float_dtypes
        for a_shape, axis in (
          ((7,), None),
          ((47, 7), 0),
          ((4, 101), 1),
        )
        for q_dtype in [onp.float32]
        for q_shape in scalar_shapes + [(4,)]
        for keepdims in [False, True]))
  @jtu.disable
  def testQuantile(self, op, a_rng, q_rng, a_shape, a_dtype, q_shape, q_dtype,
                   axis, keepdims):
    # [FUNCTION DOC]
    # Disabled parameterized test of percentile, quantile and median over float data with scalar
    # or vector quantile levels along an axis with keepdims. Skips quantile on very old NumPy.
    # Compares against NumPy (bfloat16 upcast) without dtype checking, then compiles and checks.
    # Never runs because of the disable decorator.
    if op == "quantile" and numpy_version < (1, 15):
      raise SkipTest("Numpy < 1.15 does not have np.quantile")
    if op == "median":
      args_maker = lambda: [a_rng(a_shape, a_dtype)]
    else:
      args_maker = lambda: [a_rng(a_shape, a_dtype), q_rng(q_shape, q_dtype)]

    def onp_fun(*args):
      # [FUNCTION DOC]
      # HELPER: NumPy reference; upcasts bfloat16 arguments to float32 and calls the named
      # function with the axis and keepdims options.
      args = [x if tnp.result_type(x) != tnp.bfloat16 else
              onp.asarray(x, onp.float32) for x in args]
      return getattr(onp, op)(*args, axis=axis, keepdims=keepdims)
    lnp_fun = partial(getattr(tnp, op), axis=axis, keepdims=keepdims)
    # TODO(phawkins): we currently set dtype=False because we aren't as
    # aggressive about promoting to float64. It's not clear we want to mimic
    # Numpy here.
    tol_spec = {onp.float32: 2e-4, onp.float64: 5e-6}
    tol = max(jtu.tolerance(a_dtype, tol_spec),
              jtu.tolerance(q_dtype, tol_spec))
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=False,
                            tol=tol)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True, rtol=tol)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}".format(
          jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in all_shapes for dtype in all_dtypes))
  def testWhereOneArgument(self, shape, dtype):
    # [FUNCTION DOC]
    # Parameterized test of the one-argument where function (indices of non-zero elements) for
    # data with many zeros. Compares against the NumPy-compatibility wrapper without dtype
    # checking, then compiles and checks with shape evaluation, unknown-rank and XLA checks off
    # because the output shape is dynamic.
    rng = jtu.rand_some_zero()
    onp_fun = lambda x: np_where(x)
    lnp_fun = lambda x: tnp.where(x)
    args_maker = lambda: [rng(shape, dtype)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=False)
    # Turns off XLA check because there are no XLA kernels for `Where`, which
    # XLA can't support because it's output shape is dynamic.
    self._CompileAndCheck(
        tnp.where,
        args_maker,
        check_dtypes=True,
        check_eval_on_shapes=False,
        check_incomplete_shape=True,
        check_unknown_rank=False,
        check_experimental_compile=False, check_xla_forced_compile=False)

  @named_parameters(jtu.cases_from_list(
    {"testcase_name": "_{}".format("_".join(
        jtu.format_shape_dtype_string(shape, dtype)
        for shape, dtype in zip(shapes, dtypes))),
     "rng_factory": jtu.rand_default, "shapes": shapes, "dtypes": dtypes}
    for shapes in filter(_shapes_are_broadcast_compatible,
                         CombosWithReplacement(all_shapes, 3))
    for dtypes in CombosWithReplacement(all_dtypes, 3)))
  def testWhereThreeArgument(self, rng_factory, shapes, dtypes):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the first generator created is unused; a second one is created for the inputs.
    # Parameterized test of the three-argument where (condition, x, y) with broadcastable shapes
    # and arbitrary dtypes. Compares against NumPy with promotion of x and y adjusted, with dtype
    # checking, then compiles and checks.
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng_factory(), shapes, dtypes)
    def onp_fun(cond, x, y):
      # [FUNCTION DOC]
      # HELPER: NumPy reference; applies where with the condition after promoting x and y to a
      # common dtype following TensorFlow NumPy rules.
      return _promote_like_lnp(partial(onp.where, cond))(x, y)
    self._CheckAgainstNumpy(onp_fun, tnp.where, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        tnp.where, args_maker, check_dtypes=True, check_incomplete_shape=True)

  def testWhereScalarPromotion(self):
    # [FUNCTION DOC]
    # Test that where with a Python integer scalar and a float32 array returns float32 (the scalar
    # does not upcast the result). Asserts the result dtype equals float32.
    x = tnp.where(tnp.array([True, False]), 3,
                  tnp.ones((2,), dtype=tnp.float32))
    self.assertEqual(x.dtype, onp.dtype(onp.float32))

  @named_parameters(jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix("", shapes,
                                                      (onp.bool_,) * n + dtypes),
         "rng_factory": jtu.rand_default, "shapes": shapes, "dtypes": dtypes}
        for n in range(0, 3)
        for shapes in filter(
          _shapes_are_broadcast_compatible,
          CombosWithReplacement(all_shapes, 2 * n + 1))
        for dtypes in CombosWithReplacement(all_dtypes, n + 1)))
  def testSelect(self, rng_factory, shapes, dtypes):
    # [FUNCTION DOC]
    # Parameterized test of selecting from choices by conditions with 0 to 2 condition/choice
    # pairs plus a default, over broadcastable shapes and mixed dtypes. Compares against NumPy
    # (dtype check off because of known float32/float64 mismatches), then compiles and checks.
    rng = rng_factory()
    n = len(dtypes) - 1
    def args_maker():
      # [FUNCTION DOC]
      # HELPER: returns a list of random boolean conditions, a list of random choice arrays and a
      # random default array, matching the shapes and dtypes under test.
      condlist = [rng(shape, onp.bool_) for shape in shapes[:n]]
      choicelist = [rng(shape, dtype)
                    for shape, dtype in zip(shapes[n:-1], dtypes[:n])]
      default = rng(shapes[-1], dtypes[-1])
      return condlist, choicelist, default
    # TODO(phawkins): float32/float64 type mismatches
    def onp_fun(condlist, choicelist, default):
      # [FUNCTION DOC]
      # HELPER: NumPy reference select; upcasts bfloat16 choices, converts all choices and the
      # default to the TensorFlow NumPy promoted dtype, then calls select.
      choicelist = [x if tnp.bfloat16 != tnp.result_type(x)
                    else x.astype(onp.float32) for x in choicelist]
      dtype = tnp.result_type(default, *choicelist).as_numpy_dtype
      return onp.select(condlist,
                        [onp.asarray(x, dtype=dtype) for x in choicelist],
                        onp.asarray(default, dtype=dtype))
    self._CheckAgainstNumpy(onp_fun, tnp.select, args_maker,
                            check_dtypes=False)
    self._CompileAndCheck(tnp.select, args_maker, check_dtypes=True,
                          check_incomplete_shape=True,
                          rtol={onp.float64: 1e-7, onp.complex128: 1e-7})

  @jtu.disable
  def testIssue330(self):
    # [FUNCTION DOC]
    # Disabled regression test (JAX issue 330): a full array filled with a one-element array's
    # item must not crash and its element must equal one. Never runs (disable decorator).
    x = tnp.full((1, 1), tnp.array([1])[0])  # doesn't crash
    self.assertEqual(x[0, 0], 1)

  @jtu.disable
  def testScalarDtypePromotion(self):
    # [FUNCTION DOC]
    # Disabled test that adding a Python scalar to a float32 identity matrix gives the same dtype
    # in TensorFlow NumPy as in NumPy. Never runs (disable decorator).
    orig_numpy_result = (1 + onp.eye(1, dtype=onp.float32)).dtype
    jax_numpy_result = (1 + tnp.eye(1, dtype=tnp.float32)).dtype
    self.assertEqual(orig_numpy_result, jax_numpy_result)

  @jtu.disable
  def testSymmetrizeDtypePromotion(self):
    # [FUNCTION DOC]
    # Disabled test that symmetrizing a float32 identity matrix (add transpose, divide by two)
    # keeps the same dtype in TensorFlow NumPy as in NumPy. Never runs (disable decorator).
    x = onp.eye(3, dtype=onp.float32)
    orig_numpy_result = ((x + x.T) / 2).dtype

    x = tnp.eye(3, dtype=tnp.float32)
    jax_numpy_result = ((x + x.T) / 2).dtype
    self.assertEqual(orig_numpy_result, jax_numpy_result)

  @jtu.disable
  def testIssue347(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled and uses a gradient API name that is not imported in this module.
    # Intended scenario (JAX issue 347): the gradient of a function using sqrt and where on a
    # matrix containing a zero row must not contain NaN values.
    # https://github.com/google/jax/issues/347
    def test_fail(x):
      # [FUNCTION DOC]
      # HELPER: computes row norms, replaces norms not above 0.5 by ones and returns the sum.
      x = tnp.sqrt(tnp.sum(x ** 2, axis=1))
      ones = tnp.ones_like(x)
      x = tnp.where(x > 0.5, x, ones)
      return tnp.sum(x)

    x = tnp.array([[1, 2], [3, 4], [0, 0]], dtype=tnp.float64)
    result = api.grad(test_fail)(x)
    assert not onp.any(onp.isnan(result))

  def testIssue453(self):
    # [FUNCTION DOC]
    # Regression test (JAX issue 453): reshaping 1..6 into 3x2 with Fortran order must equal the
    # NumPy result, with dtype checking.
    # https://github.com/google/jax/issues/453
    a = onp.arange(6) + 1
    ans = tnp.reshape(a, (3, 2), order="F")
    expected = onp.reshape(a, (3, 2), order="F")
    self.assertAllClose(ans, expected, check_dtypes=True)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name": "_op={}_dtype={}".format(op, pytype.__name__),
       "pytype": pytype, "dtype": dtype, "op": op}
      for pytype, dtype in [(int, tnp.int_), (float, tnp.float64),
                            (bool, tnp.bool_), (complex, tnp.complex128)]
      for op in ["atleast_1d", "atleast_2d", "atleast_3d"]))
  def testAtLeastNdLiterals(self, pytype, dtype, op):
    # [FUNCTION DOC]
    # Parameterized regression test (JAX issue 634) of the at-least-1D/2D/3D functions applied to
    # Python literals (int, float, bool, complex). The input is the literal 2 converted to the
    # type; compares against NumPy (cast to the matching dtype) and compiles and checks.
    # Fixes: https://github.com/google/jax/issues/634
    onp_fun = lambda arg: getattr(onp, op)(arg).astype(dtype)
    lnp_fun = lambda arg: getattr(tnp, op)(arg)
    args_maker = lambda: [pytype(2)]
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  def testLongLong(self):
    # [FUNCTION DOC]
    # Test that an identity function compiled with jit accepts a NumPy long-long scalar and returns
    # a 64-bit integer with the same value (compared with dtype checking).
    self.assertAllClose(
        onp.int64(7), nje.jit(lambda x: x)(onp.longlong(7)), check_dtypes=True
    )

  def testArange(self):
    # [FUNCTION DOC]
    # Test of arange with one, two and three positional arguments, negative steps, float steps and
    # explicit float/int dtypes (cases inspired by dask). Each is compared with NumPy with dtype
    # checking; calling with no arguments must raise a type error. Commented-out parts relate to
    # JAX specific behavior and are not executed.
    # test cases inspired by dask tests at
    # https://github.com/dask/dask/blob/master/dask/array/tests/test_creation.py#L92
    self.assertAllClose(tnp.arange(77),
                        onp.arange(77, dtype=tnp.int_), check_dtypes=True)
    self.assertAllClose(tnp.arange(2, 13),
                        onp.arange(2, 13, dtype=tnp.int_), check_dtypes=True)
    self.assertAllClose(tnp.arange(4, 21, 9),
                        onp.arange(4, 21, 9, dtype=tnp.int_), check_dtypes=True)
    self.assertAllClose(tnp.arange(53, 5, -3),
                        onp.arange(53, 5, -3, dtype=tnp.int_),
                        check_dtypes=True)
    # TODO(mattjj): make these tests work when enable_x64=True
    self.assertAllClose(
        tnp.arange(77, dtype=float),
        onp.arange(77, dtype=float),
        check_dtypes=True)
    self.assertAllClose(
        tnp.arange(2, 13, dtype=int),
        onp.arange(2, 13, dtype=int),
        check_dtypes=True)
    self.assertAllClose(tnp.arange(0, 1, -0.5),
                        onp.arange(0, 1, -0.5, dtype=tnp.float64),
                        check_dtypes=True)

    self.assertRaises(TypeError, lambda: tnp.arange())

    # # The following have been disabled since they test JAX specific behavior
    # # test that tnp.arange(N) doesn't instantiate an ndarray
    # self.assertFalse(type(tnp.arange(77)) == type(onp.arange(77)))
    # self.assertTrue(type(tnp.arange(77)) == type(lax.iota(onp.int32, 77)))

    # # test that tnp.arange(N, dtype=int32) doesn't instantiate an ndarray
    # self.assertFalse(type(tnp.arange(77, dtype=tnp.int32)) ==
    #                  type(onp.arange(77, dtype=onp.int32)))
    # self.assertTrue(type(tnp.arange(77, dtype=tnp.int32)) ==
    #                 type(lax.iota(onp.int32, 77)))

  def testIssue830(self):
    # [FUNCTION DOC]
    # Regression test (JAX issue 830): arange with complex64 dtype yields an array whose dtype is
    # complex64.
    a = tnp.arange(4, dtype=tnp.complex64)
    self.assertEqual(a.dtype, tnp.complex64)

  def testIssue728(self):
    # [FUNCTION DOC]
    # Regression test (JAX issue 728): large identity matrices (5000 and 1050) are built correctly;
    # asserts allclose with NumPy for 5000 and that the sum of the difference for 1050 is zero.
    assert tnp.allclose(tnp.eye(5000), onp.eye(5000))
    self.assertEqual(0, onp.sum(tnp.eye(1050) - onp.eye(1050)))

  def testIssue746(self):
    # [FUNCTION DOC]
    # Regression test (JAX issue 746): reshaping the result of arange to 3 by 4 must not crash.
    # There are no assertions.
    tnp.arange(12).reshape(3, 4)  # doesn't crash

  def testIssue764(self):
    # [FUNCTION DOC]
    # Regression test (JAX issue 764): the gradient of the sum of tanh over linspace(190, 200, 4)
    # is tiny but must match reference values computed in float64 (about 1e-165 down to 1e-174);
    # asserts closeness without dtype checking.
    x = tnp.linspace(190, 200, 4)
    f = nje.grad(lambda x: tnp.sum(tnp.tanh(x)))
    # Expected values computed with autograd in float64 precision.
    expected = onp.array([3.71669453e-165, 4.72999108e-168, 6.01954653e-171,
                          7.66067839e-174], onp.float64)
    self.assertAllClose(f(x), expected, check_dtypes=False)

  @jtu.disable
  def testIssue776(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled; it uses an unimported gradient API name and calls a functional
    # in-place update accessor that NumPy arrays do not have, so it would fail if enabled.
    # Intended scenario: gradient of a function that scatter-adds into ones is zeros.
    """Tests that the scatter-add transpose rule instantiates symbolic zeros."""
    def f(u):
      # [FUNCTION DOC]
      # HELPER: performs a scatter-add into an array of ones and then ignores it, returning the
      # constant 7 (so the gradient with respect to the input should be zero).
      _ = onp.ones(10,).at[[2, 4, 5]].add(u)
      # The transpose rule for lax.tie_in returns a symbolic zero for its first
      # argument.
      return 7.

    self.assertAllClose(onp.zeros(3,), api.grad(f)(onp.ones(3,)),
                        check_dtypes=True)

  @jtu.disable
  def testIssue777(self):
    # [FUNCTION DOC]
    # Disabled regression test (JAX issue 777): the gradient of the summed logistic function at
    # large negative inputs must be finite and match 0 0 0 0.25 style values in float32. Never
    # runs (disable decorator).
    x = tnp.linspace(-200, 0, 4, dtype=onp.float32)
    f = nje.grad(lambda x: tnp.sum(1 / (1 + tnp.exp(-x))))
    self.assertAllClose(f(x), onp.array([0., 0., 0., 0.25], dtype=onp.float32),
                        check_dtypes=True)

  @named_parameters(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(op, [()], [dtype]),
         "dtype": dtype, "op": op}
      for dtype in float_dtypes
      for op in ("sqrt", "arccos", "arcsin", "arctan", "sin", "cos", "tan",
                 "sinh", "cosh", "tanh", "arccosh", "arcsinh", "arctanh", "exp",
                 "log", "expm1", "log1p")))
  def testMathSpecialFloatValues(self, op, dtype):
    # [FUNCTION DOC]
    # Parameterized test of elementwise math functions (sqrt, trig, hyperbolic, exp, log...) on
    # special float values: NaN, infinities, huge and ordinary values. For each value, asserts
    # TensorFlow NumPy matches NumPy within a dtype-based tolerance with dtype checking. Skips
    # trig functions on TPU and several functions for float16 (known open bugs).
    onp_op = getattr(onp, op)
    lnp_op = getattr(tnp, op)
    dtype = onp.dtype(tnp.canonicalize_dtype(dtype)).type
    for x in (onp.nan, -onp.inf, -100., -2., -1., 0., 1., 2., 100., onp.inf,
              tnp.finfo(dtype).max, onp.sqrt(tnp.finfo(dtype).max),
              onp.sqrt(tnp.finfo(dtype).max) * 2.):
      if (op in ("sin", "cos", "tan", "arctan") and
          jtu.device_under_test() == "tpu"):
        continue  # TODO(b/132196789, b/134175194): fix and reenable.
      # TODO(b/158006398): fix and reenable.
      if (op in ("cosh", "arccosh", "arcsinh", "arcsin", "sinh", "arccos",
                 "arctan", "arctanh") and dtype == onp.float16):
        continue
      x = dtype(x)
      expected = onp_op(x)
      actual = lnp_op(x)
      tol = jtu.tolerance(dtype, {onp.float32: 1e-3, onp.float64: 1e-7})
      self.assertAllClose(expected, actual, check_dtypes=True, atol=tol,
                          rtol=tol)

  def testIssue883(self):
    # [FUNCTION DOC]
    # Regression test (JAX issue 883): calling a jit-compiled function twice with the same array
    # arguments, where the second argument is declared static, must not crash. No assertions.
    # from https://github.com/google/jax/issues/883

    @partial(nje.jit, static_argnums=(1,))
    def f(x, v):
      # [FUNCTION DOC]
      # HELPER: returns the first argument unchanged; the second is only a static argument.
      return x

    x = tnp.ones((10, 10))
    v = tnp.array([1, 2, 3])
    first_call = f(x, v)
    second_call = f(x, v)  # doesn't crash

  def testReductionOfOutOfBoundsAxis(self):  # Issue 888
    # [FUNCTION DOC]
    # Regression test (issue 888): summing a 3x4 array over axis 2 (out of bounds) must raise the
    # TensorFlow invalid-argument error.
    x = tnp.ones((3, 4))
    self.assertRaises(
        errors_impl.InvalidArgumentError, lambda: tnp.sum(x, axis=2)
    )

  @jtu.disable
  def testIssue956(self):
    # [FUNCTION DOC]
    # Disabled regression test (JAX issue 956): directly constructing the array class with a shape
    # should raise a type error. Never runs (disable decorator).
    self.assertRaises(TypeError, lambda: tnp.ndarray((1, 1)))

  @named_parameters(
      jtu.cases_from_list(
        {"testcase_name":
         "_shape={}_dtype={}_out_dtype={}_axis={}_ddof={}_keepdims={}"
         .format(shape, dtype, out_dtype, axis, ddof, keepdims),
         "shape": shape, "dtype": dtype, "out_dtype": out_dtype, "axis": axis,
         "ddof": ddof, "keepdims": keepdims, "rng_factory": rng_factory}
        for shape in [(5,), (10, 5)]
        for dtype in all_dtypes
        for out_dtype in inexact_dtypes
        for axis in [None, 0, -1]
        for ddof in [0, 1, 2]
        for keepdims in [False, True]
        for rng_factory in [jtu.rand_default]))
  def testVar(self, shape, dtype, out_dtype, axis, ddof, keepdims, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of the variance with output dtype, axis, delta degrees of freedom and
    # keepdims over all dtypes. Compares against NumPy (input promoted to at least float32, result
    # cast to the output dtype) with dtype checking and an output-dtype tolerance, then compiles
    # and checks with incomplete-shape checking.
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    def onp_fun(x):
      # [FUNCTION DOC]
      # HELPER: NumPy reference variance computed after promoting the input with float32 and then
      # cast to the output dtype.
      out = onp.var(x.astype(tnp.promote_types(onp.float32, dtype)),
                    axis=axis, ddof=ddof, keepdims=keepdims)
      return out.astype(out_dtype)
    lnp_fun = partial(
        tnp.var, dtype=out_dtype, axis=axis, ddof=ddof, keepdims=keepdims)
    tol = jtu.tolerance(out_dtype, {onp.float16: 1e-1, onp.float32: 1e-3,
                                    onp.float64: 1e-3, onp.complex128: 1e-6})
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True,
                            tol=tol)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True, rtol=tol,
                          atol=tol, check_incomplete_shape=True)

  @named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_shape={}_dtype={}_rowvar={}_ddof={}_bias={}".format(
            shape, dtype, rowvar, ddof, bias),
         "shape": shape, "dtype": dtype, "rowvar": rowvar, "ddof": ddof,
         "bias": bias, "rng_factory": rng_factory}
        for shape in [(5,), (10, 5), (5, 10)]
        for dtype in all_dtypes
        for rowvar in [True, False]
        for bias in [True, False]
        for ddof in [None, 2, 3]
        for rng_factory in [jtu.rand_default]))
  @jtu.skip_on_devices("gpu")  # TODO(b/138003641): test fails on GPU.
  @jtu.disable
  def testCov(self, shape, dtype, rowvar, ddof, bias, rng_factory):
    # [FUNCTION DOC]
    # Disabled parameterized test (also skipped on GPU) of the covariance matrix with rowvar,
    # degrees of freedom and bias options. Compares against NumPy without dtype checking using a
    # looser tolerance on TPU, then compiles and checks. Never runs (disable decorator).
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    onp_fun = partial(onp.cov, rowvar=rowvar, ddof=ddof, bias=bias)
    lnp_fun = partial(tnp.cov, rowvar=rowvar, ddof=ddof, bias=bias)
    tol = {onp.float32: 1e-5, onp.float64: 1e-13, onp.complex128: 1e-13}
    tol = 7e-2 if jtu.device_under_test() == "tpu" else tol
    tol = jtu.join_tolerance(tol, jtu.tolerance(dtype))
    self._CheckAgainstNumpy(
        onp_fun, lnp_fun, args_maker, check_dtypes=False, tol=tol)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True, atol=tol,
                          rtol=tol)

  def testIssue967(self):
    # [FUNCTION DOC]
    # Regression test (JAX issue 967): creating zeros with a non-integer shape (1.5) must raise a
    # type error.
    self.assertRaises(TypeError, lambda: tnp.zeros(1.5))

  @named_parameters(
      jtu.cases_from_list(
        {"testcase_name": "_shape={}_dtype={}_rowvar={}_ddof={}_bias={}".format(
            shape, dtype, rowvar, ddof, bias),
         "shape": shape, "dtype": dtype, "rowvar": rowvar, "ddof": ddof,
         "bias": bias, "rng_factory": rng_factory}
        for shape in [(5,), (10, 5), (3, 10)]
        for dtype in number_dtypes
        for rowvar in [True, False]
        for bias in [True, False]
        for ddof in [None, 2, 3]
        for rng_factory in [jtu.rand_default]))
  @jtu.disable
  def testCorrCoef(self, shape, dtype, rowvar, ddof, bias, rng_factory):
    # [FUNCTION DOC]
    # Disabled parameterized test of the correlation coefficient matrix. The NumPy comparison only
    # runs when the data's standard deviation is not close to zero; the compile-and-check always
    # runs. Never runs (disable decorator).
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, [shape], [dtype])
    mat = onp.asarray([rng(shape, dtype)])
    onp_fun = partial(onp.corrcoef, rowvar=rowvar, ddof=ddof, bias=bias)
    lnp_fun = partial(tnp.corrcoef, rowvar=rowvar, ddof=ddof, bias=bias)
    if not onp.any(onp.isclose(onp.std(mat), 0.0)):
      self._CheckAgainstNumpy(
          onp_fun, lnp_fun, args_maker, check_dtypes=False,
          tol=1e-2 if jtu.device_under_test() == "tpu" else None)
    self._CompileAndCheck(lnp_fun, args_maker, check_dtypes=True)

  @named_parameters(
      jtu.cases_from_list(
          {
              "testcase_name":
                  "_shapes={}_dtype={}_indexing={}_sparse={}".format(
                      shapes, jtu.dtype_str(dtype), indexing, sparse),
              "shapes":
                  shapes,
              "dtype":
                  dtype,
              "indexing":
                  indexing,
              "sparse":
                  sparse,
              "rng_factory":
                  rng_factory
          } for shapes in [(), (5,), (5, 3)] for dtype in number_dtypes
          for indexing in ["xy", "ij"]
          for sparse in [False]  # TODO(nareshmodi): Make sparse work
          for rng_factory in [jtu.rand_default]))
  def testMeshGrid(self, shapes, dtype, indexing, sparse, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of coordinate matrices from coordinate vectors, with Cartesian or matrix
    # indexing. The sparse option is always false (sparse output not supported yet). Compares
    # against NumPy and compiles and checks.
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, [(x,) for x in shapes],
                                    [dtype] * len(shapes))
    onp_fun = partial(onp.meshgrid, indexing=indexing, sparse=sparse)
    lnp_fun = partial(tnp.meshgrid, indexing=indexing, sparse=sparse)
    self._CheckAgainstNumpy(onp_fun, lnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_fun, args_maker, check_dtypes=True, check_incomplete_shape=True)

  @named_parameters(
      jtu.cases_from_list(
          {
              "testcase_name": (
                  "_start_shape={}_stop_shape={}_num={}_endpoint={}"
                  "_retstep={}_dtype={}"
              ).format(start_shape, stop_shape, num, endpoint, retstep, dtype),
              "start_shape": start_shape,
              "stop_shape": stop_shape,
              "num": num,
              "endpoint": endpoint,
              "retstep": retstep,
              "dtype": dtype,
              "rng_factory": rng_factory,
          }
          for start_shape in [(), (2,), (2, 2)]
          for stop_shape in [(), (2,), (2, 2)]
          for num in [0, 1, 2, 5, 20]
          for endpoint in [True, False]
          for retstep in [True, False]
          for dtype in (
              (
                  float_dtypes
                  + complex_dtypes
                  + [
                      None,
                  ]
              )
              if onp.__version__ >= onp.lib.NumpyVersion("2.0.0")
              else (number_dtypes + [None])
          )
          for rng_factory in [jtu.rand_default]
      )
  )
  def testLinspace(self, start_shape, stop_shape, num, endpoint,
                   retstep, dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of evenly spaced numbers over an interval with broadcastable start and
    # stop shapes, count, endpoint and step-return options. Returns early for integer dtypes
    # without endpoint (rounding problems). For every axis, compares against NumPy without dtype
    # checking at a tolerance ten times the dtype default, and compiles and checks for inexact
    # or unspecified dtypes.
    if not endpoint and onp.issubdtype(dtype, onp.integer):
      # TODO(b/157597565): Support all dtypes when the tf op supports endpoint
      # Currently, subtracting the step early leads to rounding errors for
      # integers.
      return
    rng = rng_factory()
    # relax default tolerances slightly
    tol = jtu.tolerance(dtype if dtype else onp.float32) * 10
    args_maker = self._GetArgsMaker(rng,
                                    [start_shape, stop_shape],
                                    [dtype, dtype])
    start, stop = args_maker()
    ndim = len(onp.shape(start + stop))
    for axis in range(-ndim, ndim):
      lnp_op = lambda start, stop: tnp.linspace(
        start, stop, num,
        endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis)
      onp_op = lambda start, stop: onp.linspace(
        start, stop, num,
        endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis)
      self._CheckAgainstNumpy(onp_op, lnp_op, args_maker,
                              check_dtypes=False, tol=tol)
      # floating-point compute between jitted platforms and non-jit + rounding
      # cause unavoidable variation in integer truncation for some inputs.
      if dtype in (inexact_dtypes + [None,]):
        self._CompileAndCheck(lnp_op, args_maker,
                              check_dtypes=False, atol=tol, rtol=tol,
                              check_incomplete_shape=True)

  @named_parameters(
      jtu.cases_from_list(
        {"testcase_name": ("_start_shape={}_stop_shape={}_num={}_endpoint={}"
                           "_base={}_dtype={}").format(
            start_shape, stop_shape, num, endpoint, base,
            dtype.__name__ if dtype else "None"),
         "start_shape": start_shape,
         "stop_shape": stop_shape,
         "num": num, "endpoint": endpoint, "base": base,
         "dtype": dtype, "rng_factory": rng_factory}
        for start_shape in [(), (2,), (2, 2)]
        for stop_shape in [(), (2,), (2, 2)]
        for num in [0, 1, 2, 5, 20]
        for endpoint in [True, False]
        for base in [10.0, 2, onp.e]
        for dtype in inexact_dtypes + [None,]
        for rng_factory in [jtu.rand_default]))
  def testLogspace(self, start_shape, stop_shape, num,
                   endpoint, base, dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of numbers spaced evenly on a log scale with broadcastable start and stop
    # shapes, count, endpoint, base (10, 2 or e) and optional dtype. Skips integer dtypes on
    # GPU/TPU without 64-bit mode (truncation mismatch). For every axis, compares against NumPy
    # without dtype checking, and compiles and checks for inexact or unspecified dtypes with a
    # looser absolute tolerance for float16.
    if (dtype in int_dtypes and
        jtu.device_under_test() in ("gpu", "tpu") and
        not FLAGS.enable_x64):
      raise unittest.SkipTest("GPUx32 truncated exponentiation"
                              " doesn't exactly match other platforms.")
    rng = rng_factory()
    # relax default tolerances slightly
    tol = {onp.float16: 2e-2, onp.float32: 1e-2, onp.float64: 1e-6,
           onp.complex64: 1e-3, onp.complex128: 1e-6}
    args_maker = self._GetArgsMaker(rng,
                                    [start_shape, stop_shape],
                                    [dtype, dtype])
    start, stop = args_maker()
    ndim = len(onp.shape(start + stop))
    for axis in range(-ndim, ndim):
      lnp_op = lambda start, stop: tnp.logspace(
        start, stop, num, endpoint=endpoint, base=base, dtype=dtype, axis=axis)
      onp_op = lambda start, stop: onp.logspace(
        start, stop, num, endpoint=endpoint, base=base, dtype=dtype, axis=axis)
      self._CheckAgainstNumpy(onp_op, lnp_op, args_maker,
                              check_dtypes=False, tol=tol)
      if dtype in (inexact_dtypes + [None,]):
        # Why do compiled and op-by-op float16 np.power numbers differ
        # slightly more than expected?
        atol = {onp.float16: 1e-2}
        self._CompileAndCheck(lnp_op, args_maker,
                              check_dtypes=False, atol=atol, rtol=tol,
                              check_incomplete_shape=True)


  @named_parameters(
      jtu.cases_from_list(
          {
              "testcase_name": (
                  "_start_shape={}_stop_shape={}_num={}_endpoint={}_dtype={}"
              ).format(start_shape, stop_shape, num, endpoint, dtype),
              "start_shape": start_shape,
              "stop_shape": stop_shape,
              "num": num,
              "endpoint": endpoint,
              "dtype": dtype,
              "rng_factory": rng_factory,
          }
          for start_shape in [(), (2,), (2, 2)]
          for stop_shape in [(), (2,), (2, 2)]
          for num in [0, 1, 2, 5, 20]
          for endpoint in [True, False]
          # NB: numpy's geomspace gives nonsense results on integer types
          for dtype in (
              (
                  float_dtypes
                  + [
                      None,
                  ]
              )
              if onp.__version__ >= onp.lib.NumpyVersion("2.0.0")
              else (
                  inexact_dtypes
                  + [
                      None,
                  ]
              )
          )
          for rng_factory in [jtu.rand_default]
      )
  )
  def testGeomspace(self, start_shape, stop_shape, num,
                    endpoint, dtype, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of numbers spaced evenly on a geometric scale, over broadcastable start and
    # stop shapes, count, endpoint and floating, complex or unspecified dtype. For every axis,
    # compares against NumPy (bfloat16 handled via float32) without dtype checking, and compiles
    # and checks for inexact or unspecified dtypes.
    rng = rng_factory()
    # relax default tolerances slightly
    tol = {onp.float16: 4e-3, onp.float32: 2e-3, onp.complex128: 1e-14}
    def args_maker():
      # [FUNCTION DOC]
      # HELPER: generates a (start, stop) pair on which NumPy's geomspace is well defined. Both
      # are broadcast to a common shape; for non-complex dtypes the start is multiplied by the
      # signs of start and stop so they never differ in sign (avoids NaN results).
      """Test the set of inputs onp.geomspace is well-defined on."""
      start, stop = self._GetArgsMaker(rng,
                                [start_shape, stop_shape],
                                [dtype, dtype])()
      # onp.geomspace can't handle differently ranked tensors
      # w. negative numbers!
      start, stop = tnp.broadcast_arrays(start, stop)
      if dtype in complex_dtypes:
        return start, stop
      # to avoid NaNs, non-complex start and stop cannot
      # differ in sign, elementwise
      start = start * tnp.sign(start) * tnp.sign(stop)
      return start, stop
    start, stop = args_maker()
    ndim = len(onp.shape(start + stop))
    for axis in range(-ndim, ndim):
      def lnp_op(start, stop):
        # [FUNCTION DOC]
        # HELPER: TensorFlow NumPy geomspace with the test's count, endpoint, dtype and axis.
        return tnp.geomspace(start, stop, num, endpoint=endpoint, dtype=dtype,
                             axis=axis)
      def onp_op(start, stop):
        # [FUNCTION DOC]
        # HELPER: NumPy reference geomspace; computes in float32 for bfloat16 and casts the
        # result back to the test dtype.
        start = start.astype(onp.float32) if dtype == tnp.bfloat16 else start
        stop = stop.astype(onp.float32) if dtype == tnp.bfloat16 else stop
        return onp.geomspace(
            start, stop, num, endpoint=endpoint,
            dtype=dtype if dtype != tnp.bfloat16 else onp.float32,
            axis=axis).astype(dtype)
      self._CheckAgainstNumpy(onp_op, lnp_op, args_maker,
                              check_dtypes=False, tol=tol)
      if dtype in (inexact_dtypes + [None,]):
        self._CompileAndCheck(lnp_op, args_maker,
                              check_dtypes=False, atol=tol, rtol=tol,
                              check_incomplete_shape=True)

  @jtu.disable
  def testDisableNumpyRankPromotionBroadcasting(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled; relies on a rank-promotion flag (a JAX flag) that may not exist in
    # this suite's flags, and its name says "Disable" though it tests allow/raise/warn modes.
    # Intended scenario: with the rank promotion flag set to allow, adding arrays of different
    # rank works; set to raise, it raises a value error; set to warn, it emits a warning for
    # array-array rank promotion but not for scalars. The flag is restored in every case.
    try:
      prev_flag = FLAGS.jax_numpy_rank_promotion
      FLAGS.jax_numpy_rank_promotion = "allow"
      tnp.ones(2) + tnp.ones((1, 2))  # works just fine
    finally:
      FLAGS.jax_numpy_rank_promotion = prev_flag

    try:
      prev_flag = FLAGS.jax_numpy_rank_promotion
      FLAGS.jax_numpy_rank_promotion = "raise"
      self.assertRaises(ValueError, lambda: tnp.ones(2) + tnp.ones((1, 2)))
    finally:
      FLAGS.jax_numpy_rank_promotion = prev_flag

    try:
      prev_flag = FLAGS.jax_numpy_rank_promotion
      FLAGS.jax_numpy_rank_promotion = "warn"
      with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tnp.ones(2) + tnp.ones((1, 2))
        assert len(w) > 0
        msg = str(w[-1].message)
        expected_msg = ("Following NumPy automatic rank promotion for add on "
                        "shapes (2,) (1, 2).")
        self.assertEqual(msg[:len(expected_msg)], expected_msg)

        prev_len = len(w)
        tnp.ones(2) + 3
        self.assertEqual(len(w), prev_len)  # don't want to warn for scalars
    finally:
      FLAGS.jax_numpy_rank_promotion = prev_flag

  def testStackArrayArgument(self):
    # [FUNCTION DOC]
    # Regression test (JAX issue 1271): stack and concatenate called inside a jit-compiled
    # function with a single array argument (rather than a list) must not crash. The same local
    # name is defined twice, once for each operation; there are no assertions.
    # tests https://github.com/google/jax/issues/1271
    @nje.jit
    def foo(x):
      # [FUNCTION DOC]
      # HELPER: stacks the argument (treated as a sequence of arrays) and returns the result.
      return tnp.stack(x)
    foo(onp.zeros(2))  # doesn't crash

    @nje.jit
    def foo(x):
      # [FUNCTION DOC]
      # HELPER: concatenates the argument (treated as a sequence of arrays) and returns the result.
      return tnp.concatenate(x)
    foo(onp.zeros((2, 2)))  # doesn't crash

  @jtu.disable
  def testReluGradientConstants(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled dead code; it refers to JAX-only names (jax, lax, linear-util and
    # partial-eval modules) that are not imported here, so it cannot run.
    # Intended scenario: constants from the gradient of maximum must not be hoisted into the
    # outermost traced program, checked by asserting no captured constant equals a 3x4 array of 2.
    # This is a regression test that verifies that constants associated with the
    # gradient of np.maximum (from lax._balanced_eq) aren't hoisted into the
    # outermost jaxpr. This was producing some large materialized constants for
    # every relu activation in a model.
    def body(i, xy):
      # [FUNCTION DOC]
      # HELPER: loop body for the disabled relu test; adds the gradient of the sum of maximum(z, 0)
      # at x to the second element of the pair and returns the updated pair.
      # KNOWN ISSUE: uses an undefined JAX module name (silenced by a lint comment).
      x, y = xy
      y = y + jax.grad(lambda z: tnp.sum(tnp.maximum(z, 0.)))(x)  # pylint: disable=undefined-variable
      return x, y

    f = lambda y: lax.fori_loop(0, 5, body, (y, y))
    wrapped = linear_util.wrap_init(f)
    pv = partial_eval.PartialVal(
      (jax.core.ShapedArray((3, 4), onp.float32), jax.core.unit))
    _, _, consts = partial_eval.trace_to_jaxpr(wrapped, [pv])
    self.assertFalse(
      any(onp.array_equal(x, onp.full((3, 4), 2., dtype=onp.float32))
          for x in consts))

  @named_parameters(
      {"testcase_name": "_from={}_to={}".format(from_shape, to_shape),
       "rng_factory": rng_factory, "from_shape": from_shape, "to_shape": to_shape}
      for from_shape, to_shape in [
          [(1, 3), (4, 3)],
          [(3,), (2, 1, 3)],
          [(3,), (3, 3)],
          [(1,), (3,)],
      ]
      for rng_factory in [jtu.rand_default])
  def testBroadcastTo(self, from_shape, to_shape, rng_factory):
    # [FUNCTION DOC]
    # Parameterized test of broadcasting a float32 array to a larger shape. Compares against NumPy
    # and compiles and checks with dtype and incomplete-shape checking.
    rng = rng_factory()
    args_maker = self._GetArgsMaker(rng, [from_shape], [onp.float32])
    onp_op = lambda x: onp.broadcast_to(x, to_shape)
    lnp_op = lambda x: tnp.broadcast_to(x, to_shape)
    self._CheckAgainstNumpy(onp_op, lnp_op, args_maker, check_dtypes=True)
    self._CompileAndCheck(
        lnp_op, args_maker, check_dtypes=True, check_incomplete_shape=True)

  def testBroadcastToIssue1522(self):
    # [FUNCTION DOC]
    # Regression test (issue 1522): broadcasting a 2x3 array to shape 1x3 is invalid and must
    # raise an exception whose message contains "Unable to broadcast".
    self.assertRaisesRegex(
        Exception, "Unable to broadcast",
        lambda: tnp.broadcast_to(onp.ones((2, 3)), (1, 3)))

  def testBroadcastToIntIssue1548(self):
    # [FUNCTION DOC]
    # Regression test (issue 1548): broadcasting the Python integer 1 to shape 3x2 gives an array
    # of ones; asserts closeness without dtype checking.
    self.assertAllClose(tnp.broadcast_to(1, (3, 2)), onp.ones((3, 2)),
                        check_dtypes=False)

  def testBroadcastToOnScalar(self):
    # [FUNCTION DOC]
    # Test that broadcasting a scalar to the empty shape returns an array object: a TensorFlow
    # NumPy array for the tnp function and a NumPy array for NumPy's function.
    self.assertIsInstance(tnp.broadcast_to(10.0, ()), tnp.ndarray)
    self.assertIsInstance(onp.broadcast_to(10.0, ()), onp.ndarray)

  @jtu.disable
  def testPrecision(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled dead code; uses a JAX-only precision enumeration (lax) that is not
    # imported here and a precision keyword that TensorFlow NumPy does not take.
    # Intended scenario: dot, matmul, vdot, tensordot, einsum and inner must pass the requested
    # highest-precision setting down to the underlying dot operation, checked through a test
    # utility.

    ones_1d = onp.ones((2,))
    ones_2d = onp.ones((2, 2))
    ones_3d = onp.ones((2, 2, 2))
    HIGHEST = lax.Precision.HIGHEST

    jtu.assert_dot_precision(None, tnp.dot, ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(tnp.dot, precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(tnp.dot, precision=HIGHEST),
        ones_3d, ones_3d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(tnp.matmul, precision=HIGHEST),
        ones_2d, ones_2d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(tnp.vdot, precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(tnp.tensordot, axes=2, precision=HIGHEST),
        ones_2d, ones_2d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(tnp.tensordot, axes=(0, 0), precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(tnp.tensordot, axes=((0,), (0,)), precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(tnp.einsum, "i,i", precision=HIGHEST),
        ones_1d, ones_1d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(tnp.einsum, "ij,ij", precision=HIGHEST),
        ones_2d, ones_2d)
    jtu.assert_dot_precision(
        HIGHEST,
        partial(tnp.inner, precision=HIGHEST),
        ones_1d, ones_1d)

  @named_parameters(jtu.cases_from_list(
      {"testcase_name":
       "_{}_{}_{}_{}".format(
           shape, jtu.dtype_str(key_dtype), jtu.dtype_str(value_dtype),
           dimension).replace(" ", ""),
       "shape": shape, "key_dtype": key_dtype, "value_dtype": value_dtype,
       "dimension": dimension, "rng_factory": rng_factory}
      for shape in all_shapes
      for key_dtype in minus(number_dtypes, complex_dtypes)
      for value_dtype in all_dtypes
      for dimension in range(-len(shape), len(shape))
      for rng_factory in [jtu.rand_default]))
  @new_test
  def testSortKeyValue(self, shape, key_dtype, value_dtype, dimension,
                       rng_factory):
    # [FUNCTION DOC]
    # Parameterized test (only when the additional-tests flag is on) of sorting keys together with
    # values along a dimension. Skipped for float32 keys with bool values (nightly build issue).
    # Compares against a NumPy reference built with argsort and fancy indexing, with dtype
    # checking, then compiles and checks (unknown-rank off; XLA only for dtypes with sort kernels).
    if key_dtype == onp.float32 and value_dtype == onp.bool_:
      self.skipTest(
          "Temporarily disable this test because of TF nightly build failure"
      )
    def onp_ref(keys, values):
      # [FUNCTION DOC]
      # HELPER: NumPy reference; sorts the keys along the dimension with argsort, and applies the
      # same permutation (built from an open mesh of aranges) to both keys and values. Returns the
      # pair of reordered keys and values.

      idxs = list(onp.ix_(*[onp.arange(d) for d in keys.shape]))
      idxs[dimension] = onp.argsort(keys, axis=dimension)
      return keys[tuple(idxs)], values[tuple(idxs)]
    rng = rng_factory()
    args_maker = self._GetArgsMaker(
        rng, [shape, shape], [key_dtype, value_dtype])
    op = partial(nje.sort_key_val, dimension=dimension)
    self._CheckAgainstNumpy(onp_ref, op, args_maker,
                            check_dtypes=True)
    # sort_key_val requires known rank.
    # XLA only has TopKV2 (used by tf.argsort) kernels on those dtypes
    # (b/169194137).
    check_xla = key_dtype in (onp.uint32, onp.int32, onp.float32, tnp.bfloat16)
    self._CompileAndCheck(op, args_maker, check_dtypes=True,
                          check_incomplete_shape=True, check_unknown_rank=False,
                          check_experimental_compile=check_xla,
                          check_xla_forced_compile=check_xla)


# Most grad tests are at the lax level (see lax_test.py), but we add some here
# as needed for e.g. particular compound ops of interest.

GradTestSpec = collections.namedtuple(
    "GradTestSpec",
    ["op", "nargs", "order", "rng_factory", "dtypes", "name", "tol"])
def grad_test_spec(op, nargs, order, rng_factory, dtypes, name=None, tol=None):
  # [FUNCTION DOC]
  # HELPER: constructor for gradient test records. Builds a named tuple holding the operation,
  # number of arguments, derivative order, random-generator factory, dtypes, a name (defaults to
  # the operation's own name) and a tolerance. Returns the record; no side effects.
  return GradTestSpec(
      op, nargs, order, rng_factory, dtypes, name or op.__name__, tol)

GRAD_TEST_RECORDS = [
    grad_test_spec(tnp.arcsinh, nargs=1, order=2,
                   rng_factory=jtu.rand_positive,
                   dtypes=[onp.float64, onp.complex64], tol=1e-4),
    grad_test_spec(tnp.arccosh, nargs=1, order=2,
                   rng_factory=jtu.rand_positive,
                   dtypes=[onp.float64, onp.complex64], tol=1e-4),
    grad_test_spec(tnp.arctanh, nargs=1, order=2,
                   rng_factory=partial(jtu.rand_uniform, -0.9, 0.9),
                   dtypes=[onp.float64, onp.complex64], tol=1e-4),
]

GradSpecialValuesTestSpec = collections.namedtuple(
    "GradSpecialValuesTestSpec", ["op", "values", "order"])

GRAD_SPECIAL_VALUE_TEST_RECORDS = [
    GradSpecialValuesTestSpec(tnp.arcsinh, [0., 1000.], 2),
    GradSpecialValuesTestSpec(tnp.arccosh, [1000.], 2),
    GradSpecialValuesTestSpec(tnp.arctanh, [0.], 2),
    # TODO(wangpeng): Add `GradSpecialValuesTestSpec(tnp.sinc, [0.], 1)`
]

# def num_float_bits(dtype):
#   return tnp.finfo(dtypes.canonicalize_dtype(dtype)).bits

class NumpyGradTests(jtu.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: every test in this class is disabled, and they call a gradient-checking function
  # that is not imported in this module, so the class currently verifies nothing.
  # Test case class meant to check gradients of selected compound operations (inverse hyperbolic
  # functions) generated from the gradient record tables. Holds no state.
  @named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
        {"testcase_name": jtu.format_test_name_suffix(
            rec.name, shapes, itertools.repeat(dtype)),
         "op": rec.op, "rng_factory": rec.rng_factory, "shapes": shapes, "dtype": dtype,
         "order": rec.order, "tol": rec.tol}
        for shapes in CombosWithReplacement(nonempty_shapes, rec.nargs)
        for dtype in rec.dtypes)
      for rec in GRAD_TEST_RECORDS))
  @jtu.disable
  def testOpGrad(self, op, rng_factory, shapes, dtype, order, tol):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the tol argument is overwritten immediately by a fixed dictionary, and the
    # gradient checker name is not imported; disabled so it never runs.
    # Intended scenario: random arguments of the given shapes and dtype; check forward and reverse
    # mode gradients of the operation up to the given order.
    rng = rng_factory()
    tol = {onp.float32: 1e-1, onp.complex64: 1e-1}
    args = tuple(rng(shape, dtype) for shape in shapes)
    check_grads(op, args, order, ["fwd", "rev"], tol, tol)

  @named_parameters(itertools.chain.from_iterable(
      jtu.cases_from_list(
          {"testcase_name": "_{}_{}".format(rec.op.__name__, special_value),
           "op": rec.op, "special_value": special_value, "order": rec.order}
          for special_value in rec.values)
      for rec in GRAD_SPECIAL_VALUE_TEST_RECORDS))
  @jtu.disable
  def testOpGradSpecialValue(self, op, special_value, order):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled, and the gradient checker name is not imported in this module.
    # Intended scenario: check forward and reverse gradients of an operation at a special input
    # value (for example zero or 1000) up to the given order with a float32 absolute tolerance.
    check_grads(op, (special_value,), order, ["fwd", "rev"],
                atol={onp.float32: 3e-3})

  @jtu.disable
  def testTakeAlongAxisIssue1521(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled, and the gradient checker name is not imported in this module.
    # Intended scenario (JAX issue 1521): the gradient of take-along-axis with repeated indices
    # must be correct to first order.
    # https://github.com/google/jax/issues/1521
    idx = tnp.repeat(tnp.arange(3), 10).reshape((30, 1))

    def f(x):
      # [FUNCTION DOC]
      # HELPER: scales a 1x3 arange by x, takes values along the last axis using the captured
      # index array and returns their sum.
      y = x * tnp.arange(3.).reshape((1, 3))
      return tnp.take_along_axis(y, idx, -1).sum()

    check_grads(f, (1.,), order=1)


if __name__ == "__main__":
  absltest.main()
  