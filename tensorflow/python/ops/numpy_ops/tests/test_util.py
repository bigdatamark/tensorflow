# [MODULE DOC]
# HELPER: Shared test-support library for the TF NumPy test suite (not a test itself).
# KNOWN ISSUE: several helpers refer to names that are never imported here (tree_all,
# tree_reduce, core, api, lax), so those helpers fail with a name error when called.
# Offers dtype/tolerance tables, numeric comparison and finite-difference gradient checkers,
# random-input generators (default, nonzero, positive, small, with infinities, NaNs or zeros,
# integers, booleans), shape/dtype naming helpers, skip decorators keyed on device or flags,
# sampling of parameterized cases, and a TestCase base class with allclose, compile-and-check
# and gradient-check assertions.
# Import-time side effects: defines the command-line flags enable_x64, test_dut and
# num_generated_cases on the shared flags object, and creates module-level tables and caches.
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
"""NumPy test utilities."""
from contextlib import contextmanager
import functools
from functools import partial
import re
import itertools as it
import os
from typing import Dict, Sequence, Union
import unittest
import warnings
import zlib

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp
import numpy.random as npr

from tensorflow.python.util import nest
from tensorflow.python.util import numpy_compat
from tensorflow.python.framework import tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker_v2

from tensorflow.python.ops.numpy_ops.tests.config import flags
import tensorflow.python.ops.numpy_ops.tests.extensions as nje
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.ops.numpy_ops import np_array_ops


tree_map = nest.map_structure
tree_multimap = nest.map_structure


FLAGS = flags.FLAGS


# https://danielms.site/zet/2023/pythons-distutil-strtobool-replacement/
def strtobool(value: str) -> bool:
  # [FUNCTION DOC]
  # HELPER: Replacement for the removed standard-library string-to-bool converter, used to
  # compute the default of the enable_x64 flag.
  # KNOWN ISSUE: unlike the function it replaces, it never raises for invalid text; every
  # unrecognized string silently yields false.
  # Lower-cases the input string and returns true when it is one of yes, y, on, 1, true or t,
  # otherwise false. No side effects.
  value = value.lower()
  if value in ('y', 'yes', 'on', '1', 'true', 't'):
    return True
  return False


# TODO(wangpeng): Remove this flag after broken tests are fixed
flags.DEFINE_bool('enable_x64',
                  strtobool('False'),
                  'Enable 64-bit types to be used.')


flags.DEFINE_enum(
    'test_dut', '',
    enum_values=['', 'cpu', 'gpu', 'tpu'],
    help=
    'Describes the device under test in case special consideration is required.'
)


flags.DEFINE_integer(
  'num_generated_cases',
  10,
  help='Number of generated cases to test')


EPS = 1e-4


# Default dtypes corresponding to Python scalars.
python_scalar_dtypes = {
  bool: onp.dtype(onp.bool_),
  int: onp.dtype(onp.int_),
  float: onp.dtype(onp.float64),
  complex: onp.dtype(onp.complex128),
}


def _dtype(x):
  # [FUNCTION DOC]
  # HELPER: Determines the numpy dtype of a value for the comparison helpers in this module.
  # KNOWN ISSUE: the middle fallback wraps a possibly missing lookup result in a dtype
  # constructor; a nothing value there yields the default float64 dtype instead of failing, so
  # the final fallback (converting to an array) is effectively unreachable for values without
  # a dtype attribute and not in the scalar table.
  # Input x: a tensor, an array-like object with a dtype attribute, or a Python scalar. For a
  # tensor returns its dtype converted to numpy; otherwise returns the object's dtype
  # attribute if present and truthy, else the table dtype for its Python type, else the dtype
  # of x converted to an array. Returns a numpy dtype; no side effects.
  if isinstance(x, tensor.Tensor):
    return x.dtype.as_numpy_dtype
  return (getattr(x, 'dtype', None) or
          onp.dtype(python_scalar_dtypes.get(type(x), None)) or
          numpy_compat.np_asarray(x).dtype)


def is_sequence(x):
  # [FUNCTION DOC]
  # HELPER: Predicate used by assertAllClose to decide whether to recurse into an object.
  # Returns true if x can be iterated over (so strings and dicts also count), false if asking
  # for an iterator raises a type error. No side effects.
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True

_default_tolerance = {
  onp.dtype(onp.bool_): 0,
  onp.dtype(onp.int8): 0,
  onp.dtype(onp.int16): 0,
  onp.dtype(onp.int32): 0,
  onp.dtype(onp.int64): 0,
  onp.dtype(onp.uint8): 0,
  onp.dtype(onp.uint16): 0,
  onp.dtype(onp.uint32): 0,
  onp.dtype(onp.uint64): 0,
  # TODO(b/154768983): onp.dtype(dtypes.bfloat16): 1e-2,
  onp.dtype(onp.float16): 1e-3,
  onp.dtype(onp.float32): 1e-6,
  onp.dtype(onp.float64): 1e-15,
  onp.dtype(onp.complex64): 1e-6,
  onp.dtype(onp.complex128): 1e-15,
}

def default_tolerance():
  # [FUNCTION DOC]
  # HELPER: Accessor for the module-wide default tolerance table.
  # Takes no inputs and returns the shared mapping from numpy dtype to default tolerance (zero
  # for integer and bool types, small values for floating and complex types). The mapping is
  # returned itself, not a copy, so callers must not mutate it. No side effects.
  return _default_tolerance

default_gradient_tolerance = {
  # TODO(b/154768983): onp.dtype(dtypes.bfloat16): 1e-1,
  onp.dtype(onp.float16): 1e-2,
  onp.dtype(onp.float32): 2e-3,
  onp.dtype(onp.float64): 1e-5,
  onp.dtype(onp.complex64): 1e-3,
  onp.dtype(onp.complex128): 1e-5,
}

def _assert_numpy_allclose(a, b, atol=None, rtol=None):
  # [FUNCTION DOC]
  # HELPER: Thin wrapper over numpy's allclose assertion used by the checkers and TestCase.
  # Inputs: a and b are arrays or array-likes to compare; atol and rtol are optional absolute
  # and relative tolerances, passed on only when they are truthy (so zero means numpy's own
  # default tolerance, not exact equality). Raises an assertion error if values differ.
  # Returns nothing.
  # TODO(b/154768983):
  #   a = a.astype(onp.float32) if a.dtype == dtypes.bfloat16 else a
  #   b = b.astype(onp.float32) if b.dtype == dtypes.bfloat16 else b
  kw = {}
  if atol: kw["atol"] = atol
  if rtol: kw["rtol"] = rtol
  onp.testing.assert_allclose(a, b, **kw)

def tolerance(dtype, tol=None):
  # [FUNCTION DOC]
  # HELPER: Resolves the tolerance to use for a given dtype in the comparison helpers.
  # Inputs: dtype is anything convertible to a numpy dtype; tol is nothing, a plain number
  # (returned unchanged for every dtype), or a mapping from dtypes to numbers. For a mapping
  # the entry for the dtype is returned, falling back to the default tolerance table, which
  # raises a key error for dtypes absent from it (for example bfloat16). No side effects.
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {onp.dtype(key): value for key, value in tol.items()}
  dtype = onp.dtype(dtype)
  return tol.get(dtype, default_tolerance()[dtype])

def _normalize_tolerance(tol):
  # [FUNCTION DOC]
  # HELPER: Normalizes a tolerance specification for join_tolerance.
  # Input tol: nothing, a number, or a mapping from dtypes to numbers. Nothing becomes zero.
  # A mapping is returned as a new mapping keyed by numpy dtypes; a number is expanded into a
  # mapping giving that number for every dtype in the default tolerance table. No side effects.
  tol = tol or 0
  if isinstance(tol, dict):
    return {onp.dtype(k): v for k, v in tol.items()}
  else:
    return {k: tol for k in _default_tolerance}

def join_tolerance(tol1, tol2):
  # [FUNCTION DOC]
  # Combines two tolerance specifications (numbers, mappings or nothing) into one mapping from
  # numpy dtype to tolerance, taking the larger value for each dtype present in either. Both
  # inputs are normalized first; the result is the normalized first mapping updated in place
  # (a fresh mapping, so the caller's inputs are not mutated). Returns that mapping.
  tol1 = _normalize_tolerance(tol1)
  tol2 = _normalize_tolerance(tol2)
  out = tol1
  for k, v in tol2.items():
    out[k] = max(v, tol1.get(k, 0))
  return out

def _assert_numpy_close(a, b, atol=None, rtol=None):
  # [FUNCTION DOC]
  # HELPER: Per-leaf comparison used by check_close.
  # Inputs: numpy arrays a and b, and optional tolerance specifications atol and rtol (see
  # tolerance). Asserts equal shapes, takes the larger per-dtype tolerance of the two arrays,
  # scales the absolute tolerance by the size of a and the relative tolerance by the size of b
  # (making the check looser for big arrays), then asserts closeness. Raises an assertion
  # error on mismatch; returns nothing.
  assert a.shape == b.shape
  atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
  rtol = max(tolerance(a.dtype, rtol), tolerance(b.dtype, rtol))
  _assert_numpy_allclose(a, b, atol=atol * a.size, rtol=rtol * b.size)


def check_eq(xs, ys):
  # [FUNCTION DOC]
  # HELPER: Structure-wide equality-style assertion for tests.
  # KNOWN ISSUE: tree_all is not defined or imported in this module, so calling this raises a
  # name error. It also uses the allclose helper with default tolerances, not exact equality.
  # Intended behavior: maps the allclose assertion over matching leaves of the nested
  # structures xs and ys and then reduces the result. Returns nothing.
  tree_all(tree_multimap(_assert_numpy_allclose, xs, ys))


def check_close(xs, ys, atol=None, rtol=None):
  # [FUNCTION DOC]
  # HELPER: Checks that two nested structures of arrays are numerically close.
  # KNOWN ISSUE: tree_all is not defined or imported in this module, so calling this raises a
  # name error (even though check_jvp and check_vjp depend on it).
  # Inputs: xs and ys are nested structures with matching layout; atol and rtol are optional
  # tolerance specifications. Applies the per-leaf closeness assertion over all leaves.
  # Raises an assertion error on mismatch; returns nothing.
  assert_close = partial(_assert_numpy_close, atol=atol, rtol=rtol)
  tree_all(tree_multimap(assert_close, xs, ys))


def inner_prod(xs, ys):
  # [FUNCTION DOC]
  # HELPER: Real inner product of two nested structures, used by check_vjp.
  # KNOWN ISSUE: tree_reduce is not defined or imported in this module, so calling this raises
  # a name error.
  # Inputs: xs and ys, nested structures of arrays with the same layout. Contracts each pair of
  # leaves (flattened conjugate of the first with the second, real part) and sums the results
  # across leaves. Returns a real scalar.
  def contract(x, y):
    # [FUNCTION DOC]
    # HELPER: Leaf-level step of inner_prod.
    # Flattens both arrays, conjugates the first, takes the dot product and returns its real
    # part as a scalar. Arrays must have the same number of elements.
    return onp.real(onp.dot(onp.conj(x).reshape(-1), y.reshape(-1)))
  return tree_reduce(onp.add, tree_multimap(contract, xs, ys))


add = partial(tree_multimap, lambda x, y: onp.add(x, y, dtype=_dtype(x)))
sub = partial(tree_multimap, lambda x, y: onp.subtract(x, y, dtype=_dtype(x)))
conj = partial(tree_map, lambda x: onp.conj(x, dtype=_dtype(x)))

def scalar_mul(xs, a):
  # [FUNCTION DOC]
  # HELPER: Multiplies every leaf of a nested structure by the scalar a, keeping each leaf's
  # own dtype for the result. Returns a new structure of the same layout; inputs are not
  # mutated. Used to scale tangents and finite-difference results.
  return tree_map(lambda x: onp.multiply(x, a, dtype=_dtype(x)), xs)


def rand_like(rng, x):
  # [FUNCTION DOC]
  # HELPER: Random-tangent generator for the JVP/VJP checks.
  # Inputs: rng, a numpy random state; x, an array-like whose shape and dtype are copied.
  # Returns a standard-normal array with x's shape and dtype; for complex dtypes the imaginary
  # part is an independent normal sample. Advances the random state (side effect).
  shape = onp.shape(x)
  dtype = _dtype(x)
  randn = lambda: numpy_compat.np_asarray(rng.randn(*shape), dtype=dtype)
  if onp.issubdtype(dtype, onp.complexfloating):
    return randn() + dtype.type(1.0j) * randn()
  else:
    return randn()


def numerical_jvp(f, primals, tangents, eps=EPS):
  # [FUNCTION DOC]
  # HELPER: Central finite-difference estimate of a Jacobian-vector product for tests.
  # Inputs: f, a function taking positional arguments; primals and tangents, matching nested
  # sequences of argument values and directions; eps, the step size (default 1e-4).
  # Evaluates f at primals shifted by plus and minus eps times the tangents and returns the
  # difference divided by twice eps, as a nested structure like f's output.
  delta = scalar_mul(tangents, eps)
  f_pos = f(*add(primals, delta))
  f_neg = f(*sub(primals, delta))
  return scalar_mul(sub(f_pos, f_neg), 0.5 / eps)


def _merge_tolerance(tol, default):
  # [FUNCTION DOC]
  # HELPER: Combines a user tolerance with a default per-dtype table for the gradient checks.
  # Inputs: tol, nothing, a number or a mapping from dtypes to numbers; default, a mapping
  # from numpy dtypes to numbers. Nothing returns the default table itself; a number is
  # returned unchanged; a mapping yields a copy of the default overridden by the given entries.
  if tol is None:
    return default
  if not isinstance(tol, dict):
    return tol
  out = default.copy()
  for k, v in tol.items():
    out[onp.dtype(k)] = v
  return out

def check_jvp(f, f_jvp, args, atol=None, rtol=None, eps=EPS):
  # [FUNCTION DOC]
  # Verifies a forward-mode derivative implementation against finite differences.
  # Inputs: f, the function under test; f_jvp, a function taking the argument list and a
  # tangent list and returning the output value and output tangent; args, the argument values;
  # atol, rtol, optional tolerances merged with the gradient tolerance table; eps, the
  # finite-difference step. Uses a fixed-seed random tangent, checks the primal output against
  # calling f directly and the output tangent against the numerical estimate. Raises an
  # assertion error on mismatch (and a name error, since the closeness helper is broken).
  atol = _merge_tolerance(atol, default_gradient_tolerance)
  rtol = _merge_tolerance(rtol, default_gradient_tolerance)
  rng = onp.random.RandomState(0)
  tangent = tree_map(partial(rand_like, rng), args)
  v_out, t_out = f_jvp(args, tangent)
  v_out_expected = f(*args)
  t_out_expected = numerical_jvp(f, args, tangent, eps=eps)
  # In principle we should expect exact equality of v_out and v_out_expected,
  # but due to nondeterminism especially on GPU (e.g., due to convolution
  # autotuning) we only require "close".
  check_close(v_out, v_out_expected, atol=atol, rtol=rtol)
  check_close(t_out, t_out_expected, atol=atol, rtol=rtol)


def check_vjp(f, f_vjp, args, atol=None, rtol=None, eps=EPS):
  # [FUNCTION DOC]
  # Verifies a reverse-mode derivative implementation via the dot-product test.
  # Inputs: f, the function under test; f_vjp, a function that takes the arguments and returns
  # the output and a function mapping output cotangents to input cotangents; args, atol, rtol
  # and eps as in check_jvp. Compares primal outputs, then checks that the inner product of a
  # random tangent with the pulled-back cotangent equals the inner product of the numerical
  # output tangent with the cotangent. Raises an assertion error on mismatch (and a name error
  # because the helpers it calls are broken); returns nothing.
  atol = _merge_tolerance(atol, default_gradient_tolerance)
  rtol = _merge_tolerance(rtol, default_gradient_tolerance)
  _rand_like = partial(rand_like, onp.random.RandomState(0))
  v_out, vjpfun = f_vjp(*args)
  v_out_expected = f(*args)
  check_close(v_out, v_out_expected, atol=atol, rtol=rtol)
  tangent = tree_map(_rand_like, args)
  tangent_out = numerical_jvp(f, args, tangent, eps=eps)
  cotangent = tree_map(_rand_like, v_out)
  cotangent_out = conj(vjpfun(conj(cotangent)))
  ip = inner_prod(tangent, cotangent_out)
  ip_expected = inner_prod(tangent_out, cotangent)
  check_close(ip, ip_expected, atol=atol, rtol=rtol)


def device_under_test():
  # [FUNCTION DOC]
  # HELPER: Returns the value of the test_dut flag (empty string, cpu, gpu or tpu) naming the
  # device under test. Takes no inputs; reads the flag only.
  return FLAGS.test_dut

def if_device_under_test(device_type: Union[str, Sequence[str]],
                         if_true, if_false):
  # [FUNCTION DOC]
  # HELPER: Selects between two values depending on the device under test.
  # KNOWN ISSUE: the docstring has a typo ("of" instead of "or"); behavior is as described
  # otherwise.
  # Inputs: device_type, a device name or a sequence of names; if_true, returned when the
  # current device (from the test_dut flag) matches; if_false, returned otherwise. Reads the
  # flag only; no other side effects.
  """Chooses `if_true` of `if_false` based on device_under_test."""
  if device_under_test() in ([device_type] if isinstance(device_type, str)
                             else device_type):
    return if_true
  else:
    return if_false

def supported_dtypes():
  # [FUNCTION DOC]
  # HELPER: Returns the set of numpy/TF scalar types that tests may use on the current device:
  # a reduced set (bool, 32-bit integers, bfloat16, float32, complex64) on TPU and the full
  # integer, float and complex set elsewhere. Takes no inputs; reads the test_dut flag.
  if device_under_test() == "tpu":
    return {onp.bool_, onp.int32, onp.uint32, dtypes.bfloat16, onp.float32,
            onp.complex64}
  else:
    return {onp.bool_, onp.int8, onp.int16, onp.int32, onp.int64,
            onp.uint8, onp.uint16, onp.uint32, onp.uint64,
            dtypes.bfloat16, onp.float16, onp.float32, onp.float64,
            onp.complex64, onp.complex128}

def skip_if_unsupported_type(dtype):
  # [FUNCTION DOC]
  # HELPER: Skips the running test when a dtype is not supported on the current device.
  # Input dtype: a scalar type to look up in the supported set. Raises the unittest skip
  # exception naming the type and device when it is unsupported; otherwise returns nothing.
  if dtype not in supported_dtypes():
    raise unittest.SkipTest(
      f"Type {dtype} not supported on {device_under_test()}")

def skip_on_devices(*disabled_devices):
  # [FUNCTION DOC]
  # HELPER: Decorator factory that skips a test method on listed devices.
  # Input: any number of device names (matched against the test_dut flag). Returns a decorator
  # that wraps a test method so it raises the unittest skip exception when the current device
  # is in the list and otherwise calls the original method with its arguments.
  """A decorator for test methods to skip the test on certain devices."""
  def skip(test_method):
    # [FUNCTION DOC]
    # HELPER: The actual decorator returned by skip_on_devices; takes a test method and
    # returns the wrapper that performs the device check. Preserves the method's metadata.
    @functools.wraps(test_method)
    def test_method_wrapper(self, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: Wrapper that reads the current device, raises a skip with the test name and
      # upper-cased device when it is disabled, else forwards all arguments to the original
      # test method and returns its result.
      device = device_under_test()
      if device in disabled_devices:
        test_name = getattr(test_method, '__name__', '[unknown test]')
        raise unittest.SkipTest(
          f"{test_name} not supported on {device.upper()}.")
      return test_method(self, *args, **kwargs)
    return test_method_wrapper
  return skip


def skip_on_flag(flag_name, skip_value):
  # [FUNCTION DOC]
  # HELPER: Decorator factory that skips a test method when a named flag has a given value.
  # Inputs: flag_name, the name of an attribute on the shared flags object; skip_value, the
  # value that triggers skipping. Returns a decorator wrapping the test method.
  """A decorator for test methods to skip the test when flags are set."""
  def skip(test_method):        # pylint: disable=missing-docstring
    # [FUNCTION DOC]
    # HELPER: The decorator returned by skip_on_flag; wraps a test method (metadata preserved).
    @functools.wraps(test_method)
    def test_method_wrapper(self, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: Reads the flag; if it equals the skip value raises the unittest skip exception
      # naming the test and flag value, otherwise calls the original method with all
      # arguments and returns its result.
      flag_value = getattr(FLAGS, flag_name)
      if flag_value == skip_value:
        test_name = getattr(test_method, '__name__', '[unknown test]')
        raise unittest.SkipTest(
          f"{test_name} not supported when FLAGS.{flag_name} is {flag_value}")
      return test_method(self, *args, **kwargs)
    return test_method_wrapper
  return skip


def format_test_name_suffix(opname, shapes, dt):
  # [FUNCTION DOC]
  # HELPER: Builds a readable suffix for generated parameterized test names.
  # Inputs: opname, the operation name; shapes and dt, parallel sequences of shapes and dtypes.
  # Returns the capitalized op name joined by underscore with the underscore-joined shape/dtype
  # descriptions of each argument. Raises type errors for unsupported shape kinds.
  arg_descriptions = (format_shape_dtype_string(shape, dtype)
                      for shape, dtype in zip(shapes, dt))
  return '{}_{}'.format(opname.capitalize(), '_'.join(arg_descriptions))


# We use special symbols, represented as singleton objects, to distinguish
# between NumPy scalars, Python scalars, and 0-D arrays.
class ScalarShape:
  # [CLASS DOC]
  # HELPER: Base marker type for the two singleton scalar-shape symbols used by the random
  # generators. Behaves like an empty shape: zero length and no valid indices. Exists so code
  # can tell NumPy scalars, Python scalars and 0-d arrays apart.
  def __len__(self):
    # [FUNCTION DOC]
    # Always reports length zero, making the marker look like an empty shape tuple.
    return 0
  def __getitem__(self, i):
    # [FUNCTION DOC]
    # Indexing is never valid for a scalar shape: always raises an index error mentioning the
    # requested index.
    raise IndexError(f'index {i} out of range.')
class _NumpyScalar(ScalarShape):
  # [CLASS DOC]
  # HELPER: Marker subclass whose single instance (the numpy-scalar shape symbol) requests
  # that generated values be cast to NumPy scalar types.
  pass
class _PythonScalar(ScalarShape):
  # [CLASS DOC]
  # HELPER: Marker subclass whose single instance (the Python-scalar shape symbol) requests
  # that generated values be converted to plain Python scalars.
  pass
NUMPY_SCALAR_SHAPE = _NumpyScalar()
PYTHON_SCALAR_SHAPE = _PythonScalar()


def _dims_of_shape(shape):
  # [FUNCTION DOC]
  # HELPER: Normalizes a shape for the random generators. A list or tuple is returned as is,
  # a scalar-shape marker becomes an empty tuple, anything else raises a type error.
  """Converts `shape` to a tuple of dimensions."""
  if type(shape) in (list, tuple):
    return shape
  elif isinstance(shape, ScalarShape):
    return ()
  else:
    raise TypeError(type(shape))


def _cast_to_shape(value, shape, dtype):
  # [FUNCTION DOC]
  # HELPER: Converts generated values to the container kind demanded by a shape spec.
  # Inputs: value, array-like data; shape, a list/tuple or one of the two scalar-shape
  # markers; dtype, the target dtype. NumPy-scalar marker: returns a numpy scalar of dtype.
  # Python-scalar marker: returns a plain Python scalar. List/tuple: asserts value's shape
  # equals it and returns value. Anything else raises a type error.
  """Casts `value` to the correct Python type for `shape` and `dtype`."""
  if shape is NUMPY_SCALAR_SHAPE:
    # explicitly cast to NumPy scalar in case `value` is a Python scalar.
    return onp.dtype(dtype).type(value)
  elif shape is PYTHON_SCALAR_SHAPE:
    # explicitly cast to Python scalar via https://stackoverflow.com/a/11389998
    return numpy_compat.np_asarray(value).item()
  elif type(shape) in (list, tuple):
    assert onp.shape(value) == tuple(shape)
    return value
  else:
    raise TypeError(type(shape))


def dtype_str(dtype):
  # [FUNCTION DOC]
  # HELPER: Returns the numpy name of a dtype (for example a short string such as the name of
  # a 32-bit float) for use in test names. Input is anything convertible to a numpy dtype.
  return onp.dtype(dtype).name


def format_shape_dtype_string(shape, dtype):
  # [FUNCTION DOC]
  # HELPER: Describes a shape/dtype pair as text for generated test names.
  # NumPy-scalar marker gives just the dtype name; Python-scalar marker prefixes it with py;
  # a list or tuple gives the dtype name with comma-joined dimensions in brackets; an integer
  # gives a one-dimensional description; a numpy array is formatted inside the brackets.
  # Any other shape raises a type error.
  if shape is NUMPY_SCALAR_SHAPE:
    return dtype_str(dtype)
  elif shape is PYTHON_SCALAR_SHAPE:
    return 'py' + dtype_str(dtype)
  elif type(shape) in (list, tuple):
    shapestr = ','.join(str(dim) for dim in shape)
    return '{}[{}]'.format(dtype_str(dtype), shapestr)
  elif type(shape) is int:
    return '{}[{},]'.format(dtype_str(dtype), shape)
  elif isinstance(shape, onp.ndarray):
    return '{}[{}]'.format(dtype_str(dtype), shape)
  else:
    raise TypeError(type(shape))


def _rand_dtype(rand, shape, dtype, scale=1., post=lambda x: x):
  # [FUNCTION DOC]
  # HELPER: Core engine behind all the rand_* sampler factories.
  # Calls the random function with the dimensions of shape, scales and converts to dtype (for
  # complex dtypes adds an independent imaginary sample), applies the post-processing callable,
  # converts again to dtype and casts to the container kind required by shape (array, numpy
  # scalar or Python scalar). Inputs and return value are as in the docstring below; raises a
  # type error for unsupported shape kinds. Advances the random state of rand.
  """Produce random values given shape, dtype, scale, and post-processor.

  Args:
    rand: a function for producing random values of a given shape, e.g. a
      bound version of either onp.RandomState.randn or onp.RandomState.rand.
    shape: a shape value as a tuple of positive integers.
    dtype: a numpy dtype.
    scale: optional, a multiplicative scale for the random values (default 1).
    post: optional, a callable for post-processing the random values (default
      identity).

  Returns:
    An ndarray of the given shape and dtype using random values based on a call
    to rand but scaled, converted to the appropriate dtype, and post-processed.
  """
  r = lambda: numpy_compat.np_asarray(scale * rand(*_dims_of_shape(shape)),
                                      dtype)
  if onp.issubdtype(dtype, onp.complexfloating):
    vals = r() + 1.0j * r()
  else:
    vals = r()
  return _cast_to_shape(numpy_compat.np_asarray(post(vals), dtype), shape,
                        dtype)


def rand_default(scale=3):
  # [FUNCTION DOC]
  # HELPER: Sampler factory for the default test inputs. Input scale (default 3) multiplies
  # standard-normal samples. Returns a callable taking a shape and dtype and returning random
  # values; the underlying random state is freshly seeded with zero on each factory call.
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=scale)


def rand_nonzero():
  # [FUNCTION DOC]
  # HELPER: Sampler factory producing normal samples (scale 3, seed zero) in which exact zeros
  # are replaced by one. Returns a callable taking shape and dtype.
  post = lambda x: onp.where(x == 0, onp.array(1, dtype=x.dtype), x)
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=3, post=post)


def rand_positive():
  # [FUNCTION DOC]
  # HELPER: Sampler factory producing uniform samples in [0, 1) scaled by 2 and shifted by 1,
  # so values fall in [1, 3). Seed zero. Returns a callable taking shape and dtype.
  post = lambda x: x + 1
  rand = npr.RandomState(0).rand
  return partial(_rand_dtype, rand, scale=2, post=post)


def rand_small():
  # [FUNCTION DOC]
  # HELPER: Sampler factory producing small-magnitude normal samples (scale 1e-3, seed zero).
  # Returns a callable taking shape and dtype.
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=1e-3)


def rand_not_small(offset=10.):
  # [FUNCTION DOC]
  # HELPER: Sampler factory producing normal samples (scale 3, seed zero) pushed away from
  # zero by adding offset (default 10) to positive values and subtracting it from others.
  # Returns a callable taking shape and dtype.
  post = lambda x: x + onp.where(x > 0, offset, -offset)
  randn = npr.RandomState(0).randn
  return partial(_rand_dtype, randn, scale=3., post=post)


def rand_small_positive():
  # [FUNCTION DOC]
  # HELPER: Sampler factory producing tiny positive uniform samples (uniform in [0, 1) times
  # 2e-5, seed zero). Returns a callable taking shape and dtype.
  rand = npr.RandomState(0).rand
  return partial(_rand_dtype, rand, scale=2e-5)

def rand_uniform(low=0.0, high=1.0):
  # [FUNCTION DOC]
  # HELPER: Sampler factory for uniform samples in the range from low (default 0) to high
  # (default 1), seed zero. Asserts low is less than high (assertion error otherwise).
  # Returns a callable taking shape and dtype.
  assert low < high
  rand = npr.RandomState(0).rand
  post = lambda x: x * (high - low) + low
  return partial(_rand_dtype, rand, post=post)


def rand_some_equal():
  # [FUNCTION DOC]
  # HELPER: Sampler factory producing large-scale (100) normal samples in which about half of
  # the entries are overwritten by the first element, to create duplicate values. Seed zero.
  # Returns a callable taking shape and dtype.
  randn = npr.RandomState(0).randn
  rng = npr.RandomState(0)

  def post(x):
    # [FUNCTION DOC]
    # HELPER: Post-processor for rand_some_equal. Returns x unchanged if empty; otherwise
    # replaces each entry with the first element with probability one half (using the
    # factory's own random state, advancing it).
    x_ravel = x.ravel()
    if len(x_ravel) == 0:
      return x
    flips = rng.rand(*onp.shape(x)) < 0.5
    return onp.where(flips, x_ravel[0], x)

  return partial(_rand_dtype, randn, scale=100., post=post)


def rand_some_inf():
  # [FUNCTION DOC]
  # HELPER: Sampler factory for inputs containing infinities (docstring below).
  # KNOWN ISSUE: the complex branch is dead code; complex types are not subtypes of the
  # floating category, so complex dtypes return from the first check with plain default
  # samples and never receive infinities (acknowledged by the stray string literal TODO).
  # Returns a callable taking shape and dtype. Seeds its own random state with one.
  """Return a random sampler that produces infinities in floating types."""
  rng = npr.RandomState(1)
  base_rand = rand_default()

  """
  TODO: Complex numbers are not correctly tested
  If blocks should be switched in order, and relevant tests should be fixed
  """
  def rand(shape, dtype):
    # [FUNCTION DOC]
    # HELPER: The sampler returned by rand_some_inf. For non-floating dtypes returns default
    # samples. For floating dtypes draws default samples and independently replaces about ten
    # percent with positive infinity and about ten percent with negative infinity (negative
    # wins on overlap), then casts to the shape kind. Raises a type error for bad shapes.
    """The random sampler function."""
    if not onp.issubdtype(dtype, onp.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    if onp.issubdtype(dtype, onp.complexfloating):
      base_dtype = onp.real(onp.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             onp.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    dims = _dims_of_shape(shape)
    posinf_flips = rng.rand(*dims) < 0.1
    neginf_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = onp.where(posinf_flips, onp.array(onp.inf, dtype=dtype), vals)
    vals = onp.where(neginf_flips, onp.array(-onp.inf, dtype=dtype), vals)

    return _cast_to_shape(numpy_compat.np_asarray(vals, dtype=dtype), shape,
                          dtype)

  return rand

def rand_some_nan():
  # [FUNCTION DOC]
  # HELPER: Sampler factory for inputs containing NaNs (docstring below). Seeds its own random
  # state with one. Returns a callable taking shape and dtype.
  """Return a random sampler that produces nans in floating types."""
  rng = npr.RandomState(1)
  base_rand = rand_default()

  def rand(shape, dtype):
    # [FUNCTION DOC]
    # HELPER: The sampler returned by rand_some_nan. Complex dtypes: builds real and imaginary
    # parts recursively, each with NaNs. Non-floating dtypes: default samples. Floating
    # dtypes: default samples with about ten percent replaced by NaN, cast to the shape kind.
    # Raises a type error for unsupported shapes.
    """The random sampler function."""
    if onp.issubdtype(dtype, onp.complexfloating):
      base_dtype = onp.real(onp.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             onp.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    if not onp.issubdtype(dtype, onp.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    dims = _dims_of_shape(shape)
    nan_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = onp.where(nan_flips, onp.array(onp.nan, dtype=dtype), vals)

    return _cast_to_shape(numpy_compat.np_asarray(vals, dtype=dtype), shape,
                          dtype)

  return rand

def rand_some_inf_and_nan():
  # [FUNCTION DOC]
  # HELPER: Sampler factory for inputs containing infinities and NaNs.
  # KNOWN ISSUE: the docstring mentions only infinities, but the sampler also inserts NaNs;
  # and, as in rand_some_inf, the complex branch is dead because complex types fail the
  # floating check first.
  # Seeds its own random state with one. Returns a callable taking shape and dtype.
  """Return a random sampler that produces infinities in floating types."""
  rng = npr.RandomState(1)
  base_rand = rand_default()

  """
  TODO: Complex numbers are not correctly tested
  If blocks should be switched in order, and relevant tests should be fixed
  """
  def rand(shape, dtype):
    # [FUNCTION DOC]
    # HELPER: The sampler returned by rand_some_inf_and_nan. Non-floating (including complex)
    # dtypes get default samples. Floating dtypes get default samples where about ten percent
    # each are overwritten by positive infinity, negative infinity and NaN, applied in that
    # order so later ones win on overlap; cast to the shape kind. Type error on bad shapes.
    """The random sampler function."""
    if not onp.issubdtype(dtype, onp.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    if onp.issubdtype(dtype, onp.complexfloating):
      base_dtype = onp.real(onp.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             onp.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    dims = _dims_of_shape(shape)
    posinf_flips = rng.rand(*dims) < 0.1
    neginf_flips = rng.rand(*dims) < 0.1
    nan_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = onp.where(posinf_flips, onp.array(onp.inf, dtype=dtype), vals)
    vals = onp.where(neginf_flips, onp.array(-onp.inf, dtype=dtype), vals)
    vals = onp.where(nan_flips, onp.array(onp.nan, dtype=dtype), vals)

    return _cast_to_shape(numpy_compat.np_asarray(vals, dtype=dtype), shape,
                          dtype)

  return rand

# TODO(mattjj): doesn't handle complex types
def rand_some_zero():
  # [FUNCTION DOC]
  # HELPER: Sampler factory for inputs in which about half of the entries are zero (docstring
  # below). Seeds its own random state with one. Returns a callable taking shape and dtype.
  # Complex types are not handled specially (see the author's TODO above).
  """Return a random sampler that produces some zeros."""
  rng = npr.RandomState(1)
  base_rand = rand_default()

  def rand(shape, dtype):
    # [FUNCTION DOC]
    # HELPER: The sampler returned by rand_some_zero. Draws default samples and replaces each
    # with zero with probability one half, then casts to the shape kind. Type error on bad
    # shapes.
    """The random sampler function."""
    dims = _dims_of_shape(shape)
    zeros = rng.rand(*dims) < 0.5

    vals = base_rand(shape, dtype)
    vals = onp.where(zeros, onp.array(0, dtype=dtype), vals)

    return _cast_to_shape(numpy_compat.np_asarray(vals, dtype=dtype), shape,
                          dtype)

  return rand


def rand_int(low, high=None):
  # [FUNCTION DOC]
  # HELPER: Sampler factory for random integers. Inputs: low, the lower bound (or exclusive
  # upper bound if high is omitted, following numpy's convention); high, optional upper bound.
  # Returns a callable taking shape and dtype. Seed zero.
  randint = npr.RandomState(0).randint
  def fn(shape, dtype):
    # [FUNCTION DOC]
    # HELPER: The sampler returned by rand_int; returns random integers of the given dtype
    # and shape within the configured bounds. Advances the shared random state.
    return randint(low, high=high, size=shape, dtype=dtype)
  return fn

def rand_unique_int():
  # [FUNCTION DOC]
  # HELPER: Sampler factory for arrays of distinct integers. Seed zero. Returns a callable
  # taking shape and dtype.
  randchoice = npr.RandomState(0).choice
  def fn(shape, dtype):
    # [FUNCTION DOC]
    # HELPER: The sampler returned by rand_unique_int; picks values without replacement from
    # zero up to the number of elements in the shape, so the output is a random permutation
    # of those integers arranged in the requested shape and dtype.
    return randchoice(onp.arange(onp.prod(shape), dtype=dtype),
                      size=shape, replace=False)
  return fn

def rand_bool():
  # [FUNCTION DOC]
  # HELPER: Sampler factory for random booleans (each true with probability one half). Seed
  # zero. Returns a callable taking shape and dtype.
  rng = npr.RandomState(0)
  def generator(shape, dtype):
    # [FUNCTION DOC]
    # HELPER: The sampler returned by rand_bool; draws uniform samples, thresholds them at
    # one half and casts the result to the shape kind. The dtype argument is only used for
    # scalar-shape casts.
    return _cast_to_shape(rng.rand(*_dims_of_shape(shape)) < 0.5, shape, dtype)
  return generator

def check_raises(thunk, err_type, msg):
  # [FUNCTION DOC]
  # HELPER: Asserts that calling a function raises a given error whose text starts with msg.
  # KNOWN ISSUE: the failure marker is an assertion placed inside the guarded region, so if
  # err_type is the assertion error class (or a base of it) the "did not raise" case is
  # swallowed and then judged by message prefix.
  # Inputs: thunk, a no-argument callable; err_type, the expected exception class; msg, the
  # required message prefix. Raises an assertion error if nothing is raised or the message
  # does not match. Returns nothing.
  try:
    thunk()
    assert False
  except err_type as e:
    assert str(e).startswith(msg), "\n{}\n\n{}\n".format(e, msg)

def check_raises_regexp(thunk, err_type, pattern):
  # [FUNCTION DOC]
  # HELPER: Like check_raises but matches the error text against a regular expression anchored
  # at the start. KNOWN ISSUE: same swallowed-assertion weakness as check_raises when the
  # expected class is the assertion error.
  # Inputs: thunk, a no-argument callable; err_type; pattern, a regular expression string.
  # Raises an assertion error when no error occurs or the message does not match.
  try:
    thunk()
    assert False
  except err_type as e:
    assert re.match(pattern, str(e)), "{}\n\n{}\n".format(e, pattern)


def _iter_eqns(jaxpr):
  # [FUNCTION DOC]
  # HELPER: Leftover from the JAX test utilities; generator over all equations of a program
  # representation including nested ones.
  # KNOWN ISSUE: depends on the name core, which is never imported here, so iterating past the
  # top-level equations raises a name error. It is not meaningful for TF NumPy.
  # Yields each equation of the given program, then recursively those of its sub-programs.
  # TODO(necula): why doesn't this search in params?
  for eqn in jaxpr.eqns:
    yield eqn
  for subjaxpr in core.subjaxprs(jaxpr):
    yield from _iter_eqns(subjaxpr)

def assert_dot_precision(expected_precision, fun, *args):
  # [FUNCTION DOC]
  # HELPER: Leftover from the JAX test utilities.
  # KNOWN ISSUE: uses the names api and lax, which are never imported here, so calling it
  # raises a name error.
  # Intended behavior: traces fun on args, collects the precision setting of every dot-general
  # equation and asserts each equals expected_precision, raising an assertion error with a
  # message otherwise. Returns nothing.
  jaxpr = api.make_jaxpr(fun)(*args)
  precisions = [eqn.params['precision'] for eqn in _iter_eqns(jaxpr.jaxpr)
                if eqn.primitive == lax.dot_general_p]
  for precision in precisions:
    msg = "Unexpected precision: {} != {}".format(expected_precision, precision)
    assert precision == expected_precision, msg


_CACHED_INDICES: Dict[int, Sequence[int]] = {}

def cases_from_list(xs):
  # [FUNCTION DOC]
  # HELPER: Subsamples a list of parameterized test cases.
  # Input xs: any iterable of cases. Returns a list of at most num_generated_cases of them,
  # chosen through a fixed-seed permutation that is cached per list length in a module-level
  # cache (side effect), so repeated calls with the same length pick the same indices.
  xs = list(xs)
  n = len(xs)
  k = min(n, FLAGS.num_generated_cases)
  # Random sampling for every parameterized test is expensive. Do it once and
  # cache the result.
  indices = _CACHED_INDICES.get(n)
  if indices is None:
    rng = npr.RandomState(42)
    _CACHED_INDICES[n] = indices = rng.permutation(n)
  return [xs[i] for i in indices[:k]]

def cases_from_gens(*gens):
  # [FUNCTION DOC]
  # HELPER: Generator of parameterized cases built from value generators.
  # Inputs: any number of callables, each taking a size and returning a value. For sizes 1, 3
  # and 10 it yields a number of cases (num_generated_cases divided by three, plus one) per
  # size; each case is a tuple of a name suffix made of size and index followed by one value
  # from each generator.
  sizes = [1, 3, 10]
  cases_per_size = int(FLAGS.num_generated_cases / len(sizes)) + 1
  for size in sizes:
    for i in range(cases_per_size):
      yield ('_{}_{}'.format(size, i),) + tuple(gen(size) for gen in gens)


def to_np(a):
  # [FUNCTION DOC]
  # HELPER: Converts every leaf of a nested structure to a TF NumPy array via the array-ops
  # asarray function. Returns a structure of the same layout.
  return nest.map_structure(np_array_ops.asarray, a)


def to_tf_fn(f):
  # [FUNCTION DOC]
  # HELPER: Wraps a function so that its positional arguments are first converted to TF NumPy
  # arrays. Input f, a callable; returns the wrapper callable (used by check_grads).
  return lambda *args: f(*to_np(args))


class TestCase(parameterized.TestCase):
  # [CLASS DOC]
  # HELPER: Base test case for the TF NumPy tests, derived from the absl parameterized test
  # case. Holds a per-test seeded random state (seeded from a checksum of the test method
  # name) and adds tolerance-aware array comparison, a compile-and-compare harness for jitted
  # functions, a numpy-reference check and a finite-difference gradient check.
  """Base class for tests including numerical checks and boilerplate."""

  # copied from jax.test_util
  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: Per-test fixture. Calls the parent setup, then stores a numpy random state
    # seeded with a checksum of the test method's name, making random inputs deterministic
    # per test. Sets the private random-state attribute.
    super().setUp()
    self._rng = npr.RandomState(zlib.adler32(self._testMethodName.encode()))

  # copied from jax.test_util
  def rng(self):
    # [FUNCTION DOC]
    # HELPER: Returns the per-test numpy random state created in setUp (the same object each
    # call, so draws advance shared state).
    return self._rng

  # TODO(mattjj): this obscures the error messages from failures, figure out how
  # to re-enable it
  # def tearDown(self) -> None:
  #   assert core.reset_trace_state()

  def assertArraysAllClose(self, x, y, check_dtypes, atol=None, rtol=None):
    # [FUNCTION DOC]
    # HELPER: Assertion that two arrays have equal shape and are close.
    # Inputs: x and y, arrays; check_dtypes, whether to also compare dtypes; atol and rtol,
    # optional tolerance specifications (the larger per-dtype tolerance of the two arrays is
    # used). Fails the test on shape mismatch or value difference; the dtype check only has
    # effect when the enable_x64 flag is on. Returns nothing.
    """Assert that x and y are close (up to numerical tolerances)."""
    self.assertEqual(x.shape, y.shape)
    atol = max(tolerance(_dtype(x), atol), tolerance(_dtype(y), atol))
    rtol = max(tolerance(_dtype(x), rtol), tolerance(_dtype(y), rtol))

    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol)

    if check_dtypes:
      self.assertDtypesMatch(x, y)

  def assertDtypesMatch(self, x, y):
    # [FUNCTION DOC]
    # HELPER: Asserts x and y have equal dtypes, but only when the enable_x64 flag is on;
    # otherwise does nothing (so dtype checks are silently skipped by default).
    if FLAGS.enable_x64:
      self.assertEqual(_dtype(x), _dtype(y))

  def assertAllClose(self, x, y, check_dtypes, atol=None, rtol=None):
    # [FUNCTION DOC]
    # HELPER: Recursive closeness assertion. Mappings must have equal key sets and close
    # values; non-array sequences must have equal length and close elements; arrays and
    # scalars are converted to arrays and compared with the array assertion; other values must
    # compare equal. Inputs: x, y, check_dtypes, atol, rtol as in assertArraysAllClose.
    # Raises a type error for incomparable unequal values; fails the test otherwise.
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x:
        self.assertAllClose(x[k], y[k], check_dtypes, atol=atol, rtol=rtol)
    elif is_sequence(x) and not hasattr(x, '__array__'):
      self.assertTrue(is_sequence(y) and not hasattr(y, '__array__'))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self.assertAllClose(x_elt, y_elt, check_dtypes, atol=atol, rtol=rtol)
    elif hasattr(x, '__array__') or onp.isscalar(x):
      self.assertTrue(hasattr(y, '__array__') or onp.isscalar(y))
      if check_dtypes:
        self.assertDtypesMatch(x, y)
      x = numpy_compat.np_asarray(x)
      y = numpy_compat.np_asarray(y)
      self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))

  def assertMultiLineStrippedEqual(self, expected, what):
    # [FUNCTION DOC]
    # HELPER: Compares two strings ignoring whitespace around line breaks and at the ends.
    # Inputs: expected and what, strings. Fails the test with a message showing both texts
    # when they differ after normalization. Returns nothing.
    """Asserts two strings are equal, after stripping each line."""
    ignore_space_re = re.compile(r'\s*\n\s*')
    expected_clean = re.sub(ignore_space_re, '\n', expected.strip())
    what_clean = re.sub(ignore_space_re, '\n', what.strip())
    self.assertMultiLineEqual(expected_clean, what_clean,
                              msg="Found\n{}\nExpecting\n{}".format(what, expected))

  def _CheckAgainstNumpy(self, numpy_reference_op, lax_op, args_maker,
                         check_dtypes=True, tol=None):
    # [FUNCTION DOC]
    # HELPER: Compares an op against its numpy reference. Inputs: the reference function,
    # the function under test, a callable producing the argument tuple, check_dtypes, and tol
    # used for both absolute and relative tolerance. Calls both on the same arguments and
    # asserts closeness. Fails the test on mismatch.
    args = args_maker()
    lax_ans = lax_op(*args)
    numpy_ans = numpy_reference_op(*args)
    self.assertAllClose(numpy_ans, lax_ans, check_dtypes=check_dtypes,
                        atol=tol, rtol=tol)

  def _CompileAndCheck(self,
                       fun,
                       args_maker,
                       check_dtypes=True,
                       rtol=None,
                       atol=None,
                       check_eval_on_shapes=True,
                       check_incomplete_shape=True,
                       check_unknown_rank=True,
                       static_argnums=(),
                       check_experimental_compile=True,
                       check_xla_forced_compile=True):
    # [FUNCTION DOC]
    # HELPER: Main harness comparing eager execution, compiled execution and shape inference.
    # KNOWN ISSUE: the compile-check path uses jit options experimental_compile and
    # xla_forced_compile of the extensions module, and the retrace test is skipped whenever
    # integer argument dtypes differ between draws; with Python-scalar arguments dtype checks
    # are disabled silently.
    # Inputs are described in the docstring below. Runs the function eagerly, verifies output
    # shapes agree with numpy, then (a) compiles and runs twice asserting the Python body is
    # not re-executed on the second call, compares results, and re-runs with fresh arguments
    # to confirm no retracing, in plain, experimental-compile and XLA-forced-compile modes as
    # enabled; (b) optionally checks shape/dtype inference; (c) optionally compiles with
    # partially known shapes and unknown rank input signatures. Fails the test on any mismatch.
    """Compiles the function and checks the results.

    Args:
      fun: the function to be checked.
      args_maker: a callable that returns a tuple which will be used as the
        positional arguments.
      check_dtypes: whether to check that the result dtypes from non-compiled
        and compiled runs agree.
      rtol: relative tolerance for allclose assertions.
      atol: absolute tolerance for allclose assertions.
      check_eval_on_shapes: whether to run `eval_on_shapes` on the function and
        check that the result shapes and dtypes are correct.
      check_incomplete_shape: whether to check that the function can handle
        incomplete shapes (including those with and without a known rank).
      check_unknown_rank: (only has effect when check_incomplete_shape is True)
        whether to check that the function can handle unknown ranks.
      static_argnums: indices of arguments to be treated as static arguments for
        `jit` and `eval_on_shapes`.
      check_experimental_compile: whether to check compilation with
        experimental_compile=True (in addition to compilation without the flag).
      check_xla_forced_compile: whether to check compilation with
        forced_compile=True (in addition to compilation without the flag). This
        flag is different from experimental_compile because it enforces
        whole-function compilation while the latter doesn't. TPU requires
        whole-function compilation.
    """
    args = args_maker()

    for x in args:
      if not hasattr(x, 'dtype'):
        # If there is a input that doesn't have dtype info, jit and
        # eval_on_shapes may pick a different dtype for it than numpy, so we
        # skip the dtype check.
        check_dtypes = False

    python_ans = fun(*args)

    python_shapes = nest.map_structure(onp.shape, python_ans)
    onp_shapes = nest.map_structure(
        lambda x: onp.shape(numpy_compat.np_asarray(x)), python_ans
    )
    self.assertEqual(python_shapes, onp_shapes)

    def check_compile(**kwargs):
      # [FUNCTION DOC]
      # HELPER: Local step of the compile-and-check harness. Takes extra jit options as
      # keyword arguments. Jits a wrapper of the function, runs it twice (the second time the
      # wrapper must not execute Python), compares to the eager answer, then runs with new
      # arguments, skipping the no-retrace condition when integer dtypes differ. Fails the
      # test on mismatch; returns nothing.
      # `wrapped_fun` and `python_should_be_executing` are used to check that
      # when the jitted function is called the second time, the original Python
      # function won't be executed.
      def wrapped_fun(*args):
        # [FUNCTION DOC]
        # HELPER: Wrapper around the function under test that first asserts the enclosing
        # flag says Python execution is expected (catching unexpected retracing), then
        # forwards the arguments and returns the result.
        self.assertTrue(python_should_be_executing)
        return fun(*args)

      cfun = nje.jit(wrapped_fun, static_argnums=static_argnums, **kwargs)
      python_should_be_executing = True
      monitored_ans = cfun(*args)

      python_should_be_executing = False
      compiled_ans = cfun(*args)

      self.assertAllClose(python_ans, monitored_ans, check_dtypes, atol, rtol)
      self.assertAllClose(python_ans, compiled_ans, check_dtypes, atol, rtol)

      # Run `cfun` with a different set of arguments to check that changing
      # arguments won't cause recompilation.

      new_args = args_maker()

      skip_retracing_test = False
      for old, new in zip(nest.flatten(args), nest.flatten(new_args)):
        if nje.most_precise_int_dtype(old) != nje.most_precise_int_dtype(new):
          # If the old and new arguments result in different dtypes (because
          # they fall into different value ranges), tf-numpy will retrace, so we
          # skip the no-retrace test.
          skip_retracing_test = True

      if not skip_retracing_test:
        python_should_be_executing = True
        new_python_ans = fun(*new_args)
        python_should_be_executing = False
        compiled_ans = cfun(*new_args)
        self.assertAllClose(new_python_ans, compiled_ans, check_dtypes, atol,
                            rtol)

    check_compile()
    if check_experimental_compile:
      check_compile(experimental_compile=True)
    if check_xla_forced_compile:
      check_compile(xla_forced_compile=True)

    if check_eval_on_shapes:
      # Check that nje.eval_on_shapes can get complete output shapes given
      # complete input shapes.
      cfun = nje.eval_on_shapes(fun, static_argnums=static_argnums)
      compiled_ans = cfun(*args)
      flat_python_ans = nest.flatten(python_ans)
      flat_compiled_ans = nest.flatten(compiled_ans)
      self.assertEqual(len(flat_python_ans), len(flat_compiled_ans))
      for a, b in zip(flat_python_ans, flat_compiled_ans):
        if hasattr(a, 'shape'):
          self.assertEqual(a.shape, b.shape)
        if check_dtypes and hasattr(a, 'dtype'):
          self.assertEqual(dtypes.as_dtype(a.dtype), b.dtype)

    # If some argument doesn't have a `dtype` attr (e.g. a Python scalar), we
    # skip incomplete-shape checks, since shape specs need dtype. It's OK to
    # skip since the same incomplete-shape checks will run for []-shaped arrays.
    if check_incomplete_shape and all(hasattr(x, 'dtype') for x in args):
      # Check partial shapes with known ranks.
      # Numpy scalars (created by e.g. np.int32(5)) have `dtype` but not
      # `shape`.
      if all(hasattr(x, 'shape') for x in args):
        specs = [
            tensor.TensorSpec([None] * len(x.shape), x.dtype) for x in args
        ]
        cfun = nje.jit(
            fun, static_argnums=static_argnums, input_signature=specs
        )
        compiled_ans = cfun(*args)
        self.assertAllClose(python_ans, compiled_ans, check_dtypes, atol, rtol)

      if check_unknown_rank:
        # Check unknown ranks.
        specs = [tensor.TensorSpec(None, x.dtype) for x in args]
        cfun = nje.jit(
            fun, static_argnums=static_argnums, input_signature=specs)
        compiled_ans = cfun(*args)
        self.assertAllClose(python_ans, compiled_ans, check_dtypes, atol, rtol)

  def check_grads(self, f, args, atol=None, rtol=None, delta=None):
    # [FUNCTION DOC]
    # HELPER: Compares analytical and numerical gradients of f at args using TF's gradient
    # checker. Inputs: f, function; args, list of values; atol, rtol tolerances; delta, step
    # size (default: cube root of machine epsilon for the result dtype of the arguments).
    # Fails the test when the Jacobians differ. Returns nothing.
    """Check gradients against finite differences.

    Args:
      f: function to check at ``f(*args)``.
      args: a list or tuple of argument values.
      atol: absolute tolerance for gradient equality.
      rtol: relative tolerance for gradient equality.
      delta: step size used for finite differences.
    """
    if delta is None:
      # Optimal stepsize for central difference is O(epsilon^{1/3}).
      dtype = np_utils.result_type(*args)
      epsilon = onp.finfo(dtype).eps
      delta = epsilon ** (1.0 / 3.0)
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        to_tf_fn(f), args, delta=delta)
    self.assertAllClose(theoretical, numerical, check_dtypes=False, atol=atol,
                        rtol=rtol)


@contextmanager
def ignore_warning(**kw):
  # [FUNCTION DOC]
  # HELPER: Context manager (generator-based) that suppresses warnings matching the given
  # filter keywords (for example category or message) inside its with-block, restoring the
  # previous warning filters on exit. Yields nothing.
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", **kw)
    yield


def disable(_):
  # [FUNCTION DOC]
  # HELPER: Decorator that disables a test. The decorated function (ignored) is replaced by a
  # wrapper that skips the test with the message that it is disabled. Returns the wrapper.

  def wrapper(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Replacement test body; skips the test, ignoring all arguments.
    self.skipTest('Test is disabled')

  return wrapper
