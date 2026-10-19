# [MODULE DOC]
# Test module for the variable initializers in the init ops module and for the range and linear
# spacing ops. Contains: module-level helper functions that compare initializer outputs for
# identity or duplication; tests for constant, random normal, truncated normal, random uniform,
# uniform unit scaling and variance scaling initializers; tests for the range, linspace and
# n-dimensional linspace math ops; a device-placement test for variables; and tests for the
# orthogonal, delta-orthogonal, 1-D/2-D/3-D convolutional orthogonal and identity initializers.
# Most tests are graph-mode (deprecated v1 style). No import-time side effects beyond imports;
# run through the test runner entry point when executed as a script.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.ops."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util.numpy_compat import np_where


# Returns true iff the two initializers produce the same tensor to
# within a tiny tolerance.
def identicaltest(tc, init1, init2, shape=None):
  # [FUNCTION DOC]
  # HELPER: shared by the initializer tests to decide whether two seeded initializers agree.
  # Inputs: tc (a test case providing a test session), init1 and init2 (callables taking a shape and
  # returning a tensor), shape (optional; default is a vector of length 100). Evaluates each
  # initializer in its own fresh graph and session and compares the results with an extremely tight
  # tolerance (1e-15 relative and absolute), so effectively exact equality. Returns True if equal,
  # False otherwise. Side effect: creates and discards two graphs/sessions.
  """Tests if two initializations are identical to within tiny tolerances.

  Args:
    tc: An instance of TensorFlowTestCase.
    init1: An Initializer that generates a tensor of a given shape
    init2: An Initializer that generates a tensor of a given shape
    shape: Shape of the tensor to initialize or `None` to use a vector of length
      100.

  Returns:
    True or False as determined by test.
  """
  if shape is None:
    shape = [100]
  with tc.test_session(graph=ops.Graph()):
    t1 = init1(shape).eval()
  with tc.test_session(graph=ops.Graph()):
    t2 = init2(shape).eval()
  return np.allclose(t1, t2, rtol=1e-15, atol=1e-15)


def duplicated_initializer(tc, init, graph_seed, shape=None):
  # [FUNCTION DOC]
  # HELPER: shared by the initializer tests to check that repeated use of a random initializer in
  # one graph yields different values.
  # Inputs: tc (test case), init (callable taking a shape and returning a tensor), graph_seed (the
  # graph-level random seed to set), shape (optional; default a vector of length 100). In a fresh
  # graph and session it sets the graph seed, evaluates the initializer twice, and compares the two
  # results with tolerance 1e-15. Returns True if the two results are equal, False if they differ
  # (callers expect False for random initializers). Side effect: sets the graph random seed.
  """Tests duplicated random initializer within the same graph.

  This test generates two random kernels from the same initializer to the same
  graph, and checks if the results are close enough. Even given the same global,
  seed, two different instances of random kernels should generate different
  results.

  Args:
    tc: An instance of TensorFlowTestCase.
    init: An Initializer that generates a tensor of a given shape
    graph_seed: A graph-level seed to use.
    shape: Shape of the tensor to initialize or `None` to use a vector of length
      100.

  Returns:
    True or False as determined by test.
  """
  if shape is None:
    shape = [100]
  with tc.test_session(graph=ops.Graph()):
    random_seed.set_random_seed(graph_seed)
    t1 = init(shape).eval()
    t2 = init(shape).eval()
    return np.allclose(t1, t2, rtol=1e-15, atol=1e-15)


def _init_sampler(tc, init, num):
  # [FUNCTION DOC]
  # HELPER: builds a sampling function from an initializer for statistical tests.
  # KNOWN ISSUE: nothing in this file calls it, so it appears to be dead code.
  # Inputs: tc (test case), init (callable taking a shape), num (length of the vector to sample).
  # Returns a zero-argument function that, each time it is called, evaluates the initializer for a
  # vector of that length in a test session and returns the resulting numpy array.
  """Returns a func to generate a random tensor of shape [num].

  Args:
    tc: An instance of TensorFlowTestCase.
    init: An Initializer that generates a tensor of a given shape
    num: Size of 1D tensor to create.

  Returns:
    Function to generate a random tensor.
  """

  def func():
    # [FUNCTION DOC]
    # HELPER: the sampler returned by the enclosing helper. Opens a test session on the captured
    # test case, evaluates the captured initializer for a vector of the captured length and
    # returns the resulting numpy array.
    with tc.test_session():
      return init([num]).eval()

  return func


class ConstantInitializersTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the zeros, ones and constant initializers used through variable-scope variable
  # creation, including integer and tuple values, n-dimensional values and invalid value types.
  # No shared fixtures; each test creates its own variables.

  @test_util.run_deprecated_v1
  def testZerosInitializer(self):
    # [FUNCTION DOC]
    # Test scenario (graph mode): create a two-by-three variable with the zeros initializer, run its
    # initializer and expect all zeros. Returns nothing.
    with self.session():
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.zeros_initializer())
      self.evaluate(x.initializer)
      self.assertAllEqual(x, np.zeros(shape))

  @test_util.run_deprecated_v1
  def testOnesInitializer(self):
    # [FUNCTION DOC]
    # Test scenario (graph mode): create a two-by-three variable with the ones initializer, run its
    # initializer and expect all ones. Returns nothing.
    with self.session():
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.ones_initializer())
      self.evaluate(x.initializer)
      self.assertAllEqual(x, np.ones(shape))

  @test_util.run_deprecated_v1
  def testConstantZeroInitializer(self):
    # [FUNCTION DOC]
    # Test scenario (graph mode): constant initializer with value 0.0 for a two-by-three variable;
    # expects all zeros. Returns nothing.
    with self.session():
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.constant_initializer(0.0))
      self.evaluate(x.initializer)
      self.assertAllEqual(x, np.zeros(shape))

  @test_util.run_deprecated_v1
  def testConstantOneInitializer(self):
    # [FUNCTION DOC]
    # Test scenario (graph mode): constant initializer with value 1.0 for a two-by-three variable;
    # expects all ones. Returns nothing.
    with self.session():
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x", shape=shape, initializer=init_ops.constant_initializer(1.0))
      self.evaluate(x.initializer)
      self.assertAllEqual(x, np.ones(shape))

  @test_util.run_deprecated_v1
  def testConstantIntInitializer(self):
    # [FUNCTION DOC]
    # Test scenario (graph mode): constant initializer with integer value 7 for a two-by-three
    # variable of 32-bit integer type. Expects the variable's base dtype to be 32-bit integer and all
    # values to be 7. Returns nothing.
    with self.session():
      shape = [2, 3]
      x = variable_scope.get_variable(
          "x",
          shape=shape,
          dtype=dtypes.int32,
          initializer=init_ops.constant_initializer(7))
      self.evaluate(x.initializer)
      self.assertEqual(x.dtype.base_dtype, dtypes.int32)
      self.assertAllEqual(x, 7 * np.ones(shape, dtype=np.int32))

  @test_util.run_deprecated_v1
  def testConstantTupleInitializer(self):
    # [FUNCTION DOC]
    # Test scenario (graph mode): constant initializer given the tuple ten, twenty, thirty for a
    # length-three 32-bit integer variable. Expects the dtype and the values to match the tuple.
    # Returns nothing.
    with self.session():
      shape = [3]
      x = variable_scope.get_variable(
          "x",
          shape=shape,
          dtype=dtypes.int32,
          initializer=init_ops.constant_initializer((10, 20, 30)))
      self.evaluate(x.initializer)
      self.assertEqual(x.dtype.base_dtype, dtypes.int32)
      self.assertAllEqual(x, [10, 20, 30])

  def _testNDimConstantInitializer(self, name, value, shape, expected):
    # [FUNCTION DOC]
    # HELPER: shared check for the n-dimensional constant initializer test.
    # Inputs: name (variable name), value (list or numpy array of values), shape (variable shape),
    # expected (flat list of expected values). Creates a 32-bit integer variable with the constant
    # initializer, runs the initializer, flattens the variable and asserts the length and every element
    # match the expected list. Returns nothing.
    with self.cached_session():
      init = init_ops.constant_initializer(value, dtype=dtypes.int32)
      x = variable_scope.get_variable(name, shape=shape, initializer=init)
      self.evaluate(x.initializer)

      actual = array_ops.reshape(x, [-1]).eval()
      self.assertEqual(len(actual), len(expected))
      for a, e in zip(actual, expected):
        self.assertEqual(a, e)

  @test_util.run_deprecated_v1
  def testNDimConstantInitializer(self):
    # [FUNCTION DOC]
    # Test scenario: initialize a two-by-three variable from six values given as a plain list, a flat
    # numpy array and a two-by-three numpy array, via the shared helper. Expects the flattened result
    # to equal the six values in each case. Graph mode. Returns nothing.
    value = [0, 1, 2, 3, 4, 5]
    shape = [2, 3]
    expected = list(value)

    self._testNDimConstantInitializer("list", value, shape, expected)
    self._testNDimConstantInitializer("ndarray", np.asarray(value), shape,
                                      expected)
    self._testNDimConstantInitializer("2D-ndarray",
                                      np.asarray(value).reshape(tuple(shape)),
                                      shape, expected)

  def _testNDimConstantInitializerLessValues(self, name, value, shape,
                                             expected):
    # [FUNCTION DOC]
    # HELPER: shared check for the constant initializer when fewer values are given than the
    # variable has elements.
    # Inputs: name, value, shape and expected (the provided values). Creates the integer variable,
    # flattens it, asserts the variable has more elements than expected values, and that each element
    # equals the corresponding expected value, with the last expected value repeated for the extra
    # positions. Returns nothing.
    with self.cached_session():
      init = init_ops.constant_initializer(value, dtype=dtypes.int32)
      x = variable_scope.get_variable(name, shape=shape, initializer=init)
      self.evaluate(x.initializer)

      actual = array_ops.reshape(x, [-1]).eval()
      self.assertGreater(len(actual), len(expected))
      for i in range(len(actual)):
        a = actual[i]
        e = expected[i] if i < len(expected) else expected[-1]
        self.assertEqual(a, e)

  @test_util.run_deprecated_v1
  def testNDimConstantInitializerLessValues(self):
    # [FUNCTION DOC]
    # Test scenario: six values (as a list, a flat array and a two-by-three array) are used to
    # initialize a two-by-four variable. Expects the first six elements to match and the last
    # value to be repeated to fill the rest, via the shared helper. Graph mode. Returns nothing.
    value = [0, 1, 2, 3, 4, 5]
    shape = [2, 4]
    expected = list(value)

    self._testNDimConstantInitializerLessValues("list", value, shape, expected)
    self._testNDimConstantInitializerLessValues("ndarray", np.asarray(value),
                                                shape, expected)
    self._testNDimConstantInitializerLessValues(
        "2D-ndarray",
        np.asarray(value).reshape(tuple([2, 3])), shape, expected)

  def _testNDimConstantInitializerMoreValues(self, value, shape):
    # [FUNCTION DOC]
    # HELPER: shared check for the constant initializer when too many values are given.
    # Inputs: value (list or array with more elements than the shape holds) and shape. Resets the
    # default graph (side effect), then asserts that creating a variable with the constant
    # initializer raises a value error. Returns nothing.
    ops.reset_default_graph()
    with self.cached_session():
      init = init_ops.constant_initializer(value, dtype=dtypes.int32)
      self.assertRaises(
          ValueError,
          variable_scope.get_variable,
          "x",
          shape=shape,
          initializer=init)

  @test_util.run_deprecated_v1
  def testNDimConstantInitializerMoreValues(self):
    # [FUNCTION DOC]
    # Test scenario: eight values (as a list, a flat array and a two-by-four array) given for a
    # two-by-three variable. Expects variable creation to raise a value error in each case, via
    # the shared helper. Graph mode. Returns nothing.
    value = [0, 1, 2, 3, 4, 5, 6, 7]
    shape = [2, 3]
    self._testNDimConstantInitializerMoreValues(value, shape)
    self._testNDimConstantInitializerMoreValues(np.asarray(value), shape)
    self._testNDimConstantInitializerMoreValues(
        np.asarray(value).reshape(tuple([2, 4])), shape)

  def testInvalidValueTypeForConstantInitializerCausesTypeError(self):
    # [FUNCTION DOC]
    # Test scenario: passing a tensor, or a variable, as the value of the constant initializer must
    # raise a type error whose message says the initial value type is invalid and names the tensor or
    # variable type. Returns nothing.
    c = constant_op.constant([1.0, 2.0, 3.0])
    with self.assertRaisesRegex(TypeError,
                                r"Invalid type for initial value=.*Tensor.*"):
      init_ops.constant_initializer(c, dtype=dtypes.float32)
    v = variables.Variable([3.0, 2.0, 1.0])
    with self.assertRaisesRegex(
        TypeError, r"Invalid type for initial value=.*Variable.*"):
      init_ops.constant_initializer(v, dtype=dtypes.float32)


class RandomNormalInitializationTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the random normal initializer: same seed gives identical values, different seeds
  # differ, repeated use in one graph gives different values, and an invalid dtype is rejected.

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    # [FUNCTION DOC]
    # Test scenario: for 32- and 64-bit floats, two random normal initializers with the same seed
    # (mean 0, standard deviation 1) must produce identical tensors. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      init2 = init_ops.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    # [FUNCTION DOC]
    # Test scenario: for 32- and 64-bit floats, two random normal initializers with seeds one and two
    # must produce different tensors. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.random_normal_initializer(0.0, 1.0, seed=1, dtype=dtype)
      init2 = init_ops.random_normal_initializer(0.0, 1.0, seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    # [FUNCTION DOC]
    # Test scenario: one unseeded random normal initializer evaluated twice in a graph with a fixed
    # graph seed must give different values. Graph mode. Returns nothing.
    init = init_ops.random_normal_initializer(0.0, 1.0)
    self.assertFalse(duplicated_initializer(self, init, 1))

  def testInvalidDataType(self):
    # [FUNCTION DOC]
    # Test scenario: constructing a random normal initializer with a string dtype must raise a value
    # error. Returns nothing.
    self.assertRaises(
        ValueError,
        init_ops.random_normal_initializer,
        0.0,
        1.0,
        dtype=dtypes.string)


class TruncatedNormalInitializationTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the truncated normal initializer, with the same four checks as the random normal
  # test case: identical for equal seeds, different for different seeds, distinct on repeated use,
  # and rejection of a string dtype.

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    # [FUNCTION DOC]
    # Test scenario: for 32- and 64-bit floats, two truncated normal initializers with the same seed
    # must produce identical tensors. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.truncated_normal_initializer(
          0.0, 1.0, seed=1, dtype=dtype)
      init2 = init_ops.truncated_normal_initializer(
          0.0, 1.0, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    # [FUNCTION DOC]
    # Test scenario: for 32- and 64-bit floats, truncated normal initializers with seeds one and two
    # must produce different tensors. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.truncated_normal_initializer(
          0.0, 1.0, seed=1, dtype=dtype)
      init2 = init_ops.truncated_normal_initializer(
          0.0, 1.0, seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    # [FUNCTION DOC]
    # Test scenario: one unseeded truncated normal initializer used twice under a fixed graph seed
    # must produce different values. Graph mode. Returns nothing.
    init = init_ops.truncated_normal_initializer(0.0, 1.0)
    self.assertFalse(duplicated_initializer(self, init, 1))

  def testInvalidDataType(self):
    # [FUNCTION DOC]
    # Test scenario: constructing a truncated normal initializer with a string dtype must raise a
    # value error. Returns nothing.
    self.assertRaises(
        ValueError,
        init_ops.truncated_normal_initializer,
        0.0,
        1.0,
        dtype=dtypes.string)


class RandomUniformInitializationTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the random uniform initializer: determinism for equal seeds, difference for
  # different seeds (including integer dtypes) and distinct values on repeated use.

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    # [FUNCTION DOC]
    # Test scenario: for 32-bit float, 64-bit float and 64-bit integer, two random uniform
    # initializers over zero to seven with the same seed must produce identical tensors. Graph mode.
    # Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64, dtypes.int64]:
      init1 = init_ops.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
      init2 = init_ops.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    # [FUNCTION DOC]
    # Test scenario: for float32, float64, int32 and int64, random uniform initializers over zero to
    # seven with seeds one and two must produce different tensors. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64]:
      init1 = init_ops.random_uniform_initializer(0, 7, seed=1, dtype=dtype)
      init2 = init_ops.random_uniform_initializer(0, 7, seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    # [FUNCTION DOC]
    # Test scenario: one unseeded random uniform initializer over zero to one used twice under a fixed
    # graph seed must produce different values. Graph mode. Returns nothing.
    init = init_ops.random_uniform_initializer(0.0, 1.0)
    self.assertFalse(duplicated_initializer(self, init, 1))


class UniformUnitScalingInitializationTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the uniform unit scaling initializer: seed determinism with default and custom
  # factor, zero-size shapes, repeated use, and rejection of an invalid dtype.

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    # [FUNCTION DOC]
    # Test scenario: for 32- and 64-bit floats, pairs of uniform unit scaling initializers with the
    # same seed (default factor, and factor 1.5) must produce identical tensors. Graph mode.
    # Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
      init2 = init_ops.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2))
      init3 = init_ops.uniform_unit_scaling_initializer(
          1.5, seed=1, dtype=dtype)
      init4 = init_ops.uniform_unit_scaling_initializer(
          1.5, seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init3, init4))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    # [FUNCTION DOC]
    # Test scenario: for 32- and 64-bit floats, three initializers (seed one, seed two, and factor
    # 1.5 with seed one) must pairwise produce different tensors. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.uniform_unit_scaling_initializer(seed=1, dtype=dtype)
      init2 = init_ops.uniform_unit_scaling_initializer(seed=2, dtype=dtype)
      init3 = init_ops.uniform_unit_scaling_initializer(
          1.5, seed=1, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2))
      self.assertFalse(identicaltest(self, init1, init3))
      self.assertFalse(identicaltest(self, init2, init3))

  @test_util.run_deprecated_v1
  def testZeroSize(self):
    # [FUNCTION DOC]
    # Test scenario: create a variable with shape zero by two using the uniform unit scaling
    # initializer, run global variable initialization and expect the evaluated shape to be zero by
    # two (no error from the zero-size case). Graph mode. Returns nothing.
    shape = [0, 2]
    with self.cached_session():
      x = variable_scope.get_variable(
          "x",
          shape=shape,
          initializer=init_ops.uniform_unit_scaling_initializer())
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(shape, self.evaluate(x).shape)

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    # [FUNCTION DOC]
    # Test scenario: one unseeded uniform unit scaling initializer used twice under a fixed graph
    # seed must produce different values. Graph mode. Returns nothing.
    init = init_ops.uniform_unit_scaling_initializer()
    self.assertFalse(duplicated_initializer(self, init, 1))

  def testInvalidDataType(self):
    # [FUNCTION DOC]
    # Test scenario: constructing a uniform unit scaling initializer with a string dtype must raise a
    # value error. Returns nothing.
    self.assertRaises(
        ValueError,
        init_ops.uniform_unit_scaling_initializer,
        dtype=dtypes.string)


class VarianceScalingInitializationTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the variance scaling initializer with each distribution option. Samples a
  # hundred-by-hundred tensor and checks the sample mean is near zero and variance near one over the
  # fan-in (0.01), and that the expected random op is used.

  @test_util.run_deprecated_v1
  def testTruncatedNormalDistribution(self):
    # [FUNCTION DOC]
    # Test scenario: initializer with the truncated normal distribution; the truncated normal random
    # op is wrapped by a mock that passes calls through. Expects that op to have been called and the
    # sample mean to be near 0 and variance near 1/100 (tolerance 0.01). Graph mode. Returns nothing.
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops.variance_scaling_initializer(
        distribution="truncated_normal")

    with self.session(), \
      test.mock.patch.object(
          random_ops, "truncated_normal", wraps=random_ops.truncated_normal) \
          as mock_truncated_normal:
      x = init(shape).eval()
      self.assertTrue(mock_truncated_normal.called)

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)

  @test_util.run_deprecated_v1
  def testNormalDistribution(self):
    # [FUNCTION DOC]
    # Test scenario: initializer with the normal distribution; expects (via a pass-through mock) that
    # the truncated normal op is what is called, and mean near 0 and variance near 1/100 (tolerance
    # 0.01). Graph mode. Returns nothing.
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops.variance_scaling_initializer(distribution="normal")

    with self.session(), \
      test.mock.patch.object(
          random_ops, "truncated_normal", wraps=random_ops.truncated_normal) \
          as mock_truncated_normal:
      x = init(shape).eval()
      self.assertTrue(mock_truncated_normal.called)

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)

  @test_util.run_deprecated_v1
  def testUntruncatedNormalDistribution(self):
    # [FUNCTION DOC]
    # Test scenario: initializer with the untruncated normal distribution; expects the plain random
    # normal op to be called (pass-through mock) and mean near 0, variance near 1/100 (tolerance
    # 0.01). Graph mode. Returns nothing.
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops.variance_scaling_initializer(
        distribution="untruncated_normal")

    with self.session(), \
      test.mock.patch.object(
          random_ops, "random_normal", wraps=random_ops.random_normal) \
          as mock_random_normal:
      x = init(shape).eval()
      self.assertTrue(mock_random_normal.called)

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)

  @test_util.run_deprecated_v1
  def testUniformDistribution(self):
    # [FUNCTION DOC]
    # Test scenario: initializer with the uniform distribution; no mock. Expects sample mean near 0 and
    # variance near 1/100 (tolerance 0.01). Graph mode. Returns nothing.
    shape = [100, 100]
    expect_mean = 0.
    expect_var = 1. / shape[0]
    init = init_ops.variance_scaling_initializer(distribution="uniform")

    with self.session():
      x = init(shape).eval()

    self.assertNear(np.mean(x), expect_mean, err=1e-2)
    self.assertNear(np.var(x), expect_var, err=1e-2)


# TODO(vrv): move to sequence_ops_test?
class RangeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the math range op (a TODO says it belongs in a sequence ops test). Covers integer
  # and float ranges, limit-only form, empty and negative-step ranges, dtype inference and
  # explicit dtype, and an overflow-prone start value.

  def _Range(self, start, limit, delta):
    # [FUNCTION DOC]
    # HELPER: runs the range op for the range tests.
    # Inputs: start, limit, delta (numbers). Builds the range op, asserts its static shape equals the
    # length numpy reports for the same arguments, evaluates it and returns the numpy result.
    with self.cached_session():
      tf_ans = math_ops.range(start, limit, delta, name="range")
      self.assertEqual([len(np.arange(start, limit, delta))],
                       tf_ans.get_shape())
      return self.evaluate(tf_ans)

  def testBasic(self):
    # [FUNCTION DOC]
    # Test scenario: several integer ranges (step one, two, seven, one hundred, with limits not
    # reached exactly) must equal the expected numpy arrays; a range of integer arguments must have
    # 32-bit integer dtype. Returns nothing.
    self.assertTrue(
        np.array_equal(self._Range(0, 5, 1), np.array([0, 1, 2, 3, 4])))
    self.assertTrue(np.array_equal(self._Range(0, 5, 2), np.array([0, 2, 4])))
    self.assertTrue(np.array_equal(self._Range(0, 6, 2), np.array([0, 2, 4])))
    self.assertTrue(
        np.array_equal(self._Range(13, 32, 7), np.array([13, 20, 27])))
    self.assertTrue(
        np.array_equal(
            self._Range(100, 500, 100), np.array([100, 200, 300, 400])))
    self.assertEqual(math_ops.range(0, 5, 1).dtype, dtypes.int32)

  @test_util.run_deprecated_v1
  def testLimitOnly(self):
    # [FUNCTION DOC]
    # Test scenario: range called with a single argument five equals the numpy arange of five.
    # Graph mode. Returns nothing.
    with self.session():
      self.assertAllEqual(np.arange(5), math_ops.range(5))

  def testEmpty(self):
    # [FUNCTION DOC]
    # Test scenario: for start values zero and five, a range whose start equals its limit is empty.
    # Returns nothing.
    for start in 0, 5:
      self.assertTrue(np.array_equal(self._Range(start, start, 1), []))

  def testNonInteger(self):
    # [FUNCTION DOC]
    # Test scenario: float steps and limits (0.5, 2.5, 0.9, and whole-number floats) give values close
    # to the expected arrays, and float arguments give 32-bit float dtype. Returns nothing.
    self.assertTrue(
        np.allclose(self._Range(0, 2, 0.5), np.array([0, 0.5, 1, 1.5])))
    self.assertTrue(np.allclose(self._Range(0, 5, 2.5), np.array([0, 2.5])))
    self.assertTrue(
        np.allclose(self._Range(0, 3, 0.9), np.array([0, 0.9, 1.8, 2.7])))
    self.assertTrue(
        np.allclose(
            self._Range(100., 500., 100.), np.array([100, 200, 300, 400])))
    self.assertEqual(math_ops.range(0., 5., 1.).dtype, dtypes.float32)

  def testNegativeDelta(self):
    # [FUNCTION DOC]
    # Test scenario: descending ranges with negative integer and float steps give the expected arrays.
    # Returns nothing.
    self.assertTrue(
        np.array_equal(self._Range(5, -1, -1), np.array([5, 4, 3, 2, 1, 0])))
    self.assertTrue(
        np.allclose(self._Range(2.5, 0, -0.5), np.array([2.5, 2, 1.5, 1, 0.5])))
    self.assertTrue(
        np.array_equal(self._Range(-5, -10, -3), np.array([-5, -8])))

  def testDType(self):
    # [FUNCTION DOC]
    # Test scenario: checks dtype inference of range. Scalar tensors of 32/64-bit integer and
    # float types as start give that dtype; mixing types promotes (int32 with int64 gives int64,
    # int64 with float32 gives float32, float32 with float64 gives float64, float64 with int32 gives
    # float64); an explicit dtype argument overrides. Returns nothing.
    zero_int32 = math_ops.cast(0, dtypes.int32)
    zero_int64 = math_ops.cast(0, dtypes.int64)
    zero_float32 = math_ops.cast(0, dtypes.float32)
    zero_float64 = math_ops.cast(0, dtypes.float64)

    self.assertEqual(math_ops.range(zero_int32, 0, 1).dtype, dtypes.int32)
    self.assertEqual(math_ops.range(zero_int64, 0, 1).dtype, dtypes.int64)
    self.assertEqual(math_ops.range(zero_float32, 0, 1).dtype, dtypes.float32)
    self.assertEqual(math_ops.range(zero_float64, 0, 1).dtype, dtypes.float64)

    self.assertEqual(
        math_ops.range(zero_int32, zero_int64, 1).dtype, dtypes.int64)
    self.assertEqual(
        math_ops.range(zero_int64, zero_float32, 1).dtype, dtypes.float32)
    self.assertEqual(
        math_ops.range(zero_float32, zero_float64, 1).dtype, dtypes.float64)
    self.assertEqual(
        math_ops.range(zero_float64, zero_int32, 1).dtype, dtypes.float64)

    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.int32).dtype, dtypes.int32)
    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.int64).dtype, dtypes.int64)
    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.float32).dtype, dtypes.float32)
    self.assertEqual(
        math_ops.range(0, 0, 1, dtype=dtypes.float64).dtype, dtypes.float64)

  def testMixedDType(self):
    # [FUNCTION DOC]
    # Test scenario (regression for a reported issue): a 32-bit integer tensor four as the only
    # argument with explicit 64-bit integer dtype must yield zero, one, two, three. Returns nothing.
    # Test case for GitHub issue 35710
    tf_ans = math_ops.range(
        constant_op.constant(4, dtype=dtypes.int32), dtype=dtypes.int64)
    self.assertAllEqual(self.evaluate(tf_ans), np.array([0, 1, 2, 3]))

  def testLargeStarts(self):
    # [FUNCTION DOC]
    # Test scenario (regression for a reported issue): range with a huge negative start (about minus
    # 1e38) and limit one must raise a value error or an invalid-argument error rather than hang or
    # overflow. Returns nothing.
    # Test case for GitHub issue 46899.
    with self.session():
      with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
        v = math_ops.range(start=-1e+38, limit=1)
        self.evaluate(v)


# TODO(vrv): move to sequence_ops_test?
class LinSpaceTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the math linspace op with scalar endpoints (a TODO says it belongs in a sequence
  # ops test). Each test loops over CPU and, when available, forced-GPU modes by setting an instance
  # attribute used by the helper. Compares with hand-computed values within 1e-5.

  def _gpu_modes(self):
    # [FUNCTION DOC]
    # HELPER: returns the list of force-GPU flags to test with: false only, plus true when a GPU is
    # available. Takes no inputs.
    if test.is_gpu_available():
      return [False, True]
    else:
      return [False]

  def _LinSpace(self, start, stop, num):
    # [FUNCTION DOC]
    # HELPER: runs linspace in a new graph and session for the tests.
    # Inputs: start, stop, num. Uses the instance's force_gpu attribute (set by the calling test's
    # loop) to force GPU placement. Asserts the static shape equals the number of points and returns
    # the evaluated numpy result.
    with ops.Graph().as_default() as graph:
      with self.session(graph=graph, force_gpu=self.force_gpu):
        tf_ans = math_ops.linspace(start, stop, num, name="linspace")
        self.assertEqual([num], tf_ans.get_shape())
        return self.evaluate(tf_ans)

  def testPositive(self):
    # [FUNCTION DOC]
    # Test scenario: linspace between positive endpoints one and five with one to four points matches
    # the expected values (within 1e-5) in every available device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(1., 5., 1), np.array([1.]), 1e-5)
      self.assertArrayNear(self._LinSpace(1., 5., 2), np.array([1., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(1., 5., 3), np.array([1., 3., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(1., 5., 4), np.array([1., 7. / 3., 11. / 3., 5.]),
          1e-5)

  def testNegative(self):
    # [FUNCTION DOC]
    # Test scenario: linspace from minus one to minus five with one to four points matches the
    # expected values in every device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(-1., -5., 1), np.array([-1.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 2), np.array([-1., -5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 3), np.array([-1., -3., -5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 4), np.array([-1., -7. / 3., -11. / 3.,
                                                 -5.]), 1e-5)

  def testNegativeToPositive(self):
    # [FUNCTION DOC]
    # Test scenario: linspace from minus one to five with one to four points matches the expected
    # values in every device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(-1., 5., 1), np.array([-1.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 2), np.array([-1., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 3), np.array([-1., 2., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 4), np.array([-1., 1., 3., 5.]), 1e-5)

  def testPoint(self):
    # [FUNCTION DOC]
    # Test scenario: linspace with equal start and stop (five) with one to four points yields
    # repeated five in every device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(5., 5., 1), np.array([5.]), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 2), np.array([5.] * 2), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 3), np.array([5.] * 3), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 4), np.array([5.] * 4), 1e-5)

  def testEndpointsAreExact(self):
    # [FUNCTION DOC]
    # Test scenario: for cases where computing the last value by repeated float arithmetic would miss
    # the stop value (zero to one with 42 points, minus one to zero, 0.1 to 0.2 with 4 points, and a
    # 64-bit float start), the first and last outputs must be exactly the endpoints in the output
    # dtype. Runs in every device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      # Test some cases that produce last values not equal to "stop" when
      # computed via start + (num - 1) * ((stop - start) / (num - 1)), since
      # float arithmetic will introduce error through precision loss.
      self.assertAllEqual(
          self._LinSpace(0., 1., 42)[[0, -1]], np.array([0., 1.], np.float32))
      self.assertAllEqual(
          self._LinSpace(-1., 0., 42)[[0, -1]], np.array([-1., 0.], np.float32))
      self.assertAllEqual(
          self._LinSpace(.1, .2, 4)[[0, -1]], np.array([.1, .2], np.float32))
      # Check a case for float64 error too.
      self.assertAllEqual(
          self._LinSpace(np.array(0., np.float64), .1, 12)[[0, -1]],
          np.array([0., .1], np.float64))


class LinSpaceNdTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the n-dimensional linspace op: scalar inputs, numpy comparisons for array
  # endpoints with different axes, num given as a constant tensor, and inputs of unknown shape at
  # graph construction time (fed through placeholders). Uses the same force-GPU attribute loop idiom
  # as the scalar linspace tests.

  def _gpu_modes(self):
    # [FUNCTION DOC]
    # HELPER: returns the list of force-GPU flags to test with: false only, plus true when a GPU is
    # available. Duplicate of the helper of the scalar linspace test case. Takes no inputs.
    if test.is_gpu_available():
      return [False, True]
    else:
      return [False]

  def _LinSpace(self, start, stop, num, axis=0):
    # [FUNCTION DOC]
    # HELPER: runs the n-dimensional linspace in a new graph and session.
    # Inputs: start, stop, num (python value) and axis (default zero). Uses the instance's force_gpu
    # attribute. Returns the evaluated numpy result (no static shape assertion, unlike the scalar
    # version).
    with ops.Graph().as_default() as graph:
      with self.session(graph=graph, force_gpu=self.force_gpu):
        tf_ans = math_ops.linspace_nd(start, stop, num, axis=axis)
        return self.evaluate(tf_ans)

  def _LinSpaceNumConstant(self, start, stop, num, axis=0):
    # [FUNCTION DOC]
    # HELPER: like the plain helper but passes num as a constant tensor created in the graph.
    # Inputs: start, stop, num, axis (default zero). Returns the evaluated numpy result.
    with ops.Graph().as_default() as graph:
      num_constant = constant_op.constant(num)
      with self.session(graph=graph, force_gpu=self.force_gpu):
        tf_ans = math_ops.linspace_nd(start, stop, num_constant, axis=axis)
        return self.evaluate(tf_ans)

  def _LinspaceNoneShape(self, start, stop, num, graph_shape=None, axis=0):
    # [FUNCTION DOC]
    # HELPER: runs linspace with endpoints and count supplied through placeholders.
    # Inputs: start and stop values, num, graph_shape (optional static shape for the start and stop
    # placeholders, default nothing meaning unknown), axis (default zero). Builds 32-bit float
    # placeholders for start and stop and an integer placeholder for num, then runs the op in a
    # session feeding the given values. Returns the numpy result.
    with ops.Graph().as_default() as graph:
      num_tensor = array_ops.placeholder(dtypes.int32)
      start_t = array_ops.placeholder(dtypes.float32, shape=graph_shape)
      stop_t = array_ops.placeholder(dtypes.float32, shape=graph_shape)
      ans_tensor = math_ops.linspace_nd(start_t, stop_t, num_tensor, axis=axis)

      with self.session(graph=graph, force_gpu=self.force_gpu) as sess:
        feed_dict = {start_t: start, stop_t: stop, num_tensor: num}
        return sess.run(ans_tensor, feed_dict=feed_dict)

  def testPositive(self):
    # [FUNCTION DOC]
    # Test scenario: same positive-endpoint linspace values as the scalar test, through the
    # n-dimensional op, in every device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(1., 5., 1), np.array([1.]), 1e-5)
      self.assertArrayNear(self._LinSpace(1., 5., 2), np.array([1., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(1., 5., 3), np.array([1., 3., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(1., 5., 4), np.array([1., 7. / 3., 11. / 3., 5.]),
          1e-5)

  def testNegative(self):
    # [FUNCTION DOC]
    # Test scenario: same negative-endpoint linspace values as the scalar test, through the
    # n-dimensional op, in every device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(-1., -5., 1), np.array([-1.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 2), np.array([-1., -5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 3), np.array([-1., -3., -5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., -5., 4), np.array([-1., -7. / 3., -11. / 3.,
                                                 -5.]), 1e-5)

  def testNegativeToPositive(self):
    # [FUNCTION DOC]
    # Test scenario: minus one to five with one to four points through the n-dimensional op, in every
    # device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(-1., 5., 1), np.array([-1.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 2), np.array([-1., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 3), np.array([-1., 2., 5.]), 1e-5)
      self.assertArrayNear(
          self._LinSpace(-1., 5., 4), np.array([-1., 1., 3., 5.]), 1e-5)

  def testPoint(self):
    # [FUNCTION DOC]
    # Test scenario: equal start and stop (five) with one to four points through the n-dimensional op,
    # in every device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      self.assertArrayNear(self._LinSpace(5., 5., 1), np.array([5.]), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 2), np.array([5.] * 2), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 3), np.array([5.] * 3), 1e-5)
      self.assertArrayNear(self._LinSpace(5., 5., 4), np.array([5.] * 4), 1e-5)

  def testEndpointsAreExact(self):
    # [FUNCTION DOC]
    # Test scenario: first and last outputs equal the endpoints exactly for the same difficult cases as
    # the scalar test, through the n-dimensional op, in every device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      # Test some cases that produce last values not equal to "stop" when
      # computed via start + (num - 1) * ((stop - start) / (num - 1)), since
      # float arithmetic will introduce error through precision loss.
      self.assertAllEqual(
          self._LinSpace(0., 1., 42)[[0, -1]], np.array([0., 1.], np.float32))
      self.assertAllEqual(
          self._LinSpace(-1., 0., 42)[[0, -1]], np.array([-1., 0.], np.float32))
      self.assertAllEqual(
          self._LinSpace(.1, .2, 4)[[0, -1]], np.array([.1, .2], np.float32))
      # Check a case for float64 error too.
      self.assertAllEqual(
          self._LinSpace(np.array(0., np.float64), .1, 12)[[0, -1]],
          np.array([0., .1], np.float64))

  def testScalarsCompareToNumpy(self):
    # [FUNCTION DOC]
    # Test scenario: scalar endpoints zero and one with 32 points must be close to numpy's linspace
    # (1e-5) in every device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      actual = self._LinSpace(0., 1., 32)
      expected = np.linspace(0., 1., 32)
      self.assertArrayNear(expected, actual, 1e-5)

  def _baseNDArrayCompareToNumpy(self, axis):
    # [FUNCTION DOC]
    # HELPER: shared body of the array-endpoint tests.
    # Input: axis (integer). In every device mode, builds the two-element start/stop arrays and expected
    # result from the creation helper, runs linspace along the axis and asserts closeness using the
    # closeness helper. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      a, b, expected, num = self.create_nd_inputs_and_expected_output(axis)
      actual = self._LinSpace(a, b, num, axis=axis)
      self.assert_close(actual, expected)

  def assert_close(self, actual, expected):
    # [FUNCTION DOC]
    # HELPER: asserts two arrays are close for the array-endpoint tests.
    # KNOWN ISSUE: the wrong-indices value is computed by negating the single overall closeness result
    # (a scalar truth value) rather than an element-wise comparison, so the reported indices carry no
    # information about which elements differ; only the pass/fail outcome is meaningful.
    # Inputs: actual and expected arrays. Builds a failure message listing "wrong indices" and asserts
    # that all elements are close under default tolerance.
    wrong_indices = np_where(~np.allclose(actual, expected))
    mess = "Wrong float answer. Wrong indices: {}".format(wrong_indices)
    self.assertTrue(np.allclose(actual, expected), mess)

  def create_nd_inputs_and_expected_output(self, axis):
    # [FUNCTION DOC]
    # HELPER: builds test data for the array-endpoint tests.
    # Input: axis (integer). Start is the 32-bit float vector zero, one; stop is five times start;
    # number of points is five. Expected output has rows zero-filled and one to five. For axis zero the
    # expected array is transposed (five by two); for any other axis (including negative) it is
    # two by five. Returns start, stop, expected, num.
    a = np.arange(2, dtype=np.float32)
    b = a * 5
    num = 5

    res = np.array([[0., 0., 0., 0., 0.], [1., 2., 3., 4., 5.]])
    expected = res if axis != 0 else res.T
    return a, b, expected, num

  def testNDArrayCompareToNumpyDefaultAxis(self):
    # [FUNCTION DOC]
    # Test scenario: array endpoints with the default axis zero compare with the expected result via
    # the shared helper. Returns nothing.
    self._baseNDArrayCompareToNumpy(0)

  def testNDArrayAxisStrictlyPositive(self):
    # [FUNCTION DOC]
    # Test scenario: array endpoints with axis one compare with the expected result via the shared
    # helper. Returns nothing.
    self._baseNDArrayCompareToNumpy(1)

  def testNDArrayAxisStrictlyNegative(self):
    # [FUNCTION DOC]
    # Test scenario: array endpoints with axis minus one compare with the expected result via the
    # shared helper. Returns nothing.
    self._baseNDArrayCompareToNumpy(-1)

  def testNumConstant(self):
    # [FUNCTION DOC]
    # Test scenario: scalar endpoints zero and one with the point count given as a constant tensor
    # must be close to numpy's linspace of 32 points, in every device mode. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      actual = self._LinSpaceNumConstant(0., 1., 32)
      expected = np.linspace(0., 1., 32)
      self.assertArrayNear(expected, actual, 1e-5)

  def testUnknownShapeAtGraphCreationTime(self):
    # [FUNCTION DOC]
    # Test scenario: delegates to the shared unknown-shape check with the static placeholder shape
    # given as the bare number two (not a one-element tuple). Returns nothing.
    self.base_test_unknown_shape((2))

  def testNoneValuesInShapeAtGraphCreationTime(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, the argument is the parenthesized null value, which is just the
    # null value itself, not a shape with unknown entries; the test therefore duplicates the
    # next test (no static shape at all).
    # Test scenario: delegates to the shared unknown-shape check with a null graph shape.
    # Returns nothing.
    self.base_test_unknown_shape((None))

  def testNoneShapeAtGraphCreationTime(self):
    # [FUNCTION DOC]
    # Test scenario: delegates to the shared unknown-shape check with no static shape for the
    # placeholders. Returns nothing.
    self.base_test_unknown_shape(None)

  def base_test_unknown_shape(self, graph_shape):
    # [FUNCTION DOC]
    # HELPER: shared body of the unknown-shape tests.
    # Input: graph_shape (static shape for the start/stop placeholders, or nothing). In every device
    # mode, with axis one, builds the array endpoints and expected result, runs linspace through
    # placeholders using the given static shape and asserts closeness. Returns nothing.
    for self.force_gpu in self._gpu_modes():
      axis = 1
      a, b, expected, num = self.create_nd_inputs_and_expected_output(axis)
      actual = self._LinspaceNoneShape(a, b, num, graph_shape, axis)
      self.assert_close(actual, expected)


class DeviceTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking that variables created under an explicit device scope record that device on
  # both the variable and its initializer, and have no device otherwise.

  def testNoDevice(self):
    # [FUNCTION DOC]
    # Test scenario: a variable created in a new graph without a device scope has no device on the
    # variable or its initializer op. Returns nothing.
    with ops.Graph().as_default():
      var = variables.Variable([[1.0, 1.0]])
    self.assertDeviceEqual(None, var.device)
    self.assertDeviceEqual(None, var.initializer.device)

  def testDevice(self):
    # [FUNCTION DOC]
    # Test scenario: a variable created under a parameter-server job device scope reports that device
    # on both the variable and its initializer op. Returns nothing.
    with ops.Graph().as_default():
      with ops.device("/job:ps"):
        var = variables.Variable([[1.0, 1.0]])
    self.assertDeviceEqual("/job:ps", var.device)
    self.assertDeviceEqual("/job:ps", var.initializer.device)


class OrthogonalInitializerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the orthogonal initializer: seed determinism, repeated use, invalid dtype and
  # shape, gain scaling and orthogonality of results for varied shapes.

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    # [FUNCTION DOC]
    # Test scenario: for 32- and 64-bit floats, orthogonal initializers with equal seeds produce
    # identical ten-by-ten tensors. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
      init2 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (10, 10)))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    # [FUNCTION DOC]
    # Test scenario: for 32- and 64-bit floats, orthogonal initializers with seeds one and two produce
    # different ten-by-ten tensors. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
      init2 = init_ops.orthogonal_initializer(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (10, 10)))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    # [FUNCTION DOC]
    # Test scenario: one unseeded orthogonal initializer used twice under a fixed graph seed produces
    # different ten-by-ten values. Graph mode. Returns nothing.
    init = init_ops.orthogonal_initializer()
    self.assertFalse(duplicated_initializer(self, init, 1, (10, 10)))

  def testInvalidDataType(self):
    # [FUNCTION DOC]
    # Test scenario: constructing an orthogonal initializer with a string dtype raises a value error.
    # Returns nothing.
    self.assertRaises(
        ValueError, init_ops.orthogonal_initializer, dtype=dtypes.string)

  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Test scenario: calling the orthogonal initializer with a one-dimensional shape of five raises a
    # value error (it needs at least two dimensions). Returns nothing.
    init1 = init_ops.orthogonal_initializer()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[5])

  @test_util.run_deprecated_v1
  def testGain(self):
    # [FUNCTION DOC]
    # Test scenario: for 32- and 64-bit floats, with the same seed, the result with gain 3.14 divided
    # by 3.14 must equal the result with default gain, for a ten-by-ten shape. Graph mode. Returns
    # nothing.
    shape = (10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.orthogonal_initializer(seed=1, dtype=dtype)
      init2 = init_ops.orthogonal_initializer(gain=3.14, seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_deprecated_v1
  def testShapesValues(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes and shapes ten-by-ten, ten-nine-eight, hundred-five-five,
    # fifty-by-forty and forty-by-fifty, the initializer output has the requested shape and, after
    # flattening leading dimensions to a matrix, has orthonormal columns (when rows exceed columns) or
    # orthonormal rows (otherwise), within tolerance 1e-5 for float32 and 1e-12 for float64.
    # Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      for shape in [(10, 10), (10, 9, 8), (100, 5, 5), (50, 40), (40, 50)]:
        init = init_ops.orthogonal_initializer(dtype=dtype)
        tol = 1e-5 if dtype == dtypes.float32 else 1e-12
        with self.session(graph=ops.Graph(), use_gpu=True):
          # Check the shape
          t = init(shape).eval()
          self.assertAllEqual(shape, t.shape)
          # Check orthogonality by computing the inner product
          t = t.reshape((np.prod(t.shape[:-1]), t.shape[-1]))
          if t.shape[0] > t.shape[1]:
            self.assertAllClose(
                np.dot(t.T, t), np.eye(t.shape[1]), rtol=tol, atol=tol)
          else:
            self.assertAllClose(
                np.dot(t, t.T), np.eye(t.shape[0]), rtol=tol, atol=tol)


class ConvolutionDeltaOrthogonalInitializerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the convolutional delta orthogonal initializer: seed determinism, repeated use,
  # invalid dtype/shape, gain, and the statistical distribution of the determinants of the central
  # kernel slice.

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes, equal-seed delta orthogonal initializers produce identical
    # tensors of shape three, three, ten, ten. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes, seeds one and two produce different tensors of shape three,
    # three, ten, ten. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_delta_orthogonal(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    # [FUNCTION DOC]
    # Test scenario: one unseeded delta orthogonal initializer used twice under a fixed graph seed
    # produces different values. Graph mode. Returns nothing.
    init = init_ops.convolutional_delta_orthogonal()
    self.assertFalse(duplicated_initializer(self, init, 1, (3, 3, 10, 10)))

  def testInvalidDataType(self):
    # [FUNCTION DOC]
    # Test scenario: constructing the delta orthogonal initializer with a string dtype raises a value
    # error. Returns nothing.
    self.assertRaises(
        ValueError,
        init_ops.convolutional_delta_orthogonal,
        dtype=dtypes.string)

  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Test scenario: calling the initializer with shape three, three, six, five (more output than input
    # channels) raises a value error. Returns nothing.
    init1 = init_ops.convolutional_delta_orthogonal()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes and shape three, three, ten, ten, the result with gain 3.14
    # divided by 3.14 equals the default-gain result for the same seed. Graph mode. Returns nothing.
    shape = (3, 3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_delta_orthogonal(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_delta_orthogonal(
          gain=3.14, seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_deprecated_v1
  def testNonuniformity(self):
    # [FUNCTION DOC]
    # Test scenario: create seventy variables of shape three, three, ten, ten with the initializer and
    # take the determinant of the center spatial slice of each. Expects the sum of determinants to be
    # strictly between minus and plus seventy (some sign variation) and the sum of absolute values to
    # equal seventy within 1e-5 (each slice is orthogonal). Graph mode. Returns nothing.
    value = 0
    abs_value = 0
    shape = [3, 3, 10, 10]
    count = 70
    tol = 1e-5
    with self.session():
      for i in range(count):
        x = variable_scope.get_variable(
            "{}".format(i),
            shape=shape,
            initializer=init_ops.convolutional_delta_orthogonal)
        self.evaluate(x.initializer)
        y = self.evaluate(x)[1, 1, :, :]
        determinant = np.linalg.det(y)
        value += determinant
        abs_value += np.abs(determinant)

      # Check there is some variation in the signs of the determinants
      self.assertLess(value, count - tol)
      self.assertLess(-count + tol, value)
      # Check all determinants have absolute value 1
      # Compute the sum of the absolute values of 'count' determinants
      self.assertAllClose(abs_value, count, rtol=tol, atol=tol)


@test_util.run_all_without_tensor_float_32(
    "Tests convolutional_orthogonal_1d, which calls matmul")
class ConvolutionOrthogonal1dInitializerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the one-dimensional convolutional orthogonal initializer: seed determinism,
  # repeated use, invalid dtype/shape, gain and determinant statistics. The whole class runs with
  # TensorFloat-32 disabled because the initializer uses matrix multiplication.

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes, equal-seed initializers give identical tensors of shape
    # three, ten, ten. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (3, 10, 10)))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes, seeds one and two give different tensors of shape three,
    # ten, ten. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_1d(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (3, 10, 10)))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    # [FUNCTION DOC]
    # Test scenario: one unseeded initializer used twice under a fixed graph seed gives different
    # values. Graph mode. Returns nothing.
    init = init_ops.convolutional_orthogonal_1d()
    self.assertFalse(duplicated_initializer(self, init, 1, (3, 10, 10)))

  def testInvalidDataType(self):
    # [FUNCTION DOC]
    # Test scenario: constructing the initializer with a string dtype raises a value error.
    # Returns nothing.
    self.assertRaises(
        ValueError, init_ops.convolutional_orthogonal_1d, dtype=dtypes.string)

  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Test scenario: calling the initializer with shape three, six, five (more outputs than inputs)
    # raises a value error. Returns nothing.
    init1 = init_ops.convolutional_orthogonal_1d()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes and shape three, ten, ten, the gain 3.14 result divided by
    # 3.14 equals the default-gain result with the same seed. Graph mode. Returns nothing.
    shape = (3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_1d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_1d(
          gain=3.14, seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_deprecated_v1
  def testNonuniformity(self):
    # [FUNCTION DOC]
    # Test scenario: create seventy variables of shape three, ten, ten, sum each over the spatial axis
    # and take the determinant of the resulting matrix. Expects the sum of determinants to lie strictly
    # between minus and plus seventy and the sum of absolute values to equal seventy within 1e-5.
    # Graph mode. Returns nothing.
    value = 0
    abs_value = 0
    shape = [3, 10, 10]
    count = 70
    tol = 1e-5
    with self.session():
      for i in range(count):
        x = variable_scope.get_variable(
            "{}".format(i),
            shape=shape,
            initializer=init_ops.convolutional_orthogonal_1d)
        self.evaluate(x.initializer)
        y = np.sum(self.evaluate(x), axis=0)
        determinant = np.linalg.det(y)
        value += determinant
        abs_value += np.abs(determinant)

      # Check there is some variation in the signs of the determinants.
      self.assertLess(value, count - tol)
      self.assertLess(-count + tol, value)
      # Check all determinants have absolute value 1
      # Compute the sum of the absolute values of 'count' determinants
      self.assertAllClose(abs_value, count, rtol=tol, atol=tol)


class ConvolutionOrthogonal2dInitializerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the two-dimensional convolutional orthogonal initializer: seed determinism,
  # repeated use, invalid dtype/shape and gain scaling.

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes, equal-seed initializers give identical tensors of shape
    # three, three, ten, ten. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes, seeds one and two give different tensors of shape three,
    # three, ten, ten. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_2d(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    # [FUNCTION DOC]
    # Test scenario: one unseeded initializer used twice under a fixed graph seed gives different
    # values. Graph mode. Returns nothing.
    init = init_ops.convolutional_orthogonal_2d()
    self.assertFalse(duplicated_initializer(self, init, 1, (3, 3, 10, 10)))

  def testInvalidDataType(self):
    # [FUNCTION DOC]
    # Test scenario: constructing the initializer with a string dtype raises a value error.
    # Returns nothing.
    self.assertRaises(
        ValueError, init_ops.convolutional_orthogonal_2d, dtype=dtypes.string)

  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Test scenario: calling the initializer with shape three, three, six, five raises a value error.
    # Returns nothing.
    init1 = init_ops.convolutional_orthogonal_2d()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes and shape three, three, ten, ten, the gain 3.14 result
    # divided by 3.14 equals the default-gain result with the same seed. Graph mode. Returns nothing.
    shape = (3, 3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_2d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_2d(
          gain=3.14, seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)


@test_util.run_all_without_tensor_float_32(
    "Tests convolutional_orthogonal_3d, which calls matmul")
class ConvolutionOrthogonal3dInitializerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the three-dimensional convolutional orthogonal initializer: seed determinism,
  # repeated use, invalid dtype/shape, gain and determinant statistics. The whole class runs with
  # TensorFloat-32 disabled because the initializer uses matrix multiplication.

  @test_util.run_deprecated_v1
  def testInitializerIdentical(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes, equal-seed initializers give identical tensors of shape
    # three, three, three, ten, ten. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
      self.assertTrue(identicaltest(self, init1, init2, (3, 3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testInitializerDifferent(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes, seeds one and two give different tensors of shape three,
    # three, three, ten, ten. Graph mode. Returns nothing.
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_3d(seed=2, dtype=dtype)
      self.assertFalse(identicaltest(self, init1, init2, (3, 3, 3, 10, 10)))

  @test_util.run_deprecated_v1
  def testDuplicatedInitializer(self):
    # [FUNCTION DOC]
    # Test scenario: one unseeded initializer used twice under a fixed graph seed gives different
    # values. Graph mode. Returns nothing.
    init = init_ops.convolutional_orthogonal_3d()
    self.assertFalse(duplicated_initializer(self, init, 1, (3, 3, 3, 10, 10)))

  def testInvalidDataType(self):
    # [FUNCTION DOC]
    # Test scenario: constructing the initializer with a string dtype raises a value error.
    # Returns nothing.
    self.assertRaises(
        ValueError, init_ops.convolutional_orthogonal_3d, dtype=dtypes.string)

  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Test scenario: calling the initializer with shape three, three, three, six, five raises a value
    # error. Returns nothing.
    init1 = init_ops.convolutional_orthogonal_3d()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init1, shape=[3, 3, 3, 6, 5])

  @test_util.run_deprecated_v1
  def testGain(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes and shape three, three, three, ten, ten, the gain 3.14 result
    # divided by 3.14 equals the default-gain result with the same seed. Graph mode. Returns nothing.
    shape = (3, 3, 3, 10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init1 = init_ops.convolutional_orthogonal_3d(seed=1, dtype=dtype)
      init2 = init_ops.convolutional_orthogonal_3d(
          gain=3.14, seed=1, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        t1 = init1(shape).eval()
        t2 = init2(shape).eval()
      self.assertAllClose(t1, t2 / 3.14)

  @test_util.run_deprecated_v1
  def testNonuniformity(self):
    # [FUNCTION DOC]
    # Test scenario: create twenty variables of shape three, three, three, five, five, sum each over
    # the three spatial axes and take the determinant. Expects the sum of determinants strictly
    # between minus and plus twenty and the sum of absolute values equal to twenty within 1e-5.
    # Graph mode. Returns nothing.
    value = 0
    abs_value = 0
    shape = [3, 3, 3, 5, 5]
    count = 20
    tol = 1e-5
    with self.session():
      for i in range(count):
        x = variable_scope.get_variable(
            "{}".format(i),
            shape=shape,
            initializer=init_ops.convolutional_orthogonal_3d)
        self.evaluate(x.initializer)
        y = np.sum(self.evaluate(x), axis=(0, 1, 2))
        determinant = np.linalg.det(y)
        value += determinant
        abs_value += np.abs(determinant)

      # Check there is some variation in the signs of the determinants
      self.assertLess(value, count - tol)
      self.assertLess(-count + tol, value)
      # Check all determinants have absolute value 1
      # Compute the sum of the absolute values of 'count' determinants
      self.assertAllClose(abs_value, count, rtol=tol, atol=tol)


class IdentityInitializerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the identity initializer: invalid arguments, non-square shapes, gain scaling and use
  # with partitioned variables.

  def testInvalidDataType(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite being in the identity initializer test case, it constructs the orthogonal
    # initializer, not the identity initializer, so it does not test the identity initializer at all.
    # Test scenario: constructing the orthogonal initializer with a string dtype raises a value error.
    # Returns nothing.
    self.assertRaises(
        ValueError, init_ops.orthogonal_initializer, dtype=dtypes.string)

  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Test scenario: calling the identity initializer with shapes of rank three, rank one and rank zero
    # raises a value error each time (only two-dimensional shapes are allowed). Returns nothing.
    init = init_ops.identity_initializer()
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertRaises(ValueError, init, shape=[5, 7, 7])
      self.assertRaises(ValueError, init, shape=[5])
      self.assertRaises(ValueError, init, shape=[])

  @test_util.run_deprecated_v1
  def testNonSquare(self):
    # [FUNCTION DOC]
    # Test scenario: the identity initializer for a ten-by-five shape equals the ten-by-five numpy
    # identity-like matrix. Graph mode. Returns nothing.
    init = init_ops.identity_initializer()
    shape = (10, 5)
    with self.session(graph=ops.Graph(), use_gpu=True):
      self.assertAllClose(init(shape), np.eye(*shape))

  @test_util.run_deprecated_v1
  def testGain(self):
    # [FUNCTION DOC]
    # Test scenario: for both float dtypes and a ten-by-ten shape, the default initializer equals the
    # identity matrix and the initializer with gain 0.9 equals 0.9 times the identity matrix.
    # Graph mode. Returns nothing.
    shape = (10, 10)
    for dtype in [dtypes.float32, dtypes.float64]:
      init_default = init_ops.identity_initializer(dtype=dtype)
      init_custom = init_ops.identity_initializer(gain=0.9, dtype=dtype)
      with self.session(graph=ops.Graph(), use_gpu=True):
        self.assertAllClose(init_default(shape), np.eye(*shape))
      with self.session(graph=ops.Graph(), use_gpu=True):
        self.assertAllClose(init_custom(shape), np.eye(*shape) * 0.9)

  @test_util.run_deprecated_v1
  def testPartitions(self):
    # [FUNCTION DOC]
    # Test scenario: a variable created under a variable scope that has a partitioner (axis size
    # one) and the identity initializer, wrapped in an identity op, is initialized and must equal the
    # ten-by-ten identity matrix, showing the initializer works with partitioned variables. Graph
    # mode. Returns nothing.
    shape = (10, 10)
    init = init_ops.identity_initializer()
    partitioner = partitioned_variables.variable_axis_size_partitioner(1)
    with self.session(graph=ops.Graph(), use_gpu=True):
      with variable_scope.variable_scope(
          "foo", partitioner=partitioner, initializer=init):
        v = array_ops.identity(variable_scope.get_variable("bar", shape=shape))
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(v, np.eye(*shape))


if __name__ == "__main__":
  test.main()
