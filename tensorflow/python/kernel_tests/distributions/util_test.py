# [MODULE DOC]
# Unit tests for the helper functions in the distributions util module (aliased as the
# distribution utilities): integer-form assertion, static value extraction, conversion between
# logits and probabilities, categorical event shape and integer-casting checks, log
# combinations, dynamic shape comparison, rotate-transpose, vector picking, static rank
# preference, appending/expanding dimensions, softmax-centered and its inverse, matrix-vector
# multiplication helpers, triangular matrix helpers, embedding checks and softplus inverse
# (including gradients). Many tests use graph-mode sessions. Optional scipy is used for
# reference values; it is imported with a helper that logs a warning on failure. Import-time
# side effects: scipy import attempt and importing the neural-network gradient module for
# registration. Runs as a test binary through the TensorFlow test main.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utility functions."""

import importlib

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import util as distribution_util
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

du = distribution_util


def try_import(name):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: optional-dependency loader used to import scipy for reference values.
  # Imports the module with the given dotted name and returns it. On an import error it logs a
  # warning (side effect) naming the module and the error and returns the null value instead of
  # raising. Other exceptions propagate.
  module = None
  try:
    module = importlib.import_module(name)
  except ImportError as e:
    tf_logging.warning("Could not import %s: %s" % (name, str(e)))
  return module


special = try_import("scipy.special")


def _logit(x):
  # [FUNCTION DOC]
  # HELPER: numpy reference implementation of the logit function for the probability tests.
  # Takes a number or array of probabilities, converts to an array and returns the log of the
  # value minus the log of one minus the value (computed stably). Values outside (0, 1) give
  # NaN or infinities with numpy warnings rather than errors.
  x = np.asarray(x)
  return np.log(x) - np.log1p(-x)


class AssertCloseTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the integer-form assertion utility. Despite its name it contains only the
  # integer-form test, not a closeness assertion test. Graph mode only.
  # KNOWN ISSUE: the class name suggests it tests a "close" assertion, but its only test
  # exercises the assertion that values are integer-valued.

  @test_util.run_deprecated_v1
  def testAssertIntegerForm(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): feeds placeholders with an all-integer vector (accepted), a
    # vector with 1.1 (rejected), one with 1.0001 (rejected, difference larger than float32
    # precision) and one with 1e-8 (rejected, not near an integer). Asserts the first passes and
    # the others raise an op error saying the tensor has non-integer components.
    # This should only be detected as an integer.
    x = array_ops.placeholder(dtypes.float32)
    y = array_ops.placeholder(dtypes.float32)
    # First component isn't less than float32.eps = 1e-7
    z = array_ops.placeholder(dtypes.float32)
    # This shouldn"t be detected as an integer.
    w = array_ops.placeholder(dtypes.float32)
    feed_dict = {x: [1., 5, 10, 15, 20], y: [1.1, 5, 10, 15, 20],
                 z: [1.0001, 5, 10, 15, 20], w: [1e-8, 5, 10, 15, 20]}
    with self.cached_session():
      with ops.control_dependencies([du.assert_integer_form(x)]):
        array_ops.identity(x).eval(feed_dict=feed_dict)

      with self.assertRaisesOpError("has non-integer components"):
        with ops.control_dependencies(
            [du.assert_integer_form(y)]):
          array_ops.identity(y).eval(feed_dict=feed_dict)

      with self.assertRaisesOpError("has non-integer components"):
        with ops.control_dependencies(
            [du.assert_integer_form(z)]):
          array_ops.identity(z).eval(feed_dict=feed_dict)

      with self.assertRaisesOpError("has non-integer components"):
        with ops.control_dependencies(
            [du.assert_integer_form(w)]):
          array_ops.identity(w).eval(feed_dict=feed_dict)


class MaybeGetStaticTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility that returns a static (graph-construction-time) value of its input
  # when available: plain numbers, numpy arrays, constants, and placeholders (unavailable).

  @test_util.run_in_graph_and_eager_modes
  def testGetStaticInt(self):
    # [FUNCTION DOC]
    # Test: a plain integer 2 returns itself, and with a float64 dtype requested returns an
    # array equal to 2.0.
    x = 2
    self.assertEqual(x, du.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.), du.maybe_get_static_value(x, dtype=np.float64))

  @test_util.run_in_graph_and_eager_modes
  def testGetStaticNumpyArray(self):
    # [FUNCTION DOC]
    # Test: a zero-dimensional int32 numpy array returns an equal value, and with a float64
    # dtype requested returns an array equal to 2.0.
    x = np.array(2, dtype=np.int32)
    self.assertEqual(x, du.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.), du.maybe_get_static_value(x, dtype=np.float64))

  @test_util.run_in_graph_and_eager_modes
  def testGetStaticConstant(self):
    # [FUNCTION DOC]
    # Test: an int32 constant tensor yields its static numpy value 2, and with a float64 dtype
    # requested yields an array equal to 2.0.
    x = constant_op.constant(2, dtype=dtypes.int32)
    self.assertEqual(np.array(2, dtype=np.int32), du.maybe_get_static_value(x))
    self.assertAllClose(
        np.array(2.), du.maybe_get_static_value(x, dtype=np.float64))

  @test_util.run_deprecated_v1
  def testGetStaticPlaceholder(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a placeholder has no static value, so the function returns
    # the null value both with and without a requested dtype.
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
    self.assertEqual(None, du.maybe_get_static_value(x))
    self.assertEqual(None, du.maybe_get_static_value(x, dtype=np.float64))


class GetLogitsAndProbsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility converting between logits and probabilities (one-dimensional
  # Bernoulli-style and multidimensional categorical-style), including argument validation and
  # class-count precision limits.

  @test_util.run_in_graph_and_eager_modes
  def testImproperArguments(self):
    # [FUNCTION DOC]
    # Test: supplying neither logits nor probabilities, or both, must raise a value error.
    with self.assertRaises(ValueError):
      du.get_logits_and_probs(logits=None, probs=None)

    with self.assertRaises(ValueError):
      du.get_logits_and_probs(logits=[0.1], probs=[0.1])

  @test_util.run_in_graph_and_eager_modes
  def testLogits(self):
    # [FUNCTION DOC]
    # Test: given logits computed from five probabilities, the function returns the same logits
    # and probabilities close to the originals (relative tolerance 1e-5), with validation on.
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)
    logits = _logit(p)

    new_logits, new_p = du.get_logits_and_probs(
        logits=logits, validate_args=True)

    self.assertAllClose(p, self.evaluate(new_p), rtol=1e-5, atol=0.)
    self.assertAllClose(logits, self.evaluate(new_logits), rtol=1e-5, atol=0.)

  @test_util.run_in_graph_and_eager_modes
  def testLogitsMultidimensional(self):
    # [FUNCTION DOC]
    # Test: with multidimensional mode and logits equal to the log of a three-class probability
    # vector, the returned logits equal the input and the probabilities equal the original
    # vector.
    p = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    logits = np.log(p)

    new_logits, new_p = du.get_logits_and_probs(
        logits=logits, multidimensional=True, validate_args=True)

    self.assertAllClose(self.evaluate(new_p), p)
    self.assertAllClose(self.evaluate(new_logits), logits)

  @test_util.run_in_graph_and_eager_modes
  def testProbability(self):
    # [FUNCTION DOC]
    # Test: given five probabilities with validation on, the returned logits equal the logit of
    # the probabilities and the returned probabilities equal the input.
    p = np.array([0.01, 0.2, 0.5, 0.7, .99], dtype=np.float32)

    new_logits, new_p = du.get_logits_and_probs(probs=p, validate_args=True)

    self.assertAllClose(_logit(p), self.evaluate(new_logits))
    self.assertAllClose(p, self.evaluate(new_p))

  @test_util.run_in_graph_and_eager_modes
  def testProbabilityMultidimensional(self):
    # [FUNCTION DOC]
    # Test: given a two-by-three matrix of class probabilities in multidimensional mode with
    # validation, the returned logits equal the log of the probabilities and the probabilities
    # are unchanged.
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)

    new_logits, new_p = du.get_logits_and_probs(
        probs=p, multidimensional=True, validate_args=True)

    self.assertAllClose(np.log(p), self.evaluate(new_logits))
    self.assertAllClose(p, self.evaluate(new_p))

  @test_util.run_in_graph_and_eager_modes
  def testProbabilityValidateArgs(self):
    # [FUNCTION DOC]
    # Test: valid probabilities pass with validation. A negative component raises an op error
    # (condition that values are non-negative failed) and a component above one raises an op
    # error (components greater than one), when validation is on; with validation off, both
    # evaluate without error.
    p = [0.01, 0.2, 0.5, 0.7, .99]
    # Component less than 0.
    p2 = [-1, 0.2, 0.5, 0.3, .2]
    # Component greater than 1.
    p3 = [2, 0.2, 0.5, 0.3, .2]

    _, prob = du.get_logits_and_probs(probs=p, validate_args=True)
    self.evaluate(prob)

    with self.assertRaisesOpError("Condition x >= 0"):
      _, prob = du.get_logits_and_probs(probs=p2, validate_args=True)
      self.evaluate(prob)

    _, prob = du.get_logits_and_probs(probs=p2, validate_args=False)
    self.evaluate(prob)

    with self.assertRaisesOpError("probs has components greater than 1"):
      _, prob = du.get_logits_and_probs(probs=p3, validate_args=True)
      self.evaluate(prob)

    _, prob = du.get_logits_and_probs(probs=p3, validate_args=False)
    self.evaluate(prob)

  @test_util.run_in_graph_and_eager_modes
  def testProbabilityValidateArgsMultidimensional(self):
    # [FUNCTION DOC]
    # Test: multidimensional probabilities. Valid rows pass (validation is not requested for
    # that call). With validation on: a negative entry raises a non-negativity error; an entry
    # above one raises an error about components exceeding one or not summing to one; rows not
    # summing to one raise a sum error. With validation off all of these evaluate without error.
    p = np.array([[0.3, 0.4, 0.3], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component less than 0. Still sums to 1.
    p2 = np.array([[-.3, 0.4, 0.9], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Component greater than 1. Does not sum to 1.
    p3 = np.array([[1.3, 0.0, 0.0], [0.1, 0.5, 0.4]], dtype=np.float32)
    # Does not sum to 1.
    p4 = np.array([[1.1, 0.3, 0.4], [0.1, 0.5, 0.4]], dtype=np.float32)

    _, prob = du.get_logits_and_probs(probs=p, multidimensional=True)
    self.evaluate(prob)

    with self.assertRaisesOpError("Condition x >= 0"):
      _, prob = du.get_logits_and_probs(
          probs=p2, multidimensional=True, validate_args=True)
      self.evaluate(prob)

    _, prob = du.get_logits_and_probs(
        probs=p2, multidimensional=True, validate_args=False)
    self.evaluate(prob)

    with self.assertRaisesOpError(
        "(probs has components greater than 1|probs does not sum to 1)"):
      _, prob = du.get_logits_and_probs(
          probs=p3, multidimensional=True, validate_args=True)
      self.evaluate(prob)

    _, prob = du.get_logits_and_probs(
        probs=p3, multidimensional=True, validate_args=False)
    self.evaluate(prob)

    with self.assertRaisesOpError("probs does not sum to 1"):
      _, prob = du.get_logits_and_probs(
          probs=p4, multidimensional=True, validate_args=True)
      self.evaluate(prob)

    _, prob = du.get_logits_and_probs(
        probs=p4, multidimensional=True, validate_args=False)
    self.evaluate(prob)

  @test_util.run_deprecated_v1
  def testProbsMultidimShape(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): probabilities in half precision with more classes (2049)
    # than the type can represent exactly: a statically shaped input raises a value error at
    # construction, and a placeholder fed at run time raises an op error that the number of
    # classes exceeds the dtype precision.
    with self.cached_session():
      with self.assertRaises(ValueError):
        p = array_ops.ones([int(2**11+1)], dtype=np.float16)
        du.get_logits_and_probs(
            probs=p, multidimensional=True, validate_args=True)

      with self.assertRaisesOpError(
          "Number of classes exceeds `dtype` precision"):
        p = array_ops.placeholder(dtype=dtypes.float16)
        _, prob = du.get_logits_and_probs(
            probs=p, multidimensional=True, validate_args=True)
        prob.eval(feed_dict={p: np.ones([int(2**11+1)])})

  @test_util.run_deprecated_v1
  def testLogitsMultidimShape(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): same as the probabilities variant but passing logits: a
    # static half-precision input with 2049 classes raises a value error, and a fed placeholder
    # raises an op error that the class count exceeds the dtype precision.
    with self.cached_session():
      with self.assertRaises(ValueError):
        l = array_ops.ones([int(2**11+1)], dtype=np.float16)
        du.get_logits_and_probs(
            logits=l, multidimensional=True, validate_args=True)

      with self.assertRaisesOpError(
          "Number of classes exceeds `dtype` precision"):
        l = array_ops.placeholder(dtype=dtypes.float16)
        logit, _ = du.get_logits_and_probs(
            logits=l, multidimensional=True, validate_args=True)
        logit.eval(feed_dict={l: np.ones([int(2**11+1)])})


class EmbedCheckCategoricalEventShapeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility that checks a categorical parameter has an acceptable number of
  # events: at least two, no more than the floating type can count exactly, and a supported
  # dtype.

  @test_util.run_deprecated_v1
  def testTooSmall(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a half-precision parameter with a single event raises a value
    # error when statically shaped; a placeholder fed a single-event value raises an op error
    # saying at least two events are required.
    with self.cached_session():
      with self.assertRaises(ValueError):
        param = array_ops.ones([1], dtype=np.float16)
        checked_param = du.embed_check_categorical_event_shape(
            param)

      with self.assertRaisesOpError(
          "must have at least 2 events"):
        param = array_ops.placeholder(dtype=dtypes.float16)
        checked_param = du.embed_check_categorical_event_shape(
            param)
        checked_param.eval(feed_dict={param: np.ones([1])})

  @test_util.run_deprecated_v1
  def testTooLarge(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a half-precision parameter with 2049 events raises a value
    # error when static; a fed placeholder raises an op error that the number of classes exceeds
    # the dtype precision.
    with self.cached_session():
      with self.assertRaises(ValueError):
        param = array_ops.ones([int(2**11+1)], dtype=dtypes.float16)
        checked_param = du.embed_check_categorical_event_shape(
            param)

      with self.assertRaisesOpError(
          "Number of classes exceeds `dtype` precision"):
        param = array_ops.placeholder(dtype=dtypes.float16)
        checked_param = du.embed_check_categorical_event_shape(
            param)
        checked_param.eval(feed_dict={param: np.ones([int(2**11+1)])})

  @test_util.disable_tfrt("b/169901260")
  @test_util.run_in_graph_and_eager_modes
  def testUnsupportedDtype(self):
    # [FUNCTION DOC]
    # Test: a quantized 16-bit integer tensor of 2049 elements passed to the check must raise a
    # type error because the dtype is unsupported. Disabled for the TFRT runtime (known bug
    # reference in the decorator).
    param = ops.convert_to_tensor(
        np.ones([2**11 + 1]).astype(dtypes.qint16.as_numpy_dtype),
        dtype=dtypes.qint16)
    with self.assertRaises(TypeError):
      du.embed_check_categorical_event_shape(param)


class EmbedCheckIntegerCastingClosedTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility that adds run-time checks that casting a tensor to a target integer
  # type is lossless: non-negativity (optional), integer valued, and within the type's range.
  # All tests use placeholders fed at run time in graph mode.

  @test_util.run_deprecated_v1
  def testCorrectlyAssertsNonnegative(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a half-precision value of -1 with a 16-bit signed target
    # raises an op error that elements must be non-negative.
    with self.cached_session():
      with self.assertRaisesOpError("Elements must be non-negative"):
        x = array_ops.placeholder(dtype=dtypes.float16)
        x_checked = du.embed_check_integer_casting_closed(
            x, target_dtype=dtypes.int16)
        x_checked.eval(feed_dict={x: np.array([1, -1], dtype=np.float16)})

  @test_util.run_deprecated_v1
  def testCorrectlyAssersIntegerForm(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name has a typo ("Assers" instead of "Asserts"); behavior is
    # unaffected.
    # Test (deprecated graph mode): a half-precision value of 1.5 with a 16-bit signed target
    # raises an op error that elements must be equivalent to the target integer type.
    with self.cached_session():
      with self.assertRaisesOpError("Elements must be int16-equivalent."):
        x = array_ops.placeholder(dtype=dtypes.float16)
        x_checked = du.embed_check_integer_casting_closed(
            x, target_dtype=dtypes.int16)
        x_checked.eval(feed_dict={x: np.array([1, 1.5], dtype=np.float16)})

  @test_util.run_deprecated_v1
  def testCorrectlyAssertsLargestPossibleInteger(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a 32-bit integer value of 32768 with a 16-bit signed target
    # raises an op error that elements cannot exceed 32767.
    with self.cached_session():
      with self.assertRaisesOpError("Elements cannot exceed 32767."):
        x = array_ops.placeholder(dtype=dtypes.int32)
        x_checked = du.embed_check_integer_casting_closed(
            x, target_dtype=dtypes.int16)
        x_checked.eval(feed_dict={x: np.array([1, 2**15], dtype=np.int32)})

  @test_util.run_deprecated_v1
  def testCorrectlyAssertsSmallestPossibleInteger(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): with an unsigned 16-bit target and the non-negativity
    # option turned off, a 32-bit integer value of -1 raises an op error that elements cannot be
    # smaller than zero (the type's minimum).
    with self.cached_session():
      with self.assertRaisesOpError("Elements cannot be smaller than 0."):
        x = array_ops.placeholder(dtype=dtypes.int32)
        x_checked = du.embed_check_integer_casting_closed(
            x, target_dtype=dtypes.uint16, assert_nonnegative=False)
        x_checked.eval(feed_dict={x: np.array([1, -1], dtype=np.int32)})


@test_util.run_all_in_graph_and_eager_modes
class LogCombinationsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the log-combinations (multinomial coefficient logarithm) utility. The class
  # decorator runs every test in both graph and eager modes.

  def testLogCombinationsBinomial(self):
    # [FUNCTION DOC]
    # Test: with totals (2, 5, 12, 15) and two-class counts that match binomial choices, asserts
    # the result has static shape four and equals the log of scipy's binomial coefficients.
    # Returns early without asserting when scipy is unavailable.
    n = [2, 5, 12, 15]
    k = [1, 2, 4, 11]

    if not special:
      return

    log_combs = np.log(special.binom(n, k))

    n = np.array(n, dtype=np.float32)
    counts = [[1., 1], [2., 3], [4., 8], [11, 4]]
    log_binom = du.log_combinations(n, counts)
    self.assertEqual([4], log_binom.get_shape())
    self.assertAllClose(log_combs, self.evaluate(log_binom))

  def testLogCombinationsShape(self):
    # [FUNCTION DOC]
    # Test: totals of shape two by two and four-class counts of shape two by two by four yield a
    # result whose static shape is two by two (the class axis is reduced).
    # Shape [2, 2]
    n = [[2, 5], [12, 15]]

    n = np.array(n, dtype=np.float32)
    # Shape [2, 2, 4]
    counts = [[[1., 1, 0, 0], [2., 2, 1, 0]], [[4., 4, 1, 3], [10, 1, 1, 4]]]
    log_binom = du.log_combinations(n, counts)
    self.assertEqual([2, 2], log_binom.get_shape())


class DynamicShapeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility that compares two values' shapes at run time. Graph mode only.

  @test_util.run_deprecated_v1
  def testSameDynamicShape(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): builds scalar, vector and matrix inputs, some constants or
    # lists and some placeholders fed at run time. Asserts the comparison is true when both have
    # the same shape (scalar-scalar, vector-vector, matrix-matrix) and false when they differ in
    # rank (scalar versus vector or matrix, vector versus matrix) or in size (three versus two
    # or one elements, two-by-two versus two-by-three).
    with self.cached_session():
      scalar = constant_op.constant(2.0)
      scalar1 = array_ops.placeholder(dtype=dtypes.float32)

      vector = [0.3, 0.4, 0.5]
      vector1 = array_ops.placeholder(dtype=dtypes.float32, shape=[None])
      vector2 = array_ops.placeholder(dtype=dtypes.float32, shape=[None])

      multidimensional = [[0.3, 0.4], [0.2, 0.6]]
      multidimensional1 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, None])
      multidimensional2 = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, None])

      # Scalar
      self.assertTrue(
          du.same_dynamic_shape(scalar, scalar1).eval({
              scalar1: 2.0
          }))

      # Vector

      self.assertTrue(
          du.same_dynamic_shape(vector, vector1).eval({
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertTrue(
          du.same_dynamic_shape(vector1, vector2).eval({
              vector1: [2.0, 3.0, 4.0],
              vector2: [2.0, 3.5, 6.0]
          }))

      # Multidimensional
      self.assertTrue(
          du.same_dynamic_shape(
              multidimensional, multidimensional1).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))
      self.assertTrue(
          du.same_dynamic_shape(
              multidimensional1, multidimensional2).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]],
                  multidimensional2: [[1.0, 3.5], [6.3, 2.3]]
              }))

      # Scalar, X
      self.assertFalse(
          du.same_dynamic_shape(scalar, vector1).eval({
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertFalse(
          du.same_dynamic_shape(scalar1, vector1).eval({
              scalar1: 2.0,
              vector1: [2.0, 3.0, 4.0]
          }))
      self.assertFalse(
          du.same_dynamic_shape(scalar, multidimensional1).eval({
              multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
          }))
      self.assertFalse(
          du.same_dynamic_shape(scalar1, multidimensional1).eval(
              {
                  scalar1: 2.0,
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))

      # Vector, X
      self.assertFalse(
          du.same_dynamic_shape(vector, vector1).eval({
              vector1: [2.0, 3.0]
          }))
      self.assertFalse(
          du.same_dynamic_shape(vector1, vector2).eval({
              vector1: [2.0, 3.0, 4.0],
              vector2: [6.0]
          }))
      self.assertFalse(
          du.same_dynamic_shape(vector, multidimensional1).eval({
              multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
          }))
      self.assertFalse(
          du.same_dynamic_shape(vector1, multidimensional1).eval(
              {
                  vector1: [2.0, 3.0, 4.0],
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]]
              }))

      # Multidimensional, X
      self.assertFalse(
          du.same_dynamic_shape(
              multidimensional, multidimensional1).eval({
                  multidimensional1: [[1.0, 3.5, 5.0], [6.3, 2.3, 7.1]]
              }))
      self.assertFalse(
          du.same_dynamic_shape(
              multidimensional1, multidimensional2).eval({
                  multidimensional1: [[2.0, 3.0], [3.0, 4.0]],
                  multidimensional2: [[1.0, 3.5, 5.0], [6.3, 2.3, 7.1]]
              }))


class RotateTransposeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the rotate-transpose utility, which cyclically shifts the order of a tensor's
  # dimensions by a given amount (static and dynamic shift). Holds a numpy reference
  # implementation as a helper.

  def _np_rotate_transpose(self, x, shift):
    # [FUNCTION DOC]
    # HELPER: numpy oracle for the tests. Inputs: an array or array-like x and an integer shift.
    # Converts x to an array and returns it transposed with its axes order cyclically rolled by
    # the shift.
    if not isinstance(x, np.ndarray):
      x = np.array(x)
    return np.transpose(x, np.roll(np.arange(len(x.shape)), shift))

  @test_util.run_in_graph_and_eager_modes
  def testRollStatic(self):
    # [FUNCTION DOC]
    # Test: passing the null value as input raises a value error (message differs between eager
    # and graph mode). For arrays of rank one to three and every shift from -5 to 4, the result
    # equals the numpy reference and its static shape equals the input shape rolled by the shift.
    if context.executing_eagerly():
      error_message = r"Attempt to convert a value \(None\)"
    else:
      error_message = "None values not supported."
    with self.assertRaisesRegex(ValueError, error_message):
      du.rotate_transpose(None, 1)
    for x in (np.ones(1), np.ones((2, 1)), np.ones((3, 2, 1))):
      for shift in np.arange(-5, 5):
        y = du.rotate_transpose(x, shift)
        self.assertAllEqual(
            self._np_rotate_transpose(x, shift), self.evaluate(y))
        self.assertAllEqual(np.roll(x.shape, shift), y.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testRollDynamic(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): input and shift are placeholders fed at run time; for float
    # arrays of rank one to three and shifts from -5 to 4 the session result equals the numpy
    # reference.
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      shift = array_ops.placeholder(dtypes.int32)
      for x_value in (np.ones(
          1, dtype=x.dtype.as_numpy_dtype()), np.ones(
              (2, 1), dtype=x.dtype.as_numpy_dtype()), np.ones(
                  (3, 2, 1), dtype=x.dtype.as_numpy_dtype())):
        for shift_value in np.arange(-5, 5):
          self.assertAllEqual(
              self._np_rotate_transpose(x_value, shift_value),
              sess.run(du.rotate_transpose(x, shift),
                       feed_dict={x: x_value,
                                  shift: shift_value}))


class PickVectorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility that chooses between two vectors based on a boolean condition tensor.
  # Graph mode only.

  @test_util.run_deprecated_v1
  def testCorrectlyPicksVector(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a true condition from a comparison yields the first vector
    # and a false one yields the second when evaluated. With constant true/false conditions the
    # result is compared directly without evaluating, showing the choice is resolved statically.
    with self.cached_session():
      x = np.arange(10, 12)
      y = np.arange(15, 18)
      self.assertAllEqual(
          x, self.evaluate(du.pick_vector(math_ops.less(0, 5), x, y)))
      self.assertAllEqual(
          y, self.evaluate(du.pick_vector(math_ops.less(5, 0), x, y)))
      self.assertAllEqual(x,
                          du.pick_vector(
                              constant_op.constant(True), x, y))  # No eval.
      self.assertAllEqual(y,
                          du.pick_vector(
                              constant_op.constant(False), x, y))  # No eval.


class PreferStaticRankTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility returning a tensor's rank, as a static numpy value when known and as a
  # tensor otherwise. Graph mode only.

  @test_util.run_deprecated_v1
  def testNonEmptyConstantTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a zeros tensor of shape two by three by four gives a numpy
    # array value equal to three.
    x = array_ops.zeros((2, 3, 4))
    rank = du.prefer_static_rank(x)
    self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(3, rank)

  @test_util.run_deprecated_v1
  def testEmptyConstantTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): an empty one-dimensional constant gives a numpy array value
    # equal to one (an empty tensor still has rank one).
    x = constant_op.constant([])
    rank = du.prefer_static_rank(x)
    self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(1, rank)

  @test_util.run_deprecated_v1
  def testScalarTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a scalar constant gives a numpy array value equal to zero.
    x = constant_op.constant(1.)
    rank = du.prefer_static_rank(x)
    self.assertIsInstance(rank, np.ndarray)
    self.assertEqual(0, rank)

  @test_util.run_deprecated_v1
  def testDynamicRankEndsUpBeingNonEmpty(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a placeholder of unknown rank fed a two-by-three array gives
    # a rank tensor that evaluates to two.
    x = array_ops.placeholder(np.float64, shape=None)
    rank = du.prefer_static_rank(x)
    with self.cached_session():
      self.assertAllEqual(2, rank.eval(feed_dict={x: np.zeros((2, 3))}))

  @test_util.run_deprecated_v1
  def testDynamicRankEndsUpBeingEmpty(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a placeholder of unknown rank fed an empty list gives a rank
    # tensor that evaluates to one.
    x = array_ops.placeholder(np.int32, shape=None)
    rank = du.prefer_static_rank(x)
    with self.cached_session():
      self.assertAllEqual(1, rank.eval(feed_dict={x: []}))

  @test_util.run_deprecated_v1
  def testDynamicRankEndsUpBeingScalar(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a placeholder of unknown rank fed a scalar gives a rank
    # tensor that evaluates to zero.
    x = array_ops.placeholder(np.int32, shape=None)
    rank = du.prefer_static_rank(x)
    with self.cached_session():
      self.assertAllEqual(0, rank.eval(feed_dict={x: 1}))


class PreferStaticShapeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility returning a tensor's shape, as a static numpy array when fully known
  # and as a tensor otherwise. Graph mode only.

  @test_util.run_deprecated_v1
  def testNonEmptyConstantTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a zeros tensor of shape two by three by four gives a numpy
    # array equal to (2, 3, 4).
    x = array_ops.zeros((2, 3, 4))
    shape = du.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual(np.array([2, 3, 4]), shape)

  @test_util.run_deprecated_v1
  def testEmptyConstantTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): an empty one-dimensional constant gives a numpy array
    # equal to (0).
    x = constant_op.constant([])
    shape = du.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual(np.array([0]), shape)

  @test_util.run_deprecated_v1
  def testScalarTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a scalar constant gives an empty numpy array as its shape.
    x = constant_op.constant(1.)
    shape = du.prefer_static_shape(x)
    self.assertIsInstance(shape, np.ndarray)
    self.assertAllEqual(np.array([]), shape)

  @test_util.run_deprecated_v1
  def testDynamicShapeEndsUpBeingNonEmpty(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a placeholder of unknown shape fed a two-by-three array
    # gives a shape tensor that evaluates to (2, 3).
    x = array_ops.placeholder(np.float64, shape=None)
    shape = du.prefer_static_shape(x)
    with self.cached_session():
      self.assertAllEqual((2, 3), shape.eval(feed_dict={x: np.zeros((2, 3))}))

  @test_util.run_deprecated_v1
  def testDynamicShapeEndsUpBeingEmpty(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a placeholder of unknown shape fed an empty list gives a
    # shape tensor that evaluates to (0).
    x = array_ops.placeholder(np.int32, shape=None)
    shape = du.prefer_static_shape(x)
    with self.cached_session():
      self.assertAllEqual(np.array([0]), shape.eval(feed_dict={x: []}))

  @test_util.run_deprecated_v1
  def testDynamicShapeEndsUpBeingScalar(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a placeholder of unknown shape fed a scalar gives a shape
    # tensor that evaluates to an empty array.
    x = array_ops.placeholder(np.int32, shape=None)
    shape = du.prefer_static_shape(x)
    with self.cached_session():
      self.assertAllEqual(np.array([]), shape.eval(feed_dict={x: 1}))


class PreferStaticValueTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility returning a tensor's value, as a static numpy array when it can be
  # determined at graph-construction time and as the tensor otherwise. Graph mode only.

  @test_util.run_deprecated_v1
  def testNonEmptyConstantTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a zeros tensor of shape two by three by four gives a numpy
    # array of zeros with that shape.
    x = array_ops.zeros((2, 3, 4))
    value = du.prefer_static_value(x)
    self.assertIsInstance(value, np.ndarray)
    self.assertAllEqual(np.zeros((2, 3, 4)), value)

  @test_util.run_deprecated_v1
  def testEmptyConstantTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): an empty constant gives an empty numpy array.
    x = constant_op.constant([])
    value = du.prefer_static_value(x)
    self.assertIsInstance(value, np.ndarray)
    self.assertAllEqual(np.array([]), value)

  @test_util.run_deprecated_v1
  def testScalarTensor(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a scalar constant gives a numpy array holding 1.0.
    x = constant_op.constant(1.)
    value = du.prefer_static_value(x)
    self.assertIsInstance(value, np.ndarray)
    self.assertAllEqual(np.array(1.), value)

  @test_util.run_deprecated_v1
  def testDynamicValueEndsUpBeingNonEmpty(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a placeholder fed a two-by-three zeros array evaluates to
    # that same array.
    x = array_ops.placeholder(np.float64, shape=None)
    value = du.prefer_static_value(x)
    with self.cached_session():
      self.assertAllEqual(np.zeros((2, 3)),
                          value.eval(feed_dict={x: np.zeros((2, 3))}))

  @test_util.run_deprecated_v1
  def testDynamicValueEndsUpBeingEmpty(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a placeholder fed an empty list evaluates to an empty array.
    x = array_ops.placeholder(np.int32, shape=None)
    value = du.prefer_static_value(x)
    with self.cached_session():
      self.assertAllEqual(np.array([]), value.eval(feed_dict={x: []}))

  @test_util.run_deprecated_v1
  def testDynamicValueEndsUpBeingScalar(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a placeholder fed the scalar one evaluates to one.
    x = array_ops.placeholder(np.int32, shape=None)
    value = du.prefer_static_value(x)
    with self.cached_session():
      self.assertAllEqual(np.array(1), value.eval(feed_dict={x: 1}))


class FillTriangularTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility that builds a lower (default) or upper triangular matrix from a
  # flattened vector of its triangle elements, including batches, unknown static shape, and
  # gradients. Holds a seeded numpy random generator and a numpy reference implementation. The
  # inverse-test subclass reuses the cases with a different run method.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent test case's setUp, so base-class per-test setup is
    # skipped.
    # Fixture: creates a numpy random generator with fixed seed 42 stored on the instance, so
    # random inputs are deterministic.
    self._rng = np.random.RandomState(42)

  def _fill_triangular(self, x, upper=False):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation of the triangular fill operation.
    # Inputs: x, array-like whose last axis has m = n(n+1)/2 elements; upper, a flag choosing an
    # upper triangular result (default lower). Solves for n and raises a value error if the
    # length is not a triangular number. Rearranges the vector elements (a reversed copy
    # combined with a tail slice) into n by n entries per batch member and zeroes the other
    # triangle. Returns the array of shape batch by n by n.
    """Numpy implementation of `fill_triangular`."""
    x = np.asarray(x)
    # Formula derived by solving for n: m = n(n+1)/2.
    m = np.int32(x.shape[-1])
    n = np.sqrt(0.25 + 2. * m) - 0.5
    if n != np.floor(n):
      raise ValueError("Invalid shape.")
    n = np.int32(n)
    # We can't do: `x[..., -(n**2-m):]` because this doesn't correctly handle
    # `m == n == 1`. Hence, we do absolute indexing.
    x_tail = x[..., (m - (n * n - m)):]
    y = np.concatenate(
        [x, x_tail[..., ::-1]] if upper else [x_tail, x[..., ::-1]],
        axis=-1)
    y = y.reshape(np.concatenate([
        np.int32(x.shape[:-1]),
        np.int32([n, n]),
    ], axis=0))
    return np.triu(y) if upper else np.tril(y)

  def _run_test(self, x_, use_deferred_shape=False, **kwargs):
    # [FUNCTION DOC]
    # HELPER: common body of the triangular-fill tests.
    # Inputs: x_, numpy array of vectors; use_deferred_shape, when true hides the static shape
    # so the shape must be inferred at run time; keyword arguments are passed on to the
    # function (such as the upper flag). Wraps x in a placeholder with default, adds a term that
    # is zero in value but makes the gradient equal the input itself (to verify each gradient
    # entry maps to the correct location), calls the triangular fill, computes its gradient and
    # runs both in a session. Asserts: static shape is unknown when deferred, else matches the
    # numpy reference; values match the reference; the gradient equals the input.
    x_ = np.asarray(x_)
    with self.cached_session() as sess:
      static_shape = None if use_deferred_shape else x_.shape
      x_pl = array_ops.placeholder_with_default(x_, shape=static_shape)
      # Add `zeros_like(x)` such that x's value and gradient are identical. We
      # do this so we can ensure each gradient value is mapped to the right
      # gradient location.  (Not doing this means the gradient wrt `x` is simple
      # `ones_like(x)`.)
      # Note:
      #   zeros_like_x_pl == zeros_like(x_pl)
      #   gradient(zeros_like_x_pl, x_pl) == x_pl - 1
      zeros_like_x_pl = (x_pl * array_ops.stop_gradient(x_pl - 1.)
                         - array_ops.stop_gradient(x_pl * (x_pl - 1.)))
      x = x_pl + zeros_like_x_pl
      actual = du.fill_triangular(x, **kwargs)
      grad_actual = gradients_impl.gradients(actual, x_pl)[0]
      [actual_, grad_actual_] = sess.run([actual, grad_actual],
                                         feed_dict={x_pl: x_})
    expected = self._fill_triangular(x_, **kwargs)
    if use_deferred_shape:
      self.assertEqual(None, actual.shape)
    else:
      self.assertAllEqual(expected.shape, actual.shape)
    self.assertAllClose(expected, actual_, rtol=1e-8, atol=1e-9)
    self.assertAllClose(x_, grad_actual_, rtol=1e-8, atol=1e-9)

  @test_util.run_deprecated_v1
  def testCorrectlyMakes1x1TriLower(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): batch of three single-element vectors yields lower
    # triangular one-by-one matrices.
    self._run_test(self._rng.randn(3, int(1*2/2)))

  @test_util.run_deprecated_v1
  def testCorrectlyMakesNoBatchTriLower(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a single 10-element vector yields a lower triangular
    # four-by-four matrix.
    self._run_test(self._rng.randn(int(4*5/2)))

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatchTriLower(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a two-by-three batch of six-element vectors yields lower
    # triangular three-by-three matrices.
    self._run_test(self._rng.randn(2, 3, int(3*4/2)))

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatchTriLowerUnknownShape(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): same as the batch lower case but with the static shape
    # hidden; asserts the static shape is unknown and values and gradient are right.
    self._run_test(self._rng.randn(2, 3, int(3*4/2)), use_deferred_shape=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatch7x7TriLowerUnknownShape(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): two-by-three batch of 28-element vectors, shape hidden,
    # yields lower triangular seven-by-seven matrices.
    self._run_test(self._rng.randn(2, 3, int(7*8/2)), use_deferred_shape=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatch7x7TriLower(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): two-by-three batch of 28-element vectors with known shape
    # yields lower triangular seven-by-seven matrices.
    self._run_test(self._rng.randn(2, 3, int(7*8/2)))

  @test_util.run_deprecated_v1
  def testCorrectlyMakes1x1TriUpper(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): batch of three single-element vectors with the upper option
    # yields one-by-one matrices.
    self._run_test(self._rng.randn(3, int(1*2/2)), upper=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesNoBatchTriUpper(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a single 10-element vector with the upper option yields an
    # upper triangular four-by-four matrix.
    self._run_test(self._rng.randn(int(4*5/2)), upper=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatchTriUpper(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): a two-by-two batch of six-element vectors with the upper
    # option yields upper triangular three-by-three matrices.
    self._run_test(self._rng.randn(2, 2, int(3*4/2)), upper=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatchTriUpperUnknownShape(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): same as the batch upper case with the static shape hidden.
    self._run_test(self._rng.randn(2, 2, int(3*4/2)),
                   use_deferred_shape=True,
                   upper=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatch7x7TriUpperUnknownShape(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): two-by-three batch of 28-element vectors, upper option and
    # hidden shape, yields upper triangular seven-by-seven matrices.
    self._run_test(self._rng.randn(2, 3, int(7*8/2)),
                   use_deferred_shape=True,
                   upper=True)

  @test_util.run_deprecated_v1
  def testCorrectlyMakesBatch7x7TriUpper(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): two-by-three batch of 28-element vectors with the upper
    # option and known shape yields upper triangular seven-by-seven matrices.
    self._run_test(self._rng.randn(2, 3, int(7*8/2)), upper=True)


class FillTriangularInverseTest(FillTriangularTest):
  # [CLASS DOC]
  # Reuses all test cases of the triangular-fill tests but overrides the run method to check the
  # inverse operation: filling a triangular matrix and then applying the inverse must recover
  # the original vector. Inherits the seeded random generator fixture.

  def _run_test(self, x_, use_deferred_shape=False, **kwargs):
    # [FUNCTION DOC]
    # HELPER: overrides the parent's common test body. Inputs: x_ array of vectors,
    # use_deferred_shape flag (hide static shape), keyword arguments forwarded (upper flag).
    # Fills a triangular matrix from the vector, applies the inverse function, runs in a session
    # and asserts the inverse's static shape is unknown when deferred (else equals the input
    # shape) and that the recovered values equal the input exactly. Gradients are not checked.
    x_ = np.asarray(x_)
    with self.cached_session() as sess:
      static_shape = None if use_deferred_shape else x_.shape
      x_pl = array_ops.placeholder_with_default(x_, shape=static_shape)
      zeros_like_x_pl = (x_pl * array_ops.stop_gradient(x_pl - 1.)
                         - array_ops.stop_gradient(x_pl * (x_pl - 1.)))
      x = x_pl + zeros_like_x_pl
      actual = du.fill_triangular(x, **kwargs)
      inverse_actual = du.fill_triangular_inverse(actual, **kwargs)

      inverse_actual_ = sess.run(
          inverse_actual,
          feed_dict={x_pl: x_})

    if use_deferred_shape:
      self.assertEqual(None, inverse_actual.shape)
    else:
      self.assertAllEqual(x_.shape, inverse_actual.shape)
    self.assertAllEqual(x_, inverse_actual_)


class ReduceWeightedLogSumExp(test.TestCase):
  # [CLASS DOC]
  # Tests for the weighted log-sum-exp reduction, which returns the log of the absolute weighted
  # sum of exponentials and optionally the sign: without weights (matches the plain log-sum-exp
  # and its gradient), with negative and zero weights, with kept dimensions, and the examples in
  # the function's documentation. Contains a numpy reference helper.

  def _reduce_weighted_logsumexp(self, logx, w, axis, keep_dims=False):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation. Inputs: logx array of log values; w array of
    # weights; axis to reduce; keep_dims flag (default false) to keep the reduced axis.
    # Subtracts the per-axis maximum for stability, computes the weighted sum of exponentials,
    # and returns a pair: the maximum plus the log of the absolute sum, and the sign of the sum.
    m = np.max(logx, axis=axis, keepdims=True)
    sum_ = np.sum(w * np.exp(logx - m), axis=axis, keepdims=keep_dims)
    sgn = np.sign(sum_)
    if not keep_dims:
      m = np.squeeze(m, axis=axis)
    return m + np.log(sgn * sum_), sgn

  @test_util.run_deprecated_v1
  def testNoWeights(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): with no weights on a three-by-three input containing a large
    # value (1000), asserts the result and its gradient exactly equal the standard log-sum-exp
    # and its gradient and that all returned signs are positive.
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    with self.cached_session() as sess:
      logx = constant_op.constant(logx_)
      expected = math_ops.reduce_logsumexp(logx, axis=-1)
      grad_expected = gradients_impl.gradients(expected, logx)[0]
      actual, actual_sgn = du.reduce_weighted_logsumexp(
          logx, axis=-1, return_sign=True)
      grad_actual = gradients_impl.gradients(actual, logx)[0]
      [actual_, actual_sgn_, grad_actual_,
       expected_, grad_expected_] = sess.run([
           actual, actual_sgn, grad_actual,
           expected, grad_expected])
    self.assertAllEqual(expected_, actual_)
    self.assertAllEqual(grad_expected_, grad_actual_)
    self.assertAllEqual([1., 1, 1], actual_sgn_)

  def testNegativeWeights(self):
    # [FUNCTION DOC]
    # Test: with weights containing negative and zero entries, asserts the result equals the
    # numpy reference exactly and the signs are (-1, -1, 1). Reduces along the last axis with
    # sign output on.
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    w_ = np.array([[1., 1, -1],
                   [1, -2, 1],
                   [1, 0, 1]])
    expected, _ = self._reduce_weighted_logsumexp(logx_, w_, axis=-1)
    with self.cached_session() as sess:
      logx = constant_op.constant(logx_)
      w = constant_op.constant(w_)
      actual, actual_sgn = du.reduce_weighted_logsumexp(
          logx, w, axis=-1, return_sign=True)
      [actual_, actual_sgn_] = self.evaluate([actual, actual_sgn])
    self.assertAllEqual(expected, actual_)
    self.assertAllEqual([-1., -1, 1], actual_sgn_)

  def testKeepDims(self):
    # [FUNCTION DOC]
    # Test: same inputs as the negative-weights test but with the reduced dimension kept;
    # asserts the result matches the reference with kept dimension and the signs have shape
    # three by one.
    logx_ = np.array([[0., -1, 1000.],
                      [0, 1, -1000.],
                      [-5, 0, 5]])
    w_ = np.array([[1., 1, -1],
                   [1, -2, 1],
                   [1, 0, 1]])
    expected, _ = self._reduce_weighted_logsumexp(
        logx_, w_, axis=-1, keep_dims=True)
    with self.cached_session() as sess:
      logx = constant_op.constant(logx_)
      w = constant_op.constant(w_)
      actual, actual_sgn = du.reduce_weighted_logsumexp(
          logx, w, axis=-1, return_sign=True, keep_dims=True)
      [actual_, actual_sgn_] = self.evaluate([actual, actual_sgn])
    self.assertAllEqual(expected, actual_)
    self.assertAllEqual([[-1.], [-1], [1]], actual_sgn_)

  def testDocString(self):
    # [FUNCTION DOC]
    # Test: reproduces the examples from the function's documentation on a two-by-three zeros
    # input with weights (-1, 1, 1) and (1, 1, 1): full reduction gives log four; reducing over
    # rows gives log of (0, 2, 2) with divide warnings suppressed; reducing over columns gives
    # log of (1, 3), also with kept dimension; reducing over both axes gives log four.
    """This test verifies the correctness of the docstring examples."""

    with self.cached_session():
      x = constant_op.constant([[0., 0, 0],
                                [0, 0, 0]])

      w = constant_op.constant([[-1., 1, 1],
                                [1, 1, 1]])

      self.assertAllClose(
          np.log(4), self.evaluate(du.reduce_weighted_logsumexp(x, w)))

      with np.errstate(divide="ignore"):
        self.assertAllClose(
            np.log([0, 2, 2]),
            self.evaluate(du.reduce_weighted_logsumexp(x, w, axis=0)))

      self.assertAllClose(
          np.log([1, 3]),
          self.evaluate(du.reduce_weighted_logsumexp(x, w, axis=1)))

      self.assertAllClose(
          np.log([[1], [3]]),
          self.evaluate(
              du.reduce_weighted_logsumexp(x, w, axis=1, keep_dims=True)))

      self.assertAllClose(
          np.log(4),
          self.evaluate(du.reduce_weighted_logsumexp(x, w, axis=[0, 1])))


class GenNewSeedTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the seed-derivation utility that produces a new seed from a seed and a salt.

  def testOnlyNoneReturnsNone(self):
    # [FUNCTION DOC]
    # Test: a seed of zero with a salt yields a non-null result, while a null seed yields the
    # null value.
    self.assertFalse(du.gen_new_seed(0, "salt") is None)
    self.assertTrue(du.gen_new_seed(None, "salt") is None)


# TODO(jvdillon): Merge this test back into:
# tensorflow/python/kernel_tests/softplus_op_test.py
# once TF core is accepting new ops.
class SoftplusTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the inverse softplus utility (and softplus gradient): round trip accuracy over
  # ranges and extreme values for several float types, finiteness and positivity, and gradients
  # of the inverse that must not be NaN. A copy of tests intended to eventually move to the core
  # softplus op tests (see the comment above the class).

  def _npSoftplus(self, np_features):
    # [FUNCTION DOC]
    # HELPER: numpy reference softplus. Takes array-like features and returns the log of the sum
    # of exponentials of zero and each feature, computed stably in the input's dtype.
    np_features = np.asarray(np_features)
    zero = np.asarray(0).astype(np_features.dtype)
    return np.logaddexp(zero, np_features)

  def _testSoftplus(self, np_features, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: shared check for the numbers test. Inputs: array-like features and a flag choosing
    # whether the session may use a GPU (default no). Computes softplus and its inverse in a
    # session, then asserts softplus matches the numpy reference within a dtype-dependent
    # relative tolerance, the inverse recovers the original features, softplus is positive and
    # finite, the inverse is finite, and static shapes agree with the reference.
    np_features = np.asarray(np_features)
    np_softplus = self._npSoftplus(np_features)
    with self.session(use_gpu=use_gpu) as sess:
      softplus = nn_ops.softplus(np_features)
      softplus_inverse = du.softplus_inverse(softplus)
      [tf_softplus, tf_softplus_inverse] = sess.run([
          softplus, softplus_inverse])
    rtol = {"float16": 0.07, "float32": 0.003, "float64": 0.002}.get(
        str(np_features.dtype), 1e-6)
    self.assertAllCloseAccordingToType(np_softplus, tf_softplus, rtol=rtol)
    # This will test that we correctly computed the inverse by verifying we
    # recovered the original input.
    self.assertAllCloseAccordingToType(
        np_features, tf_softplus_inverse,
        atol=0., rtol=rtol)
    self.assertAllEqual(
        np.ones_like(tf_softplus).astype(np.bool_), tf_softplus > 0)

    self.assertShapeEqual(np_softplus, softplus)
    self.assertShapeEqual(np_softplus, softplus_inverse)

    self.assertAllEqual(
        np.ones_like(tf_softplus).astype(np.bool_), np.isfinite(tf_softplus))
    self.assertAllEqual(
        np.ones_like(tf_softplus_inverse).astype(np.bool_),
        np.isfinite(tf_softplus_inverse))

  @test_util.run_deprecated_v1
  def testNumbers(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second list passed in the GPU run is not the same as the one in the
    # CPU run: it drops the entry for log-eps plus ten with the sign of log-eps and instead has
    # an entry computed as log-eps plus ten minus log-eps, so the two runs test different values
    # (and that list has one fewer element). Looks like an editing mistake.
    # Test (deprecated graph mode): for half, single and double precision, runs the shared check
    # on a linear range of 1000 values split into two rows (lower bound -15 for half, -50 for the
    # others, upper bound 50), once without and once with GPU, and on lists of values near
    # plus and minus the log of the machine epsilon shifted by one and ten.
    for t in [np.float16, np.float32, np.float64]:
      lower = {np.float16: -15, np.float32: -50, np.float64: -50}.get(t, -100)
      upper = {np.float16: 50, np.float32: 50, np.float64: 50}.get(t, 100)
      self._testSoftplus(
          np.array(np.linspace(lower, upper, int(1e3)).astype(t)).reshape(
              [2, -1]),
          use_gpu=False)
      self._testSoftplus(
          np.array(np.linspace(lower, upper, int(1e3)).astype(t)).reshape(
              [2, -1]),
          use_gpu=True)
      log_eps = np.log(np.finfo(t).eps)
      one = t(1)
      ten = t(10)
      self._testSoftplus(
          [
              log_eps, log_eps - one, log_eps + one, log_eps - ten,
              log_eps + ten, -log_eps, -log_eps - one, -log_eps + one,
              -log_eps - ten, -log_eps + ten
          ],
          use_gpu=False)
      self._testSoftplus(
          [
              log_eps, log_eps - one, log_eps + one, log_eps - ten,
              log_eps + ten - log_eps, -log_eps - one, -log_eps + one,
              -log_eps - ten, -log_eps + ten
          ],
          use_gpu=True)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): numerically checks the gradient of softplus on a two-by-five
    # input between -0.9 and 0.9 using the gradient checker, logs the error at verbosity level
    # two and asserts it is below 1e-4.
    with self.cached_session():
      x = constant_op.constant(
          [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9],
          shape=[2, 5],
          name="x")
      y = nn_ops.softplus(x, name="softplus")
      x_init = np.asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F")
      err = gradient_checker.compute_gradient_error(
          x, [2, 5], y, [2, 5], x_init_value=x_init)
    tf_logging.vlog(2, "softplus (float) gradient err = ", err)
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testInverseSoftplusGradientNeverNan(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): half-precision inputs spanning 1e-8 to 1e6 (which includes
    # values that become zero and infinity in half precision); the gradient of the inverse
    # softplus must contain no NaN values.
    with self.cached_session():
      # Note that this range contains both zero and inf.
      x = constant_op.constant(np.logspace(-8, 6).astype(np.float16))
      y = du.softplus_inverse(x)
      grads = self.evaluate(gradients_impl.gradients(y, x)[0])
      # Equivalent to `assertAllFalse` (if it existed).
      self.assertAllEqual(
          np.zeros_like(grads).astype(np.bool_), np.isnan(grads))

  @test_util.run_deprecated_v1
  def testInverseSoftplusGradientFinite(self):
    # [FUNCTION DOC]
    # Test (deprecated graph mode): half-precision inputs from about 1.6e-5 to 3e4, a range where
    # the reciprocal is finite; the gradient of the inverse softplus must be finite everywhere.
    with self.cached_session():
      # This range of x is all finite, and so is 1 / x.  So the
      # gradient and its approximations should be finite as well.
      x = constant_op.constant(np.logspace(-4.8, 4.5).astype(np.float16))
      y = du.softplus_inverse(x)
      grads = self.evaluate(gradients_impl.gradients(y, x)[0])
      # Equivalent to `assertAllTrue` (if it existed).
      self.assertAllEqual(
          np.ones_like(grads).astype(np.bool_), np.isfinite(grads))


@test_util.run_all_in_graph_and_eager_modes
class ArgumentsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the utility that returns a mapping of the calling function's argument names to
  # their current values: positional, keyword, mixed, overwritten and variadic cases. The class
  # decorator runs each test in both graph and eager modes.

  def testNoArguments(self):
    # [FUNCTION DOC]
    # Test: a function with no parameters that calls the utility yields an empty mapping.
    def foo():
      # [FUNCTION DOC]
      # HELPER: nested function under test. Takes no arguments; returns the utility's view of
      # its own arguments (empty).
      return du.parent_frame_arguments()

    self.assertEqual({}, foo())

  def testPositionalArguments(self):
    # [FUNCTION DOC]
    # Test: for functions called with positional arguments the mapping holds exactly the
    # parameters with their passed values; local variables defined in the body are not included.
    def foo(a, b, c, d):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: nested function under test with four positional parameters; returns the
      # utility's mapping of its arguments.
      return du.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3, "d": 4}, foo(1, 2, 3, 4))

    # Tests that it does not matter where this function is called, and
    # no other local variables are returned back.
    def bar(a, b, c):
      # [FUNCTION DOC]
      # HELPER: nested function under test with three parameters and two extra local variables;
      # returns the utility's mapping, which must contain only the parameters.
      unused_x = a * b
      unused_y = c * 3
      return du.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3}, bar(1, 2, 3))

  def testOverloadedArgumentValues(self):
    # [FUNCTION DOC]
    # Test: when the function reassigns its parameters before calling the utility, the mapping
    # contains the new (current) values rather than the originally passed ones.
    def foo(a, b, c):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: nested function under test. Overwrites its three parameters with new values
      # and returns the utility's mapping.
      a = 42
      b = 31
      c = 42
      return du.parent_frame_arguments()
    self.assertEqual({"a": 42, "b": 31, "c": 42}, foo(1, 2, 3))

  def testKeywordArguments(self):
    # [FUNCTION DOC]
    # Test: a function taking only variable keyword arguments yields a mapping of the passed
    # keywords to their values.
    def foo(**kwargs):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: nested function under test accepting arbitrary keyword arguments; returns the
      # utility's mapping.
      return du.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3, "d": 4}, foo(a=1, b=2, c=3, d=4))

  def testPositionalKeywordArgs(self):
    # [FUNCTION DOC]
    # Test: a function with three named parameters plus variable keywords yields the named
    # parameters, and any extra keyword (an unexpected name with a null value) is merged into
    # the mapping too.
    def foo(a, b, c, **kwargs):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: nested function under test with named parameters and variable keywords; returns
      # the utility's mapping.
      return du.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3}, foo(a=1, b=2, c=3))
    self.assertEqual({"a": 1, "b": 2, "c": 3, "unicorn": None},
                     foo(a=1, b=2, c=3, unicorn=None))

  def testNoVarargs(self):
    # [FUNCTION DOC]
    # Test: for a function with named parameters, extra positional arguments and variable
    # keywords, the mapping contains the named parameters and keywords but not the extra
    # positional arguments.
    def foo(a, b, c, *varargs, **kwargs):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: nested function under test with named parameters, variable positional arguments
      # and variable keywords; returns the utility's mapping.
      return du.parent_frame_arguments()

    self.assertEqual({"a": 1, "b": 2, "c": 3}, foo(a=1, b=2, c=3))
    self.assertEqual({"a": 1, "b": 2, "c": 3}, foo(1, 2, 3, *[1, 2, 3]))
    self.assertEqual({"a": 1, "b": 2, "c": 3, "unicorn": None},
                     foo(1, 2, 3, unicorn=None))
    self.assertEqual({"a": 1, "b": 2, "c": 3, "unicorn": None},
                     foo(1, 2, 3, *[1, 2, 3], unicorn=None))


if __name__ == "__main__":
  test.main()
