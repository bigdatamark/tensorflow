# [MODULE DOC]
# Unit tests for the Bijector base class of the distributions package of TensorFlow ops.
# Contains: a test of the base-class defaults and abstractness; a deliberately broken concrete
# bijector (forward and inverse are not true inverses and can be made to raise on demand) used to
# probe the forward/inverse caching of results; tests of event-dimension argument validation; a
# reusable caching test mixin; and two small bijectors with only Jacobian terms (non-constant and
# constant) used to test how log-determinant-Jacobians are reduced over event dimensions.
# The helper classes are specific to these tests. No import-time side effects other than class
# definitions; running the file as a script runs the tests.
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
"""Tests for Bijector."""

import abc

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class BaseBijectorTest(test.TestCase):
  # [CLASS DOC]
  # Test case (graph and eager modes) for properties of the abstract Bijector base class: that it
  # cannot be instantiated directly and what a minimal subclass inherits by default.
  """Tests properties of the Bijector base-class."""

  def testIsAbstract(self):
    # [FUNCTION DOC]
    # Scenario: instantiate the Bijector base class directly.
    # Expected: raises a type error saying an abstract class cannot be instantiated because of the
    # abstract constructor method; the message pattern tolerates the singular or plural wording and
    # optional quoting used by different Python versions.
    # In Python 3.9, "abstract methods" become "abstract method"
    with self.assertRaisesRegex(TypeError,
                                r"Can't instantiate abstract class Bijector "
                                r"with.* abstract method '?__init__'?"):
      bijector.Bijector()  # pylint: disable=abstract-class-instantiated

  def testDefaults(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local variables are crossed: the one named forward event shape holds the result
    # of the inverse-event-shape-tensor call and the inverse one holds the forward result; the
    # assertions still pass because the default shape transformations are identities, so the mix-up
    # is harmless but misleading.
    # Scenario: a minimal Bijector subclass defined inside the test, with only the minimum event rank.
    # Expected: graph parents are an empty list, is_constant_jacobian and validate_args are false,
    # dtype is nothing, and the name is derived from the class name as bare_bones_bijector. Forward and
    # inverse event shapes (static and tensor forms) return the input shape unchanged for scalar, rank-1
    # and rank-2 shapes. Calling forward, inverse, forward and inverse log-det-Jacobian each raises a
    # not-implemented error whose message names the missing method.
    class _BareBonesBijector(bijector.Bijector):
      # [CLASS DOC]
      # HELPER: minimal subclass of Bijector defined inside testDefaults, used only to test defaults.
      # Implements only a constructor that passes a minimum forward event rank of zero, so everything
      # else is inherited behavior.
      """Minimal specification of a `Bijector`."""

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor of the minimal test bijector: takes no arguments and calls the base-class
        # constructor with forward minimum event ndims zero, leaving every other option at its default.
        super().__init__(forward_min_event_ndims=0)

    bij = _BareBonesBijector()
    self.assertEqual([], bij.graph_parents)
    self.assertEqual(False, bij.is_constant_jacobian)
    self.assertEqual(False, bij.validate_args)
    self.assertEqual(None, bij.dtype)
    self.assertEqual("bare_bones_bijector", bij.name)

    for shape in [[], [1, 2], [1, 2, 3]]:
      forward_event_shape_ = self.evaluate(
          bij.inverse_event_shape_tensor(shape))
      inverse_event_shape_ = self.evaluate(
          bij.forward_event_shape_tensor(shape))
      self.assertAllEqual(shape, forward_event_shape_)
      self.assertAllEqual(shape, bij.forward_event_shape(shape))
      self.assertAllEqual(shape, inverse_event_shape_)
      self.assertAllEqual(shape, bij.inverse_event_shape(shape))

    with self.assertRaisesRegex(NotImplementedError, "inverse not implemented"):
      bij.inverse(0)

    with self.assertRaisesRegex(NotImplementedError, "forward not implemented"):
      bij.forward(0)

    with self.assertRaisesRegex(NotImplementedError,
                                "inverse_log_det_jacobian not implemented"):
      bij.inverse_log_det_jacobian(0, event_ndims=0)

    with self.assertRaisesRegex(NotImplementedError,
                                "forward_log_det_jacobian not implemented"):
      bij.forward_log_det_jacobian(0, event_ndims=0)


class IntentionallyMissingError(Exception):
  # [CLASS DOC]
  # HELPER: marker exception raised by BrokenBijector to simulate an unimplemented direction.
  # Used by the caching tests to prove that a value was served from the cache rather than
  # recomputed. Has no behavior of its own.
  pass


class BrokenBijector(bijector.Bijector):
  # [CLASS DOC]
  # HELPER: test bijector used by the caching and event-ndims tests.
  # KNOWN ISSUE: the docstring says forward and inverse are not inverses of each other, but the code
  # implements an exact inverse pair (multiply by two and divide by two) with consistent Jacobian
  # terms; it is only "broken" in that either direction can be made to raise on demand.
  # A Bijector with minimum forward event ndims zero and name broken whose forward is multiplication
  # by two and inverse is division by two. Two switches given at construction make the
  # forward direction or the inverse direction (including its log-det-Jacobian) raise
  # IntentionallyMissingError, so tests can show that results came from the cache. Stores the two
  # switches as private attributes.
  """Forward and inverse are not inverses of each other."""

  def __init__(
      self, forward_missing=False, inverse_missing=False, validate_args=False):
    # [FUNCTION DOC]
    # Constructor of the broken test bijector. Inputs: forward_missing (default false) and
    # inverse_missing (default false) select which direction raises the marker exception, and
    # validate_args (default false) is forwarded to the base class. Calls the base constructor with
    # minimum forward event ndims zero and the name broken, then records the two switches.
    super().__init__(
        validate_args=validate_args, forward_min_event_ndims=0, name="broken")
    self._forward_missing = forward_missing
    self._inverse_missing = inverse_missing

  def _forward(self, x):
    # [FUNCTION DOC]
    # Forward transform: raises the marker exception if the forward direction was disabled at
    # construction, otherwise returns the input multiplied by two.
    if self._forward_missing:
      raise IntentionallyMissingError
    return 2 * x

  def _inverse(self, y):
    # [FUNCTION DOC]
    # Inverse transform: raises the marker exception if the inverse direction was disabled at
    # construction, otherwise returns the input divided by two.
    if self._inverse_missing:
      raise IntentionallyMissingError
    return y / 2.

  def _inverse_log_det_jacobian(self, y):  # pylint:disable=unused-argument
    # [FUNCTION DOC]
    # The argument is ignored (kept for the interface). Raises the marker exception if the inverse
    # direction was disabled, otherwise returns the constant minus log two as a scalar tensor.
    if self._inverse_missing:
      raise IntentionallyMissingError
    return -math_ops.log(2.)

  def _forward_log_det_jacobian(self, x):  # pylint:disable=unused-argument
    # [FUNCTION DOC]
    # The argument is ignored (kept for the interface). Raises the marker exception if the forward
    # direction was disabled, otherwise returns the constant log two as a scalar tensor.
    if self._forward_missing:
      raise IntentionallyMissingError
    return math_ops.log(2.)


class BijectorTestEventNdims(test.TestCase):
  # [CLASS DOC]
  # Test case checking validation of the event_ndims argument of the log-det-Jacobian methods of
  # Bijector, using BrokenBijector. Runs in the default mode (no graph-and-eager decorator on the
  # class).

  def testBijectorNonIntegerEventNdims(self):
    # [FUNCTION DOC]
    # Scenario: event_ndims given as the non-integer 1.5.
    # Expected: both forward and inverse log-det-Jacobian raise a value error mentioning that an
    # integer was expected.
    bij = BrokenBijector()
    with self.assertRaisesRegex(ValueError, "Expected integer"):
      bij.forward_log_det_jacobian(1., event_ndims=1.5)
    with self.assertRaisesRegex(ValueError, "Expected integer"):
      bij.inverse_log_det_jacobian(1., event_ndims=1.5)

  def testBijectorArrayEventNdims(self):
    # [FUNCTION DOC]
    # Scenario: event_ndims given as a tuple of two numbers instead of a scalar.
    # Expected: both forward and inverse log-det-Jacobian raise a value error mentioning that a scalar
    # was expected.
    bij = BrokenBijector()
    with self.assertRaisesRegex(ValueError, "Expected scalar"):
      bij.forward_log_det_jacobian(1., event_ndims=(1, 2))
    with self.assertRaisesRegex(ValueError, "Expected scalar"):
      bij.inverse_log_det_jacobian(1., event_ndims=(1, 2))

  @test_util.run_deprecated_v1
  def testBijectorDynamicEventNdims(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): bijector with argument validation on; event_ndims supplied through
    # an integer placeholder of unknown shape and fed a pair of values.
    # Expected: evaluating forward and inverse log-det-Jacobian raises an op error (at run time)
    # saying a scalar was expected.
    bij = BrokenBijector(validate_args=True)
    event_ndims = array_ops.placeholder(dtype=np.int32, shape=None)
    with self.cached_session():
      with self.assertRaisesOpError("Expected scalar"):
        bij.forward_log_det_jacobian(1., event_ndims=event_ndims).eval({
            event_ndims: (1, 2)})
      with self.assertRaisesOpError("Expected scalar"):
        bij.inverse_log_det_jacobian(1., event_ndims=event_ndims).eval({
            event_ndims: (1, 2)})


class BijectorCachingTestBase(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # HELPER: mixin (an abstract base class, not a test case itself) holding caching tests shared by
  # concrete test classes. Subclasses must also derive from the test case class and provide the
  # class of the bijector under test through the broken_bijector_cls property; the bijector must
  # raise IntentionallyMissingError for directions that are switched off.

  @abc.abstractproperty
  def broken_bijector_cls(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: declared with the deprecated abstract-property decorator, and the abstract
    # implementation itself raises the marker exception instead of leaving the body empty;
    # subclasses must override it.
    # Abstract property: returns the bijector class that the caching tests instantiate. The base
    # version always raises IntentionallyMissingError with a not-implemented message.
    # return a BrokenBijector type Bijector, since this will test the caching.
    raise IntentionallyMissingError("Not implemented")

  def testCachingOfForwardResults(self):
    # [FUNCTION DOC]
    # Scenario: a bijector whose inverse is disabled is used. Forward and forward log-det-Jacobian
    # are computed separately for a constant input.
    # Expected: afterwards inverse and inverse log-det-Jacobian of the output succeed (served from the
    # cache even though the inverse is disabled), but asking for the inverse log-det-Jacobian with a
    # different event_ndims raises the marker exception because that was not cached.
    broken_bijector = self.broken_bijector_cls(inverse_missing=True)
    x = constant_op.constant(1.1)

    # Call forward and forward_log_det_jacobian one-by-one (not together).
    y = broken_bijector.forward(x)
    _ = broken_bijector.forward_log_det_jacobian(x, event_ndims=0)

    # Now, everything should be cached if the argument is y.
    broken_bijector.inverse(y)
    broken_bijector.inverse_log_det_jacobian(y, event_ndims=0)

    # Different event_ndims should not be cached.
    with self.assertRaises(IntentionallyMissingError):
      broken_bijector.inverse_log_det_jacobian(y, event_ndims=1)

  def testCachingOfInverseResults(self):
    # [FUNCTION DOC]
    # Scenario: a bijector whose forward is disabled is used. Inverse and inverse log-det-Jacobian
    # are computed separately for a constant input.
    # Expected: forward and forward log-det-Jacobian of the result succeed from the cache, while the
    # forward log-det-Jacobian with a different event_ndims raises the marker exception.
    broken_bijector = self.broken_bijector_cls(forward_missing=True)
    y = constant_op.constant(1.1)

    # Call inverse and inverse_log_det_jacobian one-by-one (not together).
    x = broken_bijector.inverse(y)
    _ = broken_bijector.inverse_log_det_jacobian(y, event_ndims=0)

    # Now, everything should be cached if the argument is x.
    broken_bijector.forward(x)
    broken_bijector.forward_log_det_jacobian(x, event_ndims=0)

    # Different event_ndims should not be cached.
    with self.assertRaises(IntentionallyMissingError):
      broken_bijector.forward_log_det_jacobian(x, event_ndims=1)


class BijectorCachingTest(BijectorCachingTestBase, test.TestCase):
  # [CLASS DOC]
  # Concrete test case that applies the shared caching tests to BrokenBijector.
  """Test caching with BrokenBijector."""

  @property
  def broken_bijector_cls(self):
    # [FUNCTION DOC]
    # Property returning the BrokenBijector class, satisfying the abstract requirement of the mixin.
    return BrokenBijector


class ExpOnlyJacobian(bijector.Bijector):
  # [CLASS DOC]
  # HELPER: test bijector that only defines log-det-Jacobian terms, with an exponential-style
  # Jacobian that is not constant. Used by BijectorReduceEventDimsTest. Has name exp, validation off,
  # and a configurable minimum forward event ndims.
  """Only used for jacobian calculations."""

  def __init__(self, forward_min_event_ndims=0):
    # [FUNCTION DOC]
    # Constructor: takes forward_min_event_ndims (default zero) and forwards it to the base class with
    # validate_args false, is_constant_jacobian false and the name exp.
    super().__init__(
        validate_args=False,
        is_constant_jacobian=False,
        forward_min_event_ndims=forward_min_event_ndims,
        name="exp")

  def _inverse_log_det_jacobian(self, y):
    # [FUNCTION DOC]
    # Returns the negative natural log of the argument, elementwise (inverse Jacobian of an
    # exponential map).
    return -math_ops.log(y)

  def _forward_log_det_jacobian(self, x):
    # [FUNCTION DOC]
    # Returns the natural log of the argument, elementwise.
    return math_ops.log(x)


class ConstantJacobian(bijector.Bijector):
  # [CLASS DOC]
  # HELPER: test bijector with a constant log-det-Jacobian (forward minus two, inverse plus two),
  # flagged as having a constant Jacobian so the base class multiplies instead of summing when
  # reducing event dimensions. Used by BijectorReduceEventDimsTest. Name is c.
  """Only used for jacobian calculations."""

  def __init__(self, forward_min_event_ndims=0):
    # [FUNCTION DOC]
    # Constructor: takes forward_min_event_ndims (default zero) and forwards it to the base class
    # with validate_args false, is_constant_jacobian true and the name c.
    super().__init__(
        validate_args=False,
        is_constant_jacobian=True,
        forward_min_event_ndims=forward_min_event_ndims,
        name="c")

  def _inverse_log_det_jacobian(self, y):
    # [FUNCTION DOC]
    # Returns the constant plus two as a scalar tensor with the dtype of the argument.
    return constant_op.constant(2., y.dtype)

  def _forward_log_det_jacobian(self, x):
    # [FUNCTION DOC]
    # Returns the constant minus two as a scalar tensor with the dtype of the argument.
    return constant_op.constant(-2., x.dtype)


class BijectorReduceEventDimsTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says it tests caching with BrokenBijector, but the class actually
  # tests how log-det-Jacobians are reduced over event dimensions using ExpOnlyJacobian and
  # ConstantJacobian; the docstring was copied from another class.
  # Test case for the reduction of log-det-Jacobian over event dimensions: summation for ordinary
  # Jacobians, multiplication by the event size for constant ones, and errors when event_ndims is
  # below the bijector minimum.
  """Test caching with BrokenBijector."""

  def testReduceEventNdimsForward(self):
    # [FUNCTION DOC]
    # Scenario: a 1x2x2 input and ExpOnlyJacobian.
    # Expected: forward log-det-Jacobian with event_ndims 0 is the elementwise log, with 1 it is the
    # sum of logs over the last axis, and with 2 it is the sum over the last two axes.
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian()
    self.assertAllClose(
        np.log(x),
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        np.sum(np.log(x), axis=-1),
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        np.sum(np.log(x), axis=(-1, -2)),
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=2)))

  def testReduceEventNdimsForwardRaiseError(self):
    # [FUNCTION DOC]
    # Scenario: a bijector requiring minimum forward event ndims of one, called with event_ndims zero.
    # Expected: forward log-det-Jacobian raises a value error saying the value must be larger than
    # the minimum.
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    with self.assertRaisesRegex(ValueError, "must be larger than"):
      bij.forward_log_det_jacobian(x, event_ndims=0)

  def testReduceEventNdimsInverse(self):
    # [FUNCTION DOC]
    # Scenario: same as the forward reduction test but for the inverse direction.
    # Expected: inverse log-det-Jacobian with event_ndims 0, 1 and 2 equals the elementwise negative
    # log, then its sum over the last axis, then over the last two axes.
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian()
    self.assertAllClose(
        -np.log(x),
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        np.sum(-np.log(x), axis=-1),
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        np.sum(-np.log(x), axis=(-1, -2)),
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=2)))

  def testReduceEventNdimsInverseRaiseError(self):
    # [FUNCTION DOC]
    # Scenario: minimum forward event ndims of one, inverse log-det-Jacobian called with event_ndims zero.
    # Expected: raises a value error saying the value must be larger than the minimum.
    x = [[[1., 2.], [3., 4.]]]
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    with self.assertRaisesRegex(ValueError, "must be larger than"):
      bij.inverse_log_det_jacobian(x, event_ndims=0)

  def testReduceEventNdimsForwardConstJacobian(self):
    # [FUNCTION DOC]
    # Scenario: ConstantJacobian on a 1x2x2 input.
    # Expected: forward log-det-Jacobian equals minus two, minus four and minus eight for event_ndims
    # 0, 1 and 2, i.e. the constant multiplied by the number of event elements.
    x = [[[1., 2.], [3., 4.]]]
    bij = ConstantJacobian()
    self.assertAllClose(
        -2.,
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        -4.,
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        -8.,
        self.evaluate(bij.forward_log_det_jacobian(x, event_ndims=2)))

  def testReduceEventNdimsInverseConstJacobian(self):
    # [FUNCTION DOC]
    # Scenario: ConstantJacobian on a 1x2x2 input.
    # Expected: inverse log-det-Jacobian equals two, four and eight for event_ndims 0, 1 and 2.
    x = [[[1., 2.], [3., 4.]]]
    bij = ConstantJacobian()
    self.assertAllClose(
        2.,
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=0)))
    self.assertAllClose(
        4.,
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=1)))
    self.assertAllClose(
        8.,
        self.evaluate(bij.inverse_log_det_jacobian(x, event_ndims=2)))

  @test_util.run_deprecated_v1
  def testHandlesNonStaticEventNdims(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only): input and event_ndims both come from placeholders with unknown
    # shape; the bijector requires minimum event ndims one.
    # Action: builds the inverse log-det-Jacobian once (to check construction works), then runs it in
    # a session with event_ndims fed as one.
    # Expected: the result equals the negative log of the input, i.e. dynamic event_ndims are handled.
    x_ = [[[1., 2.], [3., 4.]]]
    x = array_ops.placeholder_with_default(x_, shape=None)
    event_ndims = array_ops.placeholder(dtype=np.int32, shape=[])
    bij = ExpOnlyJacobian(forward_min_event_ndims=1)
    bij.inverse_log_det_jacobian(x, event_ndims=event_ndims)
    with self.cached_session() as sess:
      ildj = sess.run(bij.inverse_log_det_jacobian(x, event_ndims=event_ndims),
                      feed_dict={event_ndims: 1})
    self.assertAllClose(-np.log(x_), ildj)


if __name__ == "__main__":
  test.main()
