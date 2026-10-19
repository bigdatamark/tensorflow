# [MODULE DOC]
# Unit tests for the KL divergence registration and dispatch mechanism in the distributions ops
# package: registering divergence functions for pairs of distribution classes, registration error
# cases, lookup through inheritance, NaN domain errors, and the distribution convenience methods
# for KL divergence and cross entropy. Reads private module state of the KL module (the registry
# mapping and the lookup function) at import time, with lint warnings disabled for that access.
# Note that tests register new classes in the global registry.
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
"""Tests for distributions KL mechanism."""

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import normal
from tensorflow.python.platform import test

# pylint: disable=protected-access
_DIVERGENCES = kullback_leibler._DIVERGENCES
_registered_kl = kullback_leibler._registered_kl

# pylint: enable=protected-access


class KLTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the KL divergence registry and dispatch. Tests define throwaway Normal subclasses
  # and register divergence functions for them in the global registry (a persistent side effect
  # shared across the process). Includes a private helper exercising inheritance-based lookup.

  def testRegistration(self):
    # [FUNCTION DOC]
    # Defines a Normal subclass, registers a divergence function for the pair that simply returns
    # the name argument, then asserts the generic KL function dispatches to it and returns the
    # passed name.

    class MyDist(normal.Normal):
      # [CLASS DOC]
      # Throwaway empty subclass of Normal used as a registration key for this test.
      pass

    # Register KL to a lambda that spits out the name parameter
    @kullback_leibler.RegisterKL(MyDist, MyDist)
    def _kl(a, b, name=None):  # pylint: disable=unused-argument,unused-variable
      # [FUNCTION DOC]
      # HELPER: registered fake divergence function for the registration test.
      # Ignores both distributions and returns the name argument unchanged (possibly null).
      return name

    a = MyDist(loc=0.0, scale=1.0)
    self.assertEqual("OK", kullback_leibler.kl_divergence(a, a, name="OK"))

  @test_util.run_deprecated_v1
  def testDomainErrorExceptions(self):
    # [FUNCTION DOC]
    # Registers a divergence function that always yields a not-a-number value for a throwaway
    # Normal subclass. With NaN statistics disallowed, asserts that evaluating the module-level
    # KL and the distribution's own KL method raise an op error saying the KL calculation returned
    # NaN values. With NaN statistics allowed, asserts KL, self KL and cross entropy all evaluate
    # to the not-a-number list without error. Runs in graph mode only (deprecated-v1 decorator).

    class MyDistException(normal.Normal):
      # [CLASS DOC]
      # Throwaway empty subclass of Normal used as a registration key for the NaN domain test.
      pass

    # Register KL to a lambda that spits out the name parameter
    @kullback_leibler.RegisterKL(MyDistException, MyDistException)
    # pylint: disable=unused-argument,unused-variable
    def _kl(a, b, name=None):
      # [FUNCTION DOC]
      # HELPER: registered fake divergence function for the NaN domain test.
      # Ignores its inputs and returns a one-element tensor holding not-a-number.
      return array_ops.identity([float("nan")])

    # pylint: disable=unused-argument,unused-variable

    with self.cached_session():
      a = MyDistException(loc=0.0, scale=1.0, allow_nan_stats=False)
      kl = kullback_leibler.kl_divergence(a, a, allow_nan_stats=False)
      with self.assertRaisesOpError(
          "KL calculation between .* and .* returned NaN values"):
        self.evaluate(kl)
      with self.assertRaisesOpError(
          "KL calculation between .* and .* returned NaN values"):
        a.kl_divergence(a).eval()
      a = MyDistException(loc=0.0, scale=1.0, allow_nan_stats=True)
      kl_ok = kullback_leibler.kl_divergence(a, a)
      self.assertAllEqual([float("nan")], self.evaluate(kl_ok))
      self_kl_ok = a.kl_divergence(a)
      self.assertAllEqual([float("nan")], self.evaluate(self_kl_ok))
      cross_ok = a.cross_entropy(a)
      self.assertAllEqual([float("nan")], self.evaluate(cross_ok))

  def testRegistrationFailures(self):
    # [FUNCTION DOC]
    # Asserts registration error handling: registering a non-callable (a string) raises a type
    # error saying it must be callable; a first valid registration succeeds; registering the same
    # class pair again raises a value error saying it has already been registered.

    class MyDist(normal.Normal):
      # [CLASS DOC]
      # Throwaway empty subclass of Normal used as a registration key for the failure test.
      pass

    with self.assertRaisesRegex(TypeError, "must be callable"):
      kullback_leibler.RegisterKL(MyDist, MyDist)("blah")

    # First registration is OK
    kullback_leibler.RegisterKL(MyDist, MyDist)(lambda a, b: None)

    # Second registration fails
    with self.assertRaisesRegex(ValueError, "has already been registered"):
      kullback_leibler.RegisterKL(MyDist, MyDist)(lambda a, b: None)

  def testExactRegistrationsAllMatch(self):
    # [FUNCTION DOC]
    # Iterates over every entry of the global divergence registry and asserts that looking up the
    # registered function for each exact class pair returns the very function stored.
    for (k, v) in _DIVERGENCES.items():
      self.assertEqual(v, _registered_kl(*k))

  def _testIndirectRegistration(self, fn):
    # [FUNCTION DOC]
    # HELPER: shared body for the indirect-registration tests.
    # Takes a callable fn of two distributions (a KL function, KL method wrapper or cross entropy
    # function). Defines Normal subclasses Sub1, Sub2 and Sub11 (a subclass of Sub1) and registers
    # divergence functions returning marker strings for the pairs (Sub1,Sub1), (Sub1,Sub2) and
    # (Sub2,Sub1). Asserts fn resolves each pair, including subclass instances, to the most
    # specific registered function by inheritance. Side effect: adds entries to the global
    # registry.

    class Sub1(normal.Normal):
      # [CLASS DOC]
      # Normal subclass used as a registration key; overrides entropy with a stub.

      def entropy(self):
        # [FUNCTION DOC]
        # Stub entropy override; returns an empty string rather than a tensor. Only needed so the
        # cross entropy path (which uses entropy) can run in tests where the result is a marker.
        return ""

    class Sub2(normal.Normal):
      # [CLASS DOC]
      # Second Normal subclass used as a registration key; overrides entropy with a stub.

      def entropy(self):
        # [FUNCTION DOC]
        # Stub entropy override; returns an empty string rather than a tensor.
        return ""

    class Sub11(Sub1):
      # [CLASS DOC]
      # Subclass of Sub1 used to test that lookup walks up the inheritance chain; overrides
      # entropy with a stub.

      def entropy(self):
        # [FUNCTION DOC]
        # Stub entropy override; returns an empty string rather than a tensor.
        return ""

    # pylint: disable=unused-argument,unused-variable
    @kullback_leibler.RegisterKL(Sub1, Sub1)
    def _kl11(a, b, name=None):
      # [FUNCTION DOC]
      # HELPER: fake divergence registered for (Sub1, Sub1); ignores inputs and returns a marker
      # string identifying that pair.
      return "sub1-1"

    @kullback_leibler.RegisterKL(Sub1, Sub2)
    def _kl12(a, b, name=None):
      # [FUNCTION DOC]
      # HELPER: fake divergence registered for (Sub1, Sub2); ignores inputs and returns a marker
      # string identifying that pair.
      return "sub1-2"

    @kullback_leibler.RegisterKL(Sub2, Sub1)
    def _kl21(a, b, name=None):
      # [FUNCTION DOC]
      # HELPER: fake divergence registered for (Sub2, Sub1); ignores inputs and returns a marker
      # string identifying that pair.
      return "sub2-1"

    # pylint: enable=unused-argument,unused_variable

    sub1 = Sub1(loc=0.0, scale=1.0)
    sub2 = Sub2(loc=0.0, scale=1.0)
    sub11 = Sub11(loc=0.0, scale=1.0)

    self.assertEqual("sub1-1", fn(sub1, sub1))
    self.assertEqual("sub1-2", fn(sub1, sub2))
    self.assertEqual("sub2-1", fn(sub2, sub1))
    self.assertEqual("sub1-1", fn(sub11, sub11))
    self.assertEqual("sub1-1", fn(sub11, sub1))
    self.assertEqual("sub1-2", fn(sub11, sub2))
    self.assertEqual("sub1-1", fn(sub11, sub1))
    self.assertEqual("sub1-2", fn(sub11, sub2))
    self.assertEqual("sub2-1", fn(sub2, sub11))
    self.assertEqual("sub1-1", fn(sub1, sub11))

  def testIndirectRegistrationKLFun(self):
    # [FUNCTION DOC]
    # Runs the shared indirect-registration check using the module-level KL divergence function.
    self._testIndirectRegistration(kullback_leibler.kl_divergence)

  def testIndirectRegistrationKLSelf(self):
    # [FUNCTION DOC]
    # Runs the shared indirect-registration check using the distribution's own KL divergence
    # method (first distribution against the second).
    self._testIndirectRegistration(
        lambda p, q: p.kl_divergence(q))

  def testIndirectRegistrationCrossEntropy(self):
    # [FUNCTION DOC]
    # Runs the shared indirect-registration check using the distribution's own cross entropy
    # method (first distribution against the second).
    self._testIndirectRegistration(
        lambda p, q: p.cross_entropy(q))

  def testFunctionCrossEntropy(self):
    # [FUNCTION DOC]
    # Runs the shared indirect-registration check using the module-level cross entropy function.
    self._testIndirectRegistration(kullback_leibler.cross_entropy)


if __name__ == "__main__":
  test.main()
