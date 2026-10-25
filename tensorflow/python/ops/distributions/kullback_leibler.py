# [MODULE DOC]
# Registry and dispatch for KL divergences between distribution types (legacy tf.distributions
# API). Holds a module-level mapping from (class A, class B) pairs to KL functions, the RegisterKL
# decorator that fills it, the kl_divergence function that looks up the best match by walking the
# classes' inheritance hierarchies, and cross_entropy built on entropy plus KL. Distribution
# modules register their KL functions at import time. Global state: the registry mapping.
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
"""Registration and usage mechanisms for KL-divergences."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


_DIVERGENCES = {}


__all__ = [
    "RegisterKL",
    "kl_divergence",
]


def _registered_kl(type_a, type_b):
  # [FUNCTION DOC]
  # HELPER: lookup used by kl_divergence.
  # Takes two classes, walks the method resolution order of each, and examines every pair of
  # ancestors in the registry. Picks the registered function whose summed distances to the two
  # classes is smallest; ties keep the first one found (favoring shorter distance on type_a's
  # side because it is the outer loop). Returns the function, or nothing if none is registered.
  # Reads global registry only; no side effects.
  """Get the KL function registered for classes a and b."""
  hierarchy_a = tf_inspect.getmro(type_a)
  hierarchy_b = tf_inspect.getmro(type_b)
  dist_to_children = None
  kl_fn = None
  for mro_to_a, parent_a in enumerate(hierarchy_a):
    for mro_to_b, parent_b in enumerate(hierarchy_b):
      candidate_dist = mro_to_a + mro_to_b
      candidate_kl_fn = _DIVERGENCES.get((parent_a, parent_b), None)
      if not kl_fn or (candidate_kl_fn and candidate_dist < dist_to_children):
        dist_to_children = candidate_dist
        kl_fn = candidate_kl_fn
  return kl_fn


@deprecation.deprecated(
    "2019-01-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.distributions`.",
    warn_once=True)
@tf_export(v1=["distributions.kl_divergence"])
def kl_divergence(distribution_a, distribution_b,
                  allow_nan_stats=True, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says name is prefixed to created ops, but the outer name scope is
  # fixed to KullbackLeibler; name is only forwarded to the registered KL function.
  # Computes the batchwise KL divergence between two distribution objects. Finds the registered
  # KL function for their types (see the registry lookup); raises a not-implemented error if none
  # exists. Calls it inside a name scope. If allow_nan_stats is true (default) returns the result
  # directly; otherwise adds a runtime assertion that no entry is NaN and returns the checked
  # tensor. Returns a tensor of KL values with the batch shape. Deprecated; warns once.
  """Get the KL-divergence KL(distribution_a || distribution_b).

  If there is no KL method registered specifically for `type(distribution_a)`
  and `type(distribution_b)`, then the class hierarchies of these types are
  searched.

  If one KL method is registered between any pairs of classes in these two
  parent hierarchies, it is used.

  If more than one such registered method exists, the method whose registered
  classes have the shortest sum MRO paths to the input types is used.

  If more than one such shortest path exists, the first method
  identified in the search is used (favoring a shorter MRO distance to
  `type(distribution_a)`).

  Args:
    distribution_a: The first distribution.
    distribution_b: The second distribution.
    allow_nan_stats: Python `bool`, default `True`. When `True`,
      statistics (e.g., mean, mode, variance) use the value "`NaN`" to
      indicate the result is undefined. When `False`, an exception is raised
      if one or more of the statistic's batch members are undefined.
    name: Python `str` name prefixed to Ops created by this class.

  Returns:
    A Tensor with the batchwise KL-divergence between `distribution_a`
    and `distribution_b`.

  Raises:
    NotImplementedError: If no KL method is defined for distribution types
      of `distribution_a` and `distribution_b`.
  """
  kl_fn = _registered_kl(type(distribution_a), type(distribution_b))
  if kl_fn is None:
    raise NotImplementedError(
        "No KL(distribution_a || distribution_b) registered for distribution_a "
        "type %s and distribution_b type %s"
        % (type(distribution_a).__name__, type(distribution_b).__name__))

  with ops.name_scope("KullbackLeibler"):
    kl_t = kl_fn(distribution_a, distribution_b, name=name)
    if allow_nan_stats:
      return kl_t

    # Check KL for NaNs
    kl_t = array_ops.identity(kl_t, name="kl")

    with ops.control_dependencies([
        control_flow_assert.Assert(
            math_ops.logical_not(math_ops.reduce_any(math_ops.is_nan(kl_t))), [
                "KL calculation between %s and %s returned NaN values "
                "(and was called with allow_nan_stats=False). Values:" %
                (distribution_a.name, distribution_b.name), kl_t
            ])
    ]):
      return array_ops.identity(kl_t, name="checked_kl")


@deprecation.deprecated(
    "2019-01-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.distributions`.",
    warn_once=True)
def cross_entropy(ref, other,
                  allow_nan_stats=True, name=None):
  # [FUNCTION DOC]
  # Computes the cross entropy H[ref, other] as the entropy of ref plus the KL divergence from
  # ref to other. ref and other are distribution objects; allow_nan_stats (bool, default true)
  # is passed through to the KL computation; name is the scope name. Returns a tensor with the
  # batch shape. Raises a not-implemented error if no KL is registered for the type pair.
  # Not listed in the module exports. Deprecated; warns once.
  """Computes the (Shannon) cross entropy.

  Denote two distributions by `P` (`ref`) and `Q` (`other`). Assuming `P, Q`
  are absolutely continuous with respect to one another and permit densities
  `p(x) dr(x)` and `q(x) dr(x)`, (Shanon) cross entropy is defined as:

  ```none
  H[P, Q] = E_p[-log q(X)] = -int_F p(x) log q(x) dr(x)
  ```

  where `F` denotes the support of the random variable `X ~ P`.

  Args:
    ref: `tfd.Distribution` instance.
    other: `tfd.Distribution` instance.
    allow_nan_stats: Python `bool`, default `True`. When `True`,
      statistics (e.g., mean, mode, variance) use the value "`NaN`" to
      indicate the result is undefined. When `False`, an exception is raised
      if one or more of the statistic's batch members are undefined.
    name: Python `str` prepended to names of ops created by this function.

  Returns:
    cross_entropy: `ref.dtype` `Tensor` with shape `[B1, ..., Bn]`
      representing `n` different calculations of (Shanon) cross entropy.
  """
  with ops.name_scope(name, "cross_entropy"):
    return ref.entropy() + kl_divergence(
        ref, other, allow_nan_stats=allow_nan_stats)


@tf_export(v1=["distributions.RegisterKL"])
class RegisterKL:
  # [CLASS DOC]
  # Decorator factory used to register a KL divergence function for an ordered pair of
  # distribution classes in the module-level registry. Instantiated with the two classes, then
  # called on the function to register; registration is a side effect on global state.
  """Decorator to register a KL divergence implementation function.

  Usage:

  @distributions.RegisterKL(distributions.Normal, distributions.Normal)
  def _kl_normal_mvn(norm_a, norm_b):
    # Return KL(norm_a || norm_b)
  """

  @deprecation.deprecated(
      "2019-01-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.distributions`.",
      warn_once=True)
  def __init__(self, dist_cls_a, dist_cls_b):
    # [FUNCTION DOC]
    # Constructor. Takes the class of the first and of the second distribution argument and
    # stores them as a pair key for later registration. No registration happens yet. Deprecated;
    # warns once.
    """Initialize the KL registrar.

    Args:
      dist_cls_a: the class of the first argument of the KL divergence.
      dist_cls_b: the class of the second argument of the KL divergence.
    """
    self._key = (dist_cls_a, dist_cls_b)

  def __call__(self, kl_fn):
    # [FUNCTION DOC]
    # Registers kl_fn (a callable taking two distributions and an optional name) under the stored
    # class pair in the global registry and returns kl_fn unchanged so it works as a decorator.
    # Raises a type error if kl_fn is not callable, and a value error if the pair is already
    # registered. Side effect: mutates the global registry.
    """Perform the KL registration.

    Args:
      kl_fn: The function to use for the KL divergence.

    Returns:
      kl_fn

    Raises:
      TypeError: if kl_fn is not a callable.
      ValueError: if a KL divergence function has already been registered for
        the given argument classes.
    """
    if not callable(kl_fn):
      raise TypeError("kl_fn must be callable, received: %s" % kl_fn)
    if self._key in _DIVERGENCES:
      raise ValueError("KL(%s || %s) has already been registered to: %s"
                       % (self._key[0].__name__, self._key[1].__name__,
                          _DIVERGENCES[self._key]))
    _DIVERGENCES[self._key] = kl_fn
    return kl_fn
