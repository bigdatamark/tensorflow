# [MODULE DOC]
# HELPER: helper module for the composition-style LinearOperator classes.
# Small pure functions that combine the matrix property hints (self-adjoint, square, positive
# definite, non-singular) of two operators into a hint for their composition. Each hint is true,
# false or unknown (null). No import-time side effects.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Common utilities for LinearOperator property hints."""


# Note: only use this method in the commuting case.
def combined_commuting_self_adjoint_hint(operator_a, operator_b):
  # [FUNCTION DOC]
  # HELPER: hint combination for compositions of commuting operators.
  # Returns true if both operators are self-adjoint, false if exactly one is known self-adjoint
  # and the other known not to be, and null (unknown) otherwise. Only valid when the operators
  # commute. Pure function.
  """Get combined hint for self-adjoint-ness."""

  # The property is preserved under composition when the operators commute.
  if operator_a.is_self_adjoint and operator_b.is_self_adjoint:
    return True

  # The property is not preserved when an operator with the property is composed
  # with an operator without the property.

  # pylint:disable=g-bool-id-comparison
  if ((operator_a.is_self_adjoint is True and
       operator_b.is_self_adjoint is False) or
      (operator_a.is_self_adjoint is False and
       operator_b.is_self_adjoint is True)):
    return False
  # pylint:enable=g-bool-id-comparison

  # The property is not known when operators are not known to have the property
  # or both operators don't have the property (the property for the complement
  # class is not closed under composition).
  return None


def is_square(operator_a, operator_b):
  # [FUNCTION DOC]
  # HELPER: hint combination for compositions.
  # Returns true if both operators are square; if both are known non-square, compares the static
  # range dimension of the first with the static domain dimension of the second and returns whether
  # they are equal (only when both are known; otherwise falls through); returns false if exactly
  # one is known square and the other known non-square; otherwise unknown (null). Pure function.
  # KNOWN ISSUE: in the both-non-square case the comparison uses the first operator's range
  # dimension and the second operator's domain dimension, which are the inner dimensions of
  # the product and do not determine squareness of the result (outer dimensions do).
  """Return a hint to whether the composition is square."""
  if operator_a.is_square and operator_b.is_square:
    return True
  if operator_a.is_square is False and operator_b.is_square is False:  # pylint:disable=g-bool-id-comparison
    # Let A have shape [B, M, N], B have shape [B, N, L].
    m = operator_a.range_dimension
    l = operator_b.domain_dimension
    if m is not None and l is not None:
      return m == l

  if (operator_a.is_square != operator_b.is_square) and (
      operator_a.is_square is not None and operator_b.is_square is not None):
    return False

  return None


# Note: Positive definiteness is only guaranteed to be preserved
# when the operators commute and are symmetric. Only use this method in
# commuting cases.
def combined_commuting_positive_definite_hint(operator_a, operator_b):
  # [FUNCTION DOC]
  # HELPER: hint combination for compositions of commuting operators.
  # Returns true only if both operators are known positive definite and known self-adjoint;
  # otherwise unknown (null). Never returns false. Pure function.
  """Get combined PD hint for compositions."""
  # pylint:disable=g-bool-id-comparison
  if (operator_a.is_positive_definite is True and
      operator_a.is_self_adjoint is True and
      operator_b.is_positive_definite is True and
      operator_b.is_self_adjoint is True):
    return True
  # pylint:enable=g-bool-id-comparison

  return None


def combined_non_singular_hint(operator_a, operator_b):
  # [FUNCTION DOC]
  # HELPER: hint combination for compositions.
  # KNOWN ISSUE: the docstring is truncated ("for when ."). Also the last line returns the
  # logical and of the two hints, which may yield a null/unknown value or false rather than a
  # strict boolean.
  # Returns false if either operator is known singular; otherwise the logical and of the two
  # hints (true if both true, else unknown or false-like value). Pure function.
  """Get combined hint for when ."""
  # If either operator is not-invertible the composition isn't.

  # pylint:disable=g-bool-id-comparison
  if (operator_a.is_non_singular is False or
      operator_b.is_non_singular is False):
    return False
  # pylint:enable=g-bool-id-comparison

  return operator_a.is_non_singular and operator_b.is_non_singular
