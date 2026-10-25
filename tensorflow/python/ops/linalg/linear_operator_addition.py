# [MODULE DOC]
# Implements efficient addition of linear operators. The public-style entry point is
# add_operators, which takes several LinearOperator objects and returns a possibly shorter list
# of operators whose sum acts the same way, by merging pairs using "adder" strategies organized
# in tiers (cheapest result type first: scaled identity, diagonal, lower triangular, and finally
# a dense full matrix as the catch-all).
# Also defines internal helpers: a hints container, an abstract adder base class, four concrete
# adders, a function classifying operators by type, static dimension/batch-shape checks, type
# name constants and the default tier list.
# The module has an empty export list, so it is not re-exported wholesale; users import the
# module directly. Import-time side effects: creates one instance of each concrete adder for the
# default tier list; no registrations or flags.
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
"""Add one or more `LinearOperators` efficiently."""

import abc

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_diag
from tensorflow.python.ops.linalg import linear_operator_full_matrix
from tensorflow.python.ops.linalg import linear_operator_identity
from tensorflow.python.ops.linalg import linear_operator_lower_triangular

__all__ = []


def add_operators(operators,
                  operator_name=None,
                  addition_tiers=None,
                  name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring lists only value errors, but a type error is also raised when any
  # element of operators is not a LinearOperator. The empty-input error message also prints the
  # already reversed (and empty) list.
  # Adds linear operators efficiently. Inputs: operators (an iterable of LinearOperator objects
  # sharing dtype, domain and range dimensions, with broadcastable batch shapes), operator_name
  # (optional string name for each created sum operator; default builds a name from the operand
  # names), addition_tiers (optional list of tiers, each a list of adder objects; default is the
  # module default tiers) and name (optional name scope, default "add_operators").
  # Validates that the input is a proper iterable, non-empty, all LinearOperators, with static
  # dimensions and batch shapes compatible. Then, for each tier in order, repeatedly takes an
  # operator, looks for a partner it can be added to within the tier, and if found replaces the
  # pair by their sum (which is retried in the same tier); unmatched operators move on to the
  # next tier. Returns the list of remaining operators (their sum equals the original sum);
  # the order follows the processing order and may change. Raises a value error for empty input
  # or incompatible static shapes and a type error for non-operators. Creates ops inside a name
  # scope; does not mutate the caller's list (it works on a reversed copy).
  """Efficiently add one or more linear operators.

  Given operators `[A1, A2,...]`, this `Op` returns a possibly shorter list of
  operators `[B1, B2,...]` such that

  ```sum_k Ak.matmul(x) = sum_k Bk.matmul(x).```

  The operators `Bk` result by adding some of the `Ak`, as allowed by
  `addition_tiers`.

  Example of efficient adding of diagonal operators.

  ```python
  A1 = LinearOperatorDiag(diag=[1., 1.], name="A1")
  A2 = LinearOperatorDiag(diag=[2., 2.], name="A2")

  # Use two tiers, the first contains an Adder that returns Diag.  Since both
  # A1 and A2 are Diag, they can use this Adder.  The second tier will not be
  # used.
  addition_tiers = [
      [_AddAndReturnDiag()],
      [_AddAndReturnMatrix()]]
  B_list = add_operators([A1, A2], addition_tiers=addition_tiers)

  len(B_list)
  ==> 1

  B_list[0].__class__.__name__
  ==> 'LinearOperatorDiag'

  B_list[0].to_dense()
  ==> [[3., 0.],
       [0., 3.]]

  B_list[0].name
  ==> 'Add/A1__A2/'
  ```

  Args:
    operators:  Iterable of `LinearOperator` objects with same `dtype`, domain
      and range dimensions, and broadcastable batch shapes.
    operator_name:  String name for returned `LinearOperator`.  Defaults to
      concatenation of "Add/A__B/" that indicates the order of addition steps.
    addition_tiers:  List tiers, like `[tier_0, tier_1, ...]`, where `tier_i`
      is a list of `Adder` objects.  This function attempts to do all additions
      in tier `i` before trying tier `i + 1`.
    name:  A name for this `Op`.  Defaults to `add_operators`.

  Returns:
    Subclass of `LinearOperator`.  Class and order of addition may change as new
      (and better) addition strategies emerge.

  Raises:
    ValueError:  If `operators` argument is empty.
    ValueError:  If shapes are incompatible.
  """
  # Default setting
  if addition_tiers is None:
    addition_tiers = _DEFAULT_ADDITION_TIERS

  # Argument checking.
  check_ops.assert_proper_iterable(operators)
  operators = list(reversed(operators))
  if len(operators) < 1:
    raise ValueError(
        f"Argument `operators` must contain at least one operator. "
        f"Received: {operators}.")
  if not all(
      isinstance(op, linear_operator.LinearOperator) for op in operators):
    raise TypeError(
        f"Argument `operators` must contain only LinearOperator instances. "
        f"Received: {operators}.")
  _static_check_for_same_dimensions(operators)
  _static_check_for_broadcastable_batch_shape(operators)

  with ops.name_scope(name or "add_operators"):

    # Additions done in one of the tiers.  Try tier 0, 1,...
    ops_to_try_at_next_tier = list(operators)
    for tier in addition_tiers:
      ops_to_try_at_this_tier = ops_to_try_at_next_tier
      ops_to_try_at_next_tier = []
      while ops_to_try_at_this_tier:
        op1 = ops_to_try_at_this_tier.pop()
        op2, adder = _pop_a_match_at_tier(op1, ops_to_try_at_this_tier, tier)
        if op2 is not None:
          # Will try to add the result of this again at this same tier.
          new_operator = adder.add(op1, op2, operator_name)
          ops_to_try_at_this_tier.append(new_operator)
        else:
          ops_to_try_at_next_tier.append(op1)

    return ops_to_try_at_next_tier


def _pop_a_match_at_tier(op1, operator_list, tier):
  # [FUNCTION DOC]
  # HELPER: partner search step used by add_operators.
  # Searches operator_list from its back toward its front for an operator that some adder in the
  # given tier (a list of adders) can add to op1. On the first match it removes that operator
  # from operator_list (mutating it) and returns the pair of the removed operator and the adder.
  # If nothing matches it returns a pair of null values and leaves the list unchanged.
  # Search from the back of list to the front in order to create nice default
  # order of operations.
  for i in range(1, len(operator_list) + 1):
    op2 = operator_list[-i]
    for adder in tier:
      if adder.can_add(op1, op2):
        return operator_list.pop(-i), adder
  return None, None


def _infer_hints_allowing_override(op1, op2, hints):
  # [FUNCTION DOC]
  # HELPER: computes matrix-property hints for the sum of two operators in Adder.add.
  # Inputs: op1 and op2 (LinearOperators) and hints (a hints object or null). Explicit hints
  # win; missing ones are inferred: the sum is self-adjoint / positive definite only if both
  # operands are (an unknown operand hint yields a falsy or unknown result), and a positive
  # definite result inferred this way implies non-singular. Otherwise non-singular is copied
  # from hints. Returns a new hints object; no side effects.
  """Infer hints from op1 and op2.  hints argument is an override.

  Args:
    op1:  LinearOperator
    op2:  LinearOperator
    hints:  _Hints object holding "is_X" boolean hints to use for returned
      operator.
      If some hint is None, try to set using op1 and op2.  If the
      hint is provided, ignore op1 and op2 hints.  This allows an override
      of previous hints, but does not allow forbidden hints (e.g. you still
      cannot say a real diagonal operator is not self-adjoint.

  Returns:
    _Hints object.
  """
  hints = hints or _Hints()
  # If A, B are self-adjoint, then so is A + B.
  if hints.is_self_adjoint is None:
    is_self_adjoint = op1.is_self_adjoint and op2.is_self_adjoint
  else:
    is_self_adjoint = hints.is_self_adjoint

  # If A, B are positive definite, then so is A + B.
  if hints.is_positive_definite is None:
    is_positive_definite = op1.is_positive_definite and op2.is_positive_definite
  else:
    is_positive_definite = hints.is_positive_definite

  # A positive definite operator is always non-singular.
  if is_positive_definite and hints.is_positive_definite is None:
    is_non_singular = True
  else:
    is_non_singular = hints.is_non_singular

  return _Hints(
      is_non_singular=is_non_singular,
      is_self_adjoint=is_self_adjoint,
      is_positive_definite=is_positive_definite)


def _static_check_for_same_dimensions(operators):
  # [FUNCTION DOC]
  # HELPER: validation step used by add_operators.
  # Given a list of operators, returns immediately if fewer than two. Otherwise collects the
  # statically known domain dimensions and the statically known range dimensions (unknown ones
  # are skipped) and raises a value error, listing operator names with their dimensions, if more
  # than one distinct value occurs for either. Returns nothing; no side effects.
  """ValueError if operators determined to have different dimensions."""
  if len(operators) < 2:
    return

  domain_dimensions = [
      (op.name, tensor_shape.dimension_value(op.domain_dimension))
      for op in operators
      if tensor_shape.dimension_value(op.domain_dimension) is not None]
  if len(set(value for name, value in domain_dimensions)) > 1:
    raise ValueError(f"All `operators` must have the same `domain_dimension`. "
                     f"Received: {domain_dimensions}.")

  range_dimensions = [
      (op.name, tensor_shape.dimension_value(op.range_dimension))
      for op in operators
      if tensor_shape.dimension_value(op.range_dimension) is not None]
  if len(set(value for name, value in range_dimensions)) > 1:
    raise ValueError(f"All operators must have the same `range_dimension`. "
                     f"Received: {range_dimensions}.")


def _static_check_for_broadcastable_batch_shape(operators):
  # [FUNCTION DOC]
  # HELPER: validation step used by add_operators.
  # Given a list of operators, returns immediately if fewer than two. Otherwise progressively
  # broadcasts their static batch shapes together; the broadcasting helper raises a value error
  # if two shapes cannot be broadcast. The result is discarded. Returns nothing.
  """ValueError if operators determined to have non-broadcastable shapes."""
  if len(operators) < 2:
    return

  # This will fail if they cannot be broadcast together.
  batch_shape = operators[0].batch_shape
  for op in operators[1:]:
    batch_shape = array_ops.broadcast_static_shape(batch_shape, op.batch_shape)


class _Hints:
  # [CLASS DOC]
  # HELPER: plain data holder for the matrix-property hints passed between the addition helpers
  # and adders.
  # Holds three attributes: is_non_singular, is_positive_definite and is_self_adjoint, each true,
  # false or null (unknown). Created by the helpers with no arguments or with keyword values and
  # read when constructing the result operator of an addition.
  """Holds 'is_X' flags that every LinearOperator is initialized with."""

  def __init__(self,
               is_non_singular=None,
               is_positive_definite=None,
               is_self_adjoint=None):
    # [FUNCTION DOC]
    # HELPER: constructor of the hints holder.
    # Stores the three optional hints (non-singular, positive definite, self-adjoint; each true,
    # false or null for unknown, all defaulting to null) as attributes of the same names.
    # Returns nothing.
    self.is_non_singular = is_non_singular
    self.is_positive_definite = is_positive_definite
    self.is_self_adjoint = is_self_adjoint


################################################################################
# Classes to add two linear operators.
################################################################################


class _Adder(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # HELPER: abstract base class for the addition strategies used by add_operators.
  # Stateless. Subclasses declare which pairs of operators they can add (can_add) and how to
  # build the sum (the private add hook); the public add method wraps that with hint inference,
  # default naming and a name scope. Uses an abstract-base-class metaclass, so subclasses must
  # implement both abstract methods.
  """Abstract base class to add two operators.

  Each `Adder` acts independently, adding everything it can, paying no attention
  as to whether another `Adder` could have done the addition more efficiently.
  """

  @property
  def name(self):
    # [FUNCTION DOC]
    # HELPER: naming support for adders.
    # Read-only property returning the adder's class name, used as the name scope of additions.
    return self.__class__.__name__

  @abc.abstractmethod
  def can_add(self, op1, op2):
    # [FUNCTION DOC]
    # Abstract predicate. Subclasses take two LinearOperators and return true if this adder is
    # able to add them, false otherwise. The base implementation does nothing (returns null).
    """Returns `True` if this `Adder` can add `op1` and `op2`.  Else `False`."""
    pass

  @abc.abstractmethod
  def _add(self, op1, op2, operator_name, hints):
    # [FUNCTION DOC]
    # Abstract hook. Subclasses build and return the LinearOperator representing op1 plus op2,
    # given the result name and a finished hints object. Callers guarantee that the operands
    # were validated (same dtype, matching dimensions). The base implementation does nothing.
    # Derived classes can assume op1 and op2 have been validated, e.g. they have
    # the same dtype, and their domain/range dimensions match.
    pass

  def add(self, op1, op2, operator_name, hints=None):
    # [FUNCTION DOC]
    # Public entry point for adding two operators with this adder. Inputs: op1 and op2
    # (LinearOperators that can be added), operator_name (string name for the result, or null to
    # build "Add/" plus both names), and hints (optional hints object overriding inferred
    # hints). Infers final hints, then calls the subclass hook inside a name scope named after
    # the adder class (without a leading underscore). Returns the new LinearOperator. Does not
    # itself verify can_add; callers are responsible.
    """Return new `LinearOperator` acting like `op1 + op2`.

    Args:
      op1:  `LinearOperator`
      op2:  `LinearOperator`, with `shape` and `dtype` such that adding to
        `op1` is allowed.
      operator_name:  `String` name to give to returned `LinearOperator`
      hints:  `_Hints` object.  Returned `LinearOperator` will be created with
        these hints.

    Returns:
      `LinearOperator`
    """
    updated_hints = _infer_hints_allowing_override(op1, op2, hints)

    if operator_name is None:
      operator_name = "Add/" + op1.name + "__" + op2.name + "/"

    scope_name = self.name
    if scope_name.startswith("_"):
      scope_name = scope_name[1:]
    with ops.name_scope(scope_name):
      return self._add(op1, op2, operator_name, updated_hints)


class _AddAndReturnScaledIdentity(_Adder):
  # [CLASS DOC]
  # HELPER: concrete adder, first tier of the default tiers.
  # Adds two operators that both belong to the identity family (identity or scaled identity) and
  # returns a scaled identity operator, since that family is closed under addition. Stateless.
  """Handles additions resulting in an Identity family member.

  The Identity (`LinearOperatorScaledIdentity`, `LinearOperatorIdentity`) family
  is closed under addition.  This `Adder` respects that, and returns an Identity
  """

  def can_add(self, op1, op2):
    # [FUNCTION DOC]
    # Returns true when both operators are identity or scaled identity operators. Raises a type
    # error (from the type classifier) if either operator is of an unsupported class.
    types = {_type(op1), _type(op2)}
    return not types.difference(_IDENTITY_FAMILY)

  def _add(self, op1, op2, operator_name, hints):
    # [FUNCTION DOC]
    # Builds a scaled identity operator equal to op1 plus op2. The multiplier of each operand is
    # its own multiplier if it is a scaled identity, otherwise a tensor of ones (matching the
    # operand's batch shape and dtype) for a plain identity. The result has the range dimension
    # of op1, the sum of the multipliers, the given hints, and the given name.
    # Will build a LinearOperatorScaledIdentity.

    if _type(op1) == _SCALED_IDENTITY:
      multiplier_1 = op1.multiplier
    else:
      multiplier_1 = array_ops.ones(op1.batch_shape_tensor(), dtype=op1.dtype)

    if _type(op2) == _SCALED_IDENTITY:
      multiplier_2 = op2.multiplier
    else:
      multiplier_2 = array_ops.ones(op2.batch_shape_tensor(), dtype=op2.dtype)

    return linear_operator_identity.LinearOperatorScaledIdentity(
        num_rows=op1.range_dimension_tensor(),
        multiplier=multiplier_1 + multiplier_2,
        is_non_singular=hints.is_non_singular,
        is_self_adjoint=hints.is_self_adjoint,
        is_positive_definite=hints.is_positive_definite,
        name=operator_name)


class _AddAndReturnDiag(_Adder):
  # [CLASS DOC]
  # HELPER: concrete adder, second tier of the default tiers.
  # Adds two diagonal-like operators (diagonal, identity or scaled identity) and returns a
  # diagonal operator. Stateless.
  """Handles additions resulting in a Diag operator."""

  def can_add(self, op1, op2):
    # [FUNCTION DOC]
    # Returns true when both operators are diagonal, identity or scaled identity operators.
    # Raises a type error (from the type classifier) for unsupported operator classes.
    types = {_type(op1), _type(op2)}
    return not types.difference(_DIAG_LIKE)

  def _add(self, op1, op2, operator_name, hints):
    # [FUNCTION DOC]
    # Builds a diagonal operator whose diagonal is the sum of the two operands' diagonal parts,
    # using the given hints and name.
    return linear_operator_diag.LinearOperatorDiag(
        diag=op1.diag_part() + op2.diag_part(),
        is_non_singular=hints.is_non_singular,
        is_self_adjoint=hints.is_self_adjoint,
        is_positive_definite=hints.is_positive_definite,
        name=operator_name)


class _AddAndReturnTriL(_Adder):
  # [CLASS DOC]
  # HELPER: concrete adder, third tier of the default tiers.
  # Adds operators that are each diagonal-like or lower triangular and returns a lower
  # triangular operator. Stateless.
  """Handles additions resulting in a TriL operator."""

  def can_add(self, op1, op2):
    # [FUNCTION DOC]
    # Returns true when each operator is diagonal, identity, scaled identity or lower triangular.
    # Raises a type error (from the type classifier) for unsupported operator classes.
    types = {_type(op1), _type(op2)}
    return not types.difference(_DIAG_LIKE.union({_TRIL}))

  def _add(self, op1, op2, operator_name, hints):
    # [FUNCTION DOC]
    # Builds a lower triangular operator. Picks as the "adding" operand the first one whose type
    # has an efficient add-to-tensor method (diagonal-like), otherwise the second, then adds it to
    # the dense form of the other operand. The resulting dense lower triangular matrix is wrapped
    # with the given hints and name. Materializes a dense matrix.
    if _type(op1) in _EFFICIENT_ADD_TO_TENSOR:
      op_add_to_tensor, op_other = op1, op2
    else:
      op_add_to_tensor, op_other = op2, op1

    return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
        tril=op_add_to_tensor.add_to_tensor(op_other.to_dense()),
        is_non_singular=hints.is_non_singular,
        is_self_adjoint=hints.is_self_adjoint,
        is_positive_definite=hints.is_positive_definite,
        name=operator_name)


class _AddAndReturnMatrix(_Adder):
  # [CLASS DOC]
  # HELPER: concrete adder, last (catch-all) tier of the default tiers.
  # KNOWN ISSUE: the docstring below starts with four quote characters, so the text begins with
  # a stray quote mark; harmless but a typo.
  # Adds any two LinearOperators by densifying and returns a full-matrix operator. Stateless.
  """"Handles additions resulting in a `LinearOperatorFullMatrix`."""

  def can_add(self, op1, op2):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Returns true if both arguments are LinearOperator instances (any kind), false otherwise.
    return isinstance(op1, linear_operator.LinearOperator) and isinstance(
        op2, linear_operator.LinearOperator)

  def _add(self, op1, op2, operator_name, hints):
    # [FUNCTION DOC]
    # Builds a full-matrix operator. Chooses as the "adding" operand the first one with an
    # efficient add-to-tensor method (else the second), adds it to the other operand's dense
    # form, and wraps the dense result with the given hints and name. Materializes the full
    # matrix, which can be memory and time intensive. Raises a type error for unsupported
    # operator classes (from the type classifier).
    if _type(op1) in _EFFICIENT_ADD_TO_TENSOR:
      op_add_to_tensor, op_other = op1, op2
    else:
      op_add_to_tensor, op_other = op2, op1
    return linear_operator_full_matrix.LinearOperatorFullMatrix(
        matrix=op_add_to_tensor.add_to_tensor(op_other.to_dense()),
        is_non_singular=hints.is_non_singular,
        is_self_adjoint=hints.is_self_adjoint,
        is_positive_definite=hints.is_positive_definite,
        name=operator_name)


################################################################################
# Constants designating types of LinearOperators
################################################################################

# Type name constants for LinearOperator classes.
_IDENTITY = "identity"
_SCALED_IDENTITY = "scaled_identity"
_DIAG = "diag"
_TRIL = "tril"
_MATRIX = "matrix"

# Groups of operators.
_DIAG_LIKE = {_DIAG, _IDENTITY, _SCALED_IDENTITY}
_IDENTITY_FAMILY = {_IDENTITY, _SCALED_IDENTITY}
# operators with an efficient .add_to_tensor() method.
_EFFICIENT_ADD_TO_TENSOR = _DIAG_LIKE

# Supported LinearOperator classes.
SUPPORTED_OPERATORS = [
    linear_operator_diag.LinearOperatorDiag,
    linear_operator_lower_triangular.LinearOperatorLowerTriangular,
    linear_operator_full_matrix.LinearOperatorFullMatrix,
    linear_operator_identity.LinearOperatorIdentity,
    linear_operator_identity.LinearOperatorScaledIdentity
]


def _type(operator):
  # [FUNCTION DOC]
  # HELPER: classifier used by the adders.
  # Maps a LinearOperator to one of the type-name constants (diagonal, lower triangular, full
  # matrix, identity, scaled identity) by checking its class. Raises a type error naming the
  # supported classes if the operator is none of them (so subclasses of other operator types,
  # such as the adjoint operator, are rejected). No side effects.
  """Returns the type name constant (e.g. _TRIL) for operator."""
  if isinstance(operator, linear_operator_diag.LinearOperatorDiag):
    return _DIAG
  if isinstance(operator,
                linear_operator_lower_triangular.LinearOperatorLowerTriangular):
    return _TRIL
  if isinstance(operator, linear_operator_full_matrix.LinearOperatorFullMatrix):
    return _MATRIX
  if isinstance(operator, linear_operator_identity.LinearOperatorIdentity):
    return _IDENTITY
  if isinstance(operator,
                linear_operator_identity.LinearOperatorScaledIdentity):
    return _SCALED_IDENTITY
  raise TypeError(f"Expected operator to be one of [LinearOperatorDiag, "
                  f"LinearOperatorLowerTriangular, LinearOperatorFullMatrix, "
                  f"LinearOperatorIdentity, LinearOperatorScaledIdentity]. "
                  f"Received: {operator}")


################################################################################
# Addition tiers:
# We attempt to use Adders in tier K before K+1.
#
# Organize tiers to
#   (i) reduce O(..) complexity of forming final operator, and
#   (ii) produce the "most efficient" final operator.
# Dev notes:
#  * Results of addition at tier K will be added at tier K or higher.
#  * Tiers may change, and we warn the user that it may change.
################################################################################

# Note that the final tier, _AddAndReturnMatrix, will convert everything to a
# dense matrix.  So it is sometimes very inefficient.
_DEFAULT_ADDITION_TIERS = [
    [_AddAndReturnScaledIdentity()],
    [_AddAndReturnDiag()],
    [_AddAndReturnTriL()],
    [_AddAndReturnMatrix()],
]
