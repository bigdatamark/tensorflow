# [MODULE DOC]
# Defines LinearOperatorAdjoint, a linear operator that lazily represents the (conjugate) transpose
# of another linear operator. Nothing is materialized: every operation (matmul, solve, matvec,
# determinant, trace, to_dense, eigenvalues, condition number, shape) is forwarded to the wrapped
# operator with the adjoint flag flipped or with conjugation applied where needed.
# Exposes only LinearOperatorAdjoint (listed in the module export list and exported publicly as
# the linalg namespace entry of the same name). Part of the linalg linear-operator family and
# built on the LinearOperator base class and its composite-tensor decorator.
# Import-time side effects: registers the class with the public API export mechanism and wraps it
# so it can act as a composite tensor.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Takes the adjoint of a `LinearOperator`."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.util.tf_export import tf_export

__all__ = ["LinearOperatorAdjoint"]


@tf_export("linalg.LinearOperatorAdjoint")
@linear_operator.make_composite_tensor
class LinearOperatorAdjoint(linear_operator.LinearOperator):
  # [CLASS DOC]
  # A linear operator representing the adjoint (conjugate transpose) of a wrapped operator.
  # Holds a single piece of state: the wrapped operator, available through the operator property.
  # Instances are created by passing an existing LinearOperator (optionally with matrix-property
  # hints and a name) and are used like any other LinearOperator; all computations delegate to
  # the wrapped operator with the adjoint direction flipped. Subclasses the LinearOperator base
  # class, overrides its private hook methods, and is decorated to be a composite tensor whose
  # only tensor-like field is the wrapped operator. Exported publicly under the linalg namespace.
  """`LinearOperator` representing the adjoint of another operator.

  This operator represents the adjoint of another operator.

  ```python
  # Create a 2 x 2 linear operator.
  operator = LinearOperatorFullMatrix([[1 - i., 3.], [0., 1. + i]])
  operator_adjoint = LinearOperatorAdjoint(operator)

  operator_adjoint.to_dense()
  ==> [[1. + i, 0.]
       [3., 1 - i]]

  operator_adjoint.shape
  ==> [2, 2]

  operator_adjoint.log_abs_determinant()
  ==> - log(2)

  x = ... Shape [2, 4] Tensor
  operator_adjoint.matmul(x)
  ==> Shape [2, 4] Tensor, equal to operator.matmul(x, adjoint=True)
  ```

  #### Performance

  The performance of `LinearOperatorAdjoint` depends on the underlying
  operators performance.

  #### Matrix property hints

  This `LinearOperator` is initialized with boolean flags of the form `is_X`,
  for `X = non_singular, self_adjoint, positive_definite, square`.
  These have the following meaning:

  * If `is_X == True`, callers should expect the operator to have the
    property `X`.  This is a promise that should be fulfilled, but is *not* a
    runtime assert.  For example, finite floating point precision may result
    in these promises being violated.
  * If `is_X == False`, callers should expect the operator to not have `X`.
  * If `is_X == None` (the default), callers should have no expectation either
    way.
  """

  def __init__(self,
               operator,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a value error is raised if operator.is_non_singular is
    # False, but this constructor never checks that. A value error is only raised by the hint
    # combining helper when a supplied hint contradicts the wrapped operator's known hint, or by
    # the base class constructor for inconsistent hint combinations.
    # Builds an adjoint operator around the given operator. Inputs: operator (a LinearOperator),
    # optional tri-state hints is_non_singular, is_self_adjoint, is_positive_definite and
    # is_square (true, false or unknown), and an optional name (default is the wrapped operator's
    # name plus an adjoint suffix). Each hint is merged with the wrapped operator's own hint
    # (the adjoint is square / non-singular / self-adjoint / positive-definite exactly when the
    # original is), raising a value error on contradiction. Stores the wrapped operator, records
    # the constructor arguments as the operator parameters, and calls the base constructor
    # inside a name scope with the wrapped operator's dtype. Returns nothing.
    r"""Initialize a `LinearOperatorAdjoint`.

    `LinearOperatorAdjoint` is initialized with an operator `A`.  The `solve`
    and `matmul` methods  effectively flip the `adjoint` argument.  E.g.

    ```
    A = MyLinearOperator(...)
    B = LinearOperatorAdjoint(A)
    x = [....]  # a vector

    assert A.matvec(x, adjoint=True) == B.matvec(x, adjoint=False)
    ```

    Args:
      operator: `LinearOperator` object.
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`. Default is `operator.name +
        "_adjoint"`.

    Raises:
      ValueError:  If `operator.is_non_singular` is False.
    """
    parameters = dict(
        operator=operator,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name,
    )

    self._operator = operator

    # The congruency of is_non_singular and is_self_adjoint was checked in the
    # base operator.
    combine_hint = (
        linear_operator_util.use_operator_or_provided_hint_unless_contradicting)

    is_square = combine_hint(
        operator, "is_square", is_square,
        "An operator is square if and only if its adjoint is square.")

    is_non_singular = combine_hint(
        operator, "is_non_singular", is_non_singular,
        "An operator is non-singular if and only if its adjoint is "
        "non-singular.")

    is_self_adjoint = combine_hint(
        operator, "is_self_adjoint", is_self_adjoint,
        "An operator is self-adjoint if and only if its adjoint is "
        "self-adjoint.")

    is_positive_definite = combine_hint(
        operator, "is_positive_definite", is_positive_definite,
        "An operator is positive-definite if and only if its adjoint is "
        "positive-definite.")

    # Initialization.
    if name is None:
      name = operator.name + "_adjoint"
    with ops.name_scope(name):
      super(LinearOperatorAdjoint, self).__init__(
          dtype=operator.dtype,
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
          parameters=parameters,
          name=name)

  @property
  def operator(self):
    # [FUNCTION DOC]
    # Read-only property returning the wrapped LinearOperator whose adjoint this object
    # represents. No side effects.
    """The operator before taking the adjoint."""
    return self._operator

  def _linop_adjoint(self) -> linear_operator.LinearOperator:
    # [FUNCTION DOC]
    # HELPER: hook used by the base class adjoint method.
    # Returns the adjoint of this operator, which is simply the wrapped operator (the adjoint of
    # an adjoint is the original). Takes no inputs and has no side effects.
    return self.operator

  def _assert_non_singular(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class assert-non-singular method.
    # Returns the assertion operation produced by the wrapped operator's own non-singularity
    # check (an adjoint is non-singular exactly when the original is). No inputs.
    return self.operator.assert_non_singular()

  def _assert_positive_definite(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class assert-positive-definite method.
    # Returns the assertion operation from the wrapped operator's positive-definiteness check,
    # since the adjoint is positive definite exactly when the original is. No inputs.
    return self.operator.assert_positive_definite()

  def _assert_self_adjoint(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class assert-self-adjoint method.
    # Returns the assertion operation from the wrapped operator's self-adjointness check, since
    # the adjoint is self-adjoint exactly when the original is. No inputs.
    return self.operator.assert_self_adjoint()

  def _shape(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class static shape property.
    # Returns the static shape of the wrapped operator with its last two (matrix) dimensions
    # swapped; the batch dimensions are kept. No inputs, no side effects.
    # Rotate last dimension
    shape = self.operator.shape
    return shape[:-2].concatenate([shape[-1], shape[-2]])

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class dynamic shape method.
    # Returns a one-dimensional integer tensor: the wrapped operator's dynamic shape with the
    # last two entries swapped (batch entries unchanged). No inputs.
    # Rotate last dimension
    shape = self.operator.shape_tensor()
    return array_ops.concat([
        shape[:-2], [shape[-1], shape[-2]]], axis=-1)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # HELPER: hook for the base class matmul method.
    # Multiplies this operator by the tensor x. Inputs: x (a tensor with compatible dtype and
    # shape), adjoint (apply the adjoint of this operator, default false) and adjoint_arg
    # (use the adjoint of x, default false). Delegates to the wrapped operator's matmul with the
    # adjoint flag negated and returns the resulting tensor.
    return self.operator.matmul(
        x, adjoint=(not adjoint), adjoint_arg=adjoint_arg)

  def _matvec(self, x, adjoint=False):
    # [FUNCTION DOC]
    # HELPER: hook for the base class matvec method.
    # Applies this operator to the vector (batch) x. Inputs: x, and adjoint (default false).
    # Delegates to the wrapped operator's matvec with the adjoint flag negated and returns the
    # resulting tensor.
    return self.operator.matvec(x, adjoint=(not adjoint))

  def _determinant(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class determinant method.
    # Returns the determinant of the wrapped operator if this operator is known to be
    # self-adjoint, otherwise its complex conjugate (the determinant of an adjoint is the
    # conjugate of the determinant). No inputs.
    if self.is_self_adjoint:
      return self.operator.determinant()
    return math_ops.conj(self.operator.determinant())

  def _log_abs_determinant(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class log-absolute-determinant method.
    # Returns the wrapped operator's log absolute determinant unchanged, since conjugation does
    # not alter the magnitude. No inputs.
    return self.operator.log_abs_determinant()

  def _trace(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class trace method.
    # Returns the wrapped operator's trace if this operator is known to be self-adjoint,
    # otherwise the complex conjugate of that trace. No inputs.
    if self.is_self_adjoint:
      return self.operator.trace()
    return math_ops.conj(self.operator.trace())

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # HELPER: hook for the base class solve method.
    # Solves a linear system with this operator. Inputs: rhs (right-hand-side tensor), adjoint
    # (solve with the adjoint of this operator, default false) and adjoint_arg (use the adjoint
    # of rhs, default false). Delegates to the wrapped operator's solve with the adjoint flag
    # negated and returns the solution tensor.
    return self.operator.solve(
        rhs, adjoint=(not adjoint), adjoint_arg=adjoint_arg)

  def _solvevec(self, rhs, adjoint=False):
    # [FUNCTION DOC]
    # HELPER: hook for the base class solvevec method.
    # Solves a linear system whose right-hand side is a vector (batch) rhs. Input adjoint
    # defaults to false. Delegates to the wrapped operator's solvevec with the adjoint flag
    # negated and returns the solution tensor.
    return self.operator.solvevec(rhs, adjoint=(not adjoint))

  def _to_dense(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class to-dense method.
    # Returns the dense matrix of the wrapped operator if this operator is known to be
    # self-adjoint, otherwise the conjugate transpose of that dense matrix. No inputs.
    if self.is_self_adjoint:
      return self.operator.to_dense()
    return linalg.adjoint(self.operator.to_dense())

  def _add_to_tensor(self, x):
    # [FUNCTION DOC]
    # HELPER: hook for the base class add-to-tensor method.
    # Returns the dense form of this operator added to the tensor x (a tensor of compatible
    # shape and dtype). Materializes the full matrix, so it is not memory efficient.
    return self.to_dense() + x

  def _eigvals(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class eigenvalues method.
    # Returns the wrapped operator's eigenvalues, complex-conjugated unless the wrapped operator
    # is known to be self-adjoint. No inputs.
    eigvals = self.operator.eigvals()
    if not self.operator.is_self_adjoint:
      eigvals = math_ops.conj(eigvals)
    return eigvals

  def _cond(self):
    # [FUNCTION DOC]
    # HELPER: hook for the base class condition-number method.
    # Returns the wrapped operator's condition number, which is the same for its adjoint.
    return self.operator.cond()

  @property
  def _composite_tensor_fields(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor support hook.
    # Property returning a tuple with the single name of the constructor argument (the wrapped
    # operator) that is treated as the composite tensor's component.
    return ("operator",)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    # [FUNCTION DOC]
    # HELPER: batching support hook for composite-tensor handling.
    # Property returning a mapping from the constructor argument name (the wrapped operator) to
    # zero, meaning that argument's batch rank equals the batch rank of this operator.
    return {"operator": 0}
