# [MODULE DOC]
# Defines LinearOperator, the abstract base class of the whole linalg linear-operator family, plus
# supporting machinery: a composite-tensor gradient helper, a composite tensor type
# specification for operators, a class decorator that turns an operator subclass into a
# composite tensor, and several small private helpers.
# A LinearOperator represents a (batch of) matrices without materializing them; subclasses
# implement private hooks (shape, matmul, solve, determinant, and so on) and the base class
# supplies the public methods with validation, name scopes, dtype checks and generic (often slow)
# default implementations built on dense conversion. Also provides operator algebra (adjoint,
# inverse, cholesky, addition, multiplication, slicing via indexing), matrix property hints and
# composite-tensor support (flattening to tensors, type specs, gradients).
# Exposes only LinearOperator in its export list (also exported publicly in the linalg namespace);
# the decorator and type-spec helpers are used by the sibling operator modules.
# Import-time side effects: registers LinearOperator for public API export and registers the
# composite type specification with the type-spec registry; no flags are defined.
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
"""Base class for linear operators."""

import abc
import contextlib

import numpy as np

from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.ops.linalg import property_hint_util
from tensorflow.python.ops.linalg import slicing
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export


__all__ = ["LinearOperator"]


# pylint: disable=protected-access
class _LinearOperatorGradient(
    composite_tensor_gradient.CompositeTensorGradient):
  # [CLASS DOC]
  # HELPER: gradient support for composite-tensor LinearOperators.
  # Tells the composite tensor gradient machinery how to split a LinearOperator into its
  # gradient-carrying components and how to rebuild an operator from component gradients.
  # Stateless; used as the gradient protocol of operator type specifications.
  """Composite tensor gradient for `LinearOperator`."""

  def get_gradient_components(self, value):
    # [FUNCTION DOC]
    # Returns the components of the given operator value (as defined by its type specification's
    # component extraction), which are the parts that receive gradients.
    return value._type_spec._to_components(value)

  def replace_gradient_components(self, value, components):
    # [FUNCTION DOC]
    # Builds a new operator from gradients. Inputs: value (the original operator, used for its
    # type specification and dtype) and components (a nested structure of component gradients,
    # entries may be null for disconnected gradients). If every component gradient is null,
    # returns null. Otherwise replaces each null component gradient with zeros shaped like the
    # corresponding original component (using the operator's dtype), repacks the structure, and
    # returns an operator reconstructed from these gradient components via the type
    # specification. No other side effects.
    flat_components = nest.flatten(components)

    # If all component gradients are disconnected, return None.
    if all(c is None for c in flat_components):
      return None

    # TODO(b/286565628): Update this once `CompositeTensorGradient` fully
    # supports `tf.UnconnectedGradients.ZERO`.
    # Replace individual disconnected component gradients with zeros.
    value_components = value._type_spec._to_components(value)
    flat_grad_components = []
    for gc, vc in zip(flat_components, nest.flatten(value_components)):
      if gc is None:
        flat_grad_components.append(
            nest.map_structure(
                lambda x: array_ops.zeros_like(x, dtype=value.dtype),
                vc,
                expand_composites=True))
      else:
        flat_grad_components.append(gc)
    grad_components = nest.pack_sequence_as(
        value_components, flat_grad_components)
    return value._type_spec._from_components(grad_components)
# pylint: enable=protected-access


# TODO(langmore) Use matrix_solve_ls for singular or non-square matrices.
@tf_export("linalg.LinearOperator")
class LinearOperator(
    module.Module, composite_tensor.CompositeTensor, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class for every linear operator: an object acting like a (batch of) matrices
  # of shape batch dimensions followed by rows and columns, without materializing the matrix.
  # State: dtype, name, the matrix-property hints (non-singular, self-adjoint, positive
  # definite, square), the dictionary of constructor parameters, and a deprecated list of graph
  # parents. Subclasses override private hook methods (at minimum the shape and matmul hooks);
  # the public methods wrap them with name scopes and dtype checks, and many have slow generic
  # fallbacks based on converting to a dense matrix. Inherits from the module base class (for
  # variable tracking) and the composite tensor base class, with an abstract-class metaclass.
  # Exported publicly under the linalg namespace.
  """Base class defining a [batch of] linear operator[s].

  Subclasses of `LinearOperator` provide access to common methods on a
  (batch) matrix, without the need to materialize the matrix.  This allows:

  * Matrix free computations
  * Operators that take advantage of special structure, while providing a
    consistent API to users.

  #### Subclassing

  To enable a public method, subclasses should implement the leading-underscore
  version of the method.  The argument signature should be identical except for
  the omission of `name="..."`.  For example, to enable
  `matmul(x, adjoint=False, name="matmul")` a subclass should implement
  `_matmul(x, adjoint=False)`.

  #### Performance contract

  Subclasses should only implement the assert methods
  (e.g. `assert_non_singular`) if they can be done in less than `O(N^3)`
  time.

  Class docstrings should contain an explanation of computational complexity.
  Since this is a high-performance library, attention should be paid to detail,
  and explanations can include constants as well as Big-O notation.

  #### Shape compatibility

  `LinearOperator` subclasses should operate on a [batch] matrix with
  compatible shape.  Class docstrings should define what is meant by compatible
  shape.  Some subclasses may not support batching.

  Examples:

  `x` is a batch matrix with compatible shape for `matmul` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  x.shape =   [B1,...,Bb] + [N, R]
  ```

  `rhs` is a batch matrix with compatible shape for `solve` if

  ```
  operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
  rhs.shape =   [B1,...,Bb] + [M, R]
  ```

  #### Example docstring for subclasses.

  This operator acts like a (batch) matrix `A` with shape
  `[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
  batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
  an `m x n` matrix.  Again, this matrix `A` may not be materialized, but for
  purposes of identifying and working with compatible arguments the shape is
  relevant.

  Examples:

  ```python
  some_tensor = ... shape = ????
  operator = MyLinOp(some_tensor)

  operator.shape()
  ==> [2, 4, 4]

  operator.log_abs_determinant()
  ==> Shape [2] Tensor

  x = ... Shape [2, 4, 5] Tensor

  operator.matmul(x)
  ==> Shape [2, 4, 5] Tensor
  ```

  #### Shape compatibility

  This operator acts on batch matrices with compatible shape.
  FILL IN WHAT IS MEANT BY COMPATIBLE SHAPE

  #### Performance

  FILL THIS IN

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

  #### Initialization parameters

  All subclasses of `LinearOperator` are expected to pass a `parameters`
  argument to `super().__init__()`.  This should be a `dict` containing
  the unadulterated arguments passed to the subclass `__init__`.  For example,
  `MyLinearOperator` with an initializer should look like:

  ```python
  def __init__(self, operator, is_square=False, name=None):
     parameters = dict(
         operator=operator,
         is_square=is_square,
         name=name
     )
     ...
     super().__init__(..., parameters=parameters)
  ```

   Users can then access `my_linear_operator.parameters` to see all arguments
   passed to its initializer.
  """

  # TODO(b/143910018) Remove graph_parents in V3.
  @deprecation.deprecated_args(None, "Do not pass `graph_parents`.  They will "
                               " no longer be used.", "graph_parents")
  def __init__(self,
               dtype,
               graph_parents=None,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None,
               name=None,
               parameters=None):
    # [FUNCTION DOC]
    # Base constructor for subclasses. Inputs: dtype (data type of tensors handled; may be null,
    # otherwise converted to its base type), graph_parents (deprecated list, default null),
    # tri-state hints is_non_singular, is_self_adjoint, is_positive_definite and is_square
    # (true, false or unknown), name (default is the class name) and parameters (mapping of the
    # subclass constructor arguments). Hint propagation: positive definite implies non-singular,
    # non-singular implies square, self-adjoint implies square. Raises a value error if a hint
    # contradicts those implications (positive definite with non-singular false, non-singular
    # with square false, self-adjoint with square false). A supplied graph_parents list is
    # validated by the setter helper, which raises a value error for non-tensor members.
    # Stores all values as private
    # attributes, wraps parameters so they are not tracked as dependencies, and marks parameters
    # as not yet sanitized. Returns nothing.
    """Initialize the `LinearOperator`.

    **This is a private method for subclass use.**
    **Subclasses should copy-paste this `__init__` documentation.**

    Args:
      dtype: The type of the this `LinearOperator`.  Arguments to `matmul` and
        `solve` will have to be this type.
      graph_parents: (Deprecated) Python list of graph prerequisites of this
        `LinearOperator` Typically tensors that are passed during initialization
      is_non_singular:  Expect that this operator is non-singular.
      is_self_adjoint:  Expect that this operator is equal to its hermitian
        transpose.  If `dtype` is real, this is equivalent to being symmetric.
      is_positive_definite:  Expect that this operator is positive definite,
        meaning the quadratic form `x^H A x` has positive real part for all
        nonzero `x`.  Note that we do not require the operator to be
        self-adjoint to be positive-definite.  See:
        https://en.wikipedia.org/wiki/Positive-definite_matrix#Extension_for_non-symmetric_matrices
      is_square:  Expect that this operator acts like square [batch] matrices.
      name: A name for this `LinearOperator`.
      parameters: Python `dict` of parameters used to instantiate this
        `LinearOperator`.

    Raises:
      ValueError:  If any member of graph_parents is `None` or not a `Tensor`.
      ValueError:  If hints are set incorrectly.
    """
    # Check and auto-set flags.
    if is_positive_definite:
      if is_non_singular is False:
        raise ValueError("A positive definite matrix is always non-singular.")
      is_non_singular = True

    if is_non_singular:
      if is_square is False:
        raise ValueError("A non-singular matrix is always square.")
      is_square = True

    if is_self_adjoint:
      if is_square is False:
        raise ValueError("A self-adjoint matrix is always square.")
      is_square = True

    self._is_square_set_or_implied_by_hints = is_square

    if graph_parents is not None:
      self._set_graph_parents(graph_parents)
    else:
      self._graph_parents = []
    self._dtype = dtypes.as_dtype(dtype).base_dtype if dtype else dtype
    self._is_non_singular = is_non_singular
    self._is_self_adjoint = is_self_adjoint
    self._is_positive_definite = is_positive_definite
    self._parameters = self._no_dependency(parameters)
    self._parameters_sanitized = False
    self._name = name or type(self).__name__

  @contextlib.contextmanager
  def _name_scope(self, name=None):  # pylint: disable=method-hidden
    # [FUNCTION DOC]
    # HELPER: context manager used by all public methods to create ops under a common scope.
    # Input: optional name suffix. Enters a name scope named after the operator, with the suffix
    # appended after a slash when provided, and yields the scope string. Side effect: changes
    # the current graph name scope for the duration of the context.
    """Helper function to standardize op scope."""
    full_name = self.name
    if name is not None:
      full_name += "/" + name
    with ops.name_scope(full_name) as scope:
      yield scope

  @property
  def parameters(self):
    # [FUNCTION DOC]
    # Property returning a shallow copy (plain mapping) of the constructor parameters recorded at
    # initialization. Modifying the returned mapping does not change the operator.
    """Dictionary of parameters used to instantiate this `LinearOperator`."""
    return dict(self._parameters)

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the data type of tensors this operator handles (may be null if the
    # operator was created without one).
    """The `DType` of `Tensor`s handled by this `LinearOperator`."""
    return self._dtype

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the operator's name, used as the prefix of all ops it creates.
    """Name prepended to all ops created by this `LinearOperator`."""
    return self._name

  @property
  @deprecation.deprecated(None, "Do not call `graph_parents`.")
  def graph_parents(self):
    # [FUNCTION DOC]
    # Deprecated property returning the stored list of graph parents (empty unless a list was
    # supplied at construction). Calling it emits a deprecation warning.
    """List of graph dependencies of this `LinearOperator`."""
    return self._graph_parents

  @property
  def is_non_singular(self):
    # [FUNCTION DOC]
    # Property returning the non-singular hint: true, false, or null if unknown. It is a promise
    # given at construction (possibly implied by the positive definite hint), not a runtime check.
    return self._is_non_singular

  @property
  def is_self_adjoint(self):
    # [FUNCTION DOC]
    # Property returning the self-adjoint hint: true, false, or null if unknown. A promise given
    # at construction, not a runtime check.
    return self._is_self_adjoint

  @property
  def is_positive_definite(self):
    # [FUNCTION DOC]
    # Property returning the positive definite hint: true, false, or null if unknown. A promise
    # given at construction, not a runtime check.
    return self._is_positive_definite

  @property
  def is_square(self):
    # [FUNCTION DOC]
    # Returns whether the operator is square. Compares the static domain and range dimensions;
    # if the user hint said not square but the dimensions are equal, raises a value error. If
    # there is no hint (or implied hint) returns the result of the dimension comparison,
    # otherwise returns the hint. Note a comparison of unknown dimensions is treated as per the
    # dimension comparison rules (unknown results are falsy or null-like).
    """Return `True/False` depending on if this operator is square."""
    # Static checks done after __init__.  Why?  Because domain/range dimension
    # sometimes requires lots of work done in the derived class after init.
    auto_square_check = self.domain_dimension == self.range_dimension
    if self._is_square_set_or_implied_by_hints is False and auto_square_check:
      raise ValueError(
          "User set is_square hint to False, but the operator was square.")
    if self._is_square_set_or_implied_by_hints is None:
      return auto_square_check

    return self._is_square_set_or_implied_by_hints

  @abc.abstractmethod
  def _shape(self):
    # [FUNCTION DOC]
    # Abstract hook. Subclasses must return the static shape of the operator (batch dimensions
    # followed by rows and columns). The base version raises a not-implemented error.
    # Write this in derived class to enable all static shape methods.
    raise NotImplementedError("_shape is not implemented.")

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property returning the statically determined shape of the operator (batch dimensions plus
    # rows and columns), by calling the subclass hook. May be partially or fully unknown.
    """`TensorShape` of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb, M, N])`, equivalent to `A.shape`.

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    return self._shape()

  def _shape_tensor(self):
    # [FUNCTION DOC]
    # Optional hook. Subclasses return the runtime shape as a one-dimensional integer tensor.
    # Not abstract so subclasses may add optional keyword arguments; the base version raises a
    # not-implemented error.
    # This is not an abstractmethod, since we want derived classes to be able to
    # override this with optional kwargs, which can reduce the number of
    # `convert_to_tensor` calls.  See derived classes for examples.
    raise NotImplementedError("_shape_tensor is not implemented.")

  def shape_tensor(self, name="shape_tensor"):
    # [FUNCTION DOC]
    # Returns the operator's shape as an integer tensor. Input: name for the name scope. If the
    # static shape is fully defined it is converted to a constant tensor, otherwise the
    # subclass's runtime shape hook is called (which may raise a not-implemented error).
    """Shape of this `LinearOperator`, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      # Prefer to use statically defined shape if available.
      if self.shape.is_fully_defined():
        return linear_operator_util.shape_tensor(self.shape.as_list())
      else:
        return self._shape_tensor()

  @property
  def batch_shape(self):
    # [FUNCTION DOC]
    # Property returning the static batch shape: the operator's static shape without its last two
    # (matrix) dimensions. May be partially or fully unknown.
    """`TensorShape` of batch dimensions of this `LinearOperator`.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns
    `TensorShape([B1,...,Bb])`, equivalent to `A.shape[:-2]`

    Returns:
      `TensorShape`, statically determined, may be undefined.
    """
    # Derived classes get this "for free" once .shape is implemented.
    return self.shape[:-2]

  def batch_shape_tensor(self, name="batch_shape_tensor"):
    # [FUNCTION DOC]
    # Returns the batch shape as an integer tensor determined at runtime. Input: name for the
    # name scope. Delegates to the private batch shape helper inside a name scope.
    """Shape of batch dimensions of this operator, determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
    `[B1,...,Bb]`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._batch_shape_tensor()

  def _batch_shape_tensor(self, shape=None):
    # [FUNCTION DOC]
    # HELPER: implementation of batch_shape_tensor, overridable by subclasses.
    # Input: optional precomputed runtime shape tensor (saves conversions). If the static batch
    # shape is fully defined returns it as a constant tensor named "batch_shape", otherwise
    # returns all but the last two entries of the runtime shape (computed here if not given).
    # `shape` may be passed in if this can be pre-computed in a
    # more efficient manner, e.g. without excessive Tensor conversions.
    if self.batch_shape.is_fully_defined():
      return linear_operator_util.shape_tensor(
          self.batch_shape.as_list(), name="batch_shape")
    else:
      shape = self.shape_tensor() if shape is None else shape
      return shape[:-2]

  @property
  def tensor_rank(self, name="tensor_rank"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this is declared as a property yet takes a name argument; as a property it can
    # never be called with a name, so the argument always has its default and is effectively dead.
    # Returns the static rank of the operator viewed as a tensor (number of batch dimensions plus
    # two), or null when the rank is unknown. Creates a name scope while reading the shape.
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op`.

    Returns:
      Python integer, or None if the tensor rank is undefined.
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):  # pylint: disable=not-callable
      return self.shape.ndims

  def tensor_rank_tensor(self, name="tensor_rank_tensor"):
    # [FUNCTION DOC]
    # Returns the tensor rank (batch dimensions plus two) as an integer scalar tensor determined
    # at runtime. Input: name for the name scope. Delegates to the private helper.
    """Rank (in the sense of tensors) of matrix corresponding to this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`, determined at runtime.
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._tensor_rank_tensor()

  def _tensor_rank_tensor(self, shape=None):
    # [FUNCTION DOC]
    # HELPER: implementation of tensor_rank_tensor, overridable by subclasses.
    # Input: optional precomputed runtime shape tensor. If the static rank is known returns it as
    # a constant tensor, otherwise returns the number of elements of the runtime shape.
    # `shape` may be passed in if this can be pre-computed in a
    # more efficient manner, e.g. without excessive Tensor conversions.
    if self.tensor_rank is not None:
      return tensor_conversion.convert_to_tensor_v2_with_dispatch(
          self.tensor_rank
      )
    else:
      shape = self.shape_tensor() if shape is None else shape
      return array_ops.size(shape)

  @property
  def domain_dimension(self):
    # [FUNCTION DOC]
    # Property returning the static size of the domain (number of columns) as a dimension object:
    # the last entry of the static shape, or an unknown dimension if the rank is unknown.
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Returns:
      `Dimension` object.
    """
    # Derived classes get this "for free" once .shape is implemented.
    if self.shape.rank is None:
      return tensor_shape.Dimension(None)
    else:
      return self.shape.dims[-1]

  def domain_dimension_tensor(self, name="domain_dimension_tensor"):
    # [FUNCTION DOC]
    # Returns the number of columns as an integer scalar tensor determined at runtime. Input:
    # name for the name scope. Delegates to the private helper.
    """Dimension (in the sense of vector spaces) of the domain of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._domain_dimension_tensor()

  def _domain_dimension_tensor(self, shape=None):
    # [FUNCTION DOC]
    # HELPER: implementation of domain_dimension_tensor, overridable by subclasses.
    # Input: optional precomputed runtime shape tensor. Returns a constant tensor if the static
    # domain dimension is known, otherwise the last entry of the runtime shape.
    # `shape` may be passed in if this can be pre-computed in a
    # more efficient manner, e.g. without excessive Tensor conversions.
    dim_value = tensor_shape.dimension_value(self.domain_dimension)
    if dim_value is not None:
      return tensor_conversion.convert_to_tensor_v2_with_dispatch(dim_value)
    else:
      shape = self.shape_tensor() if shape is None else shape
      return shape[-1]

  @property
  def range_dimension(self):
    # [FUNCTION DOC]
    # Property returning the static size of the range (number of rows) as a dimension object: the
    # second to last entry of the static shape, or an unknown dimension if no dimensions exist.
    """Dimension (in the sense of vector spaces) of the range of this operator.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Returns:
      `Dimension` object.
    """
    # Derived classes get this "for free" once .shape is implemented.
    if self.shape.dims:
      return self.shape.dims[-2]
    else:
      return tensor_shape.Dimension(None)

  def range_dimension_tensor(self, name="range_dimension_tensor"):
    # [FUNCTION DOC]
    # Returns the number of rows as an integer scalar tensor determined at runtime. Input: name
    # for the name scope. Delegates to the private helper.
    """Dimension (in the sense of vector spaces) of the range of this operator.

    Determined at runtime.

    If this operator acts like the batch matrix `A` with
    `A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

    Args:
      name:  A name for this `Op`.

    Returns:
      `int32` `Tensor`
    """
    # Derived classes get this "for free" once .shape() is implemented.
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._range_dimension_tensor()

  def _range_dimension_tensor(self, shape=None):
    # [FUNCTION DOC]
    # HELPER: implementation of range_dimension_tensor, overridable by subclasses.
    # Input: optional precomputed runtime shape tensor. Returns a constant tensor if the static
    # range dimension is known, otherwise the second to last entry of the runtime shape.
    # `shape` may be passed in if this can be pre-computed in a
    # more efficient manner, e.g. without excessive Tensor conversions.
    dim_value = tensor_shape.dimension_value(self.range_dimension)
    if dim_value is not None:
      return tensor_conversion.convert_to_tensor_v2_with_dispatch(dim_value)
    else:
      shape = self.shape_tensor() if shape is None else shape
      return shape[-2]

  def _assert_non_singular(self):
    # [FUNCTION DOC]
    # Default (slow) non-singularity check. Logs a warning that it needs dense conversion and
    # cubic cost. If Cholesky can be used (see the helper that checks hints) it delegates to the
    # positive definite assertion; otherwise computes singular values of the dense matrix, takes
    # the ratio of largest to smallest, and returns an assertion op that this condition number is
    # below the maximum allowed. The assertion fails at run time with a message about a singular
    # matrix up to precision.
    """Private default implementation of _assert_non_singular."""
    logging.warn(
        "Using (possibly slow) default implementation of assert_non_singular."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      return self.assert_positive_definite()
    else:
      singular_values = linalg_ops.svd(self.to_dense(), compute_uv=False)
      # TODO(langmore) Add .eig and .cond as methods.
      cond = (math_ops.reduce_max(singular_values, axis=-1) /
              math_ops.reduce_min(singular_values, axis=-1))
      return check_ops.assert_less(
          cond,
          self._max_condition_number_to_be_non_singular(),
          message="Singular matrix up to precision epsilon.")

  def _max_condition_number_to_be_non_singular(self):
    # [FUNCTION DOC]
    # HELPER: threshold for the default non-singular assertion.
    # Returns a scalar tensor equal to the reciprocal of machine epsilon of the operator's dtype
    # times the larger of 100 and the range and domain dimensions; condition numbers above it
    # count as singular. Creates ops in a name scope.
    """Return the maximum condition number that we consider nonsingular."""
    with ops.name_scope("max_nonsingular_condition_number"):
      dtype_eps = np.finfo(self.dtype.as_numpy_dtype).eps
      eps = math_ops.cast(
          math_ops.reduce_max([
              100.,
              math_ops.cast(self.range_dimension_tensor(), self.dtype),
              math_ops.cast(self.domain_dimension_tensor(), self.dtype)
          ]), self.dtype) * dtype_eps
      return 1. / eps

  def assert_non_singular(self, name="assert_non_singular"):
    # [FUNCTION DOC]
    # Public method returning an assertion op that fails at run time with an invalid-argument
    # error if the operator is (numerically) singular. Input: name for the name scope. Delegates
    # to the private hook, which subclasses may override with something cheaper.
    """Returns an `Op` that asserts this operator is non singular.

    This operator is considered non-singular if

    ```
    ConditionNumber < max{100, range_dimension, domain_dimension} * eps,
    eps := np.finfo(self.dtype.as_numpy_dtype).eps
    ```

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is singular.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._assert_non_singular()

  def _assert_positive_definite(self):
    # [FUNCTION DOC]
    # Default (slow) positive definiteness check. Logs a warning about dense conversion and cubic
    # cost. If the operator is hinted self-adjoint, returns an assertion that the diagonal of the
    # Cholesky factor of the dense matrix is positive (message says the matrix was not positive
    # definite). Otherwise raises a not-implemented error since no generic check exists.
    """Default implementation of _assert_positive_definite."""
    logging.warn(
        "Using (possibly slow) default implementation of "
        "assert_positive_definite."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    # If the operator is self-adjoint, then checking that
    # Cholesky decomposition succeeds + results in positive diag is necessary
    # and sufficient.
    if self.is_self_adjoint:
      return check_ops.assert_positive(
          array_ops.matrix_diag_part(linalg_ops.cholesky(self.to_dense())),
          message="Matrix was not positive definite.")
    # We have no generic check for positive definite.
    raise NotImplementedError("assert_positive_definite is not implemented.")

  def assert_positive_definite(self, name="assert_positive_definite"):
    # [FUNCTION DOC]
    # Public method returning an assertion op that fails at run time if the operator is not
    # positive definite (positive real part of the quadratic form; self-adjointness not
    # required). Input: name for the name scope. Delegates to the private hook, which may raise a
    # not-implemented error.
    """Returns an `Op` that asserts this operator is positive definite.

    Here, positive definite means that the quadratic form `x^H A x` has positive
    real part for all nonzero `x`.  Note that we do not require the operator to
    be self-adjoint to be positive definite.

    Args:
      name:  A name to give this `Op`.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not positive definite.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._assert_positive_definite()

  def _assert_self_adjoint(self):
    # [FUNCTION DOC]
    # Default (slow) self-adjointness check. Converts the operator to a dense matrix, logs a
    # warning, and returns an assertion op that the dense matrix exactly equals its conjugate
    # transpose (message says the matrix was not equal to its adjoint).
    dense = self.to_dense()
    logging.warn(
        "Using (possibly slow) default implementation of assert_self_adjoint."
        "  Requires conversion to a dense matrix.")
    return check_ops.assert_equal(
        dense,
        linalg.adjoint(dense),
        message="Matrix was not equal to its adjoint.")

  def assert_self_adjoint(self, name="assert_self_adjoint"):
    # [FUNCTION DOC]
    # Public method returning an assertion op that fails at run time if the operator is not
    # exactly equal to its hermitian transpose. Input: name for the name scope. Delegates to the
    # private hook.
    """Returns an `Op` that asserts this operator is self-adjoint.

    Here we check that this operator is *exactly* equal to its hermitian
    transpose.

    Args:
      name:  A string name to prepend to created ops.

    Returns:
      An `Assert` `Op`, that, when run, will raise an `InvalidArgumentError` if
        the operator is not self-adjoint.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._assert_self_adjoint()

  def _check_input_dtype(self, arg):
    # [FUNCTION DOC]
    # HELPER: dtype validation used by matmul, solve and similar public methods.
    # Input: arg (a tensor-like object with a data type). Raises a type error stating expected
    # and found dtype if the argument's base dtype differs from the operator's dtype. Returns
    # nothing.
    """Check that arg.dtype == self.dtype."""
    if arg.dtype.base_dtype != self.dtype:
      raise TypeError(
          "Expected argument to have dtype %s.  Found: %s in tensor %s" %
          (self.dtype, arg.dtype, arg))

  @abc.abstractmethod
  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Abstract hook. Subclasses multiply the operator (or its adjoint) with a batch matrix x
    # (optionally adjointed) and return the product tensor. The base version raises a
    # not-implemented error.
    raise NotImplementedError("_matmul is not implemented.")

  def matmul(
      self,
      x,
      adjoint=False,
      adjoint_arg=False,
      name="matmul",
  ):
    # [FUNCTION DOC]
    # Left-multiplies: computes the product of this operator (or its adjoint) with x. Inputs: x (a
    # LinearOperator or a tensor-like batch matrix with the operator's dtype and a compatible
    # shape), adjoint (use the adjoint of this operator, default false), adjoint_arg (use the
    # adjoint of x, default false), name (name scope). If x is a LinearOperator, checks that the
    # dimensions match (raising a value error otherwise) and returns a new operator representing
    # the product through the private operator-matmul helper. Otherwise converts x to a tensor,
    # checks the dtype (type error on mismatch), statically checks the inner dimensions are
    # compatible (shape error otherwise) and returns the tensor produced by the subclass hook.
    """Transform [batch] matrix `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    X = ... # shape [..., N, R], batch matrix, R > 0.

    Y = operator.matmul(X)
    Y.shape
    ==> [..., M, R]

    Y[..., :, r] = sum_j A[..., :, j] X[j, r]
    ```

    Args:
      x: `LinearOperator` or `Tensor` with compatible shape and same `dtype` as
        `self`. See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      adjoint_arg:  Python `bool`.  If `True`, compute `A x^H` where `x^H` is
        the hermitian transpose (transposition and complex conjugation).
      name:  A name for this `Op`.

    Returns:
      A `LinearOperator` or `Tensor` with shape `[..., M, R]` and same `dtype`
        as `self`.
    """
    if isinstance(x, LinearOperator):
      left_operator = self.adjoint() if adjoint else self
      right_operator = x.adjoint() if adjoint_arg else x

      if (right_operator.range_dimension is not None and
          left_operator.domain_dimension is not None and
          right_operator.range_dimension != left_operator.domain_dimension):
        raise ValueError(
            "Operators are incompatible. Expected `x` to have dimension"
            " {} but got {}.".format(
                left_operator.domain_dimension, right_operator.range_dimension))

      with self._name_scope(name):  # pylint: disable=not-callable
        return self._linop_matmul(left_operator, right_operator)

    with self._name_scope(name):  # pylint: disable=not-callable
      x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name="x")
      self._check_input_dtype(x)

      self_dim = -2 if adjoint else -1
      arg_dim = -1 if adjoint_arg else -2
      tensor_shape.dimension_at_index(
          self.shape, self_dim).assert_is_compatible_with(
              x.shape[arg_dim])

      return self._matmul(x, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _linop_matmul(
      self, left_operator: "LinearOperator", right_operator: "LinearOperator"
  ) -> "LinearOperator":
    # [FUNCTION DOC]
    # HELPER: operator-times-operator branch of matmul.
    # Inputs: left_operator and right_operator (LinearOperators, already adjointed as requested).
    # If the right one is a plain identity operator (detected by attributes, not by class) returns
    # the left one unchanged. If the right one is a zeros operator, requires both to be square
    # (otherwise raises a value error saying this is unsupported) and returns the right one.
    # Otherwise builds a composition operator of the two, with hints: non-singular combined from
    # both when the product is square, and non-singular, self-adjoint and positive definite all
    # false when it is known to be non-square. Imports the composition module lazily to avoid a
    # cyclic import.
    # instance of linear_operator_identity.LinearOperatorIdentity
    if hasattr(right_operator, "_ones_diag") and not hasattr(
        right_operator, "multiplier"
    ):
      return left_operator

    # instance of linear_operator_zeros.LinearOperatorZeros
    elif hasattr(right_operator, "_zeros_diag"):
      if not right_operator.is_square or not left_operator.is_square:
        raise ValueError(
            "Matmul with non-square `LinearOperator`s or "
            "non-square `LinearOperatorZeros` not supported at this time."
        )
      return right_operator

    else:
      # Generic matmul of two `LinearOperator`s.
      is_square = property_hint_util.is_square(left_operator, right_operator)
      is_non_singular = None
      is_self_adjoint = None
      is_positive_definite = None

      if is_square:
        is_non_singular = property_hint_util.combined_non_singular_hint(
            left_operator, right_operator
        )
      # is_square can be None, so the explicit check for False is needed.
      elif is_square is False:  # pylint:disable=g-bool-id-comparison
        is_non_singular = False
        is_self_adjoint = False
        is_positive_definite = False

      # LinearOperator outputs a LinearOperatorComposition instance, which
      # inherits from LinearOperator. The inline import is necessary to avoid
      # errors due to this cyclic dependency.
      from tensorflow.python.ops.linalg import linear_operator_composition  # pylint: disable=g-import-not-at-top

      return linear_operator_composition.LinearOperatorComposition(
          operators=[left_operator, right_operator],
          is_non_singular=is_non_singular,
          is_self_adjoint=is_self_adjoint,
          is_positive_definite=is_positive_definite,
          is_square=is_square,
      )

  def __matmul__(self, other):
    # [FUNCTION DOC]
    # Implements the matrix-multiplication infix operator: multiplies this operator by other (a
    # tensor or another LinearOperator) with default flags, i.e. calls matmul. Returns its result
    # and raises whatever matmul raises.
    return self.matmul(other)

  def _matvec(self, x, adjoint=False):
    # [FUNCTION DOC]
    # Default hook for matvec. Inputs: x (batch vector) and adjoint (default false). Appends a
    # trailing dimension to make a column matrix, calls matmul, and removes that dimension from
    # the result. Returns the resulting batch vector.
    x_mat = array_ops.expand_dims(x, axis=-1)
    y_mat = self.matmul(x_mat, adjoint=adjoint)
    return array_ops.squeeze(y_mat, axis=-1)

  def matvec(self, x, adjoint=False, name="matvec"):
    # [FUNCTION DOC]
    # Left-multiplies a batch of vectors: computes the operator (or its adjoint) applied to x.
    # Inputs: x (tensor-like batch vector with the operator's dtype; the last dimension is the
    # vector), adjoint (default false), name (name scope). Converts x to a tensor, checks dtype
    # (type error on mismatch) and static compatibility of the last dimension with the operator
    # (shape error otherwise), and returns the tensor from the subclass hook.
    """Transform [batch] vector `x` with left multiplication:  `x --> Ax`.

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)

    X = ... # shape [..., N], batch vector

    Y = operator.matvec(X)
    Y.shape
    ==> [..., M]

    Y[..., :] = sum_j A[..., :, j] X[..., j]
    ```

    Args:
      x: `Tensor` with compatible shape and same `dtype` as `self`.
        `x` is treated as a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, left multiply by the adjoint: `A^H x`.
      name:  A name for this `Op`.

    Returns:
      A `Tensor` with shape `[..., M]` and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name="x")
      self._check_input_dtype(x)
      self_dim = -2 if adjoint else -1
      tensor_shape.dimension_at_index(
          self.shape, self_dim).assert_is_compatible_with(x.shape[-1])
      return self._matvec(x, adjoint=adjoint)

  def _determinant(self):
    # [FUNCTION DOC]
    # Default (slow) determinant hook. Logs a warning about dense conversion and cubic cost. If
    # the operator is hinted positive definite and self-adjoint, returns the exponential of the
    # log absolute determinant; otherwise returns the matrix determinant of the dense form.
    logging.warn(
        "Using (possibly slow) default implementation of determinant."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      return math_ops.exp(self.log_abs_determinant())
    return linalg_ops.matrix_determinant(self.to_dense())

  def determinant(self, name="det"):
    # [FUNCTION DOC]
    # Returns the determinant of every batch member as a tensor with the batch shape and the
    # operator's dtype. Input: name for the name scope. Raises a not-implemented error if the
    # operator is hinted not square. Delegates to the subclass hook.
    """Determinant for every batch member.

    Args:
      name:  A name for this `Op`.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    """
    if self.is_square is False:
      raise NotImplementedError(
          "Determinant not implemented for an operator that is expected to "
          "not be square.")
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._determinant()

  def _log_abs_determinant(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the warning text says "determinant" rather than "log absolute determinant"
    # (copied from the determinant hook).
    # Default (slow) hook. Logs a warning about dense conversion and cubic cost. If the operator
    # is hinted positive definite and self-adjoint, returns twice the sum of logs of the diagonal
    # of the Cholesky factor of the dense matrix; otherwise returns the log absolute determinant
    # part of the sign-and-log determinant of the dense matrix.
    logging.warn(
        "Using (possibly slow) default implementation of determinant."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    if self._can_use_cholesky():
      diag = array_ops.matrix_diag_part(linalg_ops.cholesky(self.to_dense()))
      return 2 * math_ops.reduce_sum(math_ops.log(diag), axis=[-1])
    _, log_abs_det = linalg.slogdet(self.to_dense())
    return log_abs_det

  def log_abs_determinant(self, name="log_abs_det"):
    # [FUNCTION DOC]
    # Returns the log of the absolute value of the determinant for every batch member, as a
    # tensor with the batch shape and operator's dtype. Input: name for the name scope. Raises a
    # not-implemented error if the operator is hinted not square. Delegates to the subclass hook.
    """Log absolute value of determinant for every batch member.

    Args:
      name:  A name for this `Op`.

    Returns:
      `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

    Raises:
      NotImplementedError:  If `self.is_square` is `False`.
    """
    if self.is_square is False:
      raise NotImplementedError(
          "Determinant not implemented for an operator that is expected to "
          "not be square.")
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._log_abs_determinant()

  def _dense_solve(self, rhs, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # HELPER: dense fallback used by the default solve hook (and reusable by subclasses).
    # Inputs: rhs (batch matrix), adjoint (solve with the adjoint operator, default false) and
    # adjoint_arg (use the adjoint of rhs, default false). Raises a not-implemented error if the
    # operator is hinted not square. Applies the conjugate transpose to rhs if requested, then
    # solves with a Cholesky solve when the operator is hinted positive definite and self-adjoint
    # (in that case the adjoint flag is not needed), otherwise with a broadcasting dense matrix
    # solve. Returns the solution tensor.
    """Solve by conversion to a dense matrix."""
    if self.is_square is False:  # pylint: disable=g-bool-id-comparison
      raise NotImplementedError(
          "Solve is not yet implemented for non-square operators.")
    rhs = linalg.adjoint(rhs) if adjoint_arg else rhs
    if self._can_use_cholesky():
      return linalg_ops.cholesky_solve(
          linalg_ops.cholesky(self.to_dense()), rhs)
    return linear_operator_util.matrix_solve_with_broadcast(
        self.to_dense(), rhs, adjoint=adjoint)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    # [FUNCTION DOC]
    # Default (slow) solve hook. Logs a warning about dense conversion and cubic cost, then
    # returns the result of the dense solve helper with the same arguments.
    """Default implementation of _solve."""
    logging.warn(
        "Using (possibly slow) default implementation of solve."
        "  Requires conversion to a dense matrix and O(N^3) operations.")
    return self._dense_solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def solve(self, rhs, adjoint=False, adjoint_arg=False, name="solve"):
    # [FUNCTION DOC]
    # Solves A X = rhs for a batch of systems. Inputs: rhs (a LinearOperator or tensor-like batch
    # matrix with the operator's dtype), adjoint (solve with the adjoint of this operator),
    # adjoint_arg (use the adjoint of rhs), name (name scope). Raises a not-implemented error if
    # the operator is hinted singular or not square. If rhs is an operator, checks dimension
    # compatibility (value error otherwise) and returns an operator representing the solution via
    # the private operator-solve helper. Otherwise converts rhs to a tensor, checks dtype (type
    # error) and static dimension compatibility, and returns the tensor from the subclass hook.
    """Solve (exact or approx) `R` (batch) systems of equations: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve R > 0 linear systems for every member of the batch.
    RHS = ... # shape [..., M, R]

    X = operator.solve(RHS)
    # X[..., :, r] is the solution to the r'th linear system
    # sum_j A[..., :, j] X[..., j, r] = RHS[..., :, r]

    operator.matmul(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator and compatible shape.
        `rhs` is treated like a [batch] matrix meaning for every set of leading
        dimensions, the last two dimensions defines a matrix.
        See class docstring for definition of compatibility.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      adjoint_arg:  Python `bool`.  If `True`, solve `A X = rhs^H` where `rhs^H`
        is the hermitian transpose (transposition and complex conjugation).
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    if self.is_non_singular is False:
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "be singular.")
    if self.is_square is False:
      raise NotImplementedError(
          "Exact solve not implemented for an operator that is expected to "
          "not be square.")
    if isinstance(rhs, LinearOperator):
      left_operator = self.adjoint() if adjoint else self
      right_operator = rhs.adjoint() if adjoint_arg else rhs

      if (right_operator.range_dimension is not None and
          left_operator.domain_dimension is not None and
          right_operator.range_dimension != left_operator.domain_dimension):
        raise ValueError(
            "Operators are incompatible. Expected `rhs` to have dimension"
            " {} but got {}.".format(
                left_operator.domain_dimension, right_operator.range_dimension))
      with self._name_scope(name):  # pylint: disable=not-callable
        return self._linop_solve(left_operator, right_operator)

    with self._name_scope(name):  # pylint: disable=not-callable
      rhs = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          rhs, name="rhs"
      )
      self._check_input_dtype(rhs)

      self_dim = -1 if adjoint else -2
      arg_dim = -1 if adjoint_arg else -2
      tensor_shape.dimension_at_index(
          self.shape, self_dim).assert_is_compatible_with(
              rhs.shape[arg_dim])

      return self._solve(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

  def _linop_solve(
      self, left_operator: "LinearOperator", right_operator: "LinearOperator"
    ) -> "LinearOperator":
    # [FUNCTION DOC]
    # HELPER: operator-solve branch of solve.
    # Inputs: left_operator and right_operator (LinearOperators, already adjointed as requested).
    # If the right operator is a plain identity (detected by attributes) returns the inverse of
    # the left one. Otherwise returns a composition of the inversion of the left operator with the
    # right operator, with hints computed as in the operator multiplication helper (combined
    # non-singular hint when square, all false when known non-square). Imports composition and
    # inversion modules lazily to avoid cyclic imports.
    # instance of linear_operator_identity.LinearOperatorIdentity
    if hasattr(right_operator, "_ones_diag") and not hasattr(
        right_operator, "multiplier"
    ):
      return left_operator.inverse()

    # Generic solve of two `LinearOperator`s.
    is_square = property_hint_util.is_square(left_operator, right_operator)
    is_non_singular = None
    is_self_adjoint = None
    is_positive_definite = None

    if is_square:
      is_non_singular = property_hint_util.combined_non_singular_hint(
          left_operator, right_operator
      )
    elif is_square is False:  # pylint:disable=g-bool-id-comparison
      is_non_singular = False
      is_self_adjoint = False
      is_positive_definite = False

    # LinearOperator outputs a LinearOperatorComposition instance that contains
    # a LinearOperatorInversion instance, both of which
    # inherit from LinearOperator. The inline import is necessary to avoid
    # errors due to this cyclic dependency.
    from tensorflow.python.ops.linalg import linear_operator_composition  # pylint: disable=g-import-not-at-top
    from tensorflow.python.ops.linalg import linear_operator_inversion  # pylint: disable=g-import-not-at-top

    return linear_operator_composition.LinearOperatorComposition(
        operators=[
            linear_operator_inversion.LinearOperatorInversion(left_operator),
            right_operator,
        ],
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
    )

  def _solvevec(self, rhs, adjoint=False):
    # [FUNCTION DOC]
    # Default hook for solvevec. Inputs: rhs (batch vector) and adjoint (default false). Appends a
    # trailing dimension to make a column matrix, calls solve, and removes that dimension from the
    # solution. Returns the solution batch vector.
    """Default implementation of _solvevec."""
    rhs_mat = array_ops.expand_dims(rhs, axis=-1)
    solution_mat = self.solve(rhs_mat, adjoint=adjoint)
    return array_ops.squeeze(solution_mat, axis=-1)

  def solvevec(self, rhs, adjoint=False, name="solve"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the default name scope is "solve" (not "solvevec"), and the docstring's
    # NotImplementedError is not raised here directly; the singular/non-square checks only
    # happen in the solve call made by the default hook (a subclass hook may skip them).
    # Solves A X = rhs for a batch of vectors. Inputs: rhs (tensor-like batch vector with the
    # operator's dtype; last dimension is the vector), adjoint (default false), name. Converts rhs
    # to a tensor, checks dtype (type error) and static dimension compatibility, and returns the
    # solution tensor from the subclass hook.
    """Solve single equation with best effort: `A X = rhs`.

    The returned `Tensor` will be close to an exact solution if `A` is well
    conditioned. Otherwise closeness will vary. See class docstring for details.

    Examples:

    ```python
    # Make an operator acting like batch matrix A.  Assume A.shape = [..., M, N]
    operator = LinearOperator(...)
    operator.shape = [..., M, N]

    # Solve one linear system for every member of the batch.
    RHS = ... # shape [..., M]

    X = operator.solvevec(RHS)
    # X is the solution to the linear system
    # sum_j A[..., :, j] X[..., j] = RHS[..., :]

    operator.matvec(X)
    ==> RHS
    ```

    Args:
      rhs: `Tensor` with same `dtype` as this operator.
        `rhs` is treated like a [batch] vector meaning for every set of leading
        dimensions, the last dimension defines a vector.  See class docstring
        for definition of compatibility regarding batch dimensions.
      adjoint: Python `bool`.  If `True`, solve the system involving the adjoint
        of this `LinearOperator`:  `A^H X = rhs`.
      name:  A name scope to use for ops added by this method.

    Returns:
      `Tensor` with shape `[...,N]` and same `dtype` as `rhs`.

    Raises:
      NotImplementedError:  If `self.is_non_singular` or `is_square` is False.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      rhs = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          rhs, name="rhs"
      )
      self._check_input_dtype(rhs)
      self_dim = -1 if adjoint else -2
      tensor_shape.dimension_at_index(
          self.shape, self_dim).assert_is_compatible_with(rhs.shape[-1])

      return self._solvevec(rhs, adjoint=adjoint)

  def adjoint(self, name: str = "adjoint") -> "LinearOperator":
    # [FUNCTION DOC]
    # Returns a LinearOperator representing the adjoint (conjugate transpose). Input: name for
    # the name scope. If the operator is hinted self-adjoint it returns itself; otherwise calls
    # the private adjoint hook (default: wraps in an adjoint operator). The attribute H is an
    # alias property for this method.
    """Returns the adjoint of the current `LinearOperator`.

    Given `A` representing this `LinearOperator`, return `A*`.
    Note that calling `self.adjoint()` and `self.H` are equivalent.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the adjoint of this `LinearOperator`.
    """
    if self.is_self_adjoint is True:  # pylint: disable=g-bool-id-comparison
      return self
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._linop_adjoint()

  # self.H is equivalent to self.adjoint().
  H = property(adjoint, None)

  def _linop_adjoint(self) -> "LinearOperator":
    # [FUNCTION DOC]
    # Default hook for adjoint. Lazily imports the adjoint operator module (to avoid a cyclic
    # import) and returns an adjoint operator wrapping this one, passing along this operator's
    # four hints. Subclasses override to return something more efficient.
    from tensorflow.python.ops.linalg import linear_operator_adjoint  # pylint: disable=g-import-not-at-top
    return linear_operator_adjoint.LinearOperatorAdjoint(
        self,
        is_non_singular=self.is_non_singular,
        is_self_adjoint=self.is_self_adjoint,
        is_positive_definite=self.is_positive_definite,
        is_square=self.is_square)

  def inverse(self, name: str = "inverse") -> "LinearOperator":
    # [FUNCTION DOC]
    # Returns a LinearOperator representing the inverse of this one. Input: name for the name
    # scope. Raises a value error if the operator is hinted non-square or singular; otherwise
    # returns the result of the private inverse hook.
    """Returns the Inverse of this `LinearOperator`.

    Given `A` representing this `LinearOperator`, return a `LinearOperator`
    representing `A^-1`.

    Args:
      name: A name scope to use for ops added by this method.

    Returns:
      `LinearOperator` representing inverse of this matrix.

    Raises:
      ValueError: When the `LinearOperator` is not hinted to be `non_singular`.
    """
    if self.is_square is False:  # pylint: disable=g-bool-id-comparison
      raise ValueError("Cannot take the Inverse: This operator represents "
                       "a non square matrix.")
    if self.is_non_singular is False:  # pylint: disable=g-bool-id-comparison
      raise ValueError("Cannot take the Inverse: This operator represents "
                       "a singular matrix.")

    with self._name_scope(name):  # pylint: disable=not-callable
      return self._linop_inverse()

  def _linop_inverse(self) -> "LinearOperator":
    # [FUNCTION DOC]
    # Default hook for inverse. Lazily imports the inversion operator module (cyclic dependency)
    # and returns an inversion operator wrapping this one with this operator's four hints.
    # The in-line import is necessary because linear_operator_inversion.py
    # depends on linear_operator.py. The in-line import works because the two
    # files are now in the same build target, but if the import were at the top
    # of the file there would be a partially-initialized module error caused by
    # the code cycle.
    from tensorflow.python.ops.linalg import linear_operator_inversion  # pylint: disable=g-import-not-at-top
    return linear_operator_inversion.LinearOperatorInversion(
        self,
        is_non_singular=self.is_non_singular,
        is_self_adjoint=self.is_self_adjoint,
        is_positive_definite=self.is_positive_definite,
        is_square=self.is_square)

  def cholesky(self, name: str = "cholesky") -> "LinearOperator":
    # [FUNCTION DOC]
    # Returns a LinearOperator for the lower triangular Cholesky factor of this operator. Input:
    # name for the name scope. Raises a value error unless the operator is hinted both positive
    # definite and self-adjoint. Otherwise returns the result of the private cholesky hook.
    """Returns a Cholesky factor as a `LinearOperator`.

    Given `A` representing this `LinearOperator`, if `A` is positive definite
    self-adjoint, return `L`, where `A = L L^T`, i.e. the cholesky
    decomposition.

    Args:
      name:  A name for this `Op`.

    Returns:
      `LinearOperator` which represents the lower triangular matrix
      in the Cholesky decomposition.

    Raises:
      ValueError: When the `LinearOperator` is not hinted to be positive
        definite and self adjoint.
    """

    if not self._can_use_cholesky():
      raise ValueError("Cannot take the Cholesky decomposition: "
                       "Not a positive definite self adjoint matrix.")
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._linop_cholesky()

  def _linop_cholesky(self) -> "LinearOperator":
    # [FUNCTION DOC]
    # Default hook for cholesky. Densifies the operator, computes the Cholesky factor, and wraps
    # it in a lower triangular operator hinted non-singular, not self-adjoint and square. Lazily
    # imports the lower triangular module. Materializes the full matrix.
    from tensorflow.python.ops.linalg import linear_operator_lower_triangular  # pylint: disable=g-import-not-at-top
    return linear_operator_lower_triangular.LinearOperatorLowerTriangular(
        linalg_ops.cholesky(self.to_dense()),
        is_non_singular=True,
        is_self_adjoint=False,
        is_square=True)

  def _to_dense(self):
    # [FUNCTION DOC]
    # Default (generic, often inefficient) dense conversion hook. Builds an identity matrix with
    # the operator's batch shape, domain dimension (static if known, otherwise runtime) and dtype,
    # and returns the operator applied to it by matmul, i.e. the dense batch matrix.
    """Generic and often inefficient implementation.  Override often."""
    if self.batch_shape.is_fully_defined():
      batch_shape = self.batch_shape
    else:
      batch_shape = self.batch_shape_tensor()

    dim_value = tensor_shape.dimension_value(self.domain_dimension)
    if dim_value is not None:
      n = dim_value
    else:
      n = self.domain_dimension_tensor()

    eye = linalg_ops.eye(num_rows=n, batch_shape=batch_shape, dtype=self.dtype)
    return self.matmul(eye)

  def to_dense(self, name="to_dense"):
    # [FUNCTION DOC]
    # Returns the dense (batch) matrix tensor represented by this operator. Input: name for the
    # name scope. Delegates to the subclass hook; can be memory and time expensive.
    """Return a dense (batch) matrix representing this operator."""
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._to_dense()

  def _diag_part(self):
    # [FUNCTION DOC]
    # Default (generic, inefficient) hook: returns the diagonal of the dense form of the operator.
    """Generic and often inefficient implementation.  Override often."""
    return array_ops.matrix_diag_part(self.to_dense())

  def diag_part(self, name="diag_part"):
    # [FUNCTION DOC]
    # Returns the batch diagonal as a tensor of shape batch dimensions plus the smaller of rows
    # and columns, with the operator's dtype. Input: name for the name scope. Delegates to the
    # subclass hook (default densifies).
    """Efficiently get the [batch] diagonal part of this operator.

    If this operator has shape `[B1,...,Bb, M, N]`, this returns a
    `Tensor` `diagonal`, of shape `[B1,...,Bb, min(M, N)]`, where
    `diagonal[b1,...,bb, i] = self.to_dense()[b1,...,bb, i, i]`.

    ```
    my_operator = LinearOperatorDiag([1., 2.])

    # Efficiently get the diagonal
    my_operator.diag_part()
    ==> [1., 2.]

    # Equivalent, but inefficient method
    tf.linalg.diag_part(my_operator.to_dense())
    ==> [1., 2.]
    ```

    Args:
      name:  A name for this `Op`.

    Returns:
      diag_part:  A `Tensor` of same `dtype` as self.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._diag_part()

  def _trace(self):
    # [FUNCTION DOC]
    # Default hook: returns the sum of the operator's diagonal part over its last axis.
    return math_ops.reduce_sum(self.diag_part(), axis=-1)

  def trace(self, name="trace"):
    # [FUNCTION DOC]
    # Returns the trace (sum of the diagonal) of every batch member as a tensor of the batch shape
    # and operator's dtype. Input: name for the name scope. Delegates to the subclass hook.
    """Trace of the linear operator, equal to sum of `self.diag_part()`.

    If the operator is square, this is also the sum of the eigenvalues.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._trace()

  def _add_to_tensor(self, x):
    # [FUNCTION DOC]
    # Default hook: returns the dense form of the operator plus the tensor x (broadcasting).
    # Subclasses override when a cheaper way exists.
    # Override if a more efficient implementation is available.
    return self.to_dense() + x

  def add_to_tensor(self, x, name="add_to_tensor"):
    # [FUNCTION DOC]
    # Adds the matrix represented by this operator to the tensor x. Inputs: x (tensor with the
    # same dtype and a shape broadcastable to the operator's shape), name (name scope). Returns a
    # dense tensor with the broadcast shape; delegates to the subclass hook.
    """Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

    Args:
      x:  `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
      name:  A name to give this `Op`.

    Returns:
      A `Tensor` with broadcast shape and same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      x = tensor_conversion.convert_to_tensor_v2_with_dispatch(x, name="x")
      self._check_input_dtype(x)
      return self._add_to_tensor(x)

  def _eigvals(self):
    # [FUNCTION DOC]
    # Default hook: returns the eigenvalues of the dense form of the operator, computed with the
    # self-adjoint eigenvalue routine (so it is only meaningful for self-adjoint operators).
    return linalg_ops.self_adjoint_eigvals(self.to_dense())

  def eigvals(self, name="eigvals"):
    # [FUNCTION DOC]
    # Returns the eigenvalues of every batch member as a tensor of shape batch dimensions plus
    # the matrix size. Input: name for the name scope. Raises a not-implemented error unless the
    # operator is hinted self-adjoint. Delegates to the subclass hook.
    """Returns the eigenvalues of this linear operator.

    If the operator is marked as self-adjoint (via `is_self_adjoint`)
    this computation can be more efficient.

    Note: This currently only supports self-adjoint operators.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb, N]` `Tensor` of same `dtype` as `self`.
    """
    if not self.is_self_adjoint:
      raise NotImplementedError("Only self-adjoint matrices are supported.")
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._eigvals()

  def _cond(self):
    # [FUNCTION DOC]
    # Default hook for the condition number. Uses singular values of the dense matrix when the
    # operator is not hinted self-adjoint, otherwise the absolute values of the eigenvalues. Returns
    # the ratio of the largest to the smallest of those values, per batch member.
    if not self.is_self_adjoint:
      # In general the condition number is the ratio of the
      # absolute value of the largest and smallest singular values.
      vals = linalg_ops.svd(self.to_dense(), compute_uv=False)
    else:
      # For self-adjoint matrices, and in general normal matrices,
      # we can use eigenvalues.
      vals = math_ops.abs(self._eigvals())

    return (math_ops.reduce_max(vals, axis=-1) /
            math_ops.reduce_min(vals, axis=-1))

  def cond(self, name="cond"):
    # [FUNCTION DOC]
    # Returns the condition number of every batch member as a tensor of the batch shape. Input:
    # name for the name scope. Delegates to the subclass hook.
    """Returns the condition number of this linear operator.

    Args:
      name:  A name for this `Op`.

    Returns:
      Shape `[B1,...,Bb]` `Tensor` of same `dtype` as `self`.
    """
    with self._name_scope(name):  # pylint: disable=not-callable
      return self._cond()

  def _can_use_cholesky(self):
    # [FUNCTION DOC]
    # HELPER: predicate used by the default determinant, solve and assertion code.
    # Returns truthy when the operator is hinted both self-adjoint and positive definite (the
    # result may be null or false when either hint is unknown or false).
    return self.is_self_adjoint and self.is_positive_definite

  def _set_graph_parents(self, graph_parents):
    # [FUNCTION DOC]
    # HELPER: deprecated graph-parent setter for subclasses and the constructor.
    # Input: graph_parents (iterable of tensors, or null meaning empty). Raises a value error if
    # any item is null or neither a variable reference nor a tensor-like object. Stores the
    # iterable as the operator's graph parents. Returns nothing.
    """Set self._graph_parents.  Called during derived class init.

    This method allows derived classes to set graph_parents, without triggering
    a deprecation warning (which is invoked if `graph_parents` is passed during
    `__init__`.

    Args:
      graph_parents: Iterable over Tensors.
    """
    # TODO(b/143910018) Remove this function in V3.
    graph_parents = [] if graph_parents is None else graph_parents
    for i, t in enumerate(graph_parents):
      if t is None or not (linear_operator_util.is_ref(t) or
                           tensor_util.is_tf_type(t)):
        raise ValueError("Graph parent item %d is not a Tensor; %s." % (i, t))
    self._graph_parents = graph_parents

  @property
  def _composite_tensor_fields(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor support hook. Property returning the names of constructor
    # arguments needed to rebuild the operator; the base version returns an empty tuple and
    # subclasses override it.
    """A tuple of parameter names to rebuild the `LinearOperator`.

    The tuple contains the names of kwargs to the `LinearOperator`'s constructor
    that the `TypeSpec` needs to rebuild the `LinearOperator` instance.

    "is_non_singular", "is_self_adjoint", "is_positive_definite", and
    "is_square" are common to all `LinearOperator` subclasses and may be
    omitted.
    """
    return ()

  @property
  def _composite_tensor_prefer_static_fields(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor support hook. Property returning the names of tensor-like
    # constructor arguments (a subset of the composite fields) that may be stored as static
    # values when statically known; the base version returns an empty tuple.
    """A tuple of names referring to parameters that may be treated statically.

    This is a subset of `_composite_tensor_fields`, and contains the names of
    of `Tensor`-like args to the `LinearOperator`s constructor that may be
    stored as static values, if they are statically known. These are typically
    shapes or axis values.
    """
    return ()

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # HELPER: placeholder property so the abstract composite tensor base class is satisfied.
    # Does nothing and returns null; the composite-tensor class decorator replaces it on each
    # decorated subclass with a property that builds the operator's type specification.
    # This property will be overwritten by the `@make_composite_tensor`
    # decorator. However, we need it so that a valid subclass of the `ABCMeta`
    # class `CompositeTensor` can be constructed and passed to the
    # `@make_composite_tensor` decorator.
    pass

  def _convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # HELPER: composite-tensor support. Extracts the operator's components through its type
    # specification, converts any variables inside them to plain tensors, and rebuilds and
    # returns a new operator from those components. The original is not modified.
    """Recursively converts ResourceVariables in the LinearOperator to Tensors.

    The usage of `self._type_spec._from_components` violates the contract of
    `CompositeTensor`, since it is called on a different nested structure
    (one containing only `Tensor`s) than `self.type_spec` specifies (one that
    may contain `ResourceVariable`s). Since `LinearOperator`'s
    `_from_components` method just passes the contents of the nested structure
    to `__init__` to rebuild the operator, and any `LinearOperator` that may be
    instantiated with `ResourceVariables` may also be instantiated with
    `Tensor`s, this usage is valid.

    Returns:
      tensor_operator: `self` with all internal Variables converted to Tensors.
    """
    # pylint: disable=protected-access
    components = self._type_spec._to_components(self)
    tensor_components = variable_utils.convert_variables_to_tensors(
        components)
    return self._type_spec._from_components(tensor_components)
    # pylint: enable=protected-access

  def __getitem__(self, slices):
    # [FUNCTION DOC]
    # Indexing support. Input: slices (an index expression applying to the batch dimensions).
    # Returns a new operator with the batch slice applied, built by the shared slicing helper
    # with no parameter overrides. Errors come from that helper (for example, for operators that
    # do not support batch slicing).
    return slicing.batch_slice(self, params_overrides={}, slices=slices)

  @property
  def _experimental_parameter_ndims_to_matrix_ndims(self):
    # [FUNCTION DOC]
    # HELPER: batch-slicing support hook.
    # KNOWN ISSUE: the docstring describes a dictionary but the base version returns an empty
    # tuple; subclasses override it with a real mapping.
    # Property mapping constructor argument names to the number of right-most dimensions
    # contributing to the matrix shape.
    """A dict of names to number of dimensions contributing to an operator.

    This is a dictionary of parameter names to `int`s specifying the
    number of right-most dimensions contributing to the **matrix** shape of the
    densified operator.
    If the parameter is a `Tensor`, this is mapped to an `int`.
    If the parameter is a `LinearOperator` (called `A`), this specifies the
    number of batch dimensions of `A` contributing to this `LinearOperator`s
    matrix shape.
    If the parameter is a structure, this is a structure of the same type of
    `int`s.
    """
    return ()

  __composite_gradient__ = _LinearOperatorGradient()


class _LinearOperatorSpec(type_spec.BatchableTypeSpec):
  # [CLASS DOC]
  # HELPER: base type specification for composite-tensor LinearOperators.
  # Holds three pieces of state: a mapping of parameter names to type specs of tensor-like
  # constructor arguments, a mapping of names to non-tensor constructor arguments, and a tuple of
  # names that prefer static storage. Concrete specs are created per operator class by the
  # composite-tensor decorator, which sets the value type. Supports batching and unbatching,
  # decomposition into components and reconstruction by calling the operator constructor.
  """A tf.TypeSpec for `LinearOperator` objects."""

  __slots__ = ("_param_specs", "_non_tensor_params", "_prefer_static_fields")

  def __init__(self, param_specs, non_tensor_params, prefer_static_fields):
    # [FUNCTION DOC]
    # Stores the three constructor-argument groups: param_specs (mapping of names to type specs
    # of tensor-like arguments), non_tensor_params (mapping of names to plain values) and
    # prefer_static_fields (tuple of names that may be stored statically). Returns nothing.
    """Initializes a new `_LinearOperatorSpec`.

    Args:
      param_specs: Python `dict` of `tf.TypeSpec` instances that describe
        kwargs to the `LinearOperator`'s constructor that are `Tensor`-like or
        `CompositeTensor` subclasses.
      non_tensor_params: Python `dict` containing non-`Tensor` and non-
        `CompositeTensor` kwargs to the `LinearOperator`'s constructor.
      prefer_static_fields: Python `tuple` of strings corresponding to the names
        of `Tensor`-like args to the `LinearOperator`s constructor that may be
        stored as static values, if known. These are typically shapes, indices,
        or axis values.
    """
    self._param_specs = param_specs
    self._non_tensor_params = non_tensor_params
    self._prefer_static_fields = prefer_static_fields

  @classmethod
  def from_operator(cls, operator):
    # [FUNCTION DOC]
    # Class method building a spec from an operator instance. Extracts the operator's composite
    # fields plus the four hint fields as constructor arguments, converts each into type specs
    # (recursively through collections). Arguments that are entirely tensor-like go to the
    # parameter specs, entirely non-tensor ones go to the plain parameters, and a mix inside one
    # argument raises a not-implemented error. Returns a new spec carrying the operator's
    # prefer-static field names.
    """Builds a `_LinearOperatorSpec` from a `LinearOperator` instance.

    Args:
      operator: An instance of `LinearOperator`.

    Returns:
      linear_operator_spec: An instance of `_LinearOperatorSpec` to be used as
        the `TypeSpec` of `operator`.
    """
    validation_fields = ("is_non_singular", "is_self_adjoint",
                         "is_positive_definite", "is_square")
    kwargs = _extract_attrs(
        operator,
        keys=set(operator._composite_tensor_fields + validation_fields))  # pylint: disable=protected-access

    non_tensor_params = {}
    param_specs = {}
    for k, v in list(kwargs.items()):
      type_spec_or_v = _extract_type_spec_recursively(v)
      is_tensor = [isinstance(x, type_spec.TypeSpec)
                   for x in nest.flatten(type_spec_or_v)]
      if all(is_tensor):
        param_specs[k] = type_spec_or_v
      elif not any(is_tensor):
        non_tensor_params[k] = v
      else:
        raise NotImplementedError(f"Field {k} contains a mix of `Tensor` and "
                                  f" non-`Tensor` values.")

    return cls(
        param_specs=param_specs,
        non_tensor_params=non_tensor_params,
        prefer_static_fields=operator._composite_tensor_prefer_static_fields)  # pylint: disable=protected-access

  def _to_components(self, obj):
    # [FUNCTION DOC]
    # Type-spec hook: returns a mapping of the tensor-like constructor argument names (those in
    # the parameter specs) to their values extracted from the given operator.
    return _extract_attrs(obj, keys=list(self._param_specs))

  def _from_components(self, components):
    # [FUNCTION DOC]
    # Type-spec hook: rebuilds an operator by calling the value type's constructor with the
    # stored non-tensor parameters merged with the given components mapping (components win on
    # name clashes). Returns the new operator.
    kwargs = dict(self._non_tensor_params, **components)
    return self.value_type(**kwargs)

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Type-spec hook (property): returns the mapping of tensor-like constructor argument names to
    # their type specs.
    return self._param_specs

  def _serialize(self):
    # [FUNCTION DOC]
    # Type-spec hook: returns a tuple of the parameter specs, non-tensor parameters and
    # prefer-static field names, which fully describes the spec.
    return (self._param_specs,
            self._non_tensor_params,
            self._prefer_static_fields)

  def _copy(self, **overrides):
    # [FUNCTION DOC]
    # HELPER: returns a new spec of the same class with the same three fields, where any keyword
    # overrides (using the constructor argument names) replace the originals. Does not mutate
    # this spec.
    kwargs = {
        "param_specs": self._param_specs,
        "non_tensor_params": self._non_tensor_params,
        "prefer_static_fields": self._prefer_static_fields
    }
    kwargs.update(overrides)
    return type(self)(**kwargs)

  def _batch(self, batch_size):
    # [FUNCTION DOC]
    # Type-spec hook: returns a copy of this spec in which every parameter spec has been batched
    # with the given batch size (an integer or unknown).
    """Returns a TypeSpec representing a batch of objects with this TypeSpec."""
    return self._copy(
        param_specs=nest.map_structure(
            lambda spec: spec._batch(batch_size),  # pylint: disable=protected-access
            self._param_specs))

  def _unbatch(self, batch_size):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the batch_size argument is accepted but never used.
    # Type-spec hook: returns a copy of this spec in which every parameter spec has been
    # unbatched (describing a single element).
    """Returns a TypeSpec representing a single element of this TypeSpec."""
    return self._copy(
        param_specs=nest.map_structure(
            lambda spec: spec._unbatch(),  # pylint: disable=protected-access
            self._param_specs))


def make_composite_tensor(cls, module_name="tf.linalg"):
  # [FUNCTION DOC]
  # Class decorator turning a LinearOperator subclass into a composite tensor. Inputs: cls (the
  # operator class) and module_name (prefix used for the registered type name, default
  # "tf.linalg"). Creates a new spec subclass named after the class, with the class as its value
  # type, registers it in the global type-spec registry (side effect, a duplicate name would be
  # rejected by the registry), and sets the class's type-spec attribute to a property building the
  # spec from an instance. Returns the same class, modified in place.
  """Class decorator to convert `LinearOperator`s to `CompositeTensor`."""

  spec_name = "{}Spec".format(cls.__name__)
  spec_type = type(spec_name, (_LinearOperatorSpec,), {"value_type": cls})
  type_spec_registry.register("{}.{}".format(module_name, spec_name))(spec_type)
  cls._type_spec = property(spec_type.from_operator)  # pylint: disable=protected-access
  return cls


def _extract_attrs(op, keys):
  # [FUNCTION DOC]
  # HELPER: used by the type specification to read constructor arguments off an operator.
  # Inputs: op (a LinearOperator) and keys (names of constructor arguments). For each name uses
  # the first value found among a public attribute, a private underscore attribute, or the
  # operator's recorded parameters; raises a value error if none exists. For names the operator
  # prefers to store statically, a tensor whose value is statically known is replaced by that
  # value. Numpy arrays and scalars are converted to plain lists or numbers. Returns a mapping of
  # names to values.
  """Extract constructor kwargs to reconstruct `op`.

  Args:
    op: A `LinearOperator` instance.
    keys: A Python `tuple` of strings indicating the names of the constructor
      kwargs to extract from `op`.

  Returns:
    kwargs: A Python `dict` of kwargs to `op`'s constructor, keyed by `keys`.
  """

  kwargs = {}
  not_found = object()
  for k in keys:
    srcs = [
        getattr(op, k, not_found), getattr(op, "_" + k, not_found),
        getattr(op, "parameters", {}).get(k, not_found),
    ]
    if any(v is not not_found for v in srcs):
      kwargs[k] = [v for v in srcs if v is not not_found][0]
    else:
      raise ValueError(
          f"Could not determine an appropriate value for field `{k}` in object "
          f" `{op}`. Looked for \n"
          f" 1. an attr called `{k}`,\n"
          f" 2. an attr called `_{k}`,\n"
          f" 3. an entry in `op.parameters` with key '{k}'.")
    if k in op._composite_tensor_prefer_static_fields and kwargs[k] is not None:  # pylint: disable=protected-access
      if tensor_util.is_tensor(kwargs[k]):
        static_val = tensor_util.constant_value(kwargs[k])
        if static_val is not None:
          kwargs[k] = static_val
    if isinstance(kwargs[k], (np.ndarray, np.generic)):
      kwargs[k] = kwargs[k].tolist()
  return kwargs


def _extract_type_spec_recursively(value):
  # [FUNCTION DOC]
  # HELPER: used by the type specification builder.
  # Input: value (any object). Composite tensors give their type spec, variables give a variable
  # spec, tensors give a tensor spec; lists, tuples and dictionaries (and trackable wrappers of
  # them) are processed element by element preserving structure; anything else is returned
  # unchanged. Returns the spec or a parallel structure. No side effects.
  """Return (collection of) `TypeSpec`(s) for `value` if it includes `Tensor`s.

  If `value` is a `Tensor` or `CompositeTensor`, return its `TypeSpec`. If
  `value` is a collection containing `Tensor` values, recursively supplant them
  with their respective `TypeSpec`s in a collection of parallel stucture.

  If `value` is none of the above, return it unchanged.

  Args:
    value: a Python `object` to (possibly) turn into a (collection of)
    `tf.TypeSpec`(s).

  Returns:
    spec: the `TypeSpec` or collection of `TypeSpec`s corresponding to `value`
    or `value`, if no `Tensor`s are found.
  """
  if isinstance(value, composite_tensor.CompositeTensor):
    return value._type_spec  # pylint: disable=protected-access
  if isinstance(value, variables.Variable):
    return resource_variable_ops.VariableSpec(
        value.shape, dtype=value.dtype, trainable=value.trainable)
  if tensor_util.is_tensor(value):
    return tensor_spec.TensorSpec(value.shape, value.dtype)
  # Unwrap trackable data structures to comply with `Type_Spec._serialize`
  # requirements. `ListWrapper`s are converted to `list`s, and for other
  # trackable data structures, the `__wrapped__` attribute is used.
  if isinstance(value, list):
    return list(_extract_type_spec_recursively(v) for v in value)
  if isinstance(value, data_structures.TrackableDataStructure):
    return _extract_type_spec_recursively(value.__wrapped__)
  if isinstance(value, tuple):
    return type(value)(_extract_type_spec_recursively(x) for x in value)
  if isinstance(value, dict):
    return type(value)((k, _extract_type_spec_recursively(v))
                       for k, v in value.items())
  return value


# Overrides for tf.linalg functions. This allows a LinearOperator to be used in
# place of a Tensor.
# For instance tf.trace(linop) and linop.trace() both work.


@dispatch.dispatch_for_types(linalg.adjoint, LinearOperator)
def _adjoint(matrix, name=None):
  # [FUNCTION DOC]
  # HELPER: dispatch override so the public linalg adjoint function accepts a LinearOperator.
  # Returns the operator's adjoint operator, using name as the name scope.
  return matrix.adjoint(name)


@dispatch.dispatch_for_types(linalg.cholesky, LinearOperator)
def _cholesky(input, name=None):   # pylint:disable=redefined-builtin
  # [FUNCTION DOC]
  # HELPER: dispatch override so the public linalg cholesky function accepts a LinearOperator.
  # Returns the operator's Cholesky factor operator; raises a value error if the operator is not
  # hinted positive definite and self-adjoint.
  return input.cholesky(name)


# The signature has to match with the one in python/op/array_ops.py,
# so we have k, padding_value, and align even though we don't use them here.
# pylint:disable=unused-argument
@dispatch.dispatch_for_types(linalg.diag_part, LinearOperator)
def _diag_part(
    input,  # pylint:disable=redefined-builtin
    name="diag_part",
    k=0,
    padding_value=0,
    align="RIGHT_LEFT"):
  # [FUNCTION DOC]
  # HELPER: dispatch override so the public linalg diag-part function accepts a LinearOperator.
  # KNOWN ISSUE: the arguments k, padding_value and align are accepted only to match the
  # signature and are silently ignored, so non-default values have no effect.
  # Returns the operator's batch diagonal, using name as the name scope.
  return input.diag_part(name)
# pylint:enable=unused-argument


@dispatch.dispatch_for_types(linalg.det, LinearOperator)
def _det(input, name=None):  # pylint:disable=redefined-builtin
  # [FUNCTION DOC]
  # HELPER: dispatch override so the public linalg determinant function accepts a
  # LinearOperator. Returns the operator's determinant tensor, using name as the name scope.
  return input.determinant(name)


@dispatch.dispatch_for_types(linalg.inv, LinearOperator)
def _inverse(input, adjoint=False, name=None):   # pylint:disable=redefined-builtin
  # [FUNCTION DOC]
  # HELPER: dispatch override so the public linalg inverse function accepts a LinearOperator.
  # Returns the inverse operator (value error if the operator is non-square or singular), and if
  # adjoint is true returns the adjoint of that inverse. Name is used as the name scope.
  inv = input.inverse(name)
  if adjoint:
    inv = inv.adjoint()
  return inv


@dispatch.dispatch_for_types(linalg.logdet, LinearOperator)
def _logdet(matrix, name=None):
  # [FUNCTION DOC]
  # HELPER: dispatch override so the public linalg log-determinant function accepts a
  # LinearOperator. If the operator is hinted positive definite and self-adjoint, returns its log
  # absolute determinant (equal to the log determinant in that case); otherwise raises a value
  # error saying a self-adjoint positive definite matrix was expected.
  if matrix.is_positive_definite and matrix.is_self_adjoint:
    return matrix.log_abs_determinant(name)
  raise ValueError("Expected matrix to be self-adjoint positive definite.")


@dispatch.dispatch_for_types(math_ops.matmul, LinearOperator)
def _matmul(  # pylint:disable=missing-docstring
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    adjoint_a=False,
    adjoint_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    output_type=None,  # pylint: disable=unused-argument
    grad_a=False,  # pylint: disable=unused-argument
    grad_b=False,  # pylint: disable=unused-argument
    name=None,
):
  # [FUNCTION DOC]
  # HELPER: dispatch override so the public matrix-multiply function accepts LinearOperators.
  # Inputs follow the public function: a and b (either may be an operator), transpose flags,
  # adjoint flags, sparsity flags, an output type and gradient flags (the last two and the
  # unused-argument markers are ignored), and name. Raises a value error if either plain
  # transpose or either sparse flag is set. If a is an operator, returns its matmul with b using
  # the adjoint flags. If only b is an operator, uses the identity that the product equals the
  # adjoint of the product of adjoints: multiplies b by a with flipped adjoint flags and returns
  # the adjoint of the result.
  if transpose_a or transpose_b:
    raise ValueError("Transposing not supported at this time.")
  if a_is_sparse or b_is_sparse:
    raise ValueError("Sparse methods not supported at this time.")
  if not isinstance(a, LinearOperator):
    # We use the identity (B^HA^H)^H =  AB
    adjoint_matmul = b.matmul(
        a,
        adjoint=(not adjoint_b),
        adjoint_arg=(not adjoint_a),
        name=name)
    return linalg.adjoint(adjoint_matmul)
  return a.matmul(
      b, adjoint=adjoint_a, adjoint_arg=adjoint_b, name=name)


@dispatch.dispatch_for_types(linalg.solve, LinearOperator)
def _solve(
    matrix,
    rhs,
    adjoint=False,
    name=None):
  # [FUNCTION DOC]
  # HELPER: dispatch override so the public linalg solve function accepts LinearOperators.
  # Inputs: matrix, rhs, adjoint (default false) and name. Raises a value error if matrix is a
  # plain tensor while rhs is an operator (unsupported). Otherwise returns the operator's solve
  # of rhs with the adjoint flag and name.
  if not isinstance(matrix, LinearOperator):
    raise ValueError("Passing in `matrix` as a Tensor and `rhs` as a "
                     "LinearOperator is not supported.")
  return matrix.solve(rhs, adjoint=adjoint, name=name)


@dispatch.dispatch_for_types(linalg.trace, LinearOperator)
def _trace(x, name=None):
  # [FUNCTION DOC]
  # HELPER: dispatch override so the public linalg trace function accepts a LinearOperator.
  # Returns the operator's trace tensor, using name as the name scope.
  return x.trace(name)
