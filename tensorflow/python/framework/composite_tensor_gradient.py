# [MODULE DOC]
# Gradient support for composite tensors. Defines an abstract gradient-component interface
# (which parts of a composite value take part in differentiation, and how to substitute
# gradients back), a runtime-checkable protocol marking objects that carry such an interface,
# a values-based default implementation, and internal and public helpers that flatten the
# differentiable tensors of a list of values and rebuild values from flat gradients. Used by the
# gradient machinery. Import-time side effect: selects typing protocol utilities from the
# standard library or the typing extensions package depending on the Python version.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Gradient support for Composite Tensors."""

import abc
import sys

from tensorflow.python.framework import composite_tensor
from tensorflow.python.util import nest


# pylint:disable=g-import-not-at-top
if sys.version_info >= (3, 8):
  from typing import Protocol
  from typing import runtime_checkable
else:
  from typing_extensions import Protocol
  from typing_extensions import runtime_checkable
# pylint:enable=g-import-not-at-top


# TODO(xjun): Add CompositeTensorGradient support for SparseTensor,
# StructuredTensor, and MaskedTensor.
class CompositeTensorGradient(object, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract interface that a composite tensor type exposes to take part in gradient
  # computation. Implementations say which components of a value are differentiated and how to
  # build a copy of the value with those components replaced by gradients. Stateless; concrete
  # subclasses are attached to composite tensor types and used by the helper functions in this
  # module.
  """Class used to help compute gradients for CompositeTensors.

  This abstract base class defines two methods: `get_gradient_components`, which
  returns the components of a value that should be included in gradients; and
  `replace_gradient_components`, which replaces the gradient components in a
  value.  These methods can be used to compute the gradient of a `y` with
  respect to `x` (`grad(y, x)`) as follows:

  * If `y` is a `CompositeTensor` with `CompositeTensorGradient` `cg` =
    `y.__composite_gradient__`, then `grad(y, x)` =
    `grad(cg.get_gradient_components(y), x)`.

  * If `x` is a `CompositeTensor` with `CompositeTensorGradient` `cg` =
    'x.__composite_gradient__', then `grad(y, x)` =
    `cg.replace_gradient_components(x, grad(y, cg.get_gradient_components(x))`.
  """

  @abc.abstractmethod
  def get_gradient_components(self, value):
    # [FUNCTION DOC]
    # Abstract method returning the components of a composite value that should be included in
    # gradients. Input: value, a composite tensor. Output: a nested structure of tensors or
    # sparse-slice objects. Implementations must not create new graph operations. The base
    # version raises a not-implemented error naming the concrete class.
    """Returns the components of `value` that should be included in gradients.

    This method may not call TensorFlow ops, since any new ops added to the
    graph would not be properly tracked by the gradient mechanisms.

    Args:
      value: A `CompositeTensor` value.

    Returns:
      A nested structure of `Tensor` or `IndexedSlices`.
    """
    raise NotImplementedError(
        f"{type(self).__name__}.get_gradient_components()")

  @abc.abstractmethod
  def replace_gradient_components(self, value, component_grads):
    # [FUNCTION DOC]
    # Abstract method building a copy of a composite value in which its gradient components are
    # replaced. Inputs: value, a composite value whose gradient components are compatible with
    # component_grads; component_grads, a nested structure of tensors, sparse-slice objects, or
    # nothing for unconnected gradients. Output: the new value, or nothing if the gradients
    # contain nothing. The base version raises a not-implemented error naming the concrete class.
    """Replaces the gradient components in `value` with `component_grads`.

    Args:
      value: A value with its gradient components compatible with
        `component_grads`.
      component_grads: A nested structure of `Tensor` or `IndexedSlices` or
        `None` (for unconnected gradients).

    Returns:
      A copy of `value`, where the components that should be included in
      gradients have been replaced by `component_grads`; or `None` (if
      `component_grads` includes `None`).
    """
    raise NotImplementedError(
        f"{type(self).__name__}.replace_gradient_components()")


@runtime_checkable
class CompositeTensorGradientProtocol(Protocol):
  # [CLASS DOC]
  # Runtime-checkable structural protocol: any object that has a composite gradient attribute
  # holding a gradient-interface object satisfies it. Used with instance checks to decide
  # whether a composite tensor supports gradients. It has no methods and is not instantiated
  # directly.
  """Protocol for adding gradient support to CompositeTensors."""
  __composite_gradient__: CompositeTensorGradient


class WithValuesCompositeTensorGradient(CompositeTensorGradient):
  # [CLASS DOC]
  # Concrete gradient interface for composite types that expose a values component and can
  # produce a copy with new values. The values component is the only differentiable part.
  # Stateless; assigned as the gradient interface of such composite types.
  """CompositeTensorGradient based on `T.values` and `T.with_values`."""

  def get_gradient_components(self, value):
    # [FUNCTION DOC]
    # Returns the values component of the given composite value as its gradient component.
    # Input: value, an object with a values attribute. Raises an attribute error if missing.
    # No side effects.
    return value.values

  def replace_gradient_components(self, value, component_grads):
    # [FUNCTION DOC]
    # Returns a copy of the value whose values component is replaced by component_grads.
    # Inputs: value, an object providing a with-values method; component_grads, the replacement
    # values. Output is whatever that method returns. No side effects of its own.
    return value.with_values(component_grads)


def _get_tensors_for_gradient(x):
  # [FUNCTION DOC]
  # HELPER: internal helper for the gradient flattening function in this module.
  # Returns the tensors of x that should be differentiated. Input: x, a tensor or composite
  # tensor. Non-composite values are returned unchanged. A composite value must satisfy the
  # gradient protocol, otherwise a value error is raised saying the type is unsupported as a
  # gradient source or target. Gets the gradient components from its gradient interface; if they
  # are the value itself, returns it, else recursively maps this function over the components
  # (so nested composites are expanded). Returns a tensor or nested structure of tensors.
  """Returns the Tensors in `x` that should be differentiated.

  Args:
    x: A `Tensor` or `CompositeTensor`.

  Returns:
    A `Tensor` or a nested structure of `Tensor`.
  """
  if not isinstance(x, composite_tensor.CompositeTensor):
    return x

  if not isinstance(x, CompositeTensorGradientProtocol):
    raise ValueError(
        f"Type {type(x).__name__} is not supported as a gradient source or "
        "gradient target.")
  composite_gradient = x.__composite_gradient__
  gradient_components = composite_gradient.get_gradient_components(x)
  if gradient_components is x:
    return x
  return nest.map_structure(_get_tensors_for_gradient, gradient_components)


def _replace_tensors_for_gradient(x, grad):
  # [FUNCTION DOC]
  # HELPER: internal helper for the gradient rebuilding function in this module.
  # Inverse of the tensor extraction helper: substitutes gradients into x. Inputs: x, a tensor or
  # composite tensor; grad, a nested structure of tensors shaped like the extraction result for x.
  # Non-composite x returns grad directly. A composite x must satisfy the gradient protocol, else
  # a value error (unsupported as gradient source) is raised. Gradient components are matched with
  # grad (recursively, up to the component structure); if the combined gradients are nothing,
  # returns nothing; otherwise returns the value rebuilt by the gradient interface.
  """Replaces the tensors in `x` that should be differentiated with `grad`.

  Args:
    x: A `Tensor` or `CompositeTensor`.
    grad: A nested structure of `Tensor`, with the same structure as the value
      returned by `_get_tensors_for_gradient(x)`.

  Returns:
    A `Tensor` or `CompositeTensor`.
  """
  if not isinstance(x, composite_tensor.CompositeTensor):
    return grad

  if not isinstance(x, CompositeTensorGradientProtocol):
    raise ValueError(
        f"Type {type(x).__name__} is not supported as a gradient source.")

  composite_gradient = x.__composite_gradient__
  x_components = composite_gradient.get_gradient_components(x)
  if x_components is x:
    grad_components = grad
  else:
    grad_components = nest.map_structure_up_to(x_components,
                                               _replace_tensors_for_gradient,
                                               x_components, grad)
  if grad_components is None:
    return None
  return composite_gradient.replace_gradient_components(x, grad_components)


def get_flat_tensors_for_gradients(xs):
  # [FUNCTION DOC]
  # Returns a single flat list of the tensors to differentiate for a list of values. Input: xs,
  # a list of tensors and composite tensors. Plain tensors are kept; composites are replaced by
  # their gradient components (expanded recursively). Raises a value error for composites that
  # do not support gradients. No side effects.
  """Returns a flat list of Tensors that should be differentiated for `xs`.

  Args:
    xs: A list of `Tensor`s or `CompositeTensor`s.

  Returns:
    A flat list of `Tensor`s constructed from `xs`, where `Tensor` values are
    left as-is, and `CompositeTensor`s are replaced with
    `_get_tensors_for_gradient(x)`.
  """
  return nest.flatten([_get_tensors_for_gradient(x) for x in xs])


def replace_flat_tensors_for_gradients(xs, flat_grads):
  # [FUNCTION DOC]
  # Inverse of the flat extraction function. Inputs: xs, the original list of tensors and
  # composite tensors; flat_grads, a flat list of gradient tensors in the same order the
  # extraction function would produce for xs. Packs the flat list into the structure of the
  # differentiable tensors, then substitutes each piece back into the matching value. Returns a
  # list of tensors or composite tensors (an entry may be nothing for unconnected gradients).
  # Raises a value error for unsupported composites. No side effects.
  """Replaces Tensors that should be differentiated in `xs` with `flat_grads`.

  Args:
    xs: A list of `Tensor`s or `CompositeTensor`s.
    flat_grads: A list of `Tensor`.

  Returns:
    A list of `Tensor` or `CompositeTensor`.
  """
  xs_structure = [_get_tensors_for_gradient(x) for x in xs]
  grads = nest.pack_sequence_as(xs_structure, flat_grads)
  return [_replace_tensors_for_gradient(x, grad) for x, grad in zip(xs, grads)]
