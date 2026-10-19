# [MODULE DOC]
# Defines the weakly typed tensor wrapper types used for TensorFlow's implicit dtype promotion.
# A WeakTensor wraps an ordinary tensor whose dtype was only inferred (for example from a Python
# number) and may therefore be overridden by the dtype of the other operand in an operation.
# Exposes WeakTensor (an extension type that is also a core tensor), its eager and graph flavors
# (EagerWeakTensor and GraphWeakTensor), a gradient helper class, a small iterator class, and
# helper functions to convert between weak and normal tensors.
# Only a restricted set of dtypes (32/64-bit integers, 32/64-bit floats, 128-bit complex) may be
# weak.
# Import-time side effect: registers a tensor conversion function so that converting a WeakTensor
# to a tensor yields the plain wrapped tensor.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""An extension type that represents WeakTensor."""

from typing import Optional

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.types import core


_ALLOWED_WEAK_DTYPES = (
    dtypes.int32,
    dtypes.int64,
    dtypes.float32,
    dtypes.float64,
    dtypes.complex128,
)


class WeakTensorGradient(composite_tensor_gradient.CompositeTensorGradient):
  # [CLASS DOC]
  # HELPER: lets gradient code treat a WeakTensor as a composite tensor with one component.
  # Describes how gradients flow through a WeakTensor: the only gradient component is the wrapped
  # tensor, and rebuilding a WeakTensor from gradient components wraps the given gradient again.
  # Stateless; one instance is attached to WeakTensor as its composite gradient.
  """CompositeTensorGradient for WeakTensor."""

  def get_gradient_components(self, weak_tensor):
    # [FUNCTION DOC]
    # HELPER: part of the composite-tensor gradient protocol for WeakTensor.
    # Input: a weak tensor. Returns the wrapped ordinary tensor, which is the single component
    # that gradients are computed with respect to. No side effects, no exceptions.
    return weak_tensor.tensor

  def replace_gradient_components(self, weak_tensor, component_grads):
    # [FUNCTION DOC]
    # HELPER: part of the composite-tensor gradient protocol for WeakTensor.
    # Inputs: a weak tensor (used only for its type description) and the gradient for its single
    # component. Returns a new weak tensor built from the type description of the given weak
    # tensor and a one-element list holding the gradient. Accesses a protected member of the type
    # description. No other side effects.
    return weak_tensor._type_spec._from_components([component_grads])  # pylint: disable=protected-access


class WeakTensor(extension_type.BatchableExtensionType, core.Tensor):
  # [CLASS DOC]
  # Extension type for a weakly typed tensor: a thin immutable wrapper around one ordinary tensor
  # stored in the single field named tensor. Its dtype is only tentative, so in operations it can
  # be overridden by the dtype of the other operand (used for automatic dtype conversion).
  # Most tensor behavior (attributes, methods, length, truth value, hashing) is delegated to the
  # wrapped tensor. The wrapped tensor must have one of the allowed weak dtypes, enforced by the
  # validation hook. Instances should be created with the from_tensor class method, which picks
  # the eager or graph subclass. Declares a name used when serializing to SavedModel and a
  # composite gradient helper.
  """A weakly typed Tensor.

  A simple wrapper class that contains a normal Tensor.

  A "weak" type means that its dtype is temporarily inferred by the system,
  and could defer to other dtypes.

  i.g. weak f64 + f16 => f16

  This information is used for auto dtype conversion.
  """

  # __name__ is required for serialization in SavedModel.
  __name__ = "tf.WeakTensor"
  tensor: tensor_lib.Tensor

  def __validate__(self):
    # [FUNCTION DOC]
    # Validation hook called by the extension type machinery after construction. Takes no inputs.
    # Raises a type error if the wrapped tensor's dtype is not one of the allowed weak dtypes
    # (int32, int64, float32, float64, complex128); the message lists the allowed dtypes.
    # Returns nothing.
    if self.tensor.dtype not in _ALLOWED_WEAK_DTYPES:
      raise TypeError(
          f"{self.tensor.dtype} not allowed "
          f"as a weak type. The allowed types are {_ALLOWED_WEAK_DTYPES}."
      )

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the human-readable string form: the wrapped tensor's string form with a marker
    # saying the tensor is weak inserted before the closing character.
    return self._format_weak_tensor(is_repr=False)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the developer-oriented representation: the wrapped tensor's repr with a marker
    # saying the tensor is weak inserted before the closing character.
    return self._format_weak_tensor(is_repr=True)

  def _format_weak_tensor(self, is_repr):
    # [FUNCTION DOC]
    # HELPER: shared formatting for the string and representation methods.
    # Input: is_repr, true to use the wrapped tensor's repr and false to use its str.
    # Takes that text, finds the last occurrence of its final character (normally a closing
    # parenthesis or bracket) and inserts the weak marker before it. Returns the new string.
    # Assumes the text is non-empty. No side effects.
    tensor_str = self.tensor.__repr__() if is_repr else self.tensor.__str__()
    closing_char = tensor_str[len(tensor_str) - 1]
    last_index = tensor_str.rfind(closing_char)
    return tensor_str[:last_index] + ", weak=True" + closing_char

  def __getattr__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Fallback attribute lookup, invoked only when normal lookup fails. Forwards the request
    # (attribute name and any extra arguments) to the wrapped tensor so tensor methods and
    # properties are exposed directly on the weak tensor. Returns whatever the wrapped tensor
    # returns; raises an attribute error if the wrapped tensor lacks the attribute.
    # Fallback to `__getattr__` if `__getattribute__` fails, so that we can
    # directly expose Tensor's methods.
    return getattr(self.tensor, *args, **kwargs)

  def _disallow(self, task):
    # [FUNCTION DOC]
    # HELPER: builds the standard error for operations that are unsupported on symbolic weak
    # tensors.
    # Input: task, a short description of the forbidden operation. Always raises an
    # operator-not-allowed-in-graph error whose message explains how to avoid the problem
    # (use eager execution or decorate with the function-tracing decorator) and links to the
    # autograph limitations documentation. Never returns.
    raise errors.OperatorNotAllowedInGraphError(
        f"{task} is not allowed. You can attempt the following resolutions to"
        " the problem: If you are running in Graph mode, use Eager execution"
        " mode or decorate this function with @tf.function. If you are using"
        " AutoGraph, you can try decorating this function with @tf.function."
        " If that does not work, then you may be using an unsupported feature"
        " or your source code may not be visible to AutoGraph. See"
        " https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code"
        " for more information."
    )

  def _disallow_iteration(self):
    # [FUNCTION DOC]
    # HELPER: used by the iteration method to reject iteration over symbolic weak tensors.
    # Takes no inputs; always raises the operator-not-allowed-in-graph error with a message about
    # iterating over a symbolic weak tensor. Never returns.
    self._disallow("Iterating over a symbolic `tf.WeakTensor`")

  def _shape_as_list(self):
    # [FUNCTION DOC]
    # HELPER: convenience conversion of the static shape to a plain list.
    # Returns a list of the dimension sizes (each may be none if unknown) when the rank is known,
    # or none when the rank is unknown. No side effects.
    if self.shape.ndims is not None:
      return [dim.value for dim in self.shape.dims]
    else:
      return None

  def __iter__(self):
    # [FUNCTION DOC]
    # Iteration over the leading dimension. In graph (non-eager) mode raises the
    # operator-not-allowed-in-graph error. In eager mode reads the size of the first dimension of
    # the wrapped tensor and returns an iterator that yields each slice as a new weak tensor.
    # May raise whatever the wrapped tensor raises if it has no first dimension (e.g. a scalar).
    if not context.executing_eagerly():
      self._disallow_iteration()
    first_dim = self.tensor._get_first_dim()
    return _WeakTensorIterator(self, first_dim)

  def __hash__(self):
    # [FUNCTION DOC]
    # Returns the hash of the wrapped tensor (which may raise if tensors are unhashable, e.g.
    # when tensor equality is enabled).
    return self.tensor.__hash__()

  def __copy__(self):
    # [FUNCTION DOC]
    # Shallow copy support. Returns the same object, since weak tensors are immutable.
    # Weak Tensors are immutable so it's safe to return themselves as a copy.
    return self

  def __len__(self):
    # [FUNCTION DOC]
    # Returns the length of the first dimension by delegating to the wrapped tensor; raises
    # whatever it raises for scalars or unknown sizes.
    return self.tensor.__len__()

  def __bool__(self):
    # [FUNCTION DOC]
    # Truth value, delegated to the wrapped tensor (works for single-element eager tensors and
    # raises in graph mode or for multi-element tensors).
    return self.tensor.__bool__()

  def __tf_tensor__(
      self, dtype: Optional[dtypes.DType] = None, name: Optional[str] = None
  ):
    # [FUNCTION DOC]
    # Tensor conversion protocol hook. Inputs: optional requested dtype and optional name, both
    # default to none. Delegates to the wrapped tensor's conversion and returns its result (an
    # ordinary tensor, not a weak one). Errors for incompatible dtypes come from the wrapped
    # tensor.
    return self.tensor.__tf_tensor__(dtype=dtype, name=name)

  def __deepcopy__(self, memo):
    # [FUNCTION DOC]
    # Deep copy support. Input: memo, the copy bookkeeping mapping, which is ignored. Returns the
    # same object since the wrapper (and its tensor) is immutable.
    # Eager Tensors are immutable so it's safe to return themselves as a copy.
    del memo
    return self

  def to_tensor(self):
    # [FUNCTION DOC]
    # Returns the wrapped ordinary (strongly typed) tensor. No copy, no side effects.
    """Converts this 'WeakTensor' into a 'tf.Tensor'."""
    return self.tensor

  def _as_graph_element(self):
    # [FUNCTION DOC]
    # Graph-element conversion hook. Returns the wrapped tensor so graph code can use it.
    """Convert `self` to a graph element."""
    return self.tensor

  @classmethod
  def from_tensor(cls, tensor):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring describes the input only as a tensor, but passing an existing
    # weak tensor also works (its subclasses are value or symbol tensors) and it is wrapped again,
    # producing a nested weak tensor rather than returning it unchanged.
    # Factory that wraps a tensor. Input: tensor, an ordinary tensor. Returns an eager weak tensor
    # if the tensor is an eager value, or a graph weak tensor if it is a symbolic tensor.
    # Raises an invalid-argument error for any other input type. Construction can also raise a
    # type error if the dtype is not an allowed weak dtype.
    """Converts a 'tf.Tensor' into a 'WeakTensor'.

    This should be the standard way of creating a WeakTensor instead
    of directly calling the WeakTensor constructor.

    Args:
      tensor: The `tf.Tensor` that should be converted into a 'WeakTensor'.

    Returns:
      A `EagerWeakTensor` or 'GraphWeakTensor' that holds the `tensor`.
    """
    if isinstance(tensor, core.Value):
      return EagerWeakTensor(tensor)
    if isinstance(tensor, core.Symbol):
      return GraphWeakTensor(tensor)
    raise errors.InvalidArgumentError(
        None,
        None,
        "WeakTensor can only be constructed from tf.Tensor or tf.WeakTensor,"
        f" but {type(tensor)} was given.",
    )

  # Redefine `shape` and `dtype` rather than relying on `getattr` because the
  # class derives from core.Tensor which returns None in the two methods.
  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Property returning the dtype of the wrapped tensor. Defined explicitly because the base
    # core tensor class would otherwise return nothing.
    return self.tensor.dtype

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Property returning the static shape of the wrapped tensor. Defined explicitly because the
    # base core tensor class would otherwise return nothing.
    return self.tensor.shape

  @property
  def is_tensor_like(self):
    # [FUNCTION DOC]
    # Property that always reports true, marking weak tensors as tensor-like for type checks.
    return True

  __composite_gradient__ = WeakTensorGradient()


# EagerWeakTensor and GraphWeakTensor are wrapper classes that are
# introduced for WeakTensor to pass instance checks for core.Value or
# core.Symbol.
class EagerWeakTensor(core.Value, WeakTensor):
  # [CLASS DOC]
  # Weak tensor wrapping an eager tensor value. Exists so instance checks against the core value
  # type succeed. Adds eager-only behavior: conversion to numpy, Python numeric conversions,
  # index conversion, formatting and array conversion, all delegated to the wrapped tensor.
  """A weakly typed Eager Tensor."""

  __name__ = "tf.EagerWeakTensor"

  # Methods that are only available for EagerTensor.
  def numpy(self):
    # [FUNCTION DOC]
    # Returns the contents of the wrapped eager tensor as a NumPy array or scalar (a copy).
    # Raises a value error if the wrapped tensor is not an eager tensor.
    """Copy of the contents of this EagerWeakTensor into a NumPy array or scalar."""
    if not isinstance(self.tensor, ops.EagerTensor):
      raise ValueError("WeakTensor.numpy() is only supported in eager mode.")
    return self.tensor.numpy()

  def __complex__(self):
    # [FUNCTION DOC]
    # Converts to a Python complex number by delegating to the wrapped tensor; raises if the
    # tensor is not a single element.
    return self.tensor.__complex__()

  def __int__(self):
    # [FUNCTION DOC]
    # Converts to a Python integer by delegating to the wrapped tensor; raises if the tensor is
    # not a single element.
    return self.tensor.__int__()

  def __float__(self):
    # [FUNCTION DOC]
    # Converts to a Python float by delegating to the wrapped tensor; raises if the tensor is not
    # a single element.
    return self.tensor.__float__()

  def __index__(self):
    # [FUNCTION DOC]
    # Converts to a Python integer usable as an index by delegating to the wrapped tensor; raises
    # if the tensor is not an integer scalar.
    return self.tensor.__index__()

  def __format__(self, format_spec):
    # [FUNCTION DOC]
    # Formatted string support. Input: format_spec, the format specification text. Returns the
    # wrapped tensor's formatted text followed by the words "weakly typed".
    return f"{self.tensor.__format__(format_spec)} weakly typed"

  def __array__(self, dtype=None):
    # [FUNCTION DOC]
    # NumPy array protocol. Input: dtype, optional requested NumPy dtype (default none).
    # Returns a NumPy array built from the wrapped tensor's array conversion; the explicit
    # array wrapping ensures a real array is returned even for scalars.
    # We need to explicitly call np.array() because
    # self_tensor.__array__() for scalars raise:
    #     ValueError: object __array__ method not producing an array
    # resource_variable_ops also follows the same pattern.
    return np.array(self.tensor.__array__(dtype))


class GraphWeakTensor(core.Symbol, WeakTensor):
  # [CLASS DOC]
  # Weak tensor wrapping a symbolic (graph) tensor. Exists so instance checks against the core
  # symbol type succeed. Adds no behavior beyond WeakTensor other than its serialization name.
  """A weakly typed Graph Tensor."""

  __name__ = "tf.GraphWeakTensor"


class _WeakTensorIterator(object):
  # [CLASS DOC]
  # HELPER: iterator returned by WeakTensor iteration. Holds the weak tensor, the next index and
  # the number of leading-dimension entries (the limit). Each step yields the slice at the current
  # index wrapped as a weak tensor. Performs no error checking.
  """Iterates over the leading dim of a WeakTensor. Performs no error checks."""

  __slots__ = ["_weak_tensor", "_index", "_limit"]

  def __init__(self, weak_tensor, dim0):
    # [FUNCTION DOC]
    # HELPER: constructor. Inputs: weak_tensor, the weak tensor to iterate over, and dim0, the
    # size of its first dimension. Stores them and starts the index at zero. Returns nothing.
    self._weak_tensor = weak_tensor
    self._index = 0
    self._limit = dim0

  def __iter__(self):
    # [FUNCTION DOC]
    # Iterator protocol: returns the iterator itself.
    return self

  def __next__(self):
    # [FUNCTION DOC]
    # Returns the next slice along the leading dimension wrapped as a weak tensor and advances
    # the index. Raises the stop-iteration signal once the index reaches the limit.
    if self._index == self._limit:
      raise StopIteration
    result = WeakTensor.from_tensor((self._weak_tensor.tensor[self._index]))
    self._index += 1
    return result


def convert_to_weak_tensor_or_tensor(t, to_weak):
  # [FUNCTION DOC]
  # HELPER: used by weak-tensor-aware operation wrappers to give results the requested weakness.
  # Inputs: t, a tensor or weak tensor; to_weak, whether the result should be weak.
  # If to_weak is true, returns a weak tensor made from t (which can raise if t is not a plain
  # tensor or has a dtype not allowed as weak). Otherwise returns the underlying ordinary tensor
  # if t is a weak tensor, or t unchanged. No side effects.
  if to_weak:
    return WeakTensor.from_tensor(t)
  # We should return a normal Tensor because is_weak = False.
  if isinstance(t, WeakTensor):
    return t.tensor
  return t


# convert_to_tensor(WeakTensor) should return a Tensor because convert_to_tensor
# is mostly used internally and we want to limit the scope of WeakTensor
# creation to tf.constant and WeakTensor patched ops.
def weak_tensor_conversion_function(t):
  # [FUNCTION DOC]
  # HELPER: conversion function registered with the tensor conversion registry for weak tensors.
  # Input: t, a value being converted. Returns the wrapped ordinary tensor if t is a weak tensor;
  # for any other value implicitly returns nothing (no conversion performed here).
  if isinstance(t, WeakTensor):
    return t.tensor


tensor_conversion_registry.register_tensor_conversion_function(
    WeakTensor, weak_tensor_conversion_function
)
