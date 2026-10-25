# [MODULE DOC]
# Fast-Fourier-Transform operations for the signal package.
# Exposes the 1/2/3-D complex FFT and inverse FFT (re-exported straight from the generated spectral
# ops), real-input FFT wrappers (rfft, irfft and their 2-D and 3-D variants) that infer the
# fft_length argument and zero-pad the input, N-dimensional wrappers (fftnd, ifftnd, rfftnd, irfftnd)
# with optional normalization modes, and the fftshift / ifftshift utilities.
# Import-time side effects: builds the public op objects from wrapper factories, registers them
# with the public API exporter (with dispatch support), and registers gradient functions for the
# FFT, IFFT, FFT2D, IFFT2D, FFT3D, IFFT3D, RFFT, IRFFT, RFFT2D and IRFFT2D ops.
# Gradients for the 3-D real transforms are intentionally not registered.
# Several private helpers here are shared by the wrappers and gradient code.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Fast-Fourier Transform ops."""
import re

import numpy as np

from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_util as _tensor_util
from tensorflow.python.ops import array_ops as _array_ops
from tensorflow.python.ops import array_ops_stack as _array_ops_stack
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import math_ops as _math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


def _infer_fft_length_for_fftn(input_tensor):
  # [FUNCTION DOC]
  # HELPER: default fft_length computation for the N-dimensional FFT wrappers.
  # Returns a tensor holding the dynamic sizes of the last rank-many dimensions of the input, that
  # is, the sizes of all dimensions, since the slice length equals the static rank. Requires the
  # static rank to be known. Takes one tensor argument; no side effects.
  return _array_ops.shape(input_tensor)[-len(input_tensor.shape) :]


def _infer_fft_length_for_irfftn(input_tensor):
  # [FUNCTION DOC]
  # HELPER: default fft_length computation for the N-dimensional inverse real FFT wrapper.
  # KNOWN ISSUE: requires a fully static input shape; with an unknown dimension the static size is
  # a null value and the arithmetic on it fails with a type error, unlike the 1-D/2-D/3-D helper
  # which falls back to dynamic shape computation.
  # Uses the static shape of the input (all dimensions), and replaces the last entry by twice
  # (size minus one), floored at zero, which is the output length of the inverse real transform.
  # Returns a 32-bit integer tensor. No side effects.
  fft_shape = input_tensor.get_shape()[-len(input_tensor.shape) :]
  fft_length = fft_shape.as_list()
  fft_length[-1] = max(0, 2 * (fft_length[-1] - 1))
  return _ops.convert_to_tensor(fft_length, _dtypes.int32)


def _infer_axes_for_fftn(input_tensor):
  # [FUNCTION DOC]
  # HELPER: default axes computation for the N-dimensional FFT wrappers.
  # Returns a 32-bit integer tensor listing every axis index of the input, from zero up to its
  # static rank minus one. Requires the static rank to be known. No side effects.
  return _ops.convert_to_tensor(
      np.arange(len(input_tensor.shape)), _dtypes.int32
  )


def _process_empty_axes(input_tensor, axes):
  # [FUNCTION DOC]
  # HELPER: normalizes the axes argument of the N-dimensional FFT wrappers.
  # If axes is nothing/null, returns the list of all axes of the input; otherwise converts the given
  # axes (list, array or tensor of integers) to a 32-bit integer tensor. Returns that tensor.
  # No side effects.
  if axes is None:
    axes = _infer_axes_for_fftn(input_tensor)
  else:
    axes = _ops.convert_to_tensor(axes, _dtypes.int32)
  return axes


def _infer_fft_length_for_rfft(input_tensor, fft_rank):
  # [FUNCTION DOC]
  # HELPER: default fft_length computation for the real FFT wrappers (1-D, 2-D, 3-D).
  # Inputs: the real input tensor and the number of innermost dimensions that are transformed.
  # Returns the sizes of those innermost dimensions: a constant 32-bit integer tensor when they are
  # all statically known, otherwise a tensor computed from the dynamic shape. No side effects.
  """Infers the `fft_length` argument for a `rank` RFFT from `input_tensor`."""
  # A TensorShape for the inner fft_rank dimensions.
  fft_shape = input_tensor.get_shape()[-fft_rank:]

  # If any dim is unknown, fall back to tensor-based math.
  if not fft_shape.is_fully_defined():
    return _array_ops.shape(input_tensor)[-fft_rank:]

  # Otherwise, return a constant.
  return _ops.convert_to_tensor(fft_shape.as_list(), _dtypes.int32)


def _infer_fft_length_for_irfft(input_tensor, fft_rank):
  # [FUNCTION DOC]
  # HELPER: default fft_length computation for the inverse real FFT wrappers (1-D, 2-D, 3-D).
  # Inputs: the complex input tensor and the number of innermost transformed dimensions.
  # Returns the sizes of those dimensions with the last one replaced by twice (size minus one),
  # floored at zero, i.e. the length of the real output. Static shapes give a constant 32-bit
  # integer tensor; otherwise the value is built from the dynamic shape with stack/unstack ops.
  # No side effects.
  """Infers the `fft_length` argument for a `rank` IRFFT from `input_tensor`."""
  # A TensorShape for the inner fft_rank dimensions.
  fft_shape = input_tensor.get_shape()[-fft_rank:]

  # If any dim is unknown, fall back to tensor-based math.
  if not fft_shape.is_fully_defined():
    fft_length = _array_ops_stack.unstack(
        _array_ops.shape(input_tensor)[-fft_rank:])
    fft_length[-1] = _math_ops.maximum(0, 2 * (fft_length[-1] - 1))
    return _array_ops_stack.stack(fft_length)

  # Otherwise, return a constant.
  fft_length = fft_shape.as_list()
  if fft_length:
    fft_length[-1] = max(0, 2 * (fft_length[-1] - 1))
  return _ops.convert_to_tensor(fft_length, _dtypes.int32)


def _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length, is_reverse=False):
  # [FUNCTION DOC]
  # HELPER: zero-pads the input of the FFT wrappers up to the requested fft_length.
  # Inputs: the tensor, the number of innermost transformed dimensions, the fft_length tensor, and
  # is_reverse (default false). When is_reverse is true (inverse real transforms) the innermost
  # dimension is only padded up to fft_length / 2 + 1 because the input is a half spectrum.
  # Only pads at the end of each dimension, never crops; dimensions already long enough are left.
  # Empty tensors (any static dimension zero) are returned unchanged. Padding is precomputed when
  # shapes are static (returning the input itself if nothing to pad), otherwise padding ops are
  # added to the graph. Returns the (possibly padded) tensor. No side effects.
  """Pads `input_tensor` to `fft_length` on its inner-most `fft_rank` dims."""
  fft_shape = _tensor_util.constant_value_as_shape(fft_length)

  # Edge case: skip padding empty tensors.
  if (input_tensor.shape.ndims is not None and
      any(dim.value == 0 for dim in input_tensor.shape.dims)):
    return input_tensor

  # If we know the shapes ahead of time, we can either skip or pre-compute the
  # appropriate paddings. Otherwise, fall back to computing paddings in
  # TensorFlow.
  if fft_shape.is_fully_defined() and input_tensor.shape.ndims is not None:
    # Slice the last FFT-rank dimensions from input_tensor's shape.
    input_fft_shape = input_tensor.shape[-fft_shape.ndims:]  # pylint: disable=invalid-unary-operand-type

    if input_fft_shape.is_fully_defined():
      # In reverse, we only pad the inner-most dimension to fft_length / 2 + 1.
      if is_reverse:
        fft_shape = fft_shape[:-1].concatenate(
            fft_shape.dims[-1].value // 2 + 1)

      paddings = [[0, max(fft_dim.value - input_dim.value, 0)]
                  for fft_dim, input_dim in zip(
                      fft_shape.dims, input_fft_shape.dims)]
      if any(pad > 0 for _, pad in paddings):
        outer_paddings = [[0, 0]] * max((input_tensor.shape.ndims -
                                         fft_shape.ndims), 0)
        return _array_ops.pad(input_tensor, outer_paddings + paddings)
      return input_tensor

  # If we can't determine the paddings ahead of time, then we have to pad. If
  # the paddings end up as zero, tf.pad has a special-case that does no work.
  input_rank = _array_ops.rank(input_tensor)
  input_fft_shape = _array_ops.shape(input_tensor)[-fft_rank:]
  outer_dims = _math_ops.maximum(0, input_rank - fft_rank)
  outer_paddings = _array_ops.zeros([outer_dims], fft_length.dtype)
  # In reverse, we only pad the inner-most dimension to fft_length / 2 + 1.
  if is_reverse:
    fft_length = _array_ops.concat([fft_length[:-1],
                                    fft_length[-1:] // 2 + 1], 0)
  fft_paddings = _math_ops.maximum(0, fft_length - input_fft_shape)
  paddings = _array_ops.concat([outer_paddings, fft_paddings], 0)
  paddings = _array_ops_stack.stack(
      [_array_ops.zeros_like(paddings), paddings], axis=1)
  return _array_ops.pad(input_tensor, paddings)


def _rfft_wrapper(fft_fn, fft_rank, default_name):
  # [FUNCTION DOC]
  # HELPER: factory that builds the public rfft, rfft2d and rfft3d functions.
  # Inputs: the generated low-level real FFT op, the transform rank, and the default op name scope.
  # Returns a function taking an input tensor, an optional fft_length and an optional name; it
  # validates the dtype, infers fft_length when absent, pads the input, and calls the generated op
  # with the matching complex output dtype. If the generated op has a docstring, the returned
  # function reuses it with the output-type argument description removed.
  """Wrapper around gen_spectral_ops.rfft* that infers fft_length argument."""

  def _rfft(input_tensor, fft_length=None, name=None):
    # [FUNCTION DOC]
    # HELPER: the real FFT function returned by the rfft wrapper factory.
    # Inputs: a real tensor (converted, preferring 32-bit float), optional fft_length (integer list
    # or tensor, inferred from the input shape when omitted) and an optional op name.
    # Raises a value error if the dtype is not 32-bit or 64-bit float, and a shape error if the rank
    # is below the transform rank. Pads the input to fft_length, uses a static fft_length when it can
    # be evaluated, and returns a complex64 (for float32) or complex128 (for float64) tensor.
    # Adds ops to the current graph under a name scope.
    """Wrapper around gen_spectral_ops.rfft* that infers fft_length argument."""
    with _ops.name_scope(name, default_name,
                         [input_tensor, fft_length]) as name:
      input_tensor = _ops.convert_to_tensor(input_tensor,
                                            preferred_dtype=_dtypes.float32)
      if input_tensor.dtype not in (_dtypes.float32, _dtypes.float64):
        raise ValueError(
            "RFFT requires tf.float32 or tf.float64 inputs, got: %s" %
            input_tensor)
      real_dtype = input_tensor.dtype
      if real_dtype == _dtypes.float32:
        complex_dtype = _dtypes.complex64
      else:
        assert real_dtype == _dtypes.float64
        complex_dtype = _dtypes.complex128
      input_tensor.shape.with_rank_at_least(fft_rank)
      if fft_length is None:
        fft_length = _infer_fft_length_for_rfft(input_tensor, fft_rank)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      input_tensor = _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length)

      fft_length_static = _tensor_util.constant_value(fft_length)
      if fft_length_static is not None:
        fft_length = fft_length_static
      return fft_fn(input_tensor, fft_length, Tcomplex=complex_dtype, name=name)
  if fft_fn.__doc__ is not None:
    _rfft.__doc__ = re.sub("    Tcomplex.*?\n", "", fft_fn.__doc__)
  return _rfft


def _irfft_wrapper(ifft_fn, fft_rank, default_name):
  # [FUNCTION DOC]
  # HELPER: factory that builds the public irfft, irfft2d and irfft3d functions.
  # Inputs: the generated low-level inverse real FFT op, the transform rank, and the default name.
  # Returns a function taking a complex tensor, optional fft_length and optional name that
  # validates dtype, infers fft_length, pads the half spectrum, and calls the generated op with
  # the matching real output dtype. The returned function reuses the generated op's docstring with
  # the output-type argument removed and the argument name input replaced by input_tensor.
  """Wrapper around gen_spectral_ops.irfft* that infers fft_length argument."""

  def _irfft(input_tensor, fft_length=None, name=None):
    # [FUNCTION DOC]
    # HELPER: the inverse real FFT function returned by the irfft wrapper factory.
    # Inputs: a complex tensor (converted, preferring complex64), optional fft_length (inferred as
    # twice the last transformed size minus two when omitted), and an optional op name.
    # Raises a value error if dtype is not complex64 or complex128. Requires rank at least the
    # transform rank. Pads the half spectrum, prefers a static fft_length, and returns a real tensor
    # of the matching float precision. Adds ops to the graph under a name scope.
    """Wrapper irfft* that infers fft_length argument."""
    with _ops.name_scope(name, default_name,
                         [input_tensor, fft_length]) as name:
      input_tensor = _ops.convert_to_tensor(input_tensor,
                                            preferred_dtype=_dtypes.complex64)
      input_tensor.shape.with_rank_at_least(fft_rank)
      if input_tensor.dtype not in (_dtypes.complex64, _dtypes.complex128):
        raise ValueError(
            "IRFFT requires tf.complex64 or tf.complex128 inputs, got: %s" %
            input_tensor)
      complex_dtype = input_tensor.dtype
      real_dtype = complex_dtype.real_dtype
      if fft_length is None:
        fft_length = _infer_fft_length_for_irfft(input_tensor, fft_rank)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      input_tensor = _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length,
                                         is_reverse=True)
      fft_length_static = _tensor_util.constant_value(fft_length)
      if fft_length_static is not None:
        fft_length = fft_length_static
      return ifft_fn(input_tensor, fft_length, Treal=real_dtype, name=name)

  if ifft_fn.__doc__ is not None:
    _irfft.__doc__ = re.sub(
        "`input`",
        "`input_tensor`",
        re.sub("    Treal.*?\n", "", ifft_fn.__doc__),
    )
  return _irfft


def _fftn_wrapper(fft_n, default_name):
  # [FUNCTION DOC]
  # HELPER: factory that builds the public fftnd function.
  # Inputs: the generated N-dimensional FFT op and the default name scope. Returns a function that
  # infers axes and fft_length, pads the input, applies the requested normalization and calls
  # the generated op. The returned function reuses the generated op's docstring with the output
  # type argument description removed.
  """Wrapper around gen_spectral_ops.fftn."""

  def _fftn(input_tensor, fft_length=None, axes=None, norm=None, name=None):
    # [FUNCTION DOC]
    # HELPER: the N-dimensional complex FFT function returned by the fftn wrapper factory.
    # KNOWN ISSUE: the axes are computed from the raw argument before conversion to a tensor, so a
    # plain list input without a shape attribute fails when axes is omitted. When fft_length is
    # omitted it covers every dimension even if a shorter axes list is given, and with a norm other
    # than backward the length loop needs a statically known fft_length.
    # Inputs: input tensor (complex64 preferred), fft_length, axes (default: all axes), norm
    # (nothing/null or backward means no scaling; forward divides by the product of the lengths;
    # ortho divides by its square root), and a name. Scaling is applied to the input before the
    # transform. Pads the input, calls the generated op and returns the complex result.
    """Wrapper around gen_spectral_ops.*fft that infers fft_length and axes arguments."""
    with _ops.name_scope(
        name, default_name, [input_tensor, fft_length, axes]
    ) as name:
      axes = _process_empty_axes(input_tensor, axes)
      fft_rank = axes.shape[0]
      input_tensor = _ops.convert_to_tensor(
          input_tensor, preferred_dtype=_dtypes.complex64
      )
      input_tensor.shape.with_rank_at_least(fft_rank)
      if fft_length is None:
        fft_length = _infer_fft_length_for_fftn(input_tensor)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      input_tensor = _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length)

      fft_length_static = _tensor_util.constant_value(fft_length)
      if fft_length_static is not None:
        fft_length = fft_length_static
      if norm is None:
        norm = "backward"
      n = 1
      if norm != "backward":
        for fft_length_i in fft_length:
          n *= fft_length_i
        if norm == "forward":
          input_tensor /= n
        elif norm == "ortho":
          input_tensor /= np.sqrt(n)  # should be sqrt(N)
      return fft_n(input_tensor, fft_length, axes, name=name)

  if fft_n.__doc__ is not None:
    _fftn.__doc__ = re.sub(r"    Tcomplex.*?\n", "", fft_n.__doc__)
  return _fftn


def _ifftn_wrapper(ifft_n, default_name):
  # [FUNCTION DOC]
  # HELPER: factory that builds the public ifftnd function.
  # Inputs: the generated N-dimensional inverse FFT op and a default name scope. Returns a function
  # that infers axes and fft_length, pads, scales per the norm mode and calls the generated op.
  # The returned function reuses the generated op's docstring minus the output-type description.
  """Wrapper around gen_spectral_ops.ifftn."""

  def _ifftn(input_tensor, fft_length=None, axes=None, norm=None, name=None):
    # [FUNCTION DOC]
    # HELPER: the N-dimensional inverse complex FFT returned by the ifftn wrapper factory.
    # KNOWN ISSUE: shares the problems of the forward version: axes are derived before the input is
    # converted to a tensor, fft_length defaults to all dimensions regardless of axes, and a non
    # backward norm needs a static fft_length. Also the docstring says it wraps the forward
    # transform while it calls the inverse op.
    # Inputs: input tensor, fft_length, axes (default all), norm (backward or null: no scaling;
    # forward multiplies the input by the product of lengths; ortho multiplies by its square root),
    # and a name. Pads, scales the input before the op, and returns the complex result.
    """Wrapper around gen_spectral_ops.*fft that infers fft_length and axes arguments."""
    with _ops.name_scope(
        name, default_name, [input_tensor, fft_length, axes]
    ) as name:
      axes = _process_empty_axes(input_tensor, axes)
      fft_rank = axes.shape[0]
      input_tensor = _ops.convert_to_tensor(
          input_tensor, preferred_dtype=_dtypes.complex64
      )
      input_tensor.shape.with_rank_at_least(fft_rank)
      if fft_length is None:
        fft_length = _infer_fft_length_for_fftn(input_tensor)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      input_tensor = _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length)

      fft_length_static = _tensor_util.constant_value(fft_length)
      if fft_length_static is not None:
        fft_length = fft_length_static
      if norm is None:
        norm = "backward"
      n = 1
      if norm != "backward":
        for fft_length_i in fft_length:
          n *= fft_length_i
        if norm == "forward":
          input_tensor *= n
        elif norm == "ortho":
          input_tensor *= np.sqrt(n)  # should be sqrt(N)
      return ifft_n(input_tensor, fft_length, axes, name=name)

  if ifft_n.__doc__ is not None:
    _ifftn.__doc__ = re.sub(r"    Tcomplex.*?\n", "", ifft_n.__doc__)
  return _ifftn


def _rfftn_wrapper(rfft_n, default_name):
  # [FUNCTION DOC]
  # HELPER: factory that builds the public rfftnd function.
  # Inputs: the generated N-dimensional real FFT op and a default name scope. Returns a function
  # that validates real dtype, infers axes and fft_length, pads, scales per norm and calls the
  # generated op with the matching complex output dtype. Reuses the op docstring minus the
  # output-type description.
  """Wrapper around gen_spectral_ops.rfftn."""

  def _rfftn(input_tensor, fft_length=None, axes=None, norm=None, name=None):
    # [FUNCTION DOC]
    # HELPER: the N-dimensional real FFT returned by the rfftn wrapper factory.
    # KNOWN ISSUE: the docstring is generic ("wrapper around the fft ops") although this validates
    # real input; axes are derived before tensor conversion, fft_length defaults to all dimensions
    # regardless of axes, and a non backward norm requires a static fft_length.
    # Inputs: real tensor (float32 preferred), fft_length, axes (default all), norm (backward or
    # null: none; forward divides by product of lengths; ortho divides by its square root), name.
    # Raises a value error for non-float32/float64 input. Returns complex64 or complex128.
    """Wrapper around gen_spectral_ops.*fft that infers fft_length and axes arguments."""
    with _ops.name_scope(
        name, default_name, [input_tensor, fft_length, axes]
    ) as name:
      axes = _process_empty_axes(input_tensor, axes)
      fft_rank = axes.shape[0]
      input_tensor = _ops.convert_to_tensor(
          input_tensor, preferred_dtype=_dtypes.float32
      )
      if input_tensor.dtype not in (_dtypes.float32, _dtypes.float64):
        raise ValueError(
            "RFFT requires tf.float32 or tf.float64 inputs, got: %s"
            % input_tensor
        )
      real_dtype = input_tensor.dtype
      if real_dtype == _dtypes.float32:
        complex_dtype = _dtypes.complex64
      else:
        assert real_dtype == _dtypes.float64
        complex_dtype = _dtypes.complex128
      input_tensor.shape.with_rank_at_least(fft_rank)
      if fft_length is None:
        fft_length = _infer_fft_length_for_fftn(input_tensor)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      input_tensor = _maybe_pad_for_rfft(input_tensor, fft_rank, fft_length)

      fft_length_static = _tensor_util.constant_value(fft_length)
      if fft_length_static is not None:
        fft_length = fft_length_static
      if norm is None:
        norm = "backward"
      n = 1
      if norm != "backward":
        for fft_length_i in fft_length:
          n *= fft_length_i
        if norm == "forward":
          input_tensor /= n
        elif norm == "ortho":
          input_tensor /= np.sqrt(n)  # should be sqrt(N)
      return rfft_n(
          input_tensor,
          fft_length,
          axes,
          Tcomplex=complex_dtype,
          name=name,
      )

  if rfft_n.__doc__ is not None:
    _rfftn.__doc__ = re.sub(r"    Tcomplex.*?\n", "", rfft_n.__doc__)
  return _rfftn


def _irfftn_wrapper(irfft_n, default_name):
  # [FUNCTION DOC]
  # HELPER: factory that builds the public irfftnd function.
  # Inputs: the generated N-dimensional inverse real FFT op and a default name scope. Returns a
  # function that validates complex dtype, infers axes and fft_length, pads the half spectrum,
  # scales per norm and calls the generated op with the real output dtype. Reuses the op docstring
  # with the output-type description removed and input renamed to input_tensor.
  """Wrapper around gen_spectral_ops.irfftn."""

  def _irfftn(input_tensor, fft_length=None, axes=None, norm=None, name=None):
    # [FUNCTION DOC]
    # HELPER: the N-dimensional inverse real FFT returned by the irfftn wrapper factory.
    # KNOWN ISSUE: axes are derived before tensor conversion, default fft_length ignores axes and
    # needs a fully static shape, and a non backward norm needs a static fft_length. The name scope
    # omits axes from its input list.
    # Inputs: complex tensor (complex64 preferred), fft_length, axes (default all), norm (backward or
    # null: none; forward multiplies by product of lengths; ortho by its square root), name.
    # Raises a value error for non-complex input. Returns a real tensor.
    """Wrapper irfft* that infers fft_length argument."""
    with _ops.name_scope(
        name, default_name, [input_tensor, fft_length]
    ) as name:
      axes = _process_empty_axes(input_tensor, axes)
      fft_rank = axes.shape[0]
      input_tensor = _ops.convert_to_tensor(
          input_tensor, preferred_dtype=_dtypes.complex64
      )
      input_tensor.shape.with_rank_at_least(fft_rank)
      if input_tensor.dtype not in (_dtypes.complex64, _dtypes.complex128):
        raise ValueError(
            "IRFFT requires tf.complex64 or tf.complex128 inputs, got: %s"
            % input_tensor
        )
      complex_dtype = input_tensor.dtype
      real_dtype = complex_dtype.real_dtype
      if fft_length is None:
        fft_length = _infer_fft_length_for_irfftn(input_tensor)
      else:
        fft_length = _ops.convert_to_tensor(fft_length, _dtypes.int32)
      input_tensor = _maybe_pad_for_rfft(
          input_tensor, fft_rank, fft_length, is_reverse=True
      )
      fft_length_static = _tensor_util.constant_value(fft_length)
      if fft_length_static is not None:
        fft_length = fft_length_static

      if norm is None:
        norm = "backward"
      n = 1
      if norm != "backward":
        for fft_length_i in fft_length:
          n *= fft_length_i
        if norm == "forward":
          input_tensor *= n
        elif norm == "ortho":
          input_tensor *= np.sqrt(n)  # should be sqrt(N)
      return irfft_n(
          input_tensor, fft_length, axes, Treal=real_dtype, name=name
      )

  if irfft_n.__doc__ is not None:
    _irfftn.__doc__ = re.sub(
        "`input`",
        "`input_tensor`",
        re.sub(r"    Treal.*?\n", "", irfft_n.__doc__),
    )
  return _irfftn


# FFT/IFFT 1/2/3D are exported via
# third_party/tensorflow/core/api_def/python_api/
fft = gen_spectral_ops.fft
ifft = gen_spectral_ops.ifft
fft2d = gen_spectral_ops.fft2d
ifft2d = gen_spectral_ops.ifft2d
fft3d = gen_spectral_ops.fft3d
ifft3d = gen_spectral_ops.ifft3d
fftnd = _fftn_wrapper(gen_spectral_ops.fftnd, "fftnd")
tf_export("signal.fftnd")(
    dispatch.add_dispatch_support(fftnd)
)
ifftnd = _ifftn_wrapper(gen_spectral_ops.ifftnd, "ifftnd")
tf_export("signal.ifftnd")(
    dispatch.add_dispatch_support(ifftnd)
)
rfft = _rfft_wrapper(gen_spectral_ops.rfft, 1, "rfft")
tf_export("signal.rfft", v1=["signal.rfft", "spectral.rfft"])(
    dispatch.add_dispatch_support(rfft))
irfft = _irfft_wrapper(gen_spectral_ops.irfft, 1, "irfft")
tf_export("signal.irfft", v1=["signal.irfft", "spectral.irfft"])(
    dispatch.add_dispatch_support(irfft))
rfft2d = _rfft_wrapper(gen_spectral_ops.rfft2d, 2, "rfft2d")
tf_export("signal.rfft2d", v1=["signal.rfft2d", "spectral.rfft2d"])(
    dispatch.add_dispatch_support(rfft2d))
irfft2d = _irfft_wrapper(gen_spectral_ops.irfft2d, 2, "irfft2d")
tf_export("signal.irfft2d", v1=["signal.irfft2d", "spectral.irfft2d"])(
    dispatch.add_dispatch_support(irfft2d))
rfft3d = _rfft_wrapper(gen_spectral_ops.rfft3d, 3, "rfft3d")
tf_export("signal.rfft3d", v1=["signal.rfft3d", "spectral.rfft3d"])(
    dispatch.add_dispatch_support(rfft3d))
irfft3d = _irfft_wrapper(gen_spectral_ops.irfft3d, 3, "irfft3d")
tf_export("signal.irfft3d", v1=["signal.irfft3d", "spectral.irfft3d"])(
    dispatch.add_dispatch_support(irfft3d))
rfftnd = _rfftn_wrapper(gen_spectral_ops.rfftnd, "rfftnd")
tf_export("signal.rfftnd")(
    dispatch.add_dispatch_support(rfftnd)
)
irfftnd = _irfftn_wrapper(gen_spectral_ops.irfftnd, "irfftnd")
tf_export("signal.irfftnd")(
    dispatch.add_dispatch_support(irfftnd)
)


def _fft_size_for_grad(grad, rank):
  # [FUNCTION DOC]
  # HELPER: gradient code utility. Returns the product of the sizes of the innermost rank
  # dimensions of the given tensor (the number of points in one transform), as a dynamic integer
  # scalar tensor.
  return _math_ops.reduce_prod(_array_ops.shape(grad)[-rank:])


@_ops.RegisterGradient("FFT")
def _fft_grad(_, grad):
  # [FUNCTION DOC]
  # Registered gradient of the 1-D FFT op. The op is ignored; the upstream gradient is passed
  # through the inverse FFT and multiplied by the transform length. Returns one gradient tensor.
  size = _math_ops.cast(_fft_size_for_grad(grad, 1), grad.dtype)
  return ifft(grad) * size


@_ops.RegisterGradient("IFFT")
def _ifft_grad(_, grad):
  # [FUNCTION DOC]
  # Registered gradient of the 1-D inverse FFT op. The upstream gradient is passed through the
  # forward FFT and scaled by the reciprocal of the transform length. Returns one tensor.
  rsize = _math_ops.cast(
      1. / _math_ops.cast(_fft_size_for_grad(grad, 1), grad.dtype.real_dtype),
      grad.dtype)
  return fft(grad) * rsize


@_ops.RegisterGradient("FFT2D")
def _fft2d_grad(_, grad):
  # [FUNCTION DOC]
  # Registered gradient of the 2-D FFT op: inverse 2-D FFT of the upstream gradient times the
  # number of points in the two innermost dimensions. Returns one gradient tensor.
  size = _math_ops.cast(_fft_size_for_grad(grad, 2), grad.dtype)
  return ifft2d(grad) * size


@_ops.RegisterGradient("IFFT2D")
def _ifft2d_grad(_, grad):
  # [FUNCTION DOC]
  # Registered gradient of the 2-D inverse FFT op: forward 2-D FFT of the upstream gradient scaled
  # by one over the number of points in the two innermost dimensions. Returns one tensor.
  rsize = _math_ops.cast(
      1. / _math_ops.cast(_fft_size_for_grad(grad, 2), grad.dtype.real_dtype),
      grad.dtype)
  return fft2d(grad) * rsize


@_ops.RegisterGradient("FFT3D")
def _fft3d_grad(_, grad):
  # [FUNCTION DOC]
  # Registered gradient of the 3-D FFT op: inverse 3-D FFT of the upstream gradient times the
  # number of points in the three innermost dimensions. Returns one gradient tensor.
  size = _math_ops.cast(_fft_size_for_grad(grad, 3), grad.dtype)
  return ifft3d(grad) * size


@_ops.RegisterGradient("IFFT3D")
def _ifft3d_grad(_, grad):
  # [FUNCTION DOC]
  # Registered gradient of the 3-D inverse FFT op: forward 3-D FFT of the upstream gradient scaled
  # by one over the number of points in the three innermost dimensions. Returns one tensor.
  rsize = _math_ops.cast(
      1. / _math_ops.cast(_fft_size_for_grad(grad, 3), grad.dtype.real_dtype),
      grad.dtype)
  return fft3d(grad) * rsize


def _rfft_grad_helper(rank, irfft_fn):
  # [FUNCTION DOC]
  # HELPER: factory for the registered gradient functions of the real FFT ops.
  # Inputs: the transform rank (must be 1 or 2; otherwise an assertion error is raised, since no
  # 3-D gradient exists) and the matching inverse real FFT function. Returns a gradient function
  # taking the forward op and the upstream gradient.
  """Returns a gradient function for an RFFT of the provided rank."""
  # Can't happen because we don't register a gradient for RFFT3D.
  assert rank in (1, 2), "Gradient for RFFT3D is not implemented."

  def _grad(op, grad):
    # [FUNCTION DOC]
    # Gradient of the real FFT for the captured rank. Computes the inverse real FFT of the upstream
    # gradient times the input size, then adds correction terms for the first and (for even lengths)
    # last frequency components that were dropped by Hermitian symmetry, and halves the sum.
    # For rank two the correction terms use a batched matrix product with a DFT-like mask matrix.
    # Returns a pair: the gradient for the input and nothing for the fft_length input.
    """A gradient function for RFFT with the provided `rank` and `irfft_fn`."""
    fft_length = op.inputs[1]
    complex_dtype = grad.dtype
    real_dtype = complex_dtype.real_dtype
    input_shape = _array_ops.shape(op.inputs[0])
    is_even = _math_ops.cast(1 - (fft_length[-1] % 2), complex_dtype)

    def _tile_for_broadcasting(matrix, t):
      # [FUNCTION DOC]
      # HELPER: gradient glue. Reshapes a 2-D matrix to have leading singleton dimensions and tiles
      # it over the batch dimensions of tensor t (all but its last two), returning a tensor that can
      # be used in a batched matrix product with t.
      expanded = _array_ops.reshape(
          matrix,
          _array_ops.concat([
              _array_ops.ones([_array_ops.rank(t) - 2], _dtypes.int32),
              _array_ops.shape(matrix)
          ], 0))
      return _array_ops.tile(
          expanded, _array_ops.concat([_array_ops.shape(t)[:-2], [1, 1]], 0))

    def _mask_matrix(length):
      # [FUNCTION DOC]
      # HELPER: gradient glue for the 2-D real FFT.
      # KNOWN ISSUE: the docstring describes a vector of quadratic-phase twiddles, but the code
      # builds a full square matrix: exp of minus two pi i times the product of row and column index
      # over the length, i.e. the DFT matrix.
      # Input: an integer length. Returns a complex matrix of that size in the gradient's dtype.
      """Computes t_n = exp(sqrt(-1) * pi * n^2 / line_len)."""
      # TODO(rjryan): Speed up computation of twiddle factors using the
      # following recurrence relation and cache them across invocations of RFFT.
      #
      # t_n = exp(sqrt(-1) * pi * n^2 / line_len)
      # for n = 0, 1,..., line_len-1.
      # For n > 2, use t_n = t_{n-1}^2 / t_{n-2} * t_1^2
      a = _array_ops.tile(
          _array_ops.expand_dims(_math_ops.range(length), 0), (length, 1))
      b = _array_ops.transpose(a, [1, 0])
      return _math_ops.exp(
          -2j * np.pi * _math_ops.cast(a * b, complex_dtype) /
          _math_ops.cast(length, complex_dtype))

    def _ymask(length):
      # [FUNCTION DOC]
      # HELPER: gradient glue. Returns a complex vector of the given length alternating between one
      # and minus one, starting with one, in the gradient's complex dtype.
      """A sequence of [1+0j, -1+0j, 1+0j, -1+0j, ...] with length `length`."""
      return _math_ops.cast(1 - 2 * (_math_ops.range(length) % 2),
                            complex_dtype)

    y0 = grad[..., 0:1]
    if rank == 1:
      ym = grad[..., -1:]
      extra_terms = y0 + is_even * ym * _ymask(input_shape[-1])
    elif rank == 2:
      # Create a mask matrix for y0 and ym.
      base_mask = _mask_matrix(input_shape[-2])

      # Tile base_mask to match y0 in shape so that we can batch-matmul the
      # inner 2 dimensions.
      tiled_mask = _tile_for_broadcasting(base_mask, y0)

      y0_term = _math_ops.matmul(tiled_mask, _math_ops.conj(y0))
      extra_terms = y0_term

      ym = grad[..., -1:]
      ym_term = _math_ops.matmul(tiled_mask, _math_ops.conj(ym))

      inner_dim = input_shape[-1]
      ym_term = _array_ops.tile(
          ym_term,
          _array_ops.concat([
              _array_ops.ones([_array_ops.rank(grad) - 1], _dtypes.int32),
              [inner_dim]
          ], 0)) * _ymask(inner_dim)

      extra_terms += is_even * ym_term

    # The gradient of RFFT is the IRFFT of the incoming gradient times a scaling
    # factor, plus some additional terms to make up for the components dropped
    # due to Hermitian symmetry.
    input_size = _math_ops.cast(
        _fft_size_for_grad(op.inputs[0], rank), real_dtype)
    the_irfft = irfft_fn(grad, fft_length)
    return 0.5 * (the_irfft * input_size + _math_ops.real(extra_terms)), None

  return _grad


def _irfft_grad_helper(rank, rfft_fn):
  # [FUNCTION DOC]
  # HELPER: factory for the registered gradient functions of the inverse real FFT ops.
  # Inputs: the transform rank (1 or 2, otherwise an assertion error) and the matching forward
  # real FFT function. Returns a gradient function taking the forward op and upstream gradient.
  """Returns a gradient function for an IRFFT of the provided rank."""
  # Can't happen because we don't register a gradient for IRFFT3D.
  assert rank in (1, 2), "Gradient for IRFFT3D is not implemented."

  def _grad(op, grad):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if the gradient dtype is neither float32 nor float64 the complex dtype variable
    # is never assigned and a later use raises an unbound-variable error.
    # Gradient of the inverse real FFT for the captured rank: forward real FFT of the upstream
    # gradient, multiplied by one over the transform size and by a mask that doubles the
    # components that were de-duplicated by Hermitian symmetry (first is one, last is one only for
    # even lengths). Uses a static fft_length when available. Returns a pair: the gradient for the
    # input and nothing for the fft_length input.
    """A gradient function for IRFFT with the provided `rank` and `rfft_fn`."""
    # Generate a simple mask like [1.0, 2.0, ..., 2.0, 1.0] for even-length FFTs
    # and [1.0, 2.0, ..., 2.0] for odd-length FFTs. To reduce extra ops in the
    # graph we special-case the situation where the FFT length and last
    # dimension of the input are known at graph construction time.
    fft_length = op.inputs[1]
    fft_length_static = _tensor_util.constant_value(fft_length)
    if fft_length_static is not None:
      fft_length = fft_length_static
    real_dtype = grad.dtype
    if real_dtype == _dtypes.float32:
      complex_dtype = _dtypes.complex64
    elif real_dtype == _dtypes.float64:
      complex_dtype = _dtypes.complex128
    is_odd = _math_ops.mod(fft_length[-1], 2)
    input_last_dimension = _array_ops.shape(op.inputs[0])[-1]
    mask = _array_ops.concat(
        [[1.0], 2.0 * _array_ops.ones(
            [input_last_dimension - 2 + is_odd], real_dtype),
         _array_ops.ones([1 - is_odd], real_dtype)], 0)

    rsize = _math_ops.reciprocal(_math_ops.cast(
        _fft_size_for_grad(grad, rank), real_dtype))

    # The gradient of IRFFT is the RFFT of the incoming gradient times a scaling
    # factor and a mask. The mask scales the gradient for the Hermitian
    # symmetric components of the RFFT by a factor of two, since these
    # components are de-duplicated in the RFFT.
    the_rfft = rfft_fn(grad, fft_length)
    return the_rfft * _math_ops.cast(rsize * mask, complex_dtype), None

  return _grad


@tf_export("signal.fftshift")
@dispatch.add_dispatch_support
def fftshift(x, axes=None, name=None):
  # [FUNCTION DOC]
  # Public op signal.fftshift: rolls the tensor so the zero-frequency component moves to the center.
  # Inputs: tensor x; axes (an integer, a list of integers possibly negative, or nothing/null for
  # all axes, which needs a static rank); optional name. Each shifted axis is rolled by half its
  # size (rounded down). Returns the rolled tensor. Adds graph ops; no other side effects.
  """Shift the zero-frequency component to the center of the spectrum.

  This function swaps half-spaces for all axes listed (defaults to all).
  Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.

  @compatibility(numpy)
  Equivalent to numpy.fft.fftshift.
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html
  @end_compatibility

  For example:

  ```python
  x = tf.signal.fftshift([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
  x.numpy() # array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
  ```

  Args:
    x: `Tensor`, input tensor.
    axes: `int` or shape `tuple`, optional Axes over which to shift.  Default is
      None, which shifts all axes.
    name: An optional name for the operation.

  Returns:
    A `Tensor`, The shifted tensor.
  """
  with _ops.name_scope(name, "fftshift") as name:
    x = _ops.convert_to_tensor(x)
    if axes is None:
      axes = tuple(range(x.shape.ndims))
      shift = _array_ops.shape(x) // 2
    elif isinstance(axes, int):
      shift = _array_ops.shape(x)[axes] // 2
    else:
      rank = _array_ops.rank(x)
      # allows negative axis
      axes = _array_ops.where(_math_ops.less(axes, 0), axes + rank, axes)
      shift = _array_ops.gather(_array_ops.shape(x), axes) // 2

    return manip_ops.roll(x, shift, axes, name)


@tf_export("signal.ifftshift")
@dispatch.add_dispatch_support
def ifftshift(x, axes=None, name=None):
  # [FUNCTION DOC]
  # Public op signal.ifftshift: inverse of fftshift. Rolls each selected axis by minus half its
  # size (rounded down), which differs from the forward shift for odd lengths.
  # Inputs: tensor x; axes (integer, list of possibly negative integers, or null for all axes,
  # which needs a static rank); optional name. Returns the rolled tensor.
  """The inverse of fftshift.

  Although identical for even-length x,
  the functions differ by one sample for odd-length x.

  @compatibility(numpy)
  Equivalent to numpy.fft.ifftshift.
  https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.ifftshift.html
  @end_compatibility

  For example:

  ```python
  x = tf.signal.ifftshift([[ 0.,  1.,  2.],[ 3.,  4., -4.],[-3., -2., -1.]])
  x.numpy() # array([[ 4., -4.,  3.],[-2., -1., -3.],[ 1.,  2.,  0.]])
  ```

  Args:
    x: `Tensor`, input tensor.
    axes: `int` or shape `tuple` Axes over which to calculate. Defaults to None,
      which shifts all axes.
    name: An optional name for the operation.

  Returns:
    A `Tensor`, The shifted tensor.
  """
  with _ops.name_scope(name, "ifftshift") as name:
    x = _ops.convert_to_tensor(x)
    if axes is None:
      axes = tuple(range(x.shape.ndims))
      shift = -(_array_ops.shape(x) // 2)
    elif isinstance(axes, int):
      shift = -(_array_ops.shape(x)[axes] // 2)
    else:
      rank = _array_ops.rank(x)
      # allows negative axis
      axes = _array_ops.where(_math_ops.less(axes, 0), axes + rank, axes)
      shift = -(_array_ops.gather(_array_ops.shape(x), axes) // 2)

    return manip_ops.roll(x, shift, axes, name)


_ops.RegisterGradient("RFFT")(_rfft_grad_helper(1, irfft))
_ops.RegisterGradient("IRFFT")(_irfft_grad_helper(1, rfft))
_ops.RegisterGradient("RFFT2D")(_rfft_grad_helper(2, irfft2d))
_ops.RegisterGradient("IRFFT2D")(_irfft_grad_helper(2, rfft2d))
