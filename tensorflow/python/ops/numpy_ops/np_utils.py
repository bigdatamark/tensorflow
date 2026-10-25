# [MODULE DOC]
# HELPER: internal utility module shared by the numpy-compatibility (tf.experimental.numpy) ops.
# Provides dtype conversion helpers, axis canonicalization, "static-if-possible" versions of
# arithmetic/logical/reduction/conditional operations (they compute eagerly with numpy when the
# value is statically known, otherwise fall back to TensorFlow ops), result-type promotion
# helpers, and the np_doc / np_doc_only decorators that attach NumPy-derived docstrings (with a
# link or inlined text) to tf-numpy functions, optionally checking signature compatibility.
# Import-time side effects: reads the environment variables controlling documentation form
# (TF_NP_DOC_FORM), link checking (TF_NP_CHECK_LINK) and signature mismatch strictness
# (TF_NP_SIG_MISMATCH_IS_ERROR) into module-level globals, and registers finfo, result_type and
# promote_types as public experimental numpy API symbols.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for internal use."""
# pylint: disable=g-direct-tensorflow-import

import inspect
import numbers
import os
import re

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.types import core
from tensorflow.python.util import nest
from tensorflow.python.util import tf_export


def _canonicalize_axis(axis, rank):
  # [FUNCTION DOC]
  # HELPER: axis normalization for the numpy ops.
  # Converts one possibly negative axis (integer or scalar tensor) into its non-negative
  # equivalent for a tensor of the given rank (integer or scalar tensor). Delegates to the
  # plural version and returns the single converted axis. No range validation is done.
  return _canonicalize_axes([axis], rank)[0]


def _canonicalize_axes(axes, rank):
  # [FUNCTION DOC]
  # HELPER: axis normalization for the numpy ops.
  # Takes an iterable of axes (integers or scalar tensors) and a rank (integer or tensor) and
  # returns a list in which every negative axis has the rank added to it. If the rank is a
  # tensor with no static value, a conditional (static if possible, else a TF cond) is used per
  # axis; otherwise plain arithmetic is used. Out-of-range axes are not detected.
  rank = _maybe_static(rank)

  if isinstance(rank, core.Tensor):
    canonicalizer = lambda axis: cond(  # pylint: disable=g-long-lambda
        axis < 0, lambda: axis + rank, lambda: axis
    )
  else:
    canonicalizer = lambda axis: axis + rank if axis < 0 else axis

  return [canonicalizer(axis) for axis in axes]


def _supports_signature():
  # [FUNCTION DOC]
  # HELPER: feature check for signature introspection.
  # KNOWN ISSUE: effectively dead code; the inspect module always provides signature support on
  # supported Python versions, so this always returns true (and it is not used in this file).
  # Takes no arguments and returns a boolean telling whether the inspect module has a signature
  # function. No side effects.
  return hasattr(inspect, 'signature')


def _to_tf_type(dtype):
  # [FUNCTION DOC]
  # HELPER: dtype conversion to TensorFlow.
  # Accepts a Python type, numpy dtype or TensorFlow DType and returns the equivalent TensorFlow
  # DType. Raises a type error if the value cannot be interpreted as a dtype. No side effects.
  """Converts a native python or numpy type to TF DType.

  Args:
    dtype: Could be a python type, a numpy type or a TF DType.

  Returns:
    A tensorflow `DType`.
  """
  return dtypes.as_dtype(dtype)


def _to_numpy_type(dtype):
  # [FUNCTION DOC]
  # HELPER: dtype conversion to numpy.
  # KNOWN ISSUE: the docstring says the result is a numpy dtype, but for a TensorFlow DType the
  # function returns the numpy scalar type class (as_numpy_dtype), while for anything else it
  # returns a numpy dtype instance.
  # Input is a TensorFlow DType, Python type, string or numpy dtype. Raises a type error when
  # numpy cannot interpret the input. No side effects.
  """Converts a native python or TF DType to numpy type.

  Args:
    dtype: Could be a python type, a numpy type or a TF DType.

  Returns:
    A NumPy `dtype`.
  """
  if isinstance(dtype, dtypes.DType):
    return dtype.as_numpy_dtype
  return np.dtype(dtype)


def isscalar(val):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring promises a boolean, but when val is a tensor of unknown rank the
  # function returns a boolean tensor (result of an equality op on the dynamic rank) instead.
  # Input is any value. A tf-numpy ndarray is first unwrapped to its underlying data. For tensors
  # with a known rank returns whether the rank is zero; for unknown rank returns a tensor
  # computing that test; for non-tensors returns numpy's scalar test result. No side effects.
  """Returns whether `val` is a scalar value or scalar Tensor."""
  if isinstance(val, np_arrays.ndarray):
    val = val.data
  if isinstance(val, core.Tensor):
    ndims = val.shape.ndims
    if ndims is not None:
      return ndims == 0
    else:
      return math_ops.equal(array_ops.rank(val), 0)
  else:
    return np.isscalar(val)


def _has_docstring(f):
  # [FUNCTION DOC]
  # HELPER: used by the docstring-building functions.
  # KNOWN ISSUE: the name suggests a boolean result, but the function returns the falsy value of
  # f itself (for example None) or the docstring text itself when present.
  # Input is any object (possibly None). Returns a truthy value (the docstring string) when the
  # object has a non-empty string docstring, otherwise a falsy value. No side effects.
  return (
      f and hasattr(f, '__doc__') and isinstance(f.__doc__, str) and f.__doc__
  )


def _add_blank_line(s):
  # [FUNCTION DOC]
  # HELPER: docstring formatting.
  # Takes a string and returns it ending with a blank line: one newline is appended if it already
  # ends with a newline, otherwise two are appended. No side effects.
  if s.endswith('\n'):
    return s + '\n'
  else:
    return s + '\n\n'


def _np_signature(f):
  # [FUNCTION DOC]
  # HELPER: signature lookup for the np_doc decorator.
  # Input is a callable (possibly a numpy ufunc) or nothing. Returns an inspect signature object,
  # or nothing if f is nothing, if inspect cannot produce a signature (value error), or if the
  # inspect module lacks signature support. For a ufunc, a synthetic signature is built: the
  # positional-only inputs (named x, or x1..xn), extra positional-only outputs when there are
  # several outputs, an out parameter, and keyword-only parameters (where, casting, order, dtype,
  # subok, signature, extobj) with numpy's default values. No side effects.
  """An enhanced inspect.signature that can handle numpy.ufunc."""
  # TODO(wangpeng): consider migrating away from inspect.signature.
  # inspect.signature is supported in Python 3.3.
  if not hasattr(inspect, 'signature'):
    return None
  if f is None:
    return None
  if not isinstance(f, np.ufunc):
    try:
      return inspect.signature(f)
    except ValueError:
      return None

  def names_from_num(prefix, n):
    # [FUNCTION DOC]
    # HELPER: builds parameter names for the synthetic ufunc signature.
    # Given a prefix string and a count, returns an empty list for a non-positive count, a list
    # holding just the prefix for a count of one, otherwise the prefix followed by numbers
    # starting at one. Pure function.
    if n <= 0:
      return []
    elif n == 1:
      return [prefix]
    else:
      return [prefix + str(i + 1) for i in range(n)]

  input_names = names_from_num('x', f.nin)
  output_names = names_from_num('out', f.nout)
  keyword_only_params = [
      ('where', True),
      ('casting', 'same_kind'),
      ('order', 'K'),
      ('dtype', None),
      ('subok', True),
      ('signature', None),
      ('extobj', None),
  ]
  params = []
  params += [
      inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY)
      for name in input_names
  ]
  if f.nout > 1:
    params += [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY, default=None)
        for name in output_names
    ]
  params += [
      inspect.Parameter(
          'out',
          inspect.Parameter.POSITIONAL_OR_KEYWORD,
          default=None if f.nout == 1 else (None,) * f.nout,
      )
  ]
  params += [
      inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
      for name, default in keyword_only_params
  ]
  return inspect.Signature(params)


# Python 2 doesn't allow keyword-only argument. Python prior to 3.8 doesn't
# allow positional-only argument. So we conflate positional-only, keyword-only
# and positional-or-keyword arguments here.
def _is_compatible_param_kind(a, b):
  # [FUNCTION DOC]
  # HELPER: signature comparison for the np_doc decorator.
  # Takes two inspect parameter kinds and returns true if they are equal after treating
  # positional-only and keyword-only as positional-or-keyword. Pure function.
  def relax(k):
    # [FUNCTION DOC]
    # HELPER: maps positional-only and keyword-only kinds to positional-or-keyword and leaves
    # other kinds unchanged. Returns the relaxed parameter kind. Pure function.
    if k in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.KEYWORD_ONLY):
      return inspect.Parameter.POSITIONAL_OR_KEYWORD
    return k

  return relax(a) == relax(b)


def _prepare_np_fun_name_and_fun(np_fun_name, np_fun):
  # [FUNCTION DOC]
  # HELPER: argument normalization for np_doc and np_doc_only.
  # Inputs: an optional numpy function name (string) and an optional numpy function (not a
  # string). If the function is missing it is looked up by name in numpy (set to nothing when
  # numpy has no such attribute); if the name is missing it is taken from the function's name.
  # Returns the pair (name, function). Fails with assertion errors if the name is not a string,
  # the function is a string, or both are missing. No side effects.
  """Mutually propagates information between `np_fun_name` and `np_fun`.

  If one is None and the other is not, we'll try to make the former not None in
  a best effort.

  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: the numpy function whose docstring will be used.

  Returns:
    Processed `np_fun_name` and `np_fun`.
  """
  if np_fun_name is not None:
    assert isinstance(np_fun_name, str)
  if np_fun is not None:
    assert not isinstance(np_fun, str)
  if np_fun is None:
    assert np_fun_name is not None
    try:
      np_fun = getattr(np, str(np_fun_name))
    except AttributeError:
      np_fun = None
  if np_fun_name is None:
    assert np_fun is not None
    np_fun_name = np_fun.__name__
  return np_fun_name, np_fun


def _np_doc_helper(
    f, np_f, np_fun_name=None, unsupported_params=None, link=None
):
  # [FUNCTION DOC]
  # HELPER: builds the final docstring for np_doc and np_doc_only.
  # Inputs: the tf function f (its own docstring is reused), the numpy function np_f, an optional
  # name (defaults to np_f's name), an optional list of unsupported argument names, and an
  # optional link specification. Returns a string: a "TensorFlow variant of" header, an
  # unsupported-arguments line if any, f's own docstring, then the numpy reference added by
  # _add_np_doc. Asserts that np_f or the name is given. May raise a value error from link
  # checking (network access). No other side effects.
  """Helper to get docs."""
  assert np_f or np_fun_name
  if not np_fun_name:
    np_fun_name = np_f.__name__
  doc = "TensorFlow variant of NumPy's `%s`.\n\n" % np_fun_name
  if unsupported_params:
    doc += (
        'Unsupported arguments: '
        + ', '.join('`' + name + '`' for name in unsupported_params)
        + '.\n\n'
    )
  if _has_docstring(f):
    doc += f.__doc__
    doc = _add_blank_line(doc)
  # TODO(wangpeng): Re-enable the following and choose inlined vs. link to numpy
  #   doc according to some global switch.
  doc = _add_np_doc(doc, np_fun_name, np_f, link=link)
  return doc


_np_doc_form = os.getenv('TF_NP_DOC_FORM', 'stable')


def get_np_doc_form():
  # [FUNCTION DOC]
  # Returns the module-level setting (initially from the TF_NP_DOC_FORM environment variable,
  # default stable) that selects how numpy documentation is referenced. No side effects.
  """Gets the form of the original numpy docstrings.

  Returns:
    See `set_np_doc_form` for the list of valid values.
  """
  return _np_doc_form


def set_np_doc_form(value):
  # [FUNCTION DOC]
  # Sets the module-level numpy documentation form to value (inlined, stable, dev, or a numpy
  # version string). Returns nothing. The value is not validated here; an unrecognized string
  # later results in no link being generated. Only affects docstrings of functions decorated
  # after the call. Side effect: mutates the module-level global.
  r"""Selects the form of the original numpy docstrings.

  This function sets a global variable that controls how a tf-numpy symbol's
  docstring should refer to the original numpy docstring. If `value` is
  `'inlined'`, the numpy docstring will be verbatim copied into the tf-numpy
  docstring. Otherwise, a link to the original numpy docstring will be
  added. Which numpy version the link points to depends on `value`:
  * `'stable'`: the current stable version;
  * `'dev'`: the current development version;
  * pattern `\d+(\.\d+(\.\d+)?)?`: `value` will be treated as a version number,
    e.g. '1.16'.

  Args:
    value: the value to set the global variable to.
  """
  global _np_doc_form
  _np_doc_form = value


class Link:
  # [CLASS DOC]
  # Marker value for the link argument of np_doc: wraps a complete URL string (attribute value)
  # to be used verbatim as the numpy documentation link. Plain data holder, no inheritance.

  def __init__(self, v):
    # [FUNCTION DOC]
    # Stores the given URL string v in the value attribute. Returns nothing.
    self.value = v


class AliasOf:
  # [CLASS DOC]
  # Marker value for the link argument of np_doc: wraps the name (attribute value) of a different
  # numpy function whose documentation page should be linked instead of the decorated name.
  # Plain data holder, no inheritance.

  def __init__(self, v):
    # [FUNCTION DOC]
    # Stores the given numpy function name v in the value attribute. Returns nothing.
    self.value = v


class NoLink:
  # [CLASS DOC]
  # Marker class for the link argument of np_doc: an instance means no link to numpy
  # documentation should be added. Carries no state or behavior.
  pass


def generate_link(flag, np_fun_name):
  # [FUNCTION DOC]
  # Builds the URL of the numpy reference page for a function. flag is the documentation form:
  # dev gives the development docs, stable gives the stable docs, a version-number string gives
  # that version's docs; any other value gives nothing. np_fun_name is the numpy function name
  # substituted into the URL. Returns the URL string or nothing. No side effects.
  """Generates link from numpy function name.

  Args:
    flag: the flag to control link form. See `set_np_doc_form`.
    np_fun_name: the numpy function name.

  Returns:
    A string.
  """
  # Only adds link in this case
  if flag == 'dev':
    template = 'https://numpy.org/devdocs/reference/generated/numpy.%s.html'
  elif flag == 'stable':
    template = 'https://numpy.org/doc/stable/reference/generated/numpy.%s.html'
  elif re.match(r'\d+(\.\d+(\.\d+)?)?$', flag):
    # `flag` is the version number
    template = f'https://numpy.org/doc/{flag}/reference/generated/numpy.%s.html'
  else:
    return None
  return template % np_fun_name


_is_check_link = os.getenv('TF_NP_CHECK_LINK', 'False') in ('True', 'true', '1')


def is_check_link():
  # [FUNCTION DOC]
  # Returns the module-level flag telling whether generated numpy documentation links are
  # verified over the network (initially from TF_NP_CHECK_LINK, default off). No side effects.
  return _is_check_link


def set_check_link(value):
  # [FUNCTION DOC]
  # Sets the module-level link-checking flag to value (truthy to enable). Returns nothing.
  # Side effect: mutates the module-level global.
  global _is_check_link
  _is_check_link = value


def _add_np_doc(doc, np_fun_name, np_f, link):
  # [FUNCTION DOC]
  # HELPER: appends the numpy reference to a docstring for _np_doc_helper.
  # KNOWN ISSUE: the docstring says np_f and link are optional, but they are required positional
  # parameters (np_f may be nothing, link may be nothing).
  # Inputs: the docstring so far, the numpy function name, the numpy function and a link
  # specification (nothing, an alias marker, an explicit link marker, or the no-link marker).
  # Behavior depends on the global doc form: if inlined, the numpy docstring (with interactive
  # prompts rewritten) is appended when it exists; if another string, a markdown link sentence
  # is appended using the chosen URL (nothing is appended when no URL results). If link checking
  # is enabled, the URL is requested over the network and a value error is raised on any
  # non-success status. Returns the extended docstring.
  """Appends the numpy docstring to `doc`, according to `set_np_doc_form`.

  See `set_np_doc_form` for how it controls the form of the numpy docstring.

  Args:
    doc: the docstring to be appended to.
    np_fun_name: the name of the numpy function.
    np_f: (optional) the numpy function.
    link: (optional) which link to use. See `np_doc` for details.

  Returns:
    `doc` with numpy docstring appended.
  """
  flag = get_np_doc_form()
  if flag == 'inlined':
    if _has_docstring(np_f):
      doc += 'Documentation for `numpy.%s`:\n\n' % np_fun_name
      # TODO(wangpeng): It looks like code snippets in numpy doc don't work
      # correctly with doctest. Fix that and remove the reformatting of the np_f
      # comment.
      doc += np_f.__doc__.replace('>>>', '>')
  elif isinstance(flag, str):
    if link is None:
      url = generate_link(flag, np_fun_name)
    elif isinstance(link, AliasOf):
      url = generate_link(flag, link.value)
    elif isinstance(link, Link):
      url = link.value
    else:
      url = None
    if url is not None:
      if is_check_link():
        # Imports locally because some builds may not have `requests`
        import requests  # pylint: disable=g-import-not-at-top

        r = requests.head(url)
        if r.status_code != 200:
          raise ValueError(
              f'Check link failed at [{url}] with status code {r.status_code}. '
              f'Argument `np_fun_name` is {np_fun_name}.'
          )
      doc += 'See the NumPy documentation for [`numpy.%s`](%s).' % (
          np_fun_name,
          url,
      )
  return doc


_is_sig_mismatch_an_error = os.getenv(
    'TF_NP_SIG_MISMATCH_IS_ERROR', 'False'
) in ('True', 'true', '1')


def is_sig_mismatch_an_error():
  # [FUNCTION DOC]
  # Returns the module-level flag telling whether np_doc raises a type error when the decorated
  # function's signature disagrees with numpy's (initially from
  # TF_NP_SIG_MISMATCH_IS_ERROR, default off). No side effects.
  return _is_sig_mismatch_an_error


def set_is_sig_mismatch_an_error(value):
  # [FUNCTION DOC]
  # Sets the module-level signature-mismatch strictness flag to value. Returns nothing.
  # Side effect: mutates the module-level global.
  global _is_sig_mismatch_an_error
  _is_sig_mismatch_an_error = value


def np_doc(np_fun_name, np_fun=None, unsupported_params=None, link=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the unsupported_params list supplied by the caller is mutated by the returned
  # decorator (numpy parameters missing from the decorated function are appended), so reusing one
  # list for several decorated functions accumulates names; the docstring does not mention this.
  # Decorator factory. Inputs: the numpy function name, an optional numpy function (looked up by
  # name if absent), an optional list of parameter names known to be unsupported, and an optional
  # link specification (nothing, alias marker, link marker, or no-link marker). The numpy
  # signature is computed once. Returns a decorator that compares signatures (raising a type
  # error on mismatch only when strict mode is on), records numpy parameters missing from the
  # decorated function as unsupported, replaces the function's docstring with the generated one
  # and returns the same function. Asserts on invalid name/function arguments.
  """Attachs numpy docstring to a function.

  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: (optional) the numpy function whose docstring will be used.
    unsupported_params: (optional) the list of parameters not supported by
      tf.numpy.
    link: (optional) which link to use. If `None`, a default link generated from
      `np_fun_name` will be used. If an instance of `AliasOf`, `link.value` will
      be used in place of `np_fun_name` for the link generation. If an instance
      of `Link`, `link.value` will be used as the whole link. If an instance of
      `NoLink`, no link will be added.

  Returns:
    A function decorator that attaches the docstring from `np_fun` to the
    decorated function.
  """
  np_fun_name_orig, np_fun_orig = np_fun_name, np_fun
  np_fun_name, np_fun = _prepare_np_fun_name_and_fun(np_fun_name, np_fun)
  np_sig = _np_signature(np_fun)
  if unsupported_params is None:
    unsupported_params = []

  def decorator(f):
    # [FUNCTION DOC]
    # Decorator returned by np_doc. Input is the function to document. When a numpy signature is
    # available, each parameter of f is checked against numpy's: unknown names, differing kinds
    # and differing default presence raise a type error only in strict mode (unknown names are
    # otherwise skipped). Numpy parameters absent from f are appended to the shared unsupported
    # list. Then f's docstring is replaced by the generated numpy-based docstring. Returns f.
    """The decorator."""
    if hasattr(inspect, 'signature') and np_sig is not None:
      try:
        sig = inspect.signature(f)
      except ValueError:
        sig = None
      if sig is not None:
        for name, param in sig.parameters.items():
          np_param = np_sig.parameters.get(name)
          if np_param is None:
            if is_sig_mismatch_an_error():
              raise TypeError(
                  f"Cannot find parameter {name} in the numpy function's "
                  'signature (which has these parameters: '
                  f'{list(np_sig.parameters.keys())}). Argument `np_fun_name` '
                  f'is {np_fun_name_orig}. Argument `np_fun` is {np_fun_orig}.'
              )
            else:
              continue
          if is_sig_mismatch_an_error() and not _is_compatible_param_kind(
              param.kind, np_param.kind
          ):
            raise TypeError(
                f'Parameter {name} is of kind {param.kind} while in numpy it '
                f'is of kind {np_param.kind}. Argument `np_fun_name` is '
                f'{np_fun_name_orig}. Argument `np_fun` is {np_fun_orig}.'
            )
          has_default = param.default != inspect.Parameter.empty
          np_has_default = np_param.default != inspect.Parameter.empty
          if is_sig_mismatch_an_error() and has_default != np_has_default:
            raise TypeError(
                'Parameter {} should{} have a default value. Argument '
                '`np_fun_name` is {}. Argument `np_fun` is {}.'.format(
                    name,
                    '' if np_has_default else ' not',
                    np_fun_name_orig,
                    np_fun_orig,
                )
            )
        for name in np_sig.parameters:
          if name not in sig.parameters:
            unsupported_params.append(name)
    f.__doc__ = _np_doc_helper(
        f,
        np_fun,
        np_fun_name=np_fun_name,
        unsupported_params=unsupported_params,
        link=link,
    )
    return f

  return decorator


def np_doc_only(np_fun_name, np_fun=None):
  # [FUNCTION DOC]
  # Decorator factory like np_doc but without any signature comparison and without unsupported
  # parameter reporting or link options. Inputs: numpy function name and optional numpy function.
  # Returns a decorator that overwrites the decorated function's docstring with the generated
  # numpy-based one. Asserts on invalid name/function arguments.
  """Attachs numpy docstring to a function.

  This differs from np_doc in that it doesn't check for a match in signature.

  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: (optional) the numpy function whose docstring will be used.

  Returns:
    A function decorator that attaches the docstring from `np_fun` to the
    decorated function.
  """
  np_fun_name, np_fun = _prepare_np_fun_name_and_fun(np_fun_name, np_fun)

  def decorator(f):
    # [FUNCTION DOC]
    # Decorator returned by np_doc_only. Replaces the docstring of f with the generated numpy
    # based docstring (side effect on f) and returns f.
    f.__doc__ = _np_doc_helper(f, np_fun, np_fun_name=np_fun_name)
    return f

  return decorator


# pylint: disable=g-short-docstring-punctuation,g-no-space-after-docstring-summary,g-docstring-missing-newline,g-doc-return-or-yield,g-doc-args
@tf_export.tf_export('experimental.numpy.finfo', v1=[])
@np_doc('finfo')
def finfo(dtype):
  # [FUNCTION DOC]
  # Public tf.experimental.numpy.finfo. Input is a TensorFlow or numpy dtype. Returns numpy's
  # machine-limits information object for the corresponding numpy type; no TensorFlow-specific
  # adjustments are made. Raises whatever numpy raises for non-float dtypes.
  """Note that currently it just forwards to the numpy namesake, while

  tensorflow and numpy dtypes may have different properties.
  """
  return np.finfo(_to_numpy_type(dtype))


# pylint: enable=g-short-docstring-punctuation,g-no-space-after-docstring-summary,g-docstring-missing-newline,g-doc-return-or-yield,g-doc-args


def _maybe_get_dtype(x):
  # [FUNCTION DOC]
  # HELPER: argument preparation for result_type.
  # Maps one argument to something numpy's type promotion can use: Python real numbers are
  # returned unchanged; indexed slices and TensorFlow tensors give their numpy dtype; TensorFlow
  # dtypes give their numpy type; anything else (including numpy arrays and dtype strings) is
  # returned unchanged. Raises a value error for lists and tuples. No side effects.
  """Returns a numpy type if available from x. Skips if x is numpy.ndarray."""
  # Don't put np.ndarray in this list, because np.result_type looks at the
  # value (not just dtype) of np.ndarray to decide the result type.
  if isinstance(x, numbers.Real):
    return x
  if isinstance(x, indexed_slices.IndexedSlices) or tensor_util.is_tf_type(x):
    return _to_numpy_type(x.dtype)
  if isinstance(x, dtypes.DType):
    return x.as_numpy_dtype
  if isinstance(x, (list, tuple)):
    raise ValueError(
        'Cannot find dtype for type inference from argument `x` of a sequence '
        f'type {type(x)}. For sequences, please call this function on each '
        'element individually.'
    )
  return x


@tf_export.tf_export('experimental.numpy.result_type', v1=[])
# Can't use np_doc because np.result_type is a builtin function.
@np_doc_only('result_type')
def result_type(*arrays_and_dtypes):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Public tf.experimental.numpy.result_type. Takes any number of arrays, tensors, dtypes, Python
  # scalars or dtype names (nested structures are flattened). When auto dtype conversion is
  # enabled it returns the dtype from the flexible-dtype promotion rules; otherwise each element
  # is prepared and the numpy-style promotion (restricted to TF-supported dtypes) is applied, using
  # an empty array when nothing is given. Raises a value error for sequences that survive flatten.
  if ops.is_auto_dtype_conversion_enabled():
    # Use auto dtype conversion semantics for type inference.
    dtype, _ = flexible_dtypes.result_type(*arrays_and_dtypes)
    return dtype
  arrays_and_dtypes = [
      _maybe_get_dtype(x) for x in nest.flatten(arrays_and_dtypes)
  ]
  if not arrays_and_dtypes:
    # If arrays_and_dtypes is an empty list, let numpy decide what the dtype is.
    arrays_and_dtypes = [np.asarray([])]
  return np_dtypes._result_type(*arrays_and_dtypes)  # pylint: disable=protected-access


def result_type_unary(a, dtype):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Computes the result dtype for a unary operation. If an explicit dtype is given (truthy) the
  # result type of that dtype is returned. Otherwise a string input yields the numpy string type,
  # a bytes input yields the numpy bytes type, and anything else goes through result_type of the
  # single input. Returns a dtype; no side effects.
  """Find the result type from a single input and a dtype."""
  if dtype:
    # We need to let np_utils.result_type decide the dtype, not tf.zeros_like
    return result_type(dtype)

  # np_utils.result_type treats string inputs as dtype strings, not as strings.
  # but for unary we want to treat it as a string input.
  if isinstance(a, str):
    return np.str_
  elif isinstance(a, bytes):
    return np.bytes_

  # TF and numpy has different interpretations of Python types such as
  # `float`, so we let `np_utils.result_type` decide.
  return result_type(a)


def _result_type_binary(t1, t2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # HELPER: fast path for binary ops.
  # Computes the promoted dtype of two arguments directly via the numpy dtype helper; if that
  # raises a value error (for example for sequences) it falls back to the general result_type.
  # Returns a dtype; no side effects.
  """A specialization of result_type for 2 arguments for performance reasons."""
  try:
    return np_dtypes._result_type(  # pylint: disable=protected-access
        _maybe_get_dtype(t1),
        _maybe_get_dtype(t2),
    )
  except ValueError:
    return result_type(t1, t2)


@tf_export.tf_export('experimental.numpy.promote_types', v1=[])
@np_doc('promote_types')
def promote_types(type1, type2):  # pylint: disable=missing-function-docstring
  # [FUNCTION DOC]
  # Public tf.experimental.numpy.promote_types. Takes two dtypes (TensorFlow, numpy or Python
  # types), converts them to numpy types, applies numpy's promotion and returns the result
  # canonicalized to a dtype TensorFlow supports. No side effects.
  type1 = _to_numpy_type(type1)
  type2 = _to_numpy_type(type2)
  return np_dtypes.canonicalize_dtype(np.promote_types(type1, type2))


def tf_broadcast(*args):
  # [FUNCTION DOC]
  # KNOWN ISSUE: inconsistent return type; with zero or one argument the args tuple is returned
  # unchanged (not broadcast to anything), otherwise a list is returned. The docstring does not
  # mention this.
  # Takes any number of tensors. Computes the common broadcast shape dynamically and returns a
  # list of the inputs broadcast to it. Raises an invalid-argument error if shapes are not
  # compatible. Adds graph ops when tracing.
  """Broadcast tensors.

  Args:
    *args: a list of tensors whose shapes are broadcastable against each other.

  Returns:
    Tensors broadcasted to the common shape.
  """
  if len(args) <= 1:
    return args
  sh = array_ops.shape(args[0])
  for arg in args[1:]:
    sh = array_ops.broadcast_dynamic_shape(sh, array_ops.shape(arg))
  return [array_ops.broadcast_to(arg, sh) for arg in args]


# TODO(wangpeng): Move the following functions to a separate file and check for
#   float dtypes in each of them.


def get_static_value(x):
  # [FUNCTION DOC]
  # Takes a tensor or value. Returns nothing when x is a tensor of floating or complex dtype (so
  # gradients are not broken); otherwise returns the constant value TensorFlow can determine
  # statically, or nothing if it is not statically known. No side effects.
  """A version of tf.get_static_value that returns None on float dtypes.

  It returns None on float dtypes in order to avoid breaking gradients.

  Args:
    x: a tensor.

  Returns:
    Same as `tf.get_static_value`, except that it returns None when `x` has a
    float dtype.
  """
  if isinstance(x, core.Tensor) and (x.dtype.is_floating or x.dtype.is_complex):
    return None
  return tensor_util.constant_value(x)


def _maybe_static(x):
  # [FUNCTION DOC]
  # HELPER: shared by the static-if-possible operations below.
  # Returns the statically known value of x when available (see get_static_value), otherwise x
  # itself. No side effects.
  value = get_static_value(x)
  if value is None:
    return x
  else:
    return value


# All the following functions exist becaues get_static_value can't handle
# their TF counterparts.


def cond(pred, true_fn, false_fn):
  # [FUNCTION DOC]
  # Conditional that avoids building graph control flow when possible. Inputs: a predicate
  # (boolean or boolean tensor) and two callbacks taking no arguments. If the predicate has a
  # static value only the chosen callback is called and its result returned; otherwise a
  # TensorFlow cond is built with both callbacks.
  """A version of tf.cond that tries to evaluate the condition."""
  v = get_static_value(pred)
  if v is None:
    return tf_cond.cond(pred, true_fn, false_fn)
  if v:
    return true_fn()
  else:
    return false_fn()


def add(a, b):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring names tf.add, but the sum is computed with Python addition on the
  # (possibly static) operands, so numpy broadcasting/dtype rules apply to static values.
  # Returns the sum of the two inputs, using their static values when known.
  """A version of tf.add that eagerly evaluates if possible."""
  return _maybe_static(a) + _maybe_static(b)


def subtract(a, b):
  # [FUNCTION DOC]
  # Returns a minus b, using the static values of the inputs when known (numpy semantics),
  # otherwise the tensors' own subtraction.
  """A version of tf.subtract that eagerly evaluates if possible."""
  return _maybe_static(a) - _maybe_static(b)


def greater(a, b):
  # [FUNCTION DOC]
  # Returns whether a is greater than b elementwise, using static values of the inputs when
  # known, otherwise tensor comparison.
  """A version of tf.greater that eagerly evaluates if possible."""
  return _maybe_static(a) > _maybe_static(b)


def greater_equal(a, b):
  # [FUNCTION DOC]
  # Returns whether a is greater than or equal to b, using static values of the inputs when
  # known, otherwise tensor comparison.
  """A version of tf.greater_equal that eagerly evaluates if possible."""
  return _maybe_static(a) >= _maybe_static(b)


def less_equal(a, b):
  # [FUNCTION DOC]
  # Returns whether a is less than or equal to b, using static values of the inputs when known,
  # otherwise tensor comparison.
  """A version of tf.less_equal that eagerly evaluates if possible."""
  return _maybe_static(a) <= _maybe_static(b)


def logical_and(a, b):
  # [FUNCTION DOC]
  # KNOWN ISSUE: when a is not static, the bitwise and operator is used (not a true logical and),
  # and a scalar static a short-circuits, returning a itself or the static value of b rather than
  # a strict boolean, so the result is not guaranteed to be boolean typed.
  # Combines two boolean values/tensors. If a is statically known: a scalar true gives b (static
  # if possible), a scalar false gives a; an array gives elementwise and with b. Otherwise a is
  # combined with b by and.
  """A version of tf.logical_and that eagerly evaluates if possible."""
  a_value = get_static_value(a)
  if a_value is not None:
    if np.isscalar(a_value):
      if a_value:
        return _maybe_static(b)
      else:
        return a_value
    else:
      return a_value & _maybe_static(b)
  else:
    return a & _maybe_static(b)


def logical_or(a, b):
  # [FUNCTION DOC]
  # KNOWN ISSUE: when a is not static, the bitwise or operator is used (not a true logical or),
  # and a scalar static a short-circuits, returning a itself or the static value of b rather than
  # a strict boolean, so the result is not guaranteed to be boolean typed.
  # Combines two boolean values/tensors. If a is statically known: a scalar true gives a, a scalar
  # false gives b (static if possible); an array gives elementwise or with b. Otherwise a is
  # combined with b by or.
  """A version of tf.logical_or that eagerly evaluates if possible."""
  a_value = get_static_value(a)
  if a_value is not None:
    if np.isscalar(a_value):
      if a_value:
        return a_value
      else:
        return _maybe_static(b)
    else:
      return a_value | _maybe_static(b)
  else:
    return a | _maybe_static(b)


def getitem(a, slice_spec):
  # [FUNCTION DOC]
  # Indexes a (static value if known, otherwise the tensor itself) with the given index or slice
  # specification and returns the selected element or slice. Errors come from the indexing.
  """A version of __getitem__ that eagerly evaluates if possible."""
  return _maybe_static(a)[slice_spec]


def reduce_all(input_tensor, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # Logical all-reduction. Inputs: a tensor/array, optional axis (default all), and keepdims flag
  # (default false). Uses the TensorFlow reduction when the input is not static, otherwise the
  # numpy all method on the static value. Returns the reduced boolean value or tensor.
  """A version of tf.reduce_all that eagerly evaluates if possible."""
  v = get_static_value(input_tensor)
  if v is None:
    return math_ops.reduce_all(input_tensor, axis=axis, keepdims=keepdims)
  else:
    return v.all(axis=axis, keepdims=keepdims)


def reduce_any(input_tensor, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # Logical any-reduction. Inputs: a tensor/array, optional axis (default all), and keepdims flag
  # (default false). Uses the TensorFlow reduction when the input is not static, otherwise the
  # numpy any method on the static value. Returns the reduced boolean value or tensor.
  """A version of tf.reduce_any that eagerly evaluates if possible."""
  v = get_static_value(input_tensor)
  if v is None:
    return math_ops.reduce_any(input_tensor, axis=axis, keepdims=keepdims)
  else:
    return v.any(axis=axis, keepdims=keepdims)


def tf_rank(t):
  # [FUNCTION DOC]
  # Returns the rank of tensor t: a plain integer when the static shape has a known rank,
  # otherwise a scalar tensor computed dynamically. No other side effects.
  r = t.shape.rank
  if r is not None:
    return r
  return array_ops.rank(t)
