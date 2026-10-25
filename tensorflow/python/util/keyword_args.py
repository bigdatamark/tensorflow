# [MODULE DOC]
# HELPER: provides a decorator that forces a function to be called with keyword arguments
# only, to prevent argument-order mistakes. Uses the shared decorator validation helper.
# No import-time side effects.
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

"""Keyword args functions."""

import functools

from tensorflow.python.util import decorator_utils


def keyword_args_only(func):
  # [FUNCTION DOC]
  # Decorator factory that forces a function to be called with keyword arguments only.
  # Input: func, a callable (validated immediately; a value error is raised if it is not
  # callable). Output: a wrapper that preserves the wrapped function's metadata. The wrapper
  # raises a value error naming the function and the offending positional arguments if called
  # with any positional argument; otherwise it forwards all keyword arguments and returns the
  # wrapped function's result. No other side effects.
  """Decorator for marking specific function accepting keyword args only.

  This decorator raises a `ValueError` if the input `func` is called with any
  non-keyword args. This prevents the caller from providing the arguments in
  wrong order.

  Args:
    func: The function or method needed to be decorated.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If `func` is not callable.
  """

  decorator_utils.validate_callable(func, "keyword_args_only")
  @functools.wraps(func)
  def new_func(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper produced by keyword_args_only. Inputs: any positional and keyword
    # arguments. Raises a value error if any positional argument is present; otherwise calls
    # the wrapped function with the keyword arguments and returns its result.
    """Keyword args only wrapper."""
    if args:
      raise ValueError(
          f"The function {func.__name__} only accepts keyword arguments. "
          "Do not pass positional arguments. Received the following positional "
          f"arguments: {args}")
    return func(**kwargs)
  return new_func
