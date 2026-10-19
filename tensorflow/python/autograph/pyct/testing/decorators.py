# [MODULE DOC]
# HELPER: fixture module of decorators used by autograph's inspection and conversion tests.
# Offers three decorator shapes: one that wraps with metadata preservation, one that wraps
# without it (a standalone wrapper), and a decorator factory that returns a decorator producing a
# wrapper. Each wrapper simply forwards all positional and keyword arguments to the wrapped
# function and returns its result. No import-time side effects.
# NOTE: the functions here are intentionally left without per-item documentation comments,
# because tests inspect their source text and line numbers.
#
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
"""Module with test decorators."""

import functools


def wrapping_decorator(f):

  @functools.wraps(f)
  def wrapper(*args, **kwargs):
    return f(*args, **kwargs)

  return wrapper


def standalone_decorator(f):

  def standalone_wrapper(*args, **kwargs):
    return f(*args, **kwargs)

  return standalone_wrapper


def functional_decorator():

  def decorator(f):

    def functional_wrapper(*args, **kwargs):
      return f(*args, **kwargs)

    return functional_wrapper

  return decorator
