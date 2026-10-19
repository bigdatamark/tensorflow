# [MODULE DOC]
# HELPER: Helper module providing a decorator-aware drop-in replacement for the standard
# library context manager decorator, used wherever Keras defines generator-based context managers.
# Exposes a single function, contextmanager. No import-time side effects.
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
"""TFDecorator-aware replacements for the contextlib module."""

import contextlib as _contextlib

from tensorflow.python.util import tf_decorator


def contextmanager(target):
  # [FUNCTION DOC]
  # HELPER: Wraps generator functions as context managers while keeping TensorFlow decorator
  # introspection working.
  # Takes target, a generator function, turns it into a context manager factory using the
  # standard library decorator, then re-wraps the result with the TensorFlow decorator utility
  # so the original target stays discoverable (unwrappable) and the decorator is named
  # contextmanager. Returns the wrapped callable, usable in a with statement. No side effects.
  """A tf_decorator-aware wrapper for `contextlib.contextmanager`.

  Usage is identical to `contextlib.contextmanager`.

  Args:
    target: A callable to be wrapped in a contextmanager.
  Returns:
    A callable that can be used inside of a `with` statement.
  """
  context_manager = _contextlib.contextmanager(target)
  return tf_decorator.make_decorator(target, context_manager, 'contextmanager')
