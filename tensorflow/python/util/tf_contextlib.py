# [MODULE DOC]
# HELPER: TFDecorator-aware replacement for the standard library context manager decorator.
# Exposes contextmanager, which behaves like the standard one but registers the result with
# tf_decorator so the original function stays discoverable (for example by documentation and
# introspection tools). No import-time side effects besides defining a type variable.

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
from collections.abc import Callable, Iterator
import contextlib as _contextlib

from typing import ContextManager, TypeVar

from tensorflow.python.util import tf_decorator

_T = TypeVar('_T')


def contextmanager(
    target: Callable[..., Iterator[_T]],
) -> Callable[..., ContextManager[_T]]:
  # [FUNCTION DOC]
  # Decorator turning a generator function into a context manager factory, keeping decorator
  # metadata. Input: target (generator function yielding once). Output: a callable returning a
  # context manager, wrapped by tf_decorator with the name contextmanager so the target remains
  # reachable. Side effects: none beyond creating the wrapper; errors from the standard decorator
  # or from the decorator-wrapping helper propagate.
  """A tf_decorator-aware wrapper for `contextlib.contextmanager`.

  Usage is identical to `contextlib.contextmanager`.

  Args:
    target: A callable to be wrapped in a contextmanager.
  Returns:
    A callable that can be used inside of a `with` statement.
  """
  context_manager = _contextlib.contextmanager(target)
  return tf_decorator.make_decorator(target, context_manager, 'contextmanager')
