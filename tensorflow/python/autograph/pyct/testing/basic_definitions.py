# [MODULE DOC]
# HELPER: fixture module for autograph's source-inspection and origin-tracking tests.
# Defines small sample entities (simple function, function with a nested function, function that
# prints, a lambda, a class with two methods, a function with a multi-line call, a pass-through
# decorator and a function wrapped twice by it) whose source text and line positions are
# inspected by tests of the inspection utilities and origin information. No import-time side
# effects beyond defining these names.
# NOTE: the individual functions and classes here are intentionally left without per-item
# documentation comments, because tests inspect their source text and line numbers.
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
"""Module with basic entity definitions for testing."""


def simple_function(x):
  """Docstring."""
  return x  # comment


def nested_functions(x):
  """Docstring."""

  def inner_fn(y):
    return y

  return inner_fn(x)


def function_with_print():
  print('foo')


simple_lambda = lambda: None


class SimpleClass(object):

  def simple_method(self):
    return self

  def method_with_print(self):
    print('foo')


def function_with_multiline_call(x):
  """Docstring."""
  return range(
      x,
      x + 1,
  )


def basic_decorator(f):
  return f


@basic_decorator
@basic_decorator
def decorated_function(x):
  if x > 0:
    return 1
  return 2
