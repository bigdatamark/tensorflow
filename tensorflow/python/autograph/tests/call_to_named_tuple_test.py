# [MODULE DOC]
# Reference test for autograph handling of named tuples: constructing a named tuple type inside
# the converted function, receiving a named tuple type as an argument, and calling a method on a
# subclass of a named tuple. Native and compiled results are compared with a Python number and a
# tensor. No import-time side effects.
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
"""Simple call to construct a namedtuple."""

import collections

import tensorflow.compat.v1 as tf

from tensorflow.python.autograph.tests import reference_test_base


def inline_namedtuple(x):
  # [FUNCTION DOC]
  # Function under test: creates a named tuple type with fields a and b, instantiates it with a
  # equal to one and b equal to x, and returns the instance.
  nt = collections.namedtuple('TestNamedTuple', ('a', 'b'))
  n = nt(a=1, b=x)
  return n


def external_namedtuple(x, nt):
  # [FUNCTION DOC]
  # Function under test: instantiates the supplied named tuple type nt with a equal to one and b
  # equal to x, and returns the instance.
  return nt(a=1, b=x)


class NamedTupleSubclass(collections.namedtuple('TestNamedTuple', ('a',))):
  # [CLASS DOC]
  # Subclass of a one-field named tuple, adding one method; used to test method calls on named
  # tuple subclasses.

  def foo(self):
    # [FUNCTION DOC]
    # Returns the field a plus one.
    return self.a + 1


def namedtuple_subclass(x):
  # [FUNCTION DOC]
  # Function under test: builds the named tuple subclass from x and returns the result of its
  # method.
  nt = NamedTupleSubclass(x)
  return nt.foo()


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Reference test case for named tuple construction and method calls.

  def test_inline(self):
    # [FUNCTION DOC]
    # Asserts the inline named tuple sample matches eager results for a number and a tensor.
    self.assertFunctionMatchesEager(inline_namedtuple, 1)
    self.assertFunctionMatchesEager(inline_namedtuple, tf.constant(1))

  def test_external(self):
    # [FUNCTION DOC]
    # Creates a named tuple type with fields a and b in the test and asserts the sample taking it
    # as an argument matches eager results for a number and a tensor.
    nt = collections.namedtuple('TestNamedTuple', ('a', 'b'))
    self.assertFunctionMatchesEager(external_namedtuple, 1, nt)
    self.assertFunctionMatchesEager(external_namedtuple, tf.constant(1), nt)

  def test_subclass(self):
    # [FUNCTION DOC]
    # Asserts the named tuple subclass method-call sample matches eager results for a number and
    # a tensor.
    self.assertFunctionMatchesEager(namedtuple_subclass, 1)
    self.assertFunctionMatchesEager(namedtuple_subclass, tf.constant(1))


if __name__ == '__main__':
  tf.test.main()
