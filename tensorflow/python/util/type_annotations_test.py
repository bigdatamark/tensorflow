# [MODULE DOC]
# Unit tests for the type_annotations helper module. Checks the generic union, tuple and mapping
# predicates over parameterized and bare typing forms and plain values, the extraction of type
# arguments from generic annotations (including ellipsis and nested forms), and the forward
# reference predicate. Uses parameterized test cases; runs as a test binary.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow/python/util/type_annotations.py."""

import typing
from absl.testing import parameterized

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.util import type_annotations


class TypeAnnotationsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the type annotation helpers: generic-type predicates, generic
  # argument extraction and forward-reference detection. No shared fixtures.

  @parameterized.parameters([
      (typing.Union[int, float], 'Union'),
      (typing.Tuple[int, ...], 'Tuple'),
      (typing.Tuple[int, float, float], 'Tuple'),
      (typing.Mapping[int, float], 'Mapping'),
      (typing.Union[typing.Tuple[int], typing.Tuple[int, ...]], 'Union'),
      # These predicates return False for Generic types w/ no parameters:
      (typing.Union, None),
      (typing.Tuple, None),
      (typing.Mapping, None),
      (int, None),
      (12, None),
  ])
  def testGenericTypePredicates(self, tp, expected):
    # [FUNCTION DOC]
    # Parameterized over (annotation, expected kind) pairs: parameterized union, tuple and mapping
    # forms, nested union, bare forms, a plain class and an integer (the last group expect no
    # kind). Asserts that each of the union, tuple and mapping predicates returns true exactly
    # when the expected kind names it.
    self.assertEqual(
        type_annotations.is_generic_union(tp), expected == 'Union')
    self.assertEqual(
        type_annotations.is_generic_tuple(tp), expected == 'Tuple')
    self.assertEqual(
        type_annotations.is_generic_mapping(tp), expected == 'Mapping')

  @parameterized.parameters([
      (typing.Union[int, float], (int, float)),
      (typing.Tuple[int, ...], (int, Ellipsis)),
      (typing.Tuple[int, float, float], (
          int,
          float,
          float,
      )),
      (typing.Mapping[int, float], (int, float)),
      (typing.Union[typing.Tuple[int],
                    typing.Tuple[int,
                                 ...]], (typing.Tuple[int], typing.Tuple[int,
                                                                         ...])),
  ])
  def testGetGenericTypeArgs(self, tp, expected):
    # [FUNCTION DOC]
    # Parameterized over (annotation, expected arguments) pairs covering unions, variable-length
    # and fixed tuples, mappings and a union of tuples. Asserts the generic argument accessor
    # returns the expected tuple of arguments.
    self.assertEqual(type_annotations.get_generic_type_args(tp), expected)

  def testIsForwardRef(self):
    # [FUNCTION DOC]
    # Builds a union of a string forward reference and an integer type, extracts its arguments
    # and asserts the forward-reference predicate is true for the first argument and false for
    # the second.
    tp = typing.Union['B', int]
    tp_args = type_annotations.get_generic_type_args(tp)
    self.assertTrue(type_annotations.is_forward_ref(tp_args[0]))
    self.assertFalse(type_annotations.is_forward_ref(tp_args[1]))


if __name__ == '__main__':
  googletest.main()
