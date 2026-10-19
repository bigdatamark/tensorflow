# [MODULE DOC]
# Reference test for autograph handling of calls to builtin functions (dict, len, list, range,
# float, int), including cases where the builtin name is shadowed by a local alias, obtained
# dynamically, or applied to a mock object. Native and compiled results are compared with Python
# values and with tensor inputs. No import-time side effects.
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
"""Simple call to a builtin function."""

import unittest

import tensorflow as tf

from tensorflow.python.autograph.tests import reference_test_base


# TODO(mdan): Add tests for all builtins.


def dict_call(x):
  # [FUNCTION DOC]
  # Function under test: returns a dictionary with the single key foo mapped to x.
  return dict(foo=x)


def dict_call_aliased(x):
  # [FUNCTION DOC]
  # Function under test: shadows the dictionary builtin with a local identity function and
  # calls it with x, so the result is x itself (conversion must not treat it as the builtin).
  def fake_dict(x):
    # [FUNCTION DOC]
    # Stand-in for the dictionary builtin: returns its argument unchanged.
    return x

  dict = fake_dict  # pylint:disable=redefined-builtin
  return dict(x)


def dict_call_dynamic(x):
  # [FUNCTION DOC]
  # Function under test: obtains the dictionary builtin through a helper function and calls it
  # with foo set to x. Returns the resulting dictionary.
  def gen_dict():
    # [FUNCTION DOC]
    # Returns the dictionary builtin itself (not a call result).
    return dict

  d = gen_dict()
  return d(foo=x)


def len_call(x):
  # [FUNCTION DOC]
  # Function under test: returns the length of x.
  return len(x)


def nested_call(x):
  # [FUNCTION DOC]
  # Function under test: returns a list of the integers from zero up to the length of x.
  return list(range(len(x)))


def nested_cast(x):
  # [FUNCTION DOC]
  # Function under test: converts x to an integer and then to a float, returning the float.
  # Not exercised by any test in this module.
  return float(int(x))


def len_call_aliased(x):
  # [FUNCTION DOC]
  # Function under test: shadows the length builtin with a local identity function and calls it
  # with x, so the result is x itself.

  def fake_len(x):
    # [FUNCTION DOC]
    # Stand-in for the length builtin: returns its argument unchanged.
    return x

  len = fake_len  # pylint:disable=redefined-builtin
  return len(x)


def len_call_dynamic(x):
  # [FUNCTION DOC]
  # Function under test: obtains the length builtin through a helper function and calls it on x.
  # Returns the length.

  def gen_len():
    # [FUNCTION DOC]
    # Returns the length builtin itself (not a call result).
    return len

  l = gen_len()
  return l(x)


def len_call_on_mock():
  # [FUNCTION DOC]
  # Function under test: creates a mock object and returns its length (the mock's default of
  # zero). Not exercised by any test in this module.
  x = unittest.mock.MagicMock()
  return len(x)


class ReferenceTest(reference_test_base.TestCase):
  # [CLASS DOC]
  # Reference test case for calls to builtin functions.

  def test_basic(self):
    # [FUNCTION DOC]
    # Asserts the dictionary, length, aliased, dynamic and nested-call samples give identical
    # results natively and compiled, using plain Python inputs.
    self.assertFunctionMatchesEager(dict_call, 1)
    self.assertFunctionMatchesEager(len_call, [1, 2])
    self.assertFunctionMatchesEager(dict_call_aliased, 1)
    self.assertFunctionMatchesEager(len_call_aliased, [1, 2])
    self.assertFunctionMatchesEager(dict_call_dynamic, 1)
    self.assertFunctionMatchesEager(len_call_dynamic, [1, 2])
    self.assertFunctionMatchesEager(nested_call, [])
    self.assertFunctionMatchesEager(nested_call, [1, 2, 3])

  def test_basic_tensor(self):
    # [FUNCTION DOC]
    # Same checks as the basic test, but with all inputs converted to tensors first.
    self.all_inputs_tensors = True
    self.assertFunctionMatchesEager(dict_call, 1)
    self.assertFunctionMatchesEager(len_call, [1, 2])
    self.assertFunctionMatchesEager(dict_call_aliased, 1)
    self.assertFunctionMatchesEager(len_call_aliased, [1, 2])
    self.assertFunctionMatchesEager(dict_call_dynamic, 1)
    self.assertFunctionMatchesEager(len_call_dynamic, [1, 2])
    self.assertFunctionMatchesEager(nested_call, [])
    self.assertFunctionMatchesEager(nested_call, [1, 2, 3])


if __name__ == '__main__':
  tf.test.main()
