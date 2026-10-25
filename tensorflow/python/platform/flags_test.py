# [MODULE DOC]
# Sanity tests for the flags compatibility wrapper. At import time it defines flags of each
# basic type (string, integer, float, boolean, bool) twice, once using the legacy keyword names
# and once using the new names, in the global absl registry. The test class checks that the
# proxy registry class forwards every operation to the real registry and that both keyword
# styles produce identical flag defaults and help text. Runs the unittest main when executed.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Sanity tests for tf.flags."""
import sys
import unittest

from absl import flags as absl_flags

from tensorflow.python.platform import flags
from tensorflow.python.platform import test


flags.DEFINE_string(  # pylint: disable=no-value-for-parameter
    flag_name='old_string',
    default_value='default',
    docstring='docstring')
flags.DEFINE_string(
    name='new_string', default='default', help='docstring')
flags.DEFINE_integer(  # pylint: disable=no-value-for-parameter
    flag_name='old_integer',
    default_value=1,
    docstring='docstring')
flags.DEFINE_integer(
    name='new_integer', default=1, help='docstring')
flags.DEFINE_float(  # pylint: disable=no-value-for-parameter
    flag_name='old_float',
    default_value=1.5,
    docstring='docstring')
flags.DEFINE_float(
    name='new_float', default=1.5, help='docstring')
flags.DEFINE_bool(  # pylint: disable=no-value-for-parameter
    flag_name='old_bool',
    default_value=True,
    docstring='docstring')
flags.DEFINE_bool(
    name='new_bool', default=True, help='docstring')
flags.DEFINE_boolean(  # pylint: disable=no-value-for-parameter
    flag_name='old_boolean',
    default_value=False,
    docstring='docstring')
flags.DEFINE_boolean(
    name='new_boolean', default=False, help='docstring')


class FlagsTest(unittest.TestCase):
  # [CLASS DOC]
  # Plain unittest case for the flag registry proxy and the legacy keyword support. Fixture: each
  # test gets a fresh private flag registry wrapped by the proxy, holding one string flag.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Creates a new empty flag registry, wraps it in the proxy class, and defines a
    # string flag named test (default value default, help text test flag) in that wrapped
    # registry. Stores both on the instance.
    self.original_flags = flags.FlagValues()
    self.wrapped_flags = flags._FlagValuesWrapper(self.original_flags)
    flags.DEFINE_string(
        'test', 'default', 'test flag', flag_values=self.wrapped_flags)

  def test_attribute_overrides(self):
    # [FUNCTION DOC]
    # Asserts that the is-parsed method looked up on the public proxy registry equals the one on
    # the absl registry, i.e. methods are forwarded rather than shadowed.
    # Test that methods defined in absl.flags.FlagValues are the same as the
    # wrapped ones.
    self.assertEqual(flags.FLAGS.is_parsed, absl_flags.FLAGS.is_parsed)

  def test_getattr(self):
    # [FUNCTION DOC]
    # Asserts the registry starts unparsed; then, with the process arguments patched to set the
    # test flag to new, reading the flag through the proxy returns new (implicit parsing), and the
    # registry is afterwards marked parsed.
    self.assertFalse(self.wrapped_flags.is_parsed())
    with test.mock.patch.object(sys, 'argv', new=['program', '--test=new']):
      self.assertEqual('new', self.wrapped_flags.test)
    self.assertTrue(self.wrapped_flags.is_parsed())

  def test_setattr(self):
    # [FUNCTION DOC]
    # Asserts the flag initially has its default value, then assigns a new value through the proxy
    # and asserts reading it back returns the new value.
    self.assertEqual('default', self.wrapped_flags.test)
    self.wrapped_flags.test = 'new'
    self.assertEqual('new', self.wrapped_flags.test)

  def test_delattr(self):
    # [FUNCTION DOC]
    # Deletes the test flag through the proxy and asserts it is no longer contained in the proxy
    # and that reading it afterwards raises an attribute error.
    del self.wrapped_flags.test
    self.assertNotIn('test', self.wrapped_flags)
    with self.assertRaises(AttributeError):
      _ = self.wrapped_flags.test

  def test_dir(self):
    # [FUNCTION DOC]
    # Asserts the directory listing of the proxy equals a list holding only the test flag name.
    self.assertEqual(['test'], dir(self.wrapped_flags))

  def test_getitem(self):
    # [FUNCTION DOC]
    # Asserts item access on the proxy returns the identical flag object as the underlying
    # registry for the test flag.
    self.assertIs(self.original_flags['test'], self.wrapped_flags['test'])

  def test_setitem(self):
    # [FUNCTION DOC]
    # Builds a flag object named fruit (default apple, help the fruit type), stores it via item
    # assignment on the proxy, and asserts the underlying registry returns the identical object
    # for it and that reading the flag value through the proxy gives apple.
    flag = flags.Flag(flags.ArgumentParser(), flags.ArgumentSerializer(),
                      'fruit', 'apple', 'the fruit type')
    self.wrapped_flags['fruit'] = flag
    self.assertIs(self.original_flags['fruit'], self.wrapped_flags['fruit'])
    self.assertEqual('apple', self.wrapped_flags.fruit)

  def test_len(self):
    # [FUNCTION DOC]
    # Asserts the proxy's length is one (the single flag defined in setUp).
    self.assertEqual(1, len(self.wrapped_flags))

  def test_iter(self):
    # [FUNCTION DOC]
    # Asserts iterating over the proxy yields exactly the list holding the test flag name.
    self.assertEqual(['test'], list(self.wrapped_flags))

  def test_str(self):
    # [FUNCTION DOC]
    # Asserts the string form of the proxy equals the string form of the underlying registry.
    self.assertEqual(str(self.wrapped_flags), str(self.original_flags))

  def test_call(self):
    # [FUNCTION DOC]
    # Calls the proxy with a program name and an argument setting the test flag to new, then
    # asserts reading the flag returns new, showing the call is forwarded and parses.
    self.wrapped_flags(['program', '--test=new'])
    self.assertEqual('new', self.wrapped_flags.test)

  def test_keyword_arguments(self):
    # [FUNCTION DOC]
    # For each of the flags defined at module import time with old and new keyword styles (string,
    # integer, float, bool, boolean), looks the flag up in the global absl registry and asserts
    # its default equals the expected value and its help text equals the docstring text.
    test_cases = (
        ('old_string', 'default'),
        ('new_string', 'default'),
        ('old_integer', 1),
        ('new_integer', 1),
        ('old_float', 1.5),
        ('new_float', 1.5),
        ('old_bool', True),
        ('new_bool', True),
        ('old_boolean', False),
        ('new_boolean', False),
    )
    for flag_name, default_value in test_cases:
      self.assertEqual(default_value, absl_flags.FLAGS[flag_name].default)
      self.assertEqual('docstring', absl_flags.FLAGS[flag_name].help)


if __name__ == '__main__':
  unittest.main()
