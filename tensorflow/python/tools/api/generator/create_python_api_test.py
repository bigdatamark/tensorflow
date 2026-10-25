# [MODULE DOC]
# Unit tests for the API generator script. Defines fake exported symbols (two functions, one
# class and a constant) registered through the export decorator, injects them into a fake loaded
# module whose name contains the base package, and checks the generated import text, compat
# module creation, nested compat modules and proxy file generation. Import-time side effect:
# the decorators register the fake symbols in the export tables.
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
# =============================================================================
"""Tests for create_python_api."""

import os
import sys
import types

from tensorflow.python.platform import test
from tensorflow.python.tools.api.generator import create_python_api
from tensorflow.python.util.tf_export import tf_export


@tf_export('test_op', 'test_op1', 'test.test_op2')
def test_op():
  # [FUNCTION DOC]
  # HELPER: fake exported function used as test input; does nothing and returns nothing.
  pass


@tf_export('test1.foo', v1=['test.foo'])
def deprecated_test_op():
  # [FUNCTION DOC]
  # HELPER: fake exported function used as test input, exported under a v2 name and a different
  # v1 name; does nothing and returns nothing.
  pass


@tf_export('TestClass', 'NewTestClass')
class TestClass(object):
  # [CLASS DOC]
  # HELPER: empty fake exported class (exported under two names) used as test input.
  pass


_TEST_CONSTANT = 5
_MODULE_NAME = 'tensorflow.python.test_module'


class CreatePythonApiTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the API init text generator and proxy file writer. The fixture installs a
  # fake module holding the fake exported symbols and an exported constant, and removes it
  # afterwards.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: registers a fake module in the loaded module table, attaches the fake functions
    # and class to it, points their defining module at it, and exports a test constant from it.
    # Side effects: modifies the global module table and export tables. Returns nothing.
    # Add fake op to a module that has 'tensorflow' in the name.
    sys.modules[_MODULE_NAME] = types.ModuleType(_MODULE_NAME)
    setattr(sys.modules[_MODULE_NAME], 'test_op', test_op)
    setattr(sys.modules[_MODULE_NAME], 'deprecated_test_op', deprecated_test_op)
    setattr(sys.modules[_MODULE_NAME], 'TestClass', TestClass)
    test_op.__module__ = _MODULE_NAME
    TestClass.__module__ = _MODULE_NAME
    tf_export('consts._TEST_CONSTANT').export_constant(
        _MODULE_NAME, '_TEST_CONSTANT')

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: removes the fake module from the loaded module table. Raises a key error
    # if it is missing. Returns nothing.
    del sys.modules[_MODULE_NAME]

  def testFunctionImportIsAdded(self):
    # [FUNCTION DOC]
    # Scenario: generate version 1 init text for the default package. Asserts that the text
    # contains the import (or lazy table entry) of the fake function under its alternate export
    # name and under its primary name, and that no compat.v1 module is produced when no compat
    # versions are requested.
    imports, _, _ = create_python_api.get_api_init_text(
        packages=[create_python_api._DEFAULT_PACKAGE],
        packages_to_ignore=[],
        output_package='tensorflow',
        api_name='tensorflow',
        api_version=1)
    if create_python_api._LAZY_LOADING:
      expected_import = (
          '\'test_op1\': '
          '(\'tensorflow.python.test_module\','
          ' \'test_op\')')
    else:
      expected_import = (
          'from tensorflow.python.test_module '
          'import test_op as test_op1')
    self.assertTrue(
        expected_import in str(imports),
        msg='%s not in %s' % (expected_import, str(imports)))

    if create_python_api._LAZY_LOADING:
      expected_import = (
          '\'test_op\': '
          '(\'tensorflow.python.test_module\','
          ' \'test_op\')')
    else:
      expected_import = (
          'from tensorflow.python.test_module '
          'import test_op')
    self.assertTrue(
        expected_import in str(imports),
        msg='%s not in %s' % (expected_import, str(imports)))
    # Also check that compat.v1 is not added to imports.
    self.assertFalse('compat.v1' in imports,
                     msg='compat.v1 in %s' % str(imports.keys()))

  def testClassImportIsAdded(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected import text is computed but never compared; the assertion only
    # checks that the class name appears somewhere in the generated text.
    # Scenario: generate version 2 init text and check that the fake class name is present.
    imports, _, _ = create_python_api.get_api_init_text(
        packages=[create_python_api._DEFAULT_PACKAGE],
        packages_to_ignore=[],
        output_package='tensorflow',
        api_name='tensorflow',
        api_version=2)
    if create_python_api._LAZY_LOADING:
      expected_import = (
          '\'NewTestClass\':'
          ' (\'tensorflow.python.test_module\','
          ' \'TestClass\')')
    else:
      expected_import = (
          'from tensorflow.python.test_module '
          'import TestClass')
    self.assertTrue(
        'TestClass' in str(imports),
        msg='%s not in %s' % (expected_import, str(imports)))

  def testConstantIsAdded(self):
    # [FUNCTION DOC]
    # Scenario: generate version 1 init text and assert that the exported test constant appears
    # as an import (static mode) or lazy table entry (lazy mode) from the fake module.
    imports, _, _ = create_python_api.get_api_init_text(
        packages=[create_python_api._DEFAULT_PACKAGE],
        packages_to_ignore=[],
        output_package='tensorflow',
        api_name='tensorflow',
        api_version=1)
    if create_python_api._LAZY_LOADING:
      expected = ('\'_TEST_CONSTANT\':'
                  ' (\'tensorflow.python.test_module\','
                  ' \'_TEST_CONSTANT\')')
    else:
      expected = ('from tensorflow.python.test_module '
                  'import _TEST_CONSTANT')
    self.assertTrue(expected in str(imports),
                    msg='%s not in %s' % (expected, str(imports)))

  def testCompatModuleIsAdded(self):
    # [FUNCTION DOC]
    # Scenario: generate version 2 text with compat version 1 requested. Asserts that the
    # resulting module mapping contains the compat.v1 module and the compat.v1.test module.
    imports, _, _ = create_python_api.get_api_init_text(
        packages=[create_python_api._DEFAULT_PACKAGE],
        packages_to_ignore=[],
        output_package='tensorflow',
        api_name='tensorflow',
        api_version=2,
        compat_api_versions=[1])
    self.assertTrue('compat.v1' in imports,
                    msg='compat.v1 not in %s' % str(imports.keys()))
    self.assertTrue('compat.v1.test' in imports,
                    msg='compat.v1.test not in %s' % str(imports.keys()))

  def testNestedCompatModulesAreAdded(self):
    # [FUNCTION DOC]
    # Scenario: generate version 2 text with compat versions 1 and 2. Asserts that all four
    # nested modules (compat.v1.compat.v1, v1.compat.v2, v2.compat.v1, v2.compat.v2) exist.
    imports, _, _ = create_python_api.get_api_init_text(
        packages=[create_python_api._DEFAULT_PACKAGE],
        packages_to_ignore=[],
        output_package='tensorflow',
        api_name='tensorflow',
        api_version=2,
        compat_api_versions=[1, 2])
    self.assertIn('compat.v1.compat.v1', imports,
                  msg='compat.v1.compat.v1 not in %s' % str(imports.keys()))
    self.assertIn('compat.v1.compat.v2', imports,
                  msg='compat.v1.compat.v2 not in %s' % str(imports.keys()))
    self.assertIn('compat.v2.compat.v1', imports,
                  msg='compat.v2.compat.v1 not in %s' % str(imports.keys()))
    self.assertIn('compat.v2.compat.v2', imports,
                  msg='compat.v2.compat.v2 not in %s' % str(imports.keys()))

  def testProxyAPIFileIsGenerated(self):
    # [FUNCTION DOC]
    # Scenario: write a proxy init file for a keras losses module under a temporary directory
    # using a proxy root. Asserts the file exists and its content is exactly one line that
    # star-imports the module from the proxy root. Side effect: creates files in the temp dir.
    save_dir = self.get_temp_dir()
    proxy_module_root = 'tf_keras.api._v2'
    module = 'keras.losses'
    module_dir = module.replace('.', '/')
    proxy_file = os.path.join(save_dir, module_dir, '__init__.py')
    expected_imports = [f'from {proxy_module_root}.{module} import *']
    create_python_api.create_proxy_api_files([proxy_file],
                                             proxy_module_root,
                                             save_dir)
    self.assertTrue(os.path.exists(proxy_file))
    with open(proxy_file, 'r') as f:
      lines = f.readlines()
    self.assertCountEqual(expected_imports, lines)


if __name__ == '__main__':
  test.main()
