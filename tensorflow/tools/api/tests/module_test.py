# [MODULE DOC]
# Smoke tests for the top-level tensorflow module object. Defines one test case class that checks
# the module can be located by the package loader, has the expected documentation text and name,
# lazily exposes common submodules, formats tensors correctly for built-in-named functions, and
# keeps the internal python package hidden from the public namespace. Importing it imports
# tensorflow; running it as a script launches the TensorFlow test runner. Not a helper module.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""Smoke tests for tensorflow module."""

import pkgutil

import tensorflow as tf

from tensorflow.python import tf2
from tensorflow.python.platform import test


class ModuleTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering basic properties of the imported tensorflow module (loadability, doc text,
  # name, attribute availability, tensor formatting, summary API, hidden internals). It has no
  # shared fixtures; each test method is independent and only reads module attributes.

  def testCanLoadWithPkgutil(self):
    # [FUNCTION DOC]
    # Asks the package loader utility to find the tensorflow package by name and asserts that a
    # loader is returned (not null). Takes no inputs, returns nothing, fails the test otherwise.
    out = pkgutil.find_loader('tensorflow')
    self.assertIsNotNone(out)

  def testDocString(self):
    # [FUNCTION DOC]
    # Asserts the tensorflow module documentation text mentions the word TensorFlow and does not
    # contain the word Wrapper (i.e. it is the real documentation, not a lazy-loader wrapper's).
    # Takes no inputs, returns nothing.
    self.assertIn('TensorFlow', tf.__doc__)
    self.assertNotIn('Wrapper', tf.__doc__)

  def testDict(self):
    # [FUNCTION DOC]
    # Accesses the nn, keras and image submodule attributes on the tensorflow module (which may
    # trigger lazy loading), then asserts each name is present in the module's attribute
    # dictionary. Takes no inputs, returns nothing.
    # Check that a few modules are in __dict__.
    # pylint: disable=pointless-statement
    tf.nn
    tf.keras
    tf.image
    # pylint: enable=pointless-statement
    self.assertIn('nn', tf.__dict__)
    self.assertIn('keras', tf.__dict__)
    self.assertIn('image', tf.__dict__)

  def testName(self):
    # [FUNCTION DOC]
    # Asserts the tensorflow module's name attribute equals the string tensorflow.
    # Takes no inputs, returns nothing.
    self.assertEqual('tensorflow', tf.__name__)

  def testBuiltInName(self):
    # [FUNCTION DOC]
    # Checks that the range function of tensorflow (which shares a name with a Python built-in)
    # works: builds a range tensor of the integers 1 to 9 and compares its string form with the
    # expected text. In TF2 behavior the eager tensor format is expected; otherwise the
    # graph-mode tensor format named range:0 is expected. Takes no inputs, returns nothing.
    # range is a built-in name in Python. Just checking that
    # tf.range works fine.
    if tf2.enabled():
      self.assertEqual(
          'tf.Tensor([1 2 3 4 5 6 7 8 9], shape=(9,), dtype=int32)',
          str(tf.range(1, 10)))
    else:
      self.assertEqual('Tensor("range:0", shape=(9,), dtype=int32)',
                       str(tf.range(1, 10)))

  def testCompatV2HasCompatV1(self):
    # [FUNCTION DOC]
    # Verifies that the compat v1 namespace is reachable from inside the compat v2 namespace and
    # exposes keras, by simply accessing the attribute; fails only if an attribute error is raised.
    # Takes no inputs, returns nothing, makes no explicit assertion.
    # pylint: disable=pointless-statement
    tf.compat.v2.compat.v1.keras
    # pylint: enable=pointless-statement

  def testSummaryMerged(self):
    # [FUNCTION DOC]
    # Checks the summary API was merged into the tensorflow module by accessing the image
    # summary function, then, depending on the major API version, either the file writer factory
    # in the summary namespace (version 2) or the legacy file writer class under compat v1 (other
    # versions). Passes if no attribute error is raised. Takes no inputs, returns nothing.
    # pylint: disable=pointless-statement
    tf.summary.image
    # If we use v2 API, check for create_file_writer,
    # otherwise check for FileWriter.
    if hasattr(tf, '_major_api_version') and tf._major_api_version == 2:
      tf.summary.create_file_writer
    else:
      tf.compat.v1.summary.FileWriter
    # pylint: enable=pointless-statement

  def testPythonModuleIsHidden(self):
    # [FUNCTION DOC]
    # Asserts that the internal python subpackage name is not listed in the directory listing of
    # the tensorflow module, i.e. it is not part of the public namespace. Takes no inputs,
    # returns nothing.
    self.assertNotIn('python', dir(tf))


if __name__ == '__main__':
  test.main()
