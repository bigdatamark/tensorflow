# [MODULE DOC]
# Tests the in-memory ("ram://") file system through the Python file APIs: directory creation and
# recursive deletion, writing, appending with seek, listing, globbing, existence checks and saving
# then loading a saved model. When run as main, enables v2 behavior and starts the test runner.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for ram_file_system.h."""

import platform

from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.saved_model import saved_model


class RamFilesystemTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the RAM file system. Each test uses paths with the in-memory scheme prefix; no
  # setup or teardown, so state in the in-memory file system persists across tests in the process.

  def test_create_and_delete_directory(self):
    # [FUNCTION DOC]
    # Creates an in-memory directory and recursively deletes it; expects no error.
    file_io.create_dir_v2('ram://testdirectory')
    file_io.delete_recursively_v2('ram://testdirectory')

  def test_create_and_delete_directory_tree_recursive(self):
    # [FUNCTION DOC]
    # Builds a nested directory tree with a text file in the deepest directory, recursively deletes
    # the root, and asserts globbing the root's children returns an empty list.
    file_io.create_dir_v2('ram://testdirectory')
    file_io.create_dir_v2('ram://testdirectory/subdir1')
    file_io.create_dir_v2('ram://testdirectory/subdir2')
    file_io.create_dir_v2('ram://testdirectory/subdir1/subdir3')
    with gfile.GFile('ram://testdirectory/subdir1/subdir3/a.txt', 'w') as f:
      f.write('Hello, world.')
    file_io.delete_recursively_v2('ram://testdirectory')
    self.assertEqual(gfile.Glob('ram://testdirectory/*'), [])

  def test_write_file(self):
    # [FUNCTION DOC]
    # Writes a string twice in one write-mode session, reads the file back, and asserts the content
    # is the string repeated twice.
    with gfile.GFile('ram://a.txt', 'w') as f:
      f.write('Hello, world.')
      f.write('Hello, world.')

    with gfile.GFile('ram://a.txt', 'r') as f:
      self.assertEqual(f.read(), 'Hello, world.' * 2)

  def test_append_file_with_seek(self):
    # [FUNCTION DOC]
    # Writes a string, reopens the file in read-write mode, seeks to the end and writes the string
    # again, then asserts reading the file gives the string repeated twice (data not truncated).
    with gfile.GFile('ram://c.txt', 'w') as f:
      f.write('Hello, world.')

    with gfile.GFile('ram://c.txt', 'w+') as f:
      f.seek(offset=0, whence=2)
      f.write('Hello, world.')

    with gfile.GFile('ram://c.txt', 'r') as f:
      self.assertEqual(f.read(), 'Hello, world.' * 2)

  def test_list_dir(self):
    # [FUNCTION DOC]
    # Creates ten empty files under two different directories and asserts listing one directory
    # returns exactly the ten file names in order.
    for i in range(10):
      with gfile.GFile('ram://a/b/%d.txt' % i, 'w') as f:
        f.write('')
      with gfile.GFile('ram://c/b/%d.txt' % i, 'w') as f:
        f.write('')

    matches = ['%d.txt' % i for i in range(10)]
    self.assertEqual(gfile.ListDirectory('ram://a/b/'), matches)

  def test_glob(self):
    # [FUNCTION DOC]
    # Creates ten empty files under two directories. Asserts a wildcard pattern returns the full
    # paths of the ten files for each existing directory in order, and an empty list for a
    # directory that does not exist.
    for i in range(10):
      with gfile.GFile('ram://a/b/%d.txt' % i, 'w') as f:
        f.write('')
      with gfile.GFile('ram://c/b/%d.txt' % i, 'w') as f:
        f.write('')

    matches = ['ram://a/b/%d.txt' % i for i in range(10)]
    self.assertEqual(gfile.Glob('ram://a/b/*'), matches)

    matches = []
    self.assertEqual(gfile.Glob('ram://b/b/*'), matches)

    matches = ['ram://c/b/%d.txt' % i for i in range(10)]
    self.assertEqual(gfile.Glob('ram://c/b/*'), matches)

  def test_file_exists(self):
    # [FUNCTION DOC]
    # Creates a file in nested directories; asserts the file and its parent directories exist, and
    # that sibling or nonexistent paths do not.
    with gfile.GFile('ram://exists/a/b/c.txt', 'w') as f:
      f.write('')
    self.assertTrue(gfile.Exists('ram://exists/a'))
    self.assertTrue(gfile.Exists('ram://exists/a/b'))
    self.assertTrue(gfile.Exists('ram://exists/a/b/c.txt'))

    self.assertFalse(gfile.Exists('ram://exists/b'))
    self.assertFalse(gfile.Exists('ram://exists/a/c'))
    self.assertFalse(gfile.Exists('ram://exists/a/b/k'))

  def test_savedmodel(self):
    # [FUNCTION DOC]
    # Skipped on Windows. Defines a module with a no-argument function returning a constant
    # one-element list, saves it to an in-memory path, loads it back, and asserts calling the loaded
    # function returns that list.
    if platform.system() == 'Windows':
      self.skipTest('RAM FS not fully supported on Windows.')

    class MyModule(module.Module):
      # [CLASS DOC]
      # Local fixture module with one traced no-argument method. No instance state.

      @def_function.function(input_signature=[])
      def foo(self):
        # [FUNCTION DOC]
        # Returns a constant one-element integer tensor holding 1.
        return constant_op.constant([1])

    saved_model.save(MyModule(), 'ram://my_module')

    loaded = saved_model.load('ram://my_module')
    self.assertAllEqual(loaded.foo(), [1])


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  test.main()
