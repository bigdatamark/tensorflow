# -*- coding: utf-8 -*-
# [MODULE DOC]
# Unit tests for the file-level io ops (read file, write file, matching files).
# Contains one test class that creates temporary files and checks reading, writing (including
# creating missing parent directories) and glob-style file matching. No import-time side effects
# besides imports; running the file launches the test main.
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
"""Tests for tensorflow.python.ops.io_ops."""

import os
import shutil
import tempfile

from tensorflow.python.framework import test_util
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class IoOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the read-file, write-file and matching-files ops. Tests create temporary files in
  # the test temp directory and clean them up themselves; one helper builds expected result sets.

  @test_util.run_deprecated_v1
  def testReadFile(self):
    # [FUNCTION DOC]
    # For an empty string, an ASCII string and a non-ASCII string, writes the bytes to a temporary
    # file, reads it with the read-file op and expects a scalar shape and identical contents.
    # Removes the temporary file afterwards. Runs in graph mode only.
    cases = ['', 'Some contents', 'Неки садржаји на српском']
    for contents in cases:
      contents = compat.as_bytes(contents)
      with tempfile.NamedTemporaryFile(
          prefix='ReadFileTest', dir=self.get_temp_dir(), delete=False) as temp:
        temp.write(contents)
      with self.cached_session():
        read = io_ops.read_file(temp.name)
        self.assertEqual([], read.get_shape())
        self.assertEqual(self.evaluate(read), contents)
      os.remove(temp.name)

  def testWriteFile(self):
    # [FUNCTION DOC]
    # Creates an empty temporary file, writes contents to it using the write-file op, then reads it
    # back with ordinary file access and expects identical bytes. Removes the file afterwards.
    # KNOWN ISSUE: the session variable bound by the context manager is never used.
    cases = ['', 'Some contents']
    for contents in cases:
      contents = compat.as_bytes(contents)
      with tempfile.NamedTemporaryFile(
          prefix='WriteFileTest', dir=self.get_temp_dir(),
          delete=False) as temp:
        pass
      with self.cached_session() as sess:
        w = io_ops.write_file(temp.name, contents)
        self.evaluate(w)
        with open(temp.name, 'rb') as f:
          file_contents = f.read()
        self.assertEqual(file_contents, contents)
      os.remove(temp.name)

  def testWriteFileCreateDir(self):
    # [FUNCTION DOC]
    # Writes to a path whose parent directories do not exist and expects the write-file op to
    # create them; reads the file back and compares contents. Removes the created directory tree.
    # KNOWN ISSUE: the session variable bound by the context manager is never used.
    cases = ['', 'Some contents']
    for contents in cases:
      contents = compat.as_bytes(contents)
      subdir = os.path.join(self.get_temp_dir(), 'subdir1')
      filepath = os.path.join(subdir, 'subdir2', 'filename')
      with self.cached_session() as sess:
        w = io_ops.write_file(filepath, contents)
        self.evaluate(w)
        with open(filepath, 'rb') as f:
          file_contents = f.read()
        self.assertEqual(file_contents, contents)
      shutil.rmtree(subdir)

  def _subset(self, files, indices):
    # [FUNCTION DOC]
    # HELPER: builds expected result sets for the matching-files test.
    # Inputs: files is a list of temporary file objects; indices is a collection of positions.
    # Returns the set of file names, as bytes, of the files at the given positions.
    return set(
        compat.as_bytes(files[i].name) for i in range(len(files))
        if i in indices)

  @test_util.run_deprecated_v1
  def testMatchingFiles(self):
    # [FUNCTION DOC]
    # Creates several temporary files with related names, then checks the matching-files op:
    # exact names, wildcard patterns (single character, any characters, and, except on Windows,
    # character sets and ranges), an empty list, multiple exact names and multiple globs.
    # Closes (and thereby deletes) the temporary files at the end. Runs in graph mode only.
    cases = [
        'ABcDEF.GH', 'ABzDEF.GH', 'ABasdfjklDEF.GH', 'AB3DEF.GH', 'AB4DEF.GH',
        'ABDEF.GH', 'XYZ'
    ]
    files = [
        tempfile.NamedTemporaryFile(
            prefix=c, dir=self.get_temp_dir(), delete=True) for c in cases
    ]

    with self.cached_session():
      # Test exact match without wildcards.
      for f in files:
        self.assertEqual(
            io_ops.matching_files(f.name).eval(), compat.as_bytes(f.name))

      # We will look for files matching "ABxDEF.GH*" where "x" is some wildcard.
      directory_path = files[0].name[:files[0].name.find(cases[0])]
      pattern = directory_path + 'AB%sDEF.GH*'

      self.assertEqual(
          set(io_ops.matching_files(pattern % 'z').eval()),
          self._subset(files, [1]))
      self.assertEqual(
          set(io_ops.matching_files(pattern % '?').eval()),
          self._subset(files, [0, 1, 3, 4]))
      self.assertEqual(
          set(io_ops.matching_files(pattern % '*').eval()),
          self._subset(files, [0, 1, 2, 3, 4, 5]))
      # NOTE(mrry): Windows uses PathMatchSpec to match file patterns, which
      # does not support the following expressions.
      if os.name != 'nt':
        self.assertEqual(
            set(io_ops.matching_files(pattern % '[cxz]').eval()),
            self._subset(files, [0, 1]))
        self.assertEqual(
            set(io_ops.matching_files(pattern % '[0-9]').eval()),
            self._subset(files, [3, 4]))

      # Test an empty list input.
      self.assertItemsEqual(io_ops.matching_files([]).eval(), [])

      # Test multiple exact filenames.
      self.assertItemsEqual(
          io_ops.matching_files([
              files[0].name, files[1].name, files[2].name]).eval(),
          self._subset(files, [0, 1, 2]))

      # Test multiple globs.
      self.assertItemsEqual(
          io_ops.matching_files([
              pattern % '?', directory_path + 'X?Z*']).eval(),
          self._subset(files, [0, 1, 3, 4, 6]))

    for f in files:
      f.close()


if __name__ == '__main__':
  test.main()
