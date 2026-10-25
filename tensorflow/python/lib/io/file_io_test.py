# This Python file uses the following encoding: utf-8
# [MODULE DOC]
# Unit tests for the file IO layer (file_io module). They exercise the file object (read,
# write, append, seek, tell, line iteration, encodings), path helpers (join), and module level
# helpers (exists, delete, glob, directory creation, copy, rename, atomic write, directory
# listing, walk, stat, file comparison, crc32, atomic-move query, registered schemes). Many
# tests run for both plain string paths and path-like objects. All tests use a temporary
# base directory created per test. Defines a small path-like helper class and a parameterized
# decorator; runs the test main when executed directly.
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
# =============================================================================
"""Testing File IO operations in file_io.py."""

import os.path

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


class PathLike(object):
  # [CLASS DOC]
  # HELPER: minimal path-like test object used to check that file IO accepts objects that
  # implement the filesystem path protocol. Holds a single name string; converts to that
  # string both as a filesystem path and as text.
  """Backport of pathlib.Path for Python < 3.6"""

  def __init__(self, name):
    # [FUNCTION DOC]
    # HELPER: stores the given path string as the name attribute. Returns nothing.
    self.name = name

  def __fspath__(self):
    # [FUNCTION DOC]
    # HELPER: path protocol hook; returns the stored path string.
    return self.name

  def __str__(self):
    # [FUNCTION DOC]
    # HELPER: text conversion; returns the stored path string.
    return self.name


run_all_path_types = parameterized.named_parameters(
    ("str", file_io.join),
    ("pathlike", lambda *paths: PathLike(file_io.join(*paths))))


class FileIoTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the file IO module. Shared fixture: a fresh base directory inside the
  # test temp directory, created before and removed after every test (stored in an attribute).
  # Parameterized path-type tests run once with plain strings and once with path-like objects.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: builds the base directory path under the temp directory, stores it, and
    # creates the directory.
    self._base_dir = file_io.join(self.get_temp_dir(), "base_dir")
    file_io.create_dir(self._base_dir)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture teardown: recursively deletes the base directory created in setup.
    file_io.delete_recursively(self._base_dir)

  def testEmptyFilename(self):
    # [FUNCTION DOC]
    # Opens a file object with an empty path in read mode and expects that reading raises a
    # not-found error.
    f = file_io.FileIO("", mode="r")
    with self.assertRaises(errors.NotFoundError):
      _ = f.read()

  def testJoinUrlLike(self):
    # [FUNCTION DOC]
    # For the in-memory, GCS and file URI schemes, joins components in several splits (scheme
    # alone, scheme plus first part, slashes inside components) and expects the same forward
    # slash joined URL each time.
    """file_io.join joins url-like filesystems with '/' on all platform."""
    for fs in ("ram://", "gcs://", "file://"):
      expected = fs + "exists/a/b/c.txt"
      self.assertEqual(file_io.join(fs, "exists", "a", "b", "c.txt"), expected)
      self.assertEqual(file_io.join(fs + "exists", "a", "b", "c.txt"), expected)
      self.assertEqual(file_io.join(fs, "exists/a", "b", "c.txt"), expected)
      self.assertEqual(file_io.join(fs, "exists", "a", "b/c.txt"), expected)

  def testJoinFilesystem(self):
    # [FUNCTION DOC]
    # For slash, backslash and the OS separator, joins plain relative and absolute components
    # (with separators at the start of later parts or the end of the last part) and expects
    # the result to equal the standard OS path join.
    """file_io.join respects the os.path.join behavior for native filesystems."""
    for sep in ("/", "\\", os.sep):
      self.assertEqual(os.path.join("a", "b", "c"), file_io.join("a", "b", "c"))
      self.assertEqual(
          os.path.join(sep + "a", "b", "c"), file_io.join(sep + "a", "b", "c"))
      self.assertEqual(
          os.path.join("a", sep + "b", "c"), file_io.join("a", sep + "b", "c"))
      self.assertEqual(
          os.path.join("a", "b", sep + "c"), file_io.join("a", "b", sep + "c"))
      self.assertEqual(
          os.path.join("a", "b", "c" + sep), file_io.join("a", "b", "c" + sep))

  @run_all_path_types
  def testFileDoesntExist(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. For a path that was never created, expects the existence
    # check to be false and reading the file to raise a not-found error.
    file_path = join(self._base_dir, "temp_file")
    self.assertFalse(file_io.file_exists(file_path))
    with self.assertRaises(errors.NotFoundError):
      _ = file_io.read_file_to_string(file_path)

  @run_all_path_types
  def testWriteToString(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Writes a string to a file, expects it to exist, and expects
    # reading it back to return the same string.
    file_path = join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    self.assertTrue(file_io.file_exists(file_path))
    file_contents = file_io.read_file_to_string(file_path)
    self.assertEqual("testing", file_contents)

  def testAtomicWriteStringToFile(self):
    # [FUNCTION DOC]
    # Performs an atomic write of a string to a new path and expects the file to exist with
    # exactly that content.
    file_path = file_io.join(self._base_dir, "temp_file")
    file_io.atomic_write_string_to_file(file_path, "testing")
    self.assertTrue(file_io.file_exists(file_path))
    file_contents = file_io.read_file_to_string(file_path)
    self.assertEqual("testing", file_contents)

  def testAtomicWriteStringToFileOverwriteFalse(self):
    # [FUNCTION DOC]
    # With overwrite disabled: the first atomic write succeeds; a second write to the same
    # path raises an already-exists error and leaves the old content; after deleting the file
    # a new write with overwrite disabled succeeds with the new content.
    file_path = file_io.join(self._base_dir, "temp_file")
    file_io.atomic_write_string_to_file(file_path, "old", overwrite=False)
    with self.assertRaises(errors.AlreadyExistsError):
      file_io.atomic_write_string_to_file(file_path, "new", overwrite=False)
    file_contents = file_io.read_file_to_string(file_path)
    self.assertEqual("old", file_contents)
    file_io.delete_file(file_path)
    file_io.atomic_write_string_to_file(file_path, "new", overwrite=False)
    file_contents = file_io.read_file_to_string(file_path)
    self.assertEqual("new", file_contents)

  @run_all_path_types
  def testReadBinaryMode(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Writes text, opens the file in bytes read mode, and expects
    # the read to return the matching bytes.
    file_path = join(self._base_dir, "temp_file")
    file_io.write_string_to_file(file_path, "testing")
    with file_io.FileIO(file_path, mode="rb") as f:
      self.assertEqual(b"testing", f.read())

  @run_all_path_types
  def testWriteBinaryMode(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Writes text through a bytes write mode file object, then
    # reads it in text mode and expects the same text.
    file_path = join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, "wb").write("testing")
    with file_io.FileIO(file_path, mode="r") as f:
      self.assertEqual("testing", f.read())

  def testAppend(self):
    # [FUNCTION DOC]
    # Writes a first line in write mode, appends two more lines in two separate append mode
    # sessions, and expects the read-back content to be all three lines in order.
    file_path = file_io.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="w") as f:
      f.write("begin\n")
    with file_io.FileIO(file_path, mode="a") as f:
      f.write("a1\n")
    with file_io.FileIO(file_path, mode="a") as f:
      f.write("a2\n")
    with file_io.FileIO(file_path, mode="r") as f:
      file_contents = f.read()
      self.assertEqual("begin\na1\na2\n", file_contents)

  def testMultipleFiles(self):
    # [FUNCTION DOC]
    # Stress test: 5000 times, opens a new file in read-write mode, writes text, flushes,
    # reads it back expecting the same text, and closes it (checks handles are released).
    file_prefix = file_io.join(self._base_dir, "temp_file")
    for i in range(5000):
      f = file_io.FileIO(file_prefix + str(i), mode="w+")
      f.write("testing")
      f.flush()
      self.assertEqual("testing", f.read())
      f.close()

  def testMultipleWrites(self):
    # [FUNCTION DOC]
    # Writes two pieces in one session inside a context manager and expects the file to hold
    # their concatenation.
    file_path = file_io.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="w") as f:
      f.write("line1\n")
      f.write("line2")
    file_contents = file_io.read_file_to_string(file_path)
    self.assertEqual("line1\nline2", file_contents)

  def testFileWriteBadMode(self):
    # [FUNCTION DOC]
    # Writing to a file object opened in read mode must raise a permission-denied error.
    file_path = file_io.join(self._base_dir, "temp_file")
    with self.assertRaises(errors.PermissionDeniedError):
      file_io.FileIO(file_path, mode="r").write("testing")

  def testFileReadBadMode(self):
    # [FUNCTION DOC]
    # Creates a file via a write mode object, then reading from a write mode object must raise
    # a permission-denied error.
    file_path = file_io.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    self.assertTrue(file_io.file_exists(file_path))
    with self.assertRaises(errors.PermissionDeniedError):
      file_io.FileIO(file_path, mode="w").read()

  @run_all_path_types
  def testFileDelete(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Creates a file, deletes it, and expects it to no longer
    # exist.
    file_path = join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    file_io.delete_file(file_path)
    self.assertFalse(file_io.file_exists(file_path))

  def testFileDeleteFail(self):
    # [FUNCTION DOC]
    # Deleting a file that does not exist must raise a not-found error.
    file_path = file_io.join(self._base_dir, "temp_file")
    with self.assertRaises(errors.NotFoundError):
      file_io.delete_file(file_path)

  def testGetMatchingFiles(self):
    # [FUNCTION DOC]
    # Creates a directory with four files (one with a literal star in its name). Expects the
    # glob of a star pattern to return all four, an empty tuple of patterns to return nothing,
    # a list of exact paths to return those paths, and after deleting the directory the paths
    # to no longer exist.
    dir_path = file_io.join(self._base_dir, "temp_dir")
    file_io.create_dir(dir_path)
    files = ["file1.txt", "file2.txt", "file3.txt", "file*.txt"]
    for name in files:
      file_path = file_io.join(dir_path, name)
      file_io.FileIO(file_path, mode="w").write("testing")
    expected_match = [file_io.join(dir_path, name) for name in files]
    self.assertItemsEqual(
        file_io.get_matching_files(file_io.join(dir_path, "file*.txt")),
        expected_match)
    self.assertItemsEqual(file_io.get_matching_files(tuple()), [])
    files_subset = [
        file_io.join(dir_path, files[0]),
        file_io.join(dir_path, files[2])
    ]
    self.assertItemsEqual(
        file_io.get_matching_files(files_subset), files_subset)
    file_io.delete_recursively(dir_path)
    self.assertFalse(file_io.file_exists(file_io.join(dir_path, "file3.txt")))

  def testGetMatchingFilesWhenParentDirContainsParantheses(self):
    # [FUNCTION DOC]
    # Creates a parent directory whose name contains parentheses holding two files (one with
    # parentheses in its name) and expects a star glob inside it to return both files.
    dir_path = file_io.join(self._base_dir, "dir_(special)")
    file_io.create_dir(dir_path)
    files = ["file1.txt", "file(2).txt"]
    for name in files:
      file_path = file_io.join(dir_path, name)
      file_io.FileIO(file_path, mode="w").write("testing")
    expected_match = [file_io.join(dir_path, name) for name in files]
    glob_pattern = file_io.join(dir_path, "*")
    self.assertItemsEqual(
        file_io.get_matching_files(glob_pattern), expected_match)

  @run_all_path_types
  def testCreateRecursiveDir(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Creates a nested directory chain twice (repeat creation must
    # succeed), writes a file in it and checks it exists, then deletes the top directory
    # recursively and expects the file to be gone.
    dir_path = join(self._base_dir, "temp_dir/temp_dir1/temp_dir2")
    file_io.recursive_create_dir(dir_path)
    file_io.recursive_create_dir(dir_path)  # repeat creation
    file_path = file_io.join(str(dir_path), "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    self.assertTrue(file_io.file_exists(file_path))
    file_io.delete_recursively(file_io.join(self._base_dir, "temp_dir"))
    self.assertFalse(file_io.file_exists(file_path))

  @run_all_path_types
  def testCopy(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Writes a file, copies it, expects the copy to exist, and
    # expects the source to be still readable with the same text and a position of seven.
    file_path = join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    copy_path = join(self._base_dir, "copy_file")
    file_io.copy(file_path, copy_path)
    self.assertTrue(file_io.file_exists(copy_path))
    f = file_io.FileIO(file_path, mode="r")
    self.assertEqual("testing", f.read())
    self.assertEqual(7, f.tell())

  def testCopyOverwrite(self):
    # [FUNCTION DOC]
    # Copies over an existing destination with overwrite enabled; expects the destination to
    # exist and the source content to remain readable.
    file_path = file_io.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    copy_path = file_io.join(self._base_dir, "copy_file")
    file_io.FileIO(copy_path, mode="w").write("copy")
    file_io.copy(file_path, copy_path, overwrite=True)
    self.assertTrue(file_io.file_exists(copy_path))
    self.assertEqual("testing", file_io.FileIO(file_path, mode="r").read())

  def testCopyOverwriteFalse(self):
    # [FUNCTION DOC]
    # Copying onto an existing destination with overwrite disabled must raise an
    # already-exists error.
    file_path = file_io.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    copy_path = file_io.join(self._base_dir, "copy_file")
    file_io.FileIO(copy_path, mode="w").write("copy")
    with self.assertRaises(errors.AlreadyExistsError):
      file_io.copy(file_path, copy_path, overwrite=False)

  @run_all_path_types
  def testRename(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Renames a file and expects the new path to exist and the old
    # one not to.
    file_path = join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    rename_path = join(self._base_dir, "rename_file")
    file_io.rename(file_path, rename_path)
    self.assertTrue(file_io.file_exists(rename_path))
    self.assertFalse(file_io.file_exists(file_path))

  def testRenameOverwrite(self):
    # [FUNCTION DOC]
    # Renames a file onto an existing destination with overwrite enabled; expects the
    # destination to exist and the source to be gone.
    file_path = file_io.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    rename_path = file_io.join(self._base_dir, "rename_file")
    file_io.FileIO(rename_path, mode="w").write("rename")
    file_io.rename(file_path, rename_path, overwrite=True)
    self.assertTrue(file_io.file_exists(rename_path))
    self.assertFalse(file_io.file_exists(file_path))

  def testRenameOverwriteFalse(self):
    # [FUNCTION DOC]
    # Renaming onto an existing destination with overwrite disabled must raise an
    # already-exists error and leave both files in place.
    file_path = file_io.join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    rename_path = file_io.join(self._base_dir, "rename_file")
    file_io.FileIO(rename_path, mode="w").write("rename")
    with self.assertRaises(errors.AlreadyExistsError):
      file_io.rename(file_path, rename_path, overwrite=False)
    self.assertTrue(file_io.file_exists(rename_path))
    self.assertTrue(file_io.file_exists(file_path))

  def testDeleteRecursivelyFail(self):
    # [FUNCTION DOC]
    # Recursively deleting a directory that does not exist must raise a not-found error.
    fake_dir_path = file_io.join(self._base_dir, "temp_dir")
    with self.assertRaises(errors.NotFoundError):
      file_io.delete_recursively(fake_dir_path)

  @run_all_path_types
  def testIsDirectory(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Expects the directory check to be false for a missing path,
    # true after creating the directory, false for a file inside it, and the statistics of the
    # directory to flag it as a directory.
    dir_path = join(self._base_dir, "test_dir")
    # Failure for a non-existing dir.
    self.assertFalse(file_io.is_directory(dir_path))
    file_io.create_dir(dir_path)
    self.assertTrue(file_io.is_directory(dir_path))
    file_path = join(str(dir_path), "test_file")
    file_io.FileIO(file_path, mode="w").write("test")
    # False for a file.
    self.assertFalse(file_io.is_directory(file_path))
    # Test that the value returned from `stat()` has `is_directory` set.
    file_statistics = file_io.stat(dir_path)
    self.assertTrue(file_statistics.is_directory)

  @run_all_path_types
  def testListDirectory(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Creates a directory with three files and a subdirectory
    # (with its own file) and expects the listing to contain exactly the three file names plus
    # the subdirectory name, not nested entries.
    dir_path = join(self._base_dir, "test_dir")
    file_io.create_dir(dir_path)
    files = ["file1.txt", "file2.txt", "file3.txt"]
    for name in files:
      file_path = join(str(dir_path), name)
      file_io.FileIO(file_path, mode="w").write("testing")
    subdir_path = join(str(dir_path), "sub_dir")
    file_io.create_dir(subdir_path)
    subdir_file_path = join(str(subdir_path), "file4.txt")
    file_io.FileIO(subdir_file_path, mode="w").write("testing")
    dir_list = file_io.list_directory(dir_path)
    self.assertItemsEqual(files + ["sub_dir"], dir_list)

  def testListDirectoryFailure(self):
    # [FUNCTION DOC]
    # Listing a directory that was never created must raise a not-found error.
    dir_path = file_io.join(self._base_dir, "test_dir")
    with self.assertRaises(errors.NotFoundError):
      file_io.list_directory(dir_path)

  def _setupWalkDirectories(self, dir_path):
    # [FUNCTION DOC]
    # HELPER: builds the directory tree used by the walk tests. Input is the root directory
    # path. Creates the root with one file and three subdirectories; the first subdirectory
    # gets one file and the second gets a nested subdirectory. Returns nothing.
    # Creating a file structure as follows
    # test_dir -> file: file1.txt; dirs: subdir1_1, subdir1_2, subdir1_3
    # subdir1_1 -> file: file3.txt
    # subdir1_2 -> dir: subdir2
    file_io.create_dir(dir_path)
    file_io.FileIO(
        file_io.join(dir_path, "file1.txt"), mode="w").write("testing")
    sub_dirs1 = ["subdir1_1", "subdir1_2", "subdir1_3"]
    for name in sub_dirs1:
      file_io.create_dir(file_io.join(dir_path, name))
    file_io.FileIO(
        file_io.join(dir_path, "subdir1_1/file2.txt"),
        mode="w").write("testing")
    file_io.create_dir(file_io.join(dir_path, "subdir1_2/subdir2"))

  @run_all_path_types
  def testWalkInOrder(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Builds the walk tree and walks it in pre-order. Expects all
    # five directories to be visited, the root first, a parent before its nested child, the
    # per-directory subdirectory and file lists to match the tree, and the root file listed
    # before the nested file.
    dir_path_str = file_io.join(self._base_dir, "test_dir")
    dir_path = join(self._base_dir, "test_dir")
    self._setupWalkDirectories(dir_path_str)
    # Now test the walk (in_order = True)
    all_dirs = []
    all_subdirs = []
    all_files = []
    for (w_dir, w_subdirs, w_files) in file_io.walk(dir_path, in_order=True):
      all_dirs.append(w_dir)
      all_subdirs.append(w_subdirs)
      all_files.append(w_files)
    self.assertItemsEqual(all_dirs, [dir_path_str] + [
        file_io.join(dir_path_str, item) for item in
        ["subdir1_1", "subdir1_2", "subdir1_2/subdir2", "subdir1_3"]
    ])
    self.assertEqual(dir_path_str, all_dirs[0])
    self.assertLess(
        all_dirs.index(file_io.join(dir_path_str, "subdir1_2")),
        all_dirs.index(file_io.join(dir_path_str, "subdir1_2/subdir2")))
    self.assertItemsEqual(all_subdirs[1:5], [[], ["subdir2"], [], []])
    self.assertItemsEqual(all_subdirs[0],
                          ["subdir1_1", "subdir1_2", "subdir1_3"])
    self.assertItemsEqual(all_files, [["file1.txt"], ["file2.txt"], [], [], []])
    self.assertLess(
        all_files.index(["file1.txt"]), all_files.index(["file2.txt"]))

  def testWalkPostOrder(self):
    # [FUNCTION DOC]
    # Builds the walk tree and walks it in post-order. Expects the root to be visited last, a
    # nested child before its parent, subdirectory and file lists matching the tree, and the
    # nested file listed before the root file.
    dir_path = file_io.join(self._base_dir, "test_dir")
    self._setupWalkDirectories(dir_path)
    # Now test the walk (in_order = False)
    all_dirs = []
    all_subdirs = []
    all_files = []
    for (w_dir, w_subdirs, w_files) in file_io.walk(dir_path, in_order=False):
      all_dirs.append(w_dir)
      all_subdirs.append(w_subdirs)
      all_files.append(w_files)
    self.assertItemsEqual(all_dirs, [
        file_io.join(dir_path, item) for item in
        ["subdir1_1", "subdir1_2/subdir2", "subdir1_2", "subdir1_3"]
    ] + [dir_path])
    self.assertEqual(dir_path, all_dirs[4])
    self.assertLess(
        all_dirs.index(file_io.join(dir_path, "subdir1_2/subdir2")),
        all_dirs.index(file_io.join(dir_path, "subdir1_2")))
    self.assertItemsEqual(all_subdirs[0:4], [[], [], ["subdir2"], []])
    self.assertItemsEqual(all_subdirs[4],
                          ["subdir1_1", "subdir1_2", "subdir1_3"])
    self.assertItemsEqual(all_files, [["file2.txt"], [], [], [], ["file1.txt"]])
    self.assertLess(
        all_files.index(["file2.txt"]), all_files.index(["file1.txt"]))

  def testWalkFailure(self):
    # [FUNCTION DOC]
    # Walking a directory that does not exist yields nothing and raises no error.
    dir_path = file_io.join(self._base_dir, "test_dir")
    # Try walking a directory that wasn't created.
    all_dirs = []
    all_subdirs = []
    all_files = []
    for (w_dir, w_subdirs, w_files) in file_io.walk(dir_path, in_order=False):
      all_dirs.append(w_dir)
      all_subdirs.append(w_subdirs)
      all_files.append(w_files)
    self.assertItemsEqual(all_dirs, [])
    self.assertItemsEqual(all_subdirs, [])
    self.assertItemsEqual(all_files, [])

  @run_all_path_types
  def testStat(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Writes seven bytes to a file and expects the statistics to
    # report length seven, a modification time (whole seconds) equal to the operating system's
    # value, and not be a directory.
    file_path = join(self._base_dir, "temp_file")
    file_io.FileIO(file_path, mode="w").write("testing")
    file_statistics = file_io.stat(file_path)
    os_statistics = os.stat(str(file_path))
    self.assertEqual(7, file_statistics.length)
    self.assertEqual(
        int(os_statistics.st_mtime), int(file_statistics.mtime_nsec / 1e9))
    self.assertFalse(file_statistics.is_directory)

  def testReadLine(self):
    # [FUNCTION DOC]
    # Writes a five-line text (including a blank line and no trailing newline) in read-write
    # mode, expects size 36, then expects successive line reads to return each line with its
    # newline and finally an empty string at end of file.
    file_path = file_io.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    self.assertEqual(36, f.size())
    self.assertEqual("testing1\n", f.readline())
    self.assertEqual("testing2\n", f.readline())
    self.assertEqual("testing3\n", f.readline())
    self.assertEqual("\n", f.readline())
    self.assertEqual("testing5", f.readline())
    self.assertEqual("", f.readline())

  def testRead(self):
    # [FUNCTION DOC]
    # Writes the same five-line text, expects size 36, then expects counted reads (9, 9, 1
    # characters) to return consecutive slices and an uncounted read to return the remainder.
    file_path = file_io.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    self.assertEqual(36, f.size())
    self.assertEqual("testing1\n", f.read(9))
    self.assertEqual("testing2\n", f.read(9))
    self.assertEqual("t", f.read(1))
    self.assertEqual("esting3\n\ntesting5", f.read())

  def testReadErrorReacquiresGil(self):
    # [FUNCTION DOC]
    # Regression test for a crash when a native read fails: reading a negative count other
    # than minus one must raise an invalid-argument error rather than crash. Brittle because it
    # relies on the Python layer passing the value through unchecked.
    file_path = file_io.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    with self.assertRaises(errors.InvalidArgumentError):
      # At present, this is sufficient to convince ourselves that the change
      # fixes the problem. That is, this test will seg fault without the change,
      # and pass with it. Unfortunately, this is brittle, as it relies on the
      # Python layer to pass the argument along to the wrapped C++ without
      # checking the argument itself.
      f.read(-2)

  def testTell(self):
    # [FUNCTION DOC]
    # Writes the five-line text, then checks the reported position starts at zero and advances
    # by the length of each line read (9, 18, 27, 28, 36), staying 36 after an end of file read.
    file_path = file_io.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    self.assertEqual(0, f.tell())
    self.assertEqual("testing1\n", f.readline())
    self.assertEqual(9, f.tell())
    self.assertEqual("testing2\n", f.readline())
    self.assertEqual(18, f.tell())
    self.assertEqual("testing3\n", f.readline())
    self.assertEqual(27, f.tell())
    self.assertEqual("\n", f.readline())
    self.assertEqual(28, f.tell())
    self.assertEqual("testing5", f.readline())
    self.assertEqual(36, f.tell())
    self.assertEqual("", f.readline())
    self.assertEqual(36, f.tell())

  def testSeek(self):
    # [FUNCTION DOC]
    # Checks absolute seeks forward and backward move the position and change the next line
    # read; a negative offset raises an invalid-argument error; calling with no arguments or
    # with both offset and the deprecated position raises a type error; the deprecated
    # position argument alone still seeks.
    file_path = file_io.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    self.assertEqual("testing1\n", f.readline())
    self.assertEqual(9, f.tell())

    # Seek to 18
    f.seek(18)
    self.assertEqual(18, f.tell())
    self.assertEqual("testing3\n", f.readline())

    # Seek back to 9
    f.seek(9)
    self.assertEqual(9, f.tell())
    self.assertEqual("testing2\n", f.readline())

    f.seek(0)
    self.assertEqual(0, f.tell())
    self.assertEqual("testing1\n", f.readline())

    with self.assertRaises(errors.InvalidArgumentError):
      f.seek(-1)

    with self.assertRaises(TypeError):
      f.seek()

    # TODO(jhseu): Delete after position deprecation.
    with self.assertRaises(TypeError):
      f.seek(offset=0, position=0)
    f.seek(position=9)
    self.assertEqual(9, f.tell())
    self.assertEqual("testing2\n", f.readline())

  def testSeekFromWhat(self):
    # [FUNCTION DOC]
    # Checks seeking relative to the current position (whence 1), the start (0) and the end
    # (2, with a negative offset equal to minus the size, landing at zero), and that an
    # unsupported whence value of 3 raises an invalid-argument error.
    file_path = file_io.join(self._base_dir, "temp_file")
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("testing1\ntesting2\ntesting3\n\ntesting5")
    self.assertEqual("testing1\n", f.readline())
    self.assertEqual(9, f.tell())

    # Seek to 18
    f.seek(9, 1)
    self.assertEqual(18, f.tell())
    self.assertEqual("testing3\n", f.readline())

    # Seek back to 9
    f.seek(9, 0)
    self.assertEqual(9, f.tell())
    self.assertEqual("testing2\n", f.readline())

    f.seek(-f.size(), 2)
    self.assertEqual(0, f.tell())
    self.assertEqual("testing1\n", f.readline())

    with self.assertRaises(errors.InvalidArgumentError):
      f.seek(0, 3)

  def testReadingIterator(self):
    # [FUNCTION DOC]
    # Writes five lines, then iterates over the file object and expects the collected lines to
    # equal the written lines in order.
    file_path = file_io.join(self._base_dir, "temp_file")
    data = ["testing1\n", "testing2\n", "testing3\n", "\n", "testing5"]
    with file_io.FileIO(file_path, mode="r+") as f:
      f.write("".join(data))
    actual_data = []
    for line in f:
      actual_data.append(line)
    self.assertSequenceEqual(actual_data, data)

  def testReadlines(self):
    # [FUNCTION DOC]
    # Writes five lines and flushes, then expects the read-all-lines call to return exactly
    # those lines.
    file_path = file_io.join(self._base_dir, "temp_file")
    data = ["testing1\n", "testing2\n", "testing3\n", "\n", "testing5"]
    f = file_io.FileIO(file_path, mode="r+")
    f.write("".join(data))
    f.flush()
    lines = f.readlines()
    self.assertSequenceEqual(lines, data)

  def testUTF8StringPath(self):
    # [FUNCTION DOC]
    # Uses a file name containing non-ASCII characters: writes text and reads it back in
    # bytes mode, expecting the matching bytes.
    file_path = file_io.join(self._base_dir, "UTF8测试_file")
    file_io.write_string_to_file(file_path, "testing")
    with file_io.FileIO(file_path, mode="rb") as f:
      self.assertEqual(b"testing", f.read())

  def testEof(self):
    # [FUNCTION DOC]
    # Writes seven characters, flushes, then reads one more than that; expects the full
    # content back with no exception.
    """Test that reading past EOF does not raise an exception."""

    file_path = file_io.join(self._base_dir, "temp_file")
    f = file_io.FileIO(file_path, mode="r+")
    content = "testing"
    f.write(content)
    f.flush()
    self.assertEqual(content, f.read(len(content) + 1))

  @run_all_path_types
  def testUTF8StringPathExists(self, join):
    # [FUNCTION DOC]
    # Parameterized by path type. Writes a file with a non-ASCII name and expects the
    # existence check to be true.
    file_path = join(self._base_dir, "UTF8测试_file_exist")
    file_io.write_string_to_file(file_path, "testing")
    v = file_io.file_exists(file_path)
    self.assertEqual(v, True)

  def testFilecmp(self):
    # [FUNCTION DOC]
    # Writes files with different sizes of content and one with equal content given as text
    # of a different string kind; expects the comparison to be false for different files and
    # true for equal ones.
    file1 = file_io.join(self._base_dir, "file1")
    file_io.write_string_to_file(file1, "This is a sentence\n" * 100)

    file2 = file_io.join(self._base_dir, "file2")
    file_io.write_string_to_file(file2, "This is another sentence\n" * 100)

    file3 = file_io.join(self._base_dir, "file3")
    file_io.write_string_to_file(file3, u"This is another sentence\n" * 100)

    self.assertFalse(file_io.filecmp(file1, file2))
    self.assertTrue(file_io.filecmp(file2, file3))

  def testFilecmpSameSize(self):
    # [FUNCTION DOC]
    # Writes files of equal size but differing content (forcing the checksum path); expects
    # the comparison false for differing files and true for equal ones.
    file1 = file_io.join(self._base_dir, "file1")
    file_io.write_string_to_file(file1, "This is a sentence\n" * 100)

    file2 = file_io.join(self._base_dir, "file2")
    file_io.write_string_to_file(file2, "This is b sentence\n" * 100)

    file3 = file_io.join(self._base_dir, "file3")
    file_io.write_string_to_file(file3, u"This is b sentence\n" * 100)

    self.assertFalse(file_io.filecmp(file1, file2))
    self.assertTrue(file_io.filecmp(file2, file3))

  def testFilecmpBinary(self):
    # [FUNCTION DOC]
    # Writes four files through bytes write mode objects (two identical, one differing by the
    # last character, one of different length); expects false for differing pairs and true for
    # the identical pair.
    file1 = file_io.join(self._base_dir, "file1")
    file_io.FileIO(file1, "wb").write("testing\n\na")

    file2 = file_io.join(self._base_dir, "file2")
    file_io.FileIO(file2, "wb").write("testing\n\nb")

    file3 = file_io.join(self._base_dir, "file3")
    file_io.FileIO(file3, "wb").write("testing\n\nb")

    file4 = file_io.join(self._base_dir, "file4")
    file_io.FileIO(file4, "wb").write("testing\n\ntesting")

    self.assertFalse(file_io.filecmp(file1, file2))
    self.assertFalse(file_io.filecmp(file1, file4))
    self.assertTrue(file_io.filecmp(file2, file3))

  def testFileCrc32(self):
    # [FUNCTION DOC]
    # Computes checksums of three text files (two identical); expects different content to
    # give different checksums and identical content to give equal ones.
    file1 = file_io.join(self._base_dir, "file1")
    file_io.write_string_to_file(file1, "This is a sentence\n" * 100)
    crc1 = file_io.file_crc32(file1)

    file2 = file_io.join(self._base_dir, "file2")
    file_io.write_string_to_file(file2, "This is another sentence\n" * 100)
    crc2 = file_io.file_crc32(file2)

    file3 = file_io.join(self._base_dir, "file3")
    file_io.write_string_to_file(file3, "This is another sentence\n" * 100)
    crc3 = file_io.file_crc32(file3)

    self.assertTrue(crc1 != crc2)
    self.assertEqual(crc2, crc3)

  def testFileCrc32WithBytes(self):
    # [FUNCTION DOC]
    # Same as the basic checksum test but using small block sizes of 24 bytes for two files
    # and minus one (read all at once) for the third; expects the checksum to be independent of
    # block size for identical content and different for different content.
    file1 = file_io.join(self._base_dir, "file1")
    file_io.write_string_to_file(file1, "This is a sentence\n" * 100)
    crc1 = file_io.file_crc32(file1, block_size=24)

    file2 = file_io.join(self._base_dir, "file2")
    file_io.write_string_to_file(file2, "This is another sentence\n" * 100)
    crc2 = file_io.file_crc32(file2, block_size=24)

    file3 = file_io.join(self._base_dir, "file3")
    file_io.write_string_to_file(file3, "This is another sentence\n" * 100)
    crc3 = file_io.file_crc32(file3, block_size=-1)

    self.assertTrue(crc1 != crc2)
    self.assertEqual(crc2, crc3)

  def testFileCrc32Binary(self):
    # [FUNCTION DOC]
    # Computes checksums of files written through bytes mode objects (two identical, one with
    # an extra newline); expects differing content to differ and identical content to match.
    file1 = file_io.join(self._base_dir, "file1")
    file_io.FileIO(file1, "wb").write("testing\n\n")
    crc1 = file_io.file_crc32(file1)

    file2 = file_io.join(self._base_dir, "file2")
    file_io.FileIO(file2, "wb").write("testing\n\n\n")
    crc2 = file_io.file_crc32(file2)

    file3 = file_io.join(self._base_dir, "file3")
    file_io.FileIO(file3, "wb").write("testing\n\n\n")
    crc3 = file_io.file_crc32(file3)

    self.assertTrue(crc1 != crc2)
    self.assertEqual(crc2, crc3)

  def testFileSeekableWithZip(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue (only visible on newer Python versions): saves a
    # compressed numpy archive, opens it through the public file class in bytes mode and loads
    # it with numpy, relying on the file object reporting itself as seekable. Success means no
    # exception while loading and iterating the items.
    # Note: Test case for GitHub issue 27276, issue only exposed in python 3.7+.
    filename = file_io.join(self._base_dir, "a.npz")
    np.savez_compressed(filename, {"a": 1, "b": 2})
    with gfile.GFile(filename, "rb") as f:
      info = np.load(f, allow_pickle=True)  # pylint: disable=unexpected-keyword-arg
    _ = [i for i in info.items()]

  def testHasAtomicMove(self):
    # [FUNCTION DOC]
    # Expects the atomic-move query to be true for an arbitrary local style path.
    self.assertTrue(file_io.has_atomic_move("/a/b/c"))

  def testGetRegisteredSchemes(self):
    # [FUNCTION DOC]
    # Expects the registered schemes to be the empty scheme, file and ram, plus optionally the
    # Google storage and hypercomputer schemes if present, each appearing exactly once.
    expected = ["", "file", "ram"]
    actual = file_io.get_registered_schemes()
    # Be flexible about additional schemes that may sometimes be registered when
    # this test is run, while still verifying each scheme appears just once.
    maybe_expected = ["gs", "hypercomputer"]
    for scheme in maybe_expected:
      if scheme in actual:
        expected.append(scheme)
    self.assertCountEqual(expected, actual)

  def testReadWriteWithEncoding(self):
    # [FUNCTION DOC]
    # Round-trips Japanese text with a non-default encoding: writes with the builtin file
    # opener and reads with the file class given that encoding, then the reverse; both
    # directions must return the original text.
    file_path = file_io.join(self._base_dir, "temp_file")
    with open(file_path, mode="w", encoding="cp932") as f:
      f.write("今日はいい天気")
    with file_io.FileIO(file_path, mode="r", encoding="cp932") as f:
      self.assertEqual(f.read(), "今日はいい天気")

    with file_io.FileIO(file_path, mode="w", encoding="cp932") as f:
      f.write("今日はいい天気")
    with open(file_path, mode="r", encoding="cp932") as f:
      self.assertEqual(f.read(), "今日はいい天気")


if __name__ == "__main__":
  test.main()
