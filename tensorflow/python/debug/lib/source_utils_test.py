# [MODULE DOC]
# Unit tests for the debugger's source_utils library. Covers the heuristic that decides whether a
# file path belongs to the TensorFlow Python library, annotating source files with the names of
# graph ops (or dumped tensors) created on each line, loading source files (including from
# archive files), and listing source files against a debug dump. Uses real sessions that write
# debug dumps to a temporary directory. Helpers here compute line numbers dynamically so the
# tests do not depend on fixed line positions. Importing has no side effects besides imports.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for source_utils."""

import ast
import os
import sys
import tempfile
import zipfile

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import googletest
from tensorflow.python.util import tf_inspect


def line_number_above():
  # [FUNCTION DOC]
  # HELPER: test utility that lets tests record line numbers of graph-building statements.
  # Inspects the call stack to find the line of the caller, then returns the line number of the
  # syntax-tree statement just before that call (older Python versions: call line minus one).
  # Takes no arguments. Reads and parses this very source file from disk. Returns an integer.
  """Get lineno of the AST node immediately above this function's call site.

  It is assumed that there is no empty line(s) between the call site and the
  preceding AST node.

  Returns:
    The lineno of the preceding AST node, at the same level of the AST.
    If the preceding AST spans multiple lines:
      - In Python 3.8+, the lineno of the first line is returned.
      - In older Python versions, the lineno of the last line is returned.
  """
  # https://bugs.python.org/issue12458: In Python 3.8, traceback started
  # to return the lineno of the first line of a multi-line continuation block,
  # instead of that of the last line. Therefore, in Python 3.8+, we use `ast` to
  # get the lineno of the first line.
  call_site_lineno = tf_inspect.stack()[1][2]
  if sys.version_info < (3, 8):
    return call_site_lineno - 1
  else:
    with open(__file__, "rb") as f:
      source_text = f.read().decode("utf-8")
    source_tree = ast.parse(source_text)
    prev_node = _find_preceding_ast_node(source_tree, call_site_lineno)
    return prev_node.lineno


def _find_preceding_ast_node(node, lineno):
  # [FUNCTION DOC]
  # HELPER: for line_number_above.
  # Recursively searches the body of a syntax-tree node (module, function, block) for the child
  # statement that starts on the given line number and returns the sibling statement right before
  # it. Inputs: node, a syntax-tree node with a body; lineno, an integer line number. Returns the
  # preceding statement node, or nothing if not found. Note that a match on the very first
  # statement of a body would pick the last statement of that body (index wraps around).
  """Find the ast node immediately before and not including lineno."""
  for i, child_node in enumerate(node.body):
    if child_node.lineno == lineno:
      return node.body[i - 1]
    if hasattr(child_node, "body"):
      found_node = _find_preceding_ast_node(child_node, lineno)
      if found_node:
        return found_node


class GuessIsTensorFlowLibraryTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the function that guesses whether a file path is part of the TensorFlow Python
  # library. Fixture: the normalized absolute path of this test file, stored per test.

  def setUp(self):
    # [FUNCTION DOC]
    # Stores the normalized absolute path of this test file as the current file path attribute.
    self.curr_file_path = os.path.normpath(os.path.abspath(__file__))

  def tearDown(self):
    # [FUNCTION DOC]
    # Resets the default graph so tests do not leak graph state.
    ops.reset_default_graph()

  def testGuessedBaseDirIsProbablyCorrect(self):
    # [FUNCTION DOC]
    # Checks that the guessed TensorFlow base directory name is either the plain package name or
    # the pip-style core package name.
    # In the non-pip world, code resides in "tensorflow/"
    # In the pip world, after virtual pip, code resides in "tensorflow_core/"
    # So, we have to check both of them
    self.assertIn(
        os.path.basename(source_utils._TENSORFLOW_BASEDIR),
        ["tensorflow", "tensorflow_core"])

  def testUnitTestFileReturnsFalse(self):
    # [FUNCTION DOC]
    # Asserts this unit test file itself is not classified as a TensorFlow library file.
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(self.curr_file_path))

  def testSourceUtilModuleReturnsTrue(self):
    # [FUNCTION DOC]
    # Asserts the source_utils module's own file is classified as a TensorFlow library file.
    self.assertTrue(
        source_utils.guess_is_tensorflow_py_library(source_utils.__file__))

  @test_util.run_v1_only("Tensor.op is not available in TF 2.x")
  def testFileInPythonKernelsPathReturnsTrue(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Creates a constant op and takes the file path of the last entry in
    # its creation traceback; asserts that path is classified as a TensorFlow library file.
    x = constant_op.constant(42.0, name="x")
    self.assertTrue(
        source_utils.guess_is_tensorflow_py_library(x.op.traceback[-1][0]))

  def testDebuggerExampleFilePathReturnsFalse(self):
    # [FUNCTION DOC]
    # Asserts that paths of the debugger example scripts (several example directories under the
    # installed package) are not classified as TensorFlow library files.
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(os.path.normpath(
            "site-packages/tensorflow/python/debug/examples/debug_mnist.py")))
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(os.path.normpath(
            "site-packages/tensorflow/python/debug/examples/v1/example_v1.py")))
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(os.path.normpath(
            "site-packages/tensorflow/python/debug/examples/v2/example_v2.py")))
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(os.path.normpath(
            "site-packages/tensorflow/python/debug/examples/v3/example_v3.py")))

  def testReturnsFalseForNonPythonFile(self):
    # [FUNCTION DOC]
    # Asserts a path with a non-Python extension next to this file is not a library Python file.
    self.assertFalse(
        source_utils.guess_is_tensorflow_py_library(
            os.path.join(os.path.dirname(self.curr_file_path), "foo.cc")))

  def testReturnsFalseForStdin(self):
    # [FUNCTION DOC]
    # Asserts the pseudo file name for standard input is not a library file.
    self.assertFalse(source_utils.guess_is_tensorflow_py_library("<stdin>"))

  def testReturnsFalseForEmptyFileName(self):
    # [FUNCTION DOC]
    # Asserts an empty file name is not a library file.
    self.assertFalse(source_utils.guess_is_tensorflow_py_library(""))


class SourceHelperTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for annotating source files with graph ops and dumped tensors, and for loading source
  # files. Fixture: a small graph (two constants, two variables, a matrix multiply) is run with
  # debugging dumps enabled; the dump directory object, the Python graph, the line number of each
  # creation statement and the helper call line are stored on the test instance.

  def createAndRunGraphHelper(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the tests in this class (called from setUp).
    # Sets the dump root to a temporary directory and records this file's path. Inside a session
    # it builds two constants, two variables and their matrix product, recording the line number
    # of each creation statement. Initializes the variables, runs the product with graph
    # partitions and watch settings so tensors are dumped to files, then loads the dump into a
    # dump-directory object and attaches the Python graph. Returns nothing; sets attributes.
    """Create and run a TensorFlow Graph to generate debug dumps.

    This is intentionally done in separate method, to make it easier to test
    the stack-top mode of source annotation.
    """

    self.dump_root = self.get_temp_dir()
    self.curr_file_path = os.path.abspath(
        tf_inspect.getfile(tf_inspect.currentframe()))

    # Run a simple TF graph to generate some debug dumps that can be used in
    # source annotation.
    with session.Session() as sess:
      self.u_init = constant_op.constant(
          np.array([[5.0, 3.0], [-1.0, 0.0]]), shape=[2, 2], name="u_init")
      self.u_init_line_number = line_number_above()

      self.u = variables.Variable(self.u_init, name="u")
      self.u_line_number = line_number_above()

      self.v_init = constant_op.constant(
          np.array([[2.0], [-1.0]]), shape=[2, 1], name="v_init")
      self.v_init_line_number = line_number_above()

      self.v = variables.Variable(self.v_init, name="v")
      self.v_line_number = line_number_above()

      self.w = math_ops.matmul(self.u, self.v, name="w")
      self.w_line_number = line_number_above()

      self.evaluate(self.u.initializer)
      self.evaluate(self.v.initializer)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options, sess.graph, debug_urls=["file://%s" % self.dump_root])
      run_metadata = config_pb2.RunMetadata()
      sess.run(self.w, options=run_options, run_metadata=run_metadata)

      self.dump = debug_data.DebugDumpDir(
          self.dump_root, partition_graphs=run_metadata.partition_graphs)
      self.dump.set_python_graph(sess.graph)

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the graph-and-dump helper, then records the line number of that call as the helper
    # line number (the line whose stack-top annotation is expected in non-stack-top mode).
    self.createAndRunGraphHelper()
    self.helper_line_number = line_number_above()

  def tearDown(self):
    # [FUNCTION DOC]
    # Deletes the dump directory if it exists and resets the default graph.
    if os.path.isdir(self.dump_root):
      file_io.delete_recursively(self.dump_root)
    ops.reset_default_graph()

  def testAnnotateWholeValidSourceFileGivesCorrectResult(self):
    # [FUNCTION DOC]
    # Annotates this whole file against the dump in default mode. Asserts each recorded creation
    # line carries the name of the op created there, and that the helper line is annotated with
    # all the ops as well.
    source_annotation = source_utils.annotate_source(self.dump,
                                                     self.curr_file_path)

    self.assertIn(self.u_init.op.name,
                  source_annotation[self.u_init_line_number])
    self.assertIn(self.u.op.name, source_annotation[self.u_line_number])
    self.assertIn(self.v_init.op.name,
                  source_annotation[self.v_init_line_number])
    self.assertIn(self.v.op.name, source_annotation[self.v_line_number])
    self.assertIn(self.w.op.name, source_annotation[self.w_line_number])

    # In the non-stack-top (default) mode, the helper line should be annotated
    # with all the ops as well.
    self.assertIn(self.u_init.op.name,
                  source_annotation[self.helper_line_number])
    self.assertIn(self.u.op.name, source_annotation[self.helper_line_number])
    self.assertIn(self.v_init.op.name,
                  source_annotation[self.helper_line_number])
    self.assertIn(self.v.op.name, source_annotation[self.helper_line_number])
    self.assertIn(self.w.op.name, source_annotation[self.helper_line_number])

  def testAnnotateWithStackTopGivesCorrectResult(self):
    # [FUNCTION DOC]
    # Annotates with the stack-top option on. Asserts the creation lines carry their op names and
    # that the helper line is absent from the annotation.
    source_annotation = source_utils.annotate_source(
        self.dump, self.curr_file_path, file_stack_top=True)

    self.assertIn(self.u_init.op.name,
                  source_annotation[self.u_init_line_number])
    self.assertIn(self.u.op.name, source_annotation[self.u_line_number])
    self.assertIn(self.v_init.op.name,
                  source_annotation[self.v_init_line_number])
    self.assertIn(self.v.op.name, source_annotation[self.v_line_number])
    self.assertIn(self.w.op.name, source_annotation[self.w_line_number])

    # In the stack-top mode, the helper line should not have been annotated.
    self.assertNotIn(self.helper_line_number, source_annotation)

  def testAnnotateSubsetOfLinesGivesCorrectResult(self):
    # [FUNCTION DOC]
    # Annotates only a two-line range starting at the first variable's line. Asserts that
    # variable's line is annotated and the later variable's line is not in the result.
    source_annotation = source_utils.annotate_source(
        self.dump,
        self.curr_file_path,
        min_line=self.u_line_number,
        max_line=self.u_line_number + 1)

    self.assertIn(self.u.op.name, source_annotation[self.u_line_number])
    self.assertNotIn(self.v_line_number, source_annotation)

  def testAnnotateDumpedTensorsGivesCorrectResult(self):
    # [FUNCTION DOC]
    # Annotates with the dumped-tensors option on. Asserts the lines show tensor names (not op
    # names) for the variables and the product, and the helper line shows them too. Constants may
    # not be dumped because of constant folding, so they are not checked.
    source_annotation = source_utils.annotate_source(
        self.dump, self.curr_file_path, do_dumped_tensors=True)

    # Note: Constant Tensors u_init and v_init may not get dumped due to
    #   constant-folding.
    self.assertIn(self.u.name, source_annotation[self.u_line_number])
    self.assertIn(self.v.name, source_annotation[self.v_line_number])
    self.assertIn(self.w.name, source_annotation[self.w_line_number])

    self.assertNotIn(self.u.op.name, source_annotation[self.u_line_number])
    self.assertNotIn(self.v.op.name, source_annotation[self.v_line_number])
    self.assertNotIn(self.w.op.name, source_annotation[self.w_line_number])

    self.assertIn(self.u.name, source_annotation[self.helper_line_number])
    self.assertIn(self.v.name, source_annotation[self.helper_line_number])
    self.assertIn(self.w.name, source_annotation[self.helper_line_number])

  def testCallingAnnotateSourceWithoutPythonGraphRaisesException(self):
    # [FUNCTION DOC]
    # Removes the Python graph from the dump, then asserts that annotating a source file raises a
    # value error.
    self.dump.set_python_graph(None)
    with self.assertRaises(ValueError):
      source_utils.annotate_source(self.dump, self.curr_file_path)

  def testCallingAnnotateSourceOnUnrelatedSourceFileDoesNotError(self):
    # [FUNCTION DOC]
    # Writes a temporary unrelated source file, annotates it against the dump, and asserts the
    # result is an empty mapping (no error). Removes the temporary file afterwards.
    # Create an unrelated source file.
    fd, unrelated_source_path = tempfile.mkstemp()
    with open(fd, "wt") as source_file:
      source_file.write("print('hello, world')\n")

    self.assertEqual({},
                     source_utils.annotate_source(self.dump,
                                                  unrelated_source_path))

    # Clean up unrelated source file.
    os.remove(unrelated_source_path)

  def testLoadingPythonSourceFileWithNonAsciiChars(self):
    # [FUNCTION DOC]
    # Writes a temporary file containing a non-ASCII character encoded as UTF-8, loads it with the
    # source loader, and asserts the returned lines match the text (with a trailing empty line).
    # Removes the temporary file afterwards.
    fd, source_path = tempfile.mkstemp()
    with open(fd, "wb") as source_file:
      source_file.write(u"print('\U0001f642')\n".encode("utf-8"))
    source_lines, _ = source_utils.load_source(source_path)
    self.assertEqual(source_lines, [u"print('\U0001f642')", u""])
    # Clean up unrelated source file.
    os.remove(source_path)

  def testLoadNonexistentNonParPathFailsWithIOError(self):
    # [FUNCTION DOC]
    # Loading a nonexistent plain path must raise an IO error whose message says the file neither
    # exists nor can be loaded and mentions archive files.
    bad_path = os.path.join(self.get_temp_dir(), "nonexistent.py")
    with self.assertRaisesRegex(IOError,
                                "neither exists nor can be loaded.*par.*"):
      source_utils.load_source(bad_path)

  def testLoadingPythonSourceFileInParFileSucceeds(self):
    # [FUNCTION DOC]
    # Creates a small Python file and zips it into an archive file with a .par extension, then
    # loads the source through a path that goes inside the archive. Asserts the loaded lines
    # equal the original two lines plus a trailing empty line.
    # Create the .par file first.
    temp_file_path = os.path.join(self.get_temp_dir(), "model.py")
    with open(temp_file_path, "wb") as f:
      f.write(b"import tensorflow as tf\nx = tf.constant(42.0)\n")
    par_path = os.path.join(self.get_temp_dir(), "train_model.par")
    with zipfile.ZipFile(par_path, "w") as zf:
      zf.write(temp_file_path, os.path.join("tensorflow_models", "model.py"))

    source_path = os.path.join(par_path, "tensorflow_models", "model.py")
    source_lines, _ = source_utils.load_source(source_path)
    self.assertEqual(
        source_lines, ["import tensorflow as tf", "x = tf.constant(42.0)", ""])

  def testLoadingPythonSourceFileInParFileFailsRaisingIOError(self):
    # [FUNCTION DOC]
    # Builds the same .par archive as the success test, but loads a nonexistent member path inside
    # it. Asserts an IO error is raised saying the file neither exists nor can be loaded.
    # Create the .par file first.
    temp_file_path = os.path.join(self.get_temp_dir(), "model.py")
    with open(temp_file_path, "wb") as f:
      f.write(b"import tensorflow as tf\nx = tf.constant(42.0)\n")
    par_path = os.path.join(self.get_temp_dir(), "train_model.par")
    with zipfile.ZipFile(par_path, "w") as zf:
      zf.write(temp_file_path, os.path.join("tensorflow_models", "model.py"))

    source_path = os.path.join(par_path, "tensorflow_models", "nonexistent.py")
    with self.assertRaisesRegex(IOError,
                                "neither exists nor can be loaded.*par.*"):
      source_utils.load_source(source_path)


@test_util.run_v1_only("Sessions are not available in TF 2.x")
class ListSourceAgainstDumpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Graph-mode-only tests for listing source files against a debug dump. Fixture: a graph with a
  # while loop is run with dumping enabled; the dump object, this file's path and the line number
  # of the first graph-building statement are stored on the instance.

  def createAndRunGraphWithWhileLoop(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for this class (called from setUp).
    # Sets the dump root to a temporary directory, records this file's path, and in a session
    # builds a while loop (add two while the value is less than sixteen, starting from ten),
    # recording the line number of the loop-body statement. Runs it with dumping enabled and
    # loads the dump with the Python graph attached. Returns nothing; sets attributes.
    """Create and run a TensorFlow Graph with a while loop to generate dumps."""

    self.dump_root = self.get_temp_dir()
    self.curr_file_path = os.path.abspath(
        tf_inspect.getfile(tf_inspect.currentframe()))

    # Run a simple TF graph to generate some debug dumps that can be used in
    # source annotation.
    with session.Session() as sess:
      loop_body = lambda i: math_ops.add(i, 2)
      self.traceback_first_line = line_number_above()

      loop_cond = lambda i: math_ops.less(i, 16)

      i = constant_op.constant(10, name="i")
      loop = while_loop.while_loop(loop_cond, loop_body, [i])

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options, sess.graph, debug_urls=["file://%s" % self.dump_root])
      run_metadata = config_pb2.RunMetadata()
      sess.run(loop, options=run_options, run_metadata=run_metadata)

      self.dump = debug_data.DebugDumpDir(
          self.dump_root, partition_graphs=run_metadata.partition_graphs)
      self.dump.set_python_graph(sess.graph)

  def setUp(self):
    # [FUNCTION DOC]
    # Builds and runs the while-loop graph with debug dumps via the helper.
    self.createAndRunGraphWithWhileLoop()

  def tearDown(self):
    # [FUNCTION DOC]
    # Deletes the dump directory if present and resets the default graph.
    if os.path.isdir(self.dump_root):
      file_io.delete_recursively(self.dump_root)
    ops.reset_default_graph()

  def testGenerateSourceList(self):
    # [FUNCTION DOC]
    # Lists source files against the dump. Asserts file paths are sorted and unique, every entry is
    # a six-element tuple, and for this file: not a library file, 12 nodes, 14 tensors, 39 dumps
    # (per the table in the comments), and the first line equals the recorded loop-body line.
    source_list = source_utils.list_source_files_against_dump(self.dump)

    # Assert that the file paths are sorted and unique.
    file_paths = [item[0] for item in source_list]
    self.assertEqual(sorted(file_paths), file_paths)
    self.assertEqual(len(set(file_paths)), len(file_paths))

    # Assert that each item of source_list has length 6.
    for item in source_list:
      self.assertTrue(isinstance(item, tuple))
      self.assertEqual(6, len(item))

    # The while loop body should have executed 3 times. The following table
    # lists the tensors and how many times each of them is dumped.
    #   Tensor name            # of times dumped:
    #   i:0                    1
    #   while/Enter:0          1
    #   while/Merge:0          4
    #   while/Merge:1          4
    #   while/Less/y:0         4
    #   while/Less:0           4
    #   while/LoopCond:0       4
    #   while/Switch:0         1
    #   while/Switch:1         3
    #   while/Identity:0       3
    #   while/Add/y:0          3
    #   while/Add:0            3
    #   while/NextIteration:0  3
    #   while/Exit:0           1
    # ----------------------------
    #   (Total)                39
    #
    # The total number of nodes is 12.
    # The total number of tensors is 14 (2 of the nodes have 2 outputs:
    #   while/Merge, while/Switch).

    _, is_tf_py_library, num_nodes, num_tensors, num_dumps, first_line = (
        source_list[file_paths.index(self.curr_file_path)])
    self.assertFalse(is_tf_py_library)
    self.assertEqual(12, num_nodes)
    self.assertEqual(14, num_tensors)
    self.assertEqual(39, num_dumps)
    self.assertEqual(self.traceback_first_line, first_line)

  def testGenerateSourceListWithNodeNameFilter(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment inside says each item has length 4, but the assertion checks six.
    # Lists source files with a node-name allowlist pattern matching the add nodes. Asserts paths
    # are sorted and unique, entries are six-element tuples, and for this file: not a library
    # file, 2 nodes, 2 tensors and 6 dumped tensors.
    source_list = source_utils.list_source_files_against_dump(
        self.dump, node_name_regex_allowlist=r"while/Add.*")

    # Assert that the file paths are sorted.
    file_paths = [item[0] for item in source_list]
    self.assertEqual(sorted(file_paths), file_paths)
    self.assertEqual(len(set(file_paths)), len(file_paths))

    # Assert that each item of source_list has length 4.
    for item in source_list:
      self.assertTrue(isinstance(item, tuple))
      self.assertEqual(6, len(item))

    # Due to the node-name filtering the result should only contain 2 nodes
    # and 2 tensors. The total number of dumped tensors should be 6:
    #   while/Add/y:0          3
    #   while/Add:0            3
    _, is_tf_py_library, num_nodes, num_tensors, num_dumps, _ = (
        source_list[file_paths.index(self.curr_file_path)])
    self.assertFalse(is_tf_py_library)
    self.assertEqual(2, num_nodes)
    self.assertEqual(2, num_tensors)
    self.assertEqual(6, num_dumps)

  def testGenerateSourceListWithPathRegexFilter(self):
    # [FUNCTION DOC]
    # Lists source files with a path allowlist pattern matching only this file's base name.
    # Asserts exactly one entry, the file path equals this file, it is not a library file, and it
    # has 12 nodes, 14 tensors, 39 dumps and the recorded first line.
    curr_file_basename = os.path.basename(self.curr_file_path)
    source_list = source_utils.list_source_files_against_dump(
        self.dump,
        path_regex_allowlist=(".*" + curr_file_basename.replace(".", "\\.") +
                              "$"))

    self.assertEqual(1, len(source_list))
    (file_path, is_tf_py_library, num_nodes, num_tensors, num_dumps,
     first_line) = source_list[0]
    self.assertEqual(self.curr_file_path, file_path)
    self.assertFalse(is_tf_py_library)
    self.assertEqual(12, num_nodes)
    self.assertEqual(14, num_tensors)
    self.assertEqual(39, num_dumps)
    self.assertEqual(self.traceback_first_line, first_line)


if __name__ == "__main__":
  googletest.main()
