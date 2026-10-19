# [MODULE DOC]
# Utilities for relating Python source files to TensorFlow graphs and debug data: guessing
# whether a file belongs to the TensorFlow library, loading source text (including from .par
# archives), annotating source lines with ops, dumped tensors or profile data, and listing source
# files involved in a dump. Defines module-level base directories for TensorFlow and Abseil,
# computed at import time, and the source-suffix constants. Used by the debugger front ends and
# the remote source sender.
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
"""Classes and functions that help to inspect Python source w.r.t. TF graphs."""

import collections
import os
import re
import zipfile

from absl import app
import numpy as np

from tensorflow.python.debug.lib import profiling


_TENSORFLOW_BASEDIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.normpath(os.path.abspath(__file__))))))

_ABSL_BASEDIR = os.path.dirname(app.__file__)


UNCOMPILED_SOURCE_SUFFIXES = (".py")
COMPILED_SOURCE_SUFFIXES = (".pyc", ".pyo")


def _norm_abs_path(file_path):
  # [FUNCTION DOC]
  # HELPER: path normalization used across this module. Returns the normalized absolute form of
  # the given path.
  return os.path.normpath(os.path.abspath(file_path))


def is_extension_uncompiled_python_source(file_path):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the uncompiled-suffix constant is a plain string (parentheses without a comma),
  # not a tuple, so the membership test is a substring test: extensions such as ".p", ".py" or
  # the empty string also count as uncompiled Python source.
  # Returns true if the lower-cased extension of the file path is considered an uncompiled
  # Python source suffix. Parameter: file path string. Pure function.
  _, extension = os.path.splitext(file_path)
  return extension.lower() in UNCOMPILED_SOURCE_SUFFIXES


def is_extension_compiled_python_source(file_path):
  # [FUNCTION DOC]
  # Returns true if the file path's extension, lower-cased, is one of the compiled Python
  # suffixes (pyc or pyo). Parameter: file path string. Pure function.
  _, extension = os.path.splitext(file_path)
  return extension.lower() in COMPILED_SOURCE_SUFFIXES


def _convert_watch_key_to_tensor_name(watch_key):
  # [FUNCTION DOC]
  # HELPER: for the source annotation function. Converts a debug watch key (tensor name followed
  # by a colon and the debug op name) to the tensor name by dropping everything from the last
  # colon on. Note: if the key has no colon, the last character is dropped (search returns
  # minus one).
  return watch_key[:watch_key.rfind(":")]


def guess_is_tensorflow_py_library(py_file_path):
  # [FUNCTION DOC]
  # Heuristically decides whether a file belongs to the TensorFlow (or Abseil) Python library.
  # Parameter: file path. Returns false if the extension is not a Python source or compiled
  # extension; otherwise normalizes the path and returns true only if it starts with the
  # TensorFlow base directory or the Abseil base directory, does not end with the test-file
  # suffix, and does not lie under the debug examples directory. Pure function, no side effects.
  """Guess whether a Python source file is a part of the tensorflow library.

  Special cases:
    1) Returns False for unit-test files in the library (*_test.py),
    2) Returns False for files under python/debug/examples.

  Args:
    py_file_path: full path of the Python source file in question.

  Returns:
    (`bool`) Whether the file is inferred to be a part of the tensorflow
      library.
  """
  if (not is_extension_uncompiled_python_source(py_file_path) and
      not is_extension_compiled_python_source(py_file_path)):
    return False
  py_file_path = _norm_abs_path(py_file_path)
  return ((py_file_path.startswith(_TENSORFLOW_BASEDIR) or
           py_file_path.startswith(_ABSL_BASEDIR)) and
          not py_file_path.endswith("_test.py") and
          (os.path.normpath("tensorflow/python/debug/examples") not in
           os.path.normpath(py_file_path)))


def load_source(source_file_path):
  # [FUNCTION DOC]
  # Reads a Python source file. If the path is an existing file, reads its bytes and decodes them
  # as UTF-8; otherwise tries to read it from inside a .par archive. Parameter: the path.
  # Returns a pair: the list of lines (split on newline, so a trailing newline yields a final
  # empty entry), and the width needed to print line numbers (ceiling of the base-ten logarithm
  # of the line count plus three). Raises an IO error if the path is neither a file nor loadable
  # from an archive.
  """Load the content of a Python source code file.

  This function covers the following case:
    1. source_file_path points to an existing Python (.py) file on the
       file system.
    2. source_file_path is a path within a .par file (i.e., a zip-compressed,
       self-contained Python executable).

  Args:
    source_file_path: Path to the Python source file to read.

  Returns:
    A length-2 tuple:
      - Lines of the source file, as a `list` of `str`s.
      - The width of the string needed to show the line number in the file.
        This is calculated based on the number of lines in the source file.

  Raises:
    IOError: if loading is unsuccessful.
  """
  if os.path.isfile(source_file_path):
    with open(source_file_path, "rb") as f:
      source_text = f.read().decode("utf-8")
    source_lines = source_text.split("\n")
  else:
    # One possible reason why the file doesn't exist is that it's a path
    # inside a .par file. Try that possibility.
    source_lines = _try_load_par_source(source_file_path)
    if source_lines is None:
      raise IOError(
          "Source path neither exists nor can be loaded as a .par file: %s" %
          source_file_path)
  line_num_width = int(np.ceil(np.log10(len(source_lines)))) + 3
  return source_lines, line_num_width


def _try_load_par_source(source_file_path):
  # [FUNCTION DOC]
  # HELPER: fallback reader for the source loader when a path lies inside a .par archive.
  # Walks the path's prefixes from longest to shortest looking for one that ends with the archive
  # extension and is an existing file; opens it as a zip, and if the remaining relative path
  # matches an archive member (after path normalization), returns that member's text decoded as
  # UTF-8 split on newlines. Returns nothing/null if no archive or member matches.
  """Try loading the source code inside a .par file.

  A .par file is a zip-compressed, self-contained Python executable.
  It contains the content of individual Python source files that can
  be read only through extracting from the zip file.

  Args:
    source_file_path: The full path to the file inside the .par file. This
      path should include the path to the .par file itself, followed by the
      intra-par path, e.g.,
      "/tmp/my_executable.par/org-tensorflow/tensorflow/python/foo/bar.py".

  Returns:
    If successful, lines of the source file as a `list` of `str`s.
    Else, `None`.
  """
  prefix_path = source_file_path
  while True:
    prefix_path, basename = os.path.split(prefix_path)
    if not basename:
      break
    suffix_path = os.path.normpath(
        os.path.relpath(source_file_path, start=prefix_path))
    if prefix_path.endswith(".par") and os.path.isfile(prefix_path):
      with zipfile.ZipFile(prefix_path) as z:
        norm_names = [os.path.normpath(name) for name in z.namelist()]
        if suffix_path in norm_names:
          with z.open(z.namelist()[norm_names.index(suffix_path)]) as zf:
            source_text = zf.read().decode("utf-8")
            return source_text.split("\n")


def annotate_source(dump,
                    source_file_path,
                    do_dumped_tensors=False,
                    file_stack_top=False,
                    min_line=None,
                    max_line=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring types do_dumped_tensors as a string, but it is used as a boolean.
  # Maps source lines of one file to the names of the ops (or dumped tensors) created there.
  # Parameters: dump is a debug dump object whose Python graph is loaded; source_file_path is the
  # file to annotate (normalized to an absolute path); do_dumped_tensors (default false) selects
  # unique tensor names derived from the op's debug watch keys instead of op names;
  # file_stack_top (default false) limits each op to the first matching frame, walking its
  # traceback from the innermost frame outward; min_line (inclusive) and max_line (exclusive) are
  # optional 1-based bounds. Returns a mapping from line number to a list of names, appended in
  # graph order. Raises a value error if the dump has no Python graph. The file is not changed.
  """Annotate a Python source file with a list of ops created at each line.

  (The annotation doesn't change the source file itself.)

  Args:
    dump: (`DebugDumpDir`) A `DebugDumpDir` object of which the Python graph
      has been loaded.
    source_file_path: (`str`) Path to the source file being annotated.
    do_dumped_tensors: (`str`) Whether dumped Tensors, instead of ops are to be
      used to annotate the source file.
    file_stack_top: (`bool`) Whether only the top stack trace in the
      specified source file is to be annotated.
    min_line: (`None` or `int`) The 1-based line to start annotate the source
      file from (inclusive).
    max_line: (`None` or `int`) The 1-based line number to end the annotation
      at (exclusive).

  Returns:
    A `dict` mapping 1-based line number to a list of op name(s) created at
      that line, or tensor names if `do_dumped_tensors` is True.

  Raises:
    ValueError: If the dump object does not have a Python graph set.
  """

  py_graph = dump.python_graph
  if not py_graph:
    raise ValueError("Cannot perform source annotation due to a lack of set "
                     "Python graph in the dump object")

  source_file_path = _norm_abs_path(source_file_path)

  line_to_op_names = {}
  for op in py_graph.get_operations():
    for file_path, line_number, _, _ in reversed(dump.node_traceback(op.name)):
      if (min_line is not None and line_number < min_line or
          max_line is not None and line_number >= max_line):
        continue

      if _norm_abs_path(file_path) != source_file_path:
        continue

      if do_dumped_tensors:
        watch_keys = dump.debug_watch_keys(op.name)
        # Convert watch keys to unique Tensor names.
        items_to_append = list(
            set(map(_convert_watch_key_to_tensor_name, watch_keys)))
      else:
        items_to_append = [op.name]

      if line_number in line_to_op_names:
        line_to_op_names[line_number].extend(items_to_append)
      else:
        line_to_op_names[line_number] = items_to_append

      if file_stack_top:
        break

  return line_to_op_names


def list_source_files_against_dump(dump,
                                   path_regex_allowlist=None,
                                   node_name_regex_allowlist=None):
  # [FUNCTION DOC]
  # Summarizes, per Python source file, the ops and tensors of a debug dump that were created by
  # that file. Parameters: dump is a debug dump directory object with its Python graph loaded;
  # path_regex_allowlist and node_name_regex_allowlist are optional regular-expression strings;
  # file paths must match the first (from the start) and op names the second, otherwise they are
  # skipped. Files that do not exist on disk or fail the path filter are remembered and skipped.
  # For each retained file it tracks the set of op names, the set of output tensor names, the
  # smallest line number, and the number of dumps of each watched tensor (looked up once per
  # tensor from the dump). Returns a list sorted by file path of tuples: path, whether it is
  # guessed to be in the TensorFlow library, node count, tensor count, dump count, and first
  # line. Raises a value error if the dump has no Python graph.
  """Generate a list of source files with information regarding ops and tensors.

  Args:
    dump: (`DebugDumpDir`) A `DebugDumpDir` object of which the Python graph
      has been loaded.
    path_regex_allowlist: A regular-expression filter for source file path.
    node_name_regex_allowlist: A regular-expression filter for node names.

  Returns:
    A list of tuples regarding the Python source files involved in constructing
    the ops and tensors contained in `dump`. Each tuple is:
      (source_file_path, is_tf_library, num_nodes, num_tensors, num_dumps,
       first_line)

      is_tf_library: (`bool`) A guess of whether the file belongs to the
        TensorFlow Python library.
      num_nodes: How many nodes were created by lines of this source file.
        These include nodes with dumps and those without.
      num_tensors: How many Tensors were created by lines of this source file.
        These include Tensors with dumps and those without.
      num_dumps: How many debug Tensor dumps were from nodes (and Tensors)
        that were created by this source file.
      first_line: The first line number (1-based) that created any nodes or
        Tensors in this source file.

    The list is sorted by ascending order of source_file_path.

  Raises:
    ValueError: If the dump object does not have a Python graph set.
  """

  py_graph = dump.python_graph
  if not py_graph:
    raise ValueError("Cannot generate source list due to a lack of set "
                     "Python graph in the dump object")

  path_to_node_names = collections.defaultdict(set)
  path_to_tensor_names = collections.defaultdict(set)
  path_to_first_line = {}
  tensor_name_to_num_dumps = {}

  path_regex = (
      re.compile(path_regex_allowlist) if path_regex_allowlist else None)
  node_name_regex = (
      re.compile(node_name_regex_allowlist)
      if node_name_regex_allowlist else None)

  to_skip_file_paths = set()
  for op in py_graph.get_operations():
    if node_name_regex and not node_name_regex.match(op.name):
      continue

    for file_path, line_number, _, _ in dump.node_traceback(op.name):
      file_path = _norm_abs_path(file_path)
      if (file_path in to_skip_file_paths or
          path_regex and not path_regex.match(file_path) or
          not os.path.isfile(file_path)):
        to_skip_file_paths.add(file_path)
        continue

      path_to_node_names[file_path].add(op.name)
      if file_path in path_to_first_line:
        if path_to_first_line[file_path] > line_number:
          path_to_first_line[file_path] = line_number
      else:
        path_to_first_line[file_path] = line_number

      for output_tensor in op.outputs:
        tensor_name = output_tensor.name
        path_to_tensor_names[file_path].add(tensor_name)

      watch_keys = dump.debug_watch_keys(op.name)
      for watch_key in watch_keys:
        node_name, output_slot, debug_op = watch_key.split(":")
        tensor_name = "%s:%s" % (node_name, output_slot)
        if tensor_name not in tensor_name_to_num_dumps:
          tensor_name_to_num_dumps[tensor_name] = len(
              dump.get_tensors(node_name, int(output_slot), debug_op))

  path_to_num_dumps = {}
  for path in path_to_tensor_names:
    path_to_num_dumps[path] = sum(
        tensor_name_to_num_dumps.get(tensor_name, 0)
        for tensor_name in path_to_tensor_names[path])

  output = []
  for file_path in path_to_node_names:
    output.append((
        file_path,
        guess_is_tensorflow_py_library(file_path),
        len(path_to_node_names.get(file_path, {})),
        len(path_to_tensor_names.get(file_path, {})),
        path_to_num_dumps.get(file_path, 0),
        path_to_first_line[file_path]))

  return sorted(output, key=lambda x: x[0])


def annotate_source_against_profile(profile_data,
                                    source_file_path,
                                    node_name_filter=None,
                                    op_type_filter=None,
                                    min_line=None,
                                    max_line=None):
  # [FUNCTION DOC]
  # Aggregates profiling data per source line of one file. Parameters: profile_data is a list of
  # profile records (each with file path, line number, op type and node execution stats);
  # source_file_path is the file to annotate (normalized to an absolute path); node_name_filter
  # and op_type_filter are optional regular-expression strings matched (from the start) against
  # the node name and op type; min_line (inclusive) and max_line (exclusive) optionally bound the
  # 1-based line numbers. Records with no file path, a different file, an out-of-range line or
  # failing a filter are skipped. Returns a mapping from line number to an aggregate profile
  # summary: the first record at a line creates the aggregate, later ones are added to it.
  # Does not modify the source file or the records' lists. Raises nothing explicitly.
  """Annotate a Python source file with profiling information at each line.

  (The annotation doesn't change the source file itself.)

  Args:
    profile_data: (`list` of `ProfileDatum`) A list of `ProfileDatum`.
    source_file_path: (`str`) Path to the source file being annotated.
    node_name_filter: Regular expression to filter by node name.
    op_type_filter: Regular expression to filter by op type.
    min_line: (`None` or `int`) The 1-based line to start annotate the source
      file from (inclusive).
    max_line: (`None` or `int`) The 1-based line number to end the annotation
      at (exclusive).

  Returns:
    A `dict` mapping 1-based line number to a the namedtuple
      `profiling.LineOrFuncProfileSummary`.
  """

  source_file_path = _norm_abs_path(source_file_path)

  node_name_regex = re.compile(node_name_filter) if node_name_filter else None
  op_type_regex = re.compile(op_type_filter) if op_type_filter else None

  line_to_profile_summary = {}
  for profile_datum in profile_data:
    if not profile_datum.file_path:
      continue

    if _norm_abs_path(profile_datum.file_path) != source_file_path:
      continue

    if (min_line is not None and profile_datum.line_number < min_line or
        max_line is not None and profile_datum.line_number >= max_line):
      continue

    if (node_name_regex and
        not node_name_regex.match(profile_datum.node_exec_stats.node_name)):
      continue

    if op_type_regex and not op_type_regex.match(profile_datum.op_type):
      continue

    if profile_datum.line_number not in line_to_profile_summary:
      line_to_profile_summary[profile_datum.line_number] = (
          profiling.AggregateProfile(profile_datum))
    else:
      line_to_profile_summary[profile_datum.line_number].add(profile_datum)

  return line_to_profile_summary
