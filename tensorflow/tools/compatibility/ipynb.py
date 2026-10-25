# [MODULE DOC]
# HELPER: helper module for the TensorFlow upgrade tool that adds Jupyter notebook support.
# Extracts the Python code lines from the code cells of a notebook file, encodes notebook-only
# constructs (magic lines starting with percent, bang or question mark, backslash-continued
# lines, trailing newlines) with marker strings so the text passes through the Python source
# upgrader, then writes the upgraded lines back into a copy of the notebook structure and saves it.
# Exposes a named tuple of cell number plus code line, and functions to test cells, process a
# file, skip magic lines and detect split lines. No import-time side effects.
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
# =============================================================================
"""A module to support operations on ipynb files"""

import collections
import copy
import json
import re
import shutil
import tempfile

CodeLine = collections.namedtuple("CodeLine", ["cell_number", "code"])

def is_python(cell):
  # [FUNCTION DOC]
  # HELPER: used by notebook code extraction and notebook update to select cells to upgrade.
  # Decides whether a notebook cell holds ordinary Python code. Input: a cell as a mapping with a
  # cell type and a source (list of lines). True-like when the type is code, the source is not empty,
  # and the first line does not begin with two percent signs (a cell magic such as a bash cell).
  # KNOWN ISSUE: it returns the value of the last evaluated operand, not strictly a boolean (for
  # example an empty source gives an empty list). Raises a missing-key error if keys are absent.
  """Checks if the cell consists of Python code."""
  return (cell["cell_type"] == "code"  # code cells only
          and cell["source"]  # non-empty cells
          and not cell["source"][0].startswith("%%"))  # multiline eg: %%bash


def process_file(in_filename, out_filename, upgrader):
  # [FUNCTION DOC]
  # Upgrades a notebook file. Inputs: in_filename (path of the input notebook), out_filename (path
  # of the output notebook), upgrader (an upgrader object offering a string-update method and a log
  # formatting method). Steps: prints a progress message, extracts code lines, joins them with
  # newlines and runs the upgrader on the text, converts the result back into a notebook written as
  # JSON to a temporary file, then moves the temporary file to the output path (overwriting it).
  # Returns a triple: whether the file was processed, the formatted report text, and the list of
  # errors. Raises a syntax error with the log text if the upgrader could not process the text.
  # Side effects: console output, a temporary file that is created and (on failure) left behind,
  # and a file written at the output path.
  # KNOWN ISSUE: the docstring is vague, a comment mentions a misspelled upgrader method, and the
  # temporary file is not removed when the syntax error is raised.
  """The function where we inject the support for ipynb upgrade."""
  print("Extracting code lines from original notebook")
  raw_code, notebook = _get_code(in_filename)
  raw_lines = [cl.code for cl in raw_code]

  # The function follows the original flow from `upgrader.process_fil`
  with tempfile.NamedTemporaryFile("w", delete=False) as temp_file:

    processed_file, new_file_content, log, process_errors = (
        upgrader.update_string_pasta("\n".join(raw_lines), in_filename))

    if temp_file and processed_file:
      new_notebook = _update_notebook(notebook, raw_code,
                                      new_file_content.split("\n"))
      json.dump(new_notebook, temp_file)
    else:
      raise SyntaxError(
          "Was not able to process the file: \n%s\n" % "".join(log))

    files_processed = processed_file
    report_text = upgrader._format_log(log, in_filename, out_filename)
    errors = process_errors

  shutil.move(temp_file.name, out_filename)

  return files_processed, report_text, errors


def skip_magic(code_line, magic_list):
  # [FUNCTION DOC]
  # HELPER: used by code extraction to hide notebook-only lines from the Python upgrader.
  # Tells whether a code line starts with any of the given magic prefixes. Inputs: code_line (a
  # string) and magic_list (a list of prefix strings). Returns true if the line starts with any
  # prefix, false otherwise. No side effects.
  # KNOWN ISSUE: the docstring is loose (the arguments are called a line of code and exceptions),
  # the example is not indented like the rest and the "returns" text is ungrammatical.
  """Checks if the cell has magic, that is not Python-based.

  Args:
      code_line: A line of Python code
      magic_list: A list of jupyter "magic" exceptions

  Returns:
    If the line jupyter "magic" line, not Python line

   >>> skip_magic('!ls -laF', ['%', '!', '?'])
  True
  """

  for magic in magic_list:
    if code_line.startswith(magic):
      return True

  return False


def check_line_split(code_line):
  # [FUNCTION DOC]
  # HELPER: used by code extraction to carry the magic marker over backslash-continued lines.
  # Checks whether a line ends with a backslash followed by optional whitespace and a newline.
  # Input: code_line (string). Returns a match object when it does and nothing (null) when it does
  # not. No side effects.
  # KNOWN ISSUE: it returns a match object or null, not the boolean the docstring suggests; the
  # docstring example calls the skip-magic function with a single argument instead of this function.
  r"""Checks if a line was split with `\`.

  Args:
      code_line: A line of Python code

  Returns:
    If the line was split with `\`

  >>> skip_magic("!gcloud ml-engine models create ${MODEL} \\\n")
  True
  """

  return re.search(r"\\\s*\n$", code_line)


def _get_code(input_file):
  # [FUNCTION DOC]
  # HELPER: internal step of notebook processing that converts a notebook into plain code lines.
  # Reads the notebook JSON file and walks its cells. For each Python cell it takes each source line
  # and: prefixes lines starting with a magic character (or following a backslash-continued magic
  # line) with the marker for hidden lines; replaces a trailing newline on the last line of the cell
  # and any embedded newline with the newline marker; strips trailing whitespace; and records the
  # line with the index of the Python cell. Non-Python cells do not advance the cell index.
  # Input: input_file (path of an existing notebook file). Returns a pair: the list of code line
  # records (cell number and code) and the full notebook mapping. Raises file or JSON errors from
  # opening and parsing. Reads a file, no other side effects.
  # KNOWN ISSUE: the docstring says it returns a list of code lines but it returns a pair of that
  # list and the notebook; the magic-continuation logic checks the split state twice in a row.
  """Loads the ipynb file and returns a list of CodeLines."""

  raw_code = []

  with open(input_file) as in_file:
    notebook = json.load(in_file)

  cell_index = 0
  for cell in notebook["cells"]:
    if is_python(cell):
      cell_lines = cell["source"]

      is_line_split = False
      for line_idx, code_line in enumerate(cell_lines):

        # Sometimes, jupyter has more than python code
        # Idea is to comment these lines, for upgrade time
        if skip_magic(code_line, ["%", "!", "?"]) or is_line_split:
          # Found a special character, need to "encode"
          code_line = "###!!!" + code_line

          # if this cell ends with `\` -> skip the next line
          is_line_split = check_line_split(code_line)

        if is_line_split:
          is_line_split = check_line_split(code_line)

        # Sometimes, people leave \n at the end of cell
        # in order to migrate only related things, and make the diff
        # the smallest -> here is another hack
        if (line_idx == len(cell_lines) - 1) and code_line.endswith("\n"):
          code_line = code_line.replace("\n", "###===")

        # sometimes a line would start with `\n` and content after
        # that's the hack for this
        raw_code.append(
            CodeLine(cell_index,
                     code_line.rstrip().replace("\n", "###===")))

      cell_index += 1

  return raw_code, notebook


def _update_notebook(original_notebook, original_raw_lines, updated_code_lines):
  # [FUNCTION DOC]
  # HELPER: internal step of notebook processing that writes upgraded lines back into a notebook.
  # Makes a deep copy of the original notebook, then for each Python cell collects the upgraded
  # lines belonging to that cell (matched using the cell numbers in the raw line records), joins
  # them with newlines, removes the hidden-line marker, turns the newline marker back into a real
  # newline and stores the result as the cell's source.
  # Inputs: original_notebook (mapping), original_raw_lines (list of code line records) and
  # updated_code_lines (list of strings, same length). Returns the new notebook; the original is not
  # modified. Raises an assertion error if the line counts differ.
  # KNOWN ISSUE: the source of each cell becomes one string rather than a list of lines, and the
  # length check relies on an assert statement that is skipped under optimized runs.
  """Updates notebook, once migration is done."""

  new_notebook = copy.deepcopy(original_notebook)

  # validate that the number of lines is the same
  assert len(original_raw_lines) == len(updated_code_lines), \
    ("The lengths of input and converted files are not the same: "
     "{} vs {}".format(len(original_raw_lines), len(updated_code_lines)))

  code_cell_idx = 0
  for cell in new_notebook["cells"]:
    if not is_python(cell):
      continue

    applicable_lines = [
        idx for idx, code_line in enumerate(original_raw_lines)
        if code_line.cell_number == code_cell_idx
    ]

    new_code = [updated_code_lines[idx] for idx in applicable_lines]

    cell["source"] = "\n".join(new_code).replace("###!!!", "").replace(
        "###===", "\n")
    code_cell_idx += 1

  return new_notebook
