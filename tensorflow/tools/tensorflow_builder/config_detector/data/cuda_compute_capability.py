# [MODULE DOC]
# HELPER: maintenance script supporting the config detector (it refreshes the golden data file).
# Scrapes the NVIDIA CUDA GPU web page for GPU names and their compute capability numbers, builds
# an ordered mapping from GPU name to a list of capabilities, writes it as a csv file in the data
# directory and compares it with the checked-in golden csv, printing a warning on differences.
# Also offers a loader for the golden file and a table printer. Run as a script through absl.
# Import-time side effects: none beyond defining path constants and the flags object.
# KNOWN ISSUE: the web retrieval uses the aliased urllib.request module through a double
# "request" attribute lookup and compares text with bytes, so it likely fails on Python 3.
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
"""Retrieves CUDA compute capability from NVIDIA webpage and creates a `.csv`.

This module is mainly written to supplement for `../config_detector.py`
which retrieves CUDA compute capability from existing golden file.

The golden file resides inside `./golden` directory.

Usage:
  python cuda_compute_capability.py

Output:
  Creates `compute_capability.csv` file in the same directory by default. If
  the file already exists, then it overwrites the file.

  In order to use the new `.csv` as the golden, then it should replace the
  original golden file (`./golden/compute_capability_golden.csv`) with the
  same file name and path.
"""
import collections
import difflib
import os
import re
import urllib.request as urllib

from absl import app
from absl import flags


FLAGS = flags.FLAGS
PATH_TO_DIR = "tensorflow/tools/tensorflow_builder/config_detector"
CUDA_CC_GOLDEN_DIR = PATH_TO_DIR + "/data/golden/compute_capability_golden.csv"


def retrieve_from_web(generate_csv=False):
  # [FUNCTION DOC]
  # Downloads the NVIDIA CUDA GPUs page and extracts GPU names and capabilities line by line
  # using patterns, stopping at the closing html tag. Input: generate_csv, optional boolean
  # (default false) passed on to request csv output. Returns the ordered mapping built by
  # create_gpu_capa_map (GPU name to list of capability strings in page order). Side effects:
  # network access; optionally writes a csv file.
  # KNOWN ISSUE: the module is imported as an alias for the request module, so the call that
  # accesses its "request" attribute fails with an attribute error; and lines read from the
  # response are bytes, so the text pattern searches raise a type error on Python 3.
  """Retrieves list of all CUDA compute capability from NVIDIA webpage.

  Args:
    generate_csv: Boolean for generating an output file containing
                  the results.

  Returns:
    OrderedDict that is a list of all CUDA compute capability listed on the
    NVIDIA page. Order goes from top to bottom of the webpage content (.html).
  """
  url = "https://developer.nvidia.com/cuda-gpus"
  source = urllib.request.urlopen(url)
  matches = []
  while True:
    line = source.readline()
    if "</html>" in line:
      break
    else:
      gpu = re.search(r"<a href=.*>([\w\S\s\d\[\]\,]+[^*])</a>(<a href=.*)?.*",
                      line)
      capability = re.search(
          r"([\d]+).([\d]+)(/)?([\d]+)?(.)?([\d]+)?.*</td>.*", line)
      if gpu:
        matches.append(gpu.group(1))
      elif capability:
        if capability.group(3):
          capability_str = capability.group(4) + "." + capability.group(6)
        else:
          capability_str = capability.group(1) + "." + capability.group(2)
        matches.append(capability_str)

  return create_gpu_capa_map(matches, generate_csv)


def retrieve_from_golden():
  # [FUNCTION DOC]
  # Loads the golden csv file (path constant for the golden compute capability file). Takes no
  # inputs. Each line is split on commas: first item is the GPU name, the rest (newline stripped)
  # are its capability strings. Returns a plain mapping from GPU name to list of capabilities.
  # Raises an OS error if the file is missing. Only reads the file.
  """Retrieves list of all CUDA compute capability from a golden file.

  The following file is set as default:
    `./golden/compute_capability_golden.csv`

  Returns:
    Dictionary that lists of all CUDA compute capability in the following
    format:
      {'<GPU name>': ['<version major>.<version minor>', ...], ...}

    If there are multiple versions available for a given GPU, then it
    appends all supported versions in the value list (in the key-value
    pair.)
  """
  out_dict = dict()
  with open(CUDA_CC_GOLDEN_DIR) as g_file:
    for line in g_file:
      line_items = line.split(",")
      val_list = []
      for item in line_items[1:]:
        val_list.append(item.strip("\n"))
      out_dict[line_items[0]] = val_list

  return out_dict


def create_gpu_capa_map(match_list,
                        generate_csv=False,
                        filename="compute_capability"):
  # [FUNCTION DOC]
  # HELPER: post-processing step of retrieve_from_web.
  # Inputs: match_list (alternating GPU name and capability strings scraped from the page, with
  # marker entries), generate_csv (optional boolean, default false) and filename (csv base name
  # without extension, default "compute_capability"). Starts collecting after an entry containing
  # "Products" and stops at one containing "www"; pairs a name with the following capability and
  # appends repeated names' capabilities to the same list. Returns an ordered mapping from GPU
  # name to list of capabilities. If requested, writes the csv (which also compares to golden).
  # KNOWN ISSUE: the counters for mismatches are computed but never used or reported.
  """Generates a map between GPU types and corresponding compute capability.

  This method is used for retrieving CUDA compute capability from the web only.

  Args:
    match_list: List of all CUDA compute capability detected from the webpage.
    generate_csv: Boolean for creating csv file to store results.
    filename: String that is the name of the csv file (without `.csv` ending).

  Returns:
    OrderedDict that lists in the incoming order of all CUDA compute capability
    provided as `match_list`.
  """
  gpu_capa = collections.OrderedDict()
  include = False
  gpu = ""
  cnt = 0
  mismatch_cnt = 0
  for match in match_list:
    if "Products" in match:
      if not include:
        include = True

      continue
    elif "www" in match:
      include = False
      break

    if include:
      if gpu:
        if gpu in gpu_capa:
          gpu_capa[gpu].append(match)
        else:
          gpu_capa[gpu] = [match]

        gpu = ""
        cnt += 1
        if len(list(gpu_capa.keys())) < cnt:
          mismatch_cnt += 1
          cnt = len(list(gpu_capa.keys()))

      else:
        gpu = match

  if generate_csv:
    f_name = filename + ".csv"
    write_csv_from_dict(f_name, gpu_capa)

  return gpu_capa


def write_csv_from_dict(filename, input_dict):
  # [FUNCTION DOC]
  # Writes the mapping as a csv file named filename inside the data directory: one line per GPU
  # with the name followed by its comma-separated capabilities. Inputs: filename (output file
  # name) and input_dict (mapping from name to list of strings). Flushes, prints a confirmation
  # and then runs the golden comparison. Returns nothing. Overwrites an existing file.
  # KNOWN ISSUE: the file is flushed but never closed explicitly.
  """Writes out a `.csv` file from an input dictionary.

  After writing out the file, it checks the new list against the golden
  to make sure golden file is up-to-date.

  Args:
    filename: String that is the output file name.
    input_dict: Dictionary that is to be written out to a `.csv` file.
  """
  f = open(PATH_TO_DIR + "/data/" + filename, "w")
  for k, v in input_dict.items():
    line = k
    for item in v:
      line += "," + item

    f.write(line + "\n")

  f.flush()
  print("Wrote to file %s" % filename)
  check_with_golden(filename)


def check_with_golden(filename):
  # [FUNCTION DOC]
  # Compares a newly written csv in the data directory with the golden csv. Input: filename of
  # the new file. If both files exist, computes a unified diff and prints either a warning with
  # the list of diff lines or a message that no difference was found. Does nothing if either file
  # is missing. Returns nothing. Only reads files and prints.
  # KNOWN ISSUE: the "no difference" message misspells golden as "golen".
  """Checks the newly created CUDA compute capability file with the golden.

  If differences are found, then it prints a list of all mismatches as
  a `WARNING`.

  Golden file must reside in `golden/` directory.

  Args:
    filename: String that is the name of the newly created file.
  """
  path_to_file = PATH_TO_DIR + "/data/" + filename
  if os.path.isfile(path_to_file) and os.path.isfile(CUDA_CC_GOLDEN_DIR):
    with open(path_to_file, "r") as f_new:
      with open(CUDA_CC_GOLDEN_DIR, "r") as f_golden:
        diff = difflib.unified_diff(
            f_new.readlines(),
            f_golden.readlines(),
            fromfile=path_to_file,
            tofile=CUDA_CC_GOLDEN_DIR
        )
        diff_list = []
        for line in diff:
          diff_list.append(line)

        if diff_list:
          print("WARNING: difference(s) found between new csv and golden csv.")
          print(diff_list)
        else:
          print("No difference found between new csv and golen csv.")


def print_dict(py_dict):
  # [FUNCTION DOC]
  # HELPER: debugging aid. Input: py_dict, mapping from GPU name to capability value. Prints one
  # line per entry as a two column table, each column padded to 25 characters. Returns nothing.
  """Prints dictionary with formatting (2 column table).

  Args:
    py_dict: Dictionary that is to be printed out in a table format.
  """
  for gpu, cc in py_dict.items():
    print("{:<25}{:<25}".format(gpu, cc))


def main(argv):
  # [FUNCTION DOC]
  # Script entry point. Input: argv, the command-line argument list from the absl runner.
  # Raises a usage error if more than two arguments are given; otherwise scrapes the NVIDIA page
  # and writes the csv file (default name compute_capability) in the data directory, then
  # compares with the golden file. Returns nothing.
  if len(argv) > 2:
    raise app.UsageError("Too many command-line arguments.")

  retrieve_from_web(generate_csv=True)


if __name__ == "__main__":
  app.run(main)
