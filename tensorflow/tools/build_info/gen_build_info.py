# [MODULE DOC]
# KNOWN ISSUE: the argument parsing and file generation run at import time (no main guard), so
# importing this module without the required options raises a runtime error.
# Build script helper that generates a small Python source file defining an ordered mapping named
# build_info. Contents come from the optional CUDA and TensorRT configuration modules (only
# present in open-source builds; otherwise treated as absent) plus key=value pairs given on the
# command line. Command line: a required output path option and an optional list of key=value
# pairs. Import-time side effects: parses the command line, writes the output file, or raises a
# runtime error when the output option is missing. Helper module for the pip package build.
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
# ==============================================================================
"""Generates a Python module containing information about the build."""
import argparse

# cuda.cuda is only valid in OSS
try:
  from cuda.cuda import cuda_config  # pylint: disable=g-import-not-at-top
except ImportError:
  cuda_config = None

# tensorrt.tensorrt is only valid in OSS
try:
  from tensorrt.tensorrt import tensorrt_config  # pylint: disable=g-import-not-at-top
except ImportError:
  tensorrt_config = None


def write_build_info(filename, key_value_list):
  # [FUNCTION DOC]
  # HELPER: generates the build information source file for the pip package build.
  # KNOWN ISSUE: the docstring has a typo (a Python what?) and each pair is split on every equals
  # sign, so a value containing an equals sign raises a value error; the output file handle is
  # never explicitly closed; values are formatted with the earlier entries, so literal braces in a
  # value cause errors.
  # Inputs: filename (output path) and key_value_list (list of key=value texts). Starts with the
  # CUDA and TensorRT configuration entries when available, then adds each pair: values equal to
  # true or false (any case) become booleans, other values are text with named placeholders
  # replaced from entries gathered so far. Writes a file with a license header, a docstring and a
  # definition of an ordered mapping holding the entries sorted by key. Overwrites the file.
  # Returns nothing.
  """Writes a Python that describes the build.

  Args:
    filename: filename to write to.
    key_value_list: A list of "key=value" strings that will be added to the
      module's "build_info" dictionary as additional entries.
  """

  build_info = {}

  if cuda_config:
    build_info.update(cuda_config.config)

  if tensorrt_config:
    build_info.update(tensorrt_config.config)

  for arg in key_value_list:
    key, value = arg.split("=")
    if value.lower() == "true":
      build_info[key] = True
    elif value.lower() == "false":
      build_info[key] = False
    else:
      build_info[key] = value.format(**build_info)

  # Sort the build info to ensure deterministic output.
  sorted_build_info_pairs = sorted(build_info.items())

  contents = """
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
\"\"\"Auto-generated module providing information about the build.\"\"\"
import collections

build_info = collections.OrderedDict(%s)
""" % sorted_build_info_pairs
  open(filename, "w").write(contents)


parser = argparse.ArgumentParser(
    description="""Build info injection into the PIP package.""")

parser.add_argument("--raw_generate", type=str, help="Generate build_info.py")

parser.add_argument(
    "--key_value", type=str, nargs="*", help="List of key=value pairs.")

args = parser.parse_args()

if args.raw_generate:
  write_build_info(args.raw_generate, args.key_value)
else:
  raise RuntimeError("--raw_generate must be used.")
