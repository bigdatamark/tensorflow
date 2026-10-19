# [MODULE DOC]
# HELPER: build-time generator script for the gradient exclusions C++ source file.
# Command-line script that writes the generated contents (from the gradient input/output
# exclusions module) to the output file path given as its single positional argument. Run as a
# main program; importing it has no side effects.
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
r"""Script to generate inputs/outputs exclusion lists for GradientTape.

To use this script:

bazel run tensorflow/python/eager:gen_gradient_input_output_exclusions -- \
  $PWD/tensorflow/python/eager/pywrap_gradient_exclusions.cc
"""

import argparse

from tensorflow.python.eager import gradient_input_output_exclusions


def main(output_file):
  # [FUNCTION DOC]
  # HELPER: entry point of the generator script.
  # Opens the file at the given path for writing (overwriting it) and writes the generated
  # exclusion-list source text into it. Returns nothing; file errors propagate.
  with open(output_file, "w") as fp:
    fp.write(gradient_input_output_exclusions.get_contents())


if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("output", metavar="O", type=str, help="Output file.")
  args = arg_parser.parse_args()
  main(args.output)
