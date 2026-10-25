# [MODULE DOC]
# HELPER: CI build helper script that prints the full TensorFlow wheel version string.
# Command line: required wheel type (nightly or release) and required wheel version (arguments
# may also be read from a file via an at-prefixed argument). Prints the version wrapped in double
# quotes; for nightly wheels a development suffix with today's date (year, month, day) is
# appended, for release wheels the version is printed unchanged. Does its work only when run as a
# script; importing has no side effects.
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Calculates full TensorFlow version."""

import argparse
import time


def parse_args():
  # [FUNCTION DOC]
  # HELPER: command line parsing for the version calculation script.
  # Defines the required wheel type option (nightly or release) and the required wheel version
  # option, enables reading arguments from a file, parses the process arguments and returns the
  # namespace of parsed values. Exits the process with a usage error on invalid input.
  """Arguments parser."""
  parser = argparse.ArgumentParser(
      description="Helper for TensorFlow version calculation",
      fromfile_prefix_chars="@",
  )
  parser.add_argument(
      "--wheel-type",
      required=True,
      choices=["nightly", "release"],
      help="Type of the wheel",
  )
  parser.add_argument("--wheel-version", required=True, help="Wheel version")
  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()
  tensorflow_version_suffix = ""
  if args.wheel_type == "nightly":
    tensorflow_version_suffix = "-dev{}".format(time.strftime("%Y%m%d"))

  print(
      '"{wheel_version}{tensorflow_version_suffix}"'.format(
          wheel_version=args.wheel_version,
          tensorflow_version_suffix=tensorflow_version_suffix,
      )
  )
