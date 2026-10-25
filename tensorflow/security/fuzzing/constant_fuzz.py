# [MODULE DOC]
# Atheris-based Python API fuzzer for the constant-tensor creation function. Standalone fuzz
# target script: the entry function passes the raw fuzzer bytes directly to the function, and the
# main function registers it with Atheris (Python coverage enabled) and starts fuzzing.
# Importing instruments the imports of sys and TensorFlow via Atheris.
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
"""This is a Python API fuzzer for tf.constant."""
import atheris
with atheris.instrument_imports():
  import sys
  import tensorflow as tf


def TestOneInput(data):
  # [FUNCTION DOC]
  # Fuzz entry function called by Atheris. Input: data, the raw fuzzer bytes, used directly as the
  # value to convert into a constant tensor. Returns nothing and discards the tensor. Errors are
  # not caught, so any exception is reported by the fuzzer as a finding.
  tf.constant(data)


def main():
  # [FUNCTION DOC]
  # Fuzz program entry point: registers the fuzz entry function with Atheris, with Python coverage
  # collection enabled and the process command-line arguments passed through, then starts the
  # fuzzing loop. Takes no inputs and returns nothing; the fuzzing loop normally runs until stopped
  # or a crash is found, and the process exit is controlled by Atheris.
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == "__main__":
  main()
