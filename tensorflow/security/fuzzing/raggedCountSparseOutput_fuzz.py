# [MODULE DOC]
# Atheris-based Python API fuzzer for the raw operation RaggedCountSparseOutput. Standalone fuzz
# target script: the entry function builds fuzzed row splits, values and weights and calls the
# operation, ignoring invalid-argument errors. The main function registers the entry function
# with Atheris and starts fuzzing. Importing instruments the imports of sys, the sibling helper
# module python_fuzzing and TensorFlow.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""This is a Python API fuzzer for tf.raw_ops.RaggedCountSparseOutput."""
import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import tensorflow as tf


@atheris.instrument_func
def TestOneInput(input_bytes):
  # [FUNCTION DOC]
  # Fuzz entry function called by Atheris. Input: input_bytes, the raw fuzzer bytes. Draws a list of
  # row splits, a list of values (all integers or all floats, chosen randomly) and a list of
  # weights, then calls RaggedCountSparseOutput with non-binary output, discarding the three
  # outputs. Invalid-argument errors are swallowed; others propagate as findings. Returns nothing.
  """Test randomized integer/float fuzzing input for tf.raw_ops.RaggedCountSparseOutput."""
  fh = FuzzingHelper(input_bytes)

  splits = fh.get_int_list()
  values = fh.get_int_or_float_list()
  weights = fh.get_int_list()
  try:
    _, _, _, = tf.raw_ops.RaggedCountSparseOutput(
        splits=splits, values=values, weights=weights, binary_output=False)
  except tf.errors.InvalidArgumentError:
    pass


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
