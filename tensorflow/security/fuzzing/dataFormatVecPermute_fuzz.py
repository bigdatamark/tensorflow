# [MODULE DOC]
# Atheris-based Python API fuzzer for the raw operation DataFormatVecPermute. Standalone fuzz
# target script: the entry function builds a random tensor and random source/destination format
# strings from the fuzzer bytes and calls the operation, tolerating expected argument errors.
# The main function registers the entry function with Atheris and starts fuzzing. Importing
# instruments the imports of sys, the sibling helper module python_fuzzing and TensorFlow.
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
"""This is a Python API fuzzer for tf.raw_ops.DataFormatVecPermute."""
import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import tensorflow as tf


@atheris.instrument_func
def TestOneInput(input_bytes):
  # [FUNCTION DOC]
  # Fuzz entry function called by Atheris. Input: input_bytes, the raw fuzzer bytes. Chooses a
  # random dtype and a short random shape (up to eight dimensions each of size up to eight), builds
  # a random tensor from a fuzzed seed, then calls DataFormatVecPermute with fuzzed digit strings
  # as source and destination formats and a fuzzed op name. Invalid-argument, value and type errors
  # are swallowed; others propagate as findings. Returns nothing.
  """Test randomized integer fuzzing input for tf.raw_ops.DataFormatVecPermute."""
  fh = FuzzingHelper(input_bytes)

  dtype = fh.get_tf_dtype()
  # Max shape can be 8 in length and randomized from 0-8 without running into
  # a OOM error.
  shape = fh.get_int_list(min_length=0, max_length=8, min_int=0, max_int=8)
  seed = fh.get_int()
  try:
    x = tf.random.uniform(shape=shape, dtype=dtype, seed=seed)
    src_format_digits = str(fh.get_int(min_int=0, max_int=999999999))
    dest_format_digits = str(fh.get_int(min_int=0, max_int=999999999))
    _ = tf.raw_ops.DataFormatVecPermute(
        x,
        src_format=src_format_digits,
        dst_format=dest_format_digits,
        name=fh.get_string())
  except (tf.errors.InvalidArgumentError, ValueError, TypeError):
    pass


def main():
  # [FUNCTION DOC]
  # Fuzz program entry point: registers the fuzz entry function with Atheris, with Python coverage
  # collection enabled and the process command-line arguments passed through, then starts the
  # fuzzing loop. Takes no inputs and returns nothing; the fuzzing loop normally runs until stopped
  # or a crash is found, and the process exit is controlled by Atheris.
  atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
  atheris.Fuzz()


if __name__ == '__main__':
  main()
