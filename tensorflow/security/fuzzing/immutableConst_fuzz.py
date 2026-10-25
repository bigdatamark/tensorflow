# [MODULE DOC]
# Atheris-based Python API fuzzer for the raw operation ImmutableConst. Standalone fuzz target
# script: the entry function writes a fuzzer-generated string to a fixed temporary file path
# and asks the operation to map that file as an immutable tensor with a fuzzed type and shape.
# Side effect: creates/overwrites a file in the temporary directory on every run. The main
# function registers the entry function with Atheris and starts fuzzing. Importing instruments
# the imports of sys, the sibling helper module python_fuzzing and TensorFlow.
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
"""This is a Python API fuzzer for tf.raw_ops.ImmutableConst."""
import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import tensorflow as tf

_DEFAULT_FILENAME = '/tmp/test.txt'


@atheris.instrument_func
def TestOneInput(input_bytes):
  # [FUNCTION DOC]
  # Fuzz entry function called by Atheris. Input: input_bytes, the raw fuzzer bytes. Draws a random
  # dtype, shape and file contents; writes the string to the fixed temporary file (side effect:
  # the file is created or overwritten each run) and calls ImmutableConst with that file name as
  # the memory region. Invalid-argument, internal and unicode encode/decode errors are swallowed;
  # others propagate as findings. Returns nothing.
  """Test randomized integer fuzzing input for tf.raw_ops.ImmutableConst."""
  fh = FuzzingHelper(input_bytes)

  dtype = fh.get_tf_dtype()
  shape = fh.get_int_list()
  try:
    with open(_DEFAULT_FILENAME, 'w') as f:
      f.write(fh.get_string())
    _ = tf.raw_ops.ImmutableConst(
        dtype=dtype, shape=shape, memory_region_name=_DEFAULT_FILENAME)
  except (tf.errors.InvalidArgumentError, tf.errors.InternalError,
          UnicodeEncodeError, UnicodeDecodeError):
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
