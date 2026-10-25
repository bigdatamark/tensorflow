# [MODULE DOC]
# Atheris-based Python API fuzzer for the raw operation SparseCountSparseOutput. Standalone fuzz
# target script: the entry function builds random sparse-tensor components (indices, values,
# dense shape, weights) plus fuzzed options and calls the operation, ignoring invalid-argument
# errors. The main function registers the entry function with Atheris and starts fuzzing.
# Importing instruments the imports of sys, the sibling helper module python_fuzzing and TensorFlow.
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
"""This is a Python API fuzzer for tf.raw_ops.SparseCountSparseOutput."""
import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import tensorflow as tf


@atheris.instrument_func
def TestOneInput(input_bytes):
  # [FUNCTION DOC]
  # Fuzz entry function called by Atheris. Input: input_bytes, the raw fuzzer bytes. Draws four
  # small shapes and a seed, builds random int64 tensors for indices, values, dense shape and
  # weights, draws a binary-output flag, min and max lengths and a name, then calls
  # SparseCountSparseOutput, discarding the outputs. Invalid-argument errors are swallowed; others
  # propagate as findings. Returns nothing.
  """Test randomized integer fuzzing input for tf.raw_ops.SparseCountSparseOutput."""
  fh = FuzzingHelper(input_bytes)

  shape1 = fh.get_int_list(min_length=0, max_length=8, min_int=0, max_int=8)
  shape2 = fh.get_int_list(min_length=0, max_length=8, min_int=0, max_int=8)
  shape3 = fh.get_int_list(min_length=0, max_length=8, min_int=0, max_int=8)
  shape4 = fh.get_int_list(min_length=0, max_length=8, min_int=0, max_int=8)

  seed = fh.get_int()
  indices = tf.random.uniform(
      shape=shape1, minval=0, maxval=1000, dtype=tf.int64, seed=seed)
  values = tf.random.uniform(
      shape=shape2, minval=0, maxval=1000, dtype=tf.int64, seed=seed)
  dense_shape = tf.random.uniform(
      shape=shape3, minval=0, maxval=1000, dtype=tf.int64, seed=seed)
  weights = tf.random.uniform(
      shape=shape4, minval=0, maxval=1000, dtype=tf.int64, seed=seed)

  binary_output = fh.get_bool()
  minlength = fh.get_int()
  maxlength = fh.get_int()
  name = fh.get_string()
  try:
    _, _, _, = tf.raw_ops.SparseCountSparseOutput(
        indices=indices,
        values=values,
        dense_shape=dense_shape,
        weights=weights,
        binary_output=binary_output,
        minlength=minlength,
        maxlength=maxlength,
        name=name)
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
