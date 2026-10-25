# [MODULE DOC]
# Atheris-based Python API fuzzer for the raw operation Acosh (floating-point dtypes only). Standalone fuzz
# target script (not imported by library code): it defines the fuzz entry function that turns
# raw fuzzer bytes into a random numeric tensor and calls the operation, plus a main function that
# registers the entry function with Atheris (with Python coverage enabled) and starts fuzzing.
# Importing it instruments the imports of sys, the sibling helper module python_fuzzing and
# TensorFlow via Atheris; it has no other import-time side effects.
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
"""This is a Python API fuzzer for tf.raw_ops.Acosh."""
import atheris
with atheris.instrument_imports():
  import sys
  from python_fuzzing import FuzzingHelper
  import tensorflow as tf


def TestOneInput(data):
  # [FUNCTION DOC]
  # Fuzz entry function called by Atheris for each generated input. Input: data, the raw fuzzer
  # bytes. It wraps the bytes in the shared fuzzing helper, picks a random floating-point dtype
  # (half, single or double precision), draws a random tensor of that dtype and passes it to the
  # Acosh raw operation, discarding the result. Returns nothing. Errors are not caught, so any
  # exception raised is reported by the fuzzer as a finding.
  """Test randomized fuzzing input for tf.raw_ops.Acosh."""
  fh = FuzzingHelper(data)

  # tf.raw_ops.Acos takes tf.bfloat16, tf.half, tf.float32, tf.float64,
  # tf.complex64, tf.complex128, but get_random_numeric_tensor only generates
  # tf.float16, tf.float32, tf.float64, tf.int32, tf.int64
  dtype = fh.get_tf_dtype(allowed_set=[tf.float16, tf.float32, tf.float64])
  input_tensor = fh.get_random_numeric_tensor(dtype=dtype)
  _ = tf.raw_ops.Acosh(x=input_tensor)


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
