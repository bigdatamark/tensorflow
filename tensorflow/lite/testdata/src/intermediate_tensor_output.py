# [MODULE DOC]
# Standalone script that generates a TFLite test model in which an intermediate tensor is also a
# model output. It defines a traced function with two float inputs of fixed shape and converts it
# with the TFLite converter, writing the model to a fixed path in the temporary directory.
# Not a library; run as a script. No import-time side effects besides tracing setup.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""A model whose intermediate tensor is also used as a model output."""

import tensorflow as tf
from tensorflow.lite.python import lite


@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[1, 4, 4, 4], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 4, 4, 4], dtype=tf.float32),
    ]
)
def func(a, b):
  # [FUNCTION DOC]
  # Traced function with two float32 inputs of shape 1x4x4x4. Computes the sum of a and b (c),
  # then adds a three more times (d, e, f). Returns the intermediate c and the final f, so c is
  # both consumed internally and exposed as an output.
  c = a + b
  d = c + a
  e = d + a
  f = e + a
  return c, f


def main():
  # [FUNCTION DOC]
  # Converts the traced function to a TFLite model with a default target spec and writes the
  # bytes to a fixed file in the temporary directory, then prints a message with that path.
  # Takes no arguments, returns nothing; side effect is the file write.
  converter = lite.TFLiteConverter.from_concrete_functions(
      [func.get_concrete_function()]
  )
  converter.target_spec = lite.TargetSpec()
  tflite_model = converter.convert()
  model_path = '/tmp/intermediate_tensor_output.tflite'
  with open(model_path, 'wb') as f:
    f.write(tflite_model)
  print(f'TFLite model {model_path} is generated.\n')


if __name__ == '__main__':
  main()
