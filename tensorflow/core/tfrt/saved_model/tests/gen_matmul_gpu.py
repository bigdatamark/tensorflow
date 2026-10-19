# [MODULE DOC]
# HELPER: generator script producing test data for runtime saved model tests (not a test itself).
# Saves a version-2 module with one variable and a function that multiplies the input by it on the
# first GPU device. Defines a flag for the output path.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Generates a toy saved model with matmul op on GPU for testing."""

from absl import app
from absl import flags
from absl import logging

from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options

flags.DEFINE_string('saved_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS


class ToyModule(module.Module):
  # [CLASS DOC]
  # HELPER: toy module used only by the generator. Holds one 3 by 1 float variable w and a traced
  # method that multiplies the input by it on a GPU.
  """Defines a toy module."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor: calls the base constructor and creates variable w with column values 1, 2, 3.
    super(ToyModule, self).__init__()
    self.w = variables.Variable(
        constant_op.constant([[1.0], [2.0], [3.0]]), name='w')

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec([1, 3], dtypes.float32, name='input')
  ])
  def toy(self, x):
    # [FUNCTION DOC]
    # Traced method taking a 1 by 3 float input. Places a matrix multiplication of the input and
    # variable w on the first GPU device and returns the 1 by 1 result.
    with ops.device('/gpu:0'):
      r = math_ops.matmul(x, self.w, name='result')
    return r


def main(argv):
  # [FUNCTION DOC]
  # HELPER: entry point of the generator script.
  # Input: argv, list of command line arguments; raises a usage error if more than the program name.
  # Enables v2 behavior, saves a new toy module to the path in the flag without debug info, and
  # logs the path. Returns nothing. Side effects: global behavior change and filesystem writes.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  v2_compat.enable_v2_behavior()

  save.save(
      ToyModule(),
      FLAGS.saved_model_path,
      options=save_options.SaveOptions(save_debug_info=False))
  logging.info('Saved model to: %s', FLAGS.saved_model_path)


if __name__ == '__main__':
  app.run(main)
