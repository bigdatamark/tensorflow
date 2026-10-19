# [MODULE DOC]
# HELPER: generator script producing test data (a saved model with checkpoint) for a kernel test.
# Defines a flag for the output path, a toy module with four integer vector variables whose
# serving function adds them all to its input, and a main function that saves the module.
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Create a saved model with checkpoint for ifrt_ops_kernel_test."""

from absl import app
from absl import flags
from absl import logging
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor
from tensorflow.python.module import module
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import saved_model

_SAVED_MODEL_PATH = flags.DEFINE_string(
    'saved_model_path', '', 'Path to save the model to.'
)


class ToyModule(module.Module):
  # [CLASS DOC]
  # HELPER: toy module used only by the generator. Holds four integer vector variables of length
  # three (w, w1, w2, w3) and a serving function that adds all of them to its input.
  """A toy module for testing checkpoing loading."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor: calls the base constructor and creates the four variables with fixed initial
    # values (1-3, 4-6, 7-9, 10-12) as attributes. Takes no arguments.
    super().__init__()
    self.w = variables.Variable(constant_op.constant([1, 2, 3]), name='w')
    self.w1 = variables.Variable(constant_op.constant([4, 5, 6]), name='w1')
    self.w2 = variables.Variable(constant_op.constant([7, 8, 9]), name='w2')
    self.w3 = variables.Variable(constant_op.constant([10, 11, 12]), name='w3')

  @polymorphic_function.function(
      input_signature=[tensor.TensorSpec([None, 3], dtypes.int32, name='input')]
  )
  def serving_default(self, x):
    # [FUNCTION DOC]
    # Traced serving function with input signature of an integer matrix with any number of rows and
    # three columns. Returns x plus each of the four variables in turn (broadcast addition).
    dummy = x + self.w
    dummy = dummy + self.w1
    dummy = dummy + self.w2
    dummy = dummy + self.w3
    return dummy


def main(argv):
  # [FUNCTION DOC]
  # HELPER: entry point of the generator script.
  # Input: argv, list of command line arguments; raises a usage error if more than the program name.
  # Enables v2 behavior, creates the toy module and saves it to the path given by the flag, with
  # debug info disabled and the serving function exposed as the serving_default signature. Logs the
  # path. Returns nothing. Side effects: global behavior change and writes files.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  v2_compat.enable_v2_behavior()

  model = ToyModule()
  saved_model.save(
      model,
      _SAVED_MODEL_PATH.value,
      options=save_options.SaveOptions(save_debug_info=False),
      signatures={
          'serving_default': model.serving_default,
      },
  )
  logging.info('Saved model to: %s', _SAVED_MODEL_PATH.value)


if __name__ == '__main__':
  app.run(main)
