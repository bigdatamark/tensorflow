# [MODULE DOC]
# HELPER: generator script producing test data for runtime saved model tests (not a test itself).
# Saves a version-2 module whose serving function uses a data pipeline (tensor dataset mapped by
# adding one, then extracting the single element). Defines a flag for the output path.
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
"""Generates a saved model that contains tf.data ops for testing."""

from absl import app
from absl import flags
from absl import logging
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options

flags.DEFINE_string('saved_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS


class ToyModule(module.Module):
  # [CLASS DOC]
  # HELPER: toy module used only by the generator. Stateless; exposes one traced method that
  # exercises data pipeline ops.
  """Defines a toy module."""

  @def_function.function(
      input_signature=[tensor_spec.TensorSpec([], dtypes.int32, name='input')])
  def map_data(self, x):
    # [FUNCTION DOC]
    # Traced method taking a scalar 32-bit integer. Builds a dataset from the single value, maps it
    # with a function adding one, and returns the single element, so the result is x plus 1.
    dataset = dataset_ops.Dataset.from_tensors(x)
    r = dataset.map(lambda x: x + 1)
    return r.get_single_element()


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
