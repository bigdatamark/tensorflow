# [MODULE DOC]
# HELPER: command-line generator of a test fixture (a toy v1 saved model) for TFRT saved model tests.
# Script that builds a small graph-mode (v1) saved model containing a while loop. The graph has an
# integer scalar placeholder named input; a loop increments the value by one until it reaches ten.
# The model is exported with a serving tag and two signatures (the name serving_default and the
# default serving signature key), both mapping input x to output r with the predict method name.
# Defines one command-line flag, saved_model_path (string, default empty), where the model is
# written. Running the script as main executes the generator; importing only defines the flag.
# Side effects: the main function first recursively deletes the directory at saved_model_path
# (it fails if the path does not exist), then writes the saved model there.

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
"""Generates a toy v1 saved model for testing."""

import shutil
from absl import app
from absl import flags
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

flags.DEFINE_string('saved_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS


def main(argv):
  # [FUNCTION DOC]
  # HELPER: entry point of the saved model generator script.
  # KNOWN ISSUE: it deletes the output directory first, which raises a file-not-found error if the
  # directory does not already exist, so the output path must be pre-created.
  # Takes the command-line argument list; raises a usage error if more than the program name is
  # given. Removes the directory named by the saved_model_path flag, builds a while-loop graph
  # (placeholder input, loop adding one until value reaches ten), starts a session, and saves a
  # v1 saved model with the serving tag and two signature entries (serving_default and the default
  # serving signature key) mapping x to r, stripping default attributes. Returns nothing.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  shutil.rmtree(FLAGS.saved_model_path)

  # Create the graph
  ten = constant_op.constant(10)
  one = constant_op.constant(1)
  x = array_ops.placeholder(dtypes.int32, shape=(), name='input')
  r = while_loop.while_loop(lambda a: a < ten, lambda a: math_ops.add(a, one),
                            [x])

  sess = session.Session()

  sm_builder = builder.SavedModelBuilder(FLAGS.saved_model_path)
  tensor_info_x = utils.build_tensor_info(x)
  tensor_info_r = utils.build_tensor_info(r)

  func_signature = (
      signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name=signature_constants.PREDICT_METHOD_NAME))

  sm_builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
          'serving_default': func_signature,
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: func_signature,
      },
      strip_default_attrs=True)
  sm_builder.save()


if __name__ == '__main__':
  app.run(main)
