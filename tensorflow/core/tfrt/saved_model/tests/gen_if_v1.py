# [MODULE DOC]
# HELPER: generator script producing test data for runtime saved model tests (not a test itself).
# Builds a version-1 saved model with a conditional on an integer scalar input that compares
# against two reference variables. Defines a flag for the output path.
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
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

flags.DEFINE_string('saved_model_path', '', 'Path to save the model to.')
FLAGS = flags.FLAGS


def main(argv):
  # [FUNCTION DOC]
  # HELPER: entry point of the generator script.
  # KNOWN ISSUE: the output directory is removed unconditionally and must already exist.
  # Input: argv, list of command line arguments; raises a usage error if more than the program name.
  # Removes the existing output directory, creates a zero constant, two variables (y holding one and
  # z holding minus one) and a scalar integer placeholder x. The result is a conditional: if x is
  # below zero, 1 or 0 for x greater than y; otherwise 1 or 0 for x greater than z (cast to
  # integer). Initializes variables and saves a model with a serving tag and a predict signature
  # (input x, output r) registered under serving_default and the default serving key.
  # Returns nothing. Side effects: filesystem deletion/writes.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  shutil.rmtree(FLAGS.saved_model_path)

  # Create the graph
  zero = constant_op.constant(0)
  one = variable_scope.get_variable(name='y', initializer=[1])
  neg_one = variable_scope.get_variable(name='z', initializer=[-1])
  x = array_ops.placeholder(dtypes.int32, shape=(), name='input')
  r = cond.cond(
      x < zero, lambda: math_ops.cast(math_ops.greater(x, one), dtypes.int32),
      lambda: math_ops.cast(math_ops.greater(x, neg_one), dtypes.int32))

  sess = session.Session()

  sess.run(variables.global_variables_initializer())

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
