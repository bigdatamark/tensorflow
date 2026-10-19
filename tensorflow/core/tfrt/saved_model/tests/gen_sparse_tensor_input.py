# [MODULE DOC]
# HELPER: generator script producing test data for runtime saved model tests (not a test itself).
# Builds a version-1 saved model with one signature taking a sparse tensor input (reduced to its
# sum) and another with a dense input. Defines a flag for the output path.
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
"""Generates a saved model with tf.Pow to trigger placer and grappler."""

import shutil
from absl import app
from absl import flags
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
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
  # Removes the existing output directory, creates a sparse integer placeholder x reduced by sum,
  # and a dense 1 by 3 placeholder x1 plus one. Saves a model with a serving tag and signatures
  # "sparse" (input x, output r) and "dense" (input x1, output r1). Returns nothing.
  # Side effects: filesystem deletion/writes.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  shutil.rmtree(FLAGS.saved_model_path)

  # Create the graph
  x = array_ops.sparse_placeholder(dtype=dtypes.int32, shape=None, name='input')
  r = sparse_ops.sparse_reduce_sum(x)

  x1 = array_ops.placeholder(dtype=dtypes.int32, shape=(1, 3), name='input1')
  r1 = math_ops.add(x1, 1)

  sess = session.Session()

  sm_builder = builder.SavedModelBuilder(FLAGS.saved_model_path)

  tensor_info_x = utils.build_tensor_info(x)
  tensor_info_r = utils.build_tensor_info(r)

  tensor_info_x1 = utils.build_tensor_info(x1)
  tensor_info_r1 = utils.build_tensor_info(r1)

  sparse_signature = (
      signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name=signature_constants.PREDICT_METHOD_NAME))

  dense_signature = (
      signature_def_utils.build_signature_def(
          inputs={'x1': tensor_info_x1},
          outputs={'r1': tensor_info_r1},
          method_name=signature_constants.PREDICT_METHOD_NAME))

  sm_builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
          'sparse': sparse_signature,
          'dense': dense_signature,
      },
      strip_default_attrs=True)
  sm_builder.save()


if __name__ == '__main__':
  app.run(main)
