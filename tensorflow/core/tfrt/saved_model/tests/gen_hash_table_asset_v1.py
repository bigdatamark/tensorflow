# [MODULE DOC]
# HELPER: generator script producing test data for runtime saved model tests (not a test itself).
# Builds a version-1 saved model containing a static vocabulary lookup table initialized from a
# text file asset, saved with the table initializer as the main op and assets included.
# Defines a required flag for the output path.
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

import os
import shutil
import tempfile

from absl import app
from absl import flags
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variables_toggle
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils

flags.DEFINE_string('saved_model_path', None, 'Path to save the model to.')
FLAGS = flags.FLAGS


def write_vocabulary_file(vocabulary):
  # [FUNCTION DOC]
  # HELPER: creates the vocabulary file used by the table initializer.
  # Input: vocabulary, a list of strings. Creates a new temporary directory (never cleaned up),
  # writes each entry on its own line to a file named for tokens, and returns that file's path.
  """Write temporary vocab file for module construction."""
  tmpdir = tempfile.mkdtemp()
  vocabulary_file = os.path.join(tmpdir, 'tokens.txt')
  with gfile.GFile(vocabulary_file, 'w') as f:
    for entry in vocabulary:
      f.write(entry + '\n')
  return vocabulary_file


def main(argv):
  # [FUNCTION DOC]
  # HELPER: entry point of the generator script.
  # KNOWN ISSUE: the output directory is removed unconditionally and must already exist.
  # Input: argv, list of command line arguments; raises a usage error if more than the program name.
  # Removes the existing output directory, enables resource variables, builds a static vocabulary
  # table (ten out-of-vocabulary buckets) from a vocabulary file with the words cat, is, on, the,
  # mat, looks up a scalar string placeholder, initializes variables, and saves a model with a
  # serving tag, default predict signature (input x, output r), the table initializer as main op, and
  # the asset file paths collection. Returns nothing. Side effects: temp files, filesystem writes.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  shutil.rmtree(FLAGS.saved_model_path)

  resource_variables_toggle.enable_resource_variables()

  # Create the graph
  table_initializer = lookup_ops.TextFileInitializer(
      write_vocabulary_file(['cat', 'is', 'on', 'the', 'mat']), dtypes.string,
      lookup_ops.TextFileIndex.WHOLE_LINE, dtypes.int64,
      lookup_ops.TextFileIndex.LINE_NUMBER)
  table = lookup_ops.StaticVocabularyTable(
      table_initializer, num_oov_buckets=10)

  key = array_ops.placeholder(dtypes.string, shape=(), name='input')
  result = table.lookup(key)

  sess = session.Session()

  sess.run(variables.global_variables_initializer())

  sm_builder = builder.SavedModelBuilder(FLAGS.saved_model_path)
  tensor_info_x = utils.build_tensor_info(key)
  tensor_info_r = utils.build_tensor_info(result)

  toy_signature = (
      signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name=signature_constants.PREDICT_METHOD_NAME))

  sm_builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: toy_signature,
      },
      main_op=lookup_ops.tables_initializer(),
      assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
      strip_default_attrs=True)
  sm_builder.save()


if __name__ == '__main__':
  flags.mark_flag_as_required('saved_model_path')
  app.run(main)
