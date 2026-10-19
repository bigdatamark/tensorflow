# [MODULE DOC]
# FileCheck test for importing a TF1 SavedModel that contains a static vocabulary hash table
# initialized from a text file asset into MLIR. The checks verify that a session initializer is
# emitted, that two asset entries are created (one for the table initializer and one for the extra
# asset file path collection entry), and that the init function binds the assets as inputs and
# initializes the hash table from the text file. Uses the common_v1 test harness.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# RUN: %p/hash_table_asset_v1| FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import os
import tempfile

import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

# CHECK: "tf_saved_model.session_initializer"() <{initializers = [@[[init:.*]]]}> : () -> ()
# CHECK: "tf_saved_model.asset"() <{filename = {{.*}}, sym_name = "[[asset1:__tf_saved_model_asset1_.*]]"}>
# CHECK: "tf_saved_model.asset"() <{filename = {{.*}}, sym_name = "[[asset0:__tf_saved_model_asset0_.*]]"}>

# CHECK:      func @[[init]]
# CHECK-SAME: [[ARG0:%.*]]: tensor<!tf_type.string> {tf_saved_model.bound_input = @[[asset0]]}
# CHECK-SAME: [[ARG1:%.*]]: tensor<!tf_type.string> {tf_saved_model.bound_input = @[[asset1]]}
# CHECK-SAME: tf_saved_model.initializer_type = "init_op"
# CHECK-NEXT: [[R0:%.*]] = "tf.HashTableV2"()
# CHECK-SAME: shared_name = "[[hash_table:.*]]"
# CHECK-NEXT: "tf.InitializeTableFromTextFileV2"([[R0]], [[ARG0]])


def write_vocabulary_file(vocabulary):
  # [FUNCTION DOC]
  # HELPER: Creates the vocabulary text file used by the hash table in the test function.
  # Takes a list of strings (vocabulary) and writes them, one per line, to a file named tokens.txt
  # inside a newly created temporary directory (never cleaned up). Returns the file path as a string.
  """Write temporary vocab file for module construction."""
  tmpdir = tempfile.mkdtemp()
  vocabulary_file = os.path.join(tmpdir, 'tokens.txt')
  with tf.io.gfile.GFile(vocabulary_file, 'w') as f:
    for entry in vocabulary:
      f.write(entry + '\n')
  return vocabulary_file


def test():
  # [FUNCTION DOC]
  # Builds the TF1 graph for the test. Writes a five word vocabulary file, creates a static
  # vocabulary table (text file initializer, ten out-of-vocabulary buckets) and registers the file
  # path also as an asset file path collection entry so a second asset binding is produced. Defines
  # a string placeholder, looks it up in the table and builds a signature definition with key "key"
  # mapping input x to output r. Takes no arguments. Returns a three element tuple: the mapping of
  # signature names to signature definitions, the table initializers operation, and the list of
  # asset file path tensors from the graph collection. Side effects: adds to the default graph and
  # creates a temporary file.

  vocabulary_file = write_vocabulary_file(['cat', 'is', 'on', 'the', 'mat'])
  table_initializer = tf.lookup.TextFileInitializer(
      vocabulary_file, tf.string, tf.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
      tf.lookup.TextFileIndex.LINE_NUMBER)
  # Incur another bound_input on the asset, but with a different sym_name, i.e.,
  # __tf_saved_model_asset1_tokens.txt vs. __tf_saved_model_asset0_tokens.txt.
  table = tf.lookup.StaticVocabularyTable(table_initializer, num_oov_buckets=10)
  vocab_file_tensor = tf.convert_to_tensor(
      vocabulary_file, tf.string, name='asset_filepath')
  tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, vocab_file_tensor)

  x = tf.placeholder(tf.string, shape=(), name='input')
  r = table.lookup(x)

  tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
  tensor_info_r = tf.compat.v1.saved_model.utils.build_tensor_info(r)

  return {
      'key': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name='some_function'))
  }, tf.tables_initializer(), tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(test, use_lite=True)
