# [MODULE DOC]
# Helper module for the multiplex_4 example showing how a multiplex op is stored in and restored
# from a SavedModel. Provides a function to save a small module containing a given multiplex op
# and a function to load it again and run it. Used by the multiplex_2_save and multiplex_4_load_use
# programs and the multiplex_4 test to demonstrate C++ kernel backward compatibility. No import-time
# side effects.
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
"""Example of using multiplex op in a SavedModel.

multiplex_2_save.py and multiplex_4_load_use.py are programs that use this.

https://www.tensorflow.org/guide/saved_model
https://www.tensorflow.org/api_docs/python/tf/saved_model/save
"""

import tensorflow as tf


def _get_example_tensors():
  # [FUNCTION DOC]
  # HELPER: supplies the fixed example inputs shared by the save and load functions.
  # Takes no arguments. Returns a triple of constant tensors: a boolean condition of length five
  # (true, false, true, false, true) and two 64-bit integer vectors, one to five and ten to fifty.
  cond = tf.constant([True, False, True, False, True], dtype=bool)
  a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
  b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
  return cond, a, b


def save(multiplex_op, path):
  # [FUNCTION DOC]
  # Saves a SavedModel containing a module that calls the given multiplex op. Parameters:
  # multiplex_op, a callable taking condition, a and b tensors (for example a multiplex wrapper of
  # any version, so old and new op versions can be exported); path, the directory to write to.
  # The module has one traced function whose input signature is taken from the example tensors,
  # and that function is also exported as the model signature. Returns nothing; writes files
  # under path.
  """Save a model that contains the given `multiplex_op`.

  Args:
    multiplex_op: A multiplex Custom Op, e.g. multiplex_4_op.multiplex. This is
      parameterized so it can also be used to create an "old" model with an
      older version of the op, e.g. multiplex_2_op.multiplex.
    path: Directory to save model to.
  """
  example_cond, example_a, example_b = _get_example_tensors()

  class UseMultiplex(tf.Module):
    # [CLASS DOC]
    # HELPER: local module class defined inside save, holding a single traced function that calls
    # the multiplex op closed over from the enclosing function. Has no state of its own.

    @tf.function(input_signature=[
        tf.TensorSpec.from_tensor(example_cond),
        tf.TensorSpec.from_tensor(example_a),
        tf.TensorSpec.from_tensor(example_b)
    ])
    def use_multiplex(self, cond, a, b):
      # [FUNCTION DOC]
      # Traced function (input signature fixed to the example tensor types) that applies the
      # multiplex op to the condition and the two value tensors and returns the result tensor.
      return multiplex_op(cond, a, b)

  model = UseMultiplex()
  tf.saved_model.save(
      model,
      path,
      signatures=model.use_multiplex.get_concrete_function(
          tf.TensorSpec.from_tensor(example_cond),
          tf.TensorSpec.from_tensor(example_a),
          tf.TensorSpec.from_tensor(example_b)))


def load_and_use(path):
  # [FUNCTION DOC]
  # Loads a SavedModel previously written by the save function and calls its multiplex function
  # on the example tensors. Parameter: path, the model directory. Returns the result tensor,
  # expected to be the integers one, twenty, three, forty, five. Reads files from path.
  """Load and used a model that was previously created by `save()`.

  Args:
    path: Directory to load model from, typically the same directory that was
      used by save().

  Returns:
    A tensor that is the result of using the multiplex op that is
    tf.constant([1, 20, 3, 40, 5], dtype=tf.int64).
  """
  example_cond, example_a, example_b = _get_example_tensors()
  restored = tf.saved_model.load(path)
  return restored.use_multiplex(example_cond, example_a, example_b)
