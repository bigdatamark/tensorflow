# [MODULE DOC]
# Example script (graph-mode, compat v1) showing debugger use while training a Keras model on a
# dataset. Creates a dummy zero-filled dataset and a one-layer model, optionally wraps the
# session in the command-line debugger or TensorBoard debugger wrapper, installs it as the Keras
# session, and trains. Flags select debug mode, UI type, config path, TensorBoard address and
# epochs. Runs only as a main program.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""tfdbg example: debugging tf.keras models training on tf.data.Dataset."""

import argparse
import sys
import tempfile

import numpy as np
import tensorflow

from tensorflow.python import debug as tf_debug

tf = tensorflow.compat.v1


def main(_):
  # [FUNCTION DOC]
  # KNOWN ISSUE: help text says debug and TensorBoard address are mutually exclusive, but no
  # check enforces it; the debug flag silently takes priority.
  # Entry point; the argument is the unused list of leftover command-line arguments.
  # Builds a repeated, batched dataset of zeros, creates a session (wrapped in the command-line
  # debugger, possibly with a temporary config file, or in the TensorBoard debugger wrapper
  # when requested), sets it as the Keras backend session, builds a one-unit dense model compiled
  # with mean squared error and gradient descent, and fits it for the requested epochs.
  # Relies on the module-level flags object.
  # Create a dummy dataset.
  num_examples = 8
  steps_per_epoch = 2
  input_dims = 3
  output_dims = 1
  xs = np.zeros([num_examples, input_dims])
  ys = np.zeros([num_examples, output_dims])
  dataset = tf.data.Dataset.from_tensor_slices(
      (xs, ys)).repeat(num_examples).batch(int(num_examples / steps_per_epoch))

  sess = tf.Session()
  if FLAGS.debug:
    # Use the command-line interface (CLI) of tfdbg.
    if FLAGS.use_random_config_path:
      _, config_file_path = tempfile.mkstemp(".tfdbg_config")
    else:
      config_file_path = None
    sess = tf_debug.LocalCLIDebugWrapperSession(
        sess, ui_type=FLAGS.ui_type, config_file_path=config_file_path)
  elif FLAGS.tensorboard_debug_address:
    # Use the TensorBoard Debugger Plugin (GUI of tfdbg).
    sess = tf_debug.TensorBoardDebugWrapperSession(
        sess, FLAGS.tensorboard_debug_address)
  tf.keras.backend.set_session(sess)

  # Create a dummy model.
  model = tf.keras.Sequential(
      [tf.keras.layers.Dense(1, input_shape=[input_dims])])
  model.compile(loss="mse", optimizer="sgd")

  # Train the model using the dummy dataset created above.
  model.fit(dataset, epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--debug",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Use debugger to track down bad values during training. "
      "Mutually exclusive with the --tensorboard_debug_address flag.")
  parser.add_argument(
      "--ui_type",
      type=str,
      default="readline",
      help="Command-line user interface type (only readline is supported).")
  parser.add_argument(
      "--use_random_config_path",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="""If set, set config file path to a random file in the temporary
      directory.""")
  parser.add_argument(
      "--tensorboard_debug_address",
      type=str,
      default=None,
      help="Connect to the TensorBoard Debugger Plugin backend specified by "
      "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
      "--debug flag.")
  parser.add_argument(
      "--epochs",
      type=int,
      default=2,
      help="Number of epochs to train the model for.")
  FLAGS, unparsed = parser.parse_known_args()
  with tf.Graph().as_default():
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
