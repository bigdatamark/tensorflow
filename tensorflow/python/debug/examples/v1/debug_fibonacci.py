# [MODULE DOC]
# Example script (graph-mode, compat v1) demonstrating the debugger on a graph that computes a
# Fibonacci-like sequence with square integer tensors. Flags control tensor size, sequence
# length, UI type, enabling the command-line debugger, or connecting to a TensorBoard debugger
# plugin address (the two are mutually exclusive). Runs only as a main program.
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
"""Demo of the tfdbg readline UI: A TF network computing Fibonacci sequence."""
import argparse
import sys

import numpy as np
import tensorflow

from tensorflow.python import debug as tf_debug

tf = tensorflow.compat.v1

FLAGS = None


def main(_):
  # [FUNCTION DOC]
  # Entry point; the argument is the unused list of leftover command-line arguments.
  # Builds two variables of ones and a chain of addition nodes up to the requested length,
  # initializes variables, and optionally wraps the session: raises a value error if both the
  # debug flag and the TensorBoard address are set; with debug, uses the command-line debugger
  # and registers two tensor filters (one for infinity or not-a-number values, one for negative
  # values); with an address, uses the TensorBoard debugger wrapper. Prints the final value.
  # Relies on the module-level flags object set in the main block.
  sess = tf.Session()

  # Construct the TensorFlow network.
  n0 = tf.Variable(
      np.ones([FLAGS.tensor_size] * 2), dtype=tf.int32, name="node_00")
  n1 = tf.Variable(
      np.ones([FLAGS.tensor_size] * 2), dtype=tf.int32, name="node_01")

  for i in range(2, FLAGS.length):
    n0, n1 = n1, tf.add(n0, n1, name="node_%.2d" % i)

  sess.run(tf.global_variables_initializer())

  # Wrap the TensorFlow Session object for debugging.
  if FLAGS.debug and FLAGS.tensorboard_debug_address:
    raise ValueError(
        "The --debug and --tensorboard_debug_address flags are mutually "
        "exclusive.")
  if FLAGS.debug:
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    def has_negative(_, tensor):
      # [FUNCTION DOC]
      # Tensor filter for the debugger: ignores its first argument (the datum) and returns
      # whether any element of the tensor value is negative.
      return np.any(tensor < 0)

    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.add_tensor_filter("has_negative", has_negative)
  elif FLAGS.tensorboard_debug_address:
    sess = tf_debug.TensorBoardDebugWrapperSession(
        sess, FLAGS.tensorboard_debug_address)

  print("Fibonacci number at position %d:\n%s" % (FLAGS.length, sess.run(n1)))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--tensor_size",
      type=int,
      default=1,
      help="""\
      Size of tensor. E.g., if the value is 30, the tensors will have shape
      [30, 30].\
      """)
  parser.add_argument(
      "--length",
      type=int,
      default=20,
      help="Length of the fibonacci sequence to compute.")
  parser.add_argument(
      "--ui_type",
      type=str,
      default="readline",
      help="Command-line user interface type (only readline is supported)")
  parser.add_argument(
      "--debug",
      dest="debug",
      action="store_true",
      help="Use TensorFlow Debugger (tfdbg). Mutually exclusive with the "
      "--tensorboard_debug_address flag.")
  parser.add_argument(
      "--tensorboard_debug_address",
      type=str,
      default=None,
      help="Connect to the TensorBoard Debugger Plugin backend specified by "
      "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
      "--debug flag.")

  FLAGS, unparsed = parser.parse_known_args()
  with tf.Graph().as_default():
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
