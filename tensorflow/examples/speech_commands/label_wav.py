# [MODULE DOC]
# Command-line script that loads a frozen audio recognition graph and a labels file, feeds one wav
# file to the graph and prints the top predicted labels with scores. Exposes graph/label loading,
# the run step, a combined label_wav function and the main entry point. The module-level FLAGS
# holds parsed arguments when run as a script. No import-time side effects besides imports.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
import argparse
import sys

import tensorflow as tf


FLAGS = None


def load_graph(filename):
  # [FUNCTION DOC]
  # HELPER: graph loading step for label_wav.
  # Reads a serialized graph definition from the file path and imports it, without a name prefix,
  # into the current default graph. Returns nothing; file or parse errors propagate.
  """Unpersists graph from file as default graph."""
  with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  # [FUNCTION DOC]
  # HELPER: labels loading step for label_wav.
  # Returns a list of the lines of the file with trailing whitespace removed, one label per line.
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.io.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  # [FUNCTION DOC]
  # HELPER: inference step for label_wav.
  # Opens a session on the default graph, feeds the raw wav bytes to the named input tensor, reads
  # the named output tensor (a single row of per-class scores), sorts descending and prints the
  # top num_top_predictions labels with scores to standard output. Returns 0 always.
  # Inputs: wav bytes, list of label strings, input tensor name, output tensor name, count.
  # Errors: unknown tensor names raise from the graph lookup; a label index beyond the list raises.
  """Runs the audio data through the graph and prints predictions."""
  with tf.compat.v1.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))

    return 0


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  # [FUNCTION DOC]
  # Validates that the wav, labels and graph paths are non-empty and exist (raises a value error
  # naming the missing file otherwise), loads the labels, imports the graph into the default
  # graph, reads the wav file bytes and prints the top how_many_labels predictions using the given
  # input and output tensor names. Returns nothing.
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.io.gfile.exists(wav):
    raise ValueError('Audio file does not exist at {0}'.format(wav))
  if not labels or not tf.io.gfile.exists(labels):
    raise ValueError('Labels file does not exist at {0}'.format(labels))

  if not graph or not tf.io.gfile.exists(graph):
    raise ValueError('Graph file does not exist at {0}'.format(graph))

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()

  run_graph(wav_data, labels_list, input_name, output_name, how_many_labels)


def main(_):
  # [FUNCTION DOC]
  # Script entry point; the argument (leftover command-line args) is ignored. Passes the parsed
  # flags (wav, labels, graph, input name, output name, number of labels) to label_wav.
  # Returns nothing.
  """Entry point for script, converts flags to arguments."""
  label_wav(FLAGS.wav, FLAGS.labels, FLAGS.graph, FLAGS.input_name,
            FLAGS.output_name, FLAGS.how_many_labels)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--wav', type=str, default='', help='Audio file to be identified.')
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')
  parser.add_argument(
      '--how_many_labels',
      type=int,
      default=3,
      help='Number of results to show.')

  FLAGS, unparsed = parser.parse_known_args()
  tf.compat.v1.app.run(main=main, argv=[sys.argv[0]] + unparsed)
