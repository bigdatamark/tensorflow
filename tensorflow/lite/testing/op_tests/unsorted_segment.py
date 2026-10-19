# [MODULE DOC]
# Test-case definition module for the TFLite converter test suite covering the unsorted segment
# reductions (product, maximum, minimum, sum). A shared builder function takes the specific
# reduction operation and generates parameterized graphs (including negative segment ids,
# empty dimensions and a two-step dynamic-shape case); four thin registered generator functions
# bind it to each reduction. Import-time side effect: registers the four generators in the
# global registry of test-making functions.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for unsorted_segment ops."""

import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def make_unsorted_segment_tests(options, unsorted_segment_op):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the four registered unsorted segment generators.
  # Inputs: options (test-run configuration passed to the zip-test builder) and
  # unsorted_segment_op (the segment reduction function to test). Defines parameter groups with
  # constant segment ids (including negative ids, ids beyond used segments, higher-rank ids and a
  # zero-sized dimension) and one multi-node group where segment ids are a runtime input and a
  # second reduction follows. Defines the graph and input builders (dispatching on a multi-node
  # flag) and runs the zip-test machinery. Returns nothing.
  """Make a set of tests for given unsorted_segment op."""
  test_parameters = [{
      "data_shape": [[5]],
      "segment_id": [[0, 1, 1, 0, 1]],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[2, 3, 4], [2, 5, 2]],
      "segment_id": [[0, 1]],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[4]],
      "segment_id": [[0, 0, 1, 8]],
      "num_segments": [9],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[3]],
      "segment_id": [[-1, -2, -1]],
      "num_segments": [1],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[3]],
      "segment_id": [[-1, 0, 1]],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[3, 2]],
      "segment_id": [[-1, 0, 0]],
      "num_segments": [1],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[3, 2]],
      "segment_id": [[-1, -2, -1]],
      "num_segments": [1],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[4]],
      "segment_id_shape": [[4]],
      "segment_id_min": [0],
      "segment_id_max": [1],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "segment_id_2": [[0, 0]],
      "num_segments_2": [1],
      "multi_node": [1]
  }, {
      "data_shape": [[2, 2, 3]],
      "segment_id": [[[1, 2], [3, 4]], [4, 5],
                     [[[1, 2, 3], [3, 4, 5]], [[1, 2, 4], [0, 0, -1]]]],
      "num_segments": [10],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[2, 0, 3]],
      "segment_id": [[1, 1]],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }]

  def build_graph_one_node(parameters):
    # [FUNCTION DOC]
    # HELPER: single-reduction graph builder of the unsorted segment tests.
    # Creates a data placeholder, constant 32-bit segment ids and a scalar constant segment
    # count, and applies the reduction. Returns a pair: list with the data placeholder and list
    # with the output.
    data_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="data", shape=parameters["data_shape"])
    segment_ids_tensor = tf.constant(
        parameters["segment_id"], dtype=tf.int32, name="segment_ids")
    num_segments_tensor = tf.constant(
        parameters["num_segments"],
        dtype=tf.int32,
        shape=[],
        name="num_segments")
    output = unsorted_segment_op(data_tensor, segment_ids_tensor,
                                 num_segments_tensor)
    return [data_tensor], [output]


# test cases for handling dynamically shaped input tensor
  def build_graph_multi_node(parameters):
    # [FUNCTION DOC]
    # HELPER: two-reduction graph builder of the unsorted segment tests (dynamic-shape case).
    # Creates a data placeholder and a segment-id placeholder, applies the reduction with a
    # constant segment count, then applies the reduction again on that result with constant
    # second segment ids and count. Returns a pair: list with the data and segment-id
    # placeholders and list with the final output.
    data_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="data", shape=parameters["data_shape"])
    segment_ids_tensor = tf.compat.v1.placeholder(
        dtype=tf.int32,
        name="segment_ids",
        shape=parameters["segment_id_shape"])
    num_segments_tensor = tf.constant(
        parameters["num_segments"],
        dtype=tf.int32,
        shape=[],
        name="num_segments")
    intermediate_tensor = unsorted_segment_op(data_tensor, segment_ids_tensor,
                                              num_segments_tensor)
    segment_ids_tensor_2 = tf.constant(
        parameters["segment_id_2"], dtype=tf.int32, name="segment_ids_2")
    num_segments_tensor_2 = tf.constant(
        parameters["num_segments_2"],
        dtype=tf.int32,
        shape=[],
        name="num_segments_2")
    output = unsorted_segment_op(intermediate_tensor, segment_ids_tensor_2,
                                 num_segments_tensor_2)
    return [data_tensor, segment_ids_tensor], [output]

  def build_graph(parameters):
    # [FUNCTION DOC]
    # Graph builder passed to the zip-test machinery: chooses the two-reduction builder when the
    # multi-node flag of the parameter combination is set, otherwise the single-reduction
    # builder, and returns its pair of input tensors list and output tensors list.
    multi_node = parameters["multi_node"]
    if multi_node:
      return build_graph_multi_node(parameters)

    return build_graph_one_node(parameters)

  def build_inputs_one_node(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input generator for the single-reduction case. Creates random data of the parameter
    # type and shape, evaluates the graph in the given session and returns a pair: list with the
    # data array and the evaluated outputs.
    data_value = create_tensor_data(
        parameters["dtype"], shape=parameters["data_shape"])
    return [data_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [data_value])))

  def build_inputs_multi_node(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: input generator for the two-reduction case. Creates random data of the parameter
    # type and shape, and random 32-bit integer segment ids of the given shape within the given
    # minimum and maximum. Evaluates the graph in the given session and returns a pair: list
    # with the data and segment-id arrays and the evaluated outputs.
    data_value = create_tensor_data(
        dtype=parameters["dtype"], shape=parameters["data_shape"])
    segment_id_value = create_tensor_data(
        dtype=tf.int32,
        shape=parameters["segment_id_shape"],
        min_value=parameters["segment_id_min"],
        max_value=parameters["segment_id_max"])
    return [data_value, segment_id_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [data_value, segment_id_value])))

  def build_inputs(parameters, sess, inputs, outputs):
    # [FUNCTION DOC]
    # Input generator passed to the zip-test machinery: dispatches to the two-reduction input
    # generator when the multi-node flag is set, otherwise to the single-reduction one, and
    # returns its pair of input arrays and evaluated outputs.
    multi_node = parameters["multi_node"]
    if multi_node:
      return build_inputs_multi_node(parameters, sess, inputs, outputs)

    return build_inputs_one_node(parameters, sess, inputs, outputs)

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)


@register_make_test_function()
def make_unsorted_segment_prod_tests(options):
  # [FUNCTION DOC]
  # Registered generator for the unsorted segment product tests: runs the shared builder with
  # the product reduction. Takes the test-run configuration; returns nothing.
  make_unsorted_segment_tests(options, tf.math.unsorted_segment_prod)


@register_make_test_function()
def make_unsorted_segment_max_tests(options):
  # [FUNCTION DOC]
  # Registered generator for the unsorted segment maximum tests: runs the shared builder with
  # the maximum reduction. Takes the test-run configuration; returns nothing.
  make_unsorted_segment_tests(options, tf.math.unsorted_segment_max)


@register_make_test_function()
def make_unsorted_segment_min_tests(options):
  # [FUNCTION DOC]
  # Registered generator for the unsorted segment minimum tests: runs the shared builder with
  # the minimum reduction. Takes the test-run configuration; returns nothing.
  make_unsorted_segment_tests(options, tf.math.unsorted_segment_min)


@register_make_test_function()
def make_unsorted_segment_sum_tests(options):
  # [FUNCTION DOC]
  # Registered generator for the unsorted segment sum tests: runs the shared builder with the
  # sum reduction. Takes the test-run configuration; returns nothing.
  make_unsorted_segment_tests(options, tf.math.unsorted_segment_sum)
