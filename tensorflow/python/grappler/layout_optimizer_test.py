# [MODULE DOC]
# Tests for the Grappler layout optimizer, which rewrites convolution-centric graphs from the channels-last
# data format to the channels-first format (and the 3-D equivalents) on GPU, inserting and cancelling
# transpose, dimension-map and vector-permute nodes. The module defines small graph-building helper
# functions (a two-layer convolution model, loops, a metagraph, a fake GPU cluster, node-name predicates)
# and one test class that builds graphs, runs them with the optimizer off and on, counts the inserted
# layout nodes in the run metadata cost graph and compares numerical outputs. Most tests do nothing unless
# a CUDA GPU is available. No import-time side effects beyond imports; runs via the test main entry point.
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
"""Tests for Grappler LayoutOptimizer."""

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import device_properties_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.layers import convolutional as conv_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver as saver_lib


def _weight(shape):
  # [FUNCTION DOC]
  # HELPER: graph-building utility for the model helpers and tests in this module.
  # Takes a shape (list of integers) and returns a tensor of truncated-normal random values with a fixed
  # seed of zero and standard deviation 0.1. Adds a random op to the current default graph.
  """Generates a weight of a given shape."""
  return random_ops.truncated_normal(shape, seed=0, stddev=0.1)


def _bias(shape):
  # [FUNCTION DOC]
  # HELPER: graph-building utility for the model helpers in this module.
  # Takes a shape (list of integers) and returns a constant tensor of that shape filled with 0.1. Adds a
  # constant op to the current default graph.
  """Generates a bias of a given shape."""
  return constant_op.constant(0.1, shape=shape)


def _conv2d(x, w):
  # [FUNCTION DOC]
  # HELPER: graph-building utility for the model helpers in this module.
  # Takes an input tensor x and filter tensor w and returns a two-dimensional convolution with unit strides
  # and same-style padding in the default channels-last format. Adds a convolution op to the graph.
  """Returns a 2d convolution layer with full stride."""
  return nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(x):
  # [FUNCTION DOC]
  # HELPER: graph-building utility for the model helpers in this module.
  # Takes a feature-map tensor and returns it max-pooled with a two-by-two window and stride two using
  # same-style padding, i.e. downsampled by a factor of two. Adds a pooling op to the graph.
  """Downsamples a feature map by 2X."""
  return nn.max_pool(
      x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Taken from tensorflow/examples/tutorials/mnist/mnist_deep.py
def _two_layer_model(x):
  # [FUNCTION DOC]
  # HELPER: shared model fixture used by most layout optimizer tests (taken from the MNIST deep tutorial).
  # Input x is a tensor reshaped to batches of 28 by 28 single-channel images. Applies two stages of
  # convolution (5 by 5 filters, 32 then 64 channels), bias add, ReLU and 2x2 max pooling using random
  # weights and constant biases. Returns the final pooled tensor (4-D, channels-last). Adds ops to the
  # default graph; no exceptions of its own.
  x_image = array_ops.reshape(x, [-1, 28, 28, 1])
  w_conv1 = _weight([5, 5, 1, 32])
  b_conv1 = _bias([32])
  h_conv1 = nn.relu(_conv2d(x_image, w_conv1) + b_conv1)
  h_pool1 = _max_pool_2x2(h_conv1)
  w_conv2 = _weight([5, 5, 32, 64])
  b_conv2 = _bias([64])
  h_conv2 = nn.relu(_conv2d(h_pool1, w_conv2) + b_conv2)
  h_pool2 = _max_pool_2x2(h_conv2)
  return h_pool2


def _model_with_second_port():
  # [FUNCTION DOC]
  # HELPER: model fixture for the second-port binary op test.
  # Seeds the random generator, creates a random 4-D tensor, runs fused batch normalization with constant
  # scale and offset and adds the first output to the batch mean (the second output port). Returns an
  # identity of that sum. Takes no inputs; sets the global random seed and adds ops to the default graph.
  random_seed.set_random_seed(0)
  x = random_ops.truncated_normal([2, 5, 5, 4], seed=0)
  scale = constant_op.constant(0.1, shape=[4])
  offset = constant_op.constant(0.3, shape=[4])
  y, mean, _ = nn.fused_batch_norm(x, scale, offset)
  mul = math_ops.add(y, mean)
  output = array_ops.identity(mul)
  return output


def _model_with_branch(x):
  # [FUNCTION DOC]
  # HELPER: model fixture used inside the looped branch model.
  # Input x is reshaped to 28 by 28 single-channel images, passed through two parallel convolutions with
  # different random 5 by 5 filters, and the two results are added. Returns the sum tensor.
  x_image = array_ops.reshape(x, [-1, 28, 28, 1])
  w_conv1 = _weight([5, 5, 1, 32])
  w_conv2 = _weight([5, 5, 1, 32])
  c_conv1 = _conv2d(x_image, w_conv1)
  c_conv2 = _conv2d(x_image, w_conv2)
  add = math_ops.add(c_conv1, c_conv2)
  return add


def _model_with_vec_and_4d(x):
  # [FUNCTION DOC]
  # HELPER: model fixture used inside the looped vector-plus-4D model.
  # Input x is reshaped to 28 by 28 single-channel images, convolved with a random filter, and a constant
  # 32-element vector is added to the 4-D result (broadcast along the channel dimension). Returns the sum.
  x_image = array_ops.reshape(x, [-1, 28, 28, 1])
  w_conv1 = _weight([5, 5, 1, 32])
  c_conv1 = _conv2d(x_image, w_conv1)
  vector = constant_op.constant(6.4, shape=[32])
  add = math_ops.add(c_conv1, vector)
  return add


def _loop():
  # [FUNCTION DOC]
  # HELPER: model fixture for the loop test.
  # Seeds the random generator, creates four random one-by-784 tensors and maps the two-layer model over
  # them with the map function, producing a while-loop based graph. Returns the stacked output tensors.
  # Sets the global random seed; takes no inputs.
  random_seed.set_random_seed(0)
  x1 = random_ops.truncated_normal([1, 784], seed=0)
  x2 = random_ops.truncated_normal([1, 784], seed=0)
  x3 = random_ops.truncated_normal([1, 784], seed=0)
  x4 = random_ops.truncated_normal([1, 784], seed=0)
  elems = (x1, x2, x3, x4)
  outputs = map_fn.map_fn(_two_layer_model, elems, dtype=dtypes.float32)
  return outputs


def _loop_with_branch():
  # [FUNCTION DOC]
  # HELPER: model fixture for the loop-with-branch test.
  # Same structure as the loop fixture, but maps the branching two-convolution model over four random
  # inputs with the map function. Returns the output tensors; sets the global random seed.
  random_seed.set_random_seed(0)
  x1 = random_ops.truncated_normal([1, 784], seed=0)
  x2 = random_ops.truncated_normal([1, 784], seed=0)
  x3 = random_ops.truncated_normal([1, 784], seed=0)
  x4 = random_ops.truncated_normal([1, 784], seed=0)
  elems = (x1, x2, x3, x4)
  outputs = map_fn.map_fn(_model_with_branch, elems, dtype=dtypes.float32)
  return outputs


def _loop_with_vec_and_4d():
  # [FUNCTION DOC]
  # HELPER: model fixture for the loop test mixing a vector and a 4-D tensor.
  # Same structure as the loop fixture, but maps the model that adds a constant vector to a convolution
  # output over four random inputs. Returns the output tensors; sets the global random seed.
  random_seed.set_random_seed(0)
  x1 = random_ops.truncated_normal([1, 784], seed=0)
  x2 = random_ops.truncated_normal([1, 784], seed=0)
  x3 = random_ops.truncated_normal([1, 784], seed=0)
  x4 = random_ops.truncated_normal([1, 784], seed=0)
  elems = (x1, x2, x3, x4)
  outputs = map_fn.map_fn(_model_with_vec_and_4d, elems, dtype=dtypes.float32)
  return outputs


def _get_config(layout_optimizer=True):
  # [FUNCTION DOC]
  # HELPER: session configuration factory for the layout optimizer tests.
  # Input layout_optimizer is a boolean (default true) selecting whether the layout optimizer is turned on
  # or off. Arithmetic optimization is always off so duplicated nodes are not merged, the minimum graph
  # node count is disabled (negative), the cost model is built so node names show up in run metadata, and
  # the optimizer opt level is set to the lowest. Returns a session configuration message; no side effects.
  if layout_optimizer:
    rewrite_options = rewriter_config_pb2.RewriterConfig(
        layout_optimizer=rewriter_config_pb2.RewriterConfig.ON,
        # do not remove duplicated nodes
        arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF)
  else:
    rewrite_options = rewriter_config_pb2.RewriterConfig(
        layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF,
        # do not remove duplicated nodes
        arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF)
  rewrite_options.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(
      rewrite_options=rewrite_options, build_cost_model=1)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  config.graph_options.optimizer_options.opt_level = -1
  return config


def _simple_metagraph(depthwise=False):
  # [FUNCTION DOC]
  # HELPER: builds the metagraph for the gradient and depthwise tests.
  # Input depthwise is a boolean (default false): when true separable convolutions are used instead of
  # ordinary ones. Seeds the random generator, creates a variable holding a random 1 by 200 by 200 by 3
  # tensor, applies two convolution layers with 32 filters of size 3 by 3, a mean loss and a gradient
  # descent training op, registers the train op in the train-op collection of the default graph and
  # returns an exported metagraph. Side effects: global random seed, default graph and variable creation.
  random_seed.set_random_seed(0)
  x = variables.Variable(random_ops.truncated_normal([1, 200, 200, 3], seed=0))
  conv = conv_layers.separable_conv2d if depthwise else conv_layers.conv2d
  y = conv(x, 32, [3, 3])
  z = conv(y, 32, [3, 3])
  optimizer = gradient_descent.GradientDescentOptimizer(1e-4)
  loss = math_ops.reduce_mean(z)
  train_op = optimizer.minimize(loss)
  graph = ops.get_default_graph()
  graph.add_to_collection('train_op', train_op)
  meta_graph = saver_lib.export_meta_graph(graph_def=graph.as_graph_def())
  return meta_graph


def _get_cluster():
  # [FUNCTION DOC]
  # HELPER: builds a fake cluster for the optimizer-based tests.
  # Takes no inputs. Describes one named GPU device with 24 cores, frequency 1000 and architecture value 4
  # and returns a Grappler cluster object built from that device list, so the layout optimizer believes a
  # GPU is present without needing a real one.
  named_device = device_properties_pb2.NamedDevice()
  named_device.name = '/GPU:0'
  named_device.properties.type = 'GPU'
  named_device.properties.num_cores = 24
  named_device.properties.frequency = 1000
  named_device.properties.environment['architecture'] = '4'
  cluster = gcluster.Cluster(devices=[named_device])
  return cluster


def _is_transpose(node):
  # [FUNCTION DOC]
  # HELPER: node-name predicate for counting layout transposes in cost graphs.
  # Takes a node name string and returns true if it ends with one of the four layout optimizer transpose
  # suffixes (channels-last to channels-first and back, for both 4-D and 5-D formats), otherwise false.
  return node.endswith('TransposeNHWCToNCHW-LayoutOptimizer') or node.endswith(
      'TransposeNCHWToNHWC-LayoutOptimizer') or node.endswith(
          'TransposeNDHWCToNCDHW-LayoutOptimizer') or node.endswith(
              'TransposeNCDHWToNDHWC-LayoutOptimizer')


def _is_permute(node):
  # [FUNCTION DOC]
  # HELPER: node-name predicate for counting vector permutes in cost graphs.
  # Takes a node name string and returns true if it ends with a layout optimizer vector-permute suffix
  # (channels-last to channels-first or the reverse), otherwise false.
  return node.endswith('VecPermuteNHWCToNCHW-LayoutOptimizer') or node.endswith(
      'VecPermuteNCHWToNHWC-LayoutOptimizer')


@test_util.for_all_test_methods(test_util.no_xla_auto_jit,
                                'Test does not apply in XLA setting')
class LayoutOptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Grappler layout optimizer. Decorated so that every test method is skipped under the
  # XLA auto-jit setting. Each test builds a model on top of convolution ops, runs it in two sessions (layout
  # optimizer off for a reference and on for the result), reads the cost graph from run metadata, counts
  # transposes and checks that particular transpose, dimension-map or vector-permute nodes exist, and
  # compares outputs with a tolerance. Holds a family of assertion helper methods for those node names and
  # a training helper used by the checkpoint test. Most tests only run when a CUDA GPU is available.
  """Tests the Grappler layout optimizer."""

  def _assert_trans_nchw_to_nhwc(self, name, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the layout optimizer tests.
    # Asserts that the node list contains the node named by the given name plus the suffix for a
    # channels-first to channels-last transpose. Fails the test otherwise.
    self.assertIn(name + '-TransposeNCHWToNHWC-LayoutOptimizer', nodes)

  def _assert_trans_nhwc_to_nchw(self, name, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the layout optimizer tests.
    # Asserts that the node list contains the node named by the given name plus the suffix for a
    # channels-last to channels-first transpose. Fails the test otherwise.
    self.assertIn(name + '-TransposeNHWCToNCHW-LayoutOptimizer', nodes)

  def _assert_trans_ncdhw_to_ndhwc(self, name, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the layout optimizer tests.
    # Asserts that the node list contains the node named by the given name plus the suffix for a 5-D
    # channels-first to channels-last transpose. Fails the test otherwise.
    self.assertIn(name + '-TransposeNCDHWToNDHWC-LayoutOptimizer', nodes)

  def _assert_trans_ndhwc_to_ncdhw(self, name, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the layout optimizer tests.
    # Asserts that the node list contains the node named by the given name plus the suffix for a 5-D
    # channels-last to channels-first transpose. Fails the test otherwise.
    self.assertIn(name + '-TransposeNDHWCToNCDHW-LayoutOptimizer', nodes)

  def _assert_map_nhwc_to_nchw(self, name, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the layout optimizer tests.
    # Asserts that the node list contains the node named by the given name plus the suffix for a dimension-map
    # node converting 4-D channels-last axis indices to channels-first. Fails the test otherwise.
    self.assertIn(name + '-DimMapNHWCToNCHW-LayoutOptimizer', nodes)

  def _assert_map_ndhwc_to_ncdhw(self, name, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the layout optimizer tests.
    # Asserts that the node list contains the node named by the given name plus the suffix for a data-format
    # dimension-map node converting 5-D channels-last axis indices to channels-first. Fails otherwise.
    self.assertIn(name + '-DataFormatDimMapNDHWCToNCDHW-LayoutOptimizer', nodes)

  def _assert_vec_nchw_to_nhwc(self, name, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the layout optimizer tests.
    # Asserts that the node list contains the node named by the given name plus the suffix for a vector
    # permute node converting a 4-D channels-first vector (such as a shape) to channels-last order.
    self.assertIn(name + '-VecPermuteNCHWToNHWC-LayoutOptimizer', nodes)

  def _assert_vec_nhwc_to_nchw(self, name, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the layout optimizer tests.
    # Asserts that the node list contains the node named by the given name plus the suffix for a vector
    # permute node converting a 4-D channels-last vector (such as sizes or paddings) to channels-first order.
    self.assertIn(name + '-VecPermuteNHWCToNCHW-LayoutOptimizer', nodes)

  def _assert_vec_ncdhw_to_ndhwc(self, name, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the layout optimizer tests.
    # Asserts that the node list contains the node named by the given name plus the suffix for a data-format
    # vector permute node converting a 5-D channels-first vector to channels-last order.
    self.assertIn(name + '-DataFormatVecPermuteNCDHWToNDHWC-LayoutOptimizer',
                  nodes)

  def _assert_vec_ndhwc_to_ncdhw(self, name, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the layout optimizer tests.
    # Asserts that the node list contains the node named by the given name plus the suffix for a data-format
    # vector permute node converting a 5-D channels-last vector to channels-first order.
    self.assertIn(name + '-DataFormatVecPermuteNDHWCToNCDHW-LayoutOptimizer',
                  nodes)

  def _train(self, checkpoint_path, layout_optimizer=False, restore=False):
    # [FUNCTION DOC]
    # HELPER: training routine used by the checkpoint compatibility test.
    # Inputs: checkpoint_path (location of the checkpoint), layout_optimizer (boolean, default false, turns the
    # optimizer on in the session configuration), restore (boolean, default false). Resets the default graph,
    # builds two convolution layers on a placeholder of shape two by six by seven by three with a gradient
    # descent training op and a version-2 saver. If restore is true the variables are restored from the
    # checkpoint, otherwise they are initialized. Runs two training steps with seeded random inputs. When
    # restoring it returns the list of values of all global variables; otherwise it saves a checkpoint to
    # the path and returns nothing. Side effects: resets the global default graph, seeds numpy, writes files.
    ops.reset_default_graph()
    graph = ops.get_default_graph()
    with session.Session(
        config=_get_config(layout_optimizer), graph=graph) as sess:
      batch = 2
      height = 6
      width = 7
      input_channels = 3
      shape = [batch, height, width, input_channels]
      image = array_ops.placeholder(dtype='float32', shape=shape)
      conv1 = conv_layers.conv2d(image, 32, [3, 3])
      conv2 = conv_layers.conv2d(conv1, 32, [3, 3])
      optimizer = gradient_descent.GradientDescentOptimizer(0.01)
      loss = math_ops.reduce_mean(conv2)
      train_op = optimizer.minimize(loss)
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

      if restore:
        saver.restore(sess, checkpoint_path)
      else:
        self.evaluate(variables.global_variables_initializer())

      np.random.seed(0)
      for _ in range(2):
        image_val = np.random.rand(*shape).astype(np.float32)
        sess.run([loss, train_op], feed_dict={image: image_val})

      if restore:
        all_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
        all_vars_values = [var.eval(session=sess) for var in all_vars]
        return all_vars_values
      else:
        saver.save(sess, checkpoint_path)

  @test_util.deprecated_graph_mode_only
  def testTwoConvLayers(self):
    # [FUNCTION DOC]
    # Scenario: only on GPU, runs the two-layer convolution model with the layout optimizer off and on.
    # Asserts exactly two transposes remain (four were added, two cancelled), a channels-last to channels-first
    # transpose after the first convolution, a channels-first to channels-last transpose after the final
    # ReLU, and outputs equal within a small tolerance.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      output = _two_layer_model(x)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Relu_1-0-0', nodes)

      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testSplitWithNonConstAxis(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Splits the convolution output in two along an axis fed through a placeholder (value
    # three) and adds two batch-normalized halves. Asserts two transposes, one in from the convolution and one
    # out at the add, a dimension-map node for the split axis input, and matching outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      dim = array_ops.placeholder(dtype='int32')
      split = array_ops.split(conv, 2, axis=dim)
      scale = constant_op.constant(0.1, shape=[32])
      offset = constant_op.constant(0.3, shape=[32])
      bn0 = nn.fused_batch_norm(split[0], scale, offset)
      bn1 = nn.fused_batch_norm(split[1], scale, offset)
      add = bn0[0] + bn1[0]
      output = array_ops.identity(add)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={dim: 3})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata, feed_dict={dim: 3})

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('add_2-0-0', nodes)
      self._assert_map_nhwc_to_nchw('split-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testSplitVWithNonConstAxis(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Splits the convolution output with the variable-size split op using a constant sizes
    # vector and a placeholder axis (fed three), then reduces the first piece to a sum. Asserts two transposes,
    # a transpose out of the split, a dimension-map node for the axis and matching outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      dim = array_ops.placeholder(dtype='int32')
      sizes = constant_op.constant([50, 10, 4], shape=[3])
      split = gen_array_ops.split_v(
          value=conv, size_splits=sizes, axis=dim, num_split=3)
      output = math_ops.reduce_sum(split[0])

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={dim: 3})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata, feed_dict={dim: 3})

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('SplitV-0-0', nodes)
      self._assert_map_nhwc_to_nchw('SplitV-2', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testPadWithConstPaddings(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Pads the convolution output with constant paddings for the four dimensions. Asserts
    # two transposes, a transpose back after the pad, that the constant paddings input was rewritten (its name
    # carries the layout optimizer suffix) and that outputs match.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      paddings_val = [[1, 2], [3, 4], [5, 6], [7, 8]]
      paddings = constant_op.constant(
          paddings_val, dtype='int32', name='PaddingsConst')
      pad = array_ops.pad(conv, paddings)
      output = array_ops.identity(pad)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Pad-0-0', nodes)
      self.assertIn('Pad-1-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testReduceSum(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Sums the whole convolution output over all axes. Asserts only one transpose (into
    # channels-first) because the scalar result needs no transpose back, and that outputs match.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv)
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testCast(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Casts the convolution output to boolean. Asserts two transposes with the transpose
    # back occurring after the cast node, and that outputs match.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      cast = math_ops.cast(conv, dtype='bool')
      output = array_ops.identity(cast)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Cast-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testSqueeze(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Reduces the convolution output over the height and width axes without keeping
    # dimensions, then squeezes all size-one axes. Asserts a single transpose in and none out, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[1, 2])
      squeeze = array_ops.squeeze(reduce_sum)
      output = array_ops.identity(squeeze)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testSqueezeAlongHW(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Reduces over height and width keeping dimensions, then squeezes exactly those two
    # axes. Asserts one transpose in (none back) and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[1, 2], keepdims=True)
      squeeze = array_ops.squeeze(reduce_sum, axis=[1, 2])
      output = array_ops.identity(squeeze)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testSqueezeAlongNHW(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Reduces over batch, height and width keeping dimensions, then squeezes those three
    # axes. Asserts one transpose in (none back) and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[0, 1, 2], keepdims=True)
      squeeze = array_ops.squeeze(reduce_sum, axis=[0, 1, 2])
      output = array_ops.identity(squeeze)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testReduceSumAlongHWC(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Sums over height, width and channel axes. Asserts one transpose and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[1, 2, 3])
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testReduceSumAlongNHW(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Sums over batch, height and width axes. Asserts one transpose and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[0, 1, 2])
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testReduceSumAlongC(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Sums over the channel axis only, without keeping dimensions. Asserts one transpose
    # and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[3])
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Three transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testReduceSumAlongCKeepDims(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Sums over the channel axis keeping dimensions, so the result is still 4-D. Asserts two
    # transposes, one transposed back after the sum node, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[3], keepdims=True)
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Sum-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testReduceSumAlongHKeepDims(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Sums over the height axis keeping dimensions. Asserts two transposes, a transpose
    # into channels-first after the convolution, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[2], keepdims=True)
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testReduceSumAlongWCKeepDims(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Sums over width and channel axes keeping dimensions. Asserts two transposes, a
    # transpose into channels-first after the convolution, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      reduce_sum = math_ops.reduce_sum(conv, axis=[2, 3], keepdims=True)
      output = array_ops.identity(reduce_sum)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testConcatWithControlDependency(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Concatenates the convolution output with itself along the channel axis while the
    # concat has a control dependency on a variable assignment. Asserts two transposes, a transpose back after
    # the concat, that the constant axis input was rewritten, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      axis = constant_op.constant(3)
      var = variables.Variable(3)
      assign = state_ops.assign(var, 6)
      with ops.control_dependencies([assign]):
        concat = array_ops.concat([conv, conv], axis)
      output = array_ops.identity(concat)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('concat-0-0', nodes)
      self.assertIn('concat-2-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testConcatWithControlDependencyFor5DTensor(self):
    # [FUNCTION DOC]
    # Scenario: skipped when no GPU is available. Same as the concat control-dependency test but for a 5-D
    # tensor from a 3-D convolution, concatenated along the last axis. Asserts two transposes, 5-D transposes
    # in and out, a dimension-map node for the axis, and equal outputs.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')
    random_seed.set_random_seed(0)
    x = random_ops.truncated_normal([2, 2, 14, 14, 1], seed=0)
    w = random_ops.truncated_normal([2, 2, 2, 1, 2], seed=0)
    strides = [1, 1, 1, 1, 1]
    y = gen_nn_ops.conv3d(x, w, strides, 'SAME')
    axis = constant_op.constant(4)
    var = variables.Variable(3)
    assign = state_ops.assign(var, 6)
    with ops.control_dependencies([assign]):
      concat = array_ops.concat([y, y], axis)
    output = array_ops.identity(concat)

    with session.Session(config=_get_config(False)) as sess:
      output_val_ref = self.evaluate(output)

    with session.Session(config=_get_config()) as sess:
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(output, run_metadata=metadata)

    nodes = []
    num_transposes = 0
    for node in metadata.cost_graph.node:
      if _is_transpose(node.name):
        num_transposes += 1
      nodes.append(node.name)

    # Four transposes were initially added in the Expand phase of
    # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
    expected_num_transposes = 2
    self.assertEqual(expected_num_transposes, num_transposes)
    self._assert_trans_ndhwc_to_ncdhw('Conv3D-0', nodes)
    self._assert_trans_ncdhw_to_ndhwc('concat-0-0', nodes)
    self._assert_map_ndhwc_to_ncdhw('concat-2', nodes)
    self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testFill(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Builds a fill whose shape comes from the shape of the convolution output (input fed as
    # a list of 784 values). Asserts two transposes, zero vector-permute nodes (they cancel), a transpose back
    # after the fill, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = array_ops.placeholder(dtype='float32')
      conv = _two_layer_model(x)
      shape = array_ops.shape(conv)
      scalar = array_ops.constant(5.7)
      fill = array_ops.fill(shape, scalar)
      output = array_ops.identity(fill)

      x_val = [3.4] * 784
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={x: x_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                x: x_val
            })

      nodes = []
      num_transposes = 0
      num_vec_permute = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        if _is_permute(node.name):
          num_vec_permute += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      # Two vector permute nodes were initially added in the Expand phase of
      # LayoutOptimizer; they cancelled out each other in the Collapse phase.
      expected_vec_permute = 0
      self.assertEqual(expected_vec_permute, num_vec_permute)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Fill-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testTile(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Tiles the convolution output by a multiples vector fed through a placeholder. Asserts
    # two transposes, a transpose back after the tile, a vector-permute node on the multiples input, and equal
    # outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      multiple = array_ops.placeholder(dtype='int32')
      tile = array_ops.tile(conv, multiple)
      output = array_ops.identity(tile)

      multiple_val = [2, 3, 4, 1]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={multiple: multiple_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                multiple: multiple_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Tile-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('Tile-1', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testReverseWithConstDims(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Reverses the convolution output along constant dimensions. Asserts two transposes, a
    # transpose back after the reverse, that the dimensions input was rewritten, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      dims = constant_op.constant([3, 1], name='DimsConst')
      reverse = array_ops.reverse(conv, dims)
      output = array_ops.identity(reverse)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('ReverseV2-0-0', nodes)
      self.assertIn('ReverseV2-1-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testReverseWithNonConstDims(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Reverses along a dimensions list fed through a placeholder. Asserts two transposes, a
    # transpose back after the reverse, a dimension-map node for the dims input, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      dims = array_ops.placeholder(dtype='int32')
      reverse = array_ops.reverse(conv, dims)
      output = array_ops.identity(reverse)

      dims_val = [2, 3]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={dims: dims_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                dims: dims_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('ReverseV2-0-0', nodes)
      self._assert_map_nhwc_to_nchw('ReverseV2-1', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testSelectOp(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Selects between the convolution output and its doubled value using a condition tensor
    # derived by comparing with the mean. Asserts two transposes, a transpose back after the select, and equal
    # outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      add = math_ops.add(conv, conv)
      mean = math_ops.reduce_mean(conv)
      condition = math_ops.less(conv, mean)
      select = gen_math_ops.select(condition, conv, add)
      output = array_ops.identity(select)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Select-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testSelectOpConditionUnknownShape(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Same select but the condition is a boolean placeholder of unknown shape fed with
    # zeros. Asserts three transposes (the condition cannot be cancelled), a transpose into channels-first
    # after the convolution, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      add = math_ops.add(conv, conv)
      condition = array_ops.placeholder(dtype='bool')
      select = gen_math_ops.select(condition, conv, add)
      output = array_ops.identity(select)

      condition_val = np.zeros((1, 7, 7, 64))
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={condition: condition_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={condition: condition_val})

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 3
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testSelectOpScalarCondition(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Select with a scalar constant true condition. Asserts two transposes, a transpose back
    # after the select, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      add = math_ops.add(conv, conv)
      condition = constant_op.constant(True)
      select = gen_math_ops.select(condition, conv, add)
      output = array_ops.identity(select)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Select-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testPadWithNonConstPaddings(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Pads the convolution output with a paddings tensor fed through a placeholder. Asserts
    # two transposes, a transpose back after the pad, a vector-permute node on the paddings input, and equal
    # outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      paddings = array_ops.placeholder(dtype='int32')
      pad = array_ops.pad(conv, paddings)
      output = array_ops.identity(pad)

      paddings_val = [[1, 2], [3, 4], [5, 6], [7, 8]]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={paddings: paddings_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                paddings: paddings_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Pad-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('Pad-1', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testMaxPoolV2(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Applies the variant of max pooling with a constant window size and strides fed through
    # a placeholder, valid padding. Asserts two transposes, a transpose back, a vector-permute node on the
    # strides input, that the constant window-size input was rewritten, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      ksize = constant_op.constant([1, 2, 3, 1], shape=[4])
      strides = array_ops.placeholder(dtype='int32', shape=[4])
      max_pool = gen_nn_ops.max_pool_v2(conv, ksize, strides, 'VALID')
      output = array_ops.identity(max_pool)

      strides_val = [1, 3, 2, 1]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={strides: strides_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                strides: strides_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('MaxPoolV2-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('MaxPoolV2-2', nodes)
      self.assertIn('MaxPoolV2-1-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testMaxPoolGradV2(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Applies the gradient of the variant max pool using the convolution output for all
    # three tensor inputs, a constant window size and fed strides. Asserts two transposes, a transpose back,
    # a vector-permute node on the strides input, a rewritten window-size input, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      ksize = constant_op.constant([1, 2, 3, 1], shape=[4])
      strides = array_ops.placeholder(dtype='int32', shape=[4])
      max_pool_grad = gen_nn_ops.max_pool_grad_v2(conv, conv, conv, ksize,
                                                  strides, 'VALID')
      output = array_ops.identity(max_pool_grad)

      strides_val = [1, 3, 2, 1]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={strides: strides_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                strides: strides_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('MaxPoolGradV2-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('MaxPoolGradV2-4', nodes)
      self.assertIn('MaxPoolGradV2-3-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testLeakyRelu(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Convolves a random 4 by 14 by 14 by 1 tensor and applies leaky ReLU with slope 0.2.
    # Asserts two transposes, a transpose back after the leaky ReLU, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([4, 14, 14, 1], seed=0)
      w = random_ops.truncated_normal([2, 2, 1, 2], seed=0)
      y = nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
      y = nn.leaky_relu(y, alpha=0.2)
      output = array_ops.identity(y)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nchw_to_nhwc('LeakyRelu-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testLeakyReluGrad(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Convolves a random tensor and feeds the result and the original input to the leaky
    # ReLU gradient op. Asserts three transposes, a transpose into channels-first for the second input, a
    # transpose back at the output, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([4, 14, 14, 1], seed=0)
      w = random_ops.truncated_normal([2, 2, 1, 1], seed=0)
      y = nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
      y = gen_nn_ops.leaky_relu_grad(y, x, alpha=0.2)
      output = array_ops.identity(y)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 3
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('LeakyReluGrad-1', nodes)
      self._assert_trans_nchw_to_nhwc('LeakyReluGrad-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testLeakyReluGradFor5DTensors(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Same as the 4-D leaky ReLU gradient test but with a 3-D convolution and 5-D tensors.
    # Asserts three transposes using the 5-D formats and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 4, 2, 3, 3], seed=0)
      w = random_ops.truncated_normal([2, 2, 2, 3, 3], seed=0)
      y = gen_nn_ops.conv3d(x, w, [1, 1, 1, 1, 1], 'SAME')
      y = gen_nn_ops.leaky_relu_grad(y, x, alpha=0.2)
      output = array_ops.identity(y)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 3
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_ndhwc_to_ncdhw('LeakyReluGrad-1', nodes)
      self._assert_trans_ncdhw_to_ndhwc('LeakyReluGrad-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testReduceOpsFor5DTensors(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Applies a mean reduction over the first four axes with kept dimensions to a 3-D
    # convolution output. Asserts two transposes (in and out), a dimension-map node for the reduction axes, and
    # equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 4, 2, 3, 3], seed=0)
      w = random_ops.truncated_normal([2, 2, 2, 3, 3], seed=0)
      conv3d = gen_nn_ops.conv3d(x, w, [1, 1, 1, 1, 1], 'SAME')
      y = math_ops.reduce_mean(conv3d, [0, 1, 2, 3], keepdims=True)
      output = array_ops.identity(y)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # The reduce op Mean needs to dim map the input reduce index to NCDHW.
      # Then, the output needs to be transposed back to NDHWC.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_ndhwc_to_ncdhw('Conv3D-0', nodes)
      self._assert_map_ndhwc_to_ncdhw('Mean-1', nodes)
      self._assert_trans_ncdhw_to_ndhwc('Mean-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testBinaryOpsFor5DTensors(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the sibling tests it never compares the optimized output with the reference output;
    # the reference and optimized runs are executed but their values are not asserted equal.
    # Scenario: GPU only. Applies manually written batch normalization (multiply and add with broadcast
    # parameters) to a 3-D convolution output. Asserts four transposes: one after the convolution, one more
    # on each of the multiply and add nodes for the other operand, and one transpose back at the end.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 4, 2, 3, 3], seed=0)
      w = random_ops.truncated_normal([2, 2, 2, 3, 3], seed=0)
      mean = random_ops.truncated_normal([1, 1, 1, 1, 3], seed=0)
      variance = random_ops.truncated_normal([1, 1, 1, 1, 3], seed=0)
      gamma = random_ops.truncated_normal([1, 1, 1, 1, 3], seed=0)
      beta = random_ops.truncated_normal([1, 1, 1, 1, 3], seed=0)
      conv3d = gen_nn_ops.conv3d(x, w, [1, 1, 1, 1, 1], 'SAME')
      y = nn.batch_normalization(
          conv3d,
          mean=mean,
          variance=variance,
          scale=gamma,
          offset=beta,
          variance_epsilon=0.001)
      output = array_ops.identity(y)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # The binary ops mul_1 and add_1 in batch norm need to transpose one of
      # the two inputs to NCDHW. The other input has already been transposed via
      # Conv3D.
      expected_num_transposes = 4
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_ndhwc_to_ncdhw('Conv3D-0', nodes)
      self._assert_trans_ndhwc_to_ncdhw('batchnorm/mul_1-1', nodes)
      self._assert_trans_ndhwc_to_ncdhw('batchnorm/add_1-1', nodes)
      self._assert_trans_ncdhw_to_ndhwc('batchnorm/add_1-0-0', nodes)

  @test_util.deprecated_graph_mode_only
  def testBatchNorm3D(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Applies fused batch normalization with the 5-D channels-last format to a 3-D
    # convolution output. Asserts two transposes (in and out around the fused op) and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x_3d = random_ops.truncated_normal([1, 4, 2, 3, 3], seed=0)
      filters = random_ops.truncated_normal([2, 2, 2, 3, 3], seed=0)
      strides_val = [1, 1, 1, 1, 1]
      scale = constant_op.constant(0.1, shape=[3])
      offset = constant_op.constant(0.3, shape=[3])
      conv3d = gen_nn_ops.conv3d(x_3d, filters, strides_val, 'SAME')
      y, _, _ = nn.fused_batch_norm(conv3d, scale, offset, data_format='NDHWC')
      output = array_ops.identity(y)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_ndhwc_to_ncdhw('Conv3D-0', nodes)
      self._assert_trans_ncdhw_to_ndhwc('FusedBatchNormV3-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testBatchNormGrad3D(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Runs fused batch normalization in training mode on a 3-D convolution output and then
    # its gradient op using the saved reserve-space values; takes the input gradient as output. Asserts three
    # transposes (convolution in, gradient second input in, gradient output out) and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x_3d = random_ops.truncated_normal([1, 4, 2, 3, 3], seed=0)
      filters = random_ops.truncated_normal([2, 2, 2, 3, 3], seed=0)
      strides_val = [1, 1, 1, 1, 1]
      scale = constant_op.constant(0.1, shape=[3])
      offset = constant_op.constant(0.3, shape=[3])
      mean = constant_op.constant(0.1, shape=[3])
      variance = constant_op.constant(0.3, shape=[3])
      conv3d = gen_nn_ops.conv3d(x_3d, filters, strides_val, 'SAME')
      y, running_mean, running_var, r0, r1, r2 = gen_nn_ops.fused_batch_norm_v3(
          conv3d,
          scale,
          offset,
          mean,
          variance,
          epsilon=1.001e-5,
          exponential_avg_factor=1.0,
          data_format='NDHWC',
          is_training=True,
          name='batch_norm')
      dx, dscale, doffset, _, _ = gen_nn_ops.fused_batch_norm_grad_v3(
          y,
          x_3d,
          scale,
          r0,
          r1,
          r2,
          epsilon=1.001e-5,
          data_format='NDHWC',
          is_training=True)
      output = array_ops.identity(dx)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 3
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_ndhwc_to_ncdhw('Conv3D-0', nodes)
      self._assert_trans_ndhwc_to_ncdhw('FusedBatchNormGradV3-1', nodes)
      self._assert_trans_ncdhw_to_ndhwc('FusedBatchNormGradV3-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testConv3D(self):
    # [FUNCTION DOC]
    # Scenario: skipped without a GPU. Applies a 3-D convolution to a random 5-D tensor. Asserts two transposes
    # using the 5-D formats (one in, one out) and equal outputs.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')
    random_seed.set_random_seed(0)
    x = random_ops.truncated_normal([2, 2, 14, 14, 1], seed=0)
    w = random_ops.truncated_normal([2, 2, 2, 1, 2], seed=0)
    strides = [1, 1, 1, 1, 1]
    y = gen_nn_ops.conv3d(x, w, strides, 'SAME')
    output = array_ops.identity(y)

    with session.Session(config=_get_config(False)) as sess:
      output_val_ref = sess.run(output)

    with session.Session(config=_get_config()) as sess:
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(output, run_metadata=metadata)

    nodes = []
    num_transposes = 0
    for node in metadata.cost_graph.node:
      if _is_transpose(node.name):
        num_transposes += 1
      nodes.append(node.name)

    expected_num_transposes = 2
    self.assertEqual(expected_num_transposes, num_transposes)
    self._assert_trans_ndhwc_to_ncdhw('Conv3D-0', nodes)
    self._assert_trans_ncdhw_to_ndhwc('Conv3D-0-0', nodes)
    self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testConv3DBackpropInput(self):
    # [FUNCTION DOC]
    # Scenario: skipped without a GPU. Computes the input gradient of a 3-D convolution with the shape taken
    # from a tensor. Asserts two transposes, a vector-permute node on the shape input, a transpose of the
    # output-gradient input and a transpose back at the result, and equal outputs.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')
    random_seed.set_random_seed(0)
    dy = random_ops.truncated_normal([2, 2, 14, 14, 1], seed=0)
    w = random_ops.truncated_normal([2, 2, 2, 1, 1], seed=0)
    strides = [1, 1, 1, 1, 1]
    x_shape = array_ops.shape(dy)
    dx = gen_nn_ops.conv3d_backprop_input_v2(x_shape, w, dy, strides, 'SAME')
    output = array_ops.identity(dx)

    with session.Session(config=_get_config(False)) as sess:
      output_val_ref = sess.run(output)

    with session.Session(config=_get_config()) as sess:
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(output, run_metadata=metadata)

    nodes = []
    num_transposes = 0
    for node in metadata.cost_graph.node:
      if _is_transpose(node.name):
        num_transposes += 1
      nodes.append(node.name)

    expected_num_transposes = 2
    self.assertEqual(expected_num_transposes, num_transposes)
    self._assert_vec_ndhwc_to_ncdhw('Conv3DBackpropInputV2-0', nodes)
    self._assert_trans_ndhwc_to_ncdhw('Conv3DBackpropInputV2-2', nodes)
    self._assert_trans_ncdhw_to_ndhwc('Conv3DBackpropInputV2-0-0', nodes)
    self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testConv3DBackpropFilter(self):
    # [FUNCTION DOC]
    # Scenario: skipped without a GPU. Computes the filter gradient of a 3-D convolution with a constant filter
    # shape. Asserts two transposes, one for each of the two tensor inputs (the filter-gradient result is not
    # transposed back), and equal outputs.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')
    random_seed.set_random_seed(0)
    x = random_ops.truncated_normal([2, 2, 14, 14, 1], seed=0)
    dy = random_ops.truncated_normal([2, 2, 14, 14, 1], seed=0)
    strides = [1, 1, 1, 1, 1]
    w_shape = constant_op.constant([2, 2, 2, 1, 1], shape=[5])
    dw = gen_nn_ops.conv3d_backprop_filter_v2(x, w_shape, dy, strides, 'SAME')
    output = array_ops.identity(dw)

    with session.Session(config=_get_config(False)) as sess:
      output_val_ref = sess.run(output)

    with session.Session(config=_get_config()) as sess:
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(output, run_metadata=metadata)

    nodes = []
    num_transposes = 0
    for node in metadata.cost_graph.node:
      if _is_transpose(node.name):
        num_transposes += 1
      nodes.append(node.name)

    expected_num_transposes = 2
    self.assertEqual(expected_num_transposes, num_transposes)
    self._assert_trans_ndhwc_to_ncdhw('Conv3DBackpropFilterV2-0', nodes)
    self._assert_trans_ndhwc_to_ncdhw('Conv3DBackpropFilterV2-2', nodes)
    self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testBiasAddFor5DTensor(self):
    # [FUNCTION DOC]
    # Scenario: skipped without a GPU. Applies bias add with channels-last format to a 3-D convolution output.
    # Asserts two transposes (in after the convolution, out after the bias add) and equal outputs.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')
    random_seed.set_random_seed(0)
    x = random_ops.truncated_normal([2, 2, 14, 14, 1], seed=0)
    w = random_ops.truncated_normal([2, 2, 2, 1, 2], seed=0)
    b = random_ops.truncated_normal([2], seed=0)
    strides = [1, 1, 1, 1, 1]
    y = gen_nn_ops.conv3d(x, w, strides, 'SAME')
    y = gen_nn_ops.bias_add(y, b, 'NHWC')
    output = array_ops.identity(y)

    with session.Session(config=_get_config(False)) as sess:
      output_val_ref = sess.run(output)

    with session.Session(config=_get_config()) as sess:
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(output, run_metadata=metadata)

    nodes = []
    num_transposes = 0
    for node in metadata.cost_graph.node:
      if _is_transpose(node.name):
        num_transposes += 1
      nodes.append(node.name)

    expected_num_transposes = 2
    self.assertEqual(expected_num_transposes, num_transposes)
    self._assert_trans_ndhwc_to_ncdhw('Conv3D-0', nodes)
    self._assert_trans_ncdhw_to_ndhwc('BiasAdd-0-0', nodes)
    self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testBiasAddGradFor5DTensor(self):
    # [FUNCTION DOC]
    # Scenario: skipped without a GPU. Applies the bias-add gradient to a 3-D convolution input gradient.
    # Asserts a single transpose (the convolution input gradient output is not converted back), a
    # vector-permute node on its shape input, a transpose on its output-gradient input, and equal outputs.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')
    random_seed.set_random_seed(0)
    dy = random_ops.truncated_normal([2, 2, 14, 14, 1], seed=0)
    w = random_ops.truncated_normal([2, 2, 2, 1, 1], seed=0)
    strides = [1, 1, 1, 1, 1]
    dy_shape = array_ops.shape(dy)
    dx = gen_nn_ops.conv3d_backprop_input_v2(dy_shape, w, dy, strides, 'SAME')
    db = gen_nn_ops.bias_add_grad(dx, 'NHWC')
    output = array_ops.identity(db)

    with session.Session(config=_get_config(False)) as sess:
      output_val_ref = sess.run(output)

    with session.Session(config=_get_config()) as sess:
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(output, run_metadata=metadata)

    nodes = []
    num_transposes = 0
    for node in metadata.cost_graph.node:
      if _is_transpose(node.name):
        num_transposes += 1
      nodes.append(node.name)

    # The output of Conv3DBackpropInputV2 won't be converted back to NDHWC
    # because of the BiasAddGrad.
    expected_num_transposes = 1
    self.assertEqual(expected_num_transposes, num_transposes)
    self._assert_vec_ndhwc_to_ncdhw('Conv3DBackpropInputV2-0', nodes)
    self._assert_trans_ndhwc_to_ncdhw('Conv3DBackpropInputV2-2', nodes)
    self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testSliceWithNonConstAxis(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Slices the convolution output with a size vector fed through a placeholder. Asserts
    # two transposes, a transpose back after the slice, a vector-permute node on the size input, and equal
    # outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      size = array_ops.placeholder(dtype='int32')
      s = array_ops.slice(conv, [0, 0, 0, 0], size)
      output = array_ops.identity(s)

      size_val = [1, 2, 3, 4]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={size: size_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                size: size_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('Slice-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('Slice-2', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testSliceWithNonConstAxisFor5DTensor(self):
    # [FUNCTION DOC]
    # Scenario: skipped without a GPU. Same as the non-constant slice test but on a 5-D tensor from a 3-D
    # convolution. Asserts two transposes using 5-D formats, a 5-D vector-permute node on the size input, and
    # equal outputs.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')
    random_seed.set_random_seed(0)
    x = random_ops.truncated_normal([2, 2, 14, 14, 1], seed=0)
    w = random_ops.truncated_normal([2, 2, 2, 1, 2], seed=0)
    strides = [1, 1, 1, 1, 1]
    y = gen_nn_ops.conv3d(x, w, strides, 'SAME')
    size = array_ops.placeholder(dtype='int32')
    s = array_ops.slice(y, [0, 0, 0, 0, 0], size)
    output = array_ops.identity(s)

    size_val = [1, 1, 2, 2, 1]
    with session.Session(config=_get_config(False)) as sess:
      output_val_ref = sess.run(output, feed_dict={size: size_val})

    with session.Session(config=_get_config()) as sess:
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(
          output, run_metadata=metadata, feed_dict={size: size_val})

    nodes = []
    num_transposes = 0
    for node in metadata.cost_graph.node:
      if _is_transpose(node.name):
        num_transposes += 1
      nodes.append(node.name)

    # Four transposes were initially added in the Expand phase of
    # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
    expected_num_transposes = 2
    self.assertEqual(expected_num_transposes, num_transposes)
    self._assert_trans_ndhwc_to_ncdhw('Conv3D-0', nodes)
    self._assert_trans_ncdhw_to_ndhwc('Slice-0-0', nodes)
    self._assert_vec_ndhwc_to_ncdhw('Slice-2', nodes)
    self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testStridedSliceWithNonConstAxis(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Strided slice with constant begin and strides and an end vector fed through a
    # placeholder. Asserts two transposes, a transpose back, a vector-permute node on the end input, rewritten
    # constant begin and strides inputs, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      end = array_ops.placeholder(dtype='int32')
      s = array_ops.strided_slice(conv, [0, 0, 0, 0], end, strides=[1, 2, 3, 1])
      output = array_ops.identity(s)

      end_val = [1, 2, 3, 4]
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={end: end_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                end: end_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('StridedSlice-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('StridedSlice-2', nodes)
      self.assertIn('StridedSlice-1-LayoutOptimizer', nodes)
      self.assertIn('StridedSlice-3-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testStridedSliceWithMask1011(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Slices the convolution output with index syntax that produces begin and end masks
    # of binary 1011. Asserts two transposes, a transpose back, rewritten begin, end and strides inputs, and
    # equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      # This will generate a StridedSlice op with begin mask and
      # end mask 11(1011).
      s = conv[:, :, 1:-1, :]
      output = array_ops.identity(s)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('strided_slice-0-0', nodes)
      self.assertIn('strided_slice-1-LayoutOptimizer', nodes)
      self.assertIn('strided_slice-2-LayoutOptimizer', nodes)
      self.assertIn('strided_slice-3-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testStridedSliceWithMask0111(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Slices the last dimension of the convolution output, producing masks of binary 0111.
    # Asserts two transposes, a transpose back, rewritten begin, end and strides inputs, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      # This will generate a StridedSlice op with begin mask and
      # end mask 7(0111).
      s = conv[:, :, :, 1:-1]
      output = array_ops.identity(s)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('strided_slice-0-0', nodes)
      self.assertIn('strided_slice-1-LayoutOptimizer', nodes)
      self.assertIn('strided_slice-2-LayoutOptimizer', nodes)
      self.assertIn('strided_slice-3-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testStridedSliceGradWithNonConstAxis(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Builds a strided slice with constant end and then the strided-slice gradient using
    # the convolution shape and an end vector fed through a placeholder. Asserts two transposes, a transpose
    # back, a vector-permute node on the gradient end input, rewritten inputs of the forward slice, and equal
    # outputs. The forward-slice rewritten-input assertions refer to the forward slice node names.
    if test.is_gpu_available(cuda_only=True):
      random_seed.set_random_seed(0)
      x = random_ops.truncated_normal([1, 784], seed=0)
      conv = _two_layer_model(x)
      end = array_ops.placeholder(dtype='int32')
      shape = array_ops.shape(conv)
      end_val = [1, 2, 3, 4]
      s = array_ops.strided_slice(
          conv, [0, 0, 0, 0], end_val, strides=[1, 2, 3, 1])
      s_grad = array_ops.strided_slice_grad(shape, [0, 0, 0, 0], end,
                                            [1, 2, 3, 1], s)
      output = array_ops.identity(s_grad)

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={end: end_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                end: end_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('StridedSliceGrad-0-0', nodes)
      self._assert_vec_nhwc_to_nchw('StridedSliceGrad-2', nodes)
      self.assertIn('StridedSlice-1-LayoutOptimizer', nodes)
      self.assertIn('StridedSlice-2-LayoutOptimizer', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testShapeN(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Calls the shape-of-many op on two copies of the convolution output (input fed
    # as 784 values) and adds the two shape vectors. Asserts one transpose, a vector-permute node (channels-first
    # to channels-last) on the first shape output, and exactly equal outputs.
    if test.is_gpu_available(cuda_only=True):
      x = array_ops.placeholder(dtype='float32')
      conv = _two_layer_model(x)
      shapen = array_ops.shape_n([conv, conv])
      output = math_ops.add(shapen[0], shapen[1])

      x_val = [1.7] * 784
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={x: x_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={
                x: x_val
            })

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 1
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self._assert_vec_nchw_to_nhwc('ShapeN-0-0', nodes)
      self.assertAllEqual(output_val_ref, output_val)

  @test_util.deprecated_graph_mode_only
  def testShapeNFor5DTensor(self):
    # [FUNCTION DOC]
    # Scenario: skipped without a GPU. Same as the shape-of-many test but on a 5-D tensor from a 3-D
    # convolution, using a reshaped placeholder. Asserts one transpose, 5-D vector-permute nodes on both shape
    # outputs, and equal outputs.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')
    h = array_ops.placeholder(dtype='float32')
    x = array_ops.reshape(h, [-1, 2, 14, 14, 1])
    w = random_ops.truncated_normal([2, 2, 2, 1, 2], seed=0)
    strides = [1, 1, 1, 1, 1]
    y = gen_nn_ops.conv3d(x, w, strides, 'SAME')
    shapen = array_ops.shape_n([y, y])
    output = math_ops.add(shapen[0], shapen[1])

    x_val = [1.7] * 784
    with session.Session(config=_get_config(False)) as sess:
      output_val_ref = sess.run(output, feed_dict={h: x_val})

    with session.Session(config=_get_config()) as sess:
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(output, run_metadata=metadata, feed_dict={h: x_val})

    nodes = []
    num_transposes = 0
    for node in metadata.cost_graph.node:
      if _is_transpose(node.name):
        num_transposes += 1
      nodes.append(node.name)

    expected_num_transposes = 1
    self.assertEqual(expected_num_transposes, num_transposes)
    self._assert_trans_ndhwc_to_ncdhw('Conv3D-0', nodes)
    self._assert_vec_ncdhw_to_ndhwc('ShapeN-0-0', nodes)
    self._assert_vec_ncdhw_to_ndhwc('ShapeN-1-0', nodes)
    self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testIdentityNFor4DAnd5DTensors(self):
    # [FUNCTION DOC]
    # Scenario: skipped without a GPU. Builds both a 4-D convolution model and a 3-D convolution result, passes
    # them together through the multi-input identity op, reshapes both and adds them. Asserts four transposes
    # (two in, two out, in 4-D and 5-D variants) and equal outputs.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')
    h = array_ops.placeholder(dtype='float32')
    x = array_ops.reshape(h, [-1, 2, 14, 14, 1])
    w = random_ops.truncated_normal([2, 2, 2, 1, 4], seed=0)
    strides = [1, 1, 1, 1, 1]
    y = gen_nn_ops.conv3d(x, w, strides, 'SAME')
    x1 = array_ops.reshape(h, [-1, 784])
    y1 = _two_layer_model(x1)
    outputs = array_ops.identity_n([y1, y])
    new_x0 = array_ops.reshape(outputs[0], [-1, 2, 14, 14, 1])
    new_x1 = array_ops.reshape(outputs[1], [-1, 2, 14, 14, 1])
    output = math_ops.add(new_x0, new_x1)

    x_val = [1.7] * 784
    with session.Session(config=_get_config(False)) as sess:
      output_val_ref = sess.run(output, feed_dict={h: x_val})

    with session.Session(config=_get_config()) as sess:
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(output, run_metadata=metadata, feed_dict={h: x_val})

    nodes = []
    num_transposes = 0
    for node in metadata.cost_graph.node:
      if _is_transpose(node.name):
        num_transposes += 1
      nodes.append(node.name)

    expected_num_transposes = 4
    self.assertEqual(expected_num_transposes, num_transposes)
    self._assert_trans_ndhwc_to_ncdhw('Conv3D-0', nodes)
    self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
    self._assert_trans_ncdhw_to_ndhwc('IdentityN-1-0', nodes)
    self._assert_trans_nchw_to_nhwc('IdentityN-0-0', nodes)
    self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testShapeNFollowedByNotConvertibleNodeReshape(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Shape-of-many is applied to the convolution output and a reshaped copy, and the second
    # shape feeds a ones op whose result is added to the reshaped tensor. The reshape cannot be converted.
    # Asserts two transposes, a transpose into channels-first after the convolution, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      x = array_ops.placeholder(dtype='float32')
      conv = _two_layer_model(x)
      conv_reshape = array_ops.reshape(conv, [1, 1, 1, -1])
      shapen = array_ops.shape_n([conv, conv_reshape])
      shape = array_ops.identity(shapen[1])
      ones = array_ops.ones(shape)
      output = math_ops.add_n([conv_reshape, ones])

      x_val = [1.7] * 784
      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = sess.run(output, feed_dict={x: x_val})

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(
            output, run_metadata=metadata, feed_dict={x: x_val})

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('Conv2D-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testLoop(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the assertion on the transpose count appears twice in a row (duplicate line); the second is
    # redundant and harmless.
    # Scenario: GPU only. Maps the two-layer model over four inputs in a while loop. Asserts two transposes, a
    # transpose into channels-first at the loop convolution, a transpose back after the final pooling in the
    # loop, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      output = _loop()

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      # Four transposes were initially added in the Expand phase of
      # LayoutOptimizer; two of them are cancelled out in the Collapse phase.
      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('map/while/Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('map/while/MaxPool_1-0-2', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testLoopWithBranch(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Maps the two-branch convolution model in a loop. Asserts three transposes, a
    # transpose in at the loop convolution, a transpose back after the add in the loop, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      output = _loop_with_branch()

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 3
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('map/while/Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('map/while/Add_1-0-2', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testLoopWithVecAnd4D(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Maps the model that adds a constant vector to a convolution output in a loop. Asserts
    # two transposes, a transpose in at the loop convolution, a transpose back after the add, and equal outputs.
    if test.is_gpu_available(cuda_only=True):
      output = _loop_with_vec_and_4d()

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('map/while/Conv2D-0', nodes)
      self._assert_trans_nchw_to_nhwc('map/while/Add_1-0-2', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testBinaryOpSecondPort(self):
    # [FUNCTION DOC]
    # Scenario: GPU only. Adds the fused batch norm output to its second output port (the mean). Asserts two
    # transposes, a transpose into channels-first at the fused batch norm, a transpose back after the add, and
    # equal outputs.
    if test.is_gpu_available(cuda_only=True):
      output = _model_with_second_port()

      with session.Session(config=_get_config(False)) as sess:
        output_val_ref = self.evaluate(output)

      with session.Session(config=_get_config()) as sess:
        metadata = config_pb2.RunMetadata()
        output_val = sess.run(output, run_metadata=metadata)

      nodes = []
      num_transposes = 0
      for node in metadata.cost_graph.node:
        if _is_transpose(node.name):
          num_transposes += 1
        nodes.append(node.name)

      expected_num_transposes = 2
      self.assertEqual(expected_num_transposes, num_transposes)
      self._assert_trans_nhwc_to_nchw('FusedBatchNormV3-0', nodes)
      self._assert_trans_nchw_to_nhwc('Add-0-0', nodes)
      self.assertAllClose(output_val_ref, output_val, atol=1e-3)

  @test_util.deprecated_graph_mode_only
  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario: builds a two-convolution training metagraph and optimizes it with the layout optimizer on and no
    # minimum node count, using a fake GPU cluster (no real GPU needed). Counts convolution, filter-gradient and
    # input-gradient nodes and asserts each uses the channels-first data format and that there are exactly five.
    meta_graph = _simple_metagraph()
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(
        rewriter_config_pb2.RewriterConfig(
            layout_optimizer=rewriter_config_pb2.RewriterConfig.ON,
            min_graph_nodes=-1))
    optimized_graph = tf_optimizer.OptimizeGraph(
        config, meta_graph, cluster=_get_cluster())

    found = 0
    for node in optimized_graph.node:
      if node.op in ['Conv2D', 'Conv2DBackpropFilter', 'Conv2DBackpropInput']:
        found += 1
        self.assertEqual(node.attr['data_format'].s, b'NCHW')
    self.assertEqual(found, 5)

  @test_util.deprecated_graph_mode_only
  def testDepthwise(self):
    # [FUNCTION DOC]
    # Scenario: same as the gradient test but with depthwise separable convolutions. Counts depthwise
    # convolution, filter-gradient and input-gradient nodes, asserts each uses the channels-first format and
    # that there are exactly six.
    meta_graph = _simple_metagraph(depthwise=True)
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.CopyFrom(
        rewriter_config_pb2.RewriterConfig(
            layout_optimizer=rewriter_config_pb2.RewriterConfig.ON,
            min_graph_nodes=-1))
    optimized_graph = tf_optimizer.OptimizeGraph(
        config, meta_graph, cluster=_get_cluster())

    found = 0
    for node in optimized_graph.node:
      if node.op in [
          'DepthwiseConv2dNative', 'DepthwiseConv2dNativeBackpropFilter',
          'DepthwiseConv2dNativeBackpropInput'
      ]:
        found += 1
        self.assertEqual(node.attr['data_format'].s, b'NCHW')
    self.assertEqual(found, 6)

  def testCheckpointCompatibility(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the other tests it has no graph-mode-only decorator even though the training helper
    # resets the default graph and uses sessions.
    # Scenario: skipped without a GPU. Trains once with the optimizer off and saves a checkpoint in a temporary
    # directory, then restores and trains again with the optimizer off and with it on. Asserts every variable
    # value from the optimized run matches the unoptimized run within a very small tolerance, showing the
    # layout rewrite does not change checkpoint-visible variables.
    if not test.is_gpu_available(cuda_only=True):
      self.skipTest('GPU required')

    checkpoint_path = self.get_temp_dir()
    self._train(checkpoint_path)
    vars_expected = self._train(checkpoint_path, restore=True)
    vars_layout_optimized = self._train(
        checkpoint_path, restore=True, layout_optimizer=True)

    for var_expected, var_layout_optimized in zip(vars_expected,
                                                  vars_layout_optimized):
      self.assertAllClose(var_expected, var_layout_optimized, atol=1e-6)


if __name__ == '__main__':
  test.main()
