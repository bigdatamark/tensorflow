# [MODULE DOC]
# Tests that the BiasAdd gradient is deterministic (bit-identical across repeated runs) on GPU.
# Reuses the shared BiasAdd test base and adds a parameterized determinism test over data layout,
# data rank and floating dtype. When run as a script it first enables op determinism globally
# (not enabled when imported by a test runner), then runs the tests.
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
"""Functional tests for deterministic BiasAdd."""

import numpy as np

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.nn_ops import bias_op_base
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class BiasAddDeterministicTest(bias_op_base.BiasAddTestBase,
                               parameterized.TestCase):
  # [CLASS DOC]
  # Test case combining the shared BiasAdd test base with parameterized testing. Inherits all the
  # base forward and gradient tests, overrides three error-check tests to do nothing, and adds a
  # GPU-only determinism test plus small helpers for building shapes and random data.

  def _makeShapeTuple(self, batch_size, channel_count, data_rank, data_dim,
                      data_layout):
    # [FUNCTION DOC]
    # HELPER: builds an input shape tuple for the determinism test.
    # Inputs: batch size, channel count, number of spatial dimensions (rank), size of each spatial
    # dimension, and the layout name (channels first or channels last). Returns the shape with the
    # channel dimension second or last. Raises a value error for any other layout name.
    data_dims = data_rank * (data_dim,)
    if data_layout == 'channels_first':
      shape = (batch_size,) + (channel_count,) + data_dims
    elif data_layout == 'channels_last':
      shape = (batch_size,) + data_dims + (channel_count,)
    else:
      raise ValueError('Unknown data format')
    return shape

  def _dataFormatFromDataLayout(self, data_layout=None):
    # [FUNCTION DOC]
    # HELPER: maps a layout name to a data format string: channels first gives the channels-first
    # format string and channels last gives the channels-last one. Any other value, including the
    # default of nothing, raises a value error.
    if data_layout == 'channels_first':
      return 'NCHW'
    elif data_layout == 'channels_last':
      return 'NHWC'
    else:
      raise ValueError('Unknown data_layout')

  def _randomNDArray(self, shape):
    # [FUNCTION DOC]
    # HELPER: returns a numpy array of the given shape of uniform random values between minus one
    # and one (uses numpy's global random state).
    return 2 * np.random.random_sample(shape) - 1

  def _randomDataOp(self, shape, data_type):
    # [FUNCTION DOC]
    # HELPER: returns a constant tensor of the given dtype holding random values (from the random
    # array helper) of the given shape.
    return constant_op.constant(self._randomNDArray(shape), dtype=data_type)

  @parameterized.named_parameters(
      *test_util.generate_combinations_with_testcase_name(
          # With the selected layer configuration, at least in TensorFlow
          # version 2.0, when data_layout='channels_last', bias_add operates
          # deterministically by default. I don't know if this is true for
          # all layer configurations. These cases are still being tested here,
          # for completeness.
          data_layout=['channels_first', 'channels_last'],
          data_rank=[1, 2, 3],
          data_type=[dtypes.float16, dtypes.float32, dtypes.float64]))
  @test_util.run_in_graph_and_eager_modes
  @test_util.run_cuda_only
  def testDeterministicGradients(self, data_layout, data_rank, data_type):
    # [FUNCTION DOC]
    # Scenario (CUDA only, graph and eager, parameterized over layout, data rank one to three and
    # float16/32/64): inside a session forced onto the GPU, derives a numpy seed from the
    # parameters, builds random input and bias of batch 10, 8 channels and spatial size 14, and
    # computes the bias gradient through bias_add with a random upstream gradient injected by
    # multiplication. Repeats five times with different upstream gradients, computing the gradient
    # twice each time, and asserts the two results are exactly equal. Eager mode uses a persistent
    # tape; graph mode uses a placeholder for the upstream gradient and colocated gradients.
    # Side effect: seeds numpy's global random state.
    with self.session(force_gpu=True):
      # Using a cached_session with force_gpu=True does not work at the time
      # of writing (2019-12-10). Before the @parameterized.named_parameters
      # decorator was added, this non-cached session context was set outside
      # the iteration loops for the parameter combinations, and so was re-used.
      seed = (
          hash(data_layout) % 256 + hash(data_rank) % 256 +
          hash(data_type) % 256)
      np.random.seed(seed)
      batch_size = 10
      channel_count = 8
      data_dim = 14
      input_shape = self._makeShapeTuple(batch_size, channel_count, data_rank,
                                         data_dim, data_layout)
      bias_shape = (channel_count,)
      output_shape = input_shape
      input_val = self._randomDataOp(input_shape, data_type)
      bias_val = self._randomDataOp(bias_shape, data_type)
      data_format = self._dataFormatFromDataLayout(data_layout)
      repeat_count = 5
      if context.executing_eagerly():

        def bias_gradients(local_seed):
          # [FUNCTION DOC]
          # HELPER: computes the bias gradient eagerly for one seed.
          # Input: an integer seed for numpy's global random state (side effect). Generates a
          # random upstream gradient, records bias_add and the multiplication on a persistent
          # tape, and returns the gradient of the product with respect to the bias.
          np.random.seed(local_seed)
          upstream_gradients = self._randomDataOp(output_shape, data_type)
          with backprop.GradientTape(persistent=True) as tape:
            tape.watch(bias_val)
            bias_add_output = nn_ops.bias_add(
                input_val, bias_val, data_format=data_format)
            gradient_injector_output = bias_add_output * upstream_gradients
          return tape.gradient(gradient_injector_output, bias_val)

        for i in range(repeat_count):
          local_seed = seed + i  # select different upstream gradients
          result_a = bias_gradients(local_seed)
          result_b = bias_gradients(local_seed)
          self.assertAllEqual(result_a, result_b)
      else:  # graph mode
        upstream_gradients = array_ops.placeholder(
            data_type, shape=output_shape, name='upstream_gradients')
        bias_add_output = nn_ops.bias_add(
            input_val, bias_val, data_format=data_format)
        gradient_injector_output = bias_add_output * upstream_gradients
        # The gradient function behaves as if grad_ys is multiplied by the op
        # gradient result, not passing the upstram gradients through the op's
        # gradient generation graph. This is the reason for using the
        # gradient injector
        bias_gradients = gradients_impl.gradients(
            gradient_injector_output,
            bias_val,
            grad_ys=None,
            colocate_gradients_with_ops=True)[0]
        for i in range(repeat_count):
          feed_dict = {upstream_gradients: self._randomNDArray(output_shape)}
          result_a = bias_gradients.eval(feed_dict=feed_dict)
          result_b = bias_gradients.eval(feed_dict=feed_dict)
          self.assertAllEqual(result_a, result_b)

  # TODO(duncanriach): Re-enable the following three tests for the error checks
  #   after deterministic functionality is implemented at the CUDA kernel level.
  def testInputDims(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this override is an empty stub, so the inherited input-dimension error check is
    # silently disabled and this test always passes (TODO in source: re-enable once deterministic
    # kernels are implemented). Does nothing.
    pass

  def testBiasVec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: empty stub that disables the inherited bias-must-be-a-vector error check; the
    # test always passes without testing anything. Does nothing.
    pass

  def testBiasInputsMatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: empty stub that disables the inherited bias-length-mismatch error check; the
    # test always passes without testing anything. Does nothing.
    pass


if __name__ == '__main__':
  # TODO(reedwm): Merge this file with bias_op_base.py and bias_op_test.py
  config.enable_op_determinism()
  test.main()
