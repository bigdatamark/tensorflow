# [MODULE DOC]
# KNOWN ISSUE: the file relies on removed numpy aliases (the bool, float and object aliases of
#   numpy) and on legacy Keras optimizers; with a recent numpy those tests fail with attribute errors.
# Test suite for the Apple Metal (MPS) TensorFlow plugin, run on macOS arm64 CI machines.
# It is a standalone test script, not a library. It defines numpy-based reference helpers (tolerance
# lookup, powerset, Adam update, direct pooling) and many test case classes that compare plugin
# (GPU) results with CPU or numpy references for: optimizers (momentum, Adam, gradient descent),
# arg-max, gather, slice, L2 loss, rounding, reverse-sequence, top-k, split, bilinear resize,
# one-hot, 2D convolution (forward and backprop), pooling, random ops (stateful and stateless),
# broadcast_to, binary and logical ops, softmax cross entropy, add_n, resource variables, bias add,
# leaky relu, relu, relu6, softmax, concat, tile, pad, batch norm, reductions, min/max, fill,
# squared difference, ones/zeros like, select/where, transpose and broadcasting arithmetic.
# Also provides a small benchmark helper. Importing the module only defines names and module level
# constants for arithmetic operator callables and a list of test dtypes; running it as a program
# starts the TensorFlow test runner.
"""Copyright 2023 The TensorFlow Authors.

All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# !/usr/bin/env python3
# pylint: disable=g-bad-todo
# pylint: disable=redefined-builtin
# pylint: disable=arguments-out-of-order
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=protected-access
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import gc
import itertools
import math
import re
import time

from absl.testing import parameterized
import numpy as np
from PIL import Image
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow import raw_ops

from tensorflow.python.client import session
from tensorflow.python.compat import compat
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import numpy_compat
from tensorflow.python.util.compat import collections_abc


_ADD = lambda x, y: x + y
_SUB = lambda x, y: x - y
_MUL = lambda x, y: x * y
_POW = lambda x, y: x**y
_TRUEDIV = lambda x, y: x / y
_FLOORDIV = lambda x, y: x // y
_MOD = lambda x, y: x % y
_NEG = lambda x: -x
_ABS = abs
_MAX_RANK = 5


def _default_tolerance(dtype):
  # [FUNCTION DOC]
  # HELPER: numeric tolerance lookup used by the unary-op comparison helpers in the test classes.
  # Maps a numpy dtype to a default comparison tolerance: 5e-3 for half precision, 1e-3 for single
  # precision float or complex, 1e-5 for double precision float or complex. For any other dtype it
  # returns the null value so that callers using it as a tolerance fail fast. No side effects.
  """Returns a sensible default tolerance for comparing results of a given type.

  Args:
    dtype: A datatype.
  """
  if dtype == np.float16:
    return 5e-3
  elif dtype in (np.float32, np.complex64):
    return 1e-3
  elif dtype in (np.float64, np.complex128):
    return 1e-5
  else:
    return None  # Fail fast for unexpected types


def _powerset(iterable):
  # [FUNCTION DOC]
  # HELPER: generates reduction-axes combinations for reduction tests (not used elsewhere in this file).
  # KNOWN ISSUE: the docstring example calls it powerset without the leading underscore, and the
  #   function is never called in this module, so it is dead code here.
  # Takes any iterable, copies it into a list and returns a lazy iterator over all subsets of that
  # list as tuples, ordered by increasing subset size, starting with the empty tuple. Nothing is mutated.
  """Helper for generating all possible reduction_axes arguments.

  Example: powerset([0,1,2]): () (0,) (1,) (2,) (0,1) (0,2) (1,2) (0,1,2)

  Args:
    iterable: An iterable of items to generate the powerset of.

  Returns:
    The powerset of all items in iterable.
  """
  s = list(iterable)
  return itertools.chain.from_iterable(
      itertools.combinations(s, r) for r in range(len(s) + 1)
  )


def adam_update_numpy(
    param, g_t, t, m, v, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
):
  # [FUNCTION DOC]
  # HELPER: numpy reference implementation of one Adam step, used by the Adam optimizer tests.
  # Inputs are the current parameter array, the gradient, the step number (starting at 1), the first
  # and second moment estimates, and optional hyper parameters (learning rate 0.001, beta1 0.9,
  # beta2 0.999, epsilon 1e-8). It computes the bias corrected step size, updates both moment
  # estimates as exponential moving averages of the gradient and squared gradient, and returns a
  # tuple of the new parameter, new first moment and new second moment. No side effects.
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


def pool_direct_single_axis(
    input,  # pylint: disable=redefined-builtin
    axis,
    window_size,
    pooling_type,
    padding,
    dilation_rate,
    stride,
):
  # [FUNCTION DOC]
  # HELPER: numpy reference for pooling along one axis, used by pool_direct and the pooling tests.
  # Inputs: the input array, the axis to pool, the window size, the pooling type (max or average,
  # given as an upper case string), the padding mode (same or valid), the dilation rate and the stride.
  # It computes the output length and left padding from the padding mode, then for every output
  # position slices the (dilated) window, clipped to the input bounds, and reduces it with maximum or
  # mean. Returns the pooled array with the pooled axis resized. Raises a value error for an
  # unsupported padding or pooling type. Padding cells are never included in a window.
  effective_window_size = (window_size - 1) * dilation_rate + 1
  input_size = input.shape[axis]
  if padding == "SAME":
    output_size = int(math.ceil(input_size / stride))
    total_padding_amount = max(
        0, (output_size - 1) * stride + effective_window_size - input_size
    )
    before_padding = total_padding_amount // 2
  elif padding == "VALID":
    output_size = int(
        math.ceil((input_size - effective_window_size + 1) / stride)
    )
    before_padding = 0
  else:
    raise ValueError("Unsupported padding type: %r" % (padding,))

  output_shape = input.shape[:axis] + (output_size,) + input.shape[axis + 1 :]
  output = np.zeros(output_shape, input.dtype)
  initial_dim_selector = tuple(np.s_[:] for _ in range(axis))
  if pooling_type == "MAX":
    pooling_func = np.max
  elif pooling_type == "AVG":
    pooling_func = np.mean
  else:
    raise ValueError("Unsupported pooling type: %r" % (pooling_type,))
  for output_pos in range(output_size):
    input_start_pos = output_pos * stride - before_padding
    input_end_pos = min(input_start_pos + effective_window_size, input_size)
    if input_start_pos < 0:
      input_start_pos += dilation_rate
    input_slice = np.s_[input_start_pos:input_end_pos:dilation_rate]

    output[initial_dim_selector + (output_pos,)] = pooling_func(
        input[initial_dim_selector + (input_slice,)], axis=axis
    )
  return output


def pool_direct(
    input,  # pylint: disable=redefined-builtin
    window_shape,
    pooling_type,
    padding,  # pylint: disable=redefined-builtin
    dilation_rate,
    strides,
    data_format=None,
):
  # [FUNCTION DOC]
  # HELPER: numpy reference for N-dimensional pooling used by PoolingTest to validate the nn pool op.
  # Pools sequentially along each spatial axis by calling the single-axis helper once per window
  # dimension. Inputs: the input array, window shape, pooling type, padding, per-axis dilation rates,
  # per-axis strides, and an optional data format; a format starting with the letters N and C means
  # channels-first so spatial axes start at 1 index 2, otherwise they start at index 1. Returns the
  # pooled array. Errors from the single-axis helper propagate.
  if data_format is None or not data_format.startswith("NC"):
    spatial_start_dim = 1
  else:
    spatial_start_dim = 2
  output = input
  for i in range(len(window_shape)):
    output = pool_direct_single_axis(
        input=output,
        axis=i + spatial_start_dim,
        window_size=window_shape[i],
        pooling_type=pooling_type,
        padding=padding,
        dilation_rate=dilation_rate[i],
        stride=strides[i],
    )
  return output


_TEST_TYPES = [dtypes.float32]


class MomentumOptimizerTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the legacy Keras SGD optimizer with momentum, run against both parameter variables and
  # numpy references. Derives from the TensorFlow test case and the absl parameterized test case; it
  # holds no state of its own beyond a numpy helper for Nesterov updates.

  def _update_nesterov_momentum_numpy(self, var, accum, g, lr, momentum):
    # [FUNCTION DOC]
    # HELPER: numpy reference for one Nesterov momentum update used by testNesterovMomentum.
    # Inputs: the variable value, the accumulator, the gradient, the learning rate and the momentum.
    # Computes the new accumulator as momentum times old accumulator minus learning rate times gradient,
    # then updates the variable with the Nesterov look-ahead formula. The variable is updated in place
    # when it is a numpy array. Returns the pair of updated variable and accumulator.
    accum = accum * momentum - g * lr
    var += accum * momentum - g * lr
    return var, accum

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: momentum SGD with learning rate 2.0 and momentum 0.9 on two float32 variables with
    # constant gradients. Creates the variables, applies the gradients, checks that slot variables
    # named momentum exist with the right shapes, initializes variables and runs two update steps.
    # Asserts that after step one the accumulators equal minus gradient times rate and the variables
    # moved by that amount, and after step two the accumulators and variables match hand computed
    # momentum values. Uses the case-level evaluate helper so it works in eager or graph mode.
    for _, dtype in enumerate([dtypes.float32]):
      var0 = variables.Variable([1.0, 2.0], dtype=dtype, name="var0")
      var1 = variables.Variable([3.0, 4.0], dtype=dtype, name="var1")
      grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
      learning_rate = 2.0
      momentum = 0.9
      mom_opt = tf.keras.optimizers.legacy.SGD(
          learning_rate=learning_rate, momentum=momentum
      )
      # self.assertFalse(mom_opt._initial_decay)
      mom_update = mom_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      # Check we have slots
      slot0 = mom_opt.get_slot(var0, "momentum")
      self.assertEqual(slot0.shape, var0.shape)
      slot1 = mom_opt.get_slot(var1, "momentum")
      self.assertEqual(slot1.shape, var1.shape)

      # Step 1: the momentum accumulators where 0. So we should see a normal
      # update: v -= grad * learning_rate
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(mom_update)
      # Check that the momentum accumulators have been updated.
      self.assertAllCloseAccordingToType(
          np.array([-0.2, -0.2]), self.evaluate(slot0)
      )
      self.assertAllCloseAccordingToType(
          np.array([-0.02, -0.02]), self.evaluate(slot1)
      )
      # Check that the parameters have been updated.
      self.assertAllCloseAccordingToType(
          np.array([1.0 - (0.1 * 2.0), 2.0 - (0.1 * 2.0)]), self.evaluate(var0)
      )
      self.assertAllCloseAccordingToType(
          np.array([3.0 - (0.01 * 2.0), 4.0 - (0.01 * 2.0)]),
          self.evaluate(var1),
      )
      # Step 2: the momentum accumulators contain the previous update.
      self.evaluate(mom_update)
      if context.executing_eagerly():
        mom_opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      # Check that the momentum accumulators have been updated.
      self.assertAllCloseAccordingToType(
          np.array([(0.9 * (-0.2) - 2.0 * 0.1), (0.9 * (-0.2) - 2.0 * 0.1)]),
          self.evaluate(slot0),
      )
      self.assertAllCloseAccordingToType(
          np.array(
              [(0.9 * (-0.02) - 2.0 * 0.01), (0.9 * (-0.02) - 2.0 * 0.01)]
          ),
          self.evaluate(slot1),
      )
      # Check that the parameters have been updated.
      self.assertAllCloseAccordingToType(
          np.array([
              1.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
              2.0 - (0.1 * 2.0) - ((0.9 * 0.1 + 0.1) * 2.0),
          ]),
          self.evaluate(var0),
      )
      self.assertAllCloseAccordingToType(
          np.array([
              2.98 - ((0.9 * 0.01 + 0.01) * 2.0),
              3.98 - ((0.9 * 0.01 + 0.01) * 2.0),
          ]),
          self.evaluate(var1),
      )

  def testNesterovMomentum(self):
    # [FUNCTION DOC]
    # Scenario: Nesterov momentum SGD (learning rate 2.0, momentum 0.9) minimizing a quadratic plus
    # linear loss over two float32 variables in a fresh graph. Runs four optimizer steps and after each
    # step advances numpy copies with the reference Nesterov update, then asserts both variables equal
    # the numpy values. Exercises minimize with a callable loss.
    with ops.Graph().as_default():
      for dtype in [dtypes.float32]:
        var0 = variables.Variable([1.0, 2.0], dtype=dtype, name="var0")
        var1 = variables.Variable([3.0, 4.0], dtype=dtype, name="var1")
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        accum0_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        accum1_np = np.array([0.0, 0.0], dtype=dtype.as_numpy_dtype)
        loss = lambda: 5 * var0 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
        mom_op = tf.keras.optimizers.legacy.SGD(
            learning_rate=2.0, momentum=0.9, nesterov=True
        )
        opt_op = mom_op.minimize(loss, [var0, var1])
        self.evaluate(variables.global_variables_initializer())
        for _ in range(1, 5):
          self.evaluate(opt_op)
          var0_np, accum0_np = self._update_nesterov_momentum_numpy(
              var0_np, accum0_np, var0_np * 10, 2.0, 0.9
          )
          var1_np, accum1_np = self._update_nesterov_momentum_numpy(
              var1_np, accum1_np, 3, 2.0, 0.9
          )
          self.assertAllClose(var0_np, self.evaluate(var0))
          self.assertAllClose(var1_np, self.evaluate(var1))


class ArgMaxTest(test.TestCase):
  # [CLASS DOC]
  # Tests the arg-max and arg-min ops against numpy. Holds helpers that run an op on GPU and CPU
  # sessions and compare with expected values, plus several test scenarios; some scenarios are
  # currently disabled by comments.

  def _testArg(
      self,
      method,
      x,
      axis,
      expected_values,
      use_gpu=False,
      expected_err_re=None,
  ):
    # [FUNCTION DOC]
    # HELPER: runs an arg-reduction op once and checks the outcome for the arg-max tests.
    # Inputs: the op callable, the input array, the axis, the expected values, whether to use the GPU
    # (default off) and an optional regular expression for an expected op error. Without an expected
    # error it evaluates the op, asserts the result dtype is 64-bit integer, equals the expected values
    # and has the expected shape. With an expected error it asserts that evaluation raises an op error
    # matching the expression. Runs inside a session with the given device preference.
    with self.session(use_gpu=use_gpu):
      ans = method(x, axis=axis)
      if expected_err_re is None:
        tf_ans = self.evaluate(ans)
        # Defaults to int64 output.
        self.assertEqual(np.int64, tf_ans.dtype)
        self.assertAllEqual(tf_ans, expected_values)
        self.assertShapeEqual(expected_values, ans)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate(ans)

  def _testBothArg(
      self, method, x, axis, expected_values, expected_err_re=None
  ):
    # [FUNCTION DOC]
    # HELPER: runs _testArg on the GPU and, unless XLA is enabled, on the CPU too.
    # Inputs as for _testArg minus the GPU flag. XLA is skipped on the CPU pass because compilation
    # time is too large. Returns nothing.
    self._testArg(method, x, axis, expected_values, True, expected_err_re)
    # Compilation time is too large with XLA/CPU autojit.
    if not test_util.is_xla_enabled():
      self._testArg(method, x, axis, expected_values, False, expected_err_re)

  def _testBasic(self, dtype):
    # [FUNCTION DOC]
    # HELPER: shared body of testFloat. Builds a shuffled array of 200 zeros and ones of the given dtype
    # and checks that the arg-max along axis 0 matches numpy. The arg-min check is commented out.
    x = np.arange(200, dtype=np.float32).astype(np.bool_).astype(dtype)
    np.random.shuffle(x)

    # Check that argmin and argmax match numpy along the primary axis
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    # self._testBothArg(math_ops.argmin, x, 0, x.argmin())

  def _testTieBreaking(self, dtype):
    # [FUNCTION DOC]
    # HELPER: checks tie breaking of arg-max and arg-min on a vector of 200 equal zeros against numpy
    # KNOWN ISSUE: the only caller (testFloat) has this call commented out, so this helper is never run.
    # (the lowest index must win). Input: a numpy dtype.
    x = np.zeros(200, dtype=dtype)

    # Check that argmin and argmax match numpy along the primary axis for
    # breaking ties.
    self._testBothArg(math_ops.argmax, x, 0, x.argmax())
    self._testBothArg(math_ops.argmin, x, 0, x.argmin())

  def _testDim(self, dtype):
    # [FUNCTION DOC]
    # HELPER: builds a shuffled seven dimensional float array and compares arg-max and arg-min along
    # KNOWN ISSUE: the only caller (testFloat) has this call commented out, so this helper is never run.
    # every axis from minus 7 to 6 with numpy. Input: a numpy dtype for the array.
    shape = (3, 2, 4, 5, 6, 3, 7)
    x = np.arange(
        functools.reduce(lambda x, y: x * y, shape), dtype=np.float32
    ).astype(dtype)
    np.random.shuffle(x)
    x = x.reshape(shape)

    # Check that argmin and argmax match numpy along all axes
    for axis in range(-7, 7):
      self._testBothArg(math_ops.argmax, x, axis, x.argmax(axis))
      self._testBothArg(math_ops.argmin, x, axis, x.argmin(axis))

  def testFloat(self):
    # [FUNCTION DOC]
    # Runs the basic arg-max check with float32 data. The tie breaking and per-dimension checks are
    # commented out so only the basic arg-max comparison actually executes.
    self._testBasic(np.float32)
    # self._testTieBreaking(np.float32)
    # self._testDim(np.float32)

  def testFloatInt32Output(self):
    # [FUNCTION DOC]
    # Scenario: random float32 vector of 200 elements. Computes arg-max and arg-min with the output
    # type set to 32-bit integer on a GPU-preferring session and asserts the result dtype is int32 and
    # the values equal numpy. Values fit in 32 bits, so comparing to numpy's 64-bit results is valid.
    x = numpy_compat.np_asarray(100 * np.random.randn(200), dtype=np.float32)
    expected_values = x.argmax()
    with self.session(use_gpu=True):
      ans = math_ops.argmax(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      # The values are equal when comparing int32 to int64 because
      # the values don't have a range that exceeds 32-bit integers.
      self.assertAllEqual(tf_ans, expected_values)
    expected_values = x.argmin()
    with self.session(use_gpu=True):
      ans = math_ops.argmin(x, axis=0, output_type=dtypes.int32)
      tf_ans = self.evaluate(ans)
      self.assertEqual(np.int32, tf_ans.dtype)
      self.assertAllEqual(tf_ans, expected_values)


class GatherTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the gather op for scalar and vector indices on one and two dimensional data. Uses the
  # module level list of test dtypes (only float32 here) and a parameterized test case base.

  def _buildParams(self, data, dtype):
    # [FUNCTION DOC]
    # HELPER: converts numpy data to the target TensorFlow dtype's numpy type; for complex dtypes it adds
    # an index dependent imaginary part (ten times the value) so wrong values are detectable. Returns the
    # converted array.
    data = data.astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testScalar1D(self):
    # [FUNCTION DOC]
    # Gathers with a scalar index and with a list of indices from a six element vector. Asserts the
    # result equals numpy indexing and that the static shape matches the numpy result shape.
    with self.cached_session(use_gpu=True):
      data = np.array([0, 1, 2, 3, 7, 5])
      for dtype in _TEST_TYPES:
        for indices in 4, [1, 2, 2, 4, 5]:
          params_np = self._buildParams(data, dtype)
          params = constant_op.constant(params_np)
          indices_tf = constant_op.constant(indices)
          gather_t = array_ops.gather(params, indices_tf)
          gather_val = self.evaluate(gather_t)
          np_val = params_np[indices]
          self.assertAllEqual(np_val, gather_val)
          self.assertEqual(np_val.shape, gather_t.get_shape())

  def testScalar2D(self):
    # [FUNCTION DOC]
    # Gathers row/column index 2 from a five by three integer matrix along every axis. Prints the
    # TensorFlow and CPU results, asserts equality with numpy take and that the static shape drops the
    # gathered axis.
    with self.session(use_gpu=True):
      data = np.array(
          [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
      )
      for dtype in _TEST_TYPES:
        for axis in range(data.ndim):
          params_np = self._buildParams(data, dtype)
          params = constant_op.constant(params_np)
          indices = constant_op.constant(2)
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = self.evaluate(gather_t)
          print("TF {}".format(gather_val))
          print("CPU {}".format(np.take(params_np, 2, axis=axis)))
          self.assertAllEqual(np.take(params_np, 2, axis=axis), gather_val)
          expected_shape = data.shape[:axis] + data.shape[axis + 1 :]
          self.assertEqual(expected_shape, gather_t.get_shape())

  def testSimpleTwoD32(self):
    # [FUNCTION DOC]
    # Gathers a list of four indices along each axis of a five by three matrix. Asserts equality with
    # numpy take and that the static output shape has four entries along the gathered axis.
    with self.session(use_gpu=True):
      data = np.array(
          [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]]
      )
      for dtype in _TEST_TYPES:
        for axis in range(data.ndim):
          params_np = self._buildParams(data, dtype)
          params = constant_op.constant(params_np)
          # The indices must be in bounds for any axis.
          indices = constant_op.constant([0, 1, 0, 2])
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = self.evaluate(gather_t)
          self.assertAllEqual(
              np.take(params_np, [0, 1, 0, 2], axis=axis), gather_val
          )
          expected_shape = data.shape[:axis] + (4,) + data.shape[axis + 1 :]
          self.assertEqual(expected_shape, gather_t.get_shape())


class SliceTest(test.TestCase):
  # [CLASS DOC]
  # Tests slicing via the slice op and Python style indexing, including empty, random, 3D conv based
  # and partial shape inference cases.

  def testEmpty(self):
    # [FUNCTION DOC]
    # Takes an empty slice of a random four by four array for every start column and asserts that the
    # result equals the numpy empty slice.
    inp = np.random.rand(4, 4).astype("f")
    for k in xrange(4):
      with self.cached_session(use_gpu=True):
        a = constant_op.constant(inp, shape=[4, 4], dtype=dtypes.float32)
        slice_t = a[2, k:k]
        slice_val = self.evaluate(slice_t)
      self.assertAllEqual(slice_val, inp[2, k:k])

  def testSimple(self):
    # [FUNCTION DOC]
    # Slices the top-left two by two block of a random four by four tensor using both the slice op and
    # indexing. Asserts both values equal numpy and the static shapes match the value shapes.
    with self.session(use_gpu=True) as _:
      inp = np.random.rand(4, 4).astype("f")
      a = constant_op.constant(
          [float(x) for x in inp.ravel(order="C")],
          shape=[4, 4],
          dtype=dtypes.float32,
      )
      slice_t = array_ops.slice(a, [0, 0], [2, 2])
      slice2_t = a[:2, :2]
      slice_val, slice2_val = self.evaluate([slice_t, slice2_t])
    self.assertAllEqual(slice_val, inp[:2, :2])
    self.assertAllEqual(slice2_val, inp[:2, :2])
    self.assertEqual(slice_val.shape, slice_t.get_shape())
    self.assertEqual(slice2_val.shape, slice2_t.get_shape())

  def testSingleDimension(self):
    # [FUNCTION DOC]
    # Repeats ten times: indexes a random vector with a scalar and a random range and asserts equality
    # with numpy for the scalar element and the sub range.
    for _ in range(10):
      with self.cached_session(use_gpu=True):
        inp = np.random.rand(10).astype("f")
        a = constant_op.constant(inp, shape=[10], dtype=dtypes.float32)

        hi = np.random.randint(0, 9)
        scalar_t = a[hi]
        scalar_val = self.evaluate(scalar_t)
        self.assertAllEqual(scalar_val, inp[hi])

        if hi > 0:
          lo = np.random.randint(0, hi)
        else:
          lo = 0
        slice_t = a[lo:hi]
        slice_val = self.evaluate(slice_t)
        self.assertAllEqual(slice_val, inp[lo:hi])

  def test3Dimension(self):
    # [FUNCTION DOC]
    # Applies a 3D convolution with a one by one by one by eight by eight filter of incrementing values
    # to a rank five input, slices out a single output position and asserts the eight values match
    # precomputed expectations within a relative tolerance of one in a million.
    with self.cached_session():
      input_shape = [8, 16, 16, 16, 8]
      total_input_size = 1
      for s in input_shape:
        total_input_size *= s
      inputs = [
          i * 1.0 / total_input_size for i in range(1, total_input_size + 1)
      ]
      a = constant_op.constant(inputs, shape=input_shape, dtype=dtypes.float32)

      filter_shape = [1, 1, 1, 8, 8]
      total_filter_size = 1
      for s in filter_shape:
        total_filter_size *= s
      filters = [
          i * 1.0 / total_filter_size for i in range(1, total_filter_size + 1)
      ]
      f = constant_op.constant(
          filters, shape=filter_shape, dtype=dtypes.float32
      )

      conv_t = nn_ops.conv3d(
          a, filter=f, strides=[1, 1, 1, 1, 1], padding="VALID"
      )
      slice_t = array_ops.slice(conv_t, [0, 1, 1, 1, 0], [1, 1, 1, 1, 8])
      result = self.evaluate(slice_t)
      expected = [
          0.03028321,
          0.03132677,
          0.03237033,
          0.03341389,
          0.03445745,
          0.035501,
          0.03654456,
          0.03758812,
      ]
      self.assertAllClose(expected, result.flatten(), rtol=1e-6)

  def testRandom(self):
    # [FUNCTION DOC]
    # Builds a random rank six tensor with random dimensions, picks random start indices and sizes,
    # slices with the slice op and with explicit strided indexing, and asserts both equal the numpy slice
    # and have the same static shape. Randomness makes dimensions vary between runs.
    # Random dims of rank 6
    input_shape = np.random.randint(0, 20, size=6)
    inp = np.random.rand(*input_shape).astype("f")
    with self.session(use_gpu=True) as _:
      a = constant_op.constant(
          [float(x) for x in inp.ravel(order="C")],
          shape=input_shape,
          dtype=dtypes.float32,
      )
      indices = [0 if x == 0 else np.random.randint(x) for x in input_shape]
      sizes = [
          np.random.randint(0, input_shape[i] - indices[i] + 1)
          for i in range(6)
      ]
      slice_t = array_ops.slice(a, indices, sizes)
      slice2_t = a[
          indices[0] : indices[0] + sizes[0],
          indices[1] : indices[1] + sizes[1],
          indices[2] : indices[2] + sizes[2],
          indices[3] : indices[3] + sizes[3],
          indices[4] : indices[4] + sizes[4],
          indices[5] : indices[5] + sizes[5],
      ]

      slice_val, slice2_val = self.evaluate([slice_t, slice2_t])

    expected_val = inp[
        indices[0] : indices[0] + sizes[0],
        indices[1] : indices[1] + sizes[1],
        indices[2] : indices[2] + sizes[2],
        indices[3] : indices[3] + sizes[3],
        indices[4] : indices[4] + sizes[4],
        indices[5] : indices[5] + sizes[5],
    ]
    self.assertAllEqual(slice_val, expected_val)
    self.assertAllEqual(slice2_val, expected_val)
    self.assertEqual(expected_val.shape, slice_t.get_shape())
    self.assertEqual(expected_val.shape, slice2_t.get_shape())

  def testPartialShapeInference(self):
    # [FUNCTION DOC]
    # Checks static shape inference for slice on a zero tensor of shape one by two by three, using
    # negative sizes and a partially dynamic size; asserts the inferred shape is one, two, three.
    z = array_ops.zeros((1, 2, 3))
    self.assertAllEqual(z.get_shape().as_list(), [1, 2, 3])

    m1 = array_ops.slice(z, [0, 0, 0], [-1, -1, -1])
    self.assertAllEqual(m1.get_shape().as_list(), [1, 2, 3])

    m2 = array_ops.slice(z, [0, 0, 0], [constant_op.constant(1) + 0, 2, -1])
    self.assertAllEqual(m2.get_shape().as_list(), [1, 2, 3])


class L2LossTest(test.TestCase):
  # [CLASS DOC]
  # Tests the L2 loss op for forward values and gradient accuracy.

  @test_util.run_in_graph_and_eager_modes
  def testL2Loss(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. For float32 and float64 computes the L2 loss of a two by two
    # tensor and asserts it equals 7.0.
    for dtype in [dtypes.float32, dtypes.float64]:
      x = constant_op.constant(
          [1.0, 0.0, 3.0, 2.0], shape=[2, 2], name="x", dtype=dtype
      )
      l2loss = nn_ops.l2_loss(x)
      value = self.evaluate(l2loss)
      self.assertAllClose(7.0, value)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Deprecated-v1 graph test: seeds numpy, builds a random float64 tensor of shape 20 by 7 by 3, and
    # uses the gradient checker to compute the error of the L2 loss gradient; prints it and asserts it
    # is below 1e-10.
    x_shape = [20, 7, 3]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    with self.cached_session():
      x = constant_op.constant(x_val, name="x")
      output = nn_ops.l2_loss(x)
      err = gradient_checker.compute_gradient_error(x, x_shape, output, [1])
    print("L2Loss gradient err = %g " % err)
    err_tolerance = 1e-10
    self.assertLess(err, err_tolerance)


class AdamOptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Tests the TF1 Adam optimizer with resource variables, callable and tensor learning rates,
  # sharing of the optimizer between updates, and multiple graphs and eager contexts.

  def doTestBasic(self, use_resource=False, use_callable_params=False):
    # [FUNCTION DOC]
    # HELPER: shared body for the basic Adam tests.
    # Inputs: use_resource (default off) selects resource variables versus reference variables, and
    # use_callable_params (default off) passes hyper parameters as callables. Skips the test when eager
    # execution is on with non resource variables. Otherwise builds two float32 variables, applies Adam
    # and checks the beta power accumulators exist, are in the optimizer variables and have the right
    # variable kind; in graph mode checks no variables leak into another graph. Runs three steps and
    # asserts the beta powers, the variables (against the numpy Adam reference) and, for resource
    # variables, the slot name.
    if context.executing_eagerly() and not use_resource:
      self.skipTest(
          "Skipping test with use_resource=False and executing eagerly."
      )
    for i, dtype in enumerate([dtypes.float32]):
      with self.session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i
          )
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i
          )
        else:
          var0 = variables.RefVariable(var0_np)
          var1 = variables.RefVariable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = lambda: 0.001
        beta1 = lambda: 0.9
        beta2 = lambda: 0.999
        epsilon = lambda: 1e-8
        if not use_callable_params:
          learning_rate = learning_rate()
          beta1 = beta1()
          beta2 = beta2()
          epsilon = epsilon()

        opt = adam.AdamOptimizer(learning_rate=learning_rate)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt.variables()
        beta1_power, beta2_power = opt._get_beta_accumulators()
        self.assertIsNotNone(beta1_power)
        self.assertIsNotNone(beta2_power)
        self.assertIn(beta1_power, opt_variables)
        self.assertIn(beta2_power, opt_variables)
        # Ensure that non-slot variables are the same type as the requested
        # variables.
        self.assertEqual(
            use_resource,
            resource_variable_ops.is_resource_variable(beta1_power),
        )
        self.assertEqual(
            use_resource,
            resource_variable_ops.is_resource_variable(beta2_power),
        )

        if not context.executing_eagerly():
          with ops.Graph().as_default():
            # Shouldn't return non-slot variables from other graphs.
            self.assertEqual(0, len(opt.variables()))
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          self.assertAllCloseAccordingToType(
              0.9 ** (t + 1), self.evaluate(beta1_power)
          )
          self.assertAllCloseAccordingToType(
              0.999 ** (t + 1), self.evaluate(beta2_power)
          )
          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)
          var0_eval = self.evaluate(var0)
          var1_eval = self.evaluate(var1)
          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, var0_eval)
          self.assertAllCloseAccordingToType(var1_np, var1_eval)
          if use_resource:
            self.assertEqual(
                "var0_%d/Adam:0" % (i,), opt.get_slot(var=var0, name="m").name
            )

  def testBasic(self):
    # [FUNCTION DOC]
    # Runs the basic Adam body with resource variables.
    self.doTestBasic(use_resource=True)

  @test_util.run_in_graph_and_eager_modes
  def testResourceBasic(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: it is functionally identical to testBasic (both pass resource variables).
    # Same as testBasic but decorated to run in both graph and eager modes.
    self.doTestBasic(use_resource=True)

  def testBasicCallableParams(self):
    # [FUNCTION DOC]
    # Runs the basic Adam body in eager mode with resource variables and callable hyper parameters.
    with context.eager_mode():
      self.doTestBasic(use_resource=True, use_callable_params=True)

  @test_util.run_deprecated_v1
  def testTensorLearningRate(self):
    # [FUNCTION DOC]
    # Deprecated-v1 graph test: Adam with a constant tensor as learning rate. Runs three steps, checking
    # the beta power accumulators and the variable values against the numpy Adam reference each step.
    for dtype in [dtypes.float32]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adam.AdamOptimizer(constant_op.constant(0.001))
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(
              0.9**t, self.evaluate(beta1_power)
          )
          self.assertAllCloseAccordingToType(
              0.999**t, self.evaluate(beta2_power)
          )
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testSharing(self):
    # [FUNCTION DOC]
    # Deprecated-v1 graph test: builds two update ops from one Adam optimizer and alternates running
    # them over three steps. Asserts the beta powers advance as expected and the variables follow the
    # numpy reference, showing shared state is consistent.
    for dtype in [dtypes.float32]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        var0 = variables.Variable(var0_np)
        var1 = variables.Variable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)
        opt = adam.AdamOptimizer()
        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        variables.global_variables_initializer().run()

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 3 steps of intertwined Adam1 and Adam2.
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(
              0.9**t, self.evaluate(beta1_power)
          )
          self.assertAllCloseAccordingToType(
              0.999**t, self.evaluate(beta2_power)
          )
          if t % 2 == 0:
            update1.run()
          else:
            update2.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testTwoSessions(self):
    # [FUNCTION DOC]
    # Uses one optimizer in an eager context, then in two separate graphs and sessions with different
    # variable dtypes. Passes if no error is raised, showing optimizer state is keyed by graph.
    optimizer = adam.AdamOptimizer()

    with context.eager_mode():
      var0 = variables.Variable(
          np.array([1.0, 2.0], dtype=np.float32), name="v0"
      )
      grads0 = constant_op.constant(np.array([0.1, 0.1], dtype=np.float32))
      optimizer.apply_gradients([(grads0, var0)])

    g = ops.Graph()
    with g.as_default():
      with session.Session():
        var0 = variables.Variable(
            np.array([1.0, 2.0], dtype=np.float32), name="v0"
        )
        grads0 = constant_op.constant(np.array([0.1, 0.1], dtype=np.float32))
        optimizer.apply_gradients([(grads0, var0)])

    gg = ops.Graph()
    with gg.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))

        # If the optimizer saves any state not keyed by graph the following line
        # fails.
        optimizer.apply_gradients([(grads0, var0)])

  def testSlotsUniqueEager(self):
    # [FUNCTION DOC]
    # Eagerly minimizes the sum of two resource variables with Adam and asserts that the optimizer
    # exposes six unique variables (two non slot accumulators plus per variable first and second moments).
    with context.eager_mode():
      v1 = resource_variable_ops.ResourceVariable(1.0)
      v2 = resource_variable_ops.ResourceVariable(1.0)
      opt = adam.AdamOptimizer(1.0)
      opt.minimize(lambda: v1 + v2)
      # There should be two non-slot variables, and two unique slot variables
      # for v1 and v2 respectively.
      self.assertEqual(6, len({id(v) for v in opt.variables()}))


class RoundingTest(test.TestCase):
  # [CLASS DOC]
  # Tests rounding ops (floor, ceil, rint) against numpy.

  def _compare_values(self, x, y=None):
    # [FUNCTION DOC]
    # HELPER: compares the rint op with an expected array. Input: values and an optional expected result
    # (default numpy rint). Asserts equal values and matching shape.
    y = np.rint(x) if y is None else numpy_compat.np_asarray(y)

    tf_rint = math_ops.rint(x)
    np_rint = self.evaluate(tf_rint)

    self.assertAllEqual(y, np_rint)
    self.assertShapeEqual(y, tf_rint)

  def _compare(self, x):
    # [FUNCTION DOC]
    # HELPER: compares floor and ceil ops with numpy for an input array, asserting equal values and shapes.
    np_floor, np_ceil = np.floor(x), np.ceil(x)

    inx = ops.convert_to_tensor(x)
    ofloor, oceil = math_ops.floor(inx), math_ops.ceil(inx)
    tf_floor, tf_ceil = self.evaluate([ofloor, oceil])

    self.assertAllEqual(np_floor, tf_floor)
    self.assertAllEqual(np_ceil, tf_ceil)
    self.assertShapeEqual(np_floor, ofloor)
    self.assertShapeEqual(np_ceil, oceil)

  def _testDtype(self, dtype):
    # [FUNCTION DOC]
    # HELPER: builds a small array of the given dtype, compares floor/ceil, then (except for half
    # precision, where rint is unsupported) compares rint on the array and on two lists with tie cases.
    data = (np.arange(-3, 3) / 4.0).reshape(1, 3, 2).astype(dtype)
    self._compare(data)
    # TODO: rint op is not supported for float16
    if dtype is np.float16:
      return
    self._compare_values(data)
    x = [0.5, 0.5000001]
    y = [0.0, 1.0]
    self._compare_values(x, y=y)

    # numpy example
    x = [-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]
    y = [-2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 2.0]
    self._compare_values(x, y=y)

  def testTypes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test skips itself immediately (with a bug reference), so the loop over half,
    #   single and double precision never executes.
    self.skipTest("b/131162241")
    for dtype in [np.float16, np.float32, np.float64]:
      self._testDtype(dtype)


class ReverseSequenceTest(test.TestCase):
  # [CLASS DOC]
  # Tests the reverse-sequence op on a five dimensional input with different sequence length dtypes.

  def _validateReverseSequence(
      self, x, batch_axis, seq_axis, seq_lengths, truth, use_gpu=False
  ):
    # [FUNCTION DOC]
    # HELPER: runs reverse-sequence with given batch axis, sequence axis and lengths and asserts the
    # result is close to the expected truth and has its shape. The use_gpu flag (default off) picks the
    # device preference of the session.
    with self.cached_session(use_gpu=use_gpu):
      ans = array_ops.reverse_sequence(
          x, batch_axis=batch_axis, seq_axis=seq_axis, seq_lengths=seq_lengths
      )
      tf_ans = self.evaluate(ans)
      self.assertAllClose(tf_ans, truth, atol=1e-10)
      self.assertShapeEqual(truth, ans)

  def _testBasic(self, dtype, len_dtype=np.int64):
    # [FUNCTION DOC]
    # HELPER: builds a three by two by four input, permutes axes, prepares expected reversed output for
    # lengths three, zero and four, and validates with the permuted batch and sequence axes. Inputs: the
    # data dtype and the length dtype (default 64-bit integer).
    x = numpy_compat.np_asarray(
        [
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[9, 10, 11, 12], [13, 14, 15, 16]],
            [[17, 18, 19, 20], [21, 22, 23, 24]],
        ],
        dtype=dtype,
    )
    x = x.reshape(3, 2, 4, 1, 1)
    x = x.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    # reverse dim 2 up to (0:3, none, 0:4) along dim=0
    seq_lengths = numpy_compat.np_asarray([3, 0, 4], dtype=len_dtype)

    truth_orig = numpy_compat.np_asarray(
        [
            [[3, 2, 1, 4], [7, 6, 5, 8]],  # reverse 0:3
            [[9, 10, 11, 12], [13, 14, 15, 16]],  # reverse none
            [[20, 19, 18, 17], [24, 23, 22, 21]],
        ],  # reverse 0:4 (all)
        dtype=dtype,
    )
    truth_orig = truth_orig.reshape(3, 2, 4, 1, 1)
    truth = truth_orig.transpose([2, 1, 0, 3, 4])  # permute axes 0 <=> 2

    seq_axis = 0  # permute seq_axis and batch_axis (originally 2 and 0, resp.)
    batch_axis = 2
    self._validateReverseSequence(
        x, batch_axis, seq_axis, seq_lengths, truth, use_gpu=True
    )

  def testFloat(self):
    # [FUNCTION DOC]
    # Runs the basic check with float32 data for both 32-bit and 64-bit length dtypes.
    self._testBasic(np.float32, len_dtype=np.int32)
    self._testBasic(np.float32, len_dtype=np.int64)


class TopKTest(test.TestCase):
  # [CLASS DOC]
  # Tests the top-k op on small and large vectors, with Python and tensor k values.

  def _validateTopK(self, inputs, k, expected_values, expected_indices):
    # [FUNCTION DOC]
    # HELPER: runs top-k and asserts the shapes of both outputs match the expectations and the values
    # KNOWN ISSUE: the expected indices are only used for a shape check; their values are never compared.
    # output is close to the expected values.
    np_expected_values = np.array(expected_values)
    np_expected_indices = np.array(expected_indices)

    with self.cached_session(use_gpu=True) as _:
      values_op, indices_op = nn_ops.top_k(inputs, k)

      self.assertShapeEqual(np_expected_values, values_op)
      self.assertShapeEqual(np_expected_indices, indices_op)

      self.assertAllClose(np_expected_values, values_op)

  def testTop1(self):
    # [FUNCTION DOC]
    # Top one over two rows with a tie in the second row; expected values and indices are passed in.
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.3, 0.2]]
    self._validateTopK(inputs, 1, [[0.4], [0.3]], [[3], [1]])

  def testTop2(self):
    # [FUNCTION DOC]
    # Top two over two rows; expects values descending per row.
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2]]
    self._validateTopK(inputs, 2, [[0.4, 0.3], [0.4, 0.3]], [[3, 1], [2, 1]])

  def testTop3(self):
    # [FUNCTION DOC]
    # Random permutation of 6140 linear values, k of five; expectations come from numpy sort.
    k = 5
    inputs = np.random.permutation(np.linspace(0, 100, 6140, dtype=np.float32))
    indices = np.argsort(-inputs)[:k]
    values = -np.sort(-inputs)[:k]
    self._validateTopK(inputs, k, values, indices)

  def testTensorK(self):
    # [FUNCTION DOC]
    # Same as the top-two case but k is given as a constant tensor.
    inputs = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.4, 0.2]]
    k = constant_op.constant(2)
    self._validateTopK(inputs, k, [[0.4, 0.3], [0.4, 0.3]], [[3, 1], [2, 1]])


class InTopKTest(test.TestCase):
  # [CLASS DOC]
  # Tests the in-top-k op, which says whether each target class is among the k highest predictions.

  def _validateInTopK(self, predictions, target, k, expected):
    # [FUNCTION DOC]
    # HELPER: runs in-top-k for predictions, targets and k, and asserts the boolean result equals the
    # KNOWN ISSUE: uses the removed numpy boolean alias to build the expected array; fails on new numpy.
    # expected list and has matching shape.
    np_ans = np.array(expected, np.bool)
    with self.cached_session(use_gpu=True) as _:
      output = nn_ops.in_top_k(predictions, target, k)
      nn_ans = self.evaluate(output)
      self.assertAllEqual(np_ans, nn_ans)
      self.assertShapeEqual(np_ans, output)

  def testInTop1(self):
    # [FUNCTION DOC]
    # Two rows, k of one: expects true for the first row and false for the second.
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [3, 2]
    self._validateInTopK(predictions, target, 1, [True, False])

  def testInTop2(self):
    # [FUNCTION DOC]
    # Two rows, k of two: expects false then true.
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [2, 2]
    self._validateInTopK(predictions, target, 2, [False, True])

  def testInTop2Tie(self):
    # [FUNCTION DOC]
    # Tied scores for classes two and three: both targets count as inside the top two, so both are true.
    # Class 2 and 3 tie for 2nd, so both are considered in top 2.
    predictions = [[0.1, 0.3, 0.2, 0.2], [0.1, 0.3, 0.2, 0.2]]
    target = [2, 3]
    self._validateInTopK(predictions, target, 2, [True, True])

  def testInTop2_int64Target(self):
    # [FUNCTION DOC]
    # Same as the top-two case but targets are 64-bit integers.
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = numpy_compat.np_asarray([0, 2]).astype(np.int64)
    self._validateInTopK(predictions, target, 2, [False, True])

  def testTensorK(self):
    # [FUNCTION DOC]
    # k given as a constant tensor of three; expects false then true.
    predictions = [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.3, 0.4]]
    target = [0, 2]
    k = constant_op.constant(3)
    self._validateInTopK(predictions, target, k, [False, True])


class SplitTest(test.TestCase):
  # [CLASS DOC]
  # Tests the split op with static sizes and random shapes.

  def testSpecialCase2(self):
    # [FUNCTION DOC]
    # Splits an 86 by 2 by 2 by 4 by 4 random array along axis zero into 19 uneven pieces. A regression
    # check for a special case in the split kernel; there is no assertion, it passes if no error occurs.
    # Test that fails if the Special case 2 is enabled in split_v_op.cc
    split_dim = 0
    shape = (86, 2, 2, 4, 4)
    size_splits = [4, 2, 4, 7, 5, 7, 4, 6, 2, 3, 7, 6, 5, 2, 5, 2, 4, 6, 5]
    x = np.random.rand(*shape).astype(np.float32)
    _ = self.evaluate(array_ops.split(x, size_splits, split_dim))

  def testRandomVariableSlices(self):
    # [FUNCTION DOC]
    # Random rank five shape, random split axis and random uneven sizes; the split axis dimension is set
    # to the sum of the sizes. Splits and asserts each piece equals the matching numpy slice.
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    split_dim = np.random.randint(-5, 5)
    num_split = np.random.randint(2, 25)
    size_splits = np.random.randint(2, 8, num_split, dtype=np.int32)
    shape[split_dim] = np.sum(size_splits)
    x = np.random.rand(*shape).astype(np.float32)
    with self.cached_session(use_gpu=True):
      result = self.evaluate(array_ops.split(x, size_splits, split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + size_splits[i])
      offset += size_splits[i]
      self.assertAllEqual(result[i], x[tuple(slices)])

  def testRegularSlices(self):
    # [FUNCTION DOC]
    # Same as above but with equal sized pieces from a requested number of splits.
    shape = np.random.randint(1, 5, size=5)
    split_dim = np.random.randint(-5, 5)
    num_split = np.random.randint(2, 10)
    shape[split_dim] = shape[split_dim] * num_split
    x = np.random.rand(*shape).astype(np.float32)
    with self.cached_session(use_gpu=True):
      result = self.evaluate(array_ops.split(x, num_split, split_dim))
    slices = [slice(0, x) for x in shape]
    offset = 0
    length = shape[split_dim] // num_split
    for i in range(num_split):
      slices[split_dim] = slice(offset, offset + length)
      offset += length
      self.assertAllEqual(result[i], x[tuple(slices)])


class ResizeBilinearTest(test.TestCase):
  # [CLASS DOC]
  # Tests bilinear image resizing with half pixel centers against a Pillow based reference.

  def _testResize(self, x, y, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: resizes a batch with the resize-bilinear op and compares with the PIL reference result,
    # asserting equal shape and close values. Inputs: images, target size pair, optional GPU flag.
    with self.cached_session(use_gpu=use_gpu):
      ans = image_ops.resize_bilinear(x, y, half_pixel_centers=True)
      tf_ans = self.evaluate(ans)
      ref_ans = self._refResize(x, y)
      self.assertAllEqual(tf_ans.shape, ref_ans.shape)
      self.assertAllClose(tf_ans, ref_ans)

  def _refResize(self, x, y):
    # [FUNCTION DOC]
    # HELPER: reference resize using Pillow for each image and channel separately; Pillow takes (width,
    # height) while TensorFlow takes (height, width), so the size pair is swapped. Returns a numpy array.
    # pylint: disable=g-doc-args
    # pylint: disable=g-doc-return-or-yield
    """PIL has to treat each channel separately.

    Additionally it expects the new shape to be given (width, height), where as
    tensorflow expects (height, width)
    """
    resized_array = []
    for array in x:
      img_channels = []
      for channel_ind in range(array.shape[-1]):
        channel = array[:, :, channel_ind]
        pil_img = Image.fromarray(channel)
        resized_img = numpy_compat.np_asarray(
            pil_img.resize(size=(y[1], y[0]), resample=Image.BILINEAR)
        )
        img_channels.append(resized_img)
      img = np.stack(img_channels, axis=-1)
      resized_array.append(img)

    resized_array = np.array(resized_array)
    return resized_array

  def testFloatBasic(self):
    # [FUNCTION DOC]
    # Random 3 by 24 by 24 by 3 float32 batch upscaled to 48 by 48.
    x = np.random.rand(3, 24, 24, 3)
    x = x.astype(np.float32)
    y = numpy_compat.np_asarray([48, 48], dtype=np.int32)
    self._testResize(x, y, use_gpu=True)

  def testFloatUneven(self):
    # [FUNCTION DOC]
    # Random 3 by 24 by 48 by 3 batch resized to 96 by 64 (uneven scaling).
    x = np.random.rand(3, 24, 48, 3)
    x = x.astype(np.float32)
    y = numpy_compat.np_asarray([96, 64])
    self._testResize(x, y, use_gpu=True)

  def testFloatLarge(self):
    # [FUNCTION DOC]
    # Random 3 by 256 by 256 by 3 batch resized to 1024 by 1024.
    x = np.random.rand(3, 256, 256, 3)
    x = x.astype(np.float32)
    y = numpy_compat.np_asarray([1024, 1024])
    self._testResize(x, y, use_gpu=True)


class OneHotTest(test.TestCase):
  # [CLASS DOC]
  # Tests the one-hot op with and without custom on/off values on both axes.

  def _testOneHot(
      self, truth, use_gpu=False, expected_err_re=None, raises=None, **inputs
  ):
    # [FUNCTION DOC]
    # HELPER: builds a one-hot tensor from keyword inputs and checks it. Inputs: expected truth, GPU flag,
    # optional expected op error expression, optional exception type expected, and the op keyword args.
    # If an exception type is given it asserts the op raises it. Otherwise it asserts shape and values
    # equal the truth, or that evaluation raises the expected op error.
    with self.cached_session(use_gpu=use_gpu):
      if raises is not None:
        with self.assertRaises(raises):
          array_ops.one_hot(**inputs)
      else:
        ans = array_ops.one_hot(**inputs)

        if expected_err_re is None:
          tf_ans = self.evaluate(ans)
          self.assertEqual(tf_ans.shape, ans.get_shape())
          self.assertAllEqual(tf_ans, truth)
        else:
          with self.assertRaisesOpError(expected_err_re):
            self.evaluate(ans)

  def _testBothOneHot(self, truth, expected_err_re=None, raises=None, **inputs):
    # [FUNCTION DOC]
    # HELPER: runs _testOneHot with the GPU on and then off.
    self._testOneHot(truth, True, expected_err_re, raises, **inputs)
    self._testOneHot(truth, False, expected_err_re, raises, **inputs)

  def _testBasic(self, dtype):
    # [FUNCTION DOC]
    # HELPER: with indices zero, two, minus one and one, depth three and on/off values of one and minus
    # one, checks default axis and axis zero (transposed truth) for the given dtype.
    indices = numpy_compat.np_asarray([0, 2, -1, 1], dtype=np.int32)
    depth = 3
    on_value = numpy_compat.np_asarray(1.0, dtype=dtype)
    off_value = numpy_compat.np_asarray(-1.0, dtype=dtype)

    truth = numpy_compat.np_asarray(
        [
            [1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
        ],
        dtype=dtype,
    )

    # axis == -1
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        dtype=dtype,
        truth=truth,
    )

    # axis == 0
    self._testBothOneHot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        axis=0,
        dtype=dtype,
        truth=truth.T,
    )  # Output is transpose version in this case

  def _testDefaultBasic(self, dtype):
    # [FUNCTION DOC]
    # HELPER: same indices with default on/off values zero and one, for the last axis and axis zero.
    indices = numpy_compat.np_asarray([0, 2, -1, 1], dtype=np.int32)
    depth = 3

    truth = numpy_compat.np_asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=dtype,
    )

    # axis == -1
    self._testBothOneHot(indices=indices, depth=depth, truth=truth)

    # axis == 0
    self._testBothOneHot(
        indices=indices, depth=depth, axis=0, truth=truth.T
    )  # Output is transpose version in this case

  def testFloatBasic(self):
    # [FUNCTION DOC]
    # Runs both helper scenarios with float32.
    self._testBasic(np.float32)
    self._testDefaultBasic(np.float32)


def get_test_configs():
  # [FUNCTION DOC]
  # Returns the list of device/format configurations that the convolution tests iterate over:
  # a list of pairs of data format name and a flag saying whether to use the GPU. Only the channels-last
  # format is listed, once with the GPU off and once with it on. Takes no input and has no side effects.
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NHWC", False), ("NHWC", True)]
  return test_configs


class Conv2DTest(test.TestCase):
  # [CLASS DOC]
  # Tests the 2D convolution op: forward values (with strides, padding, dilation, explicit padding)
  # and input backprop. Helpers create numpy tensors, build graphs per device and data format, and
  # compare against expected values or reference convolution.

  def _DtypesToTest(self, use_gpu):
    # [FUNCTION DOC]
    # HELPER: chooses the dtypes to test. Input: whether the GPU is used. Returns float32 and double
    # (unless built for ROCm), plus half precision when the GPU is not used or supports half matmul/conv.
    # Float32 must come before float16 because it serves as a reference.
    # double datatype is currently not supported for convolution ops
    # on the ROCm platform
    optional_float64 = [] if test.is_built_with_rocm() else [dtypes.float64]
    if use_gpu and not test_util.GpuSupportsHalfMatMulAndConv():
      return [dtypes.float32] + optional_float64
    else:
      # It is important that float32 comes before float16 here,
      # as we will be using its gradients as reference for fp16 gradients.
      return [dtypes.float32, dtypes.float16] + optional_float64

  def _CreateNumpyTensor(self, shape):
    # [FUNCTION DOC]
    # HELPER: returns a float32 array of the given shape filled with incrementing values from one.
    total_size = 1
    for s in shape:
      total_size *= s
    return np.arange(1, total_size + 1, dtype=np.float32).reshape(shape)

  def _SetupValuesForDevice(
      self,
      tensor_in_sizes,
      filter_in_sizes,
      dilations,
      strides,
      padding,
      data_format,
      dtype,
      use_gpu,
  ):
    # [FUNCTION DOC]
    # HELPER: builds a convolution op for given sizes, dilations, strides, padding, data format and dtype
    # KNOWN ISSUE: docstring claims verification, but the function only constructs and returns the op.
    # on the requested device and returns the result tensor, in channels-last layout.
    # Pads strides and dilations with ones for batch/channel, adds zero padding for those dims when
    # padding is explicit, and converts layouts when the format is channels-first. Asserts the result
    # dtype matches. The docstring says "Verifies the output values", but it only builds and returns
    # the symbolic tensor; comparison is done by callers.
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,
        input_depth, output_depth].
      dilations: Dilated rate: [col_dilation, row_dilation]
      strides: Stride: [col_stride, row_stride]
      padding: Padding type.
      data_format: Format of the data tensors.
      dtype: Data type for inputs and outputs.
      use_gpu: True if the operations should be run on GPU

    Returns:
      Symbolic tensor value that can be used to execute the computation
    """
    x1 = self._CreateNumpyTensor(tensor_in_sizes)
    x2 = self._CreateNumpyTensor(filter_in_sizes)

    with test_util.device(use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtype)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtype)
      strides = [1] + strides + [1]
      dilations = [1] + dilations + [1]
      if isinstance(padding, (list, tuple)):
        padding = [(0, 0)] + padding + [(0, 0)]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
        dilations = test_util.NHWCToNCHW(dilations)
        if isinstance(padding, (list, tuple)):
          padding = test_util.NHWCToNCHW(padding)
      conv = nn_ops.conv2d(
          t1,
          t2,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format,
      )
      self.assertEqual(conv.dtype, dtype)
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)

      return conv

  def _CompareFwdValues(
      self, tensor_in_sizes, filter_in_sizes, conv_strides, padding
  ):
    # [FUNCTION DOC]
    # HELPER: checks that all device/format configurations give close forward convolution results.
    # KNOWN ISSUE: never called in this file (dead helper).
    # Inputs: input sizes, filter sizes, row/col strides and padding. Uses random float32 data, evaluates
    # a convolution for each configuration, and asserts all results are close to the first.
    """Verifies that CPU and GPU produce the same values.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,
        input_depth, output_depth].
      conv_strides: [row_stride, col_stride] for the convolution;
      padding: Padding type.
    """
    x1 = np.random.rand(*tensor_in_sizes).astype(np.float32)
    x2 = np.random.rand(*filter_in_sizes).astype(np.float32)

    def _setup_val(data_format, use_gpu):
      # [FUNCTION DOC]
      # HELPER: builds the convolution tensor for one data format and device flag using the enclosing
      # random inputs, converting layouts for channels-first, and returns the channels-last result.
      with test_util.device(use_gpu):
        t1 = constant_op.constant(x1, shape=tensor_in_sizes)
        t2 = constant_op.constant(x2, shape=filter_in_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t1 = test_util.NHWCToNCHW(t1)
          strides = test_util.NHWCToNCHW(strides)
        conv = nn_ops.conv2d(
            t1, t2, strides=strides, padding=padding, data_format=data_format
        )
        if data_format == "NCHW":
          conv = test_util.NCHWToNHWC(conv)
        return conv

    tensors = []
    for data_format, use_gpu in get_test_configs():
      tensors.append(_setup_val(data_format, use_gpu))
    values = self.evaluate(tensors)
    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-3, atol=1e-3)

  def _ComputeReferenceDilatedConv(
      self,
      tensor_in_sizes,
      filter_in_sizes,
      stride,
      dilation,
      padding,
      data_format,
      use_gpu,
  ):
    # [FUNCTION DOC]
    # HELPER: computes expected and actual dilated convolutions. Inputs: sizes, stride (single value or
    # iterable), dilation list, padding, data format and GPU flag. The expected value comes from the
    # generic convolution op with a dilation rate, the computed one from the conv2d op with dilations.
    # Both are converted to channels-last. Returns the pair (expected, computed).
    x1 = self._CreateNumpyTensor(tensor_in_sizes)
    x2 = self._CreateNumpyTensor(filter_in_sizes)
    with test_util.device(use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)
      if isinstance(stride, collections_abc.Iterable):
        strides = list(stride)
      else:
        strides = [stride, stride]
      if data_format == "NCHW":
        t1 = test_util.NHWCToNCHW(t1)
        full_strides = [1, 1] + strides
        full_dilation = [1, 1] + dilation
      else:
        full_strides = [1] + strides + [1]
        full_dilation = [1] + dilation + [1]
      expected = nn_ops.convolution(
          t1,
          t2,
          padding=padding,
          strides=strides,
          dilation_rate=dilation,
          data_format=data_format,
      )
      computed = nn_ops.conv2d(
          t1,
          t2,
          strides=full_strides,
          dilations=full_dilation,
          padding=padding,
          data_format=data_format,
      )
      if data_format == "NCHW":
        expected = test_util.NCHWToNHWC(expected)
        computed = test_util.NCHWToNHWC(computed)
    return expected, computed

  def _VerifyDilatedConvValues(
      self,
      tensor_in_sizes,
      filter_in_sizes,
      strides,
      padding,
      dilations,
      rtol=1e-4,
  ):
    # [FUNCTION DOC]
    # HELPER: for each configuration compares conv2d with dilations against the generic convolution
    # KNOWN ISSUE: the evaluate/compare code sits inside the loop and evaluates the accumulated lists on
    #   every iteration, so earlier configurations are rechecked (harmless but redundant).
    # reference, with tolerance larger on GPU. Inputs: sizes, strides, padding, dilations, relative
    # tolerance (default 1e-4).
    expected_results = []
    computed_results = []
    for data_format, use_gpu in get_test_configs():
      expected, computed = self._ComputeReferenceDilatedConv(
          tensor_in_sizes,
          filter_in_sizes,
          strides,
          dilations,
          padding,
          data_format,
          use_gpu,
      )
      expected_results.append(expected)
      computed_results.append(computed)
      tolerance = 1e-2 if use_gpu else 1e-5
      expected_values = self.evaluate(expected_results)
      computed_values = self.evaluate(computed_results)
      for e_value, c_value in zip(expected_values, computed_values):
        tf_logging.debug("expected = %s", e_value)
        tf_logging.debug("actual = %s", c_value)
        self.assertAllClose(
            e_value.flatten(), c_value.flatten(), atol=tolerance, rtol=rtol
        )

  def _VerifyValues(
      self,
      tensor_in_sizes,
      filter_in_sizes,
      strides,
      padding,
      expected,
      dilations=(1, 1),
      gpu_only=False,
      test_grappler_layout_optimizer=False,
      tol=1e-5,
      fp16_tol=1e-3,
  ):
    # [FUNCTION DOC]
    # HELPER: core forward verification. Inputs: input and filter sizes, strides, padding, flat expected
    # values, dilations (default one by one), flag gpu_only, flag to keep layout optimizer behavior, a
    # tolerance and a half precision tolerance. Builds the convolution for every configuration and dtype
    # (adding int32 for channels-last when the layout optimizer path is off), evaluates, and compares to
    # the expected values (exact for integers) and the shape and dtype. Returns early if gpu_only is set
    # and no CUDA GPU exists.
    if gpu_only and not test.is_gpu_available(cuda_only=True):
      return
    tensors = []
    dilations = list(dilations)
    for data_format, use_gpu in get_test_configs():
      if gpu_only and not use_gpu:
        continue
      dtypes_to_test = self._DtypesToTest(use_gpu)
      if not test_grappler_layout_optimizer and data_format == "NHWC":
        dtypes_to_test.append(dtypes.int32)
      for dtype in dtypes_to_test:
        result = self._SetupValuesForDevice(
            tensor_in_sizes,
            filter_in_sizes,
            dilations,
            strides,
            padding,
            data_format,
            dtype,
            use_gpu=use_gpu,
        )
        if test_grappler_layout_optimizer and data_format == "NHWC" and use_gpu:
          # Grappler's layout optimizer will not optimize a fetch node, so
          # this identity allows Grappler to optimize the Conv2D node.
          result = array_ops.identity(result)
        tensors.append(result)
      values = self.evaluate(tensors)
      for i in range(len(tensors)):
        conv = tensors[i]
        value = values[i]
        tf_logging.debug("expected = %s", expected)
        tf_logging.debug("actual = %s", value)
        tol_to_use = fp16_tol if value.dtype == np.float16 else tol
        if np.issubdtype(value.dtype, np.integer):
          self.assertAllEqual(np.rint(expected), np.ravel(value))
        else:
          self.assertAllClose(
              expected, np.ravel(value), atol=tol_to_use, rtol=tol_to_use
          )
        self.assertShapeEqual(value, conv)
        self.assertEqual(value.dtype, conv.dtype.as_numpy_dtype)

  def _VerifyExplicitPaddings(
      self,
      tensor_in_sizes,
      filter_in_sizes,
      strides,
      padding,
      dilations=(1, 1),
      test_grappler_layout_optimizer=False,
      tol=1e-5,
      fp16_tol=1e-3,
  ):
    # [FUNCTION DOC]
    # HELPER: verifies convolution with explicit padding by padding the input manually, running a valid
    # convolution to get expected values, then calling _VerifyValues with the explicit padding.
    # Inputs: sizes, strides, padding amounts, dilations, layout optimizer flag, tolerances.
    """Verifies Conv2D with explicit padding generates correct values.

    It does this by comparing with Conv2D without explicit padding. This
    function assumes Conv2D without explicit padding works correctly.

    Args:
      tensor_in_sizes: Input tensor dimensions in [batch, input_rows,
        input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in [kernel_rows, kernel_cols,
        input_depth, output_depth].
      strides: [row_stride, col_stride] for the convolution;
      padding: Explicit padding amounts.
      dilations: Dilation values
      test_grappler_layout_optimizer: If True, allow the Grappler layout
        optimizer to run, which turns NHWC Conv2Ds on the GPU to NCHW Conv2Ds.
      tol: The absolute and relative tolerance for non-fp16 dtypes.
      fp16_tol: The absolute and relative tolerance for fp16.
    """
    input_tensor = self._CreateNumpyTensor(tensor_in_sizes)
    filter_tensor = self._CreateNumpyTensor(filter_in_sizes)
    input_tensor = array_ops.pad(input_tensor, [(0, 0)] + padding + [(0, 0)])
    dilations = list(dilations)
    conv2d_result = nn_ops.conv2d(
        input_tensor,
        filter_tensor,
        [1] + list(strides) + [1],
        "VALID",
        dilations=[1] + dilations + [1],
    )
    expected = list(self.evaluate(array_ops.reshape(conv2d_result, [-1])))
    self._VerifyValues(
        tensor_in_sizes,
        filter_in_sizes,
        strides,
        padding,
        expected,
        dilations,
        test_grappler_layout_optimizer=test_grappler_layout_optimizer,
        tol=tol,
        fp16_tol=fp16_tol,
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2D1x1Filter(self):
    # [FUNCTION DOC]
    # One by one filter on a two by three by three image; expected output given as 18 numbers; valid padding.
    expected_output = [
        30.0,
        36.0,
        42.0,
        66.0,
        81.0,
        96.0,
        102.0,
        126.0,
        150.0,
        138.0,
        171.0,
        204.0,
        174.0,
        216.0,
        258.0,
        210.0,
        261.0,
        312.0,
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output,
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2Filter2x1Dilation(self):
    # [FUNCTION DOC]
    # Two by two filter with dilation two by one on a four by four image, compared against the reference.
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID",
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2DEmpty(self):
    # [FUNCTION DOC]
    # Convolution of an empty batch with one by one filter; expects an empty result.
    expected_output = []
    self._VerifyValues(
        tensor_in_sizes=[0, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output,
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2DEmptyDilation(self):
    # [FUNCTION DOC]
    # Empty batch with dilation two by one, compared against the dilated reference.
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[0, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID",
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2Filter(self):
    # [FUNCTION DOC]
    # Two by two filter on two by three by three input, valid padding; expected six values.
    # The outputs are computed using third_party/py/IPython/notebook.
    expected_output = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output,
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterDilation(self):
    # [FUNCTION DOC]
    # Same sizes with dilation one by two against the reference.
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        dilations=[1, 2],
        padding="VALID",
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2D1x2Filter(self):
    # [FUNCTION DOC]
    # One by two filter, valid padding; expected twelve values.
    # The outputs are computed using third_party/py/IPython/notebook.
    expected_output = [
        231.0,
        252.0,
        273.0,
        384.0,
        423.0,
        462.0,
        690.0,
        765.0,
        840.0,
        843.0,
        936.0,
        1029.0,
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output,
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2D1x2FilterDilation(self):
    # [FUNCTION DOC]
    # One by two filter with dilation two by one against the reference.
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 2, 3, 3],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID",
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterStride2(self):
    # [FUNCTION DOC]
    # Two by two filter with stride two, valid padding; expected three values.
    expected_output = [2271.0, 2367.0, 2463.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="VALID",
        expected=expected_output,
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterStride2Same(self):
    # [FUNCTION DOC]
    # Two by two filter, stride two, same padding; expected six values.
    expected_output = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="SAME",
        expected=expected_output,
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2D2x2FilterStride1x2(self):
    # [FUNCTION DOC]
    # Strides one and two on a three by six input; expected six values.
    expected_output = [58.0, 78.0, 98.0, 118.0, 138.0, 158.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 6, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[1, 2],
        padding="VALID",
        expected=expected_output,
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2DKernelSmallerThanStrideValid(self):
    # [FUNCTION DOC]
    # Two by two kernel and stride three on seven by seven input, valid padding; four values expected.
    expected_output = [65, 95, 275, 305]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[3, 3],
        padding="VALID",
        expected=expected_output,
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2DKernelSmallerThanStrideSame(self):
    # [FUNCTION DOC]
    # Three sub cases with same padding where the kernel is not larger than the stride, checking the
    # strided sampling values.
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1],
        strides=[2, 2],
        padding="SAME",
        expected=[1, 3, 7, 9],
    )

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[1, 1, 1, 1],
        strides=[2, 2],
        padding="SAME",
        expected=[1, 3, 9, 11],
    )

    self._VerifyValues(
        tensor_in_sizes=[1, 4, 4, 1],
        filter_in_sizes=[2, 2, 1, 1],
        strides=[3, 3],
        padding="SAME",
        expected=[44, 28, 41, 16],
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2DKernelSizeMatchesInputSize(self):
    # [FUNCTION DOC]
    # Kernel equal to the input size with two output channels; expects two values.
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 2, 1],
        filter_in_sizes=[2, 2, 1, 2],
        strides=[1, 1],
        padding="VALID",
        expected=[50, 60],
    )

  @test_util.run_in_graph_and_eager_modes
  def testConv2DKernelSizeMatchesInputSizeDilation(self):
    # [FUNCTION DOC]
    # Kernel equal to input size under dilation two by two, compared with the reference.
    self._VerifyDilatedConvValues(
        tensor_in_sizes=[1, 3, 3, 1],
        filter_in_sizes=[2, 2, 1, 2],
        strides=[1, 1],
        dilations=[2, 2],
        padding="VALID",
    )

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D0x0Padding(self):
    # [FUNCTION DOC]
    # Explicit zero padding on two shapes, comparing to a padded valid convolution.
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        padding=[[0, 0], [0, 0]],
    )

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[3, 4, 3, 2],
        filter_in_sizes=[1, 1, 2, 1],
        strides=[2, 2],
        padding=[[0, 0], [0, 0]],
    )

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D1x1Padding(self):
    # [FUNCTION DOC]
    # Explicit padding of one on every side for two shapes.
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[2, 2, 2, 2],
        strides=[1, 1],
        padding=[[1, 1], [1, 1]],
    )

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 2, 1],
        filter_in_sizes=[1, 1, 1, 2],
        strides=[1, 1],
        padding=[[1, 1], [1, 1]],
    )

  @test_util.run_in_graph_and_eager_modes()
  def testConv2D2x2Padding(self):
    # [FUNCTION DOC]
    # Explicit padding of two on every side for two shapes, one with strides two by one.
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 1, 2],
        filter_in_sizes=[2, 1, 2, 1],
        strides=[1, 1],
        padding=[[2, 2], [2, 2]],
    )

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 1, 2],
        filter_in_sizes=[1, 1, 2, 1],
        strides=[2, 1],
        padding=[[2, 2], [2, 2]],
    )

  @test_util.run_in_graph_and_eager_modes()
  def testConv2DOnlyTopRightPadding(self):
    # [FUNCTION DOC]
    # Asymmetric padding (top and right only), with a looser tolerance for the first case.
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 2],
        strides=[1, 1],
        padding=[[1, 0], [0, 2]],
        tol=5e-5,
    )

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 4, 2],
        filter_in_sizes=[2, 2, 2, 2],
        strides=[1, 3],
        padding=[[1, 0], [0, 2]],
    )

  @test_util.run_in_graph_and_eager_modes()
  def testConv2DLotsPadding(self):
    # [FUNCTION DOC]
    # Large asymmetric padding amounts larger than the kernel for two shapes.
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 1, 1, 3],
        filter_in_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        padding=[[3, 4], [4, 2]],
    )

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 1, 1],
        filter_in_sizes=[2, 2, 1, 3],
        strides=[2, 1],
        padding=[[3, 4], [4, 2]],
    )

  @test_util.run_in_graph_and_eager_modes()
  def testConv2DExplicitPaddingWithDilations(self):
    # [FUNCTION DOC]
    # Explicit padding combined with dilations two by one and two by three.
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 3, 2, 1],
        filter_in_sizes=[1, 2, 1, 2],
        strides=[1, 1],
        padding=[[1, 0], [0, 1]],
        dilations=[2, 1],
    )

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[3, 2, 2, 1],
        strides=[1, 1],
        padding=[[2, 1], [1, 2]],
        dilations=[2, 3],
    )

  def testConv2DExplicitPaddingWithLayoutOptimizer(self):
    # [FUNCTION DOC]
    # Explicit padding with the layout optimizer flag enabled, two shapes, checking the optimizer
    # handles padding correctly.
    # Test with Grappler's layout optimizer, to ensure the layout optimizer
    # handles explicit padding correctly.
    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 3, 2, 1],
        filter_in_sizes=[1, 2, 1, 2],
        strides=[1, 1],
        padding=[[1, 0], [0, 1]],
        dilations=[2, 1],
        test_grappler_layout_optimizer=True,
    )

    self._VerifyExplicitPaddings(
        tensor_in_sizes=[1, 2, 3, 2],
        filter_in_sizes=[3, 2, 2, 1],
        strides=[1, 1],
        padding=[[2, 1], [1, 2]],
        dilations=[2, 3],
        test_grappler_layout_optimizer=True,
    )

  # Testing for backprops
  def _RunAndVerifyBackpropInput(
      self,
      input_sizes,
      filter_sizes,
      output_sizes,
      strides,
      padding,
      expected,
      data_format,
      use_gpu,
      err,
      dilations=(1, 1),
  ):
    # [FUNCTION DOC]
    # HELPER: runs the conv2d input backprop op and checks the flattened result against expected values.
    # KNOWN ISSUE: the parameter err is accepted but never used; tolerance is hard coded.
    # Inputs: input sizes, filter sizes, output sizes, strides, padding, expected, data format, GPU flag,
    # an error value, dilations (default one by one). Returns early when the GPU flag is set but no CUDA
    # GPU exists. Converts layouts for channels-first, evaluates and asserts shape and closeness (atol 1e-5).
    if use_gpu and not test.is_gpu_available(cuda_only=True):
      return
    x1 = self._CreateNumpyTensor(filter_sizes)
    x2 = self._CreateNumpyTensor(output_sizes)
    dilations = list(dilations)
    with test_util.device(use_gpu):
      if data_format == "NCHW":
        input_sizes = test_util.NHWCToNCHW(input_sizes)
      t0 = constant_op.constant(input_sizes, shape=[len(input_sizes)])
      t1 = constant_op.constant(x1, shape=filter_sizes)
      t2 = constant_op.constant(x2, shape=output_sizes)
      strides = [1] + strides + [1]
      dilations = [1] + dilations + [1]
      if isinstance(padding, (list, tuple)):
        padding = [(0, 0)] + padding + [(0, 0)]
      if data_format == "NCHW":
        t2 = test_util.NHWCToNCHW(t2)
        strides = test_util.NHWCToNCHW(strides)
        dilations = test_util.NHWCToNCHW(dilations)
        if isinstance(padding, (list, tuple)):
          padding = test_util.NHWCToNCHW((padding))
      conv = nn_ops.conv2d_backprop_input(
          t0,
          t1,
          t2,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations,
      )
      if data_format == "NCHW":
        conv = test_util.NCHWToNHWC(conv)
      # "values" consists of two tensors for two backprops
      value = self.evaluate(conv)
      self.assertShapeEqual(value, conv)
    tf_logging.debug("expected = %s", expected)
    tf_logging.debug("actual = %s", value)
    self.assertAllCloseAccordingToType(expected, value.flatten(), atol=1e-5)

  def _CompareBackpropInput(
      self, input_sizes, filter_sizes, output_sizes, conv_strides, padding
  ):
    # [FUNCTION DOC]
    # HELPER: compares input backprop across all configurations; inputs are sizes, strides, padding.
    # KNOWN ISSUE: never called in this file (dead helper).
    x1 = np.random.rand(*filter_sizes).astype(np.float32)
    x2 = np.random.rand(*output_sizes).astype(np.float32)

    def _get_val(data_format, use_gpu):
      # [FUNCTION DOC]
      # HELPER: builds and evaluates the backprop op for one data format and GPU flag using enclosing random
      # data, asserts the shape matches, and returns the numpy value.
      with test_util.device(use_gpu):
        if data_format == "NCHW":
          new_input_sizes = test_util.NHWCToNCHW(input_sizes)
        else:
          new_input_sizes = input_sizes
        t0 = constant_op.constant(new_input_sizes, shape=[len(new_input_sizes)])
        t1 = constant_op.constant(x1, shape=filter_sizes)
        t2 = constant_op.constant(x2, shape=output_sizes)
        strides = [1] + conv_strides + [1]
        if data_format == "NCHW":
          t2 = test_util.NHWCToNCHW(t2)
          strides = test_util.NHWCToNCHW(strides)
        conv = nn_ops.conv2d_backprop_input(
            t0,
            t1,
            t2,
            strides=strides,
            padding=padding,
            data_format=data_format,
        )
        if data_format == "NCHW":
          conv = test_util.NCHWToNHWC(conv)
        ret = self.evaluate(conv)
        self.assertShapeEqual(ret, conv)
        return ret

    values = []
    for data_format, use_gpu in get_test_configs():
      values.append(_get_val(data_format, use_gpu))

    for i in range(1, len(values)):
      self.assertAllClose(values[0], values[i], rtol=1e-2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def testConv2DEmptyBackpropInput(self):
    # [FUNCTION DOC]
    # Backprop input with an empty batch; expects an empty result for each configuration.
    expected_output = []
    for data_format, use_gpu in get_test_configs():
      self._RunAndVerifyBackpropInput(
          input_sizes=[0, 2, 3, 1],
          filter_sizes=[2, 2, 1, 1],
          output_sizes=[0, 1, 2, 1],
          strides=[1, 1],
          padding="VALID",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu,
          err=1e-5,
      )

  @test_util.run_in_graph_and_eager_modes
  def testConv2DStrideTwoFilterOneSameBackpropInput(self):
    # [FUNCTION DOC]
    # Stride two, one by one filter, same padding: expects the output gradient scattered onto every
    # second position of a four by four map (sixteen expected values) for each configuration.
    expected_output = [
        1.0,
        0.0,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0,
        0.0,
        4.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    for data_format, use_gpu in get_test_configs():
      self._RunAndVerifyBackpropInput(
          input_sizes=[1, 4, 4, 1],
          filter_sizes=[1, 1, 1, 1],
          output_sizes=[1, 2, 2, 1],
          strides=[2, 2],
          padding="SAME",
          expected=expected_output,
          data_format=data_format,
          use_gpu=use_gpu,
          err=1e-5,
      )


class PoolingTest(test.TestCase):
  # [CLASS DOC]
  # Tests the generic pool op against the numpy reference, for forward values and gradient error.

  def _test(self, input_shape, dtype, **kwargs):
    # [FUNCTION DOC]
    # HELPER: forward check. Input shape, dtype and keyword args for the pool op. Builds negative
    # incrementing data (so zero padding would show), computes the reference and the op, asserts close.
    # Use negative numbers to make sure there isn't any zero padding getting
    # used.
    x = -np.arange(np.prod(input_shape), dtype=dtype).reshape(input_shape) - 1
    y1 = pool_direct(input=x, **kwargs)
    y2 = nn_ops.pool(input=x, **kwargs)
    self.assertAllClose(y1, self.evaluate(y2), rtol=1e-2, atol=1e-2)

  def _test_gradient(self, input_shape, dtype, **kwargs):
    # [FUNCTION DOC]
    # HELPER: gradient check. Same inputs; computes the numeric gradient error of the pool op and asserts
    # it is under 1e-2 (1.1 for half precision, noted as too high in a TODO).
    x_val = (
        -np.arange(np.prod(input_shape), dtype=dtype).reshape(input_shape) - 1
    )
    x = constant_op.constant(x_val, name="x", dtype=dtype)
    output = nn_ops.pool(input=x, **kwargs)
    y_shape = output.get_shape().as_list()
    err = gradient_checker.compute_gradient_error(
        [x], [input_shape], output, y_shape, x_init_value=[x_val]
    )
    err_tolerance = 1e-2
    if dtype == dtypes.float16:
      err_tolerance = 1.1
      # TODO: this is too high.
      # investigate precision issues.
    self.assertLess(err, err_tolerance)

  def testPoolSimple(self):
    # [FUNCTION DOC]
    # One dimensional style pooling on a one by one by ten by one tensor for both padding modes, both
    # pooling types and float32/float16.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for pooling_type in ["MAX", "AVG"]:
          for dtype in [np.float32, np.float16]:
            self._test(
                input_shape=[1, 1, 10, 1],
                window_shape=[1, 3],
                padding=padding,
                pooling_type=pooling_type,
                dilation_rate=[1, 1],
                strides=[1, 2],
                dtype=dtype,
            )

  def testPool1D(self):
    # [FUNCTION DOC]
    # Exhaustive 1D pooling over paddings, dtypes, pooling types, input and window shapes, dilation
    # rates (valid only) and strides not larger than the window.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for dtype in [np.float32, np.float16]:
          for pooling_type in ["MAX", "AVG"]:
            for input_shape in [[2, 9, 2], [2, 10, 2]]:
              for window_shape in [[1], [2], [3]]:
                if padding != "SAME":
                  for dilation_rate in [[1], [2], [3]]:
                    self._test(
                        input_shape=input_shape,
                        window_shape=window_shape,
                        padding=padding,
                        pooling_type=pooling_type,
                        dilation_rate=dilation_rate,
                        strides=[1],
                        dtype=dtype,
                    )
                for strides in [[1], [2], [3]]:
                  if np.any(np.array(strides) > window_shape):
                    continue
                  self._test(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=[1],
                      strides=strides,
                      dtype=dtype,
                  )

  def testPool2D(self):
    # [FUNCTION DOC]
    # Same as the 1D test but with two spatial dimensions.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for dtype in [np.float32, np.float16]:
          for pooling_type in ["MAX", "AVG"]:
            for input_shape in [[2, 9, 10, 2], [2, 10, 9, 2]]:
              for window_shape in [[1, 1], [2, 1], [2, 3]]:
                if padding != "SAME":
                  for dilation_rate in [[1, 1], [2, 1], [1, 2], [2, 3]]:
                    self._test(
                        input_shape=input_shape,
                        window_shape=window_shape,
                        padding=padding,
                        pooling_type=pooling_type,
                        dilation_rate=dilation_rate,
                        strides=[1, 1],
                        dtype=dtype,
                    )
                for strides in [[1, 1], [2, 1], [1, 2], [2, 3]]:
                  if np.any(np.array(strides) > window_shape):
                    continue
                  self._test(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=[1, 1],
                      strides=strides,
                      dtype=dtype,
                  )

  @test_util.run_deprecated_v1
  def testGradient2D(self):
    # [FUNCTION DOC]
    # Deprecated-v1 test of 2D pooling gradients across paddings, dtypes, pooling types, shapes, windows,
    # dilations and strides.
    with self.session(use_gpu=test.is_gpu_available()):
      for padding in ["SAME", "VALID"]:
        for dtype in [np.float32, np.float16]:
          for pooling_type in ["AVG", "MAX"]:
            for input_shape in [[2, 4, 5, 2], [1, 5, 4, 1]]:
              for window_shape in [[1, 1], [2, 1], [2, 2]]:
                if padding != "SAME":
                  for dilation_rate in [[1, 1], [2, 1], [2, 2]]:
                    self._test_gradient(
                        input_shape=input_shape,
                        window_shape=window_shape,
                        padding=padding,
                        pooling_type=pooling_type,
                        dilation_rate=dilation_rate,
                        strides=[1, 1],
                        dtype=dtype,
                    )
                for strides in [[1, 1], [2, 1], [1, 2], [2, 2]]:
                  if np.any(np.array(strides) > window_shape):
                    continue
                  self._test_gradient(
                      input_shape=input_shape,
                      window_shape=window_shape,
                      padding=padding,
                      pooling_type=pooling_type,
                      dilation_rate=[1, 1],
                      strides=strides,
                      dtype=dtype,
                  )


class FractionalMaxPoolGradTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: name and test method suggest fractional max pool, but the code only uses the plain
  #   max pool and its grad op.
  # Despite the name, tests the gradient of regular max pooling with non overlapping windows.
  # Class attributes: a seeded random state and a seed.
  _PRNG = np.random.RandomState(341261)
  _SEED = 123456

  def _GenerateUniqueRandomInputTensor(self, shape):
    # [FUNCTION DOC]
    # HELPER: builds a shuffled array of distinct float32 values with the given shape, so the max is unique.
    num_elements = 1
    for size in shape:
      num_elements *= size
    x = np.arange(num_elements, dtype=np.float32)
    self._PRNG.shuffle(x)
    return x.reshape(shape)

  def testDirectNotUseOverlapping(self):
    # [FUNCTION DOC]
    # For several window sizes, builds a tensor equal to one window, runs max pool with stride equal to
    # the window, builds random output backprop and runs the max pool grad op. There is no value
    # assertion; it passes if evaluation succeeds.
    for num_batches in [1]:
      for row_window_size in [2, 5]:
        for col_window_size in [2, 4]:
          num_rows = row_window_size
          num_cols = col_window_size
          for num_channels in [1]:
            input_shape = (num_batches, num_rows, num_cols, num_channels)
            with self.cached_session() as _:
              input_tensor = constant_op.constant(
                  self._GenerateUniqueRandomInputTensor(input_shape)
              )
              window_size = [1, row_window_size, col_window_size, 1]
              stride_size = [1, row_window_size, col_window_size, 1]
              padding = "VALID"
              output_tensor = nn_ops.max_pool(
                  input_tensor, window_size, stride_size, padding
              )
              output_data = self.evaluate(output_tensor)
              output_backprop = self._PRNG.randint(100, size=output_data.shape)
              input_backprop_tensor = gen_nn_ops.max_pool_grad(
                  input_tensor,
                  output_tensor,
                  output_backprop,
                  window_size,
                  stride_size,
                  padding,
              )
              _ = self.evaluate(input_backprop_tensor)


class RandomOpTestCommon(test.TestCase):
  # [CLASS DOC]
  # Base class holding a shared check used by random op tests.

  # Checks that executing the same rng_func multiple times rarely produces the
  # same result.
  def _testSingleSessionNotConstant(
      self,
      rng_func,
      num,
      dtype,
      min_or_mean,
      max_or_stddev,
      use_gpu,
      op_seed=None,
      graph_seed=None,
  ):
    # [FUNCTION DOC]
    # HELPER: evaluates the same random op three times in one session and asserts the three outputs are
    # not all identical. Inputs: the generating callable, element count, dtype, min or mean, max or
    # standard deviation, GPU flag, optional op seed and optional graph seed (which is set on the graph).
    with self.session(use_gpu=use_gpu, graph=ops.Graph()) as _:
      if graph_seed is not None:
        random_seed.set_random_seed(graph_seed)
      x = rng_func([num], min_or_mean, max_or_stddev, dtype=dtype, seed=op_seed)

      y = self.evaluate(x)
      z = self.evaluate(x)
      w = self.evaluate(x)

      # We use exact equality here. If the random-number generator is producing
      # the same output, all three outputs will be bitwise identical.
      self.assertTrue(
          (not np.array_equal(y, z))
          or (not np.array_equal(z, w))
          or (not np.array_equal(y, w))
      )


@test_util.for_all_test_methods(
    test_util.disable_xla, "This never passed on XLA"
)
class RandomUniformTest(RandomOpTestCommon):
  # [CLASS DOC]
  # Tests the uniform random op for range, distinctness, integer uniformity, seeds, determinism and
  # common subexpression behavior. Decorated at class level to disable XLA runs since it never passed.

  def _Sampler(self, num, minv, maxv, dtype, use_gpu, seed=None):
    # [FUNCTION DOC]
    # HELPER: returns a callable that draws ten rounds of samples from the uniform op in a fresh graph.
    # Inputs: count, min, max, dtype, GPU flag, optional seed.
    def func():
      # [FUNCTION DOC]
      # HELPER: the callable returned by the sampler; builds the op in a new graph and returns a ten row
      # numpy array of evaluated samples.
      with self.session(use_gpu=use_gpu, graph=ops.Graph()) as _:
        rng = random_ops.random_uniform(
            [num], minval=minv, maxval=maxv, dtype=dtype, seed=seed
        )
        ret = np.empty([10, num])
        for i in xrange(10):
          ret[i, :] = self.evaluate(rng)
      return ret

    return func

  def testRange(self):
    # [FUNCTION DOC]
    # For five dtypes draws 1000 values in the range minus two to eight and asserts bounds.
    for dt in (
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.int32,
        dtypes.int64,
    ):
      sampler = self._Sampler(1000, minv=-2, maxv=8, dtype=dt, use_gpu=True)
      x = sampler()
      self.assertLessEqual(-2, np.min(x))
      self.assertLess(np.max(x), 8)

  # Asserts that different trials (1000 samples per trial) is unlikely
  # to see the same sequence of values. Will catch buggy
  # implementations which uses the same random number seed.
  def testDistinct(self):
    # [FUNCTION DOC]
    # Draws two samples per dtype and asserts few equal entries, catching implementations that reuse
    # the seed. Prints diagnostic values when too many are equal.
    for dt in (
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.int32,
        dtypes.int64,
    ):
      maxv = 1.0 if dt.is_floating else 1 << 30
      sampler = self._Sampler(1000, minv=0, maxv=maxv, dtype=dt, use_gpu=True)
      x = sampler()
      y = sampler()
      count = (x == y).sum()
      count_limit = 50 if dt == dtypes.float16 else 10
      if count >= count_limit:
        print("x = ", x)
        print("y = ", y)
        print("count = ", count)
      self.assertLess(count, count_limit)

  @test_util.run_deprecated_v1
  def testUniformIntsWithInvalidShape(self):
    # [FUNCTION DOC]
    # Asserts that integer uniform with non scalar min or max raises a value error with a specific message.
    for dtype in dtypes.int32, dtypes.int64:
      with self.assertRaisesRegex(
          ValueError, "minval must be a scalar; got a tensor of shape"
      ):
        random_ops.random_uniform([1000], minval=[1, 2], maxval=3, dtype=dtype)
      with self.assertRaisesRegex(
          ValueError, "maxval must be a scalar; got a tensor of shape"
      ):
        random_ops.random_uniform([1000], minval=1, maxval=[2, 3], dtype=dtype)

  # Check that uniform ints actually follow a uniform distribution.
  @test_util.run_deprecated_v1
  def testUniformInts(self):
    # [FUNCTION DOC]
    # Draws 100000 integers in minus two to fifteen with a fixed seed and checks the histogram counts
    # are within five standard deviations of the binomial mean.
    minv = -2
    maxv = 15
    n = 100000
    p = 1 / (maxv - minv)
    # The counts should follow an (n, p) binomial distribution.
    mean = p * n
    std = np.sqrt(n * p * (1 - p))
    for dt in dtypes.int32, dtypes.int64:
      # Use a fixed seed here to make the test deterministic.
      # Without the fixed seed, the 5 * std bound will (very rarely) fail.
      sampler = self._Sampler(
          n // 10, minv=minv, maxv=maxv, dtype=dt, use_gpu=True, seed=17
      )
      x = sampler().ravel()
      self.assertEqual(x.shape, (n,))
      counts, _ = np.histogram(x, bins=maxv - minv)
      self.assertEqual(counts.shape, (maxv - minv,))
      self.assertEqual(counts.sum(), n)
      error = np.abs(counts - mean)
      self.assertLess(error.max(), 5 * std)

  # Check that minval = maxval is fine iff we're producing no numbers
  def testUniformIntsDegenerate(self):
    # [FUNCTION DOC]
    # Min equal to max is fine only when zero numbers are produced: zero samples give an empty result of
    # the right shape, one sample raises an op error saying min must be less than max.
    for dt in dtypes.int32, dtypes.int64:

      def sample(n, dtype=dt):
        # [FUNCTION DOC]
        # HELPER: draws count samples with min and max zero using the sampler; dtype default from the loop.
        return self._Sampler(n, minv=0, maxv=0, dtype=dtype, use_gpu=True)()

      self.assertEqual(sample(0, dt).shape, (10, 0))
      with self.assertRaisesOpError("Need minval < maxval, got 0 >= 0"):
        sample(1)

  @test_util.run_deprecated_v1
  def testSeed(self):
    # [FUNCTION DOC]
    # With the same seed (including huge ones) two samplers produce equal outputs for all dtypes.
    for dt in (
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.int32,
        dtypes.int64,
    ):
      for seed in [345, 2**100, -(2**100)]:
        sx = self._Sampler(1000, 0, 17, dtype=dt, use_gpu=True, seed=seed)
        sy = self._Sampler(1000, 0, 17, dtype=dt, use_gpu=True, seed=seed)
        self.assertAllEqual(sx(), sy())

  @test_util.run_deprecated_v1
  def testNoCSE(self):
    # [FUNCTION DOC]
    # Two independent uniform ops of equal shape must differ (not merged by common subexpression
    # elimination); asserts norm of the difference exceeds 0.1.
    shape = [2, 3, 4]
    for dtype in dtypes.float16, dtypes.float32, dtypes.int32:
      with self.session(use_gpu=True):
        rnd1 = random_ops.random_uniform(shape, 0, 17, dtype=dtype)
        rnd2 = random_ops.random_uniform(shape, 0, 17, dtype=dtype)
        diff = (rnd2 - rnd1).eval()
        self.assertGreater(np.linalg.norm(diff), 0.1)

  @test_util.run_deprecated_v1
  def testSingleSessionNotConstant(self):
    # [FUNCTION DOC]
    # Uses the common check for every dtype and both device flags without seeds.
    for use_gpu in [False, True]:
      for dt in (
          dtypes.float16,
          dtypes.float32,
          dtypes.float64,
          dtypes.int32,
          dtypes.int64,
      ):
        self._testSingleSessionNotConstant(
            random_ops.random_uniform, 100, dt, 0, 17, use_gpu=use_gpu
        )

  @test_util.run_deprecated_v1
  def testSingleSessionOpSeedNotConstant(self):
    # [FUNCTION DOC]
    # Same with an op seed of 1345 and range 10 to 20.
    for use_gpu in [False, True]:
      for dt in (
          dtypes.float16,
          dtypes.float32,
          dtypes.float64,
          dtypes.int32,
          dtypes.int64,
      ):
        self._testSingleSessionNotConstant(
            random_ops.random_uniform,
            100,
            dt,
            10,
            20,
            use_gpu=use_gpu,
            op_seed=1345,
        )

  @test_util.run_deprecated_v1
  def testSingleSessionGraphSeedNotConstant(self):
    # [FUNCTION DOC]
    # Same with a graph seed of 965 and range 20 to 200.
    for use_gpu in [False, True]:
      for dt in (
          dtypes.float16,
          dtypes.float32,
          dtypes.float64,
          dtypes.int32,
          dtypes.int64,
      ):
        self._testSingleSessionNotConstant(
            random_ops.random_uniform,
            100,
            dt,
            20,
            200,
            use_gpu=use_gpu,
            graph_seed=965,
        )


class BroadcastToTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the broadcast-to op: values and static shapes for many dtypes and ranks, error handling for
  # impossible shapes, and gradient error for several broadcast patterns.

  @test_util.run_deprecated_v1
  def testBroadcastToBasic(self):
    # [FUNCTION DOC]
    # Broadcasts a three element vector of each small integer dtype to three by three and compares with
    # numpy broadcast.
    for dtype in [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64]:
      with self.session(use_gpu=True):
        x = np.array([1, 2, 3], dtype=dtype)
        v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToString(self):
    # [FUNCTION DOC]
    # Broadcasts a vector of byte strings to three by three and compares with numpy.
    with self.session(use_gpu=True):
      x = np.array([b"1", b"2", b"3"])
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToBool(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: builds the input with the removed numpy boolean alias; fails on new numpy.
    # Broadcasts a boolean vector to three by three and compares with numpy.
    with self.session(use_gpu=True):
      x = np.array([True, False, True], dtype=np.bool)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShape(self):
    # [FUNCTION DOC]
    # For every input rank from one to five and output rank from that up to five (all dims of size
    # two) broadcasts a random integer array and compares with numpy.
    for input_dim in range(1, 6):
      for output_dim in range(input_dim, 6):
        with self.cached_session(use_gpu=True):
          input_shape = [2] * input_dim
          output_shape = [2] * output_dim
          x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
          v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
          v_np = np.broadcast_to(x, output_shape)
          self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeInnerDim(self):
    # [FUNCTION DOC]
    # Expands an inner dimension from one to five (shape two, one, three to two, five, three).
    input_shape = [2, 1, 3]
    output_shape = [2, 5, 3]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeLargerDim(self):
    # [FUNCTION DOC]
    # Broadcasts a rank six array to a rank fifteen shape that has many leading size one dims.
    input_shape = [2, 1, 3, 2, 2, 2]
    output_shape = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 15, 3, 2, 2, 2]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeLargerDim2(self):
    # [FUNCTION DOC]
    # Broadcasts a rank nine array to a rank twelve shape with size changes in several dims.
    input_shape = [2, 1, 3, 2, 2, 2, 1, 1, 1]
    output_shape = [1, 1, 1, 2, 5, 3, 2, 2, 2, 3, 3, 3]
    with self.cached_session(use_gpu=True):
      x = np.array(np.random.randint(5, size=input_shape), dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), output_shape)
      v_np = np.broadcast_to(x, output_shape)
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToScalar(self):
    # [FUNCTION DOC]
    # Broadcasts a scalar integer to three by three and compares with numpy.
    with self.session(use_gpu=True):
      x = np.array(1, dtype=np.int32)
      v_tf = array_ops.broadcast_to(constant_op.constant(x), [3, 3])
      v_np = np.broadcast_to(x, [3, 3])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastScalarToNonScalar(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the removed numpy float alias to build the reference; fails on new numpy.
    # Broadcasts the scalar 1.0 to a rank six shape and compares with numpy.
    with self.session(use_gpu=True):
      x = np.array(1.0, dtype=np.float)
      v_tf = array_ops.broadcast_to(
          constant_op.constant(1.0), [2, 3, 4, 1, 1, 1]
      )
      v_np = np.broadcast_to(x, [2, 3, 4, 1, 1, 1])
      self.assertAllEqual(v_tf.eval(), v_np)

  @test_util.run_deprecated_v1
  def testBroadcastToShapeTypeAndInference(self):
    # [FUNCTION DOC]
    # For a 32-bit and a 64-bit shape tensor, broadcasts a vector to three by three and asserts both
    # the value and the statically inferred shape equal the numpy result.
    for dtype in [dtypes.int32, dtypes.int64]:
      with self.cached_session(use_gpu=True):
        x = np.array([1, 2, 3])
        v_tf = array_ops.broadcast_to(
            constant_op.constant(x), constant_op.constant([3, 3], dtype=dtype)
        )
        shape = v_tf.get_shape().as_list()
        v_np = np.broadcast_to(x, [3, 3])
        self.assertAllEqual(v_tf.eval(), v_np)
        # check shape inference when shape input is constant
        self.assertAllEqual(shape, v_np.shape)

  def testBroadcastToBadOutputShape(self):
    # [FUNCTION DOC]
    # In eager mode, broadcasting a two element vector to shape two by one must raise an invalid
    # argument error mentioning inability to broadcast.
    with context.eager_mode():
      with self.assertRaisesRegex(
          errors.InvalidArgumentError, "Unable to broadcast tensor of shape"
      ):
        self.evaluate(
            array_ops.broadcast_to(
                constant_op.constant([0, 1]), constant_op.constant([2, 1])
            )
        )

  @test_util.run_deprecated_v1
  def testGradientForScalar(self):
    # [FUNCTION DOC]
    # Gradient error check (below 1e-4) of twice a scalar broadcast to two by four by three.
    x = constant_op.constant(1, dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [2, 4, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(
          x, x.get_shape(), out, out.get_shape()
      )
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithSameRank(self):
    # [FUNCTION DOC]
    # Gradient error check when input and output have the same rank (two, one, three to two, five, three).
    x = constant_op.constant(
        np.reshape(np.arange(6), (2, 1, 3)), dtype=dtypes.float32
    )
    v = array_ops.broadcast_to(x, [2, 5, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(
          x, x.get_shape(), out, out.get_shape()
      )
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithIncreasingRank(self):
    # [FUNCTION DOC]
    # Gradient error check when the output has higher rank than the input.
    x = constant_op.constant([[1], [2]], dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [5, 2, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(
          x, x.get_shape(), out, out.get_shape()
      )
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithBroadcastAllDimensions(self):
    # [FUNCTION DOC]
    # Gradient error check when a one element input is broadcast along all dimensions.
    x = constant_op.constant([1], dtype=dtypes.float32)
    v = array_ops.broadcast_to(x, [5, 2, 3])
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(
          x, x.get_shape(), out, out.get_shape()
      )
    self.assertLess(err, 1e-4)

  @test_util.run_deprecated_v1
  def testGradientWithLargeDim(self):
    # [FUNCTION DOC]
    # Gradient error check for a rank nine input broadcast to a rank twelve output.
    input_shape = [2, 1, 3, 2, 2, 2, 1, 1, 1]
    output_shape = [1, 1, 1, 2, 5, 3, 2, 2, 2, 3, 3, 3]
    x = constant_op.constant(
        np.array(np.random.randn(*input_shape), dtype=np.float32)
    )
    v = array_ops.broadcast_to(x, output_shape)
    out = 2 * v
    with self.cached_session():
      err = gradient_checker.compute_gradient_error(
          x, x.get_shape(), out, out.get_shape()
      )
    self.assertLess(err, 1e-4)


class GPUBinaryOpsTest(test.TestCase):
  # [CLASS DOC]
  # Compares binary arithmetic ops run on the GPU with CPU results.

  def _compareGPU(self, x, y, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: runs the given TensorFlow binary function on the GPU-preferring session, then on a CPU
    # session, with the same inputs, and asserts the two results are close. The numpy function argument
    # is accepted but unused.
    with self.cached_session(use_gpu=True) as _:
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = self.evaluate(out)

    with self.cached_session(use_gpu=False) as _:
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = self.evaluate(out)

    self.assertAllClose(tf_cpu, tf_gpu)

  def testFloatBasic(self):
    # [FUNCTION DOC]
    # Compares add, subtract, multiply, true division, floor division and power for float32 vectors.
    x = np.linspace(-5, 20, 15).reshape((1, 3, 5)).astype(np.float32)
    y = np.linspace(20, -5, 15).reshape((1, 3, 5)).astype(np.float32)
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareGPU(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareGPU(x, y, np.power, math_ops.pow)

  def testFloatWithBCast(self):
    # [FUNCTION DOC]
    # Same as the basic case for add, subtract, multiply and divide, with broadcasting between shapes
    # three by five and two by three by five.
    x = np.linspace(-5, 20, 15).reshape((3, 5)).astype(np.float32)
    y = np.linspace(20, -5, 30).reshape((2, 3, 5)).astype(np.float32)
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)

  def testHalfBasic(self):
    # [FUNCTION DOC]
    # Same operations as the float test but with float16 data.
    x = (
        np.linspace(-5, 20, 15, dtype=np.float16)
        .reshape((1, 3, 5))
        .astype(np.float16)
    )
    y = (
        np.linspace(20, -5, 15, dtype=np.float16)
        .reshape((1, 3, 5))
        .astype(np.float16)
    )
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareGPU(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareGPU(x, y, np.power, math_ops.pow)


class LogicalOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests logical and, or, xor and not against numpy on CPU and GPU, including broadcasting.

  def _compareBinary(self, x, y, np_func, tf_func, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: applies a TensorFlow binary function on the chosen device and asserts the dtype is boolean,
    # values equal numpy's result and shapes match.
    np_ans = np_func(x, y)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_val = self.evaluate(out)
    self.assertEqual(out.dtype, dtypes.bool)
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def _not(self, x, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: same as above for logical not.
    np_ans = np.logical_not(x)
    with test_util.device(use_gpu=use_gpu):
      out = math_ops.logical_not(ops.convert_to_tensor(x))
      tf_val = self.evaluate(out)
    self.assertEqual(out.dtype, dtypes.bool)
    self.assertAllEqual(np_ans, tf_val)
    self.assertShapeEqual(np_ans, out)

  def testScalar(self):
    # [FUNCTION DOC]
    # All combinations of true/false single element arrays for not, and, or, xor on both devices.
    data = [np.array([True]), np.array([False])]
    for use_gpu in [True, False]:
      for x in data:
        self._not(x, use_gpu)
      for x in data:
        for y in data:
          self._compareBinary(
              x, y, np.logical_and, math_ops.logical_and, use_gpu
          )
          self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
          self._compareBinary(
              x, y, np.logical_xor, math_ops.logical_xor, use_gpu
          )

  def testTensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the removed numpy boolean alias to cast random data; fails on new numpy.
    # Random boolean arrays of shape one by three by two for all four ops on both devices.
    x = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    y = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    for use_gpu in [True, False]:
      self._not(x, use_gpu)
      self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
      self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
      self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor, use_gpu)

  def testBCast(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the removed numpy boolean alias to cast random data; fails on new numpy.
    # Binary ops over a list of shape pairs requiring broadcasting, including zero sized dimensions.
    shapes = [
        ([1, 3, 2], [1]),
        ([1, 3, 2], [2]),
        ([1, 3, 2], [3, 2]),
        ([1, 3, 2], [3, 1]),
        ([1, 3, 2], [1, 3, 2]),
        ([1, 3, 2], [2, 3, 1]),
        ([1, 3, 2], [2, 1, 1]),
        ([1, 3, 2], [1, 3, 1]),
        ([2, 1, 5], [2, 3, 1]),
        ([2, 0, 5], [2, 0, 1]),
        ([2, 3, 0], [2, 3, 1]),
    ]
    for xs, ys in shapes:
      x = np.random.randint(0, 2, np.prod(xs)).astype(np.bool).reshape(xs)
      y = np.random.randint(0, 2, np.prod(ys)).astype(np.bool).reshape(ys)
      for use_gpu in [True, False]:
        self._compareBinary(x, y, np.logical_and, math_ops.logical_and, use_gpu)
        self._compareBinary(x, y, np.logical_or, math_ops.logical_or, use_gpu)
        self._compareBinary(x, y, np.logical_xor, math_ops.logical_xor, use_gpu)


class XentTest(test.TestCase):
  # [CLASS DOC]
  # Tests the softmax cross-entropy op's backprop against a numpy reference.

  def _npXent(self, features, labels, dim=-1):
    # [FUNCTION DOC]
    # HELPER: numpy reference. Inputs: features, labels and a dimension (default last). Computes
    # softmax probabilities with max subtraction for stability, the backprop (probabilities minus labels)
    # and the loss as negative sum of labels times log probabilities. Prints the dimension and the
    # intermediate product. Returns the pair (loss, backprop).
    if dim == -1:
      dim = len(features.shape) - 1
    print("dim ", dim)
    one_only_on_dim = list(features.shape)
    one_only_on_dim[dim] = 1
    e = np.exp(
        features - np.reshape(np.amax(features, axis=dim), one_only_on_dim)
    )
    probs = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    bp = probs - labels
    tmp = labels * np.log(probs + 1.0e-20)
    print("before reduction ", tmp)
    l = -np.sum(tmp, axis=dim)
    return l, bp

  # TODO(b/123860949): The values are constant folded for XLA, so placeholders
  # are needed.
  def _testXent(
      self, np_features, np_labels, use_gpu=True, with_placeholders=False
  ):
    # [FUNCTION DOC]
    # HELPER: runs the fused softmax cross-entropy op and compares the backprop with the numpy one,
    # according to dtype. Inputs: features, labels, GPU flag (default on) and a flag to feed through
    # placeholders in a session (needed for XLA, which constant folds otherwise).
    _, np_backprop = self._npXent(np_features, np_labels)
    with self.cached_session(use_gpu=use_gpu) as sess:
      if with_placeholders:
        features_placeholder = array_ops.placeholder(np_features.dtype)
        labels_placeholder = array_ops.placeholder(np_labels.dtype)
        loss, backprop_ = gen_nn_ops.softmax_cross_entropy_with_logits(
            labels=labels_placeholder, features=features_placeholder
        )
        _, tf_backprop = sess.run(
            [loss, backprop_],
            feed_dict={
                labels_placeholder: np_labels,
                features_placeholder: np_features,
            },
        )
      else:
        loss, backprop_ = gen_nn_ops.softmax_cross_entropy_with_logits(
            np_features, np_labels
        )
        _, tf_backprop = self.evaluate([loss, backprop_])
    self.assertAllCloseAccordingToType(np_backprop, tf_backprop)

  def _testXentWrapper(self, np_features, np_labels, dim=-1, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares the loss only (passing a dimension argument to the op). Never called in this
    # file, so it is currently dead code.
    np_loss, _ = self._npXent(np_features, np_labels, dim=dim)
    with self.cached_session(use_gpu=use_gpu) as _:
      loss = gen_nn_ops.softmax_cross_entropy_with_logits(
          labels=np_labels, logits=np_features, dim=dim
      )
      tf_loss = self.evaluate(loss)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)

  def _testAll(self, features, labels, with_placeholders=False):
    # [FUNCTION DOC]
    # HELPER: runs _testXent on the GPU with the placeholder flag passed through.
    self._testXent(
        features, labels, use_gpu=True, with_placeholders=with_placeholders
    )

  def testFloat(self):
    # [FUNCTION DOC]
    # Two rows of float32 features and labels, one with a one hot label and one with soft labels.
    self._testAll(
        np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).astype(
            np.float32
        ),
        np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 0.5, 0.0]]).astype(
            np.float32
        ),
    )

  def testHalf(self):
    # [FUNCTION DOC]
    # Same data as float16.
    self._testAll(
        np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).astype(
            np.float16
        ),
        np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.5, 0.5, 0.0]]).astype(
            np.float16
        ),
    )


class AddNTest(test.TestCase):
  # [CLASS DOC]
  # Tests the add_n op for counts one to thirty and several dtypes. The class comment explains that
  # add_n has special cases for small counts and then loops over groups of eight.
  # AddN special-cases adding the first M inputs to make (N - M) divisible by 8,
  # after which it adds the remaining (N - M) tensors 8 at a time in a loop.
  # Test N in [1, 10] so we check each special-case from 1 to 9 and one
  # iteration of the loop.
  _MAX_N = 10

  def _supported_types(self):
    # [FUNCTION DOC]
    # HELPER: returns the dtype list to test, shorter when a GPU is available (no small integer types).
    if test.is_gpu_available():
      return [
          dtypes.float16,
          dtypes.float32,
          dtypes.float64,
          dtypes.complex64,
          dtypes.complex128,
          dtypes.int64,
          dtypes.bfloat16,
      ]
    return [
        dtypes.int8,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.complex64,
        dtypes.complex128,
        dtypes.bfloat16,
    ]

  def _buildData(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: random normal array of the given shape and dtype; for complex types adds an imaginary part
    # that is ten times the real part.
    data = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testAddN(self):
    # [FUNCTION DOC]
    # Seeds numpy, then for each dtype and count from one to ten adds that many random two by two
    # arrays with add_n and compares to the numpy sum, with looser tolerances for half and bfloat16.
    np.random.seed(12345)
    with self.session(use_gpu=True) as _:
      for dtype in self._supported_types():
        for count in range(1, self._MAX_N + 1):
          data = [self._buildData((2, 2), dtype) for _ in range(count)]
          actual = self.evaluate(math_ops.add_n(data))
          expected = np.sum(
              np.vstack([np.expand_dims(d, 0) for d in data]), axis=0
          )
          tol = 5e-3 if dtype == dtypes.float16 else 5e-7
          if dtype == dtypes.bfloat16:
            tol = 2e-2
          self.assertAllClose(expected, actual, rtol=tol, atol=tol)

  def testBigAddN(self):
    # [FUNCTION DOC]
    # Same but for counts from ten to thirty, with larger tolerances for half-width floats.
    np.random.seed(12345)
    with self.session(use_gpu=True) as _:
      for dtype in self._supported_types():
        for count in range(10, 31):
          data = [self._buildData((2, 2), dtype) for _ in range(count)]
          actual = self.evaluate(math_ops.add_n(data))
          expected = np.sum(
              np.vstack([np.expand_dims(d, 0) for d in data]), axis=0
          )
          tol = 5e-2 if dtype in [dtypes.float16, dtypes.bfloat16] else 5e-6
          self.assertAllClose(expected, actual, rtol=tol, atol=tol)


class ResourceVariableOpsTest(
    test_util.TensorFlowTestCase, parameterized.TestCase
):
  # [CLASS DOC]
  # Tests resource variable ops (creation, read, assign, scatter variants) in eager and graph modes.

  def tearDown(self):
    # [FUNCTION DOC]
    # Forces garbage collection after each test and asserts there is no uncollectable garbage (cycles
    # with finalizers), then calls the parent tear down.
    gc.collect()
    # This will only contain uncollectable garbage, i.e. reference cycles
    # involving objects with __del__ defined.
    self.assertEmpty(gc.garbage)
    super(ResourceVariableOpsTest, self).tearDown()

  @test_util.run_gpu_only
  def testGPUInt64(self):
    # [FUNCTION DOC]
    # GPU only test: creates a 64-bit integer resource variable on the first GPU in eager mode and
    # asserts its value is one.
    with context.eager_mode(), context.device("gpu:0"):
      v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.int64)
      self.assertAllEqual(1, v.numpy())

  def testEagerNameNotIdentity(self):
    # [FUNCTION DOC]
    # Two eager variables with the same name keep independent values (one and two).
    with context.eager_mode():
      v0 = resource_variable_ops.ResourceVariable(1.0, name="a")
      v1 = resource_variable_ops.ResourceVariable(2.0, name="a")
      self.assertAllEqual(v0.numpy(), 1.0)
      self.assertAllEqual(v1.numpy(), 2.0)

  def testEagerNameNotNeeded(self):
    # [FUNCTION DOC]
    # An eager variable can be created without a name and holds its initial value.
    with context.eager_mode():
      v0 = resource_variable_ops.ResourceVariable(1.0)
      self.assertAllEqual(v0.numpy(), 1.0)

  def testReadVariableDtypeMismatchEager(self):
    # [FUNCTION DOC]
    # Creates an integer variable handle, assigns one, then reads it as float; expects an invalid
    # argument error about reading with the wrong dtype.
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo"
      )
      resource_variable_ops.assign_variable_op(handle, 1)
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "Trying to read variable with wrong dtype. Expected float got int32",
      ):
        _ = resource_variable_ops.read_variable_op(handle, dtype=dtypes.float32)

  def testEagerInitializedValue(self):
    # [FUNCTION DOC]
    # Eager variable value and its initialized value are both 1.0.
    with context.eager_mode():
      variable = resource_variable_ops.ResourceVariable(1.0, name="eager-init")
      self.assertAllEqual(variable.numpy(), 1.0)
      self.assertAllEqual(variable.initialized_value().numpy(), 1.0)

  def testInitializeVariableUsingInitializedValue(self):
    # [FUNCTION DOC]
    # Creates a variable whose initial value is another variable's initialized value; asserts it is 1.0.
    var1 = resource_variable_ops.ResourceVariable(1.0, name="var1")
    var2 = resource_variable_ops.ResourceVariable(
        var1.initialized_value(), name="var2"
    )
    self.assertAllEqual(var2.initialized_value(), 1.0)

  def testEagerBool(self):
    # [FUNCTION DOC]
    # Converting an eager boolean variable holding false to a bool gives false.
    with context.eager_mode():
      v = resource_variable_ops.ResourceVariable(False, name="bool_test")
      self.assertAllEqual(bool(v), False)

  def testEagerDeepCopy(self):
    # [FUNCTION DOC]
    # Deep copies an eager variable; asserts name, shape and device are equal, values equal, and that
    # assigning to the copy does not change the original.
    with context.eager_mode():
      init_value = np.ones((4, 4, 4))
      variable = resource_variable_ops.ResourceVariable(init_value, name="init")

      copied_variable = copy.deepcopy(variable)
      self.assertEqual(variable.name, copied_variable.name)
      self.assertEqual(variable.shape, copied_variable.shape)
      self.assertEqual(variable.device, copied_variable.device)

      # The copied variable should have the same value as the original.
      self.assertAllEqual(variable.numpy(), copied_variable.numpy())

      # Updates to the copy should not be reflected in the original.
      copied_variable.assign(4 * np.ones((4, 4, 4)))
      self.assertNotAllEqual(variable.numpy(), copied_variable.numpy())

  def testVariableShape(self):
    # [FUNCTION DOC]
    # Gets the shape op result for a two element variable handle and asserts it is two.
    v = resource_variable_ops.ResourceVariable([1.0, 1.0])
    self.assertAllEqual(
        tensor_util.constant_value(
            resource_variable_ops.variable_shape(v.handle)
        ),
        [2],
    )

  def testAssignVariableDtypeMismatchEager(self):
    # [FUNCTION DOC]
    # Assigns an integer then tries assigning a float to the same handle; expects an invalid argument
    # error about a wrong dtype.
    with context.eager_mode():
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1], name="foo"
      )
      resource_variable_ops.assign_variable_op(
          handle, constant_op.constant([1])
      )
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "Trying to assign variable with wrong "
          "dtype. Expected int32 got float",
      ):
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([1.0], dtype=dtypes.float32)
        )

  @test_util.run_in_graph_and_eager_modes
  def testCreateRead(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes: creates a scalar int variable, assigns one, reads it back and
    # asserts the value is one.
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant(1, dtype=dtypes.int32)
        )
    )
    value = self.evaluate(
        resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    )
    self.assertAllEqual(1, value)

  @test_util.run_in_graph_and_eager_modes
  def testManyAssigns(self):
    # [FUNCTION DOC]
    # Chains read, assign two and read with control dependencies; asserts the reads give one and two.
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    create = resource_variable_ops.assign_variable_op(
        handle, constant_op.constant(1, dtype=dtypes.int32)
    )
    with ops.control_dependencies([create]):
      first_read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.int32
      )
    with ops.control_dependencies([first_read]):
      write = resource_variable_ops.assign_variable_op(
          handle, constant_op.constant(2, dtype=dtypes.int32)
      )
    with ops.control_dependencies([write]):
      second_read = resource_variable_ops.read_variable_op(
          handle, dtype=dtypes.int32
      )
    f, s = self.evaluate([first_read, second_read])
    self.assertEqual(f, 1)
    self.assertEqual(s, 2)

  def testAssignAdd(self):
    # [FUNCTION DOC]
    # Creates, assigns one, adds one with the assign-add op and asserts the read value is two.
    handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant(1, dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.assign_add_variable_op(
            handle, constant_op.constant(1, dtype=dtypes.int32)
        )
    )
    read = self.evaluate(
        resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    )
    self.assertEqual(read, 2)

  @test_util.run_in_graph_and_eager_modes
  def testAssignAddMethod(self):
    # [FUNCTION DOC]
    # Uses the variable method to add one; checks the value, then the read_value option: with it the
    # result is the value; without it the result is null in eager mode or an operation in graph mode.
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign_add(1.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign_add(1.0, read_value=True)
    self.assertEqual(3.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign_add(1.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(4.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes
  def testAssignSubMethod(self):
    # [FUNCTION DOC]
    # Mirror of the previous test for subtraction starting from three.
    v = resource_variable_ops.ResourceVariable(3.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign_sub(1.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign_sub(1.0, read_value=True)
    self.assertEqual(1.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign_sub(1.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(0.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes
  def testScatterAdd(self):
    # [FUNCTION DOC]
    # Scatter-add of two into index zero of a one by one integer variable holding one; expects three.
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1]
    )
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.resource_scatter_add(
            handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)
        )
    )
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterSub(self):
    # [FUNCTION DOC]
    # Scatter-sub of two from one; expects minus one.
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1]
    )
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.resource_scatter_sub(
            handle, [0], constant_op.constant([[2]], dtype=dtypes.int32)
        )
    )
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[-1]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMul(self):
    # [FUNCTION DOC]
    # Scatter-mul of five into one; expects five.
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1]
    )
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.resource_scatter_mul(
            handle, [0], constant_op.constant([[5]], dtype=dtypes.int32)
        )
    )
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[5]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterDiv(self):
    # [FUNCTION DOC]
    # Scatter-div of six by three; expects two.
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1]
    )
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.resource_scatter_div(
            handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)
        )
    )
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[2]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMin(self):
    # [FUNCTION DOC]
    # Scatter-min of three into six, pinned to the CPU device; expects three.
    with ops.device("cpu:0"):
      handle = resource_variable_ops.var_handle_op(
          dtype=dtypes.int32, shape=[1, 1]
      )
      self.evaluate(
          resource_variable_ops.assign_variable_op(
              handle, constant_op.constant([[6]], dtype=dtypes.int32)
          )
      )
      self.evaluate(
          resource_variable_ops.resource_scatter_min(
              handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)
          )
      )
      read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
      self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMax(self):
    # [FUNCTION DOC]
    # Scatter-max of three into six; expects six.
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1]
    )
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.resource_scatter_max(
            handle, [0], constant_op.constant([[3]], dtype=dtypes.int32)
        )
    )
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[6]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterSubScalar(self):
    # [FUNCTION DOC]
    # Scatter-sub with a scalar update value; expects minus one.
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1]
    )
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.resource_scatter_sub(
            handle, [0], constant_op.constant(2, dtype=dtypes.int32)
        )
    )
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[-1]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMulScalar(self):
    # [FUNCTION DOC]
    # Scatter-mul with a scalar update value; expects five.
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1]
    )
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[1]], dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.resource_scatter_mul(
            handle, [0], constant_op.constant(5, dtype=dtypes.int32)
        )
    )
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[5]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterDivScalar(self):
    # [FUNCTION DOC]
    # Scatter-div with a scalar update value; expects two.
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1]
    )
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.resource_scatter_div(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)
        )
    )
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[2]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMinScalar(self):
    # [FUNCTION DOC]
    # Scatter-min with a scalar update value; expects three.
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1]
    )
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.resource_scatter_min(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)
        )
    )
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[3]])

  @test_util.run_in_graph_and_eager_modes
  def testScatterMaxScalar(self):
    # [FUNCTION DOC]
    # Scatter-max with a scalar update value; expects six.
    handle = resource_variable_ops.var_handle_op(
        dtype=dtypes.int32, shape=[1, 1]
    )
    self.evaluate(
        resource_variable_ops.assign_variable_op(
            handle, constant_op.constant([[6]], dtype=dtypes.int32)
        )
    )
    self.evaluate(
        resource_variable_ops.resource_scatter_max(
            handle, [0], constant_op.constant(3, dtype=dtypes.int32)
        )
    )
    read = resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
    self.assertEqual(self.evaluate(read), [[6]])


class GradientDescentOptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Tests the TF1 gradient descent optimizer with dense, sparse, resource and reference variables.
  # Class attribute: the list of dtypes tested (half and single precision).
  dtypes_ = [dtypes.float16, dtypes.float32]

  def testBasic(self):
    # [FUNCTION DOC]
    # In a fresh graph, applies one step of learning rate three to two variables; checks initial and
    # updated values and that the optimizer owns no variables.
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        optimizer = gradient_descent.GradientDescentOptimizer(3.0)
        sgd_op = optimizer.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], self.evaluate(var0)
        )
        self.assertAllCloseAccordingToType(
            [3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], self.evaluate(var1)
        )
        self.assertEqual(0, len(optimizer.variables()))

  def testBasicResourceVariable(self):
    # [FUNCTION DOC]
    # Same with resource variables, initialized through the resources helper.
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1])
        )
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], self.evaluate(var0)
        )
        self.assertAllCloseAccordingToType(
            [3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], self.evaluate(var1)
        )

  def testBasicCallableParams(self):
    # [FUNCTION DOC]
    # Same with the learning rate given as a callable.
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        lr = lambda: 3.0
        sgd_op = gradient_descent.GradientDescentOptimizer(lr).apply_gradients(
            zip([grads0, grads1], [var0, var1])
        )
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], self.evaluate(var0)
        )
        self.assertAllCloseAccordingToType(
            [3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], self.evaluate(var1)
        )

  def testMinimizeResourceVariable(self):
    # [FUNCTION DOC]
    # Minimizes a squared prediction of a matrix product plus bias for one step; checks updated values
    # against hand computed gradients.
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(var0, x) + var1
        loss = pred * pred
        sgd_op = gradient_descent.GradientDescentOptimizer(1.0).minimize(loss)
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        resources.initialize_resources([var0, var1]).run()
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
        np_grad = 2 * np_pred
        self.assertAllCloseAccordingToType(
            [[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], self.evaluate(var0)
        )
        self.assertAllCloseAccordingToType([3.0 - np_grad], self.evaluate(var1))

  def testMinimizeSparseResourceVariable(self):
    # [FUNCTION DOC]
    # Same but the matrix variable is read through an embedding lookup, giving sparse gradients.
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([3.0], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        pred += var1
        loss = pred * pred
        sgd_op = gradient_descent.GradientDescentOptimizer(1.0).minimize(loss)
        # TODO(apassos) calling initialize_resources on all resources here
        # doesn't work because the sessions and graph are reused across unit
        # tests and this would mean trying to reinitialize variables. Figure out
        # a long-term solution for this.
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        np_pred = 1.0 * 4.0 + 2.0 * 5.0 + 3.0
        np_grad = 2 * np_pred
        self.assertAllCloseAccordingToType(
            [[1.0 - np_grad * 4.0, 2.0 - np_grad * 5.0]], self.evaluate(var0)
        )
        self.assertAllCloseAccordingToType([3.0 - np_grad], self.evaluate(var1))

  def testTensorLearningRate(self):
    # [FUNCTION DOC]
    # Learning rate given as a constant tensor; checks one step.
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        lrate = constant_op.constant(3.0)
        sgd_op = gradient_descent.GradientDescentOptimizer(
            lrate
        ).apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], self.evaluate(var0)
        )
        self.assertAllCloseAccordingToType(
            [3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], self.evaluate(var1)
        )

  def testGradWrtRef(self):
    # [FUNCTION DOC]
    # Computes gradients of the sum of two variables with the optimizer and asserts each is one.
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        opt = gradient_descent.GradientDescentOptimizer(3.0)
        values = [1.0, 3.0]
        vars_ = [variables.Variable([v], dtype=dtype) for v in values]
        grads_and_vars = opt.compute_gradients(vars_[0] + vars_[1], vars_)
        self.evaluate(variables.global_variables_initializer())
        for grad, _ in grads_and_vars:
          self.assertAllCloseAccordingToType([1.0], self.evaluate(grad))

  def testWithGlobalStep(self):
    # [FUNCTION DOC]
    # Applies gradients with a global step variable; asserts the variables updated and the step is one.
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        global_step = variables.Variable(0, trainable=False)
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.1], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.01], dtype=dtype)
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step
        )
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params and global_step
        self.assertAllCloseAccordingToType(
            [1.0 - 3.0 * 0.1, 2.0 - 3.0 * 0.1], self.evaluate(var0)
        )
        self.assertAllCloseAccordingToType(
            [3.0 - 3.0 * 0.01, 4.0 - 3.0 * 0.01], self.evaluate(var1)
        )
        self.assertAllCloseAccordingToType(1, self.evaluate(global_step))

  def testSparseBasic(self):
    # [FUNCTION DOC]
    # Applies indexed-slices gradients; only the indexed rows must change.
    for dtype in self.dtypes_:
      # train.GradientDescentOptimizer is V1 only API.
      with ops.Graph().as_default(), self.cached_session():
        var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
        var1 = variables.Variable([[3.0], [4.0]], dtype=dtype)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
            constant_op.constant([0]),
            constant_op.constant([2, 1]),
        )
        grads1 = indexed_slices.IndexedSlices(
            constant_op.constant([0.01], shape=[1, 1], dtype=dtype),
            constant_op.constant([1]),
            constant_op.constant([2, 1]),
        )
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0).apply_gradients(
            zip([grads0, grads1], [var0, var1])
        )
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0], [2.0]], self.evaluate(var0))
        self.assertAllCloseAccordingToType([[3.0], [4.0]], self.evaluate(var1))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType(
            [[1.0 - 3.0 * 0.1], [2.0]], self.evaluate(var0)
        )
        self.assertAllCloseAccordingToType(
            [[3.0], [4.0 - 3.0 * 0.01]], self.evaluate(var1)
        )


class BiasAddTestBase(test.TestCase):
  # [CLASS DOC]
  # Tests the bias-add op for forward values (both data formats) and gradient error, including empty
  # inputs. Despite the name Base, it is a runnable test case.

  def _npBias(self, inputs, bias):
    # [FUNCTION DOC]
    # HELPER: numpy reference; adds a vector bias along the last axis (asserts the bias is rank one and
    # matches the last dimension). Returns the sum.
    assert len(bias.shape) == 1
    assert inputs.shape[-1] == bias.shape[0]
    return inputs + bias.reshape(
        ([1] * (len(inputs.shape) - 1)) + [bias.shape[0]]
    )

  def testNpBias(self):
    # [FUNCTION DOC]
    # Checks the numpy helper on a small two by three example.
    self.assertAllClose(
        np.array([[11, 22, 33], [41, 52, 63]]),
        self._npBias(
            np.array([[10, 20, 30], [40, 50, 60]]), np.array([1, 2, 3])
        ),
    )

  def _testBias(self, np_inputs, np_bias, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares bias-add with the numpy helper. The use_gpu flag is accepted but unused.
    np_val = self._npBias(np_inputs, np_bias)
    tf_val = nn_ops.bias_add(np_inputs, np_bias)
    self.assertAllCloseAccordingToType(np_val, tf_val)

  def _AtLeast3d(self, np_value):
    # [FUNCTION DOC]
    # HELPER: reshapes an array with fewer than three dimensions by adding leading size one dims.
    # fill the input value to at least 3-dimension
    if np_value.ndim < 3:
      return np.reshape(np_value, (1,) * (3 - np_value.ndim) + np_value.shape)
    return np_value

  def _NHWCToNCHW(self, np_value):
    # [FUNCTION DOC]
    # HELPER: numpy transpose moving the last dimension to position one (first making the array at least
    # three dimensional).
    # fill the input value to at least 3-dimension
    np_value = self._AtLeast3d(np_value)
    # move the last dimension to second
    np_dim = list(range(np_value.ndim))
    np_dim_new = list(np_dim[0:1]) + list(np_dim[-1:]) + list(np_dim[1:-1])
    return np.transpose(np_value, np_dim_new)

  def _NCHWToNHWC(self, np_value):
    # [FUNCTION DOC]
    # HELPER: inverse transpose moving dimension one to the end; asserts at least rank three.
    assert len(np_value.shape) >= 3
    np_dim = list(range(np_value.ndim))
    # move the second dimension to the last
    np_dim_new = list(np_dim[0:1]) + list(np_dim[2:]) + list(np_dim[1:2])
    return np.transpose(np_value, np_dim_new)

  def _testBiasNCHW(self, np_inputs, np_bias, use_gpu):
    # [FUNCTION DOC]
    # HELPER: runs bias-add in channels-first format on a converted input, converts back and compares
    # with the numpy reference. use_gpu is unused.
    np_val = self._npBias(np_inputs, np_bias)
    np_inputs = self._NHWCToNCHW(np_inputs)
    tf_val = nn_ops.bias_add(np_inputs, np_bias, data_format="NCHW")
    tf_val = self._NCHWToNHWC(tf_val)
    self.assertAllCloseAccordingToType(self._AtLeast3d(np_val), tf_val)

  def _testAll(self, np_inputs, np_bias):
    # [FUNCTION DOC]
    # HELPER: for float32 and float16 inputs runs the NHWC and NCHW comparisons.
    if np_inputs.dtype in [np.float32, np.float16]:
      self._testBias(np_inputs, np_bias, use_gpu=True)
      self._testBiasNCHW(np_inputs, np_bias, use_gpu=True)

  def testFloatTypes(self):
    # [FUNCTION DOC]
    # Random float32 and float16 inputs of three shapes with matching bias sizes.
    for t in [np.float32, np.float16]:
      self._testAll(
          np.random.rand(4, 3, 3).astype(t), np.random.rand(3).astype(t)
      )
      self._testAll(
          np.random.rand(7, 5, 13).astype(t), np.random.rand(13).astype(t)
      )
      self._testAll(np.random.rand(9, 9).astype(t), np.random.rand(9).astype(t))

  def _testGradient(self, np_input, bias, dtype, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: gradient test. Inputs: input array, bias, dtype, data format, GPU flag. Builds tensors,
    # computes theoretical and numeric Jacobians for input and bias and for the gradient of the
    # bias gradient (via an L2 loss), and asserts them close with a threshold depending on dtype.
    with self.cached_session(use_gpu=use_gpu):
      if data_format == "NCHW":
        np_input = self._NHWCToNCHW(np_input)
      input_tensor = constant_op.constant(
          np_input, shape=np_input.shape, dtype=dtype
      )
      bias_tensor = constant_op.constant(bias, shape=bias.shape, dtype=dtype)

      if dtype == dtypes.float16:
        delta = 4.0 / 1024
      else:
        delta = 1.0 / 1024

      output_tensor = nn_ops.bias_add(
          input_tensor, bias_tensor, data_format=data_format
      )
      tensor_jacob_t, tensor_jacob_n = gradient_checker.compute_gradient(
          input_tensor,
          np_input.shape,
          output_tensor,
          np_input.shape,
          delta=delta,
      )
      bias_jacob_t, bias_jacob_n = gradient_checker.compute_gradient(
          bias_tensor, bias.shape, output_tensor, np_input.shape, delta=delta
      )

      # Test gradient of BiasAddGrad
      bias_add_grad = gradients_impl.gradients(
          nn_ops.l2_loss(output_tensor), bias_tensor
      )[0]
      grad_jacob_t, grad_jacob_n = gradient_checker.compute_gradient(
          output_tensor, np_input.shape, bias_add_grad, bias.shape, delta=delta
      )

      threshold = 5e-3
      if dtype == dtypes.float64:
        threshold = 1e-10
      if dtype == dtypes.float16:
        threshold = 2e-2
        # threshold for fp16 < threshold for fp32 since precision is lower.

      self.assertAllClose(tensor_jacob_t, tensor_jacob_n, threshold, threshold)
      self.assertAllClose(bias_jacob_t, bias_jacob_n, threshold, threshold)
      self.assertAllClose(grad_jacob_t, grad_jacob_n, threshold, threshold)

  @test_util.run_deprecated_v1
  def testGradientTensor2D(self):
    # [FUNCTION DOC]
    # Deprecated-v1 gradient test on a two dimensional input for float32 and float16.
    for data_format, use_gpu in [("NHWC", True)]:
      for dtype in [dtypes.float32, dtypes.float16]:
        np_input = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=dtype.as_numpy_dtype
        ).reshape(3, 2)
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  @test_util.run_deprecated_v1
  def testGradientTensor3D(self):
    # [FUNCTION DOC]
    # Gradient test on a three dimensional input for three dtypes (prints the format).
    for data_format, use_gpu in [("NHWC", True)]:
      for dtype in (dtypes.float32, dtypes.float64, dtypes.float16):
        print(data_format)
        np_input = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            dtype=dtype.as_numpy_dtype,
        ).reshape((2, 3, 2))
        bias = np.array([1.3, 2.4], dtype=dtype.as_numpy_dtype)
        self._testGradient(np_input, bias, dtype, data_format, use_gpu)

  @test_util.run_deprecated_v1
  def testEmpty(self):
    # [FUNCTION DOC]
    # Empty inputs of several shapes with random biases, verified by _testAll.
    np.random.seed(7)
    for shape in (0, 0), (2, 0), (0, 2), (4, 3, 0), (4, 0, 3), (0, 4, 3):
      self._testAll(np.random.randn(*shape), np.random.randn(shape[-1]))

  @test_util.run_deprecated_v1
  def testEmptyGradient(self):
    # [FUNCTION DOC]
    # Gradient test with empty inputs in channels-last, then also channels-first formats.
    for data_format, use_gpu in ("NHWC", False), ("NHWC", True):
      for shape in (0, 0), (2, 0), (0, 2):
        self._testGradient(
            np.random.randn(*shape),
            np.random.randn(shape[-1]),
            dtypes.float64,
            data_format,
            use_gpu,
        )

    for data_format, use_gpu in [
        ("NHWC", False),
        ("NHWC", True),
        ("NCHW", False),
        ("NCHW", True),
    ]:
      for shape in (4, 3, 0), (4, 0, 3), (0, 4, 3):
        self._testGradient(
            np.random.randn(*shape),
            np.random.randn(shape[-1]),
            dtypes.float64,
            data_format,
            use_gpu,
        )


class LeakyReluTest(test.TestCase):
  # [CLASS DOC]
  # Tests leaky relu: forward values on CPU/GPU for several dtypes and gradients (first and second
  # order) with the gradient checker.

  def _npLeakyRelu(self, np_features, alpha=0.1):
    # [FUNCTION DOC]
    # HELPER: numpy reference; maximum of the features and alpha times features (alpha default 0.1).
    return np.maximum(np_features, alpha * np_features)

  def testNpLeakyRelu(self):
    # [FUNCTION DOC]
    # Checks the numpy helper on a hand computed example.
    self.assertAllClose(
        np.array(
            [[-0.09, 0.7, -0.05, 0.3, -0.01], [0.1, -0.03, 0.5, -0.07, 0.9]]
        ),
        self._npLeakyRelu(
            np.array(
                [[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7, 0.9]]
            ),
            alpha=0.1,
        ),
    )

  def _testLeakyRelu(self, np_features, alpha):
    # [FUNCTION DOC]
    # HELPER: compares the op with the numpy reference for given features and alpha.
    np_leaky_relu = self._npLeakyRelu(np_features, alpha)
    tf_leaky_relu = nn_ops.leaky_relu(np_features, alpha)
    self.assertAllClose(np_leaky_relu, tf_leaky_relu)
    self.assertShapeEqual(np_leaky_relu, tf_leaky_relu)

  def testNumbersCPU(self):
    # [FUNCTION DOC]
    # Pins to the CPU and checks integers, half, single and double precision with alpha 0.2.
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testLeakyRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t),
            alpha=0.2,
        )

  def testNumbersGPU(self):
    # [FUNCTION DOC]
    # Skipped when no GPU is available; otherwise checks floating dtypes with alpha 0.1.
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float32, np.float64]:
      self._testLeakyRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t), alpha=0.1
      )

  def testGradGradFloat16(self):
    # [FUNCTION DOC]
    # Second order gradient check for float16 using a nested function that returns the first gradient.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: function under test; asserts the input is float16, records a tape, applies leaky relu and
        # returns the gradient of the output with respect to the input.
        assert x.dtype == dtypes.float16
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.leaky_relu(x)
        return tape.gradient(y, x)

      x = numpy_compat.np_asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float16,
          order="F",
      )
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x])
      )
    self.assertLess(err, 1e-4)

  def testGradientFloat16(self):
    # [FUNCTION DOC]
    # Gradient check for float16, error below 6e-2 (the comment questions whether that is too high).
    with self.cached_session():
      x = numpy_compat.np_asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float16,
          order="F",
      )
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x])
      )
      print(err)
    self.assertLess(err, 6e-2)  # check if this is too high.

  def testGradientFloat32(self):
    # [FUNCTION DOC]
    # Gradient check for float32, error below 1e-4.
    with self.cached_session():
      x = numpy_compat.np_asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F",
      )
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x])
      )
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    # [FUNCTION DOC]
    # Gradient check for float64, error below 1e-10.
    with self.cached_session():
      x = numpy_compat.np_asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float64,
          order="F",
      )
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.leaky_relu, [x])
      )
    self.assertLess(err, 1e-10)

  def testGradGradFloat32(self):
    # [FUNCTION DOC]
    # Second order gradient check for float32.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: function under test; asserts float32 input, returns the gradient of leaky relu.
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.leaky_relu(x)
        return tape.gradient(y, x)

      x = numpy_compat.np_asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F",
      )
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x])
      )
    self.assertLess(err, 1e-4)


class ReluTest(test.TestCase):
  # [CLASS DOC]
  # Tests relu forward values and second order gradients.

  def _npRelu(self, np_features):
    # [FUNCTION DOC]
    # HELPER: numpy reference, maximum of features and zero.
    return np.maximum(np_features, np.zeros(np_features.shape))

  def testNpRelu(self):
    # [FUNCTION DOC]
    # Checks the numpy helper on a hand computed example.
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 0.0], [0.1, 0.0, 0.5, 0.0, 0.9]]),
        self._npRelu(
            np.array(
                [[-0.9, 0.7, -0.5, 0.3, -0.1], [0.1, -0.3, 0.5, -0.7, 0.9]]
            )
        ),
    )

  def _testRelu(self, np_features):
    # [FUNCTION DOC]
    # HELPER: compares the relu op with the numpy reference and the shape.
    np_relu = self._npRelu(np_features)
    tf_relu = nn_ops.relu(np_features)
    self.assertAllClose(np_relu, tf_relu)
    self.assertShapeEqual(np_relu, tf_relu)

  def testNumbersCPU(self):
    # [FUNCTION DOC]
    # CPU pinned check for integer and floating dtypes.
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testRelu(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t)
        )

  def testNumbersGPU(self):
    # [FUNCTION DOC]
    # Skipped without a GPU; checks half and single precision.
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float32]:
      self._testRelu(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t)
      )

  def testNoElement(self):
    # [FUNCTION DOC]
    # Relu on an empty two by zero array.
    self._testRelu(np.array([[], []], dtype=np.float32))

  def testGradGradFloat32(self):
    # [FUNCTION DOC]
    # Second order gradient check for float32.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: function under test, asserts float32 and returns the first gradient of relu.
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
          dy = tape.gradient(y, x)
          return dy

      x = numpy_compat.np_asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float32,
          order="F",
      )
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x])
      )
    self.assertLess(err, 1e-4)

  def testGradGradFloat16(self):
    # [FUNCTION DOC]
    # Second order gradient check for float16.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: function under test, asserts float16 and returns the first gradient of relu.
        assert x.dtype == dtypes.float16
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.relu(x)
          dy = tape.gradient(y, x)
          return dy

      x = numpy_compat.np_asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [0.1, 0.3, 0.5, 0.7, 0.9]],
          dtype=np.float16,
          order="F",
      )
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x])
      )
    self.assertLess(err, 1e-4)


class Relu6Test(test.TestCase):
  # [CLASS DOC]
  # Tests relu6 forward values and gradients.

  def _npRelu6(self, np_features):
    # [FUNCTION DOC]
    # HELPER: numpy reference, features clipped between zero and six.
    sixes = np.copy(np_features)
    sixes.fill(6.0)
    return np.minimum(
        np.maximum(np_features, np.zeros(np_features.shape)), sixes
    )

  def testNpRelu6(self):
    # [FUNCTION DOC]
    # Checks the numpy helper on a hand computed example.
    self.assertAllClose(
        np.array([[0.0, 0.7, 0.0, 0.3, 6.0], [0.1, 0.0, 6.0, 0.0, 0.9]]),
        self._npRelu6(
            np.array([[-0.9, 0.7, -0.5, 0.3, 6.0], [0.1, -0.3, 6.5, -0.7, 0.9]])
        ),
    )

  def _testRelu6(self, np_features):
    # [FUNCTION DOC]
    # HELPER: compares the relu6 op with the numpy reference and shape.
    np_relu6 = self._npRelu6(np_features)
    tf_relu6 = nn_ops.relu6(np_features)
    self.assertAllClose(np_relu6, tf_relu6)
    self.assertShapeEqual(np_relu6, tf_relu6)

  def testNumbersCPU(self):
    # [FUNCTION DOC]
    # CPU pinned check for integer and floating dtypes.
    for t in [np.int32, np.int64, np.float16, np.float32, np.float64]:
      # Force execution on CPU even if a GPU kernel is available for the type.
      with ops.device("/device:CPU:0"):
        self._testRelu6(
            np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t)
        )

  def testNumbersGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the removed numpy float alias in the dtype list; fails on new numpy.
    # Skipped without a GPU; checks half, float and double, printing each dtype.
    if not test.is_gpu_available():
      self.skipTest("No GPU available")
    for t in [np.float16, np.float, np.double]:
      print(t)
      self._testRelu6(
          np.array([[-9, 7, -5, 3, -1], [1, -3, 5, -7, 9]]).astype(t)
      )

  # The gradient test for ReLU6 is a bit tricky as the derivative is
  # not well defined at around zero and six and we want to avoid that
  # in terms of input values.
  def testGradientFloat32(self):
    # [FUNCTION DOC]
    # Gradient check away from the non differentiable points zero and six (see comment above it).
    with self.cached_session():
      x = numpy_compat.np_asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float32,
          order="F",
      )
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x])
      )
    self.assertLess(err, 1e-4)

  def testGradientFloat16(self):
    # [FUNCTION DOC]
    # Same for float16.
    with self.cached_session():
      x = numpy_compat.np_asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float16,
          order="F",
      )
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x])
      )
    self.assertLess(err, 1e-4)

  def testGradientFloat64(self):
    # [FUNCTION DOC]
    # Same for float64, error below 1e-10.
    with self.cached_session():
      x = numpy_compat.np_asarray(
          [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
          dtype=np.float64,
          order="F",
      )
      err = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x])
      )
    self.assertLess(err, 1e-10)


class SoftmaxTest(test.TestCase):
  # [CLASS DOC]
  # Tests softmax and log-softmax against numpy, including overflow behavior.

  def _npSoftmax(self, features, dim=-1, log=False):
    # [FUNCTION DOC]
    # HELPER: numpy reference. Inputs: features, dimension (default last) and a flag for log softmax.
    # Subtracts the maximum for stability, computes in float32 for half inputs and casts back.
    if dim == -1:
      dim = len(features.shape) - 1
    one_only_on_dim = list(features.shape)
    one_only_on_dim[dim] = 1
    is_fp16 = features.dtype == np.float16
    if is_fp16:
      # Do the compute in fp32 and cast the input back to fp32.
      features = features.astype(np.float32)
    e = np.exp(
        features - np.reshape(np.amax(features, axis=dim), one_only_on_dim)
    )
    softmax = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    if log:
      res = np.log(softmax)
    else:
      res = softmax
    if is_fp16:
      res = res.astype(np.float16)
    return res

  def _testSoftmax(self, np_features, dim=-1, log=False, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares softmax or log-softmax with the reference using an arbitrary op name (to catch a
    # past bug that inferred the op kind from the name), checks shape and, for softmax, that the
    # results sum to one along the dimension.
    # A previous version of the code checked the op name rather than the op type
    # to distinguish between log and non-log.  Use an arbitrary name to catch
    # this bug in future.
    name = "arbitrary"
    np_softmax = self._npSoftmax(np_features, dim=dim, log=log)
    with self.cached_session(use_gpu=use_gpu):
      if log:
        tf_softmax = nn_ops.log_softmax(np_features, axis=dim, name=name)
      else:
        tf_softmax = nn_ops.softmax(np_features, axis=dim, name=name)
      out = self.evaluate(tf_softmax)
    self.assertAllCloseAccordingToType(np_softmax, out)
    self.assertShapeEqual(np_softmax, tf_softmax)
    if not log:
      # Bonus check: the softmaxes should add to one in dimension dim.
      sum_along_dim = np.sum(out, axis=dim)
      self.assertAllCloseAccordingToType(
          np.ones(sum_along_dim.shape), sum_along_dim
      )

  def _testAll(self, features):
    # [FUNCTION DOC]
    # HELPER: runs softmax, log softmax and the overflow check, on the GPU preference.
    self._testSoftmax(features, use_gpu=True)
    self._testSoftmax(features, log=True, use_gpu=True)
    self._testOverflow(use_gpu=True)

  def testNpSoftmax(self):
    # [FUNCTION DOC]
    # Verifies the numpy helper against hand computed values for softmax and log softmax.
    features = [[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]
    # Batch 0: All exps are 1.  The expected result is
    # Softmaxes = [0.25, 0.25, 0.25, 0.25]
    # LogSoftmaxes = [-1.386294, -1.386294, -1.386294, -1.386294]
    #
    # Batch 1:
    # exps = [1., 2.718, 7.389, 20.085]
    # sum = 31.192
    # Softmaxes = exps / sum = [0.0320586, 0.08714432, 0.23688282, 0.64391426]
    # LogSoftmaxes = [-3.44019 , -2.44019 , -1.44019 , -0.44019]
    np_sm = self._npSoftmax(np.array(features))
    self.assertAllClose(
        np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.0320586, 0.08714432, 0.23688282, 0.64391426],
        ]),
        np_sm,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    np_lsm = self._npSoftmax(np.array(features), log=True)
    self.assertAllClose(
        np.array([
            [-1.386294, -1.386294, -1.386294, -1.386294],
            [-3.4401897, -2.4401897, -1.4401897, -0.4401897],
        ]),
        np_lsm,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

  def _testOverflow(self, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: log-softmax with a feature equal to the largest finite value of the dtype (float32 with
    # GPU preference, otherwise float64); expects the large entry to give zero and others minus max.
    if use_gpu:
      type = np.float32  # pylint: disable=redefined-builtin
    else:
      type = np.float64  # pylint: disable=redefined-builtin
    max = np.finfo(type).max  # pylint: disable=redefined-builtin
    features = np.array([[1.0, 1.0, 1.0, 1.0], [max, 1.0, 2.0, 3.0]]).astype(
        type
    )
    with self.cached_session(use_gpu=use_gpu):
      tf_log_softmax = nn_ops.log_softmax(features)
      out = self.evaluate(tf_log_softmax)
    self.assertAllClose(
        np.array([
            [-1.386294, -1.386294, -1.386294, -1.386294],
            [0, -max, -max, -max],
        ]),
        out,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

  def testFloat(self):
    # [FUNCTION DOC]
    # Runs the complete check for float32.
    self._testAll(
        np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).astype(
            np.float32
        )
    )

  def testHalf(self):
    # [FUNCTION DOC]
    # Runs the complete check for float16.
    self._testAll(
        np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]]).astype(
            np.float16
        )
    )


class BaseReductionTest(test.TestCase):
  # [CLASS DOC]
  # Abstract base for reduction op tests; subclasses override the TensorFlow and numpy reducers.
  # Has no test methods of its own.

  def _tf_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # Abstract hook; always raises a not-implemented error. Subclasses return the TensorFlow reduction
    # of the input over the given axes with keepdims.
    raise NotImplementedError()

  def _np_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # Abstract hook; always raises a not-implemented error. Subclasses return the numpy reduction.
    raise NotImplementedError()

  def _makeIncremental(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: array of incrementing values of the given shape and dtype; complex dtypes get a negative
    # imaginary part (minus twice the value).
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype.as_numpy_dtype)
    if dtype.is_complex:
      data -= 2j * data
    return data

  def _makeRandom(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: uniform random array of given shape and dtype, with the same complex adjustment.
    data = np.random.rand(*shape).astype(dtype.as_numpy_dtype)
    if dtype.is_complex:
      data -= 2j * data
    return data

  def _compareGradient(self, x, reduction_axes, rtol=1e-8, atol=1e-8):
    # [FUNCTION DOC]
    # HELPER: compares theoretical and numeric Jacobians for the reduction of x over given axes with
    # tolerances. When the axes is a single element list, it also recursively tests a scalar axis.
    if reduction_axes is not None and np.shape(reduction_axes) == (1,):
      # Test scalar reduction_axes argument
      self._compareGradient(x, reduction_axes[0], rtol=rtol, atol=atol)
    with self.cached_session(use_gpu=True):
      t = ops.convert_to_tensor(x)
      su = self._tf_reduce(t, reduction_axes, False)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, x.shape, su, su.get_shape().as_list(), x_init_value=x, delta=1
      )
    self.assertAllClose(jacob_t, jacob_n, rtol=rtol, atol=atol)

  def _compareGradientAxes(self, x, rtol=1e-8, atol=1e-8):
    # [FUNCTION DOC]
    # HELPER: runs the gradient comparison for no axes, empty axes, and several single and combined axes.
    self._compareGradient(x, None, rtol=rtol, atol=atol)
    self._compareGradient(x, [], rtol=rtol, atol=atol)
    self._compareGradient(x, 0, rtol=rtol, atol=atol)
    self._compareGradient(x, [1], rtol=rtol, atol=atol)
    self._compareGradient(x, [2], rtol=rtol, atol=atol)
    self._compareGradient(x, [1, 2], rtol=rtol, atol=atol)
    self._compareGradient(x, [0, 1, 2, 3], rtol=rtol, atol=atol)


class ConcatOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests the concat op with random shapes and its gradient.

  def _testRandom(self, dtype):
    # [FUNCTION DOC]
    # HELPER: random rank five shapes, several placeholders concatenated along a random dimension.
    # Input: dtype (bfloat16 values are fed as float32 and cast around the op). Asserts the result shape
    # and that each slice of the output equals the fed input (exactly or within 0.01 for casts).
    # Random dims of rank 5
    shape = np.random.randint(1, 5, size=5)
    # Random number of tensors, but always > 1.
    num_tensors = np.random.randint(2, 10)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    params = {}
    if dtype == dtypes.bfloat16:
      dtype_feed = dtypes.float32
    else:
      dtype_feed = dtype
    with self.cached_session(use_gpu=True):
      p = []
      for i in np.arange(num_tensors):
        input_shape = shape
        input_shape[concat_dim] = np.random.randint(1, 5)
        placeholder = array_ops.placeholder(dtype_feed, shape=input_shape)
        p.append(placeholder)

        t = dtype_feed.as_numpy_dtype
        params[placeholder] = np.random.rand(*input_shape).astype(t)

      if dtype != dtype_feed:
        concat_inputs = [math_ops.cast(p_i, dtype) for p_i in p]
      else:
        concat_inputs = p
      c = array_ops.concat(concat_inputs, concat_dim)
      if dtype != dtype_feed:
        c = math_ops.cast(c, dtype_feed)
      result = c.eval(feed_dict=params)

    self.assertEqual(result.shape, c.get_shape())
    cur_offset = 0

    for i in np.arange(num_tensors):
      # The index into the result is the ':' along all dimensions
      # except the concat_dim. slice(0, size) is used for ':', and
      # a list of slices is used to index into result.
      ind = [slice(0, params[p[i]].shape[j]) for j in np.arange(5)]
      ind[concat_dim] = slice(
          cur_offset, cur_offset + params[p[i]].shape[concat_dim]
      )
      cur_offset += params[p[i]].shape[concat_dim]
      if dtype == dtype_feed:
        self.assertAllEqual(result[tuple(ind)], params[p[i]])
      else:
        self.assertAllClose(result[tuple(ind)], params[p[i]], 0.01)

  @test_util.run_deprecated_v1
  def testRandom(self):
    # [FUNCTION DOC]
    # Deprecated-v1 test calling the helper for bfloat16, float16, float32, int32 and int64.
    self._testRandom(dtypes.bfloat16.as_numpy_dtype)
    self._testRandom(dtypes.float16)
    self._testRandom(dtypes.float32)
    self._testRandom(dtypes.int32)
    self._testRandom(dtypes.int64)

  def _RunAndVerifyGradientsRandom(self, dtype=dtypes.float32.as_numpy_dtype):
    # [FUNCTION DOC]
    # HELPER: concatenates 12 to 19 random tensors, backpropagates a random gradient and asserts the
    # concatenated gradients equal that gradient. Input: numpy dtype (default float32).
    # Random dims of rank 5
    input_shape = np.random.randint(1, 5, size=5)
    # Random number of tensors
    num_tensors = np.random.randint(12, 20)
    # Random dim to concat on
    concat_dim = np.random.randint(5)
    concat_dim_sizes = np.random.randint(1, 5, size=num_tensors)
    with test_util.use_gpu():
      inp = []
      inp_tensors = []
      for x in concat_dim_sizes:
        shape = input_shape
        shape[concat_dim] = x
        t = np.random.rand(*shape).astype(dtype)
        inp.append(t)
        inp_tensors.append(
            constant_op.constant(t.flatten(), shape=shape, dtype=dtype)
        )
      c = array_ops.concat(inp_tensors, concat_dim)
      output_shape = input_shape
      output_shape[concat_dim] = concat_dim_sizes.sum()
      grad_inp = np.random.rand(*output_shape).astype(dtype)
      grad_tensor = constant_op.constant(grad_inp.flatten(), shape=output_shape)
      grad = gradients_impl.gradients([c], inp_tensors, [grad_tensor])
      concated_grad = array_ops.concat(grad, concat_dim)
      result = self.evaluate(concated_grad)

    self.assertAllEqual(result, grad_inp)

  @test_util.run_deprecated_v1
  def testGradientsRandom(self):
    # [FUNCTION DOC]
    # Deprecated-v1 test; five rounds with float32 and bfloat16.
    for _ in range(5):
      self._RunAndVerifyGradientsRandom()
      self._RunAndVerifyGradientsRandom(dtypes.bfloat16.as_numpy_dtype)


class TileTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the tile op.

  def testSimple(self):
    # [FUNCTION DOC]
    # Tiles a random four by one array by multiples one and four (32-bit or 64-bit multiples) for float32
    # and bfloat16, asserting shape, static shape and values.
    # multiples could be int32 or int64
    for dtype in [dtypes.int32, dtypes.int64]:
      for in_type in [np.float32, dtypes.bfloat16.as_numpy_dtype]:
        with self.cached_session(use_gpu=True):
          inp = np.random.rand(4, 1).astype(in_type)
          a = constant_op.constant(inp)
          tiled = array_ops.tile(a, constant_op.constant([1, 4], dtype=dtype))
          result = self.evaluate(tiled)
        self.assertEqual(result.shape, (4, 4))
        self.assertEqual([4, 4], tiled.get_shape())
        self.assertTrue((result == np.tile(inp, (1, 4))).all())


class PadOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests the pad op in constant, reflect and symmetric modes, with gradient checks.

  def _npPad(self, inp, paddings, mode, constant_values=0):
    # [FUNCTION DOC]
    # HELPER: numpy pad reference with a lower-cased mode and constant values for constant mode only.
    mode = mode.lower()
    if mode == "constant":
      return np.pad(inp, paddings, mode=mode, constant_values=constant_values)
    else:
      return np.pad(inp, paddings, mode=mode)

  def _testPad(self, np_inputs, paddings, mode, constant_values):
    # [FUNCTION DOC]
    # HELPER: compares pad op output with the numpy reference and shape.
    np_val = self._npPad(
        np_inputs, paddings, mode=mode, constant_values=constant_values
    )
    with self.cached_session(use_gpu=True):
      tf_val = array_ops.pad(
          np_inputs, paddings, mode=mode, constant_values=constant_values
      )
      out = self.evaluate(tf_val)
    self.assertAllEqual(np_val, out)
    self.assertShapeEqual(np_val, tf_val)

  def _testPadGradient(self, x, a, mode, constant_values):
    # [FUNCTION DOC]
    # HELPER: gradient check of the pad op with given paddings and mode.
    with self.cached_session(use_gpu=True):
      inx = ops.convert_to_tensor(x)
      xs = list(x.shape)
      ina = ops.convert_to_tensor(a)
      y = array_ops.pad(inx, ina, mode=mode, constant_values=constant_values)
      # Expected y's shape to be:
      ys = list(np.array(x.shape) + np.sum(np.array(a), axis=1))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, y, ys, x_init_value=x
      )
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _testPaddingAll(self, np_inputs, paddings, constant_values):
    # [FUNCTION DOC]
    # HELPER: loops over six mode spellings; skips reflect for empty inputs; checks values, and the
    # gradient for float32 inputs.
    for mode in (
        "CONSTANT",
        "REFLECT",
        "SYMMETRIC",
        "reflect",
        "symmetric",
        "constant",
    ):
      # Zero-sized input is not allowed for REFLECT mode, but we still want
      # zero-sized input test cases for the other modes.
      if np_inputs.size or mode.upper() != "REFLECT":
        self._testPad(
            np_inputs, paddings, mode=mode, constant_values=constant_values
        )
        if np_inputs.dtype == np.float32:
          self._testPadGradient(
              np_inputs, paddings, mode=mode, constant_values=constant_values
          )

  @test_util.run_deprecated_v1
  def testPadding(self):
    # [FUNCTION DOC]
    # Deprecated-v1 test for three input shapes (including an empty one) with different paddings.
    for t in [np.float32]:
      self._testPaddingAll(
          np.random.rand(2, 5).astype(t), [[1, 0], [2, 0]], 0.0
      )
      self._testPaddingAll(
          np.random.rand(2, 3, 4).astype(t), [[0, 0], [0, 0], [0, 0]], -1234.0
      )
      self._testPaddingAll(
          np.random.rand(0, 3, 4).astype(t), [[0, 0], [2, 1], [2, 3]], 0.0
      )


class RandomOpsCorrectnessTest(test.TestCase):
  # [CLASS DOC]
  # Checks that GPU random ops with a fixed seed match CPU results, for uniform, normal and
  # truncated normal ops, over several shapes. Class attributes hold shapes, seeds, ranges, means
  # and standard deviations.
  shapes = [[1, 5], [2, 6, 5], [5, 3, 6, 2], [100, 100]]
  seeds = [2, 16, 1582, 12]
  minvals = [-10.0, 0.5, 10.0, 1000.0]
  maxvals = [-5.0, 1.0, 20.0, 2000.0]
  means = [-5.0, 1.0, 100.0, 1000.0]
  stddevs = [0.1, 1.0, 10.0, 100.0]

  def _testRandomDefault(self, rnfunc, shape, seed, dtype):
    # [FUNCTION DOC]
    # HELPER: runs the generator on CPU and GPU with the same seed and dtype and asserts closeness
    # (looser tolerance for half precision).
    with test_util.device(use_gpu=False):
      ref = rnfunc(shape, seed=seed, dtype=dtype)
    with test_util.device(use_gpu=True):
      result = rnfunc(shape, seed=seed, dtype=dtype)
    if dtype == dtypes.float16:
      self.assertAllClose(result, ref, atol=1e-3)
    else:
      self.assertAllClose(result, ref, atol=1e-5)

  def _testRandomMinvalMaxval(
      self, rnfunc, shape, seed, minvalue, maxvalue, dtype
  ):
    # [FUNCTION DOC]
    # HELPER: same with a range.
    with test_util.device(use_gpu=False):
      ref = rnfunc(
          shape, seed=seed, minval=minvalue, maxval=maxvalue, dtype=dtype
      )
    with test_util.device(use_gpu=True):
      result = rnfunc(
          shape, seed=seed, minval=minvalue, maxval=maxvalue, dtype=dtype
      )
    if dtype == dtypes.float16:
      self.assertAllClose(result, ref, atol=1e-3)
    else:
      self.assertAllClose(result, ref, atol=1e-5)

  def _testRandomMeanStd(self, rnfunc, shape, seed, mean, stddev, dtype):
    # [FUNCTION DOC]
    # HELPER: same with mean and standard deviation, using a very loose half precision tolerance
    # (a TODO says it is too high).
    with test_util.device(use_gpu=False):
      ref = rnfunc(shape, seed=seed, mean=mean, stddev=stddev, dtype=dtype)
    with test_util.device(use_gpu=True):
      result = rnfunc(shape, seed=seed, mean=mean, stddev=stddev, dtype=dtype)
    if dtype == dtypes.float16:
      self.assertAllClose(result, ref, atol=5e-1)
      # TODO: this tolerance is too high.
      # For the random normal case & truncated normal, this particular test has
      # 1/10000 mismatched element in the (100, 100) shape.
    else:
      self.assertAllClose(result, ref, atol=1e-5)

  def testRandomUniformCorrectness_1(self):
    # [FUNCTION DOC]
    # Uniform default-range check across dtypes and shapes.
    for dtype in [dtypes.float32, dtypes.float16]:
      for i in range(len(self.shapes)):
        self._testRandomDefault(
            random_ops.random_uniform, self.shapes[i], self.seeds[i], dtype
        )

  def testRandomUniformCorrectness_2(self):
    # [FUNCTION DOC]
    # Uniform check with the min and max for each shape.
    for dtype in [dtypes.float32, dtypes.float16]:
      for i in range(len(self.shapes)):
        self._testRandomMinvalMaxval(
            random_ops.random_uniform,
            self.shapes[i],
            self.seeds[i],
            self.minvals[i],
            self.maxvals[i],
            dtype,
        )

  def testRandomNormalCorrectness_1(self):
    # [FUNCTION DOC]
    # Normal default check.
    for dtype in [dtypes.float32, dtypes.float16]:
      for i in range(len(self.shapes)):
        self._testRandomDefault(
            random_ops.random_normal, self.shapes[i], self.seeds[i], dtype
        )

  def testRandomNormalCorrectness_2(self):
    # [FUNCTION DOC]
    # Normal check with mean and standard deviation.
    for dtype in [dtypes.float32, dtypes.float16]:
      for i in range(len(self.shapes)):
        self._testRandomMeanStd(
            random_ops.random_normal,
            self.shapes[i],
            self.seeds[i],
            self.means[i],
            self.stddevs[i],
            dtype,
        )

  def testRandomTruncatedCorrectness_1(self):
    # [FUNCTION DOC]
    # Truncated normal default check.
    for dtype in [dtypes.float32, dtypes.float16]:
      for i in range(len(self.shapes)):
        self._testRandomDefault(
            random_ops.truncated_normal, self.shapes[i], self.seeds[i], dtype
        )

  def testRandomTruncatedCorrectness_2(self):
    # [FUNCTION DOC]
    # Truncated normal check with mean and standard deviation.
    for dtype in [dtypes.float32, dtypes.float16]:
      for i in range(len(self.shapes)):
        self._testRandomMeanStd(
            random_ops.truncated_normal,
            self.shapes[i],
            self.seeds[i],
            self.means[i],
            self.stddevs[i],
            dtype,
        )


class StatelessRandomOpsCorrectnessTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the class attribute named dtypes hides the module name inside the class body, which is
  #   confusing; methods still use the module through global lookup.
  # Checks stateless random raw ops give the same results on CPU and GPU for a given seed.
  shapes = [[1, 5], [2, 6, 5], [5, 3, 6, 2], [100, 100]]
  seeds = [[2, 1], [16, 12], [1582, 10230], [12, 23101]]
  dtypes = [dtypes.float32, dtypes.float32, dtypes.half, dtypes.half]

  def _testStatelessRandomDefault(self, rnfunc, shape, seed, dtype):
    # [FUNCTION DOC]
    # HELPER: runs the raw op on CPU and GPU with the same shape, seed and dtype; compares with a
    # tolerance depending on dtype (no comparison for other dtypes).
    with test_util.device(use_gpu=False):
      ref = rnfunc(shape=shape, seed=seed, dtype=dtype)
    with test_util.device(use_gpu=True):
      result = rnfunc(shape=shape, seed=seed, dtype=dtype)
    if dtype == dtypes.float32:
      self.assertAllClose(result, ref, atol=1e-5)
    elif dtype == dtypes.float16:
      self.assertAllClose(result, ref, atol=1e-3)

  def testRandomUniformCorrectness_1(self):
    # [FUNCTION DOC]
    # Stateless uniform over the shapes, seeds and dtypes.
    for i in range(len(self.shapes)):
      self._testStatelessRandomDefault(
          raw_ops.StatelessRandomUniform,
          self.shapes[i],
          self.seeds[i],
          self.dtypes[i],
      )

  def testRandomNormalCorrectness_1(self):
    # [FUNCTION DOC]
    # Stateless normal.
    for i in range(len(self.shapes)):
      self._testStatelessRandomDefault(
          raw_ops.StatelessRandomNormal,
          self.shapes[i],
          self.seeds[i],
          self.dtypes[i],
      )

  def testTruncatedNormalCorrectness_1(self):
    # [FUNCTION DOC]
    # Stateless truncated normal.
    for i in range(len(self.shapes)):
      self._testStatelessRandomDefault(
          raw_ops.StatelessTruncatedNormal,
          self.shapes[i],
          self.seeds[i],
          self.dtypes[i],
      )


class StatelessRandomOpsCorrectnessTestV2(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the class attribute named dtypes hides the module name inside the class body.
  # Same idea for the V1 and V2 stateless ops (V2 take a key, counter and algorithm).
  shapes = [[1, 5], [2, 6, 5], [5, 3, 6, 2], [100, 100]]
  seeds = [[2, 1], [16, 12], [1582, 10230], [12, 23101]]
  key = [[2], [16], [1582], [12]]
  counters = [[23, 11], [11, 23], [2000312, 0], [0, 0]]
  itypes = [dtypes.int32, dtypes.uint32, dtypes.int64, dtypes.uint64]
  dtypes = [dtypes.float32, dtypes.float32, dtypes.half, dtypes.half]

  def _testStatelessRandomDefault(self, rnfunc, shape, seed, dtype):
    # [FUNCTION DOC]
    # HELPER: same as the V1 class: compare CPU and GPU output for a seed and dtype.
    with test_util.device(use_gpu=False):
      ref = rnfunc(shape=shape, seed=seed, dtype=dtype)
    with test_util.device(use_gpu=True):
      result = rnfunc(shape=shape, seed=seed, dtype=dtype)
    if dtype == dtypes.float32:
      self.assertAllClose(result, ref, atol=1e-5)
    elif dtype == dtypes.float16:
      self.assertAllClose(result, ref, atol=1e-3)

  def _testStatelessRandomDefaultV2(
      self, rnfunc, shape, key, counter, dtype, alg=1
  ):
    # [FUNCTION DOC]
    # HELPER: runs a V2 op with shape, key (only the first key element is used), algorithm one and counter
    # on CPU and GPU and asserts closeness. The dtype argument is unused.
    with test_util.device(use_gpu=False):
      ref = rnfunc(shape=shape, key=[key[0]], alg=alg, counter=counter)
    with test_util.device(use_gpu=True):
      result = rnfunc(shape=shape, key=[key[0]], alg=alg, counter=counter)
    self.assertAllClose(result, ref, atol=1e-5)

  def _testStatelessRandomUniformFullIntV2(
      self, rnfunc, shape, key, counter, dtype
  ):
    # [FUNCTION DOC]
    # HELPER: runs the full-integer uniform op on CPU and GPU and asserts only that the shapes are equal.
    with test_util.device(use_gpu=False):
      ref = rnfunc(shape=shape, alg=1, key=key, counter=counter, dtype=dtype)
    with test_util.device(use_gpu=True):
      result = rnfunc(shape=shape, alg=1, key=key, counter=counter, dtype=dtype)
    self.assertEqual(result.shape, ref.shape)

  def testRandomUniformCorrectness_1(self):
    # [FUNCTION DOC]
    # V1 stateless uniform.
    for i in range(len(self.shapes)):
      self._testStatelessRandomDefault(
          raw_ops.StatelessRandomUniform,
          self.shapes[i],
          self.seeds[i],
          self.dtypes[i],
      )

  def testRandomUniformV2Correctness_1(self):
    # [FUNCTION DOC]
    # V2 stateless uniform; the seeds list is passed as the key and counters as counters.
    for i in range(len(self.shapes)):
      self._testStatelessRandomDefaultV2(
          raw_ops.StatelessRandomUniformV2,
          self.shapes[i],
          self.seeds[i],
          self.counters[i],
          self.dtypes[i],
      )

  def testRandomNormalCorrectness_1(self):
    # [FUNCTION DOC]
    # V1 stateless normal.
    for i in range(len(self.shapes)):
      self._testStatelessRandomDefault(
          raw_ops.StatelessRandomNormal,
          self.shapes[i],
          self.seeds[i],
          self.dtypes[i],
      )

  def testRandomNormalV2Correctness_1(self):
    # [FUNCTION DOC]
    # V2 stateless normal.
    for i in range(len(self.shapes)):
      self._testStatelessRandomDefaultV2(
          raw_ops.StatelessRandomNormalV2,
          self.shapes[i],
          self.seeds[i],
          self.counters[i],
          self.dtypes[i],
      )

  def testTruncatedNormalCorrectness_1(self):
    # [FUNCTION DOC]
    # V1 stateless truncated normal.
    for i in range(len(self.shapes)):
      self._testStatelessRandomDefault(
          raw_ops.StatelessTruncatedNormal,
          self.shapes[i],
          self.seeds[i],
          self.dtypes[i],
      )

  def testTruncatedNormalV2Correctness_1(self):
    # [FUNCTION DOC]
    # V2 stateless truncated normal.
    for i in range(len(self.shapes)):
      self._testStatelessRandomDefaultV2(
          raw_ops.StatelessTruncatedNormalV2,
          self.shapes[i],
          self.seeds[i],
          self.counters[i],
          self.dtypes[i],
      )

  def testRandomUniformFullIntV2Functional_1(self):
    # [FUNCTION DOC]
    # Full-integer uniform for 32/64-bit signed and unsigned integer types, checking shapes only.
    for i in range(len(self.shapes)):
      self._testStatelessRandomUniformFullIntV2(
          raw_ops.StatelessRandomUniformFullIntV2,
          self.shapes[i],
          self.key[i],
          self.counters[i],
          self.itypes[i],
      )


class BatchNormTest(test.TestCase):
  # [CLASS DOC]
  # Tests fused batch normalization (training and inference, both data formats, half and single
  # precision, several shapes) against a manual reference, and its gradients.

  def _batch_norm(self, x, mean, var, offset, scale, epsilon):
    # [FUNCTION DOC]
    # HELPER: manual batch norm reference (the library function lacks float16 support, see TODO).
    # Computes inverse std times scale, applies to the input and returns the result cast back.
    # We compute the batch norm manually in this function because
    # nn_impl.batch_normalization does not support float16 yet.
    # TODO(reedwm): Add float16 support to nn_impl.batch_normalization.
    inv = math_ops.rsqrt(var + epsilon) * scale
    y = math_ops.cast(x, scale.dtype) * inv + (offset - mean * inv)
    return math_ops.cast(y, x.dtype)

  def _running_mean(self, old_mean, new_val, factor):
    # [FUNCTION DOC]
    # HELPER: exponential moving average update; returns the new value if the factor is one, else a blend.
    if factor == 1.0:
      return new_val
    else:
      return (1.0 - factor) * old_mean + factor * new_val

  def _training_ref(
      self,
      x,
      scale,
      offset,
      old_mean,
      old_var,
      exponential_avg_factor,
      epsilon,
      data_format,
  ):
    # [FUNCTION DOC]
    # HELPER: reference for training mode. Computes the batch mean and variance (moments), the output,
    # Bessel corrected variance and updated running statistics. Raises a value error for a data format
    # other than the two supported ones. Returns evaluated output, mean and variance.
    if data_format not in ["NHWC", "NCHW"]:
      raise ValueError(
          "data_format must be NCHW or NHWC, got %s." % data_format
      )
    if data_format == "NCHW":
      x = array_ops.transpose(x, [0, 2, 3, 1])
    batch_mean, batch_var = nn_impl.moments(
        math_ops.cast(x, scale.dtype), [0, 1, 2], keep_dims=False
    )

    y = self._batch_norm(x, batch_mean, batch_var, offset, scale, epsilon)
    if data_format == "NCHW":
      y = array_ops.transpose(y, [0, 3, 1, 2])

    # This is for Bessel's correction. tf.nn.moments uses n, instead of n-1, as
    # the denominator in the formula to calculate variance, while
    # tf.compat.v1.nn.fused_batch_norm has Bessel's correction built in.
    sample_size = math_ops.cast(
        array_ops.size(x) / array_ops.size(scale), scale.dtype
    )
    batch_var_corrected = (
        batch_var * sample_size / (math_ops.maximum(sample_size - 1.0, 1.0))
    )

    mean = self._running_mean(old_mean, batch_mean, exponential_avg_factor)
    var = self._running_mean(
        old_var, batch_var_corrected, exponential_avg_factor
    )
    return self.evaluate(y), self.evaluate(mean), self.evaluate(var)

  def _test_training(
      self,
      x_shape,
      x_dtype,
      scale_shape,
      scale_dtype,
      use_gpu=True,
      exponential_avg_factor=1.0,
      data_format="NHWC",
  ):
    # [FUNCTION DOC]
    # HELPER: random inputs; calls the fused batch norm in training mode and asserts output, mean and
    # variance close to the reference, with a looser output tolerance for half precision.
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    if exponential_avg_factor == 1.0:
      old_mean_val = None
      old_var_val = None
    else:
      old_mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
      old_var_val = np.random.random_sample(scale_shape).astype(scale_dtype)

    with self.cached_session(use_gpu=use_gpu) as _:
      x = constant_op.constant(x_val, name="x")
      scale = constant_op.constant(scale_val, name="scale")
      offset = constant_op.constant(offset_val, name="offset")
      epsilon = 0.001
      y, mean, var = nn_impl.fused_batch_norm(
          x,
          scale,
          offset,
          mean=old_mean_val,
          variance=old_var_val,
          epsilon=epsilon,
          exponential_avg_factor=exponential_avg_factor,
          data_format=data_format,
          is_training=True,
      )
      y_val, mean_val, var_val = self.evaluate([y, mean, var])
      y_ref, mean_ref, var_ref = self._training_ref(
          x,
          scale,
          offset,
          old_mean_val,
          old_var_val,
          exponential_avg_factor,
          epsilon,
          data_format,
      )
    y_atol = 2e-3 if x_dtype == np.float16 else 1e-3
    self.assertAllClose(y_ref, y_val, atol=y_atol)
    self.assertAllClose(mean_ref, mean_val, atol=1e-3)
    self.assertAllClose(var_ref, var_val, atol=1e-3)

  def _inference_ref(self, x, scale, offset, mean, var, epsilon, data_format):
    # [FUNCTION DOC]
    # HELPER: reference for inference mode (value error for bad data format). Returns the evaluated output.
    if data_format not in ["NHWC", "NCHW"]:
      raise ValueError(
          "data_format must be NCHW or NHWC, got %s." % data_format
      )
    if data_format == "NCHW":
      x = array_ops.transpose(x, [0, 2, 3, 1])
    y = self._batch_norm(x, mean, var, offset, scale, epsilon)
    if data_format == "NCHW":
      y = array_ops.transpose(y, [0, 3, 1, 2])
    return self.evaluate(y)

  def _test_inference(
      self,
      x_shape,
      x_dtype,
      scale_shape,
      scale_dtype,
      use_gpu=True,
      exponential_avg_factor=1.0,
      data_format="NHWC",
  ):
    # [FUNCTION DOC]
    # HELPER: random inputs; calls the fused batch norm in inference mode with given mean and variance
    # and compares with the reference.
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    mean_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    var_val = np.random.random_sample(scale_shape).astype(scale_dtype)

    with self.cached_session(use_gpu=use_gpu) as _:
      x = constant_op.constant(x_val, name="x")
      scale = constant_op.constant(scale_val, name="scale")
      offset = constant_op.constant(offset_val, name="offset")
      mean = constant_op.constant(mean_val, name="mean")
      var = constant_op.constant(var_val, name="variance")
      epsilon = 0.001
      y, _, _ = nn_impl.fused_batch_norm(
          x,
          scale,
          offset,
          mean=mean,
          variance=var,
          epsilon=epsilon,
          exponential_avg_factor=exponential_avg_factor,
          data_format=data_format,
          is_training=False,
      )
      y_val = self.evaluate(y)
      y_ref = self._inference_ref(
          x, scale, offset, mean, var, epsilon, data_format
      )
    # An atol value of 1e-3 is too small for float16's, because some adjacent
    # float16 values that y_val can take are greater than 1e-3 apart, e.g.
    # 2.16602 and 2.16797.
    atol = 2e-3 if x_dtype == np.float16 else 1e-3
    self.assertAllClose(y_ref, y_val, atol=atol)

  def _runtests(self, x_shape, is_training, gradient_test=False):
    # [FUNCTION DOC]
    # HELPER: drives a shape through all combinations of dtype (half, single), GPU flags (CPU, plus GPU
    # when CUDA is available), data format and averaging factors (one, plus 0.6 under a forward
    # compatibility check), calling the gradient, training or inference checks.
    use_gpu_vals = [False]
    if test.is_gpu_available(cuda_only=True):
      use_gpu_vals += [True]
    factors = [
        1.0,
    ]
    if compat.forward_compatible(2020, 3, 6):
      factors += [
          0.6,
      ]
    for dtype in [np.float16, np.float32]:
      for use_gpu in use_gpu_vals:
        for data_format in ["NHWC", "NCHW"]:
          if data_format == "NHWC":
            scale_shape = x_shape[-1:]
          else:
            scale_shape = x_shape[1:2]
          for exponential_avg_factor in factors:
            if gradient_test:
              self._test_gradient(
                  x_shape,
                  dtype,
                  scale_shape,
                  np.float32,
                  use_gpu=use_gpu,
                  data_format=data_format,
                  is_training=is_training,
              )
            else:
              if is_training:
                self._test_training(
                    x_shape,
                    dtype,
                    scale_shape,
                    np.float32,
                    use_gpu=use_gpu,
                    data_format=data_format,
                    exponential_avg_factor=exponential_avg_factor,
                )
              else:
                self._test_inference(
                    x_shape,
                    dtype,
                    scale_shape,
                    np.float32,
                    use_gpu=use_gpu,
                    data_format=data_format,
                    exponential_avg_factor=exponential_avg_factor,
                )

  def testInferenceShape1(self):
    # [FUNCTION DOC]
    # Inference on shape one, one, six, one.
    x_shape = [1, 1, 6, 1]
    self._runtests(x_shape, False)

  def testInferenceShape2(self):
    # [FUNCTION DOC]
    # Inference on shape one, one, six, two.
    x_shape = [1, 1, 6, 2]
    self._runtests(x_shape, False)

  def testInferenceShape3(self):
    # [FUNCTION DOC]
    # Inference on shape one, two, one, six.
    x_shape = [1, 2, 1, 6]
    self._runtests(x_shape, False)

  def testInferenceShape4(self):
    # [FUNCTION DOC]
    # Inference on a large 27 by 131 by 127 by 6 shape.
    x_shape = [27, 131, 127, 6]
    self._runtests(x_shape, False)

  def testInferenceShape5(self):
    # [FUNCTION DOC]
    # Inference on an empty batch.
    x_shape = [0, 131, 127, 6]
    self._runtests(x_shape, False)

  def testTrainingShape1(self):
    # [FUNCTION DOC]
    # Training on shape one, one, six, one.
    x_shape = [1, 1, 6, 1]
    self._runtests(x_shape, True)

  def testTrainingShape2(self):
    # [FUNCTION DOC]
    # Training on shape one, one, six, two.
    x_shape = [1, 1, 6, 2]
    self._runtests(x_shape, True)

  def testTrainingShape3(self):
    # [FUNCTION DOC]
    # Training on shape one, two, one, six.
    x_shape = [1, 2, 1, 6]
    self._runtests(x_shape, True)

  def testTrainingShape4(self):
    # [FUNCTION DOC]
    # Training on the large shape.
    x_shape = [27, 131, 127, 6]
    self._runtests(x_shape, True)

  def _test_gradient(
      self,
      x_shape,
      x_dtype,
      scale_shape,
      scale_dtype,
      use_gpu=True,
      data_format="NHWC",
      is_training=True,
  ):
    # [FUNCTION DOC]
    # HELPER: gradient error check of the fused batch norm w.r.t. input, scale and offset.
    # KNOWN ISSUE: the half precision branch calls a method of the class that does not exist in this
    #   file, so that branch would raise an attribute error if reached.
    # Inputs: shape, dtype, scale shape and dtype, GPU flag, data format, and training flag.
    np.random.seed(1)
    x_val = np.random.random_sample(x_shape).astype(x_dtype)
    scale_val = np.random.random_sample(scale_shape).astype(scale_dtype)
    offset_val = np.random.random_sample(scale_shape).astype(scale_dtype)

    with self.cached_session(use_gpu=use_gpu):
      x = constant_op.constant(x_val, name="x")
      scale = constant_op.constant(scale_val, name="scale")
      offset = constant_op.constant(offset_val, name="offset")
      if is_training:
        pop_mean = None
        pop_var = None
      else:
        pop_mean = np.random.random_sample(scale_shape).astype(scale_dtype)
        pop_var = np.random.random_sample(scale_shape).astype(scale_dtype)
      y, _, _ = nn_impl.fused_batch_norm(
          x,
          scale,
          offset,
          mean=pop_mean,
          variance=pop_var,
          data_format=data_format,
          is_training=is_training,
      )
      if x_dtype != np.float16:
        err_x = gradient_checker.compute_gradient_error(x, x_shape, y, x_shape)
        err_scale = gradient_checker.compute_gradient_error(
            scale, scale_shape, y, x_shape
        )
        err_offset = gradient_checker.compute_gradient_error(
            offset, scale_shape, y, x_shape
        )
      else:
        x32 = constant_op.constant(x_val, name="x32", dtype=dtypes.float32)
        y32, _, _ = nn_impl.fused_batch_norm(
            x32,
            scale,
            offset,
            mean=pop_mean,
            variance=pop_var,
            data_format=data_format,
            is_training=is_training,
        )
        err_x = self._compute_gradient_error_float16(
            x, x32, x_shape, y, y32, x_shape
        )
        err_scale = self._compute_gradient_error_float16(
            scale, scale, scale_shape, y, y32, x_shape
        )
        err_offset = self._compute_gradient_error_float16(
            offset, offset, scale_shape, y, y32, x_shape
        )

    x_err_tolerance = 2e-3 if x_dtype == np.float16 else 1e-3
    scale_err_tolerance = 1e-3
    self.assertLess(err_x, x_err_tolerance)
    self.assertLess(err_scale, scale_err_tolerance)
    self.assertLess(err_offset, scale_err_tolerance)

  @test_util.run_deprecated_v1
  def testBatchNormGradShape1(self):
    # [FUNCTION DOC]
    # Deprecated-v1 gradient test on shape one, one, six, one for training and inference, both formats,
    # adding GPU runs when CUDA exists.
    for is_training in [True, False]:
      x_shape = [1, 1, 6, 1]
      for dtype in [np.float32]:
        if test.is_gpu_available(cuda_only=True):
          self._test_gradient(
              x_shape,
              dtype,
              [1],
              np.float32,
              use_gpu=True,
              data_format="NHWC",
              is_training=is_training,
          )
          self._test_gradient(
              x_shape,
              dtype,
              [1],
              np.float32,
              use_gpu=True,
              data_format="NCHW",
              is_training=is_training,
          )
        self._test_gradient(
            x_shape,
            dtype,
            [1],
            np.float32,
            use_gpu=False,
            data_format="NHWC",
            is_training=is_training,
        )
        self._test_gradient(
            x_shape,
            dtype,
            [1],
            np.float32,
            use_gpu=False,
            data_format="NCHW",
            is_training=is_training,
        )

  @test_util.run_deprecated_v1
  def testBatchNormGradShape2(self):
    # [FUNCTION DOC]
    # Gradient test on shape one, one, six, two in the channels-last format.
    for is_training in [True, False]:
      x_shape = [1, 1, 6, 2]
      for dtype in [np.float32]:
        if test.is_gpu_available(cuda_only=True):
          self._test_gradient(
              x_shape,
              dtype,
              [2],
              np.float32,
              use_gpu=True,
              data_format="NHWC",
              is_training=is_training,
          )
        self._test_gradient(
            x_shape,
            dtype,
            [2],
            np.float32,
            use_gpu=False,
            data_format="NHWC",
            is_training=is_training,
        )

  @test_util.run_deprecated_v1
  def testBatchNormGradShape3(self):
    # [FUNCTION DOC]
    # Gradient test on shape one, two, one, six in the channels-first format.
    for is_training in [True, False]:
      x_shape = [1, 2, 1, 6]
      for dtype in [np.float32]:
        if test.is_gpu_available(cuda_only=True):
          self._test_gradient(
              x_shape,
              dtype,
              [2],
              np.float32,
              use_gpu=True,
              data_format="NCHW",
              is_training=is_training,
          )
        self._test_gradient(
            x_shape,
            dtype,
            [2],
            np.float32,
            use_gpu=False,
            data_format="NCHW",
            is_training=is_training,
        )


class SumReductionTest(BaseReductionTest):
  # [CLASS DOC]
  # Tests the sum reduction using the base class hooks.

  def _tf_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # Returns the TensorFlow reduce-sum.
    return math_ops.reduce_sum(x, reduction_axes, keepdims)

  def _np_reduce(self, x, reduction_axes, keepdims):
    # [FUNCTION DOC]
    # Returns the numpy sum, converting a list or array of axes to a tuple.
    if isinstance(reduction_axes, list) or isinstance(
        reduction_axes, np.ndarray
    ):
      reduction_axes = tuple(reduction_axes)
    return np.sum(x, axis=reduction_axes, keepdims=keepdims)

  def testAxesType(self):
    # [FUNCTION DOC]
    # Reduce-sum with an axis tensor of 32-bit or 64-bit integer type; asserts zero.
    for dtype in [dtypes.int64, dtypes.int32]:
      with self.cached_session(use_gpu=True) as _:
        v = math_ops.reduce_sum([0, 0], constant_op.constant(0, dtype=dtype))
        tf_v = self.evaluate(v)
      self.assertAllEqual(tf_v, 0)

  def testFloat32(self):
    # [FUNCTION DOC]
    # Random sizes of ones matrices, checks column, row and last-axis sums, then a grid of 3D sizes
    # checking sums over combined axes.
    for _ in range(5):
      size_x = int(2 ** np.random.uniform(0, 15))
      size_y = int(2 ** np.random.uniform(0, 15))
      if size_x * size_y > 1e7:
        size_y = int(1e7 / size_x)
      if size_x % 2:
        size_x = size_x + 1
      if size_y % 2:
        size_y = size_y + 1
      arr = np.ones([size_x, size_y], dtype=np.float32)
      col_sum = np.sum(arr, axis=0)
      row_sum = np.sum(arr, axis=1)
      full_sum = np.sum(arr, axis=-1, keepdims=True)
      with self.cached_session(use_gpu=True) as _:
        tf_row_sum = self._tf_reduce(arr, 1, False)
        tf_col_sum = self._tf_reduce(arr, 0, False)
        tf_full_sum = self._tf_reduce(arr, -1, keepdims=True)
        tf_out_col = self.evaluate(tf_col_sum)
        tf_out_row = self.evaluate(tf_row_sum)
        tf_out_full = self.evaluate(tf_full_sum)
      self.assertAllClose(col_sum, tf_out_col)
      self.assertAllClose(row_sum, tf_out_row)
      self.assertAllClose(full_sum, tf_out_full)

    for size_x in [4, 16, 32]:
      for size_y in [4, 16, 32]:
        for size_z in [4, 16, 32]:
          arr = np.ones([size_x, size_y, size_z], dtype=np.float32)
          sum_y = np.sum(arr, axis=1)
          sum_xz = np.sum(arr, axis=(0, 2))

          with self.cached_session(use_gpu=True) as _:
            tf_sum_xz = self._tf_reduce(arr, [0, 2], False)
            tf_sum_y = self._tf_reduce(arr, 1, False)
            tf_out_sum_xz, tf_out_sum_y = self.evaluate([tf_sum_xz, tf_sum_y])
          self.assertAllClose(sum_y, tf_out_sum_y)
          self.assertAllClose(sum_xz, tf_out_sum_xz)


class MinMaxOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests element-wise minimum and maximum, broadcast and gradients.

  def _compare(self, x, y, use_gpu):
    # [FUNCTION DOC]
    # HELPER: compares minimum and maximum ops with numpy on the chosen device.
    np_min, np_max = np.minimum(x, y), np.maximum(x, y)
    with test_util.device(use_gpu=use_gpu):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      omin, omax = math_ops.minimum(inx, iny), math_ops.maximum(inx, iny)
      tf_min, tf_max = self.evaluate([omin, omax])
    self.assertAllEqual(np_min, tf_min)
    self.assertAllEqual(np_max, tf_max)

  def testBasic(self):
    # [FUNCTION DOC]
    # Several dtypes on CPU and GPU.
    x = np.random.rand(1, 3, 2) * 100.0
    y = np.random.rand(1, 3, 2) * 100.0
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      self._compare(x.astype(t), y.astype(t), use_gpu=False)
      self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testDifferentShapes(self):
    # [FUNCTION DOC]
    # Broadcasting a vector against a three dimensional array.
    x = np.random.rand(1, 3, 2) * 100.0
    y = np.random.rand(2) * 100.0  # should broadcast
    for t in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      self._compare(x.astype(t), y.astype(t), use_gpu=False)
      self._compare(x.astype(t), y.astype(t), use_gpu=True)

  def testScalar(self):
    # [FUNCTION DOC]
    # Broadcasting a scalar (float32 and int32 only, since double is converted).
    x = np.random.rand(1, 3, 2) * 100.0
    y = np.random.rand(1).item() * 100.0  # should broadcast
    # dropped np.float64, int64 because TF automatically converts to 32 bit
    for t in [np.float32, np.int32]:
      self._compare(x.astype(t), t(y), use_gpu=False)
      self._compare(x.astype(t), t(y), use_gpu=True)

  def _compareGradientX(self, func, x, y):
    # [FUNCTION DOC]
    # HELPER: numeric versus theoretical gradient for the first input with dtype dependent tolerance.
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, s, out, s, x_init_value=x
      )
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, func, x, y):
    # [FUNCTION DOC]
    # HELPER: same for the second input.
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = func(inx, iny)
      s = list(np.shape(x))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, s, out, s, x_init_value=y
      )
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Chooses y distinct from x (plus or minus one) and checks both gradients of maximum and minimum.
    x = np.random.rand(1, 3, 2) * 100.0
    # ensure x != y
    y = x + (np.random.randint(2, size=x.shape) - 0.5) * 2  # -1 or +1
    self._compareGradientX(math_ops.maximum, x, y)
    self._compareGradientY(math_ops.maximum, x, y)
    self._compareGradientX(math_ops.minimum, x, y)
    self._compareGradientY(math_ops.minimum, x, y)


class MPSFillTest(test.TestCase):
  # [CLASS DOC]
  # Tests the fill op on the GPU against CPU.

  def _compare(self, dims, val, np_ans, use_gpu):
    # [FUNCTION DOC]
    # HELPER: fills on the GPU (when available) or CPU and compares with the expected array.
    ctx = context.context()
    device = "GPU:0" if (use_gpu and ctx.num_gpus()) else "CPU:0"
    with ops.device(device):
      tf_ans = array_ops.fill(dims, val, name="fill")
      out = tf_ans.numpy()
    self.assertAllClose(np_ans, out)

  def _compareAll(self, dims, val, np_ans):
    # [FUNCTION DOC]
    # HELPER: runs the comparison with both device flags.
    self._compare(dims, val, np_ans, False)
    self._compare(dims, val, np_ans, True)

  def testFillFloat(self):
    # [FUNCTION DOC]
    # Fills a two by three tensor with pi-like float32 and checks it.
    np_ans = np.array([[3.1415] * 3] * 2).astype(np.float32)
    self._compareAll([2, 3], np_ans[0][0], np_ans)


@test_util.run_all_in_graph_and_eager_modes
class SquaredDifferenceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests squared difference; class decorated to run in graph and eager modes.

  def testSquaredDifference(self):
    # [FUNCTION DOC]
    # Checks real dtypes with broadcasting against the explicit formula.
    for dtype in [np.float16, np.float32, np.float64, np.int32, np.int64]:
      x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
      y = np.array([-3, -2, -1], dtype=dtype)
      z = (x - y) * (x - y)
      with test_util.device(use_gpu=True):
        z_tf = self.evaluate(math_ops.squared_difference(x, y))
        self.assertAllClose(z, z_tf)

  def testComplexSquaredDifference(self):
    # [FUNCTION DOC]
    # Checks complex dtypes against conjugate-based formula, on CPU.
    for dtype in [np.complex64, np.complex128]:
      x = np.array(
          [[1 + 3j, 2 + 2j, 3 + 1j], [4 - 1j, 5 - 2j, 6 - 3j]], dtype=dtype
      )
      y = np.array([-3 + 1j, -2 + 2j, -1 + 3j], dtype=dtype)
      z = np.conj(x - y) * (x - y)
      with test_util.device(use_gpu=False):
        z_tf = self.evaluate(math_ops.squared_difference(x, y))
        self.assertAllClose(z, z_tf)


class MPSOnesLikeTest(test.TestCase):
  # [CLASS DOC]
  # Tests the ones-like op.

  def testOnesLike(self):
    # [FUNCTION DOC]
    # For many dtypes, builds a two by three tensor of ones, applies ones-like and asserts dtype, values
    # and static shape. The inline comment inside it says zeros where ones are meant.
    for dtype in [
        dtypes.float32,
        dtypes.float64,
        dtypes.int32,
        dtypes.uint8,
        dtypes.int16,
        dtypes.int8,
        dtypes.complex64,
        dtypes.complex128,
        dtypes.int64,
    ]:
      numpy_dtype = dtype.as_numpy_dtype
      # Creates a tensor of non-zero values with shape 2 x 3.
      d = constant_op.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
      # Constructs a tensor of zeros of the same dimensions and type as "d".
      z_var = array_ops.ones_like(d)
      # Test that the type is correct
      self.assertEqual(z_var.dtype, dtype)
      z_value = z_var.numpy()

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[1] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())


class SelectOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests select/where ops (V1 and V2) with broadcasting, gradients, errors and NaNs.

  def _compare(self, fn, c, x, y, use_gpu):
    # [FUNCTION DOC]
    # HELPER: compares the op result with numpy where for the chosen device.
    np_ans = np.where(c, x, y)
    with test_util.device(use_gpu=use_gpu):
      out = fn(c, x, y)
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)
    self.assertShapeEqual(np_ans, out)

  def _compareGradientX(
      self, fn, c, x, y, numeric_gradient_type=None, x_init_value=None
  ):
    # [FUNCTION DOC]
    # HELPER: gradient check for the first value input, optionally with a numeric gradient in another type.
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = fn(c, inx, iny)
      s = list(np.shape(c))
      if x_init_value is None:
        x_init_value = x
      if x.shape != y.shape:
        x_init_value = np.broadcast_to(y, x.shape)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, s, out, s, x_init_value=x_init_value
      )
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = fn(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, s, outf, s, x_init_value=xf
        )
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _compareGradientY(self, fn, c, x, y, numeric_gradient_type=None):
    # [FUNCTION DOC]
    # HELPER: same for the second value input.
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = fn(c, inx, iny)
      s = list(np.shape(c))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, s, out, s, x_init_value=x, delta=1.0
      )
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = fn(c, inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inyf, s, outf, s, x_init_value=yf
        )
        jacob_n = jacob_n.astype(x.dtype)
    if x.dtype == np.float16:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float32:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)
    elif x.dtype == np.float64:
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-5, atol=1e-5)

  def _testScalar(self, fn):
    # [FUNCTION DOC]
    # HELPER: scalar condition over seven dtypes, GPU for float types.
    c = True
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.complex64,
        np.complex128,
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testScalar(self):
    # [FUNCTION DOC]
    # Runs scalar test for where and where_v2.
    self._testScalar(array_ops.where)
    self._testScalar(array_ops.where_v2)

  def _testScalarBroadcast(self, fn, c, x, y):
    # [FUNCTION DOC]
    # HELPER: scalar condition with broadcast values.
    for t in [
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.complex64,
        np.complex128,
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testScalarBroadcast(self):
    # [FUNCTION DOC]
    # Many broadcast shape pairs, both orders (V2 only).
    c = True
    # where_v2 only
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(3, 2) * 100
    self._testScalarBroadcast(array_ops.where_v2, c, x, y)
    self._testScalarBroadcast(array_ops.where_v2, c, y, x)

  def _testBasic(self, fn):
    # [FUNCTION DOC]
    # HELPER: basic element-wise select.
    # KNOWN ISSUE: uses the removed numpy boolean alias; also the CPU comparison is commented out, so
    #   only float32 on the GPU is checked.
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float32]:
      xt = x.astype(t)
      yt = y.astype(t)
      # self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float32]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testBasic(self):
    # [FUNCTION DOC]
    # Runs basic select for where and where_v2.
    self._testBasic(array_ops.where)
    self._testBasic(array_ops.where_v2)

  def _testBasicBroadcast(self, fn, c, x, y):
    # [FUNCTION DOC]
    # HELPER: condition tensor with broadcast values.
    for t in [
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.complex64,
        np.complex128,
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(fn, c, xt, yt, use_gpu=False)
      if t in [np.float16, np.float32, np.float64]:
        self._compare(fn, c, xt, yt, use_gpu=True)

  def testBasicBroadcast(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the removed numpy boolean alias to build conditions; fails on new numpy.
    # Four condition shapes against many value shapes (V2 only).
    c0 = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    c1 = np.random.randint(0, 2, 2).astype(np.bool).reshape(1, 1, 2)
    c2 = np.random.randint(0, 2, 3).astype(np.bool).reshape(1, 3, 1)
    c3 = np.random.randint(0, 2, 1).astype(np.bool).reshape(1, 1, 1)
    for c in [c0, c1, c2, c3]:
      # where_v2 only
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 3, 1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 2) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 2) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(3, 2) * 100
      self._testBasicBroadcast(array_ops.where_v2, c, x, y)
      self._testBasicBroadcast(array_ops.where_v2, c, y, x)

  def _testGradients(self, fn):
    # [FUNCTION DOC]
    # HELPER: gradient checks for half precision (against single precision numeric gradients), single and double.
    # KNOWN ISSUE: uses the removed numpy boolean and float aliases; fails on new numpy.
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(1, 3, 2) * 100
    for t in [np.float16, np.float32, np.float64]:
      xt = x.astype(t)
      yt = y.astype(t)
      if t == np.float16:
        # Compare fp16 theoretical gradients to fp32 numerical gradients,
        # since fp16 numerical gradients are too imprecise unless great
        # care is taken with choosing the inputs and the delta. This is
        # a weaker check (in particular, it does not test the op itself,
        # only its gradient), but it's much better than nothing.
        self._compareGradientX(fn, c, xt, yt, np.float)
        self._compareGradientY(fn, c, xt, yt, np.float)
      else:
        self._compareGradientX(fn, c, xt, yt)
        self._compareGradientY(fn, c, xt, yt)

  @test_util.run_deprecated_v1
  def testGradients(self):
    # [FUNCTION DOC]
    # Gradient tests for where and where_v2.
    self._testGradients(array_ops.where)
    self._testGradients(array_ops.where_v2)

  @test_util.run_deprecated_v1
  def testGradientsBroadcast(self):
    # [FUNCTION DOC]
    # Gradient with broadcast values (V2), single and double precision.
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    for t in [np.float32, np.float64]:
      # where_v2 only
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 3, 1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1, 2) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(1, 2) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))
      x = np.random.rand(1, 3, 2) * 100
      y = np.random.rand(3, 2) * 100
      self._compareGradientX(array_ops.where_v2, c, x.astype(t), y.astype(t))

  def _testShapeMismatch(self, fn):
    # [FUNCTION DOC]
    # HELPER: asserts a value error when value shapes are incompatible.
    c = np.random.randint(0, 2, 6).astype(np.bool).reshape(1, 3, 2)
    x = np.random.rand(1, 3, 2) * 100
    y = np.random.rand(2, 5, 3) * 100
    for t in [
        np.float16,
        np.float32,
        np.float64,
        np.int32,
        np.int64,
        np.complex64,
        np.complex128,
    ]:
      xt = x.astype(t)
      yt = y.astype(t)
      with self.assertRaises(ValueError):
        fn(c, xt, yt)

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    # [FUNCTION DOC]
    # Shape mismatch tests for both ops.
    self._testShapeMismatch(array_ops.where)
    self._testShapeMismatch(array_ops.where_v2)

  def _testEmptyTensor(self, fn):
    # [FUNCTION DOC]
    # HELPER: empty condition and values give an empty float32 result.
    # KNOWN ISSUE: uses the removed numpy boolean alias; fails on new numpy.
    c = np.random.randint(0, 3, 0).astype(np.bool).reshape(1, 3, 0)
    x = np.random.rand(1, 3, 0) * 100
    y = np.random.rand(1, 3, 0) * 100
    z_expected = np.zeros((1, 3, 0), dtype=np.float32)
    with self.cached_session():
      xt = x.astype(np.float32)
      yt = y.astype(np.float32)
      z = fn(c, xt, yt).eval()
      self.assertAllEqual(z_expected, z)

  @test_util.run_deprecated_v1
  def testEmptyTensor(self):
    # [FUNCTION DOC]
    # Empty tensor tests for both ops.
    self._testEmptyTensor(array_ops.where)
    self._testEmptyTensor(array_ops.where_v2)

  def _testNan(self, fn):
    # [FUNCTION DOC]
    # HELPER: for all combinations of condition and NaN or regular values, asserts NaN propagation
    # matches the selected element only.
    with self.cached_session():
      for c in False, True:
        for a in 7.0, np.nan:
          for b in 5.0, np.nan:
            x = fn(c, a, b).eval()
            y = a if c else b
            self.assertEqual(np.isnan(x), np.isnan(y))

  @test_util.run_deprecated_v1
  def testNan(self):
    # [FUNCTION DOC]
    # Verifies that NaNs do not propagate from the unselected branch.
    """Verify that nans don't propagate where they shouldn't."""
    self._testNan(array_ops.where)
    self._testNan(array_ops.where_v2)


class ZerosLikeTest(test.TestCase):
  # [CLASS DOC]
  # Tests the zeros-like op.

  def _compareZeros(self, dtype, use_gpu):
    # [FUNCTION DOC]
    # HELPER: builds ones of two by three of the dtype (strings use byte strings), applies zeros-like and
    # asserts dtype, static shape and all zero values. The use_gpu flag is unused.
    # Creates a tensor of non-zero values with shape 2 x 3.
    # NOTE(kearnes): The default numpy dtype associated with tf.string is
    # np.object (and can't be changed without breaking a lot things), which
    # causes a TypeError in constant_op.constant below. Here we catch the
    # special case of tf.string and set the numpy dtype appropriately.
    if dtype == dtypes.string:
      numpy_dtype = np.bytes_
    else:
      numpy_dtype = dtype.as_numpy_dtype
    d = constant_op.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
    # Constructs a tensor of zeros of the same dimensions and type as "d".
    z_var = array_ops.zeros_like(d)
    # Test that the type is correct
    self.assertEqual(z_var.dtype, dtype)
    # Test that the shape is correct
    self.assertEqual([2, 3], z_var.get_shape())

    # Test that the value is correct
    z_value = z_var.numpy()
    self.assertFalse(np.any(z_value))
    self.assertEqual((2, 3), z_value.shape)

  def testZerosLikeCPU(self):
    # [FUNCTION DOC]
    # Several numeric dtypes, flag off.
    for dtype in [
        dtypes.float32,
        dtypes.float64,
        dtypes.int32,
        dtypes.uint8,
        dtypes.int16,
        dtypes.int8,
        dtypes.complex64,
        dtypes.complex128,
        dtypes.int64,
    ]:
      self._compareZeros(dtype, use_gpu=False)

  def testZerosLikeGPU(self):
    # [FUNCTION DOC]
    # Fewer dtypes including boolean, flag on.
    for dtype in [
        dtypes.float32,
        dtypes.float64,
        dtypes.int32,
        dtypes.bool,
        dtypes.int64,
    ]:
      self._compareZeros(dtype, use_gpu=True)

  def testZerosLikeDtype(self):
    # [FUNCTION DOC]
    # Zeros-like with dtype conversions between single float and complex; asserts dtype, shape, values.
    # Make sure zeros_like works even for dtypes that cannot be cast between
    shape = (3, 5)
    dtypes_ = np.float32, np.complex64
    for in_type in dtypes_:
      x = np.arange(15).astype(in_type).reshape(*shape)
      for out_type in dtypes_:
        y = array_ops.zeros_like(x, dtype=out_type).numpy()
        self.assertEqual(y.dtype, out_type)
        self.assertEqual(y.shape, shape)
        self.assertAllEqual(y, np.zeros(shape, dtype=out_type))


class MpsTest(test.TestCase):
  # [CLASS DOC]
  # Main plugin test: compares GPU results with CPU or numpy for arithmetic, comparison, gradients,
  # unary ops, convolution, transpose and broadcasting, plus variable assign tests.
  # Class attribute: gradient tolerance per dtype.

  def _compareCpu(self, x, y, np_func, tf_func, also_compare_variables=False):
    # [FUNCTION DOC]
    # HELPER: compares a TensorFlow binary function on the CPU with the numpy function; also with numpy
    # KNOWN ISSUE: uses the removed numpy object alias in a dtype comparison; fails on new numpy.
    # operands on either side (to check precedence) and optionally with variable operands. Prints types
    # in the variable case. Skips the value comparison for object dtype results, then checks shape.
    np_ans = np_func(x, y)
    with test_util.force_cpu():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = self.evaluate(out)
      # Test that the op takes precedence over numpy operators.
      np_left = self.evaluate(tf_func(x, iny))
      np_right = self.evaluate(tf_func(inx, y))

      if also_compare_variables:
        var_x = variables.Variable(x)
        var_y = variables.Variable(y)
        self.evaluate(variables.global_variables_initializer())
        print(type(x), type(y), type(var_x), type(var_y))
        print(type(tf_func(x, var_y)), type(tf_func(var_x, y)))
        np_var_left = self.evaluate(tf_func(x, var_y))
        np_var_right = self.evaluate(tf_func(var_x, y))

    if np_ans.dtype != np.object:
      self.assertAllClose(np_ans, tf_cpu)
      self.assertAllClose(np_ans, np_left)
      self.assertAllClose(np_ans, np_right)
      if also_compare_variables:
        self.assertAllClose(np_ans, np_var_left)
        self.assertAllClose(np_ans, np_var_right)
    self.assertShapeEqual(np_ans, out)

  def _inv(self, x):
    # [FUNCTION DOC]
    # HELPER: reciprocal reference.
    return 1.0 / x

  def _rsqrt(self, x):
    # [FUNCTION DOC]
    # HELPER: reciprocal square root reference.
    return self._inv(np.sqrt(x))

  def _sigmoid(self, x):
    # [FUNCTION DOC]
    # HELPER: logistic sigmoid reference.
    return 1.0 / (1.0 + np.exp(-x))

  def _log_sigmoid(self, x):
    # [FUNCTION DOC]
    # HELPER: log of sigmoid reference; not used elsewhere in this file.
    return np.log(self._sigmoid(x))

  def _replace_domain_error_with_inf(self, fn):
    # [FUNCTION DOC]
    # HELPER: wraps a function so that a value error mentioning a domain error becomes an infinity
    # array of the same shape; other errors are re-raised. Not used elsewhere in this file.
    def func(x):
      # [FUNCTION DOC]
      # HELPER: the wrapper returned: calls the function and maps domain errors to infinity.
      try:
        return fn(x)
      except ValueError as e:
        if "domain error" in str(e):
          return np.inf * np.ones_like(x)
        else:
          raise e

    return func

  def _compareTanhGrad(self, x, y):
    # [FUNCTION DOC]
    # HELPER: compares the tanh gradient op on the default device with the CPU.
    default = gen_math_ops.tanh_grad(x, y)

    with test_util.device(use_gpu=False):
      cpu = gen_math_ops.tanh_grad(x, y)

    self.assertAllClose(cpu, default)

  def testTanhGrad(self):
    # [FUNCTION DOC]
    # Random inputs in the range minus two to two.
    x = np.random.uniform(-2.0, 2.0, size=[4, 4]).astype(np.float32)
    y = np.random.uniform(-2.0, 2.0, size=[4, 4]).astype(np.float32)
    self._compareTanhGrad(x, y)

  _GRAD_TOL = {
      dtypes.float16: 1e-3,
      dtypes.float32: 1e-3,
      dtypes.complex64: 1e-2,
      dtypes.float64: 1e-5,
      dtypes.complex128: 1e-4,
  }

  def _compareGradientX(
      self, x, y, np_func, tf_func, numeric_gradient_type=None
  ):
    # [FUNCTION DOC]
    # HELPER: numeric vs theoretical gradient for the first input; scales output by 1.1 for float32 and
    # float64; dtype specific tolerance; optional numeric gradient dtype.
    z = np_func(x, y)
    zs = list(z.shape)
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      xs = list(x.shape)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          inx, xs, out, zs, x_init_value=x
      )
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = tf_func(inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, xs, outf, zs, x_init_value=xf, delta=1e-3
        )
        jacob_n = jacob_n.astype(x.dtype)
      tol = self._GRAD_TOL[dtypes.as_dtype(x.dtype)]
      self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def _compareGradientY(
      self, x, y, np_func, tf_func, numeric_gradient_type=None
  ):
    # [FUNCTION DOC]
    # HELPER: same for the second input.
    z = np_func(x, y)
    zs = list(z.shape)
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      if x.dtype in (np.float32, np.float64):
        out = 1.1 * tf_func(inx, iny)
      else:
        out = tf_func(inx, iny)
      ys = list(np.shape(y))
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          iny, ys, out, zs, x_init_value=y
      )
      if numeric_gradient_type is not None:
        xf = x.astype(numeric_gradient_type)
        yf = y.astype(numeric_gradient_type)
        inxf = ops.convert_to_tensor(xf)
        inyf = ops.convert_to_tensor(yf)
        outf = tf_func(inxf, inyf)
        _, jacob_n = gradient_checker.compute_gradient(
            inyf, ys, outf, zs, x_init_value=yf
        )
        jacob_n = jacob_n.astype(x.dtype)
    tol = self._GRAD_TOL[dtypes.as_dtype(x.dtype)]
    self.assertAllClose(jacob_t, jacob_n, rtol=tol, atol=tol)

  def compareUnaryGradient_CPU_GPU(self, inx, func, test_name):
    # [FUNCTION DOC]
    # HELPER: records a gradient tape on the CPU and then the GPU, prints each and asserts closeness.
    with test_util.force_cpu():
      with backprop.GradientTape() as t:
        t.watch(inx)
        y = func(inx)
      cpu_gradient = t.gradient(y, inx)
      print(test_name, " (CPU) = ", cpu_gradient)

    with test_util.force_gpu():
      with backprop.GradientTape() as t:
        t.watch(inx)
        y = func(inx)
      gpu_gradient = t.gradient(y, inx)
      print(test_name, " (GPU) = ", gpu_gradient)

    tol = self._GRAD_TOL[dtypes.as_dtype(inx.dtype)]
    self.assertAllClose(cpu_gradient, gpu_gradient, rtol=tol, atol=tol)

  def _compareGpu(self, x, y, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: GPU result compared with numpy, plus shape.
    np_ans = np_func(x, y)
    with test_util.use_gpu():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = self.evaluate(out)
    self.assertAllClose(np_ans, tf_gpu)
    self.assertShapeEqual(np_ans, out)

  def _compareBoth(self, x, y, np_func, tf_func, also_compare_variables=False):
    # [FUNCTION DOC]
    # HELPER: runs the CPU comparison then the GPU comparison.
    self._compareCpu(x, y, np_func, tf_func, also_compare_variables)
    self._compareGpu(x, y, np_func, tf_func)

  def _compare(self, x, y, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: GPU result compared exactly with numpy (used for comparisons).
    np_ans = np_func(x, y)
    with test_util.use_gpu():
      out = tf_func(ops.convert_to_tensor(x), ops.convert_to_tensor(y))
      tf_ans = self.evaluate(out)
    self.assertAllEqual(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testGradGrad(self):
    # [FUNCTION DOC]
    # Deprecated-v1 test; checks tanh-grad second order derivatives with the gradient checker over
    # several dtypes.
    np.random.seed(7)
    shape = (5,)
    dtype_tols = [
        (np.float32, 5e-4),
        (np.float64, 1e-6),
        (np.complex64, 5e-4),
        (np.complex128, 1e-6),
    ]
    op_range = [
        (gen_math_ops.tanh_grad, [-2, 2]),
    ]

    def rand(dtype, real_range):
      # [FUNCTION DOC]
      # HELPER: random array in a range of given dtype and the enclosing shape.
      x = np.random.uniform(real_range[0], real_range[1], size=shape[0]).astype(
          dtype
      )
      return x

    for op, real_range in op_range:
      with self.cached_session():
        for dtype, tol in dtype_tols:
          x = constant_op.constant(rand(dtype, real_range))
          y = constant_op.constant(rand(dtype, real_range))
          z = op(x, y)
          grads = gradient_checker.compute_gradient(
              [x, y],
              [shape, shape],
              z,
              shape,
              x_init_value=[rand(dtype, real_range), rand(dtype, real_range)],
          )
          if isinstance(grads, tuple):
            grads = [grads]
          for analytical, numerical in grads:
            self.assertAllClose(analytical, numerical, rtol=tol, atol=tol)

  def testFloatCompareTensor(self):
    # [FUNCTION DOC]
    # Compares less, less-equal, greater, greater-equal, equal and not-equal on GPU for two float types.
    x = np.linspace(-15, 15, 6).reshape((1, 3, 2))
    y = np.linspace(20, -10, 6).reshape((1, 3, 2))
    for t in [np.float32, np.float16]:
      xt = x.astype(t)
      yt = y.astype(t)
      self._compare(xt, yt, np.less, math_ops.less)
      self._compare(xt, yt, np.less_equal, math_ops.less_equal)
      self._compare(xt, yt, np.greater, math_ops.greater)
      self._compare(xt, yt, np.greater_equal, math_ops.greater_equal)
      self._compare(xt, yt, np.equal, math_ops.equal)
      self._compare(xt, yt, np.not_equal, math_ops.not_equal)

  def testFloatBasic(self):
    # [FUNCTION DOC]
    # Add, subtract, multiply, divide and floor-divide for float32, plus operator overloads.
    x = np.linspace(-5, 20, 30).reshape((1, 2, 3, 5)).astype(np.float32)
    y = np.linspace(20, -5, 30).reshape((1, 2, 3, 5)).astype(np.float32)
    self._compareBoth(x, y, np.add, math_ops.add, True)
    self._compareBoth(x, y, np.subtract, math_ops.subtract, True)
    self._compareBoth(x, y, np.multiply, math_ops.multiply, True)
    self._compareBoth(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testHalfBasic(self):
    # [FUNCTION DOC]
    # Same for float16.
    x = np.linspace(-5, 20, 30).reshape((1, 2, 3, 5)).astype(np.float16)
    y = np.linspace(20, -5, 30).reshape((1, 2, 3, 5)).astype(np.float16)
    self._compareBoth(x, y, np.add, math_ops.add, True)
    self._compareBoth(x, y, np.subtract, math_ops.subtract, True)
    self._compareBoth(x, y, np.multiply, math_ops.multiply, True)
    self._compareBoth(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testIntBasic(self):
    # [FUNCTION DOC]
    # Same for int32 including modulus and a GPU-only modulus check.
    x = np.arange(1, 13, 2).reshape(1, 3, 2).astype(np.int32)
    y = np.arange(1, 7, 1).reshape(1, 3, 2).astype(np.int32)
    self._compareBoth(x, y, np.add, math_ops.add)
    self._compareBoth(x, y, np.subtract, math_ops.subtract)
    self._compareBoth(x, y, np.multiply, math_ops.multiply)
    self._compareBoth(x, y, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.floor_divide, math_ops.floordiv)
    self._compareBoth(x, y, np.mod, math_ops.mod)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)
    self._compareBoth(x, y, np.true_divide, _TRUEDIV)
    self._compareBoth(x, y, np.floor_divide, _FLOORDIV)
    self._compareBoth(x, y, np.mod, _MOD)
    # _compareBoth tests on GPU only for floating point types, so test
    # _MOD for int32 on GPU by calling _compareGpu
    self._compareGpu(x, y, np.mod, _MOD)

  def testZeroElementBinaryOp(self):
    # [FUNCTION DOC]
    # Binary ops with a zero element tensor and a scalar.
    x = array_ops.ones([0, 3])
    y = 4.0
    self._compareBoth(x, y, np.add, math_ops.add, True)
    self._compareBoth(x, y, np.subtract, math_ops.subtract, True)
    self._compareBoth(x, y, np.multiply, math_ops.multiply, True)
    self._compareBoth(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareBoth(x, y, np.add, _ADD)
    self._compareBoth(x, y, np.subtract, _SUB)
    self._compareBoth(x, y, np.multiply, _MUL)

  def testAssignMethod(self):
    # [FUNCTION DOC]
    # Assign with and without read_value; checks the values, null in eager or an operation in graph.
    v = resource_variable_ops.ResourceVariable(1.0, name="var0")
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(v.assign(2.0))
    self.assertEqual(2.0, self.evaluate(v.value()))

    # Tests for the 'read_value' argument:
    assign_with_read = v.assign(3.0, read_value=True)
    self.assertEqual(3.0, self.evaluate(assign_with_read))
    assign_without_read = v.assign(4.0, read_value=False)
    if context.executing_eagerly():
      self.assertIsNone(assign_without_read)
    else:
      self.assertIsInstance(assign_without_read, ops.Operation)
    self.evaluate(assign_without_read)
    self.assertEqual(4.0, self.evaluate(v.value()))

  @test_util.run_in_graph_and_eager_modes
  def testAssignIncompatibleShape(self):
    # [FUNCTION DOC]
    # Assign-add of a scalar into a vector variable; expects an error mentioning shapes.
    v = resource_variable_ops.ResourceVariable([0, 1, 2, 3])
    self.evaluate(v.initializer)
    pattern = re.compile("shapes must be equal", re.IGNORECASE)
    with self.assertRaisesRegex(Exception, pattern):
      self.evaluate(v.assign_add(1))

  def _compareUnaryCpu(
      self, x, np_func, tf_func, grad_rtol=None, grad_atol=None
  ):
    # [FUNCTION DOC]
    # HELPER: unary op accuracy and gradient check against numpy on the CPU. Inputs: array, numpy
    # KNOWN ISSUE: uses the removed numpy float alias; fails on new numpy. Also never called in this file.
    # function, TensorFlow function, optional gradient tolerances (defaults per dtype).
    if grad_rtol is None:
      grad_rtol = _default_tolerance(x.dtype)
    if grad_atol is None:
      grad_atol = _default_tolerance(x.dtype)
    np_ans = np_func(x)
    with self.cached_session(use_gpu=False):
      inx = ops.convert_to_tensor(x)
      if x.dtype in (
          np.float32,
          np.float64,
          dtypes.bfloat16.as_numpy_dtype,
      ):
        y = 1.1 * tf_func(inx)
        np_ans *= 1.1
      else:
        y = tf_func(inx)
      tf_cpu = self.evaluate(y)
      self.assertShapeEqual(np_ans, y)
      if x.dtype == np.float16:
        self.assertAllClose(np_ans, tf_cpu, rtol=1e-3, atol=1e-3)
      elif x.dtype == dtypes.bfloat16.as_numpy_dtype:
        self.assertAllClose(np_ans, tf_cpu, rtol=1e-2, atol=1e-2)
      else:
        self.assertAllClose(np_ans, tf_cpu)

      if x.dtype in (np.complex64, np.complex128) and tf_func == math_ops.sign:
        return  # Return early

      if x.dtype == np.float16:
        s = list(np.shape(x))
        jacob_t, _ = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x
        )
        xf = x.astype(np.float)
        inxf = ops.convert_to_tensor(xf)
        yf = tf_func(inxf)
        _, jacob_n = gradient_checker.compute_gradient(
            inxf, s, yf, s, x_init_value=xf, delta=1e-2
        )
        jacob_n = jacob_n.astype(np.float16)
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
      elif x.dtype in (np.float32, np.complex64):
        s = list(np.shape(x))
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x, delta=1e-3
        )
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)
      elif x.dtype in (np.float64, np.complex128):
        s = list(np.shape(x))
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            inx, s, y, s, x_init_value=x, delta=1e-5
        )
        self.assertAllClose(jacob_t, jacob_n, rtol=grad_rtol, atol=grad_atol)

  def _compareUnaryGpu(self, x, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: unary op on GPU compared with numpy.
    np_ans = np_func(x)
    with test_util.use_gpu():
      result = tf_func(ops.convert_to_tensor(x))
      tf_gpu = self.evaluate(result)
    if x.dtype == np.float16:
      self.assertAllClose(np_ans, tf_gpu, rtol=1e-3, atol=1e-3)
    else:
      self.assertAllClose(np_ans, tf_gpu)

  def _compareUnaryBoth(self, x, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: wrapper around the GPU comparison.
    # KNOWN ISSUE: name says both devices but only the GPU comparison runs.
    self._compareUnaryGpu(x, np_func, tf_func)

  def compareConv2d(
      self, input, filter, padding, format="NHWC", dilations=None
  ):
    # [FUNCTION DOC]
    # HELPER: runs conv2d with stride two on GPU and CPU for a padding, format and dilations; prints
    # pass or error information and asserts equality. Input, filter, padding (string or list), data
    # format (default channels-last), dilations (default none). Channels-first CPU inputs are transposed.
    stride = 2

    strides = [stride, stride]

    with test_util.force_gpu():
      gpu = nn_ops.conv2d(
          input=input,
          filter=filter,
          strides=strides,
          padding=padding,
          data_format=format,
          dilations=dilations,
      )

    with test_util.force_cpu():
      if format == "NCHW":
        input = array_ops.transpose(input, [0, 2, 3, 1])
        if not isinstance(padding, str):
          padding = [padding[0], padding[2], padding[3], padding[1]]
      cpu = nn_ops.conv2d(
          input=input,
          filter=filter,
          strides=strides,
          padding=padding,
          data_format="NHWC",
          dilations=dilations,
      )
      if format == "NCHW":
        cpu = array_ops.transpose(cpu, [0, 3, 1, 2])

      if math_ops.reduce_any(math_ops.not_equal(cpu, gpu)):
        print(
            "Error: padding: {0} format: {1} dilations: {2}".format(
                padding, format, dilations
            )
        )
        print("CPU: ", cpu)
        print("GPU: ", gpu)
      else:
        print(
            "Passed: padding: {0} format: {1} dilations: {2}".format(
                padding, format, dilations
            )
        )
        print("CPU: ", cpu)
        print("GPU: ", gpu)

    self.assertAllEqual(cpu, gpu)

  def testConvolution(self):
    # [FUNCTION DOC]
    # Many input and filter combinations, with valid, same and explicit padding, dilations and
    # channels-first.
    input = constant_op.constant([[
        [[1], [2.0], [3.0], [4.0]],
        [[6], [7], [8], [9]],
        [[10], [11], [12], [13]],
        [[14], [15], [16], [17]],
    ]])

    input2 = constant_op.constant([[
        [[1], [2.0], [3.0], [4.0], [5.0]],
        [[6], [7], [8], [9], [15.0]],
        [[10], [11], [12], [13], [25.0]],
        [[14], [15], [16], [17], [35.0]],
    ]])

    input4 = constant_op.constant([[
        [[1], [2.0], [3.0], [4.0], [5.0], [1], [2.0]],
        [[6], [7], [8], [9], [15.0], [1], [2.0]],
        [[10], [11], [12], [13], [25.0], [1], [2.0]],
        [[14], [15], [16], [17], [35.0], [1], [2.0]],
        [[6], [7], [8], [9], [15.0], [1], [2.0]],
        [[10], [11], [12], [13], [25.0], [1], [2.0]],
    ]])

    print("input: ", input)

    ## (2,2,1,1)
    filter2x2 = constant_op.constant(
        [
            [[[1.0]], [[1]]],
            [[[1.0]], [[1]]],
        ],
    )

    ## (3,2,1,1)
    filter3x2 = constant_op.constant(
        [[[[1.0]], [[1]]], [[[1.0]], [[1]]], [[[1.0]], [[1]]]],
    )

    ## (4,2,1,1)
    filter4x2 = constant_op.constant(
        [
            [[[1.0]], [[1]]],
            [[[1.0]], [[1]]],
            [[[1.0]], [[1]]],
            [[[1.0]], [[1]]],
        ],
    )

    ## (5,2,1,1)
    filter5x2 = constant_op.constant(
        [
            [[[1.0]], [[1]]],
            [[[1.0]], [[1]]],
            [[[1.0]], [[1]]],
            [[[1.0]], [[1]]],
            [[[1.0]], [[1]]],
        ],
    )

    print("filter2x2: ", filter2x2)

    self.compareConv2d(input, filter2x2, "VALID")
    self.compareConv2d(input, filter3x2, "VALID")
    self.compareConv2d(input, filter4x2, "VALID")
    self.compareConv2d(input, filter5x2, "VALID")

    self.compareConv2d(input, filter2x2, "SAME")
    self.compareConv2d(input, filter3x2, "SAME")
    self.compareConv2d(input, filter4x2, "SAME")
    self.compareConv2d(input, filter5x2, "SAME")

    self.compareConv2d(input2, filter2x2, "VALID")
    self.compareConv2d(input2, filter2x2, "SAME")

    pad_top = 2
    pad_bottom = 3
    pad_left = 1
    pad_right = 5
    self.compareConv2d(
        input2,
        filter2x2,
        [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
    )

    self.compareConv2d(input2, filter2x2, "VALID", dilations=[2, 2])
    self.compareConv2d(input2, filter2x2, "SAME", dilations=[2, 2])

    self.compareConv2d(input4, filter2x2, "VALID", dilations=[2, 3])
    self.compareConv2d(input4, filter2x2, "SAME", dilations=[3, 2])

    self.compareConv2d(input4, filter3x2, "VALID", dilations=[2, 3])
    self.compareConv2d(input4, filter3x2, "SAME", dilations=[3, 2])

    self.compareConv2d(input4, filter5x2, "VALID", dilations=[2, 3])
    self.compareConv2d(input4, filter5x2, "SAME", dilations=[3, 2])
    self.compareConv2d(
        input2,
        filter2x2,
        [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        dilations=[2, 2],
    )

    input3 = constant_op.constant([[[
        [1, 2.0, 3.0, 4.0, 5.0],
        [6, 7, 8, 9, 15],
        [10, 11, 12, 13, 25.0],
        [14, 15, 16, 17, 35.0],
    ]]])

    self.compareConv2d(input3, filter2x2, "VALID", "NCHW")
    self.compareConv2d(input3, filter2x2, "SAME", "NCHW")
    self.compareConv2d(
        input3,
        filter2x2,
        [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]],
        "NCHW",
    )

  def compareTranspose(self, input, perm):
    # [FUNCTION DOC]
    # HELPER: transpose on GPU and CPU compared for a permutation; prints the outcome.
    with test_util.force_gpu():
      gpu = array_ops.transpose(input, perm)

    with test_util.force_cpu():
      cpu = array_ops.transpose(input, perm)

      if math_ops.reduce_any(math_ops.not_equal(cpu, gpu)):
        print("Error")
        print("CPU: ", cpu)
        print("GPU: ", gpu)
      else:
        print("Passed")

    self.assertAllEqual(cpu, gpu)

  def testTranspose(self):
    # [FUNCTION DOC]
    # Rank three and four tensors, float32 and bfloat16, with several permutations.
    for dtype in [dtypes.float32, dtypes.bfloat16]:
      input = tf.convert_to_tensor(np.arange(0.0, 5 * 2 * 13), dtype=dtype)
      input = array_ops.reshape(input, [5, 2, 13])

      self.compareTranspose(input, [1, 2, 0])
      self.compareTranspose(input, [0, 2, 1])
      self.compareTranspose(input, [2, 0, 1])
      self.compareTranspose(input, [2, 1, 0])

      input = tf.convert_to_tensor(np.arange(0.0, 2 * 4 * 3 * 5), dtype=dtype)
      input = array_ops.reshape(input, [2, 4, 3, 5])

      self.compareTranspose(input, [1, 0, 2, 3])
      self.compareTranspose(input, [0, 3, 1, 2])
      self.compareTranspose(input, [3, 2, 1, 0])

  def testUnaryHalfBasic(self):
    # [FUNCTION DOC]
    # Several unary ops (abs, negative, reciprocal, log, sigmoid, sqrt, rsqrt, exp, square, sign, tanh)
    # for float16 on GPU.
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float16)
    _ = x - x.min() + 1.02  # all greater than 1
    y = (x + 0.5).astype(np.float16)  # no zero
    z = (x + 15.5).astype(np.float16)  # all positive
    _ = np.arange(-0.90, 0.90, 0.25).astype(np.float16)  # between -1 and 1

    self._compareUnaryBoth(x, np.abs, math_ops.abs)
    self._compareUnaryBoth(x, np.abs, _ABS)
    self._compareUnaryBoth(x, np.negative, math_ops.negative)
    self._compareUnaryBoth(x, np.negative, _NEG)
    self._compareUnaryBoth(y, self._inv, math_ops.reciprocal)
    self._compareUnaryBoth(z, np.log, math_ops.log)
    self._compareUnaryBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareUnaryBoth(z, np.sqrt, math_ops.sqrt)
    self._compareUnaryBoth(z, self._rsqrt, math_ops.rsqrt)
    self._compareUnaryBoth(x, np.exp, math_ops.exp)
    self._compareUnaryBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareUnaryBoth(x, np.square, math_ops.square)
    self._compareUnaryBoth(y, np.sign, math_ops.sign)
    self._compareUnaryBoth(x, np.tanh, math_ops.tanh)

  def testUnaryFloatBasic(self):
    # [FUNCTION DOC]
    # Same for float32 plus log1p, then compares the sigmoid gradient CPU vs GPU and prints the result of
    # the sigmoid-grad op.
    x = np.arange(-3, 3).reshape(1, 3, 2).astype(np.float32)
    _ = x - x.min() + 1.02  # all greater than 1
    y = (x + 0.5).astype(np.float32)  # no zero
    z = (x + 15.5).astype(np.float32)  # all positive
    _ = np.arange(-0.90, 0.90, 0.25).astype(np.float32)  # between -1 and 1

    self._compareUnaryBoth(x, np.abs, math_ops.abs)
    self._compareUnaryBoth(x, np.abs, _ABS)
    self._compareUnaryBoth(x, np.negative, math_ops.negative)
    self._compareUnaryBoth(x, np.negative, _NEG)
    self._compareUnaryBoth(y, self._inv, math_ops.reciprocal)
    self._compareUnaryBoth(z, np.log, math_ops.log)
    self._compareUnaryBoth(x, np.square, math_ops.square)
    self._compareUnaryBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareUnaryBoth(z, np.sqrt, math_ops.sqrt)
    self._compareUnaryBoth(z, self._rsqrt, math_ops.rsqrt)
    self._compareUnaryBoth(x, np.exp, math_ops.exp)
    self._compareUnaryBoth(x, self._sigmoid, math_ops.sigmoid)
    self._compareUnaryBoth(z, np.log1p, math_ops.log1p)
    self._compareUnaryBoth(x, np.square, math_ops.square)
    self._compareUnaryBoth(y, np.sign, math_ops.sign)
    self._compareUnaryBoth(x, np.tanh, math_ops.tanh)

    x = np.array([0.5, 0.7], np.float32)
    inx = ops.convert_to_tensor(x)

    print("\nsigmoidGrad:\n")

    self.compareUnaryGradient_CPU_GPU(inx, gen_math_ops.sigmoid, "sigmoidGrad")

    gradient = gen_math_ops.sigmoid_grad(
        gen_math_ops.sigmoid(inx), constant_op.constant(1.0)
    )
    print("gen_math_ops.sigmoid_grad(y) = ", gradient)

  def _compareBCast(self, xs, ys, dtype, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: builds ranges of given shapes and compares CPU, plus GPU for float types.
    x = (1 + np.linspace(0, 5, np.prod(xs))).astype(dtype).reshape(xs)
    y = (1 + np.linspace(0, 5, np.prod(ys))).astype(dtype).reshape(ys)
    self._compareCpu(x, y, np_func, tf_func)
    if x.dtype in (np.float16, np.float32, np.float64):
      self._compareGpu(x, y, np_func, tf_func)

  def _testBCastByFunc(self, funcs, xs, ys):
    # [FUNCTION DOC]
    # HELPER: for each function pair compares both operand orders.
    dtypes_ = [
        np.float32,
    ]
    for dtype in dtypes_:
      for np_func, tf_func in funcs:
        self._compareBCast(xs, ys, dtype, np_func, tf_func)
        self._compareBCast(ys, xs, dtype, np_func, tf_func)

  def _testBCastA(self, xs, ys):
    # [FUNCTION DOC]
    # HELPER: add.
    funcs = [
        (np.add, math_ops.add),
        (np.add, _ADD),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastB(self, xs, ys):
    # [FUNCTION DOC]
    # HELPER: subtract and power.
    funcs = [
        (np.subtract, math_ops.subtract),
        (np.subtract, _SUB),
        (np.power, math_ops.pow),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastC(self, xs, ys):
    # [FUNCTION DOC]
    # HELPER: multiply.
    funcs = [
        (np.multiply, math_ops.multiply),
        (np.multiply, _MUL),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def _testBCastD(self, xs, ys):
    # [FUNCTION DOC]
    # HELPER: true division.
    funcs = [
        (np.true_divide, math_ops.truediv),
        (np.true_divide, _TRUEDIV),
    ]
    self._testBCastByFunc(funcs, xs, ys)

  def testBCast_0A(self):
    # [FUNCTION DOC]
    # Broadcast addition of shape one, three, two with shape one.
    self._testBCastA([1, 3, 2], [1])

  def testBCast_0B(self):
    # [FUNCTION DOC]
    # Subtract and power for the same shapes.
    self._testBCastB([1, 3, 2], [1])

  def testBCast_0C(self):
    # [FUNCTION DOC]
    # Multiply for the same shapes.
    self._testBCastC([1, 3, 2], [1])

  def testBCast_0D(self):
    # [FUNCTION DOC]
    # Division for the same shapes.
    self._testBCastD([1, 3, 2], [1])

  def testBCast_1A(self):
    # [FUNCTION DOC]
    # Addition of shape two, three, two with shape two.
    self._testBCastA([2, 3, 2], [2])

  def testBCast_1B(self):
    # [FUNCTION DOC]
    # Subtract and power of one, three, two with two.
    self._testBCastB([1, 3, 2], [2])

  def testBCast_1C(self):
    # [FUNCTION DOC]
    # Multiply of those shapes.
    self._testBCastC([1, 3, 2], [2])

  def testBCast_1D(self):
    # [FUNCTION DOC]
    # Division of those shapes.
    self._testBCastD([1, 3, 2], [2])

  def testBCast_2A(self):
    # [FUNCTION DOC]
    # Addition of two, three, two with three, two.
    self._testBCastA([2, 3, 2], [3, 2])

  def testBCast_2B(self):
    # [FUNCTION DOC]
    # Subtract and power of one, three, two with three, two.
    self._testBCastB([1, 3, 2], [3, 2])

  def testBCast_2C(self):
    # [FUNCTION DOC]
    # Multiply of those shapes.
    self._testBCastC([1, 3, 2], [3, 2])

  def testBCast_2D(self):
    # [FUNCTION DOC]
    # Division of those shapes.
    self._testBCastD([1, 3, 2], [3, 2])

  def testBCast_3A(self):
    # [FUNCTION DOC]
    # Addition of one, three, two with three, one.
    self._testBCastA([1, 3, 2], [3, 1])

  def testBCast_3B(self):
    # [FUNCTION DOC]
    # Subtract and power.
    self._testBCastB([1, 3, 2], [3, 1])

  def testBCast_3C(self):
    # [FUNCTION DOC]
    # Multiply.
    self._testBCastC([1, 3, 2], [3, 1])

  def testBCast_3D(self):
    # [FUNCTION DOC]
    # Division.
    self._testBCastD([1, 3, 2], [3, 1])

  def testBCast_4A(self):
    # [FUNCTION DOC]
    # Addition of identical shapes.
    self._testBCastA([1, 3, 2], [1, 3, 2])

  def testBCast_4B(self):
    # [FUNCTION DOC]
    # Subtract and power of identical shapes.
    self._testBCastB([1, 3, 2], [1, 3, 2])

  def testBCast_4C(self):
    # [FUNCTION DOC]
    # Multiply of identical shapes.
    self._testBCastC([1, 3, 2], [1, 3, 2])

  def testBCast_4D(self):
    # [FUNCTION DOC]
    # Division of identical shapes.
    self._testBCastD([1, 3, 2], [1, 3, 2])

  def testBCast_5A(self):
    # [FUNCTION DOC]
    # Addition of one, three, two with two, three, one.
    self._testBCastA([1, 3, 2], [2, 3, 1])

  def testBCast_5B(self):
    # [FUNCTION DOC]
    # Subtract and power.
    self._testBCastB([1, 3, 2], [2, 3, 1])

  def testBCast_5C(self):
    # [FUNCTION DOC]
    # Multiply.
    self._testBCastC([1, 3, 2], [2, 3, 1])

  def testBCast_5D(self):
    # [FUNCTION DOC]
    # Division.
    self._testBCastD([1, 3, 2], [2, 3, 1])


def run_benchmark(func, num_iters, execution_mode=None):
  # [FUNCTION DOC]
  # HELPER: simple wall-clock benchmarking utility (not referenced by any test in this file).
  # Runs the given callable once to warm up, then the requested number of times inside the given
  # execution mode of the eager context (synchronous or asynchronous), waiting for the asynchronous
  # executor before and after timing when the mode is asynchronous. Returns the elapsed time in
  # seconds as a number. The execution mode is optional and defaults to the null value.
  ctx = context.context()
  with context.execution_mode(execution_mode):
    # call func to warm up
    func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    start = time.time()
    for _ in xrange(num_iters):
      func()
    if execution_mode == context.ASYNC:
      ctx.executor.wait()
    end = time.time()

    return end - start

if __name__ == "__main__":
  test.main()
