# [MODULE DOC]
# KNOWN ISSUE: the three factory-built comparison tests are only registered inside the main-script
# guard, so they never run under a test runner that imports the module.
# Functional tests for the TensorFlow 2D/1D pooling operations: average pooling and max pooling
# (including the v2 max pool with tensor-valued window and stride, explicit padding, depthwise
# max pooling, max pooling with argmax, and the first- and second-order gradient ops).
# It defines one parameterized test case class (PoolingTest) plus module-level helpers that
# enumerate the (data format, device, data type) configurations to run, a generator of shrunk
# Inception max-pool shapes, and three factories that build CPU-vs-GPU comparison test methods.
# Tests verify numeric outputs against hand-computed expected values, check gradients with the
# numerical gradient checker, and check that invalid arguments raise the expected errors.
# Import-time side effects: imports the nn gradient registrations; the parameterized test
# configurations are computed at class-definition time (which queries GPU availability).
# The comparison tests built by the factories are attached to the test class only when the file
# is run as a script, not when it is collected by another runner.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for pooling operations."""

import collections
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
import tensorflow.python.framework.config as config_exec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def GetDeviceScope(self, use_gpu=False):
  # [FUNCTION DOC]
  # HELPER: picks the device or session context in which the pooling tests build and run ops.
  # Takes the test case instance and a flag asking for GPU (default false). When executing eagerly
  # it returns a device context for the first GPU if a GPU was requested and is available,
  # otherwise for the first CPU. In graph mode it returns a test session context created by the
  # test case with the same GPU flag. Returns a context manager; no other side effects.
  if context.executing_eagerly():
    if use_gpu and test.is_gpu_available():
      return ops.device("GPU:0")
    return ops.device("CPU:0")
  else:
    return self.session(use_gpu=use_gpu)


# TODO(jlebar): Convert the rest of this file to parameters.parameterized().
# Then remove GetTestConfigs() and rename GetTestConfigsDicts().
def GetTestConfigsDicts(v1_fn,
                        v2_fn=None,
                        one_dimensional=False,
                        allow_gpu=True):
  # [FUNCTION DOC]
  # HELPER: builds the list of parameter sets that drive the parameterized pooling tests.
  # Inputs: v1_fn is the pooling function to test; v2_fn is an optional second (v2-style) pooling
  # function; one_dimensional selects 1D layouts (width-channel formats) instead of 2D ones;
  # allow_gpu (default true) controls whether GPU entries are kept.
  # Behavior: starts from (data format, GPU flag) pairs; for 2D adds a vectorized-channel format
  # only when the function under test is a max pool. Each pair is expanded to float32 and, except
  # for the vectorized format (which always uses float32), float16, float64 and bfloat16.
  # Each expansion yields a mapping with the pool function, data format, data type, GPU flag and a
  # v2 flag. With v2_fn present, two more mappings are added: v2_fn with v2 false and v2_fn with
  # v2 true. If GPUs are not allowed, all GPU entries are dropped.
  # Returns the list of mappings (suitable as keyword arguments of the test methods); no side
  # effects and no exceptions.
  # (data_format, use_gpu) tuple
  if one_dimensional:
    configs0 = [
        ("NWC", False),
        ("NWC", True),
        ("NCW", True),
    ]
  else:
    configs0 = [
        ("NHWC", False),
        ("NHWC", True),
        ("NCHW", True),
    ]
    # NCHW_VECT_C only supported for max_pool.
    if (v1_fn == nn_ops.max_pool or v1_fn == nn_ops.max_pool1d or
        v2_fn == nn_ops.max_pool_v2 or v2_fn == gen_nn_ops.max_pool_v2):
      configs0.append(("NCHW_VECT_C", True))

  # (data_format, use_gpu, data_type) tuple
  configs1 = []
  for data_format, use_gpu in configs0:
    configs1.append((data_format, use_gpu, dtypes.float32))

    # In our test, VECT_C always uses float32.  (It gets converted to int8 in
    # the test runner.)
    if data_format == "NCHW_VECT_C":
      continue

    configs1 += [
        (data_format, use_gpu, dtypes.float16),
        (data_format, use_gpu, dtypes.float64),
        (data_format, use_gpu, dtypes.bfloat16),
    ]

  # Convert from tuple to dict and add v1/v2 versions.
  ret = []
  for data_format, use_gpu, data_type in configs1:
    ret.append({
        "pool_func": v1_fn,
        "data_format": data_format,
        "data_type": data_type,
        "use_gpu": use_gpu,
        "v2": False
    })
    if v2_fn:
      ret.append({
          "pool_func": v2_fn,
          "data_format": data_format,
          "data_type": data_type,
          "use_gpu": use_gpu,
          "v2": False
      })
      ret.append({
          "pool_func": v2_fn,
          "data_format": data_format,
          "data_type": data_type,
          "use_gpu": use_gpu,
          "v2": True
      })

  # Filter out GPU configs if necessary.
  if not allow_gpu:
    ret = [c for c in ret if not c["use_gpu"]]

  return ret


def GetTestConfigs(include_nchw_vect_c=False, one_dimensional=False):
  # [FUNCTION DOC]
  # HELPER: enumerates the (data format, use GPU) combinations that the older non-parameterized
  # tests loop over.
  # Inputs: include_nchw_vect_c (default false) asks for the vectorized-channel format when the
  # hardware supports it; one_dimensional (default false) selects 1D formats.
  # Behavior: always includes the channels-last format on CPU and GPU. Channels-first formats are
  # added only when a CUDA GPU is available (the 1D channels-first format) or, for 2D, the
  # channels-first format; the vectorized-channel format additionally needs compute capability
  # 6.1 or newer. When skipping, it logs an informational message.
  # Returns a list of (data format string, boolean) tuples. Queries GPU availability.
  """Get all the valid tests configs to run.

  Args:
    include_nchw_vect_c: Whether to include NCHW_VECT_C in the test configs.
    one_dimensional: If it's a 1D test

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  if one_dimensional:
    test_configs = [("NWC", False), ("NWC", True)]
    if test.is_gpu_available(cuda_only=True):
      test_configs += [("NCW", True)]
    return test_configs
  test_configs = [("NHWC", False), ("NHWC", True)]
  if not test.is_gpu_available(cuda_only=True):
    tf_logging.info("NCHW and NCHW_VECT_C tests skipped because not run with "
                    "--config=cuda or no GPUs available.")
    return test_configs
  # "NCHW" format is currently supported exclusively on CUDA GPUs.
  test_configs += [("NCHW", True)]
  if include_nchw_vect_c:
    if test.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(6, 1)):
      test_configs += [("NCHW_VECT_C", True)]
    else:
      tf_logging.info("NCHW_VECT_C test skipped because no GPUs with "
                      "compute capability >= 6.1 are available.")

  return test_configs


def GetShrunkInceptionMaxPoolShapes(shrink=30):
  # [FUNCTION DOC]
  # HELPER: generator of max pool test shapes taken from four max pool layers of the Inception
  # 2015 model, with the depth dimension reduced to keep the tests small.
  # Input: shrink (default 30) is the integer factor by which each input and output depth is
  # divided (integer division). Yields, for each of the four layers, a tuple of name, input
  # size, filter size, output size, strides and padding type. Only the lists local to the
  # call are modified; no external side effects.
  """Iterator for some of the max pool ops in the Inception 2015 model.

  Args:
    shrink: Factor to shrink depth relative to Inception.

  Yields:
    Tuple (name, input_size, filter_size, out_size, strides, padding)
  """
  names = ["maxpool2", "maxpool3", "maxpool4", "maxpool5"]
  input_sizes = [[32, 71, 71, 192], [32, 35, 35, 288], [32, 17, 17, 1248],
                 [32, 8, 8, 2048]]
  filter_sizes = [[1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1], [1, 3, 3, 1]]
  output_sizes = [[32, 35, 35, 192], [32, 17, 17, 288], [32, 8, 8, 1248],
                  [32, 8, 8, 2048]]
  strides = [[1, 2, 2, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]]
  # Shrink each depth value
  for i in input_sizes:
    i[3] //= shrink
  for o in output_sizes:
    o[3] //= shrink
  paddings = ["VALID", "VALID", "VALID", "SAME"]
  for n, i, f, o, s, p in zip(names, input_sizes, filter_sizes, output_sizes,
                              strides, paddings):
    yield n, i, f, o, s, p


@test_util.with_eager_op_as_function
class PoolingTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering average pooling, max pooling (v1 and v2 ops, explicit padding, depthwise
  # pooling, argmax variants) and their first- and second-order gradients, plus error handling.
  # Combines the TensorFlow test case base class with parameterized testing; the class is
  # wrapped so that eager ops run as functions. Holds no instance state. Shared fixtures are the
  # private verification helpers (_VerifyOneType, _VerifyOneTest, _VerifyValues,
  # _ConstructAndTestGradient, _ConstructAndTestSecondGradient, the _Compare* methods and the
  # direct gradient helpers). Additional comparison tests are attached to the class at run time
  # from the main-script section when the file is run directly.

  def _isMaxPool(self, func):
    # [FUNCTION DOC]
    # HELPER: tells whether a pooling function is one of the max pool entry points.
    # Input: func, a pooling function. Returns true if it is the v1 or v2 max pool function,
    # false otherwise. No side effects.
    return func in (nn_ops.max_pool, nn_ops.max_pool_v2)

  def _VerifyOneType(
      self,
      pool_func,
      input_sizes,
      ksize,
      strides,
      padding,
      data_format,
      data_type,
      expected,
      use_gpu,
      v2,
      use_negative_input=False,
      bfloat16_rtol=1e-2,
  ):
    # [FUNCTION DOC]
    # HELPER: core check shared by most pooling tests (one pooling run per configuration).
    # KNOWN ISSUE: the docstring says pool_func is the max or avg pool op "or the Lua version",
    # which is stale; any pooling function with the same keyword signature is accepted.
    # Runs one pooling op for one configuration and compares the flattened result with expected values.
    # Inputs: pool_func (pooling function), input_sizes (4D shape, or 3D for 1D pooling), ksize
    # and strides (window and stride lists), padding (string or explicit list), data_format,
    # data_type, expected (flat list of expected outputs), use_gpu, v2 (use tensor-valued window
    # and stride placeholders fed at run time), use_negative_input (negate the generated input,
    # default false), bfloat16_rtol (relative tolerance used for bfloat16, default 1e-2).
    # Behavior: skips the test (via the skip mechanism) when a GPU is requested but missing, for
    # float64 on ROCm, for the vectorized-channel format without a capable GPU, for v2 with
    # non-channels-last formats or explicit padding, and for the vectorized format with non-float32
    # type or depth not divisible by four. Otherwise builds an input of incrementing values that
    # wrap into the int8 range, converts layout (and quantizes for the vectorized format) as
    # needed, applies the pooling function, converts back, evaluates and compares with expected
    # using type-dependent tolerances. For non-v2 it also checks the static shape matches.
    # Returns nothing; raises assertion failures on mismatch.
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool, or the Lua
        version.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      data_format: The data format we use to run the pooling operation.
      data_type: The data type to use to run the pooling operation.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether we are running on GPU.
      v2: Whether to use v2 version.
      use_negative_input: If the input values should be negative.
      bfloat16_rtol: relative tolerance for bfloat16.
    """
    # Check that this test is compatible with the hardware we have.  (Really
    # this should be done in GetTestConfigsDicts(), but when that runs, we
    # haven't initialized enough of TF to know what our hardware is!)
    if use_gpu and not test.is_gpu_available():
      self.skipTest("No GPU is available.")
    if use_gpu and data_type == dtypes.float64 and test.is_built_with_rocm():
      self.skipTest("ROCm pooling ops don't support float64.")
    if use_gpu and data_format == "NCHW_VECT_C" and not test.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(6, 1)):
      self.skipTest("NCHW_VECT_C requires sm61+.")

    if v2 and data_format != "NHWC":
      self.skipTest("v2 not supported for %s" % data_format)
    if v2 and not isinstance(padding, str):
      self.skipTest("non-constant ksize/strides requires nonexplicit padding")
    if data_format == "NCHW_VECT_C":
      if data_type != dtypes.float32:
        self.skipTest("quantization to qint8 not implemented for %r" %
                      data_type)
      if input_sizes[-1] % 4 != 0:
        self.skipTest("Skipping test for depth %d" % input_sizes[-1])

    total_size = 1
    for s in input_sizes:
      total_size *= s
    tf_logging.info("Running %s test. %r %r %d %r %r %r %s", data_format, v2,
                    input_sizes, total_size, pool_func, ksize, strides,
                    data_type)
    # Initializes the input tensor with array containing incrementing
    # numbers from 1, wrapping round to -127 after 127 to support int8.
    y = -1 if use_negative_input else 1
    x = [(((f + 128) % 255) - 127)*y for f in range(total_size)]
    with self.cached_session(use_gpu=use_gpu):
      t = constant_op.constant(x, shape=input_sizes, dtype=data_type)
      if data_format in ("NCHW", "NCHW_VECT_C", "NCW"):
        if data_format == "NCHW_VECT_C":
          t = test_util.NHWCToNCHW_VECT_C(t)
          t, _, _ = gen_array_ops.quantize_v2(t, -128.0, 127.0, dtypes.qint8)
        else:
          t = test_util.NHWCToNCHW(t)
        ksize = test_util.NHWCToNCHW(ksize)
        strides = test_util.NHWCToNCHW(strides)
        if isinstance(padding, list):
          padding = test_util.NHWCToNCHW(padding)
      ksize_placeholder = array_ops.placeholder(dtypes.int32, shape=[4])
      strides_placeholder = array_ops.placeholder(dtypes.int32, shape=[4])
      if v2:
        t = pool_func(
            t,
            ksize=ksize_placeholder,
            strides=strides_placeholder,
            padding=padding,
            data_format=data_format)
      else:
        t = pool_func(
            t,
            ksize=ksize,
            strides=strides,
            padding=padding,
            data_format=data_format)
      if data_format == "NCHW_VECT_C":
        t = gen_array_ops.dequantize(t, -128, 127)
        t = test_util.NCHW_VECT_CToNHWC(t)
      elif data_format == "NCHW":
        t = test_util.NCHWToNHWC(t)
      if v2:
        actual = t.eval(feed_dict={
            ksize_placeholder: ksize,
            strides_placeholder: strides
        })
      else:
        actual = self.evaluate(t)
        self.assertShapeEqual(actual, t)
      self.assertAllCloseAccordingToType(
          expected, actual.flatten(), bfloat16_rtol=bfloat16_rtol
      )

  def _VerifyOneTest(self, pool_func, input_sizes, ksize, strides, padding,
                     data_format, expected, use_gpu, v2,
                     use_negative_input=False):
    # [FUNCTION DOC]
    # HELPER: runs the value check for one data format across several data types.
    # KNOWN ISSUE: the docstring does not say that three data types are exercised, and its
    # use_negative_input line ends with a stray quote character.
    # Inputs: same as the per-type helper except that the data type is chosen internally:
    # pooling function, input sizes, window sizes, strides, padding, data format, expected
    # outputs, GPU flag, v2 flag and optional negative-input flag (default false).
    # Behavior: for the vectorized-channel format it returns silently (logging a message) when
    # the function is average pool, or max pool with explicit padding, since those are not
    # implemented. Otherwise it runs the per-type check for float32, then float64 (skipped on
    # ROCm), then float16 (only on CPU or on a GPU that supports half precision). Returns nothing.
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      data_format: The data format we use to run the pooling operation.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether we are running on GPU.
      v2: Whether to use v2 version.
      use_negative_input: If the input values should be negative."
    """
    if data_format == "NCHW_VECT_C":
      avg_pool_func = nn_ops.avg_pool
      tf_logging.info("pool_func=%s", pool_func)
      if pool_func == avg_pool_func:
        tf_logging.info("NCHW_VECT_C not yet implemented for avg_pool")
        return
      if (self._isMaxPool(pool_func) and isinstance(padding, list)):
        tf_logging.info("NCHW_VECT_C not yet implemented for max pool" +
                        " with explicit padding")
        return

    self._VerifyOneType(pool_func, input_sizes, ksize, strides, padding,
                        data_format, dtypes.float32, expected, use_gpu, v2,
                        use_negative_input)
    if not test.is_built_with_rocm():
      # double datatype is not supported for pooling ops on the ROCm platform
      self._VerifyOneType(pool_func, input_sizes, ksize, strides, padding,
                          data_format, dtypes.float64, expected, use_gpu, v2,
                          use_negative_input)

    if not use_gpu or test_util.GpuSupportsHalfMatMulAndConv():
      self._VerifyOneType(pool_func, input_sizes, ksize, strides, padding,
                          data_format, dtypes.float16, expected, use_gpu, v2,
                          use_negative_input)

  def _VerifyValues(self,
                    pool_func,
                    input_sizes,
                    ksize,
                    strides,
                    padding,
                    expected,
                    use_gpu,
                    v2=False,
                    one_dim=False,
                    use_negative_input=False):
    # [FUNCTION DOC]
    # HELPER: runs the value check over every applicable (data format, device) configuration.
    # Inputs: pool_func, input_sizes, ksize, strides, padding, expected outputs, use_gpu (only
    # configurations whose device matches this flag are run), v2 (default false), one_dim (use 1D
    # configurations, default false) and use_negative_input (default false).
    # Behavior: loops over the configurations from the test-config helper, including the
    # vectorized-channel format, and for each matching device calls the multi-type check.
    # Returns nothing; failures surface as assertion errors or skips from the callee.
    """Verifies the output values of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      ksize: The kernel size dimensions
      strides: The stride dimensions
      padding: Padding type.
      expected: An array containing the expected operation outputs.
      use_gpu: Whether we are running on GPU.
      v2: Whether to use v2 version.
      one_dim: If one dimensional pools should be done instead of two
        dimensional pools.
      use_negative_input: If the input values should be negative.
    """
    for (data_format, use_gpu_2) in GetTestConfigs(
        include_nchw_vect_c=True, one_dimensional=one_dim):
      if use_gpu_2 == use_gpu:
        self._VerifyOneTest(pool_func, input_sizes, ksize, strides, padding,
                            data_format, expected, use_gpu, v2,
                            use_negative_input)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolValidPadding(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (one run per data format, device and data type configuration passed as
    # keyword arguments). Average-pools a 1x3x3x3 input with a 2x2 window, stride 2 and valid
    # padding; expects the three depth values 7, 8, 9 as the only output window.
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        expected=[7.0, 8.0, 9.0],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolEmpty(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Average pooling over an
    # input whose depth dimension is zero (1x3x3x0) with a 2x2 window, stride 2, valid padding;
    # expects an empty output without errors.
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 0],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        expected=[],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePadding(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Average pooling of a
    # 1x2x4x3 input with a 2x2 window, stride 2 and same padding; expects six averaged values
    # (two window positions across three channels).
    self._VerifyOneType(
        input_sizes=[1, 2, 4, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[8.5, 9.5, 10.5, 14.5, 15.5, 16.5],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingNonSquareWindow(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Average pooling of a 2x2
    # single-channel image with a 1x2 (one row, two columns) window, stride 1 and same padding.
    # Expects 1.5, 2.0, 3.5, 4.0: the window at the right edge averages only the in-image value
    # (padding is not counted in the divisor).
    # input is:
    # [1.0, 2.0
    #  3.0  4.0]
    #
    # Window of [x, x] should do:
    #  [avg(1.0, 2.0), avg(2.0, padded0),
    #   avg(3.0, 4.0), avg(4.0, padded0)]
    self._VerifyOneType(
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[1.5, 2.0, 3.5, 4.0],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingNonSquareWindow_2(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Same as the previous
    # non-square case but with a 2x1 (two rows, one column) window: average pooling of a 2x2
    # image with stride 1 and same padding; expects 2.0, 3.0, 3.0, 4.0 (bottom row windows only
    # average the real pixel).
    # Window of [x,
    #            x] should do:
    #  [avg(1.0, 3.0), avg(2.0, 4.0)
    #   avg(3.0, padded0), avg(4.0, padded0)]
    self._VerifyOneType(
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 2, 1, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[2.0, 3.0, 3.0, 4.0],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingNonSquareWindowMultiBatch(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Average pooling of a
    # 2x2x2x2 input (two batches, two channels) with a 1x2 window, stride 1, same padding;
    # expects sixteen values checking that batches and channels are pooled independently.
    self._VerifyOneType(
        input_sizes=[2, 2, 2, 2],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[
            2.0, 3.0, 3.0, 4.0, 6.0, 7.0, 7.0, 8.0, 10.0, 11.0, 11.0, 12.0,
            14.0, 15.0, 15.0, 16.0
        ],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingNonSquareWindowMultiBatch_2(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Multi-batch,
    # multi-channel 2x2x2x2 input average-pooled with a 2x1 window, stride 1, same padding;
    # expects sixteen values (the column-direction counterpart of the previous test).
    self._VerifyOneType(
        input_sizes=[2, 2, 2, 2],
        ksize=[1, 2, 1, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[
            3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 11.0, 12.0, 13.0, 14.0,
            13.0, 14.0, 15.0, 16.0
        ],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolValidPaddingUnevenStride(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Average pooling of a
    # 1x3x3x3 input, 2x2 window, strides 1 along rows and 2 along columns, valid padding;
    # expects two output positions of three channels each (six values).
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 1, 2, 1],
        padding="VALID",
        expected=[7.0, 8.0, 9.0, 16.0, 17.0, 18.0],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolValidPaddingUnevenStride_2(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Same as the previous test
    # with the strides swapped (2 along rows, 1 along columns); expects six values.
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 1, 1],
        padding="VALID",
        expected=[7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePadding_2(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Average pooling of a
    # 1x4x4x4 input with a 2x2 window, stride 2, same padding; expects sixteen averaged values
    # (2x2 positions times four channels).
    expected_output = [
        11.0, 12.0, 13.0, 14.0, 19.0, 20.0, 21.0, 22.0, 43.0, 44.0, 45.0, 46.0,
        51.0, 52.0, 53.0, 54.0
    ]
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingPacket_4(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Average pooling of a
    # 1x4x4x4 input (depth four, a vector-friendly packet size) with a 3x3 window, stride 2 and
    # same padding; expects sixteen values, with edge windows averaging fewer elements.
    expected_output = [
        21.0, 22.0, 23.0, 24.0, 27.0, 28.0, 29.0, 30.0, 45.0, 46.0, 47.0, 48.0,
        51.0, 52.0, 53.0, 54.0
    ]
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolSamePaddingPacket_8(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Average pooling of a
    # 1x8x8x8 input (depth eight, a wider vector packet size) with a 3x3 window, stride 2, same
    # padding, comparing against a long list of precomputed values. Uses a looser relative
    # tolerance (3e-2) for bfloat16.
    expected_output = [
        -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, 4.0, 5.0, 6.0, 7.0,
        8.0, 9.0, 10.0, 11.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0,
        32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, -3.5, -54.0, -53.0, -52.0,
        -51.0, -50.0, -49.0, -48.0, -47.0, -38.0, -37.0, -36.0, -35.0, -34.0,
        -33.0, -32.0, -31.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0,
        -15.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -11.0, -10.0,
        -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
        12.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 33.0, 34.0, 35.0,
        36.0, 37.0, 38.0, -3.5, -2.5, -85.0, -84.0, -83.0, -82.0, -81.0, -80.0,
        -79.0, -78.0, -69.0, -68.0, -67.0, -66.0, -65.0, -64.0, -63.0, -62.0,
        -53.0, -52.0, -51.0, -50.0, -49.0, -48.0, -47.0, -46.0, -41.0, -40.0,
        -39.0, -38.0, -37.0, -36.0, -35.0, -34.0
    ]
    self._VerifyOneType(
        input_sizes=[1, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        bfloat16_rtol=3e-2,
        **kwargs,
    )

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolEmptyInput(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (configuration passed as keyword arguments). Average pooling of an
    # input with zero batch size (0x8x8x8), 3x3 window, stride 2, same padding; expects an empty
    # result without errors.
    self._VerifyOneType(
        input_sizes=[0, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[],
        **kwargs)

  @test_util.run_in_graph_and_eager_modes
  def testRawAvgPoolLargeKsizeRaiseError(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Calls the raw average pool op on a 1x1x1x1 input with
    # an enormous window height (the largest signed 64-bit value) and expects either a value
    # error or an invalid-argument error rather than a crash or overflow.
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      with self.cached_session():
        t = gen_nn_ops.avg_pool(
            value=np.ones([1, 1, 1, 1]),
            ksize=[1, 9223372036854775807, 1, 1],
            strides=[1, 1, 1, 1],
            padding="SAME",
            data_format="NHWC")
        self.evaluate(t)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolValidPadding(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test over v1 and v2 max pool configurations (passed as keyword arguments).
    # Max-pools a 1x3x3x3 input with a 2x2 window, stride 2, valid padding; expects the three
    # channel maxima 13, 14, 15.
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="VALID",
        expected=[13.0, 14.0, 15.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolSamePadding(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test over v1 and v2 max pool configurations. Max pooling of a 1x2x3x3 input
    # with a 2x2 window, stride 2, same padding; expects six values (two window positions by
    # three channels).
    self._VerifyOneType(
        input_sizes=[1, 2, 3, 3],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolZeroExplicitPadding(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test (XLA may fall back since it lacks explicit padding). Max pooling of a
    # 3x3 single-channel image with a 3x3 window, stride 2 and an explicit padding list of all
    # zeros; expects the single maximum 9.
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
        expected=[9.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolNegativeInputExpPadding(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test with explicit padding and negative input values. Pads a 3x3 image by 2
    # before and 1 after in both spatial dimensions, 3x3 window, stride 2; expects four outputs
    # of -1, showing that padding cells never win the maximum when all real values are negative.
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [2, 1], [2, 1], [0, 0]],
        expected=[-1, -1, -1, -1],
        use_negative_input=True,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolExplicitPadding(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test with explicit padding. 3x3 image, 3x3 window, stride 2, padding of 2 at
    # the bottom and 1 at the right; expects two outputs, both 9.
    expected_output = [9.0, 9.0]
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [0, 2], [0, 1], [0, 0]],
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolExplicitPaddingAdvanced(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test with asymmetric explicit padding on a 6x6 single-channel image, 3x3
    # window, stride 2; padding is 1 top, 2 bottom, 2 left, 1 right. Expects sixteen precomputed
    # window maxima.
    expected_output = [7, 9, 11, 12, 19, 21, 23, 24, 31, 33, 35, 36, 31, 33,
                       35, 36]
    self._VerifyOneType(
        input_sizes=[1, 6, 6, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [1, 2], [2, 1], [0, 0]],
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolNegativeInputExpPaddingAdv(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized test: the asymmetric explicit padding scenario of the advanced test but with
    # the input negated (negative-input flag set), so the expected maxima are the negated
    # smallest-magnitude values and padding must not leak in as zeros. Sixteen expected values.
    expected_output = [-1, -1, -3, -5, -7, -7, -9, -11, -19, -19, -21, -23, -31,
                       -31, -33, -35]

    self._VerifyOneType(
        input_sizes=[1, 6, 6, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [1, 2], [2, 1], [0, 0]],
        expected=expected_output,
        use_negative_input=True,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, nn_ops.max_pool_v2))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolExplicitPadding2_(self, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the body is identical to the test of explicit padding without the suffix
    # (same input, window, strides, padding and expected output), so it adds no coverage.
    # Parameterized max pool test with explicit padding: 3x3 image, 3x3 window, stride 2,
    # bottom padding 2 and right padding 1; expects two outputs, both 9.
    expected_output = [9.0, 9.0]
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding=[[0, 0], [0, 2], [0, 1], [0, 0]],
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(
          nn_ops.max_pool1d, nn_ops.max_pool_v2, one_dimensional=True))
  @test_util.xla_allow_fallback("XLA doesn't support explicit padding")
  @test_util.run_deprecated_v1
  def testMaxPoolExplicitPadding_1D(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized one-dimensional max pool test using 1D configurations (width-channel
    # formats). Input shape 1x3x1, window 2, stride 2, padding of 1 at the end; expects 2 and 3.
    self._VerifyOneType(
        input_sizes=[1, 3, 1],
        ksize=[1, 2, 1],
        strides=[1, 2, 1],
        padding=[[0, 0], [0, 1], [0, 0]],
        expected=[2.0, 3.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolSamePaddingNonSquareWindow(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized max pool test: 2x2 single-channel image, 1x2 window, stride 1, same padding;
    # expects 2, 2, 4, 4 (right-edge windows see only the real pixel).
    # input is:
    # [1.0, 2.0
    #  3.0  4.0]
    #
    # Window of [x, x] should do:
    #
    #  [max(1.0, 2.0), max(2.0, padded0),
    #   max(3.0, 4.0), max(4.0, padded0)]
    self._VerifyOneType(
        input_sizes=[1, 2, 2, 1],
        ksize=[1, 1, 2, 1],
        strides=[1, 1, 1, 1],
        padding="SAME",
        expected=[2.0, 2.0, 4.0, 4.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolValidPaddingUnevenStride(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized max pool test: 4x4 single-channel image, 2x2 window, stride 1 along rows and
    # 2 along columns, valid padding; expects six values.
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 1, 2, 1],
        padding="VALID",
        expected=[6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolValidPaddingUnevenStride2_(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized max pool test with the strides swapped relative to the previous test (2 along
    # rows, 1 along columns) on a 4x4 image, 2x2 window, valid padding; expects six values.
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 1, 1],
        padding="VALID",
        expected=[6.0, 7.0, 8.0, 14.0, 15.0, 16.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolSamePaddingPacket4_(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized max pool test on a 1x4x4x4 input (depth four) with a 2x2 window, stride 2,
    # same padding; expects sixteen window maxima.
    expected_output = [
        21.0, 22.0, 23.0, 24.0, 29.0, 30.0, 31.0, 32.0, 53.0, 54.0, 55.0, 56.0,
        61.0, 62.0, 63.0, 64.0
    ]
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 4],
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolSamePaddingPacket8_(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized max pool test on a 1x8x8x8 input (depth eight) with a 3x3 window, stride 2,
    # same padding, compared against a long list of precomputed maxima. The input values wrap
    # around the signed 8-bit range, which explains the negative and wrapped entries.
    expected_output = [
        81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 97.0, 98.0, 99.0, 100.0,
        101.0, 102.0, 103.0, 104.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0,
        119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 120.0,
        18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 34.0, 35.0, 36.0, 37.0,
        38.0, 39.0, 40.0, 41.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0,
        58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 82.0, 83.0, 84.0, 85.0,
        86.0, 87.0, 88.0, 89.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0,
        105.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0,
        123.0, 124.0, 125.0, 126.0, 127.0, 120.0, 121.0, -45.0, -44.0, -43.0,
        -42.0, -41.0, -40.0, -39.0, -38.0, -29.0, -28.0, -27.0, -26.0, -25.0,
        -24.0, -23.0, -22.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0,
        -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0
    ]
    self._VerifyOneType(
        input_sizes=[1, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=expected_output,
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolEmptyInput(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized max pool test with an empty batch (0x8x8x8 input), 3x3 window, stride 2, same
    # padding; expects an empty result and no error.
    self._VerifyOneType(
        input_sizes=[0, 8, 8, 8],
        ksize=[1, 3, 3, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2))
  @test_util.run_deprecated_v1
  def testMaxPoolInvalidFilterSize(self, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: it is parameterized over many configurations but ignores all of them; the
    # body never uses the keyword arguments, so every parameter set runs the same check.
    # Max-pools a 1x1x1x1 tensor with a window of width 2 and valid padding and expects an
    # invalid-argument or value error mentioning a negative dimension size.
    with self.cached_session(use_gpu=test.is_gpu_available()):
      t = constant_op.constant(1.0, shape=[1, 1, 1, 1])
      with self.assertRaisesRegex(
          (errors_impl.InvalidArgumentError, ValueError),
          "Negative dimension size"):
        t = self.evaluate(
            nn_ops.max_pool(t, ksize=[1, 1, 2, 1], strides=1, padding="VALID"))

  @test_util.run_in_graph_and_eager_modes
  def testMaxPoolWithArgmaxKsizeOverflow(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Calls the raw max-pool-with-argmax op with a window height
    # of minus two to the power of 31 and a huge stride; expects a value error or invalid-argument
    # error saying the window size must be a positive 32-bit integer.
    with self.assertRaisesRegex(
        (ValueError, errors_impl.InvalidArgumentError),
        "ksize must be a positive int32 value",
    ):
      with self.cached_session():
        t = gen_nn_ops.max_pool_with_argmax(
            input=[[[[1, 1, 1]]]],
            ksize=[1, -2**31, 4, 1],
            strides=[1, 1000, 1, 7],
            padding="SAME")
        self.evaluate(t)

  # Tests for DepthwiseMaxPooling on CPU only.
  @parameterized.parameters(
      GetTestConfigsDicts(
          nn_ops.max_pool, gen_nn_ops.max_pool_v2, allow_gpu=False))
  @test_util.run_deprecated_v1
  def testDepthwiseMaxPool1x1DepthWindow(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized CPU-only depthwise max pool test. Input is a single pixel with ten channels
    # of 1 to 10; pools along depth with window and stride 2; expects 2, 4, 6, 8, 10.
    # input is:
    # [1.0, ..., 10.0] along depth,
    #
    # We maxpool by depth in patches of 2.
    self._VerifyOneType(
        input_sizes=[1, 1, 1, 10],
        ksize=[1, 1, 1, 2],
        strides=[1, 1, 1, 2],
        padding="SAME",
        expected=[2.0, 4.0, 6.0, 8.0, 10.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(
          nn_ops.max_pool, gen_nn_ops.max_pool_v2, allow_gpu=False))
  @test_util.run_deprecated_v1
  def testDepthwiseMaxPool2x2DepthWindow(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized CPU-only depthwise max pool test. A 2x2 spatial input with six channels is
    # pooled along depth with window and stride 3, giving a 2x2x2 output whose values are
    # multiples of 3.
    # input is:
    #
    # a 2x2x6 cube, and we depthwise max across 3 to produce a 2x2x2
    # output.  Each node has contiguous values, so the depthwise max
    # should be multiples of 3.0.
    self._VerifyOneType(
        input_sizes=[1, 2, 2, 6],
        ksize=[1, 1, 1, 3],
        strides=[1, 1, 1, 3],
        padding="SAME",
        expected=[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(
          nn_ops.max_pool, gen_nn_ops.max_pool_v2, allow_gpu=False))
  @test_util.run_deprecated_v1
  def testMaxPoolKernelSmallerThanStrideValid(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized CPU-only max pool test where the window (2x2) is smaller than the stride (3)
    # on a 7x7 single-channel image with valid padding; expects four maxima (9, 12, 30, 33).
    self._VerifyOneType(
        input_sizes=[1, 7, 7, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 3, 3, 1],
        padding="VALID",
        expected=[9, 12, 30, 33],
        **kwargs)

  @parameterized.parameters(GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testAvgPoolKernelSmallerThanStride(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized average pool test where the 2x2 window is smaller than the stride of 3 on a
    # 7x7 single-channel image with valid padding; expects four averages (5, 8, 26, 29).
    self._VerifyOneType(
        input_sizes=[1, 7, 7, 1],
        ksize=[1, 2, 2, 1],
        strides=[1, 3, 3, 1],
        padding="VALID",
        expected=[5, 8, 26, 29],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2) +
      GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testKernelSmallerThanStrideSame1_(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized over both max pool and average pool configurations. A 1x1 window with stride
    # 2 and same padding on a 3x3 single-channel image samples every other pixel; expects 1, 3,
    # 7, 9.
    self._VerifyOneType(
        input_sizes=[1, 3, 3, 1],
        ksize=[1, 1, 1, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[1, 3, 7, 9],
        **kwargs)

  @parameterized.parameters(
      GetTestConfigsDicts(nn_ops.max_pool, gen_nn_ops.max_pool_v2) +
      GetTestConfigsDicts(nn_ops.avg_pool))
  @test_util.run_deprecated_v1
  def testKernelSmallerThanStrideSame2_(self, **kwargs):
    # [FUNCTION DOC]
    # Parameterized over both max pool and average pool configurations. A 1x1 window with stride
    # 2 and same padding on a 4x4 single-channel image; expects 1, 3, 9, 11.
    self._VerifyOneType(
        input_sizes=[1, 4, 4, 1],
        ksize=[1, 1, 1, 1],
        strides=[1, 2, 2, 1],
        padding="SAME",
        expected=[1, 3, 9, 11],
        **kwargs)

  def _testDepthwiseMaxPoolInvalidConfig(self,
                                         in_size,
                                         ksize,
                                         strides,
                                         error_msg,
                                         use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: for testDepthwiseMaxPoolInvalidConfigs; asserts one depthwise max pool config is rejected.
    # Inputs: in_size (shape of an all-ones input), ksize and strides (window and stride lists),
    # error_msg (regular expression the error text must match), use_gpu (default false).
    # Builds the input, applies same-padded max pooling and evaluates it, expecting an
    # unimplemented-operation error whose message matches error_msg. Returns nothing.
    with self.cached_session(use_gpu=use_gpu):
      t = constant_op.constant(1.0, shape=in_size)
      with self.assertRaisesRegex(errors_impl.UnimplementedError, error_msg):
        t = nn_ops.max_pool(
            t, ksize=ksize, strides=strides, padding="SAME").eval()

  @test_util.disable_xla("b/123338077")  # Passes with XLA
  def testDepthwiseMaxPoolInvalidConfigs(self):
    # [FUNCTION DOC]
    # Checks that unsupported depthwise max pool configurations are rejected with an
    # unimplemented error: pooling over depth and space at once, depth window different from
    # depth stride, and a depth window that does not evenly divide the depth. When a GPU is
    # present it also checks that depthwise pooling reports it is only supported on CPU devices.
    # XLA is disabled for this test because it behaves differently.
    self._testDepthwiseMaxPoolInvalidConfig(
        [1, 2, 2, 4], [1, 2, 2, 2], [1, 1, 1, 2],
        "exactly one of pooling across depth")
    self._testDepthwiseMaxPoolInvalidConfig(
        [1, 2, 2, 4], [1, 1, 1, 2], [1, 1, 1, 1],
        "depth window to equal the depth stride")
    self._testDepthwiseMaxPoolInvalidConfig([1, 2, 2, 4], [1, 1, 1, 3],
                                            [1, 1, 1, 3], "evenly divide")
    if test.is_gpu_available():
      with self.session():
        t = variables.Variable(np.ones([1, 2, 2, 4]))
        self.evaluate(variables.global_variables_initializer())
        with self.assertRaisesOpError("for CPU devices"):
          nn_ops.max_pool(
              t, ksize=[1, 1, 1, 2], strides=[1, 1, 1, 2],
              padding="SAME").eval()

  # The following are tests that verify that the CPU and GPU implementations
  # produce the same results.
  def _CompareMaxPoolingFwd(self, input_shape, ksize, strides, padding):
    # [FUNCTION DOC]
    # HELPER: used by the factory-built forward comparison tests.
    # KNOWN ISSUE: operator precedence makes the loop list be the float32/float16 list plus float64
    # when not on ROCm, but an empty list on ROCm, so on ROCm no data type is compared at all
    # (the intent was only to drop float64). Also the variable named gpu_val is computed with
    # the argmax op using the default session, not necessarily on the GPU.
    # Inputs: input_shape, ksize, strides, padding. For each data type, builds a random input,
    # runs max-pool-with-argmax (first output only) and the plain max pool on CPU, and asserts
    # the two results agree within type-dependent tolerance. Returns nothing.
    # double datatype is currently not supported for pooling ops
    # on the ROCm platform
    for dtype in [np.float32, np.float16
                 ] + [np.float64] if not test.is_built_with_rocm() else []:
      tensor_input = np.random.rand(*input_shape).astype(dtype)
      with self.cached_session():
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op, _ = nn_ops.max_pool_with_argmax(t, ksize, strides, padding)
        gpu_val = self.evaluate(out_op)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op = nn_ops.max_pool(t, ksize, strides, padding)
        cpu_val = self.evaluate(out_op)
      self.assertAllCloseAccordingToType(cpu_val, gpu_val)

  def _CompareMaxPoolingBk(self, input_shape, output_shape, ksize, strides,
                           padding):
    # [FUNCTION DOC]
    # HELPER: used by the factory-built gradient comparison tests.
    # KNOWN ISSUE: the data-type list has the same precedence bug as the forward comparison
    # helper: on ROCm the list is empty so nothing is compared. Also uses a deprecated random
    # integer generator from the numerical library.
    # Inputs: input_shape, output_shape, ksize, strides, padding. For each data type, builds a
    # random input with few distinct values (many ties) and a random output gradient; computes
    # the argmax-based max pool gradient and the standard max pool gradient on CPU, and
    # asserts they match within loosened half/bfloat16 tolerances. Returns nothing.
    # double datatype is currently not supported for pooling ops
    # on the ROCm platform
    for dtype in [np.float32, np.float16, dtypes.bfloat16.as_numpy_dtype
                 ] + [np.float64] if not test.is_built_with_rocm() else []:
      # Generate numbers in a narrow range, so that there are many duplicates
      # in the input.
      tensor_input = np.random.random_integers(0, 3, input_shape).astype(dtype)
      tensor_output = np.random.rand(*output_shape).astype(dtype)
      with self.cached_session():
        t = constant_op.constant(tensor_input, shape=input_shape)
        _, argmax_op = nn_ops.max_pool_with_argmax(t, ksize, strides, padding)
        argmax = self.evaluate(argmax_op)
        grad_in = constant_op.constant(tensor_output, shape=output_shape)
        out_op = gen_nn_ops.max_pool_grad_with_argmax(t, grad_in, argmax, ksize,
                                                      strides, padding)
        gpu_val = self.evaluate(out_op)
        self.assertShapeEqual(gpu_val, out_op)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op = nn_ops.max_pool(t, ksize, strides, padding)
        orig_out = self.evaluate(out_op)
        grad_in = constant_op.constant(tensor_output, shape=output_shape)
        out_op = gen_nn_ops.max_pool_grad(t, orig_out, grad_in, ksize, strides,
                                          padding)
        cpu_val = self.evaluate(out_op)
        self.assertShapeEqual(cpu_val, out_op)
      # The CPU version accumulates its gradient on fp16, so it's less
      # accurate than the GPU version that does the accumulation on fp32
      self.assertAllCloseAccordingToType(
          cpu_val,
          gpu_val,
          half_rtol=0.01,
          half_atol=0.01,
          bfloat16_rtol=0.02,
          bfloat16_atol=0.1)

  def _CompareMaxPoolingGradBk(self, input_shape, output_shape, ksize, strides,
                               padding):
    # [FUNCTION DOC]
    # HELPER: used by the factory-built second-order gradient comparison tests.
    # KNOWN ISSUE: same precedence bug in the data-type list (empty on ROCm); both halves of the
    # comparison run under a CPU-only session although one is labelled as the GPU value; and it
    # uses a deprecated random integer generator.
    # Inputs: input_shape, output_shape (unused except by callers), ksize, strides, padding. For
    # each data type, builds a tie-heavy random input, computes the argmax-based grad-of-grad and
    # the standard max pool grad-of-grad, and asserts they agree within half-precision tolerances.
    # Returns nothing.
    # double datatype is currently not supported for pooling ops
    # on the ROCm platform
    for dtype in [np.float32, np.float16, dtypes.bfloat16.as_numpy_dtype
                 ] + [np.float64] if not test.is_built_with_rocm() else []:
      # Generate numbers in a narrow range, so that there are many duplicates
      # in the input.
      tensor_input = np.random.random_integers(0, 3, input_shape).astype(dtype)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        _, argmax_op = nn_ops.max_pool_with_argmax(t, ksize, strides, padding)
        argmax = self.evaluate(argmax_op)
        grad_in = constant_op.constant(tensor_input, shape=input_shape)
        out_op = gen_nn_ops.max_pool_grad_grad_with_argmax(
            t, grad_in, argmax, ksize, strides, padding)
        gpu_val = self.evaluate(out_op)
        self.assertShapeEqual(gpu_val, out_op)
      with self.cached_session(use_gpu=False):
        t = constant_op.constant(tensor_input, shape=input_shape)
        out_op = nn_ops.max_pool(t, ksize, strides, padding)
        orig_out = self.evaluate(out_op)
        grad_in = constant_op.constant(tensor_input, shape=input_shape)
        out_op = gen_nn_ops.max_pool_grad_grad(t, orig_out, grad_in, ksize,
                                               strides, padding)
        cpu_val = self.evaluate(out_op)
        self.assertShapeEqual(cpu_val, out_op)
      # The CPU version accumulates its gradient on fp16, so it's less
      # accurate than the GPU version that does the accumulation on fp32
      self.assertAllCloseAccordingToType(
          cpu_val, gpu_val, half_rtol=0.01, half_atol=0.01)

  def testMaxPoolingWithArgmax(self):
    # [FUNCTION DOC]
    # Runs max-pool-with-argmax on a two-batch 3x3 input, 2x2 window, stride 1, valid padding for
    # several configurations of device, whether argmax indices include the batch offset, and the
    # argmax integer type. Asserts the output shapes, that all pooled values are 1, and that the
    # argmax indices equal the expected flat positions (with batch offsets when requested).
    tensor_input = [
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0
    ]

    Config = collections.namedtuple(
        "Config", ["use_gpu", "include_batch_in_index", "argmax", "Targmax"])
    configs = [
        Config(False, False, [0, 1, 3, 5, 0, 2, 6, 8], dtypes.int64),
        Config(False, True, [0, 1, 3, 5, 9, 11, 15, 17], dtypes.int64),
        Config(False, False, [0, 1, 3, 5, 0, 2, 6, 8], dtypes.int32),
        Config(False, True, [0, 1, 3, 5, 9, 11, 15, 17], dtypes.int32),
        Config(True, False, [0, 1, 3, 5, 0, 2, 6, 8], dtypes.int64),
        Config(True, True, [0, 1, 3, 5, 9, 11, 15, 17], dtypes.int64),
    ]

    for config in configs:
      with GetDeviceScope(self, use_gpu=config.use_gpu):
        t = constant_op.constant(tensor_input, shape=[2, 3, 3, 1])
        out_op, argmax_op = nn_ops.max_pool_with_argmax(
            t,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            Targmax=config.Targmax,
            padding="VALID",
            include_batch_in_index=config.include_batch_in_index)
        out, argmax = self.evaluate([out_op, argmax_op])
        self.assertShapeEqual(out, out_op)
        self.assertShapeEqual(argmax, argmax_op)
        self.assertAllClose(out.ravel(),
                            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertAllEqual(argmax.ravel(), config.argmax)

  def testDepthwiseMaxPoolingWithArgmax(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: shape assertions are commented out pending a bug (b/259733542), so only values
    # are checked.
    # Max-pool-with-argmax along the depth dimension (window and stride 3) of a one-pixel,
    # three-channel input, on CPU and GPU scope with same and valid padding. Asserts the output
    # is the original three values and argmax is 0, 1, 2.
    tensor_input = [89, 73, -109]
    Config = collections.namedtuple("Config", ["use_gpu", "padding"])
    configs = [
        Config(False, "SAME"),
        Config(False, "VALID"),
        Config(True, "SAME"),
        Config(True, "VALID"),
    ]

    for config in configs:
      with GetDeviceScope(self, use_gpu=config.use_gpu):
        t = constant_op.constant(tensor_input, shape=[1, 1, 1, 3])
        out_op, argmax_op = nn_ops.max_pool_with_argmax(
            t,
            ksize=[1, 1, 1, 3],
            strides=[1, 1, 1, 3],
            padding=config.padding,
        )
        out, argmax = self.evaluate([out_op, argmax_op])
        # TODO(b/259733542): Fix below asserts once bug is fixed.
        # self.assertShapeEqual(out, out_op)
        # self.assertShapeEqual(argmax, argmax_op)
        self.assertAllClose(out.ravel(), [89, 73, -109])
        self.assertAllClose(argmax.ravel(), [0, 1, 2])

  def testMaxPoolingGradWithArgmax(self):
    # [FUNCTION DOC]
    # Tests the argmax-based max pool gradient op on a two-batch 3x3 input with a 2x2 window,
    # stride 1, valid padding, with and without batch offsets in the argmax indices, on CPU and
    # GPU scopes. Asserts the gradient scatters each incoming value to its argmax position and
    # accumulates where indices coincide, matching an explicit eighteen-element list.
    orig_input = [
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0
    ]
    tensor_input = [11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0]

    Config = collections.namedtuple(
        "Config", ["use_gpu", "include_batch_in_index", "argmax"])
    configs = [
        Config(False, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(False, True, [0, 1, 3, 5, 9, 11, 15, 17]),
        Config(True, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(True, True, [0, 1, 3, 5, 9, 11, 15, 17])
    ]

    for config in configs:
      with GetDeviceScope(self, config.use_gpu):
        orig_in = constant_op.constant(orig_input, shape=[2, 3, 3, 1])
        t = constant_op.constant(tensor_input, shape=[2, 2, 2, 1])
        argmax_t = constant_op.constant(
            config.argmax, shape=[2, 2, 2, 1], dtype=dtypes.int64)
        out_op = gen_nn_ops.max_pool_grad_with_argmax(
            orig_in,
            t,
            argmax_t,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            include_batch_in_index=config.include_batch_in_index)
        out = self.evaluate(out_op).flatten()
        self.assertAllClose(out, [
            11.0, 12.0, 0.0, 13.0, 0.0, 14.0, 0.0, 0.0, 0.0, 21.0, 0.0, 22.0,
            0.0, 0.0, 0.0, 23.0, 0.0, 24.0
        ])

  def testMaxPoolingGradThrowDeterminismError(self):
    # [FUNCTION DOC]
    # Enables op determinism, then runs the argmax-based max pool gradient. With a CUDA GPU it
    # expects an unimplemented error saying determinism is not supported for that op; without a
    # GPU it runs on CPU and expects success. Side effect: toggles global op determinism, always
    # disabling it again in a cleanup block.
    if test.is_gpu_available(cuda_only=True):
      try:
        config_exec.enable_op_determinism()
        orig_input = [
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 1.0
        ]
        tensor_input = [11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0]

        with GetDeviceScope(self, True):
          orig_in = constant_op.constant(orig_input, shape=[2, 3, 3, 1])
          t = constant_op.constant(tensor_input, shape=[2, 2, 2, 1])
          argmax_t = constant_op.constant(
              [0, 1, 3, 5, 0, 2, 6, 8], shape=[2, 2, 2, 1], dtype=dtypes.int64)
          with self.assertRaisesRegex(
              errors_impl.UnimplementedError, "Determinism is not yet supported"
              " for MaxPoolGradWithArgmax."):
            out_op = gen_nn_ops.max_pool_grad_with_argmax(
                orig_in,
                t,
                argmax_t,
                ksize=[1, 2, 2, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                include_batch_in_index=False)
            self.evaluate(out_op)
      finally:
        config_exec.disable_op_determinism()
    else:
      try:
        config_exec.enable_op_determinism()
        orig_input = [
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 1.0
        ]
        tensor_input = [11.0, 12.0, 13.0, 14.0, 21.0, 22.0, 23.0, 24.0]

        with GetDeviceScope(self, False):
          orig_in = constant_op.constant(orig_input, shape=[2, 3, 3, 1])
          t = constant_op.constant(tensor_input, shape=[2, 2, 2, 1])
          argmax_t = constant_op.constant(
              [0, 1, 3, 5, 0, 2, 6, 8], shape=[2, 2, 2, 1], dtype=dtypes.int64)
          out_op = gen_nn_ops.max_pool_grad_with_argmax(
              orig_in,
              t,
              argmax_t,
              ksize=[1, 2, 2, 1],
              strides=[1, 1, 1, 1],
              padding="VALID",
              include_batch_in_index=False)
          self.evaluate(out_op)
      finally:
        config_exec.disable_op_determinism()

  def testMaxPoolingGradGradWithArgmax(self):
    # [FUNCTION DOC]
    # Tests the argmax-based second-order max pool gradient on GPU only (returns immediately,
    # reporting a pass, when no CUDA GPU exists). Uses a two-batch 3x3 input, 2x2 window,
    # stride 1, valid padding, with and without batch offsets in the indices, and asserts the
    # output equals the eight gathered values 11, 12, 14, 16, 21, 23, 27, 29.
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    orig_input = [
        1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 1.0
    ]
    tensor_input = [
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 21.0, 22.0, 23.0,
        24.0, 25.0, 26.0, 27.0, 28.0, 29.0
    ]

    Config = collections.namedtuple(
        "Config", ["use_gpu", "include_batch_in_index", "argmax"])
    configs = [
        Config(True, False, [0, 1, 3, 5, 0, 2, 6, 8]),
        Config(True, True, [0, 1, 3, 5, 9, 11, 15, 17])
    ]

    for config in configs:
      with GetDeviceScope(self, config.use_gpu):
        orig_in = constant_op.constant(orig_input, shape=[2, 3, 3, 1])
        t = constant_op.constant(tensor_input, shape=[2, 3, 3, 1])
        argmax_t = constant_op.constant(
            config.argmax, shape=[2, 2, 2, 1], dtype=dtypes.int64)
        out_op = gen_nn_ops.max_pool_grad_grad_with_argmax(
            orig_in,
            t,
            argmax_t,
            ksize=[1, 2, 2, 1],
            strides=[1, 1, 1, 1],
            padding="VALID",
            include_batch_in_index=config.include_batch_in_index)
        out = self.evaluate(out_op).flatten()
        self.assertAllClose(out,
                            [11.0, 12.0, 14.0, 16.0, 21.0, 23.0, 27.0, 29.0])

  def _ConstructAndTestGradient(self,
                                pool_func,
                                input_sizes,
                                output_sizes,
                                window_rows,
                                window_cols,
                                row_stride,
                                col_stride,
                                padding,
                                data_format,
                                use_gpu,
                                x_init_value=None):
    # [FUNCTION DOC]
    # HELPER: numerical first-order gradient check shared by the max/avg gradient tests.
    # Inputs: pool_func (max or average pool function), input_sizes and output_sizes (4D shapes in
    # channels-last order), window_rows/window_cols (window size), row_stride/col_stride,
    # padding (string or explicit list), data_format (channels-last or channels-first), use_gpu,
    # x_init_value (optional point at which to evaluate; defaults to increasing values for max
    # pool so that maxima are unambiguous).
    # Behavior: asserts batch and depth sizes of input and output agree, builds an input of
    # increasing values, converts layout for the channels-first format, applies the pooling,
    # converts back and computes the maximum error between analytic and numeric Jacobians with a
    # step of 0.01. Asserts the error is below 1e-4 for average pool or 1e-3 for max pool, and
    # logs it. Returns nothing.
    """Verifies the gradients of the max or avg pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      output_sizes: Output tensor dimensions.
      window_rows: kernel size in row dim
      window_cols: kernel size in col dim
      row_stride: Row Stride.
      col_stride: Col Stride.
      padding: Padding type.
      data_format: Data format.
      use_gpu: whether we are running on GPU
      x_init_value: Values to be passed to the gradient checker.
    """
    assert input_sizes[0] == output_sizes[0]
    assert input_sizes[3] == output_sizes[3]
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
    with self.cached_session(use_gpu=use_gpu):
      input_tensor = constant_op.constant(x, shape=input_sizes, name="input")
      if pool_func == nn_ops.avg_pool:
        func_name = "avg_pool"
        err_tolerance = 1e-4
      else:
        if x_init_value is None:
          x_init_value = np.asarray(
              np.arange(1, total_size + 1),
              dtype=np.float32).reshape(input_sizes)
        func_name = "max_pool"
        err_tolerance = 1e-3
      if data_format == "NCHW":
        ksize = [1, 1, window_rows, window_cols]
        strides = [1, 1, row_stride, col_stride]
        if isinstance(padding, list):
          padding = test_util.NHWCToNCHW(padding)
        t = test_util.NHWCToNCHW(input_tensor)
      else:
        ksize = [1, window_rows, window_cols, 1]
        strides = [1, row_stride, col_stride, 1]
        t = input_tensor
      t = pool_func(
          t,
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format,
          name=func_name)
      if data_format == "NCHW":
        t = test_util.NCHWToNHWC(t)

      err = gradient_checker.compute_gradient_error(
          input_tensor,
          input_sizes,
          t,
          output_sizes,
          x_init_value=x_init_value,
          delta=1e-2)
    tf_logging.info("%s gradient error = %.4f" % (func_name, err))
    self.assertLess(err, err_tolerance)

  def _ConstructAndTestSecondGradient(self,
                                      pool_func,
                                      input_sizes,
                                      output_sizes,
                                      window_rows,
                                      window_cols,
                                      row_stride,
                                      col_stride,
                                      padding,
                                      data_format,
                                      use_gpu,
                                      x_init_value=None):
    # [FUNCTION DOC]
    # HELPER: numerical second-order gradient check shared by the max pool grad-grad tests.
    # KNOWN ISSUE: the channels-first branch builds the kernel size from the row window size twice
    # (the column window size parameter is ignored), and after pooling it converts the result with
    # the channels-last-to-channels-first conversion again instead of converting back, so the
    # channels-first path is wrong (only exercised when a GPU is available). Callers always pass
    # equal row and column windows so the ksize slip is hidden. The channels-last path also uses
    # the row window size for both window dimensions.
    # Inputs: same as the first-order helper (pool function, shapes, window and stride sizes,
    # padding, data format, GPU flag, optional initial value).
    # Behavior: pools the input, takes the gradient of the squared output with respect to the
    # input, and computes the numerical error of that gradient; asserts it is below 1e-3 for
    # average pool or 1e-2 for max pool. Returns nothing.
    """Verifies the second-order gradients of the pooling function.

    Args:
      pool_func: Function to be called, co.MaxPool, co.AvgPool,
        or the Lua version.
      input_sizes: Input tensor dimensions.
      output_sizes: Output tensor dimensions.
      window_rows: kernel size in row dim
      window_cols: kernel size in col dim
      row_stride: Row Stride.
      col_stride: Col Stride.
      padding: Padding type.
      data_format: Data format.
      use_gpu: whether we are running on GPU
      x_init_value: Values to be passed to the gradient checker.
    """
    assert input_sizes[0] == output_sizes[0]
    assert input_sizes[3] == output_sizes[3]
    total_size = 1
    for s in input_sizes:
      total_size *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x = [f * 1.0 for f in range(1, total_size + 1)]
    with self.cached_session(use_gpu=use_gpu):
      input_tensor = constant_op.constant(x, shape=input_sizes, name="input")
      if pool_func == nn_ops.avg_pool:
        func_name = "avg_pool"
        err_tolerance = 1e-3
      else:
        if x_init_value is None:
          x_init_value = np.asarray(
              np.arange(1, total_size + 1),
              dtype=np.float32).reshape(input_sizes)
        func_name = "max_pool"
        err_tolerance = 1e-2
      if data_format == "NCHW":
        ksize = [1, 1, window_rows, window_rows]
        strides = [1, 1, row_stride, col_stride]
        t = test_util.NHWCToNCHW(input_tensor)
      else:
        ksize = [1, window_rows, window_rows, 1]
        strides = [1, row_stride, col_stride, 1]
        t = input_tensor
      t = pool_func(
          t,
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format,
          name=func_name)
      if data_format == "NCHW":
        t = test_util.NHWCToNCHW(t)

      t_g = gradients_impl.gradients(t**2, input_tensor)[0]
      err = gradient_checker.compute_gradient_error(
          input_tensor,
          input_sizes,
          t_g,
          input_sizes,
          x_init_value=x_init_value,
          delta=1e-2)
    tf_logging.info("%s second-order gradient error = %.4f" % (func_name, err))
    self.assertLess(err, err_tolerance)

  def _testMaxPoolGradValidPadding1_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. For both the v2 and v1 max pool functions, runs the
    # numerical gradient check on a 1x3x3x1 input with a 1x1 window, stride 1, valid padding.
    # Inputs: data_format and use_gpu. Returns nothing; fails by assertion.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 3, 3, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_1_6(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for v2 and v1 max pool on a 2x6x6x3
    # input, 2x2 window, stride 1, valid padding (output 2x5x5x3).
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 6, 6, 3],
          output_sizes=[2, 5, 5, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_1_7(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for v2 and v1 max pool on a 2x7x7x3
    # input (odd spatial size), 2x2 window, stride 1, valid padding (output 2x6x6x3).
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 7, 7, 3],
          output_sizes=[2, 6, 6, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding1_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for v2 and v1 max pool on a 1x3x3x1
    # input, 1x1 window, stride 2, valid padding (output 1x2x2x1).
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 3, 3, 1],
          output_sizes=[1, 2, 2, 1],
          window_rows=1,
          window_cols=1,
          row_stride=2,
          col_stride=2,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradValidPadding2_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for v2 and v1 max pool on a 2x2x2x3
    # input, 2x2 window, stride 2, valid padding (output 2x1x1x3).
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 2, 3],
          output_sizes=[2, 1, 1, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding1_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for v2 and v1 max pool on a 2x2x4x3
    # input, 1x1 window, stride 1, same padding (output same shape as input).
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding1_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for v2 and v1 max pool on a 2x2x4x3
    # input, 1x1 window, stride 2, same padding (output 2x1x2x3).
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 1, 2, 3],
          window_rows=1,
          window_cols=1,
          row_stride=2,
          col_stride=2,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding2_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for v2 and v1 max pool on a 2x2x4x3
    # input, 2x2 window, stride 1, same padding (output same shape as input).
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding2_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for v2 and v1 max pool on a 2x2x4x3
    # input, 2x2 window, stride 2, same padding (output 2x1x2x3).
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 1, 2, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradSamePadding3_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for v2 and v1 max pool on a 1x7x7x1
    # input, 3x3 window, stride 1, same padding (output same shape as input).
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 7, 7, 1],
          window_rows=3,
          window_cols=3,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolExplicitPadding_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for the v1 max pool only on a 1x7x7x1
    # input, 3x3 window, stride 1, explicit padding of one on every spatial side (same size
    # output). Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 7, 7, 1],
          window_rows=3,
          window_cols=3,
          row_stride=1,
          col_stride=1,
          padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolExplicitPadding_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for the v1 max pool on a 1x7x7x1
    # input with a 3x5 window, stride 1 and asymmetric explicit padding (rows 0 and 1, columns 2
    # and 3), giving a 1x6x8x1 output. Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 6, 8, 1],
          window_rows=3,
          window_cols=5,
          row_stride=1,
          col_stride=1,
          padding=[[0, 0], [0, 1], [2, 3], [0, 0]],
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolExplicitPaddingLeftGreater(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for the v1 max pool on a 1x7x7x1
    # input, 3x5 window, stride 1, explicit padding where the left column padding (3) exceeds the
    # right (2); output 1x6x8x1. Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 6, 8, 1],
          window_rows=3,
          window_cols=5,
          row_stride=1,
          col_stride=1,
          padding=[[0, 0], [0, 1], [3, 2], [0, 0]],
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolExplicitPaddingBatchChannel(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for the v1 max pool with explicit
    # padding on a 4x7x7x3 input (several batches and channels), 3x5 window, stride 1; output
    # 4x6x8x3. Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[4, 7, 7, 3],
          output_sizes=[4, 6, 8, 3],
          window_rows=3,
          window_cols=5,
          row_stride=1,
          col_stride=1,
          padding=[[0, 0], [0, 1], [3, 2], [0, 0]],
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolExplicitPaddingStrides(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGrad. Gradient check for the v1 max pool on a 1x7x7x1
    # input, 3x3 window, row stride 2, column stride 3, explicit padding of one on each spatial
    # side; output 1x4x3x1. Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [nn_ops.max_pool]:
      self._ConstructAndTestGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 4, 3, 1],
          window_rows=3,
          window_cols=3,
          row_stride=2,
          col_stride=3,
          padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
          data_format=data_format,
          use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testMaxPoolGrad(self):
    # [FUNCTION DOC]
    # Runs every max pool gradient scenario helper (valid, same and explicit padding variants)
    # for each (data format, GPU) configuration returned by the config helper, asserting
    # the numerical gradient error is within tolerance. Graph-mode only (deprecated v1 style).
    for (data_format, use_gpu) in GetTestConfigs():
      self._testMaxPoolGradValidPadding1_1(data_format, use_gpu)
      self._testMaxPoolGradValidPadding1_2(data_format, use_gpu)
      self._testMaxPoolGradValidPadding2_1_6(data_format, use_gpu)
      self._testMaxPoolGradValidPadding2_1_7(data_format, use_gpu)
      self._testMaxPoolGradValidPadding2_2(data_format, use_gpu)
      self._testMaxPoolGradSamePadding1_1(data_format, use_gpu)
      self._testMaxPoolGradSamePadding1_2(data_format, use_gpu)
      self._testMaxPoolGradSamePadding2_1(data_format, use_gpu)
      self._testMaxPoolGradSamePadding2_2(data_format, use_gpu)
      self._testMaxPoolGradSamePadding3_1(data_format, use_gpu)
      self._testMaxPoolExplicitPadding_1(data_format, use_gpu)
      self._testMaxPoolExplicitPadding_2(data_format, use_gpu)
      self._testMaxPoolExplicitPaddingStrides(data_format, use_gpu)
      self._testMaxPoolExplicitPaddingLeftGreater(data_format, use_gpu)
      self._testMaxPoolExplicitPaddingBatchChannel(data_format, use_gpu)

  def _MaxPoolGrad(self, orig_input, orig_output, grad, window_rows,
                   window_cols, row_stride, col_stride, padding, v2):
    # [FUNCTION DOC]
    # HELPER: invokes the raw max pool gradient op for the direct gradient tests.
    # KNOWN ISSUE: the docstring omits the v2 parameter, and says padding is a padding enum
    # type while it is a padding string or explicit padding list.
    # Inputs: original input and output tensors, incoming gradient, window sizes, strides,
    # padding, and v2 (choose the gradient op taking tensor window and stride). For v2 it passes
    # the padding as given; otherwise it first converts the padding into a padding string plus
    # explicit padding list. Returns the resulting input-gradient tensor.
    """Max Pooling Gradient.

    Args:
      orig_input: A float Tensor. The original input tensor.
      orig_output: A float Tensor. The original output tensor.
      grad: A float Tensor.
        The 4D (batch x rows x cols x depth) output backprop.
      window_rows: integer. Kernel size along rows dimension.
      window_cols: integer. Kernel size along cols dimension.
      row_stride: integer. Stride along rows dimension
      col_stride: integer. Stride along cols dimension
      padding: PoolingOpDef.Padding.  Padding type.

    Returns:
      A Tensor.
    """
    pool_func = gen_nn_ops.max_pool_grad_v2 if v2 else gen_nn_ops.max_pool_grad
    if v2:
      return pool_func(orig_input, orig_output, grad,
                       [1, window_rows, window_cols, 1],
                       [1, row_stride, col_stride, 1], padding)
    else:
      padding, explicit_paddings = nn_ops.convert_padding(padding)
      return pool_func(orig_input, orig_output, grad,
                       [1, window_rows, window_cols, 1],
                       [1, row_stride, col_stride, 1], padding,
                       explicit_paddings)

  def _testMaxPoolGradDirect(self, input_data, output_backprop,
                             expected_input_backprop, input_sizes, output_sizes,
                             window_rows, window_cols, row_stride, col_stride,
                             padding, use_gpu, v2):
    # [FUNCTION DOC]
    # HELPER: verifies the max pool gradient op against hand-computed values.
    # Inputs: input_data (flat list of input values), output_backprop (flat incoming gradient),
    # expected_input_backprop (flat expected result), input_sizes and output_sizes (shapes),
    # window sizes, strides, padding, use_gpu, and v2 (use the tensor-window variants).
    # Behavior: puts the input in an initialized variable, runs the max pool, then the gradient op
    # and compares the flattened gradient with the expected values (tolerance 1e-6), also
    # checking the static shape. It also evaluates the forward output (unused afterwards).
    # Returns nothing; failures are assertion errors.
    pool_func = gen_nn_ops.max_pool_v2 if v2 else nn_ops.max_pool
    with self.cached_session(use_gpu=use_gpu):
      input_tensor = variables.Variable(
          np.array(input_data, dtype=np.float32).reshape(input_sizes))
      self.evaluate(variables.global_variables_initializer())
      output_tensor = pool_func(input_tensor, [1, window_rows, window_cols, 1],
                                [1, row_stride, col_stride, 1], padding)
      output_backprop_tensor = constant_op.constant(
          output_backprop, shape=output_sizes)

      input_backprop_tensor = self._MaxPoolGrad(
          input_tensor, output_tensor, output_backprop_tensor, window_rows,
          window_cols, row_stride, col_stride, padding, v2)

      actual_input_backprop = self.evaluate(input_backprop_tensor)
      self.assertShapeEqual(actual_input_backprop, input_backprop_tensor)
      actual_input_backprop = actual_input_backprop.flatten()
      actual_input_backprop = self._GetNdArray(actual_input_backprop)

      actual_output = self.evaluate(output_tensor).flatten()
      actual_output = self._GetNdArray(actual_output)

      self.assertAllClose(
          expected_input_backprop, actual_input_backprop, rtol=1e-6, atol=1e-6)

  def _testMaxPoolGradDirect1_1(self):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradDirect. All-ones 4x4 input, 2x2 window, stride 1,
    # valid padding, so ties go to the first element of each window; runs on GPU and CPU flags
    # and for v2 true and false, expecting the gradient values to land on the top-left cells of
    # each window.
    input_data = [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    expected_input_backprop = [
        11.0, 12.0, 13.0, 0.0, 15.0, 16.0, 17.0, 0.0, 19.0, 20.0, 21.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ]

    for use_gpu in True, False:
      for v2 in [True, False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 3, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding="VALID",
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradDirect1_2(self):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradDirect. 4x4 input with a checkerboard of ones and zeros,
    # 2x2 window, stride 1, valid padding; gradients overlap and accumulate on the ones. Runs
    # both device flags and both v2 settings.
    input_data = [
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    expected_input_backprop = [
        11.0, 0.0, 25.0, 0.0, 0.0, 31.0, 0.0, 17.0, 19.0, 0.0, 41.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]

    for use_gpu in True, False:
      for v2 in [True, False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 3, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding="VALID",
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradDirect1_3(self):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradDirect. Checkerboard 4x4 input, 3x3 window, stride 1,
    # same padding, sixteen incoming gradient values; checks accumulation of gradients where
    # several windows pick the same maximum. Runs both device flags and both v2 settings.
    input_data = [
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
    ]
    output_backprop = [
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0,
        23.0, 24.0, 25.0, 26.0
    ]
    expected_input_backprop = [
        54,
        0.0,
        62,
        0.0,
        0.0,
        60,
        0.0,
        22.0,
        47,
        0.0,
        51,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    for use_gpu in True, False:
      for v2 in [True, False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 4, 4, 1],
            window_rows=3,
            window_cols=3,
            row_stride=1,
            col_stride=1,
            padding="SAME",
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradZeroExplicitPadding(self):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradDirect. Same data as the checkerboard valid-padding case
    # but with an explicit padding list of all zeros; v1 only (explicit padding unsupported by v2).
    # Runs on both device flags and expects the same gradient as valid padding.
    input_data = [
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    expected_input_backprop = [
        11.0, 0.0, 25.0, 0.0, 0.0, 31.0, 0.0, 17.0, 19.0, 0.0, 41.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]

    for use_gpu in True, False:
      for v2 in [False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 3, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding=[[0, 0], [0, 0], [0, 0], [0, 0]],
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradExplicitPadding_1(self):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradDirect. Checkerboard 4x4 input, 2x2 window, stride 1,
    # explicit padding of one extra column on the right (output 3x4); v1 only; both device flags.
    # Checks gradient routing when windows overlap the padded column.
    input_data = [
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                       20.0, 21.0, 22.0]
    expected_input_backprop = [
        11.0, 0.0, 25.0, 0.0, 0.0, 31.0, 0.0, 49.0, 19.0, 0.0, 41.0, 0.0, 0.0,
        0.0, 0.0, 22.0
    ]

    for use_gpu in True, False:
      for v2 in [False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 4, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding=[[0, 0], [0, 0], [0, 1], [0, 0]],
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradExplicitPadding_2(self):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradDirect. Checkerboard 4x4 input, 3x3 window, stride 2,
    # explicit padding of 2 before and 1 after in both spatial dimensions (output 3x3); v1 only;
    # both device flags. Checks gradient placement with large leading padding.
    input_data = [
        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    expected_input_backprop = [
        54.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 39.0, 0.0, 21.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ]

    for use_gpu in True, False:
      for v2 in [False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 3, 1],
            window_rows=3,
            window_cols=3,
            row_stride=2,
            col_stride=2,
            padding=[[0, 0], [2, 1], [2, 1], [0, 0]],
            use_gpu=use_gpu,
            v2=v2)

  def _testMaxPoolGradExplicitPadding_3(self):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradDirect. Same shapes as the first explicit-padding
    # scenario but with a negative-valued checkerboard input (minus one and minus five), so
    # padding cells must not win; expects the same gradient as that scenario. v1 only.
    input_data = [
        -1.0, -5.0, -1.0, -5.0, -5.0, -1.0, -5.0, -1.0, -1.0, -5.0, -1.0, -5.0,
        -5.0, -1.0, -5.0, -1.0
    ]
    output_backprop = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
                       20.0, 21.0, 22.0]
    expected_input_backprop = [
        11.0, 0.0, 25.0, 0.0, 0.0, 31.0, 0.0, 49.0, 19.0, 0.0, 41.0, 0.0, 0.0,
        0.0, 0.0, 22.0
    ]

    for use_gpu in True, False:
      for v2 in [False]:
        self._testMaxPoolGradDirect(
            input_data,
            output_backprop,
            expected_input_backprop,
            input_sizes=[1, 4, 4, 1],
            output_sizes=[1, 3, 4, 1],
            window_rows=2,
            window_cols=2,
            row_stride=1,
            col_stride=1,
            padding=[[0, 0], [0, 0], [0, 1], [0, 0]],
            use_gpu=use_gpu,
            v2=v2)

  @test_util.no_xla_auto_jit("b/123923733")  # NaNs handled differently
  def _testMaxPoolGradDirectWithNans2_1(self):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradDirect with NaN inputs.
    # KNOWN ISSUE: the environment variable is restored only on success; a failed assertion in
    # between leaves it modified for the rest of the process.
    # Input made entirely of not-a-number
    # values, 2x2 window, stride 1, valid padding. The CPU result must propagate the incoming
    # gradient to the first window cell. If a non-ROCm GPU exists it temporarily sets the
    # NaN-propagation environment variable: with it off the expected gradient is all zeros; with
    # it on the expected gradient equals the CPU result. Side effect: modifies the process
    # environment variable and restores or removes it at the end (not restored on failure).
    input_data = [float("nan")] * 16
    output_backprop = [11.0, 12.0, 13.0, 15.0, 16.0, 17.0, 19.0, 20.0, 21.0]
    # Test the CPU implementation, which propagates diffs in case of NaN
    expected_input_backprop_tf_cpu = [
        11.0, 12.0, 13.0, 0.0, 15.0, 16.0, 17.0, 0.0, 19.0, 20.0, 21.0, 0.0,
        0.0, 0.0, 0.0, 0.0
    ]
    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_tf_cpu,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=False,
          v2=v2)

    if not test.is_gpu_available():
      return

    # The functionality associated with TF_ENABLE_NANPROP is currently
    # not supported on the ROCm platform, so skip this part of the test
    # NANs in input lead to non-deterministic results, and hence skipping
    # the remaining tests altogether on the ROCm platform
    if test.is_built_with_rocm():
      return

    # Test the GPU implementation that uses cudnn for now.
    saved_nanprop = os.environ.get("TF_ENABLE_MAXPOOL_NANPROP")
    # Do not propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "0"
    expected_input_backprop_cudnn = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    # Propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "1"
    expected_input_backprop_cudnn = expected_input_backprop_tf_cpu

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    if saved_nanprop:
      os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = saved_nanprop
    else:
      del os.environ["TF_ENABLE_MAXPOOL_NANPROP"]

  @test_util.no_xla_auto_jit("b/123923733")  # NaNs handled differently
  def _testMaxPoolGradDirectWithNans2_2(self):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradDirect with NaN inputs.
    # KNOWN ISSUE: the environment variable is restored only on success; a failed assertion in
    # between leaves it modified for the rest of the process.
    # All-NaN input, 2x2 window, stride 1, valid padding, where some incoming gradient values are
    # themselves NaN. Same CPU and GPU expectations and the same environment variable handling
    # as the first NaN scenario.
    input_data = [float("nan")] * 16
    output_backprop = [
        float("nan"), 12.0, 13.0, 15.0,
        float("nan"), 17.0, 19.0, 20.0,
        float("nan")
    ]
    # Test the CPU implementation, which propagates diffs in case of NaN
    expected_input_backprop_tf_cpu = [
        float("nan"), 12.0, 13.0, 0.0, 15.0,
        float("nan"), 17.0, 0.0, 19.0, 20.0,
        float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0
    ]
    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_tf_cpu,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=False,
          v2=v2)

    if not test.is_gpu_available():
      return

    # The functionality associated with TF_ENABLE_NANPROP is currently
    # not supported on the ROCm platform, so skip this part of the test
    # NANs in input lead to non-deterministic results, and hence skipping
    # the remaining tests altogether on the ROCm platform
    if test.is_built_with_rocm():
      return

    # Test the GPU implementation that uses cudnn for now.
    saved_nanprop = os.environ.get("TF_ENABLE_MAXPOOL_NANPROP")
    # Do not propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "0"
    expected_input_backprop_cudnn = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0
    ]

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    # Propagate the diff in cases of NaNs
    os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = "1"
    expected_input_backprop_cudnn = expected_input_backprop_tf_cpu

    for v2 in [True, False]:
      self._testMaxPoolGradDirect(
          input_data,
          output_backprop,
          expected_input_backprop_cudnn,
          input_sizes=[1, 4, 4, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          use_gpu=True,
          v2=v2)

    if saved_nanprop:
      os.environ["TF_ENABLE_MAXPOOL_NANPROP"] = saved_nanprop
    else:
      del os.environ["TF_ENABLE_MAXPOOL_NANPROP"]

  @test_util.run_deprecated_v1
  def testMaxPoolGradDirect(self):
    # [FUNCTION DOC]
    # Runs all direct max pool gradient scenarios in turn: the three basic cases, two NaN cases,
    # zero explicit padding and three explicit-padding cases. Each scenario asserts the computed
    # input gradient against hard-coded expected values. Graph-mode only.
    self._testMaxPoolGradDirect1_1()
    self._testMaxPoolGradDirect1_2()
    self._testMaxPoolGradDirect1_3()
    self._testMaxPoolGradDirectWithNans2_1()
    self._testMaxPoolGradDirectWithNans2_2()
    self._testMaxPoolGradZeroExplicitPadding()
    self._testMaxPoolGradExplicitPadding_1()
    self._testMaxPoolGradExplicitPadding_2()
    self._testMaxPoolGradExplicitPadding_3()

  def _testMaxPoolGradGradValidPadding1_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradGrad. Second-order gradient check for v2 and v1 max
    # pool on a 1x3x3x1 input, 1x1 window, stride 1, valid padding.
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[1, 3, 3, 1],
          output_sizes=[1, 3, 3, 1],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradValidPadding2_1_6(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradGrad. Second-order gradient check for v2 and v1 max
    # pool on a 2x6x6x3 input, 2x2 window, stride 1, valid padding.
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 6, 6, 3],
          output_sizes=[2, 5, 5, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradValidPadding2_1_7(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradGrad. Second-order gradient check for v2 and v1 max
    # pool on a 2x7x7x3 input, 2x2 window, stride 1, valid padding.
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 7, 7, 3],
          output_sizes=[2, 6, 6, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradValidPadding2_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradGrad. Second-order gradient check for v2 and v1 max
    # pool on a 2x2x2x3 input, 2x2 window, stride 2, valid padding.
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 2, 3],
          output_sizes=[2, 1, 1, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="VALID",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding1_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradGrad. Second-order gradient check for v2 and v1 max
    # pool on a 2x2x4x3 input, 1x1 window, stride 1, same padding.
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=1,
          window_cols=1,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding2_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradGrad. Second-order gradient check for v2 and v1 max
    # pool on a 2x2x4x3 input, 2x2 window, stride 1, same padding.
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 2, 4, 3],
          window_rows=2,
          window_cols=2,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding2_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradGrad. Second-order gradient check for v2 and v1 max
    # pool on a 2x2x4x3 input, 2x2 window, stride 2, same padding.
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[2, 2, 4, 3],
          output_sizes=[2, 1, 2, 3],
          window_rows=2,
          window_cols=2,
          row_stride=2,
          col_stride=2,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  def _testMaxPoolGradGradSamePadding3_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testMaxPoolGradGrad. Second-order gradient check for v2 and v1 max
    # pool on a 1x7x7x1 input, 3x3 window, stride 1, same padding.
    # Inputs: data_format and use_gpu. Returns nothing.
    for pool_func in [gen_nn_ops.max_pool_v2, nn_ops.max_pool]:
      self._ConstructAndTestSecondGradient(
          pool_func,
          input_sizes=[1, 7, 7, 1],
          output_sizes=[1, 7, 7, 1],
          window_rows=3,
          window_cols=3,
          row_stride=1,
          col_stride=1,
          padding="SAME",
          data_format=data_format,
          use_gpu=use_gpu)

  @test_util.run_deprecated_v1
  def testMaxPoolGradGrad(self):
    # [FUNCTION DOC]
    # Runs every second-order max pool gradient scenario helper for each (data format, GPU)
    # configuration, asserting numerical error bounds. Graph-mode only.
    for (data_format, use_gpu) in GetTestConfigs():
      self._testMaxPoolGradGradValidPadding1_1(data_format, use_gpu)
      self._testMaxPoolGradGradValidPadding2_1_6(data_format, use_gpu)
      self._testMaxPoolGradGradValidPadding2_1_7(data_format, use_gpu)
      self._testMaxPoolGradGradValidPadding2_2(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding1_1(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding2_1(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding2_2(data_format, use_gpu)
      self._testMaxPoolGradGradSamePadding3_1(data_format, use_gpu)

  def _MaxPoolGradGrad(self, orig_input, orig_output, grad, window_rows,
                       window_cols, row_stride, col_stride, padding):
    # [FUNCTION DOC]
    # HELPER: thin wrapper over the raw max pool grad-of-grad op.
    # KNOWN ISSUE: unused anywhere in this file (dead code); the docstring's padding type
    # description is also not accurate since a padding string is expected.
    # Inputs: original input and output tensors, the gradient tensor, window sizes, strides and
    # padding string. Returns the second-order gradient tensor computed with channels-last
    # window and stride lists.
    """Max Pooling Second-Order Gradient.

    Args:
      orig_input: A float Tensor. The original input tensor.
      orig_output: A float Tensor. The original output tensor.
      grad: A float Tensor.
        The 4D (batch x out_rows x out_cols x depth) output backprop.
      window_rows: integer. Kernel size along rows dimension.
      window_cols: integer. Kernel size along cols dimension.
      row_stride: integer. Stride along rows dimension
      col_stride: integer. Stride along cols dimension
      padding: PoolingOpDef.Padding.  Padding type.

    Returns:
      A Tensor.
    """
    return gen_nn_ops.max_pool_grad_grad(
        orig_input, orig_output, grad, [1, window_rows, window_cols, 1],
        [1, row_stride, col_stride, 1], padding)

  @test_util.run_deprecated_v1
  def testAvgPoolGrad(self):
    # [FUNCTION DOC]
    # Runs every average pool gradient scenario helper (valid and same padding variants) for each
    # (data format, GPU) configuration, asserting numerical gradient error bounds. Graph-mode only.
    for (data_format, use_gpu) in GetTestConfigs():
      self._testAvgPoolGradValidPadding1_1(data_format, use_gpu)
      self._testAvgPoolGradValidPadding1_2(data_format, use_gpu)
      self._testAvgPoolGradValidPadding2_1(data_format, use_gpu)
      self._testAvgPoolGradValidPadding2_2(data_format, use_gpu)
      self._testAvgPoolGradSamePadding1_1(data_format, use_gpu)
      self._testAvgPoolGradSamePadding1_2(data_format, use_gpu)
      self._testAvgPoolGradSamePadding2_1(data_format, use_gpu)
      self._testAvgPoolGradSamePadding2_2(data_format, use_gpu)
      self._testAvgPoolGradSamePadding3_1(data_format, use_gpu)

  def _testAvgPoolGradValidPadding1_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testAvgPoolGrad. Gradient check for average pool on a 2x3x3x3 input,
    # 1x1 window, stride 1, valid padding. Inputs: data_format and use_gpu. Returns nothing.
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 3, 3, 3],
        window_rows=1,
        window_cols=1,
        row_stride=1,
        col_stride=1,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding1_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testAvgPoolGrad. Gradient check for average pool on a 2x3x3x3 input,
    # 1x1 window, stride 2, valid padding. Inputs: data_format and use_gpu. Returns nothing.
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 2, 2, 3],
        window_rows=1,
        window_cols=1,
        row_stride=2,
        col_stride=2,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding2_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testAvgPoolGrad. Gradient check for average pool on a 2x3x3x3 input,
    # 2x2 window, stride 1, valid padding. Inputs: data_format and use_gpu. Returns nothing.
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 3, 3, 3],
        output_sizes=[2, 2, 2, 3],
        window_rows=2,
        window_cols=2,
        row_stride=1,
        col_stride=1,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradValidPadding2_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testAvgPoolGrad. Gradient check for average pool on a 2x2x2x3 input,
    # 2x2 window, stride 2, valid padding. Inputs: data_format and use_gpu. Returns nothing.
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 2, 3],
        output_sizes=[2, 1, 1, 3],
        window_rows=2,
        window_cols=2,
        row_stride=2,
        col_stride=2,
        padding="VALID",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding1_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testAvgPoolGrad. Gradient check for average pool on a 2x2x4x3 input,
    # 1x1 window, stride 1, same padding. Inputs: data_format and use_gpu. Returns nothing.
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 2, 4, 3],
        window_rows=1,
        window_cols=1,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding1_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testAvgPoolGrad. Gradient check for average pool on a 2x2x4x3 input,
    # 1x1 window, stride 2, same padding. Inputs: data_format and use_gpu. Returns nothing.
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 1, 2, 3],
        window_rows=1,
        window_cols=1,
        row_stride=2,
        col_stride=2,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding2_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testAvgPoolGrad. Gradient check for average pool on a 2x2x4x3 input,
    # 2x2 window, stride 1, same padding. Inputs: data_format and use_gpu. Returns nothing.
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 2, 4, 3],
        window_rows=2,
        window_cols=2,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding2_2(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testAvgPoolGrad. Gradient check for average pool on a 2x2x4x3 input,
    # 2x2 window, stride 2, same padding. Inputs: data_format and use_gpu. Returns nothing.
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[2, 2, 4, 3],
        output_sizes=[2, 1, 2, 3],
        window_rows=2,
        window_cols=2,
        row_stride=2,
        col_stride=2,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  def _testAvgPoolGradSamePadding3_1(self, data_format, use_gpu):
    # [FUNCTION DOC]
    # HELPER: scenario for testAvgPoolGrad. Gradient check for average pool on a 1x7x7x1 input,
    # 3x3 window, stride 1, same padding. Inputs: data_format and use_gpu. Returns nothing.
    self._ConstructAndTestGradient(
        nn_ops.avg_pool,
        input_sizes=[1, 7, 7, 1],
        output_sizes=[1, 7, 7, 1],
        window_rows=3,
        window_cols=3,
        row_stride=1,
        col_stride=1,
        padding="SAME",
        data_format=data_format,
        use_gpu=use_gpu)

  @test_util.disable_xla("Xla does not raise error on out of bounds access")
  def testAvgPoolGradOutputMemoryOutOfBounds(self):
    # [FUNCTION DOC]
    # Regression test (XLA disabled, since it does not raise on out-of-bounds access). Calls the
    # raw average pool gradient op with a gradient whose shape (3 rows) does not match the
    # original input shape (1 row) given strides that would index past the output. Expects an
    # invalid-argument error with either the CPU message about output elements or the GPU
    # message about the expected gradient shape.
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        (
            # CPU error message
            "(Output only has 3 elements but computation requested would use"
            " element with index=6"
            ")|("
            # GPU error message
            r"Expected grad shape to be \[1,1,3,1\], but got \[3,1,3,1\])"
        ),
    ):
      self.evaluate(
          gen_nn_ops.AvgPoolGrad(
              orig_input_shape=[1, 1, 3, 1],
              grad=[
                  [[[1.0], [2.0], [3.0]]],
                  [[[4.0], [5.0], [6.0]]],
                  [[[7.0], [8.0], [9.0]]],
              ],
              ksize=[1, 1, 1, 1],
              strides=[1, 1, 1, 2],
              padding="VALID",
              data_format="NHWC",
          )
      )

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # [FUNCTION DOC]
    # Checks static shape inference for pooling ops: with a fully unknown input shape, max pool,
    # average pool and max-pool-with-argmax produce outputs of unknown size in four dimensions;
    # with an input of the wrong rank (two dimensions) all three raise a value error.
    # Graph-mode only.
    # All shapes unknown.
    for pool_func in [nn_ops.max_pool, nn_ops.avg_pool]:
      p = pool_func(
          array_ops.placeholder(dtypes.float32),
          ksize=[1, 1, 1, 1],
          strides=[1, 1, 1, 1],
          padding="SAME")
      self.assertEqual([None, None, None, None], p.get_shape().as_list())
    p, am = nn_ops.max_pool_with_argmax(
        array_ops.placeholder(dtypes.float32),
        ksize=[1, 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding="SAME")
    self.assertEqual([None, None, None, None], p.get_shape().as_list())
    self.assertEqual([None, None, None, None], am.get_shape().as_list())

    # Incorrect input shape.
    for pool_func in [
        nn_ops.max_pool, nn_ops.avg_pool, nn_ops.max_pool_with_argmax
    ]:
      with self.assertRaises(ValueError):
        pool_func(
            array_ops.placeholder(dtypes.float32, shape=[1, 3]),
            ksize=[1, 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding="SAME")

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123337890")  # Error messages differ
  def testOpEdgeCases(self):
    # [FUNCTION DOC]
    # Checks error handling for pooling ops in a session (XLA disabled; messages differ).
    # For average pool (and argmax pool on GPU) a batch stride of 2 must raise an unimplemented
    # error about batch pooling; for all functions a window larger than the input with valid
    # padding must raise a value error about a negative dimension size, both at run time and at
    # graph construction. Graph-mode only.
    with self.session(use_gpu=test.is_gpu_available()) as sess:
      pool_funcs = [nn_ops.max_pool, nn_ops.avg_pool]
      if test.is_gpu_available():
        pool_funcs.append(nn_ops.max_pool_with_argmax)
      for pool_func in pool_funcs:
        if pool_func != nn_ops.max_pool:
          # Illegal strides.
          with self.assertRaisesRegex(
              errors_impl.UnimplementedError,
              "Pooling is not yet supported on the batch"):
            sess.run(
                pool_func(
                    array_ops.placeholder(dtypes.float32),
                    ksize=[1, 1, 1, 1],
                    strides=[2, 1, 1, 1],
                    padding="SAME"))

        # Filter too large.
        with self.assertRaisesRegex(ValueError, "Negative dimension size"):
          sess.run(
              pool_func(
                  array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 3]),
                  ksize=[1, 20, 21, 1],
                  strides=[1, 1, 1, 1],
                  padding="VALID"))
        with self.assertRaisesRegex(ValueError, "Negative dimension size"):
          pool_func(
              array_ops.placeholder(dtypes.float32, shape=[32, 20, 20, 3]),
              ksize=[1, 21, 20, 1],
              strides=[1, 1, 1, 1],
              padding="VALID")

  @test_util.run_deprecated_v1
  def testEdgeCasesRaiseErrors(self):
    # [FUNCTION DOC]
    # Checks argument validation errors: explicit padding with the vectorized-channel format (or
    # XLA's lack of explicit padding), explicit padding with a rank-5 input to the v2 max pool,
    # and the raw v2 max pool op given padding mode "explicit" without a padding list. Each case
    # expects a value error with the matching message. Graph-mode only.
    with self.assertRaisesRegex(
        ValueError, "NCHW_VECT_C.*is not supported with "
        "explicit padding|XLA does not support pooling ops with explicit "
        "padding"):
      nn_ops.max_pool(
          array_ops.placeholder(dtypes.float32, shape=[1, 3, 3, 1]),
          ksize=[1, 2, 2, 1],
          strides=[1, 2, 2, 1],
          padding=[[0, 0], [0, 1], [0, 1], [0, 0]],
          data_format="NCHW_VECT_C")
    with self.assertRaisesRegex(
        ValueError, "Explicit padding is not supported with an input "
                    "tensor of rank 5"):
      nn_ops.max_pool_v2(
          array_ops.placeholder(dtypes.float32, shape=[1, 3, 3, 1, 1]),
          ksize=[1, 2, 2, 1, 1],
          strides=[1, 2, 2, 1, 1],
          padding=[[0, 0], [0, 1], [0, 1], [0, 0]],
          data_format="NCHW")
    with self.assertRaisesRegex(
        ValueError, "Attr 'padding' of 'MaxPoolV2' Op passed "
                    "string 'EXPLICIT'"):
      gen_nn_ops.max_pool_v2(
          array_ops.placeholder(dtypes.float32, shape=[1, 3, 3, 1, 1]),
          ksize=[1, 2, 2, 1, 1],
          strides=[1, 2, 2, 1, 1],
          padding="EXPLICIT",
          data_format="NHWC")

  @test_util.run_deprecated_v1
  def testEdgeCasesExcessPadding(self):
    # [FUNCTION DOC]
    # Runs the raw max pool op on a 3x3 image, 2x2 window, with explicit right padding of 2 (not
    # smaller than the window); expects an unimplemented or invalid-argument error saying right
    # padding must be smaller than the window size (or XLA's explicit padding message).
    with self.session(use_gpu=test.is_gpu_available()) as sess:
      with self.assertRaisesRegex(
          (errors_impl.UnimplementedError, errors_impl.InvalidArgumentError),
          "Right padding 2 needs to be smaller than the window size 2|"
          "XLA does not support pooling ops with explicit padding"):
        input_sizes = [1, 3, 3, 1]
        x = [(((f + 128) % 255) - 127) for f in range(9)]
        t = constant_op.constant(x, shape=input_sizes, dtype=dtypes.float32)
        sess.run(gen_nn_ops.max_pool(
            t,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="EXPLICIT",
            explicit_paddings=[0, 0, 0, 1, 0, 2, 0, 0],
            data_format="NHWC"))

  @test_util.run_deprecated_v1
  def testNegativePadding(self):
    # [FUNCTION DOC]
    # Runs the raw max pool op with explicit padding containing negative numbers; expects a value
    # error saying all elements of the explicit padding must be nonnegative.
    with self.session(use_gpu=test.is_gpu_available()) as sess:
      with self.assertRaisesRegex(
          ValueError, "All elements of explicit_paddings must be "
                      "nonnegative for"):
        input_sizes = [1, 3, 3, 1]
        x = [(((f + 128) % 255) - 127) for f in range(9)]
        t = constant_op.constant(x, shape=input_sizes, dtype=dtypes.float32)
        sess.run(gen_nn_ops.max_pool(
            t,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="EXPLICIT",
            explicit_paddings=[0, 0, -1, -1, -1, -1, 0, 0],
            data_format="NHWC"))

  @test_util.run_deprecated_v1
  def testExplicitPaddingBatch(self):
    # [FUNCTION DOC]
    # Runs the raw max pool op with nonzero explicit padding on the batch and depth dimensions;
    # expects a value error saying such padding is not supported.
    with self.session(use_gpu=test.is_gpu_available()) as sess:
      with self.assertRaisesRegex(
          ValueError, "Nonzero explicit padding in the batch or depth "
                      "dimensions is not supported"):
        input_sizes = [1, 3, 3, 1]
        x = [(((f + 128) % 255) - 127) for f in range(9)]
        t = constant_op.constant(x, shape=input_sizes, dtype=dtypes.float32)
        sess.run(gen_nn_ops.max_pool(
            t,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="EXPLICIT",
            explicit_paddings=[1, 1, 1, 1, 1, 1, 0, 0],
            data_format="NHWC"))

  @test_util.disable_xla(
      "b/205634417")  # XLA is not throwing shape errors for multiple *Grad ops.
  def testMaxPoolGradEagerShapeErrors(self):
    # [FUNCTION DOC]
    # Eager-mode test (XLA disabled: it does not raise shape errors for gradient ops). Calls the
    # raw max pool gradient and grad-of-grad ops with an original output whose depth is wrong,
    # and then with a gradient whose depth is wrong; each call must raise an invalid-argument
    # error stating the expected shape and the shape received.
    with context.eager_mode():
      orig_in = array_ops.ones((1, 1, 1, 1))

      # Test invalid orig_out shape
      orig_out = array_ops.ones((1, 1, 1, 2))
      grad = array_ops.ones((1, 1, 1, 1))
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected orig_output shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected orig_output shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")

      # Test invalid grad shape
      orig_out = array_ops.ones((1, 1, 1, 1))
      grad = array_ops.ones((1, 1, 1, 2))
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected grad shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected grad shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad_grad(
            orig_in, orig_out, grad, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")

  def testMaxPoolGradWithArgmaxEagerShapeErrors(self):
    # [FUNCTION DOC]
    # Eager-mode test. Calls the argmax-based max pool gradient op with a wrong-shaped gradient
    # and then a wrong-shaped argmax tensor; each must raise an invalid-argument error stating the
    # expected and received shape. The grad-of-grad argmax op is checked the same way only when
    # a GPU is available since it is implemented only there.
    with context.eager_mode():
      inp = array_ops.ones((1, 1, 1, 1))

      # Test invalid grad shape
      grad = array_ops.ones((1, 1, 1, 2))
      argmax = array_ops.zeros((1, 1, 1, 1), dtype=dtypes.int64)
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected grad shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad_with_argmax(
            inp, grad, argmax, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")
      # max_pool_grad_grad_with_argmax is only implemented for GPUs
      if test.is_gpu_available():
        with self.assertRaisesRegex(
            errors_impl.InvalidArgumentError,
            r"Expected grad shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
          gen_nn_ops.max_pool_grad_grad_with_argmax(
              inp, grad, argmax, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
              padding="VALID")

      # Test invalid argmax shape
      grad = array_ops.ones((1, 1, 1, 1))
      argmax = array_ops.ones((1, 1, 1, 2), dtype=dtypes.int64)
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          r"Expected argmax shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
        gen_nn_ops.max_pool_grad_with_argmax(
            inp, grad, argmax, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
            padding="VALID")
      # max_pool_grad_grad_with_argmax is only implemented for GPUs
      if test.is_gpu_available():
        with self.assertRaisesRegex(
            errors_impl.InvalidArgumentError,
            r"Expected argmax shape to be \[1,1,1,1\], but got \[1,1,1,2\]"):
          gen_nn_ops.max_pool_grad_grad_with_argmax(
              inp, grad, argmax, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1],
              padding="VALID")

  def testAvgPoolGradInvalidInputShapeRaiseError(self):
    # [FUNCTION DOC]
    # Calls the raw average pool gradient op with a hugely negative original input shape
    # (minus 536870912 in each dimension) and a mismatched gradient; expects a value error or
    # invalid-argument error instead of a crash.
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      with self.cached_session():
        orig_input_shape = constant_op.constant(
            -536870912, shape=[4], dtype=dtypes.int32)
        grad = constant_op.constant(
            .0890338004362538, shape=[1, 5, 7, 1], dtype=dtypes.float64)
        t = gen_nn_ops.AvgPoolGrad(
            orig_input_shape=orig_input_shape,
            grad=grad,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding="VALID",
            data_format="NHWC")
        self.evaluate(t)

  def testAvgPoolGradInvalidStrideRaiseErrorProperly(self):
    # [FUNCTION DOC]
    # Calls the raw average pool gradient op with a very large window and strides that are
    # inconsistent with the supplied gradient shape (float64 gradient of shape 16x16x16x16) and
    # expects an invalid-argument error rather than a crash.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      with self.cached_session():
        orig_input_shape = [11, 9, 78, 9]
        grad = constant_op.constant(
            0.1, shape=[16, 16, 16, 16], dtype=dtypes.float64)
        t = gen_nn_ops.AvgPoolGrad(
            orig_input_shape=orig_input_shape,
            grad=grad,
            ksize=[1, 40, 128, 1],
            strides=[1, 128, 128, 30],
            padding="SAME",
            data_format="NHWC")
        self.evaluate(t)


def GetMaxPoolFwdTest(input_size, filter_size, strides, padding):
  # [FUNCTION DOC]
  # HELPER: factory producing a test method that compares the forward max pool on CPU with
  # the argmax-based max pool for one shape. Inputs: input shape, window (filter) size, strides
  # and padding. Returns a function to attach to the test class; no side effects at creation.

  def Test(self):
    # [FUNCTION DOC]
    # HELPER: the generated test body (takes the test case). Returns immediately without CUDA GPU;
    # otherwise runs the forward comparison helper with the captured shape parameters.
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    self._CompareMaxPoolingFwd(input_size, filter_size, strides, padding)

  return Test


def GetMaxPoolGradTest(input_size, filter_size, output_size, strides, padding):
  # [FUNCTION DOC]
  # HELPER: factory producing a test method that compares the argmax-based max pool gradient
  # with the standard max pool gradient for one shape. Inputs: input shape, window size, output
  # shape, strides and padding. Returns a function to attach to the test class.

  def Test(self):
    # [FUNCTION DOC]
    # HELPER: the generated test body. Returns immediately without a CUDA GPU; otherwise runs the
    # gradient comparison helper with the captured parameters.
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    self._CompareMaxPoolingBk(input_size, output_size, filter_size, strides,
                              padding)

  return Test


def GetMaxPoolGradGradTest(input_size, filter_size, output_size, strides,
                           padding):
  # [FUNCTION DOC]
  # HELPER: factory producing a test method that compares the argmax-based second-order max pool
  # gradient with the standard one for one shape. Inputs: input shape, window size, output shape,
  # strides and padding. Returns a function to attach to the test class.

  def Test(self):
    # [FUNCTION DOC]
    # HELPER: the generated test body. Returns immediately without a CUDA GPU; otherwise runs the
    # second-order gradient comparison helper with the captured parameters.
    # MaxPoolWithArgMax is implemented only on CUDA.
    if not test.is_gpu_available(cuda_only=True):
      return
    self._CompareMaxPoolingGradBk(input_size, output_size, filter_size, strides,
                                  padding)

  return Test


if __name__ == "__main__":
  for (name_, input_size_, filter_size_, output_size_, stride_,
       padding_) in GetShrunkInceptionMaxPoolShapes():
    setattr(PoolingTest, "testMaxPoolFwd_" + name_,
            GetMaxPoolFwdTest(input_size_, filter_size_, stride_, padding_))
    setattr(PoolingTest, "testMaxPoolGrad_" + name_,
            GetMaxPoolGradTest(input_size_, filter_size_, output_size_, stride_,
                               padding_))
    setattr(PoolingTest, "testMaxPoolGradGrad_" + name_,
            GetMaxPoolGradGradTest(input_size_, filter_size_, output_size_,
                                   stride_, padding_))
  test.main()
