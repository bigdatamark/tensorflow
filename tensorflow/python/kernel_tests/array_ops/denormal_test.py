# [MODULE DOC]
# Tests that TensorFlow ops flush denormal (subnormal) floating point values to zero while ordinary
# numpy code still handles denormals correctly. Multiplies the smallest normal float by a small
# factor through a constant op and expects zeros, for a small shape (main thread) and a large shape
# (thread pool), then verifies the processor flags did not leak into numpy.
# Import-time effect: none. When run as a program it prepends a flag enabling flush-to-zero in the
# XLA GPU backend to the XLA flags environment variable before starting the tests.
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
"""Tests for denormal handling."""

import os
import platform

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
class DenormalTest(test.TestCase):
  # [CLASS DOC]
  # Test case for denormal flushing behavior; decorated to run ops as functions in eager mode. Holds
  # a numpy sanity check and a shared helper run for CPU and GPU variants.

  def testPythonHasDenormals(self):
    # [FUNCTION DOC]
    # Tests that numpy itself keeps denormals: for 32-bit and 64-bit floats, the smallest normal value
    # divided by sixteen and multiplied back equals the original. Also reused as a leak check by the
    # flush test.
    """Non-tf numpy code should treat denormals correctly."""
    for dtype in np.float32, np.float64:
      tiny = np.finfo(dtype).tiny
      self.assertEqual(tiny, tiny / 16 * 16)

  def _flushDenormalsTest(self, dtypes):
    # [FUNCTION DOC]
    # HELPER: shared body of the CPU and GPU flush tests.
    # KNOWN ISSUE: on ppc64le, s390x and aarch64 machines it returns immediately, so the test
    # passes without checking anything there (disabled per a linked issue).
    # Takes dtypes, a collection of numpy float types.
    # For each dtype, scales the smallest normal value by a tenth through a constant op for a scalar
    # and a large vector shape, expects all zeros, then runs the numpy denormal check to make sure
    # flags did not leak. Returns nothing.
    if (platform.machine() == "ppc64le" or platform.machine() == "s390x" or
        platform.machine() == "aarch64"):
      # Disabled denormal_test on power/s390x/aarch64 platform
      # Check relevant discussion -
      # https://github.com/tensorflow/tensorflow/issues/11902
      return
    for dtype in dtypes:
      tiny = np.finfo(dtype).tiny
      # Small shape to test main thread, large shape to test thread pool
      for shape in (), (1 << 20,):
        flush = 0.1 * constant_op.constant(tiny, shape=shape)
        self.assertAllEqual(self.evaluate(flush), np.zeros(shape))
        # Make sure the flags don't leak out
        self.testPythonHasDenormals()

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testFlushDenormalsCPU(self):
    # [FUNCTION DOC]
    # Tests flushing on CPU in graph and eager modes (without GPU): both single and double precision
    # must flush, since the processor flags apply to both.
    # On CPUs, the processor flags flush for both single and double precision.
    self._flushDenormalsTest(dtypes=(np.float32, np.float64))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testFlushDenormalsGPU(self):
    # [FUNCTION DOC]
    # Tests flushing with GPU allowed in graph and eager modes: only single precision is checked since
    # only that can flush to zero on GPUs.
    # On GPUs, only single precision can flush to zero.
    self._flushDenormalsTest(dtypes=(np.float32,))


if __name__ == "__main__":
  # When eager_op_as_function mode is enabled xla auto-clustering kicks in.
  # By default xla does not enable flush-to-zero semantics in the GPU backend.
  # This env flag has to be set before the test is setup. Setting it using the
  # decorator does not seem to propagate the flag to all required locations.
  original_xla_flags = os.environ.get("XLA_FLAGS")
  new_xla_flags = "--xla_gpu_ftz=true"
  if original_xla_flags:
    new_xla_flags = new_xla_flags + " " + original_xla_flags
  os.environ["XLA_FLAGS"] = new_xla_flags

  test.main()
