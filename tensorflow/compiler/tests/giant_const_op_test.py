# [MODULE DOC]
# TPU test verifying that graphs containing giant constant tensors (too large for memory) are
# compiled correctly by XLA. Defines command-line flags for the TPU name, cloud project and
# zone at import time, helper functions to create a cluster resolver and TPU strategy, and one
# test class. When run as a program it sets an environment variable that disables XLA constant
# folding (asserting it was not already set) before running the tests.
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
"""Tests for giant const op compilation."""

import os
import numpy as np

from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


def get_tpu_cluster_resolver():
  # [FUNCTION DOC]
  # HELPER: builds the TPU cluster resolver for the test strategy.
  # Takes no arguments; reads the TPU name, zone and project flags and returns a resolver
  # created from them.
  resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver


def get_tpu_strategy():
  # [FUNCTION DOC]
  # HELPER: creates the TPU distribution strategy used by the test.
  # Gets the resolver, connects to the remote cluster, initializes the TPU system (side
  # effects on the runtime) and returns a TPU strategy (version 2) built from the resolver.
  resolver = get_tpu_cluster_resolver()
  remote.connect_to_cluster(resolver)
  tpu_cluster_resolver.initialize_tpu_system(resolver)
  return tpu_lib.TPUStrategyV2(resolver)


# This test doesn't use XLATestCase like the other tests in this directory.
# The Const op xla op kernel is compilation only and therefore is not executed
# with XLA in the on demand compilation mode. Also, here we want to feed the
# full program to XLA to verify handling of programs with giant constant
# tensors.
class GiantConstOp(test.TestCase):
  # [CLASS DOC]
  # Eager-mode test case (not the XLA test base class) that runs a program with a giant
  # constant on a TPU through the full XLA compilation path. Stateless.

  # Verifies that graphs containing giant const tensors that won't fit in memory
  # are compiled correctly to HLO.
  def testGiantConst(self):
    # [FUNCTION DOC]
    # Disables the MLIR bridge, creates the TPU strategy and, for many data types (bool,
    # signed and unsigned integers, half, bfloat16, float32, float64) and test values (one,
    # plus false for bool, plus type min and max except for float64), runs a compiled function
    # that builds a constant of shape 1024 to the fourth power and returns a one-element slice.
    # Asserts the first local result equals a one-element array filled with the value.
    # Disabling Mlir bridge since using the tf2xla implementation of
    # StridedSliceop which would get executed in this GiantConst test.
    config.disable_mlir_bridge()
    strategy = get_tpu_strategy()
    types = {
        dtypes.bool,
        dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64,
        dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64,
        dtypes.float16, dtypes.bfloat16,
        dtypes.float32, dtypes.float64,
    }
    for dtype in types:
      values = [True if dtype is dtypes.bool else 1]

      if dtype is dtypes.bool:
        values.append(False)
      elif dtype is not dtypes.float64:
        # TPUs don't follow IEEE 754 float64 standard for 64 bit floating point
        # numbers so it could return different output even with just data
        # transformation ops without any arithmetic operations.
        values.extend([dtype.min, dtype.max])

      for value in values:

        @def_function.function
        def train_step():
          # [FUNCTION DOC]
          # Compiled function that runs the inner computation on every replica of the strategy
          # and returns the per-replica results.

          # pylint: disable=cell-var-from-loop
          def computation():
            # [FUNCTION DOC]
            # Builds a giant constant of the loop's dtype and value with shape 1024 in each of
            # four dimensions and returns only its first element as a 1x1x1x1 slice.
            const = constant_op.constant(value, dtype=dtype, shape=[1024]*4)
            return const[:1, :1, :1, :1]

          return strategy.run(computation, args=())

        output = strategy.experimental_local_results(train_step())[0]
        expected = np.full((1, 1, 1, 1), value)
        self.assertAllEqual(output, expected)

if __name__ == "__main__":
  # Make sure TF_XLA_FLAGS is not already set to avoid dropping the existing
  # value silently.
  assert "TF_XLA_FLAGS" not in os.environ

  # Disable tfxla constant folding that always creates full Tensors and will
  # fail for giant tensors.
  os.environ["TF_XLA_FLAGS"] = "--tf_xla_disable_constant_folding=true"

  test.main()
