# [MODULE DOC]
# Compilation test for TPUStrategy (the version 2 strategy) that checks distinct functions with the
# same signature compile independently. Defines the command-line flags for TPU name, GCP project
# and GCP zone at import time. Meant to be merged into the main TPUStrategy tests once the MLIR
# bridge is enabled by default. Requires a real TPU; runs the test main entry point when executed.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TPUStrategy in regards to compiling programs."""

from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


def get_tpu_cluster_resolver():
  # [FUNCTION DOC]
  # HELPER: builds the TPU cluster resolver for the compilation test.
  # Creates a TPU cluster resolver from the command-line flags for TPU name, zone and project.
  # Takes no arguments and returns the resolver. No connection is made here.
  resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver


def get_tpu_strategy():
  # [FUNCTION DOC]
  # HELPER: sets up a TPU strategy for the compilation test.
  # Takes no arguments. Builds a resolver from the flags, connects the eager context to the remote
  # cluster, initializes the TPU system, and returns a new TPUStrategyV2 built on the resolver.
  # Side effects: global eager context is connected to the cluster and the TPU system initialized.
  resolver = get_tpu_cluster_resolver()
  remote.connect_to_cluster(resolver)
  tpu_cluster_resolver.initialize_tpu_system(resolver)
  strategy = tpu_lib.TPUStrategyV2(resolver)
  return strategy


# TODO(b/158494076): Merge this test back into TPUStrategy tests
# (tpu_strategy_test) once MLIR bridge is enabled by default.
class TPUStrategyCompilationTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering how TPUStrategy compiles programs. Needs a live TPU configured by flags.
  # Has no shared fixtures; each test creates its own strategy.

  def test_functions_compile_same_signature(self):
    # [FUNCTION DOC]
    # Scenario: two separate traced functions each run a different computation (returning constant
    # one, then constant two) through the strategy with identical signatures. After building the
    # strategy, each result is converted to a per-replica list of local results. Asserts that every
    # replica returned one for the first function and two for the second, i.e. the compiled
    # programs do not get mixed up.
    """Tests compiling different functions with the same signature."""
    strategy = get_tpu_strategy()

    @def_function.function
    def return_one():
      # [FUNCTION DOC]
      # Outer traced function; runs the inner computation on all replicas and returns the
      # per-replica result of returning constant one.

      def computation():
        # [FUNCTION DOC]
        # Replica computation returning the constant integer one.
        return constant_op.constant(1)

      return strategy.run(computation)

    @def_function.function
    def return_two():
      # [FUNCTION DOC]
      # Outer traced function; runs the inner computation on all replicas and returns the
      # per-replica result of returning constant two.

      def computation():
        # [FUNCTION DOC]
        # Replica computation returning the constant integer two.
        return constant_op.constant(2)

      return strategy.run(computation)

    expected_result_ones = [1 for _ in range(0, strategy.num_replicas_in_sync)]
    self.assertAllEqual(expected_result_ones,
                        strategy.experimental_local_results(return_one()))

    expected_result_twos = [2 for _ in range(0, strategy.num_replicas_in_sync)]
    self.assertAllEqual(expected_result_twos,
                        strategy.experimental_local_results(return_two()))


if __name__ == "__main__":
  test.main()
