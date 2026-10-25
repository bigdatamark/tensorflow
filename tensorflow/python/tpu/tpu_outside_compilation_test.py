# [MODULE DOC]
# Hardware tests for TPU outside compilation: running selected host-side computations (printing,
# summaries, string ops, control flow, gradients) from inside a TPU computation, both when
# explicitly requested and when automatically extracted because an op is unsupported on TPU.
# Requires a real TPU. Import-time side effects: defines command-line flags for choosing the TPU
# (local TPU use, TPU name, cloud project and zone) and detects whether the tensorboard summary
# libraries are importable, setting a module flag used to skip summary tests. Also contains
# small helper functions to connect to the TPU, read summary events, and wrap functions for
# TPU rewrite / partitioned-call execution.
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
"""Tests for TPU outside compilation."""

import os
import tempfile
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.core.util import event_pb2
from tensorflow.python.distribute import tpu_strategy as tpu_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile
from tensorflow.python.tpu import functional as tpu_functional
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu import tpu_replication
from tensorflow.python.tpu.ops import tpu_ops
try:
  from tensorboard.plugins.histogram import summary_v2 as histogram_summary_v2  # pylint: disable=g-import-not-at-top
  from tensorboard.plugins.image import summary_v2 as image_summary_v2  # pylint: disable=g-import-not-at-top
  from tensorboard.plugins.scalar import summary_v2 as scalar_summary_v2  # pylint: disable=g-import-not-at-top
  _TENSORBOARD_AVAILABLE = True
except ImportError:
  _TENSORBOARD_AVAILABLE = False

FLAGS = flags.FLAGS
flags.DEFINE_bool(
    "use_local_tpu",
    False,
    "use local TPUs on a TPU VM instead of connecting to a GCP TPU VM or node.",
)
flags.DEFINE_string("tpu", "", "Name of TPU to connect to.")
flags.DEFINE_string("project", None, "Name of GCP project with TPU.")
flags.DEFINE_string("zone", None, "Name of GCP zone with TPU.")


def get_tpu_cluster_resolver():
  # [FUNCTION DOC]
  # HELPER: test utility returning a TPU cluster resolver chosen by the command-line flags.
  # If the local-TPU flag is set, resolves the local TPU; otherwise uses the TPU name, zone and
  # project flags. Returns the resolver; no other side effects.
  if FLAGS.use_local_tpu:
    return tpu_cluster_resolver.TPUClusterResolver("local")
  resolver = tpu_cluster_resolver.TPUClusterResolver(
      tpu=FLAGS.tpu,
      zone=FLAGS.zone,
      project=FLAGS.project,
  )
  return resolver


def get_tpu_strategy():
  # [FUNCTION DOC]
  # HELPER: test utility that connects to the TPU cluster and returns a TPU distribution strategy.
  # Side effects: connects the eager runtime to the remote cluster and initialises the TPU system.
  # Returns the strategy object used by the tests to run computations on replicas.
  resolver = get_tpu_cluster_resolver()
  remote.connect_to_cluster(resolver)
  tpu_cluster_resolver.initialize_tpu_system(resolver)
  return tpu_lib.TPUStrategyV2(resolver)


def computation_with_string_ops(x):
  # [FUNCTION DOC]
  # HELPER: sample computation using string ops, which are unsupported on TPU and so must run on
  # the host. Formats the input into a string with a leading "1" and converts it back to a
  # number; for input 0 the result is 10. Input: a scalar tensor; returns a numeric tensor.
  output = string_ops.string_format("1{}", x)
  return string_ops.string_to_number(output)


def _events_from_logdir(test_case, logdir):
  # [FUNCTION DOC]
  # HELPER: test utility that reads summary events written to a log directory.
  # Inputs: test_case, the running test (used for assertions); logdir, directory path. Asserts
  # the directory exists and contains exactly one file, then parses every record of that file
  # into an event message. Returns the list of events in file order.
  """Reads summary events from log directory."""
  test_case.assertTrue(gfile.Exists(logdir))
  files = gfile.ListDirectory(logdir)
  test_case.assertLen(files, 1)
  records = list(tf_record.tf_record_iterator(os.path.join(logdir, files[0])))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def _rewrite_func_wrapper(tf_func):
  # [FUNCTION DOC]
  # HELPER: wraps a traced function so that calling it runs the function through the TPU rewrite
  # step. Input: tf_func, a graph function. Returns a new graph function that, when called,
  # builds a concrete function and rewrites it for TPU execution.

  def tpu_fn(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: body of the wrapper. Flattens keyword arguments after positional ones (the rewrite
    # accepts only a list of tensors), obtains the concrete function and returns the result of
    # rewriting it for TPU with those arguments.
    # tpu.rewrite only accepts list of tensors as input. We need to flatten
    # keyword arguments to meet this requirement.
    concrete = tf_func.get_concrete_function(*(list(args) +
                                               list(kwargs.values())))
    return tpu.rewrite(concrete.__call__, list(args) + list(kwargs.values()))

  return def_function.function(tpu_fn)


def _tpu_partitioned_call_wrapper(tf_func):
  # [FUNCTION DOC]
  # HELPER: wraps a graph function so that it is invoked through the TPU partitioned-call op.
  # Input: tf_func, a graph function. Returns a new graph function that performs the call.
  """Wrap a tensorflow Function with TPUPartitionedCall."""

  def inner_func(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: body of the wrapper. Gets the concrete function, builds the argument list as
    # positional arguments, flattened keyword arguments and captured inputs, and invokes the TPU
    # partitioned-call op with a device ordinal chosen by the ordinal selector op and output types
    # taken from the function signature. Returns the call outputs.
    concrete = tf_func.get_concrete_function(*args, **kwargs)
    # TPUPartitionedCall only accepts list of tensors as input args.
    # Flatten keyword arguments and do some basic ordering:
    # Positional args + Flattened keyword args + Captured args.
    op_args = list(args) + list(kwargs.values()) + concrete.captured_inputs
    return tpu_functional.TPUPartitionedCall(
        args=op_args,
        device_ordinal=tpu_ops.tpu_ordinal_selector(),
        Tout=[o.type for o in concrete.function_def.signature.output_arg],
        f=concrete)

  return def_function.function(inner_func)


class TpuOutsideCompilationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests explicit outside compilation on TPU: host-side functions marked by the test are
  # extracted from the TPU computation and run on the host. Covers no/some/multiple inputs and
  # outputs, several clusters, control flow, summaries, head/tail placement and gradients.
  # Fixture: setUp disables soft device placement. Requires a TPU strategy from the flags.

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the base setup, then turns off soft device placement so ops that cannot run on TPU
    # fail instead of silently moving.
    super(TpuOutsideCompilationTest, self).setUp()
    config.set_soft_device_placement(False)

  def testHostNoInput(self):
    # [FUNCTION DOC]
    # Test: outside compile a host function with no inputs that only prints. The TPU computation
    # adds 5, calls the host function, adds 5 again. Expects 35 on every replica from input 25.
    strategy = get_tpu_strategy()

    def outside_fn():
      # [FUNCTION DOC]
      # Host-side function with no inputs; prints a message. Returns nothing.
      logging_ops.print_v2("Outside compiled")

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the TPU function on all replicas with input 25 and returns the
      # per-replica results.

      def tpu_fn(x):
        # [FUNCTION DOC]
        # Replica computation: adds 5 to x, triggers outside compilation of the host function,
        # then adds 5 again and returns the sum.
        x2 = x + 5.0
        tpu_replication.outside_compilation(outside_fn)
        return x2 + 5.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(35., shape=(strategy.num_replicas_in_sync)))

  def testHostInputOnly(self):
    # [FUNCTION DOC]
    # Test: outside compile a host function that takes one input and only prints it (no output).
    # Expects 35 on every replica from input 25.
    strategy = get_tpu_strategy()

    def outside_fn(x):
      # [FUNCTION DOC]
      # Host-side function; prints its input. Returns nothing.
      logging_ops.print_v2("Outside compiled", x)

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the TPU function on all replicas with input 25 and returns the
      # per-replica results.

      def tpu_fn(x):
        # [FUNCTION DOC]
        # Replica computation: adds 5, sends the value to the host function, adds 5 again and
        # returns the result.
        x2 = x + 5.0
        tpu_replication.outside_compilation(outside_fn, x2)
        return x2 + 5.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(35., shape=(strategy.num_replicas_in_sync)))

  def testJitCompile(self):
    # [FUNCTION DOC]
    # Test: same scenario as the input-only test, but the outer graph function requests jit
    # compilation, which should have no effect on TPU. Expects 35 on every replica.
    strategy = get_tpu_strategy()

    def outside_fn(x):
      # [FUNCTION DOC]
      # Host-side function; prints its input. Returns nothing.
      logging_ops.print_v2("Outside compiled", x)

    # jit_compile=True should have no effect for TPU.
    @def_function.function(jit_compile=True)
    def train_step():
      # [FUNCTION DOC]
      # Jit-compiled graph function that runs the TPU function on all replicas with input 25.

      def tpu_fn(x):
        # [FUNCTION DOC]
        # Replica computation: adds 5, sends the value to the host function, adds 5 again and
        # returns the result.
        x2 = x + 5.0
        tpu_replication.outside_compilation(outside_fn, x2)
        return x2 + 5.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(35., shape=(strategy.num_replicas_in_sync)))

  def testHostInputOutput(self):
    # [FUNCTION DOC]
    # Test: outside compile a host function with one input and one output (prints, adds 6) and use
    # its output as the TPU result. Input 25 gives 25 + 5 + 6 = 36 on every replica.
    strategy = get_tpu_strategy()

    def outside_fn(x):
      # [FUNCTION DOC]
      # Host-side function; prints its input and returns it plus 6.
      logging_ops.print_v2("Outside compiled", x)
      return x + 6.0

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the TPU function on all replicas with input 25.

      def tpu_fn(x):
        # [FUNCTION DOC]
        # Replica computation: adds 5, passes the value through the host function and returns the
        # host function output.
        x2 = x + 5.0
        output = tpu_replication.outside_compilation(outside_fn, x2)
        return output

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(36., shape=(strategy.num_replicas_in_sync)))

  def testHostMultipleInputs(self):
    # [FUNCTION DOC]
    # Test: outside compile a host function with two inputs of different shapes (2x3 and 3x2)
    # returning three outputs of different shapes. The TPU function reduces them to one scalar;
    # expects 213 on every replica.
    strategy = get_tpu_strategy()
    val0 = np.arange(6).reshape((2, 3)).astype(np.float32)
    val1 = np.arange(6).reshape((3, 2)).astype(np.float32)

    def outside_fn(arg0, arg1):
      # [FUNCTION DOC]
      # Host-side function. Reshapes the second input to the first input's shape and returns
      # three tensors: their sum, the matrix product of the inputs, and the first input
      # concatenated with the reshaped one along the first axis.
      tmp = array_ops.reshape(arg1, array_ops.shape(arg0))
      ret0 = arg0 + tmp
      ret1 = math_ops.matmul(arg0, arg1)
      ret2 = array_ops.concat([arg0, tmp], 0)
      return ret0, ret1, ret2

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the TPU function on all replicas with the two numeric arrays.

      def tpu_fn(x, y):
        # [FUNCTION DOC]
        # Replica computation: adds 7 to x, doubles y, runs the host function on both, then
        # returns max of the first output plus min of the second plus sum of the third.
        a = x + 7.0
        b = y * 2.0
        c, d, e = tpu_replication.outside_compilation(outside_fn, a, b)
        return (math_ops.reduce_max(c) + math_ops.reduce_min(d) +
                math_ops.reduce_sum(e))

      return strategy.run(tpu_fn, args=(val0, val1))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(213., shape=(strategy.num_replicas_in_sync)))

  def testMultipleClusters(self):
    # [FUNCTION DOC]
    # Test: two separate outside-compiled host functions in sequence inside one TPU computation.
    # Input 25 goes: +5, host +6, +3, host -18, giving 21 on every replica.
    strategy = get_tpu_strategy()

    def outside_fn1(x):
      # [FUNCTION DOC]
      # First host function; prints its input and returns it plus 6.
      logging_ops.print_v2("Outside compiled", x)
      return x + 6.0

    def outside_fn2(x):
      # [FUNCTION DOC]
      # Second host function; prints its input and returns it minus 18.
      logging_ops.print_v2("Outside compiled", x)
      return x - 18.0

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the TPU function on all replicas with input 25.

      def tpu_fn(x):
        # [FUNCTION DOC]
        # Replica computation: adds 5, runs the first host function, adds 3, runs the second host
        # function and returns its output.
        x2 = x + 5.0
        output1 = tpu_replication.outside_compilation(outside_fn1, x2)
        x3 = output1 + 3.0
        output2 = tpu_replication.outside_compilation(outside_fn2, x3)
        return output2

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(21., shape=(strategy.num_replicas_in_sync)))

  @parameterized.parameters((True), (False))
  def testOutsideCompilationControlFlowIf(self, take_true_branch):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the parameter name is misleading. When it is true the input is 51, so the
    # condition (input below 50) is false and the host-free else path runs; the host function
    # runs when it is false.
    # Parameterized test (true or false): outside compilation inside a conditional of the TPU
    # computation. Input 51 expects 56 on every replica; input 25 expects 36.
    strategy = get_tpu_strategy()

    def outside_fn(x):
      # [FUNCTION DOC]
      # Host function; prints its input and returns it plus 6.
      logging_ops.print_v2("Outside compiled", x)
      return x + 6.0

    input_value = 51.0 if take_true_branch else 25.0

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the TPU function on all replicas with the selected input.

      def tpu_fn(x):
        # [FUNCTION DOC]
        # Replica computation: adds 5; if the original input is below 50 the host function is
        # applied to that value, otherwise the value is returned unchanged.
        x2 = x + 5.0
        if x < 50.0:
          return tpu_replication.outside_compilation(outside_fn, x2)
        else:
          return x2

      return strategy.run(tpu_fn, args=(input_value,))

    output_value = 36.0
    if take_true_branch:
      output_value = 56.0
    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(
            output_value, shape=(strategy.num_replicas_in_sync)))

  def testOutsideCompilationControlFlowWhile(self):
    # [FUNCTION DOC]
    # Test: outside compilation inside a while loop of the TPU computation. Starting at 25 + 5 the
    # host function adds 6 until the value reaches 50, then 4 is added; expects 58 per replica.
    strategy = get_tpu_strategy()

    def outside_fn(x):
      # [FUNCTION DOC]
      # Host function; prints its input and returns it plus 6.
      logging_ops.print_v2("Outside compiled", x)
      return x + 6.0

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the TPU function on all replicas with input 25.

      def tpu_fn(x):
        # [FUNCTION DOC]
        # Replica computation: adds 5, loops while the value is below 50 applying the host
        # function each time, then adds 4 and returns.
        x2 = x + 5.0
        while x2 < 50.0:
          x2 = tpu_replication.outside_compilation(outside_fn, x2)
        return x2 + 4.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(58., shape=(strategy.num_replicas_in_sync)))

  def testOutsideCompilationHostControlFlow(self):
    # [FUNCTION DOC]
    # Test: the host function itself contains a loop (four iterations adding 6 each). Input 25
    # yields 25 + 5 + 24 + 4 = 58 on every replica.
    """Tests that control flow on host for outside_compilation works."""
    strategy = get_tpu_strategy()

    def outside_fn(x):
      # [FUNCTION DOC]
      # Host function with a loop; adds 6 to the input four times and returns the total.
      n = 0
      while n < 4:
        x = x + 6.0
        n = n + 1
      return x

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the TPU function on all replicas with input 25.

      def tpu_fn(x):
        # [FUNCTION DOC]
        # Replica computation: adds 5, applies the host function, adds 4 and returns.
        x2 = x + 5.0
        x2 = tpu_replication.outside_compilation(outside_fn, x2)
        return x2 + 4.0

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(58., shape=(strategy.num_replicas_in_sync)))

  @unittest.skipIf(not _TENSORBOARD_AVAILABLE, "Tensorboard is not installed.")
  def testSummary(self):
    # [FUNCTION DOC]
    # Test (skipped without tensorboard): a scalar summary written inside an outside-compiled host
    # function, called twice in one TPU computation, under a file writer that always records.
    # Expects (2 + 1) * 2 + 1 = 7 on every replica.
    strategy = get_tpu_strategy()

    def host_computation(x):
      # [FUNCTION DOC]
      # Host function; writes a scalar summary named x at step 0 and returns the input doubled.
      scalar_summary_v2.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():
      # [FUNCTION DOC]
      # Graph function that runs the replica computation on all replicas with input 2.

      def computation(x):
        # [FUNCTION DOC]
        # Replica computation: adds 1, calls the host function twice on the same value (the
        # second result is kept), adds 1 and returns.
        x = x + 1.0
        y = tpu_replication.outside_compilation(host_computation, x)
        y = tpu_replication.outside_compilation(host_computation, x)
        return y + 1.0

      return strategy.run(computation, args=(2.0,))

    summary_writer = summary.create_file_writer(
        os.path.join(os.getenv("TEST_TMPDIR", "/tmp")), flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(7., shape=(strategy.num_replicas_in_sync)))

  @parameterized.parameters((True), (False))
  @unittest.skipIf(not _TENSORBOARD_AVAILABLE, "Tensorboard is not installed.")
  def testSummaryInCond(self, take_true_branch):
    # [FUNCTION DOC]
    # Parameterized test (skipped without tensorboard): summaries written by outside-compiled
    # host calls inside a conditional of the TPU computation. When take_true_branch is true the
    # input is 2 (condition holds, host path runs, expects 7); otherwise input 10 (expects 12).
    strategy = get_tpu_strategy()

    def host_computation(x):
      # [FUNCTION DOC]
      # Host function; writes a scalar summary named x at step 0 and returns the input doubled.
      scalar_summary_v2.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step(take_true_branch):
      # [FUNCTION DOC]
      # Graph function; runs the replica computation with input 2 or 10 depending on the flag.

      def computation(x):
        # [FUNCTION DOC]
        # Replica computation: adds 1; if the value is below 5, calls the host function twice
        # and keeps the second result; then adds 1 and returns.
        x = x + 1.0
        if x < 5.0:
          y = tpu_replication.outside_compilation(host_computation, x)
          y = tpu_replication.outside_compilation(host_computation, x)
          x = y
        return x + 1.0

      if take_true_branch:
        return strategy.run(computation, args=(2.0,))
      else:
        return strategy.run(computation, args=(10.0,))

    summary_writer = summary.create_file_writer(
        os.path.join(os.getenv("TEST_TMPDIR", "/tmp")), flush_millis=10000)

    output_value = 12.
    if take_true_branch:
      output_value = 7.
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step(take_true_branch)),
          constant_op.constant(
              output_value, shape=(strategy.num_replicas_in_sync)))

  @unittest.skipIf(not _TENSORBOARD_AVAILABLE, "Tensorboard is not installed.")
  def testSummaryInWhile(self):
    # [FUNCTION DOC]
    # Test (skipped without tensorboard): summaries written by outside-compiled host calls inside
    # a three-iteration loop of the TPU computation. Expects 31 on every replica from input 2.
    strategy = get_tpu_strategy()

    def host_computation(x):
      # [FUNCTION DOC]
      # Host function; writes a scalar summary named x at step 0 and returns the input doubled.
      scalar_summary_v2.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():
      # [FUNCTION DOC]
      # Graph function that runs the replica computation on all replicas with input 2.

      def computation(x):
        # [FUNCTION DOC]
        # Replica computation: three times, adds 1 and applies the host function twice (keeping
        # the second result); finally adds 1 and returns.
        n = 0
        while n < 3:
          x = x + 1.0
          y = tpu_replication.outside_compilation(host_computation, x)
          y = tpu_replication.outside_compilation(host_computation, x)
          x = y
          n = n + 1
        return x + 1.0

      return strategy.run(computation, args=(2.0,))

    summary_writer = summary.create_file_writer(
        os.path.join(os.getenv("TEST_TMPDIR", "/tmp")), flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(31., shape=(strategy.num_replicas_in_sync)))

  def testOutsideCompilationAtHeadAndTail(self):
    # [FUNCTION DOC]
    # Test: outside compilation applied directly to the computation input (head) and to a
    # value that feeds the final op (tail). Input 2: doubled, +1, doubled, +5 gives 15.
    """Tests that outside_compilation at head/tail of TPU computation works."""
    strategy = get_tpu_strategy()

    def host_computation(x):
      # [FUNCTION DOC]
      # Host function; returns the input doubled.
      return x * 2.0

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the replica computation on all replicas with input 2.

      def computation(x):
        # [FUNCTION DOC]
        # Replica computation: host-doubles the input, adds 1, host-doubles again, adds 5.
        w = tpu_replication.outside_compilation(host_computation, x)
        y = w + 1.0
        z = tpu_replication.outside_compilation(host_computation, y)
        return z + 5.0

      return strategy.run(computation, args=(2.0,))
    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(15., shape=(strategy.num_replicas_in_sync)))

  def testGradientAcrossOutsideCompilation(self):
    # [FUNCTION DOC]
    # Test: the gradient of a value computed through an outside-compiled host function (fourth
    # power) is itself computed in the TPU computation. With x=2, y=3 the derivative of
    # y * (x+7)^4 with respect to x is 4 * 3 * 9^3 = 8748, expected on every replica.
    """Tests compiled gradients can contain host computations."""
    strategy = get_tpu_strategy()

    def host_computation(a):
      # [FUNCTION DOC]
      # Host function; returns the fourth power of the input (squares it twice).
      b = a * a
      c = b * b
      return c

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the replica computation on all replicas with inputs 2 and 3.
      def computation(x, y):
        # [FUNCTION DOC]
        # Replica computation: applies the host function to x plus 7, multiplies by y and returns
        # the gradient of that product with respect to x, colocating gradient ops with forward ops.
        a = x + 7.0
        b = tpu_replication.outside_compilation(host_computation, a)
        c = b * y
        d = gradients_impl.gradients(
            [c], [x], colocate_gradients_with_ops=True)[0]
        return d

      return strategy.run(computation, args=(2.0, 3.0))
    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(8748., shape=(strategy.num_replicas_in_sync)))

  def testGradientOfGradientAcrossOutsideCompilation(self):
    # [FUNCTION DOC]
    # Test: like the single-gradient test but takes the gradient of the gradient (second
    # derivative of y * (x+7)^4 is 12 * 3 * 9^2 = 2916 for x=2, y=3), expected on every replica.
    """Tests compiled gradients of gradients can contain host computations."""
    strategy = get_tpu_strategy()

    def host_computation(a):
      # [FUNCTION DOC]
      # Host function; returns the fourth power of the input (squares it twice).
      b = a * a
      c = b * b
      return c

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the replica computation on all replicas with inputs 2 and 3.
      def computation(x, y):
        # [FUNCTION DOC]
        # Replica computation: applies the host function to x plus 7, multiplies by y, takes the
        # gradient with respect to x and then the gradient of that gradient; returns the latter.
        a = x + 7.0
        b = tpu_replication.outside_compilation(host_computation, a)
        c = b * y
        d = gradients_impl.gradients(
            [c], [x], colocate_gradients_with_ops=True)[0]
        e = gradients_impl.gradients(
            [d], [x], colocate_gradients_with_ops=True)[0]
        return e

      return strategy.run(computation, args=(2.0, 3.0))
    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(2916., shape=(strategy.num_replicas_in_sync)))

  def testColocateGradientWithOutsideCompiledOp(self):
    # [FUNCTION DOC]
    # Test: the gradient op of an outside-compiled square root must carry the outside compilation
    # attribute of the forward op, extended with a gradients suffix. Expects the gradient 0.1 at
    # input 25 on every replica.
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the nested TPU function on all replicas with input 25.

      @def_function.function
      def tpu_fn(x):
        # [FUNCTION DOC]
        # Replica computation: outside-compiles the square root, takes the gradient with
        # colocation, then asserts (plain assertions at trace time) that the forward op has
        # outside-compilation attribute zero and the gradient op the gradients-suffixed one.
        # Returns the gradient.
        x1 = tpu_replication.outside_compilation(math_ops.sqrt, x)
        grad = gradients_impl.gradients([x1], [x],
                                        colocate_gradients_with_ops=True)[0]
        sqrt = [
            op for op in ops.get_default_graph().get_operations()
            if op.type == "Sqrt"
        ][0]
        sqrt_grad = [
            op for op in ops.get_default_graph().get_operations()
            if op.type == "SqrtGrad"
        ][0]
        assert sqrt.get_attr(
            tpu_replication._OUTSIDE_COMPILATION_ATTR) == b"0"
        assert (sqrt_grad.get_attr(
            tpu_replication._OUTSIDE_COMPILATION_ATTR) == b"0.gradients/uid"
               )
        return grad

      return strategy.run(tpu_fn, args=(25.0,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step()),
        constant_op.constant(.1, shape=(strategy.num_replicas_in_sync)))


class OutsideCompilationOnUnsupportedOpTest(test.TestCase,
                                            parameterized.TestCase):
  # [CLASS DOC]
  # Tests that ops unsupported on TPU (strings, summaries, random ops, functional control flow)
  # run correctly on the host, both through manual outside compilation and through automatic
  # extraction. Fixture: setUp enables soft device placement. Requires a TPU strategy.

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the base setup, then turns soft device placement on so unsupported ops may move to
    # the host.
    super(OutsideCompilationOnUnsupportedOpTest, self).setUp()
    config.set_soft_device_placement(True)

  def testStringOpWithManualOutsideCompilation(self):
    # [FUNCTION DOC]
    # Test: a string-formatting computation explicitly wrapped in outside compilation. Input 0
    # gives 10 on every replica.
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step(x):
      # [FUNCTION DOC]
      # Graph function that runs the replica computation on all replicas with argument x.

      def computation(x):
        # [FUNCTION DOC]
        # Replica computation: runs the string-op function through explicit outside compilation.
        return tpu_replication.outside_compilation(
            computation_with_string_ops, x)

      return strategy.run(computation, args=(x,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step(0)),
        constant_op.constant(10, shape=(strategy.num_replicas_in_sync)))

  def testStringOpWithAutoOutsideCompilation(self):
    # [FUNCTION DOC]
    # Test: the same string computation called directly, relying on automatic outside compilation
    # of the unsupported ops. Input 0 gives 10 on every replica.
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step(x):
      # [FUNCTION DOC]
      # Graph function that runs the replica computation on all replicas with argument x.

      def computation(x):
        # [FUNCTION DOC]
        # Replica computation: calls the string-op function directly (no explicit marking).
        return computation_with_string_ops(x)

      return strategy.run(computation, args=(x,))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step(0)),
        constant_op.constant(10, shape=(strategy.num_replicas_in_sync)))

  # Regression test for b/180509859.
  @unittest.skipIf(not _TENSORBOARD_AVAILABLE, "Tensorboard is not installed.")
  def testImageSummary(self):
    # [FUNCTION DOC]
    # Regression test (skipped without tensorboard): an image summary written inside a TPU
    # computation that also contains a loop filling a tensor array with random values. Runs the
    # computation under a file writer in a temporary directory, reads back the events, decodes
    # the image of the second event and asserts it is not all zeros.
    strategy = get_tpu_strategy()

    def run():
      # [FUNCTION DOC]
      # Builds and executes the sampling graph function once.

      @def_function.function
      def sample_sequence():
        # [FUNCTION DOC]
        # Graph function: runs the replica function f via the strategy with no arguments.
        bsz = 3
        max_length = 32 * 32

        def f():
          # [FUNCTION DOC]
          # Replica computation: loops max_length steps writing random batch-size vectors into a
          # tensor array, stacks them into a 32 by 32 by 3-channel image and writes an image
          # summary with step 5. Returns nothing.

          def body(step, tokens):
            # [FUNCTION DOC]
            # Loop body: writes a fresh random vector at the current step and increments step.
            # Returns the new step and tensor array.
            next_token = random_ops.random_uniform([bsz])
            tokens = tokens.write(step, next_token)
            return (step + 1, tokens)

          def cond_fn(step, tokens):
            # [FUNCTION DOC]
            # Loop condition: true while step is below max_length; ignores the tensor array.
            del tokens
            return math_ops.less(step, max_length)

          tokens_var = tensor_array_ops.TensorArray(
              dtype=dtypes.float32,
              size=max_length,
              dynamic_size=False,
              clear_after_read=False,
              element_shape=(bsz,),
              name="tokens_accumulator",
          )

          step = constant_op.constant(0)
          step, tokens_var = while_loop.while_loop(
              cond_fn, body, [step, tokens_var]
          )

          image_flat = array_ops.transpose(tokens_var.stack(), [1, 0])
          image = array_ops.tile(
              array_ops.reshape(image_flat, [bsz, 32, 32, 1]), [1, 1, 1, 3]
          )
          image_summary_v2.image(
              "image_sample", image, constant_op.constant(5, dtype=dtypes.int64)
          )

        return strategy.run(f)

      sample_sequence()

    logdir = tempfile.mkdtemp()
    summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      run()
    events = _events_from_logdir(self, logdir)
    decoded_image = image_ops.decode_png(
        events[1].summary.value[0].tensor.string_val[2]).numpy()
    # Ensure that non-zero values were written to the image summary.
    self.assertNotAllEqual(
        array_ops.zeros((3072,), dtype=dtypes.float32),
        list(decoded_image.flat))

  @unittest.skipIf(not _TENSORBOARD_AVAILABLE, "Tensorboard is not installed.")
  def testSummaryWithAutoOutsideCompilation(self):
    # [FUNCTION DOC]
    # Test (skipped without tensorboard): a scalar summary in a plain function called from the TPU
    # computation is moved to the host automatically. Expects 7 per replica, and exactly two
    # events in the log (file header and the host-written one) with tag x.
    strategy = get_tpu_strategy()

    def host_computation(x):
      # [FUNCTION DOC]
      # Function with a scalar summary named x at step 0; returns the input doubled.
      scalar_summary_v2.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():
      # [FUNCTION DOC]
      # Graph function that runs the replica computation on all replicas with input 2.

      def computation(x):
        # [FUNCTION DOC]
        # Replica computation: adds 1, calls the summary function directly, adds 1.
        x = x + 1.0
        y = host_computation(x)
        return y + 1.0

      return strategy.run(computation, args=(2.0,))

    logdir = tempfile.mkdtemp()
    summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(7., shape=(strategy.num_replicas_in_sync)))
    events = _events_from_logdir(self, logdir)
    # There will be 2 entries: 1 summary file header entry, and 1 entry
    # written by host.
    self.assertLen(events, 2)
    self.assertEqual(events[1].summary.value[0].tag, "x")

  @unittest.skipIf(not _TENSORBOARD_AVAILABLE, "Tensorboard is not installed.")
  def testNestedFunctionScalarSummary(self):
    # [FUNCTION DOC]
    # Test (skipped without tensorboard): like the automatic scalar summary test but the replica
    # computation is itself a graph function nested in the outer one. Expects 7 per replica and
    # two events in the log, the second with tag x.
    strategy = get_tpu_strategy()

    def host_computation(x):
      # [FUNCTION DOC]
      # Function with a scalar summary named x at step 0; returns the input doubled.
      scalar_summary_v2.scalar("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():
      # [FUNCTION DOC]
      # Outer graph function that runs the nested replica computation with input 2.

      @def_function.function
      def computation(x):
        # [FUNCTION DOC]
        # Nested replica computation: adds 1, calls the summary function, adds 1.
        x = x + 1.0
        y = host_computation(x)
        return y + 1.0

      return strategy.run(computation, args=(2.0,))

    logdir = tempfile.mkdtemp()
    summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(7., shape=(strategy.num_replicas_in_sync)))
    events = _events_from_logdir(self, logdir)
    # There will be 2 entries: 1 summary file header entry, and 1 entry
    # written by host.
    self.assertLen(events, 2)
    self.assertEqual(events[1].summary.value[0].tag, "x")

  @unittest.skipIf(not _TENSORBOARD_AVAILABLE, "Tensorboard is not installed.")
  def testHistogramSummaryWithAutoOutsideCompilation(self):
    # [FUNCTION DOC]
    # Test (skipped without tensorboard): a histogram summary in a function called from the TPU
    # computation is moved to the host automatically. Expects 7 per replica and two events in the
    # log, the second with tag x.
    strategy = get_tpu_strategy()

    def host_computation(x):
      # [FUNCTION DOC]
      # Function with a histogram summary named x at step 0; returns the input doubled.
      histogram_summary_v2.histogram("x", x, step=0)
      return x * 2.0

    @def_function.function
    def step():
      # [FUNCTION DOC]
      # Graph function that runs the replica computation on all replicas with input 2.

      def computation(x):
        # [FUNCTION DOC]
        # Replica computation: adds 1, calls the summary function, adds 1.
        x = x + 1.0
        y = host_computation(x)
        return y + 1.0

      return strategy.run(computation, args=(2.0,))

    logdir = tempfile.mkdtemp()
    summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(7., shape=(strategy.num_replicas_in_sync)))
    events = _events_from_logdir(self, logdir)
    # There will be 2 entries: 1 summary file header entry, and 1 entry
    # written by host.
    self.assertLen(events, 2)
    self.assertEqual(events[1].summary.value[0].tag, "x")

  @parameterized.parameters((True), (False))
  @unittest.skipIf(not _TENSORBOARD_AVAILABLE, "Tensorboard is not installed.")
  def testSummaryControlFlowIfWithAutoOutsideCompilation(
      self, take_true_branch):
    # [FUNCTION DOC]
    # Parameterized test (skipped without tensorboard): a scalar summary inside a conditional of
    # the TPU computation, extracted automatically. Input 2 (flag true) expects 7 and, in the log,
    # two events with the second tagged as inside the conditional; input 10 expects 12 and
    # the log is not checked.
    strategy = get_tpu_strategy()

    @def_function.function
    def step():
      # [FUNCTION DOC]
      # Graph function; runs the replica computation with input 2 or 10 depending on the flag.

      def computation(x):
        # [FUNCTION DOC]
        # Replica computation: adds 1; if the value is below 5, writes a scalar summary and
        # doubles the value; then adds 1 and returns.
        x = x + 1.0
        if x < 5:
          scalar_summary_v2.scalar("x", x, step=0)
          x = x * 2.0
        return x + 1.0

      if take_true_branch:
        return strategy.run(computation, args=(2.0,))
      else:
        return strategy.run(computation, args=(10.0,))

    logdir = tempfile.mkdtemp()
    summary_writer = summary.create_file_writer(logdir, flush_millis=10000)
    output_value = 12.
    if take_true_branch:
      output_value = 7.
    with summary_writer.as_default(), summary.always_record_summaries():
      self.assertAllEqual(
          strategy.experimental_local_results(step()),
          constant_op.constant(
              output_value, shape=(strategy.num_replicas_in_sync)))
    if take_true_branch:
      events = _events_from_logdir(self, logdir)
      # There will be 2 entries: 1 summary file header entry, and 1 entry
      # written by host.
      #
      self.assertLen(events, 2)
      self.assertEqual(events[1].summary.value[0].tag, "cond/x")

  def testAutoOutsideCompilationWithFunctionalNodes(self):
    # [FUNCTION DOC]
    # Test: a functional conditional whose branches call the string-op function, requiring the
    # whole control-flow node to run on the host. With a=0 and b=-1 the predicate is true and the
    # result is 10 on every replica.
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step(a, b):
      # [FUNCTION DOC]
      # Graph function that runs the replica function on all replicas with arguments a and b.

      def fn(a, b):
        # [FUNCTION DOC]
        # Replica computation: chooses between the string-op function applied to a times 100 and
        # applied to a, based on whether a is at least b; returns the named identity of the result.
        fn1 = lambda: computation_with_string_ops(a * 100)
        fn2 = lambda: computation_with_string_ops(a)
        pred = math_ops.greater_equal(a, b)
        result = array_ops.identity(
            cond.cond(pred, fn1, fn2),
            name="uncompilable_control_flow")
        return result

      return strategy.run(fn, args=(a, b))

    self.assertAllEqual(
        strategy.experimental_local_results(train_step(0.0, -1.0)),
        constant_op.constant(10, shape=(strategy.num_replicas_in_sync)))

  def testRandomOpsWithAutoOutsideCompilation(self):
    # [FUNCTION DOC]
    # Test: random normal generation in the TPU computation works with soft placement. Asserts
    # only that the first replica's result has shape one by two by three.
    strategy = get_tpu_strategy()

    @def_function.function
    def train_step():
      # [FUNCTION DOC]
      # Graph function that runs the replica computation on all replicas with no arguments.

      def computation():
        # [FUNCTION DOC]
        # Replica computation: returns a random normal tensor of shape one by two by three.
        return random_ops.random_normal(shape=[1, 2, 3])

      return strategy.run(computation, args=())

    self.assertAllEqual(
        strategy.experimental_local_results(train_step())[0].shape, [1, 2, 3])

  def testOutsideCompilationWithTPUPartitionedCallOp(self):
    # [FUNCTION DOC]
    # Test: a function with outside compilation, rewritten for TPU and then invoked through the
    # TPU partitioned-call op, can be traced with a float vector of size one input and executed;
    # asserts the first output is a tensor.
    """Tests that control flow with TPUPartitionedCall including outside_compilation works."""
    get_tpu_strategy()

    def host_computation(x):
      # [FUNCTION DOC]
      # Host function; returns the input plus 1.
      return x + 1

    @def_function.function()
    def train_step(x):
      # [FUNCTION DOC]
      # Graph function: adds 5, prints the value, outside-compiles the host function and adds 4.
      x2 = x + 5.0
      logging_ops.print_v2(x2)
      x2 = tpu_replication.outside_compilation(host_computation, x2)
      return x2 + 4.0

    tpu_fn = _rewrite_func_wrapper(train_step)
    partitioned_tpu_fn = _tpu_partitioned_call_wrapper(tpu_fn)

    concrete = partitioned_tpu_fn.get_concrete_function(
        x=tensor.TensorSpec(
            shape=(1), dtype=dtypes.float32, name="input_tensor"))

    self.assertIsInstance(
        concrete(array_ops.ones((1), dtype=dtypes.float32))[0], tensor.Tensor)


if __name__ == "__main__":
  test.main()
