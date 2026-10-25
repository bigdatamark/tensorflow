# [MODULE DOC]
# KNOWN ISSUE: the module-level options variable is only created when the file runs as the main
# program, so importing it and running the benchmark elsewhere fails with a missing-name error.
# End-to-end benchmark script comparing three implementations of batch normalization: the fused
# kernel ("op"), the library Python implementation ("py") and a naive arithmetic version ("slow").
# Builds graphs of stacked normalization layers on CPU (and GPU when enabled), optionally with
# backpropagation, times repeated session runs, prints per-run timing and relative differences,
# and reports each run through the benchmark reporting facility. Contains helper functions plus one
# benchmark class. Run as a script it defines a command-line option for enabling GPU runs (default
# on), stores parsed options in a module-level variable and starts the test runner.
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
"""End-to-end benchmark for batch normalization."""

import argparse
import sys
import time

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def batch_norm_op(tensor, mean, variance, beta, gamma, scale):
  # [FUNCTION DOC]
  # HELPER: "op" implementation used by the benchmark graph builder.
  # KNOWN ISSUE: the docstring calls this the fused kernel, but it uses the deprecated global
  # normalization kernel and forces the default graph's producer version down to 8 as a side
  # effect (needed because the kernel is deprecated since version 9).
  # Inputs: tensor to normalize, mean, variance, beta (offset) and gamma (scale) tensors, and a
  # boolean scale flag passed to the kernel with a fixed epsilon of 0.001. Returns the normalized
  # tensor. Side effect: mutates the default graph's producer version.
  """Fused kernel for batch normalization."""
  # _batch_norm_with_global_normalization is deprecated in v9
  test_util.set_producer_version(ops.get_default_graph(), 8)
  # pylint: disable=protected-access
  return gen_nn_ops._batch_norm_with_global_normalization(
      tensor, mean, variance, beta, gamma, 0.001, scale)
  # pylint: enable=protected-access


# Note that the naive implementation is much slower:
# batch_norm = (tensor - mean) * tf.math.rsqrt(variance + 0.001)
# if scale:
#   batch_norm *= gamma
# return batch_norm + beta
def batch_norm_py(tensor, mean, variance, beta, gamma, scale):
  # [FUNCTION DOC]
  # HELPER: "py" implementation used by the benchmark graph builder.
  # Applies the library's composite batch normalization routine with a fixed epsilon of 0.001.
  # Inputs: tensor to normalize, mean, variance, beta (offset), gamma (scale factor, used only
  # when the boolean scale flag is true, otherwise no scale is applied). Returns the normalized
  # tensor. No side effects beyond adding graph ops.
  """Python implementation of batch normalization."""
  return nn_impl.batch_normalization(tensor, mean, variance, beta, gamma if
                                     scale else None, 0.001)


def batch_norm_slow(tensor, mean, variance, beta, gamma, scale):
  # [FUNCTION DOC]
  # HELPER: "slow" naive implementation used by the benchmark graph builder (baseline).
  # Computes (tensor minus mean) times reciprocal square root of (variance plus 0.001), then
  # multiplies by gamma only when the boolean scale flag is true, and finally adds beta. Returns
  # the normalized tensor. No side effects beyond adding graph ops.
  batch_norm = (tensor - mean) * math_ops.rsqrt(variance + 0.001)
  if scale:
    batch_norm *= gamma
  return batch_norm + beta


def build_graph(device, input_shape, axes, num_layers, mode, scale, train):
  # [FUNCTION DOC]
  # HELPER: graph construction routine for the benchmark.
  # Builds, in the current default graph on the given device (cpu or gpu, device index 0), a
  # chain of num_layers batch normalization layers over a variable initialized from a truncated
  # normal of input_shape. Inputs: device name string; input_shape list of dimensions; axes list
  # of dimension indices normalized across; num_layers count; mode one of "op", "py", "slow"
  # selecting the implementation (any other string adds no layers but still creates variables);
  # scale boolean; train boolean. When train is true, mean and variance come from computing
  # moments of the running tensor and the result is the list of gradients of the final tensor
  # with respect to all trainable variables; otherwise mean is zeros and variance is ones and the
  # result is a one-element list holding the final tensor. Moment shapes keep reduced axes as
  # size one for "py" and "slow", and drop them for "op". Creates beta and gamma variables for
  # each layer. Returns the list of tensors to run.
  """Build a graph containing a sequence of batch normalizations.

  Args:
    device: string, the device to run on.
    input_shape: shape of the input tensor.
    axes: axes that are to be normalized across.
    num_layers: number of batch normalization layers in the graph.
    mode: "op", "py" or "slow" depending on the implementation.
    scale: scale after normalization.
    train: if true, also run backprop.

  Returns:
    An array of tensors to run()
  """
  moment_shape = []
  keep_dims = mode == "py" or mode == "slow"
  if keep_dims:
    for axis in range(len(input_shape)):
      if axis in axes:
        moment_shape.append(1)
      else:
        moment_shape.append(input_shape[axis])
  else:
    for axis in range(len(input_shape)):
      if axis not in axes:
        moment_shape.append(input_shape[axis])
  with ops.device("/%s:0" % device):
    tensor = variables.Variable(random_ops.truncated_normal(input_shape))
    for _ in range(num_layers):
      if train:
        mean, variance = nn_impl.moments(tensor, axes, keep_dims=keep_dims)
      else:
        mean = array_ops.zeros(moment_shape)
        variance = array_ops.ones(moment_shape)
      beta = variables.Variable(array_ops.zeros(moment_shape))
      gamma = variables.Variable(constant_op.constant(1.0, shape=moment_shape))
      if mode == "py":
        tensor = batch_norm_py(tensor, mean, variance, beta, gamma, scale)
      elif mode == "op":
        tensor = batch_norm_op(tensor, mean, variance, beta, gamma, scale)
      elif mode == "slow":
        tensor = batch_norm_slow(tensor, mean, variance, beta, gamma, scale)
    if train:
      return gradients_impl.gradients([tensor], variables.trainable_variables())
    else:
      return [tensor]


def print_difference(mode, t1, t2):
  # [FUNCTION DOC]
  # HELPER: reporting utility for the benchmark.
  # Prints a line with the label mode and the percentage by which timing t2 exceeds timing t1
  # (relative to t1; negative if t2 is faster). Inputs: label string, two durations in seconds
  # (t1 must be nonzero). Returns nothing. Side effect: writes to standard output.
  """Print the difference in timing between two runs."""
  difference = (t2 - t1) / t1 * 100.0
  print("=== %s: %.1f%% ===" % (mode, difference))


class BatchNormBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class comparing the fused, library and naive batch normalization implementations
  # on convolution-like (four dimensional) and fully-connected (two dimensional) shapes, forward
  # only and forward plus backward, on CPU and optionally GPU. Holds no state of its own; results
  # are printed and reported through the benchmark base class.
  """Benchmark batch normalization."""

  def _run_graph(self, device, input_shape, axes, num_layers, mode, scale,
                 train, num_iters):
    # [FUNCTION DOC]
    # HELPER: runs one benchmark configuration for the benchmark methods.
    # Builds the graph with the graph builder in a fresh graph, opens a session, initializes
    # global variables, does one warm-up run, then times num_iters runs of the output ops.
    # Inputs: device, input_shape, axes, num_layers, mode, scale, train as for the graph builder,
    # plus num_iters, the number of timed iterations. Prints a one-line summary with the average
    # seconds per iteration and reports a named benchmark entry (name encodes device, shape,
    # axes, mode, layers, scale and train, with spaces removed) with iteration count and average
    # wall time. Returns the total duration in seconds of the timed loop (not the average).
    """Run the graph and print its execution time.

    Args:
      device: string, the device to run on.
      input_shape: shape of the input tensor.
      axes: axes that are to be normalized across.
      num_layers: number of batch normalization layers in the graph.
      mode: "op", "py" or "slow" depending on the implementation.
      scale: scale after normalization.
      train: if true, also run backprop.
      num_iters: number of steps to run.

    Returns:
      The duration of the run in seconds.
    """
    graph = ops.Graph()
    with graph.as_default():
      outputs = build_graph(device, input_shape, axes, num_layers, mode, scale,
                            train)
    with session_lib.Session(graph=graph) as session:
      variables.global_variables_initializer().run()
      _ = session.run([out.op for out in outputs])  # warm up.
      start_time = time.time()
      for _ in range(num_iters):
        _ = session.run([out.op for out in outputs])
      duration = time.time() - start_time
    print("%s shape:%d/%d #layers:%d mode:%s scale:%r train:%r - %f secs" %
          (device, len(input_shape), len(axes), num_layers, mode, scale, train,
           duration / num_iters))

    name_template = (
        "batch_norm_{device}_input_shape_{shape}_axes_{axes}_mode_{mode}_"
        "layers_{num_layers}_scale_{scale}_"
        "train_{train}")

    self.report_benchmark(
        name=name_template.format(
            device=device,
            mode=mode,
            num_layers=num_layers,
            scale=scale,
            train=train,
            shape=str(input_shape).replace(" ", ""),
            axes=str(axes)).replace(" ", ""),
        iters=num_iters,
        wall_time=duration / num_iters)

    return duration

  def benchmark_batch_norm(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the iteration counts for the GPU fully-connected runs are inconsistent with the
    # CPU pattern (forward-only uses 50 on GPU but backward uses only 5, while the CPU backward
    # case uses 50), which looks like swapped values.
    # Benchmark entry point (no inputs, returns nothing). Runs the three implementations on CPU
    # (and on GPU when the command-line option is on) for four scenarios: forward and
    # forward/backward on a lower-layer convolution shape, the same on a higher-layer shape, and
    # forward and forward/backward on a fully connected shape (only the library and naive
    # versions). After each group prints relative timing differences between implementations.
    # Reads the module-level options variable, which only exists when run as the main program.
    print("Forward convolution (lower layers).")
    shape = [8, 128, 128, 32]
    axes = [0, 1, 2]
    t1 = self._run_graph("cpu", shape, axes, 10, "op", True, False, 5)
    t2 = self._run_graph("cpu", shape, axes, 10, "py", True, False, 5)
    t3 = self._run_graph("cpu", shape, axes, 10, "slow", True, False, 5)
    print_difference("op vs py", t1, t2)
    print_difference("py vs slow", t2, t3)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "op", True, False, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "py", True, False, 50)
      t3 = self._run_graph("gpu", shape, axes, 10, "slow", True, False, 50)
      print_difference("op vs py", t1, t2)
      print_difference("py vs slow", t2, t3)
    print("Forward/backward convolution (lower layers).")
    t1 = self._run_graph("cpu", shape, axes, 10, "op", True, True, 5)
    t2 = self._run_graph("cpu", shape, axes, 10, "py", True, True, 5)
    t3 = self._run_graph("cpu", shape, axes, 10, "slow", True, True, 5)
    print_difference("op vs py", t1, t2)
    print_difference("py vs slow", t2, t3)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "op", True, True, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "py", True, True, 50)
      t3 = self._run_graph("gpu", shape, axes, 10, "slow", True, True, 50)
      print_difference("op vs py", t1, t2)
      print_difference("py vs slow", t2, t3)
    print("Forward convolution (higher layers).")
    shape = [256, 17, 17, 32]
    axes = [0, 1, 2]
    t1 = self._run_graph("cpu", shape, axes, 10, "op", True, False, 5)
    t2 = self._run_graph("cpu", shape, axes, 10, "py", True, False, 5)
    t3 = self._run_graph("cpu", shape, axes, 10, "slow", True, False, 5)
    print_difference("op vs py", t1, t2)
    print_difference("py vs slow", t2, t3)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "op", True, False, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "py", True, False, 50)
      t3 = self._run_graph("gpu", shape, axes, 10, "slow", True, False, 50)
      print_difference("op vs py", t1, t2)
      print_difference("py vs slow", t2, t3)
    print("Forward/backward convolution (higher layers).")
    t1 = self._run_graph("cpu", shape, axes, 10, "op", True, True, 5)
    t2 = self._run_graph("cpu", shape, axes, 10, "py", True, True, 5)
    t3 = self._run_graph("cpu", shape, axes, 10, "slow", True, True, 5)
    print_difference("op vs py", t1, t2)
    print_difference("py vs slow", t2, t3)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "op", True, True, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "py", True, True, 50)
      t3 = self._run_graph("gpu", shape, axes, 10, "slow", True, True, 50)
      print_difference("op vs py", t1, t2)
      print_difference("py vs slow", t2, t3)
    print("Forward fully-connected.")
    shape = [1024, 32]
    axes = [0]
    t1 = self._run_graph("cpu", shape, axes, 10, "py", True, False, 5)
    t2 = self._run_graph("cpu", shape, axes, 10, "slow", True, False, 5)
    print_difference("py vs slow", t1, t2)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "py", True, False, 50)
      t2 = self._run_graph("gpu", shape, axes, 10, "slow", True, False, 50)
      print_difference("py vs slow", t1, t2)
    print("Forward/backward fully-connected.")
    t1 = self._run_graph("cpu", shape, axes, 10, "py", True, True, 50)
    t2 = self._run_graph("cpu", shape, axes, 10, "slow", True, True, 50)
    print_difference("py vs slow", t1, t2)
    if FLAGS.use_gpu:
      t1 = self._run_graph("gpu", shape, axes, 10, "py", True, True, 5)
      t2 = self._run_graph("gpu", shape, axes, 10, "slow", True, True, 5)
      print_difference("py vs slow", t1, t2)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--use_gpu",
      type="bool",
      nargs="?",
      const=True,
      default=True,
      help="Run GPU benchmarks."
  )
  global FLAGS  # pylint:disable=global-at-module-level
  FLAGS, unparsed = parser.parse_known_args()
  test.main(argv=[sys.argv[0]] + unparsed)
