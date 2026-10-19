# [MODULE DOC]
# Tests and benchmarks for the Python bindings of the unified (eager/graph agnostic) API. The test
# class checks math and neural-network operations and their gradients, both traced with the
# experimental function wrapper and run directly, under the graph and MLIR tracing
# implementations. The benchmark class times gradient computation of a small multi-layer
# perceptron with the unified tape versus the regular eager tape. Defines small module-level
# helpers. Runs as a standalone program with eager execution enabled.
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
"""Tests for Unified APIs' python bindings."""

import timeit

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.experimental import _unified_api
from tensorflow.python.framework.experimental import context_stack as context_lib
from tensorflow.python.framework.experimental import def_function
from tensorflow.python.framework.experimental import math_ops as unified_math_ops
from tensorflow.python.framework.experimental import nn_ops as unified_nn_ops
from tensorflow.python.framework.experimental import tape as tape_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test

SetTracingImplementation = _unified_api.SetTracingImplementation
TensorCastHelper = _unified_api.EagerTensorToImmediateExecutionTensorHandle


def get_immediate_execution_context():
  # [FUNCTION DOC]
  # HELPER: shared test utility. Resets the global eager context, initializes it, and wraps it as
  # an immediate-execution context for the unified API. Takes no inputs; returns that wrapped
  # context. Side effect: discards any existing eager context state.
  context._reset_context()
  context.context().ensure_initialized()
  return _unified_api.EagerContextToImmediateExecutionContext(
      context.context()._handle)


def maybe_cast(t, perform_cast):
  # [FUNCTION DOC]
  # HELPER: shared test utility. Inputs: t, an eager tensor; perform_cast, whether to convert it.
  # Returns the tensor converted to a unified-API tensor handle when perform_cast is true,
  # otherwise the tensor unchanged.
  if perform_cast:
    return TensorCastHelper(t)
  return t


class UnifiedApiTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the unified API bindings. Each test checks one operation (add, relu, negation,
  # subtraction, multiplication, log of one plus x, safe division) or its gradient under both the
  # Graph and MLIR tracing implementations, comparing traced and directly executed results.

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testAdd(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: add the vectors [1,2] and [3,4] through the unified add op, once traced with the unified function wrapper and once run directly, inside a default context from a fresh eager context. Asserts both results equal [4,6].
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Takes two tensor handles and returns their unified-API sum.
      return unified_math_ops.add(a, b)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_output = def_function.function(model)(a, b)
      self.assertAllEqual(func_output.numpy(), [4., 6.])

      eager_output = model(a, b)
      self.assertAllEqual(eager_output.numpy(), [4., 6.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testAddGrad(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: gradient of the sum of two watched vectors with respect to both, traced and direct. Asserts both gradients are [1,1] in each mode.
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Records the unified add of both inputs under a unified gradient tape with both watched, and returns the list of gradients of the sum with respect to the two inputs.
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        result = unified_math_ops.add(a, b)
      grads = tape.gradient(result, [a, b])
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_outputs = def_function.function(model)(a, b)
      self.assertAllEqual(func_outputs[0].numpy(), [1.0, 1.0])
      self.assertAllEqual(func_outputs[1].numpy(), [1.0, 1.0])

      eager_outputs = model(a, b)
      self.assertAllEqual(eager_outputs[0].numpy(), [1.0, 1.0])
      self.assertAllEqual(eager_outputs[1].numpy(), [1.0, 1.0])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testRelu(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: apply the unified relu to [1] and to [-1], traced (same traced function reused) and direct. Asserts outputs [1] and [0].
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(t):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Takes one tensor handle and returns its unified-API relu.
      return unified_nn_ops.relu(t)

    with context_lib.set_default(get_immediate_execution_context()):
      positive = TensorCastHelper(constant_op.constant([1.]))
      negative = TensorCastHelper(constant_op.constant([-1.]))

      model_fn = def_function.function(model)
      func_output = model_fn(positive)
      self.assertAllEqual(func_output.numpy(), [1.])
      func_output = model_fn(negative)
      self.assertAllEqual(func_output.numpy(), [0.])

      eager_output = model(positive)
      self.assertAllEqual(eager_output.numpy(), [1.])
      eager_output = model(negative)
      self.assertAllEqual(eager_output.numpy(), [0.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testReluGrad(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: gradient of relu with respect to its input for [1] and [-1], traced and direct. Asserts gradients [1] and [0].
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(t):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Records relu of the input under a unified gradient tape with the input watched, and returns the gradient with respect to the input.
      with tape_lib.GradientTape() as tape:
        tape.watch(t)
        result = unified_nn_ops.relu(t)
      grads = tape.gradient(result, t)
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      positive = TensorCastHelper(constant_op.constant([1.]))
      negative = TensorCastHelper(constant_op.constant([-1.]))

      model_fn = def_function.function(model)
      func_output = model_fn(positive)
      self.assertAllEqual(func_output.numpy(), [1.])
      func_output = model_fn(negative)
      self.assertAllEqual(func_output.numpy(), [0.])

      eager_output = model(positive)
      self.assertAllEqual(eager_output.numpy(), [1.])
      eager_output = model(negative)
      self.assertAllEqual(eager_output.numpy(), [0.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testNeg(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: negate [2] with the unified neg op, traced and direct. Asserts the result is [-2].
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Takes one tensor handle and returns its unified-API negation.
      return unified_math_ops.neg(a)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([2.]))

      func_output = def_function.function(model)(a)
      self.assertAllEqual(func_output.numpy(), [-2.])

      eager_output = model(a)
      self.assertAllEqual(eager_output.numpy(), [-2.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testNegGrad(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: gradient of negation of [2] with respect to its input, traced and direct. Asserts the gradient is [-1].
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Records negation under a unified gradient tape with the input watched, and returns the gradient with respect to the input.
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        result = unified_math_ops.neg(a)
      grads = tape.gradient(result, a)
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([2.]))

      func_outputs = def_function.function(model)(a)
      self.assertAllEqual(func_outputs.numpy(), [-1.0])

      eager_outputs = model(a)
      self.assertAllEqual(eager_outputs.numpy(), [-1.0])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testSub(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: subtract [3,4] from [1,2] with the unified sub op, traced and direct. Asserts the result is [-2,-2].
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Takes two tensor handles and returns their unified-API difference.
      return unified_math_ops.sub(a, b)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_output = def_function.function(model)(a, b)
      self.assertAllEqual(func_output.numpy(), [-2., -2.])

      eager_output = model(a, b)
      self.assertAllEqual(eager_output.numpy(), [-2., -2.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testSubGrad(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: gradients of a minus b for [1,2] and [3,4], traced and direct. Asserts the gradient for a is [1,1] and for b is [-1,-1].
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Records the unified subtraction under a unified gradient tape with both inputs watched, and returns the list of gradients with respect to the two inputs.
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        result = unified_math_ops.sub(a, b)
      grads = tape.gradient(result, [a, b])
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_outputs = def_function.function(model)(a, b)
      self.assertAllEqual(func_outputs[0].numpy(), [1.0, 1.0])
      self.assertAllEqual(func_outputs[1].numpy(), [-1.0, -1.0])

      eager_outputs = model(a, b)
      self.assertAllEqual(eager_outputs[0].numpy(), [1.0, 1.0])
      self.assertAllEqual(eager_outputs[1].numpy(), [-1.0, -1.0])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testMul(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: multiply [1,2] and [3,4] with the unified mul op, traced and direct. Asserts the result is [3,8].
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Takes two tensor handles and returns their unified-API product.
      return unified_math_ops.mul(a, b)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_output = def_function.function(model)(a, b)
      self.assertAllEqual(func_output.numpy(), [3., 8.])

      eager_output = model(a, b)
      self.assertAllEqual(eager_output.numpy(), [3., 8.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testMulGrad(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: gradients of a times b for [1,2] and [3,4], traced and direct. Asserts the gradient for a is [3,4] (the value of b) and for b is [1,2] (the value of a).
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Records the unified multiplication under a unified gradient tape with both inputs watched, and returns the list of gradients with respect to the two inputs.
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        result = unified_math_ops.mul(a, b)
      grads = tape.gradient(result, [a, b])
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1., 2.]))
      b = TensorCastHelper(constant_op.constant([3., 4.]))

      func_outputs = def_function.function(model)(a, b)
      self.assertAllEqual(func_outputs[0].numpy(), [3., 4.])
      self.assertAllEqual(func_outputs[1].numpy(), [1., 2.])

      eager_outputs = model(a, b)
      self.assertAllEqual(eager_outputs[0].numpy(), [3., 4.])
      self.assertAllEqual(eager_outputs[1].numpy(), [1., 2.])

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testLog1p(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: apply the unified log-one-plus op to [1], traced and direct. Asserts the result is close to 0.69314 within 0.001.
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Takes one tensor handle and returns its unified-API log of one plus the value.
      return unified_math_ops.log1p(a)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1.]))

      func_output = def_function.function(model)(a)
      self.assertArrayNear(func_output.numpy(), [0.69314], 0.001)

      eager_output = model(a)
      self.assertArrayNear(eager_output.numpy(), [0.69314], 0.001)

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testLog1pGrad(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: gradient of log-one-plus at [1], traced and direct. Asserts the gradient is close to 0.5 within 0.001.
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Records the unified log-one-plus under a unified gradient tape with the input watched, and returns the gradient with respect to the input.
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        result = unified_math_ops.log1p(a)
      grads = tape.gradient(result, a)
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([1.]))

      func_outputs = def_function.function(model)(a)
      self.assertArrayNear(func_outputs.numpy(), [0.5], 0.001)

      eager_outputs = model(a)
      self.assertArrayNear(eager_outputs.numpy(), [0.5], 0.001)

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testDivNoNan(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: divide [2] by [4] with the unified safe-division op, traced and direct. Asserts the result is close to 0.5 within 0.001.
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Takes two tensor handles and returns their unified-API safe quotient.
      return unified_math_ops.div_no_nan(a, b)

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([2.]))
      b = TensorCastHelper(constant_op.constant([4.]))

      func_output = def_function.function(model)(a, b)
      self.assertArrayNear(func_output.numpy(), [0.5], 0.001)

      eager_output = model(a, b)
      self.assertArrayNear(eager_output.numpy(), [0.5], 0.001)

  @parameterized.named_parameters([
      ("Graph", False),
      ("Mlir", True),
  ])
  def testDivNoNanGrad(self, use_mlir):
    # [FUNCTION DOC]
    # Parameterized over a flag selecting the Graph tracing implementation (false) or the MLIR
    # tracing implementation (true); when true, switches the global tracing implementation to MLIR
    # (side effect on global state that persists for later tests).
    # Scenario: gradients of the safe quotient of [2] over [4], traced and direct. Asserts the gradient for the numerator is close to 0.25 and for the denominator close to -0.125 (tolerance 0.001).
    if use_mlir:
      SetTracingImplementation("mlir")

    def model(a, b):
      # [FUNCTION DOC]
      # HELPER: model function traced and also run directly by the enclosing test.
      # Records the unified safe division under a unified gradient tape with both inputs watched, and returns the list of gradients with respect to numerator and denominator.
      with tape_lib.GradientTape() as tape:
        tape.watch(a)
        tape.watch(b)
        result = unified_math_ops.div_no_nan(a, b)
      grads = tape.gradient(result, [a, b])
      return grads

    with context_lib.set_default(get_immediate_execution_context()):
      a = TensorCastHelper(constant_op.constant([2.]))
      b = TensorCastHelper(constant_op.constant([4.]))

      func_outputs = def_function.function(model)(a, b)
      self.assertArrayNear(func_outputs[0].numpy(), [0.25], 0.001)
      self.assertArrayNear(func_outputs[1].numpy(), [-0.125], 0.001)

      eager_outputs = model(a, b)
      self.assertArrayNear(eager_outputs[0].numpy(), [0.25], 0.001)
      self.assertArrayNear(eager_outputs[1].numpy(), [-0.125], 0.001)


class UnifiedTapeBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmarks comparing gradient computation speed of a multi-layer perceptron using the unified
  # API tape against the regular eager tape, for two and ten hidden layers. Has one shared timing
  # helper and four benchmark methods.

  def _computeMnistMlpGrads(self, math_ops_lib, nn_ops_lib, backprop_lib, cast,
                            num_iters, hidden_layers, hidden_size, batch_size):
    # [FUNCTION DOC]
    # HELPER: shared timing routine for the benchmarks. Inputs: math_ops_lib and nn_ops_lib, the
    # operation modules to use; backprop_lib, the module providing the gradient tape; cast, whether
    # to convert inputs to unified-API handles; num_iters, number of timing repeats; hidden_layers,
    # number of hidden layers; hidden_size, width of hidden layers; batch_size (ignored: the code
    # overwrites it with one).
    # KNOWN ISSUE: the batch_size parameter is immediately overwritten with one, so the argument
    # has no effect.
    # Builds a model of ones input, random weights and zero labels, runs ten warm-up gradient
    # computations, then times ten gradient computations per repeat inside a fresh default
    # context. Returns the smallest repeat time divided by ten (seconds per computation).
    batch_size = 1
    image_size = 28 * 28
    num_classes = 10

    def model(x, hidden_weights, softmax_weight, labels):
      # [FUNCTION DOC]
      # HELPER: one training-style step. Inputs: x, input batch; hidden_weights, list of weight
      # matrices; softmax_weight, final weight matrix; labels, class indices. Records, under a
      # gradient tape with every weight watched, a matrix multiply plus relu per hidden layer, a
      # final matrix multiply and the sparse softmax cross entropy loss. Returns the list of loss
      # gradients with respect to all weights.
      with backprop_lib.GradientTape() as tape:
        for weight in hidden_weights + [softmax_weight]:
          tape.watch(weight)
        for hidden_weight in hidden_weights:
          x = math_ops_lib.mat_mul(x, hidden_weight)
          x = nn_ops_lib.relu(x)
        logits = math_ops_lib.mat_mul(x, softmax_weight)
        loss = nn_ops_lib.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)

      grads = tape.gradient(loss, hidden_weights + [softmax_weight])
      return grads

    x = maybe_cast(array_ops.ones([batch_size, image_size]), cast)
    hidden_weights = []
    for i in range(hidden_layers):
      hidden_weights.append(
          maybe_cast(
              random_ops.random_uniform(
                  [hidden_size if i else image_size, hidden_size]), cast))
    softmax_weight = maybe_cast(
        random_ops.random_uniform([hidden_size, num_classes]), cast)
    labels = maybe_cast(array_ops.zeros([batch_size], dtype=dtypes.int32), cast)

    with context_lib.set_default(get_immediate_execution_context()):
      # Warm up.
      for _ in range(10):
        model(x, hidden_weights, softmax_weight, labels)
      runtimes = timeit.repeat(
          lambda: model(x, hidden_weights, softmax_weight, labels),
          repeat=num_iters,
          number=10)
    return min(runtimes) / 10

  def benchmarkTwoHiddenLayerMnistEagerUnified(self):
    # [FUNCTION DOC]
    # Benchmark. Times gradient computation for two hidden layers (100 units each) using the unified ops and tape with unified-API tensor handles; 100 repeats. Reports the result under the unified two-layer benchmark name.
    num_iters = 100
    duration = self._computeMnistMlpGrads(
        unified_math_ops,
        unified_nn_ops,
        tape_lib,
        True,
        num_iters,
        hidden_layers=2,
        hidden_size=100,
        batch_size=1)
    self.report_benchmark(
        name="TwoHiddenLayerMnistEagerUnified",
        iters=num_iters,
        wall_time=duration)

  def benchmarkTwoHiddenLayerMnistEager(self):
    # [FUNCTION DOC]
    # Benchmark. Times gradient computation for two hidden layers (100 units each) using the regular ops and eager tape; 100 repeats. Reports the result under the regular two-layer benchmark name.
    num_iters = 100
    duration = self._computeMnistMlpGrads(
        math_ops,
        nn_ops,
        backprop,
        False,
        num_iters,
        hidden_layers=2,
        hidden_size=100,
        batch_size=1)
    self.report_benchmark(
        name="TwoHiddenLayerMnistEager", iters=num_iters, wall_time=duration)

  def benchmarkTenHiddenLayerMnistEagerUnified(self):
    # [FUNCTION DOC]
    # Benchmark. Times gradient computation for ten hidden layers (100 units each) using the unified ops and tape with unified-API tensor handles; 100 repeats. Reports the result under the unified ten-layer benchmark name.
    num_iters = 100
    duration = self._computeMnistMlpGrads(
        unified_math_ops,
        unified_nn_ops,
        tape_lib,
        True,
        num_iters,
        hidden_layers=10,
        hidden_size=100,
        batch_size=1)
    self.report_benchmark(
        name="TenHiddenLayerMnistEagerUnified",
        iters=num_iters,
        wall_time=duration)

  def benchmarkTenHiddenLayerMnistEager(self):
    # [FUNCTION DOC]
    # Benchmark. Times gradient computation for ten hidden layers (100 units each) using the regular ops and eager tape; 100 repeats. Reports the result under the regular ten-layer benchmark name.
    num_iters = 100
    duration = self._computeMnistMlpGrads(
        math_ops,
        nn_ops,
        backprop,
        False,
        num_iters,
        hidden_layers=10,
        hidden_size=100,
        batch_size=1)
    self.report_benchmark(
        name="TenHiddenLayerMnistEager", iters=num_iters, wall_time=duration)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
