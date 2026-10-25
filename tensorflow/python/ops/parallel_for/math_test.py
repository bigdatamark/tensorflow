# [MODULE DOC]
# Tests for pfor (parallel-for) vectorization of math, reduction, cumulative, segment, selection
# and linear algebra kernels. Each test defines a per-iteration function that gathers slices of
# random tensors by loop index and applies an op, then uses the shared pfor test case helper to
# run it both vectorized and as a sequential loop and compare the results (including gradients
# for unary ops). Defines two test case classes, one for math ops and one for linear algebra,
# both run in graph and eager modes. No import-time side effects besides imports.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for vectorization of math kernels."""
import itertools

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as framework_ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class MathTest(PForTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that pfor vectorization of math kernels (elementwise unary and binary ops, matmul,
  # reductions, argmin/argmax, bucketize, clipping, cumulative ops, bias add, segment reductions,
  # cast, select) matches a sequential loop. Inherits the pfor test case helper that compares
  # vectorized and looped results, and supports parameterized tests. Runs in graph and eager mode.

  def _test_unary_cwise_ops(self, ops, is_complex):
    # [FUNCTION DOC]
    # HELPER: shared body for the unary elementwise op tests.
    # For each op in the given list, builds a random 3x5 input (made complex from two random
    # tensors when is_complex is true) watched by a persistent gradient tape. The per-iteration
    # function applies the op to the whole input and to row i, and, for float outputs, collects
    # gradients with respect to the row using both no output gradient and a loop-variant one.
    # Then compares pfor with the loop over 3 iterations. Returns nothing.
    for op in ops:
      with backprop.GradientTape(persistent=True) as g:
        x = random_ops.random_uniform([3, 5])
        g.watch(x)
        if is_complex:
          y = random_ops.random_uniform([3, 5])
          g.watch(y)
          x = math_ops.complex(x, y)

      # pylint: disable=cell-var-from-loop

      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: iteration body. Applies the op to the full input (loop invariant) and to row i,
        # then, for float32 outputs, appends gradients of each output with respect to row i for
        # both a default and a loop-variant output gradient. Returns the list of outputs.
        with g:
          y = op(x)
          x_i = array_ops.gather(x, i)
          y_i = op(x_i)
          outputs = [y_i]
          # Build cross product of loop variant/invariant outputs and gradients.
          for out in (y, y_i):
            if out.dtype == dtypes.float32:
              for output_gradients in (None, out * math_ops.cast(i, out.dtype)):
                grad = g.gradient(out, x_i, output_gradients=output_gradients)
                if grad is not None:
                  outputs.append(grad)
        return outputs

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3)

  def test_unary_cwise_complex_ops(self):
    # [FUNCTION DOC]
    # Scenario: complex-valued unary ops (angle, imag, complex abs, real, conj) on random complex
    # input. Expects pfor and loop results (and gradients) to match.
    complex_ops = [
        math_ops.angle,
        math_ops.imag,
        math_ops.complex_abs,
        math_ops.real,
        math_ops.conj,
    ]
    self._test_unary_cwise_ops(complex_ops, True)

  def test_unary_cwise_real_ops_1(self):
    # [FUNCTION DOC]
    # Scenario: first group of real unary ops (inverse trig and hyperbolic functions, exp, log,
    # erf family, gamma family, finiteness checks, Bessel functions and others). Expects pfor
    # and loop results (and gradients) to match.
    real_ops = [
        lambda x: math_ops.acosh(1 + math_ops.square(x)),
        math_ops.abs,
        math_ops.acos,
        math_ops.asin,
        math_ops.asinh,
        math_ops.atan,
        math_ops.atanh,
        math_ops.cos,
        math_ops.cosh,
        math_ops.digamma,
        math_ops.erf,
        math_ops.erfc,
        math_ops.erfinv,
        math_ops.exp,
        math_ops.expm1,
        math_ops.inv,
        math_ops.is_finite,
        math_ops.is_inf,
        math_ops.lgamma,
        math_ops.log,
        math_ops.log1p,
        math_ops.ndtri,
        special_math_ops.bessel_i0e,
        special_math_ops.bessel_i1e,
    ]
    self._test_unary_cwise_ops(real_ops, False)

  def test_unary_cwise_real_ops_2(self):
    # [FUNCTION DOC]
    # Scenario: second group of real unary ops (negation, reciprocal, rounding, sqrt, sigmoid,
    # trig, tanh and neural-network activations such as elu, relu, leaky relu, selu, softplus).
    # Expects pfor and loop results (and gradients) to match.
    real_ops = [
        math_ops.neg,
        math_ops.negative,
        math_ops.reciprocal,
        math_ops.rint,
        math_ops.round,
        math_ops.rsqrt,
        math_ops.sigmoid,
        math_ops.sign,
        math_ops.sin,
        math_ops.sinh,
        math_ops.sqrt,
        math_ops.square,
        math_ops.tan,
        math_ops.tanh,
        nn.elu,
        nn.relu,
        nn.relu6,
        lambda t: nn.leaky_relu(t, alpha=0.1),
        nn.selu,
        nn.softplus,
        nn.softsign,
    ]
    self._test_unary_cwise_ops(real_ops, False)

  def test_unary_cwise_no_grad(self):
    # [FUNCTION DOC]
    # Scenario: unary ops without gradients (ceil, floor, logical not; the latter gets a boolean
    # input). For each, applies the op to row i per iteration and expects pfor and loop to match.
    for op in [math_ops.ceil, math_ops.floor, math_ops.logical_not]:
      x = random_ops.random_uniform([3, 5])
      if op == math_ops.logical_not:
        x = x > 0

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: iteration body. Applies the current op to row i of the input and returns it.
        return op(array_ops.gather(x, i))

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3)

  def test_binary_cwise_ops(self):
    # [FUNCTION DOC]
    # Scenario: many binary elementwise ops (logical, arithmetic, comparison, igamma, polygamma,
    # zeta, modulo, floor division and others). Temporarily enables tensor equality so equal and
    # not-equal work, and restores the previous setting in a finally block (global side effect).
    # Floor division is skipped under XLA and zeta is skipped under XLA or on GPU. For every op it
    # builds a 7x3x5 first input and a 3x5 second input (boolean for logical ops) and compares
    # pfor and loop for combinations of loop-variant and loop-invariant arguments over 3 iterations.
    # Enable tensor equality to test `equal` and `not_equal` ops below.
    default_equality = tensor.Tensor._USE_EQUALITY
    tensor.enable_tensor_equality()
    try:
      logical_ops = [
          math_ops.logical_and, math_ops.logical_or, math_ops.logical_xor
      ]

      # Wrapper functions restricting the range of inputs of zeta and polygamma.
      def safe_polygamma(x, y):
        # [FUNCTION DOC]
        # HELPER: wrapper restricting polygamma inputs to a valid range. Rounds y clipped to the
        # range 1 to 10 for the order and uses x squared plus one as the argument. Returns the
        # polygamma result.
        return math_ops.polygamma(
            math_ops.round(clip_ops.clip_by_value(y, 1, 10)), x * x + 1)

      def safe_zeta(x, y):
        # [FUNCTION DOC]
        # HELPER: wrapper restricting zeta inputs to a valid range (first argument x squared plus
        # one, second argument y squared). Returns the zeta result.
        return math_ops.zeta(x * x + 1, y * y)

      float_ops = [
          math_ops.add,
          math_ops.add_v2,
          math_ops.atan2,
          math_ops.complex,
          math_ops.div,
          math_ops.divide,
          math_ops.div_no_nan,
          math_ops.equal,
          lambda x, y: framework_ops.convert_to_tensor(x == y),
          lambda x, y: framework_ops.convert_to_tensor(x != y),
          math_ops.floor_mod,
          math_ops.greater,
          math_ops.greater_equal,
          math_ops.igamma,
          math_ops.igammac,
          math_ops.igamma_grad_a,
          math_ops.less,
          math_ops.less_equal,
          math_ops.maximum,
          math_ops.minimum,
          math_ops.mod,
          math_ops.multiply,
          math_ops.not_equal,
          math_ops.pow,
          math_ops.squared_difference,
          math_ops.subtract,
          math_ops.truncate_mod,
          safe_polygamma,
      ]
      # FloorDiv fails on XLA due floor's discontinuities exacerbating small
      # division differences.
      if not test_util.is_xla_enabled():
        float_ops += [math_ops.floor_div]
        # TODO(b/168912036): Re-enable once GPU + XLA issues for Zeta are
        # resolved.
        if not test_util.is_gpu_available():
          float_ops += [safe_zeta]
      for op in logical_ops + float_ops:
        x = random_ops.random_uniform([7, 3, 5])
        y = random_ops.random_uniform([3, 5])
        if op in logical_ops:
          x = x > 0
          y = y > 0

        output_dtypes = []

        # pylint: disable=cell-var-from-loop
        def loop_fn(i):
          # [FUNCTION DOC]
          # HELPER: iteration body. Gathers element i from both inputs and applies the op to all
          # combinations of invariant and variant arguments, recording their dtypes.
          # KNOWN ISSUE: it clears and refills the output dtype list, which is never read
          # afterwards (dead bookkeeping). Returns the list of five outputs.
          x1 = array_ops.gather(x, i)
          y1 = array_ops.gather(y, i)
          outputs = [op(x, y), op(x1, y), op(x, y1), op(x1, y1), op(x1, x1)]
          del output_dtypes[:]
          output_dtypes.extend(t.dtype for t in outputs)
          return outputs

        # pylint: enable=cell-var-from-loop

        self._test_loop_fn(loop_fn, 3)
    finally:
      if not default_equality:
        tensor.disable_tensor_equality()

  def test_approximate_equal(self):
    # [FUNCTION DOC]
    # Scenario: approximate equality of corresponding rows of two random 3x5 tensors. Expects
    # pfor and loop results to match.
    x = random_ops.random_uniform([3, 5])
    y = random_ops.random_uniform([3, 5])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Gathers row i of both inputs and returns their approximate
      # equality.
      x1 = array_ops.gather(x, i)
      y1 = array_ops.gather(y, i)
      return math_ops.approximate_equal(x1, y1)

    self._test_loop_fn(loop_fn, 3)

  def test_abs_complex(self):
    # [FUNCTION DOC]
    # Scenario: vectorized map of absolute value over a complex128 vector holding zero and
    # minus one. Expects the result to equal zero and one.
    r = pfor_control_flow_ops.vectorized_map(
        math_ops.abs, math_ops.cast([0, -1], dtype=dtypes.complex128))
    self.assertAllEqual(self.evaluate(r), [0, 1])

  def test_colocate_with(self):
    # [FUNCTION DOC]
    # Scenario: the loop body multiplies a gathered row by two inside a device colocation scope
    # with that row. Expects pfor to handle the colocation and match the sequential loop.
    a = random_ops.random_uniform([3, 5])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Gathers row i and returns twice its value, computed under a
      # colocation scope with the row.
      x1 = array_ops.gather(a, i)
      with framework_ops.colocate_with(x1):
        return x1 * 2

    self._test_loop_fn(loop_fn, 3)

  def test_addn(self):
    # [FUNCTION DOC]
    # Scenario: add_n of one loop-variant tensor (row i of a stack of two) and two loop-invariant
    # tensors. Expects pfor and loop results to match.
    x = random_ops.random_uniform([2, 3, 5])
    y = random_ops.random_uniform([3, 5])
    z = random_ops.random_uniform([3, 5])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Gathers slice i of the stacked input and returns the sum of it
      # with the two invariant tensors.
      x1 = array_ops.gather(x, i)
      return math_ops.add_n([x1, y, z])

    self._test_loop_fn(loop_fn, 2)

  def test_cross(self):
    # [FUNCTION DOC]
    # Scenario: cross product of 3-vectors from two random 4x2x3 tensors, with both loop-variant
    # arguments and one loop-invariant argument. Expects pfor and loop results to match.
    x = random_ops.random_uniform([4, 2, 3])
    y = random_ops.random_uniform([4, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Returns the cross product of slices i of both inputs and the
      # cross product of the invariant first slice of x with slice i of y.
      x_i = array_ops.gather(x, i)
      y_i = array_ops.gather(y, i)
      x_0 = array_ops.gather(x, 0)
      return math_ops.cross(x_i, y_i), math_ops.cross(x_0, y_i)

    self._test_loop_fn(loop_fn, 4)

  @test_util.run_without_tensor_float_32(
      "Calls matmul in parallel for-loop and compares result to calling matmul "
      "in sequential for-loop")
  def test_matmul(self):
    # [FUNCTION DOC]
    # Scenario: matrix multiplication over all combinations of transposing either operand and of
    # making either operand loop-variant (stacked) or loop-invariant. Runs with TF32 disabled so
    # results are comparable. Expects pfor and loop results to match over 2 iterations.
    for tr_a in (True, False):
      for tr_b in (True, False):
        for stack_a in (True, False):
          for stack_b in (True, False):
            shape_a = (5, 3) if tr_a else (3, 5)
            if stack_a:
              shape_a = (2,) + shape_a
            shape_b = (7, 5) if tr_b else (5, 7)
            if stack_b:
              shape_b = (2,) + shape_b

            x = random_ops.random_uniform(shape_a)
            y = random_ops.random_uniform(shape_b)

            # pylint: disable=cell-var-from-loop
            def loop_fn(i):
              # [FUNCTION DOC]
              # HELPER: iteration body. Takes slice i of an operand only if it is stacked, then
              # returns their product with the configured transpose flags.
              a = array_ops.gather(x, i) if stack_a else x
              b = array_ops.gather(y, i) if stack_b else y
              return math_ops.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b)

            # pylint: enable=cell-var-from-loop

            self._test_loop_fn(loop_fn, 2)

  def test_batch_matmul(self):
    # [FUNCTION DOC]
    # Scenario: batched matrix multiplication (batch of 4) over all combinations of transposing
    # either operand and of stacking either operand across the loop. Expects pfor and loop
    # results to match over 2 iterations.
    for tr_a in (True, False):
      for tr_b in (True, False):
        for stack_a in (True, False):
          for stack_b in (True, False):
            shape_a = (4, 5, 3) if tr_a else (4, 3, 5)
            if stack_a:
              shape_a = (2,) + shape_a
            shape_b = (4, 7, 5) if tr_b else (4, 5, 7)
            if stack_b:
              shape_b = (2,) + shape_b

            x = random_ops.random_uniform(shape_a)
            y = random_ops.random_uniform(shape_b)

            # pylint: disable=cell-var-from-loop
            def loop_fn(i):
              # [FUNCTION DOC]
              # HELPER: iteration body. Takes slice i of an operand only if it is stacked, then
              # returns their batched product with the configured transpose flags.
              a = array_ops.gather(x, i) if stack_a else x
              b = array_ops.gather(y, i) if stack_b else y
              return math_ops.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b)

            # pylint: enable=cell-var-from-loop

            self._test_loop_fn(loop_fn, 2)

  def test_batch_matmul_broadcast(self):
    # [FUNCTION DOC]
    # Scenario: batched matrix multiplication where either operand may have fewer batch
    # dimensions (broadcast) and either may be stacked across the loop. Expects pfor and loop
    # results to match over 2 iterations.
    for broadcast_a in (True, False):
      for broadcast_b in (True, False):
        for stack_a in (True, False):
          for stack_b in (True, False):
            shape_a = (2, 3, 5) if broadcast_a else (4, 2, 3, 5)
            shape_b = (2, 5, 7) if broadcast_b else (4, 2, 5, 7)
            shape_a = (2,) + shape_a if stack_a else shape_a
            shape_b = (2,) + shape_b if stack_b else shape_b
            x = random_ops.random_uniform(shape_a)
            y = random_ops.random_uniform(shape_b)

            # pylint: disable=cell-var-from-loop
            def loop_fn(i):
              # [FUNCTION DOC]
              # HELPER: iteration body. Takes slice i of an operand only if it is stacked, then
              # returns their product with broadcasting of batch dimensions.
              a = array_ops.gather(x, i) if stack_a else x
              b = array_ops.gather(y, i) if stack_b else y
              return math_ops.matmul(a, b)

            # pylint: enable=cell-var-from-loop
            self._test_loop_fn(loop_fn, 2)

  def test_reduction(self):
    # [FUNCTION DOC]
    # Scenario: sum, product, max, min and mean reductions of a random 2x3x4x5 tensor per
    # iteration, over several axis choices (a list, nothing/all axes, multiple axes, a tensor
    # axis) and with and without keeping dimensions. Expects pfor and loop results to match.
    x = random_ops.random_uniform([2, 3, 4, 5])
    for op in [
        math_ops.reduce_sum,
        math_ops.reduce_prod,
        math_ops.reduce_max,
        math_ops.reduce_min,
        math_ops.reduce_mean,
    ]:
      for axis in ([1], None, [0, 2], constant_op.constant([1], dtypes.int64)):
        for keepdims in (True, False):

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            # [FUNCTION DOC]
            # HELPER: iteration body. Gathers slice i and returns the current numeric reduction of
            # it over the current axis with the current keep-dimensions setting.
            a = array_ops.gather(x, i)
            return op(a, axis=axis, keepdims=keepdims)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_boolean_reduction(self):
    # [FUNCTION DOC]
    # Scenario: logical any and all reductions of a random boolean 2x3x4x5 tensor per iteration,
    # over several axis choices and keep-dimensions settings. Expects pfor and loop to match.
    x = random_ops.random_uniform([2, 3, 4, 5]) > 0.5
    for op in [math_ops.reduce_any, math_ops.reduce_all]:
      for axis in ([1], None, [0, 2], constant_op.constant([1], dtypes.int64)):
        for keepdims in (True, False):

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            # [FUNCTION DOC]
            # HELPER: iteration body. Gathers slice i and returns the current logical reduction of
            # it over the current axis with the current keep-dimensions setting.
            a = array_ops.gather(x, i)
            return op(a, axis=axis, keepdims=keepdims)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_argmin_argmax(self):
    # [FUNCTION DOC]
    # Scenario: argmin and argmax of slices of a random 2x3x4x5 tensor over axes 1, all, and -1,
    # with int32, int64 and default output types. Expects pfor and loop results to match.
    x = random_ops.random_uniform([2, 3, 4, 5])
    for op in [math_ops.argmin, math_ops.argmax]:
      for axis in (1, None, -1):
        for output_dtype in (dtypes.int32, dtypes.int64, None):
          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            # [FUNCTION DOC]
            # HELPER: iteration body. Gathers slice i and returns the current arg-reduction of it
            # over the current axis and output type.
            a = array_ops.gather(x, i)
            return op(a, axis=axis, output_type=output_dtype)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_bucketize(self):
    # [FUNCTION DOC]
    # Scenario: bucketize slices of a random 2x3x4 tensor with boundaries minus one, one half and
    # one. Expects pfor and loop results to match.
    x = random_ops.random_uniform([2, 3, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Gathers slice i and returns its bucket indices.
      a = array_ops.gather(x, i)
      return math_ops.bucketize(a, [-1, 0.5, 1])

    self._test_loop_fn(loop_fn, 2)

  def test_clip_by_value(self):
    # [FUNCTION DOC]
    # Scenario: clip slices of a random 2x3x4 tensor to the range 0.5 to 1.0. Expects pfor and
    # loop results to match.
    x = random_ops.random_uniform([2, 3, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Gathers slice i and returns it clipped to 0.5 through 1.0.
      a = array_ops.gather(x, i)
      return clip_ops.clip_by_value(a, 0.5, 1.0)

    self._test_loop_fn(loop_fn, 2)

  def test_cum_sum(self):
    # [FUNCTION DOC]
    # Scenario: cumulative sum of slices of a random 2x3x4x5 tensor over axes 1, -2 and a tensor
    # axis, in all combinations of exclusive and reverse flags. Expects pfor and loop to match.
    x = random_ops.random_uniform([2, 3, 4, 5])
    for axis in (1, -2, constant_op.constant(1, dtypes.int64)):
      for exclusive in (True, False):
        for reverse in (True, False):

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            # [FUNCTION DOC]
            # HELPER: iteration body. Gathers slice i and returns its cumulative sum with the
            # current axis, exclusive and reverse settings.
            a = array_ops.gather(x, i)
            return math_ops.cumsum(
                a, axis=axis, exclusive=exclusive, reverse=reverse)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_cum_prod(self):
    # [FUNCTION DOC]
    # Scenario: cumulative product of slices of a random 2x3x4x5 tensor over axes 1, -2 and a
    # tensor axis, in all combinations of exclusive and reverse flags. Expects pfor and loop to
    # match.
    x = random_ops.random_uniform([2, 3, 4, 5])
    for axis in (1, -2, constant_op.constant(1, dtypes.int64)):
      for exclusive in (True, False):
        for reverse in (True, False):

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            # [FUNCTION DOC]
            # HELPER: iteration body. Gathers slice i and returns its cumulative product with the
            # current axis, exclusive and reverse settings.
            a = array_ops.gather(x, i)
            return math_ops.cumprod(
                a, axis=axis, exclusive=exclusive, reverse=reverse)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_bias_add(self):
    # [FUNCTION DOC]
    # Scenario: bias add in channels-first and channels-last formats with the value and/or the
    # bias stacked across the loop (at least one stacked). The bias is watched by a persistent
    # gradient tape and the loop returns the output and the gradient of a squared-sum loss with
    # respect to the bias (values and indices when the bias is gathered, which yields a sparse
    # gradient). Expects pfor and loop results to match over 2 iterations.
    # KNOWN ISSUE: the output dtype list built before the comparison is never used (dead code).
    for data_format in ("NCHW", "NHWC"):
      for stacked_value in (True, False):
        x_shape = [3, 4, 5, 6]
        if stacked_value:
          x_shape = [2] + x_shape
        x = random_ops.random_uniform(x_shape)
        for stacked_bias in (True, False):
          if not (stacked_value or stacked_bias):
            continue
          with backprop.GradientTape(persistent=True) as g:
            bias_dim = -1
            if data_format == "NCHW":
              bias_dim = 2 if stacked_value else 1
            bias_shape = [x_shape[bias_dim]]
            if stacked_bias:
              bias_shape = [2] + bias_shape
            bias = random_ops.random_uniform(bias_shape)
            g.watch(bias)

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            # [FUNCTION DOC]
            # HELPER: iteration body. Takes slice i of the value and/or bias when stacked, adds
            # the bias, and computes the gradient of the summed squared output with respect to
            # the bias. Returns output and gradient (gradient values and indices if the bias is
            # stacked, since that gradient is sparse).
            with g:
              a = array_ops.gather(x, i) if stacked_value else x
              b = array_ops.gather(bias, i) if stacked_bias else bias
              y = nn.bias_add(a, b, data_format=data_format)
              loss = math_ops.reduce_sum(y * y)
            grad = g.gradient(loss, bias)
            if stacked_bias:
              # If we gather over bias in loop_fn, the gradient will be an
              # instance of `IndexedSlices` with attrs `values` and `indices`.
              return y, grad.values, grad.indices
            else:
              return y, grad

          # pylint: enable=cell-var-from-loop

          out_dtypes = [dtypes.float32, dtypes.float32]
          if stacked_bias:
            out_dtypes = out_dtypes + [dtypes.int32]
          self._test_loop_fn(loop_fn, 2)

  @parameterized.parameters(
      *itertools.product(
          (
              math_ops.unsorted_segment_mean,
              math_ops.unsorted_segment_sum,
              math_ops.unsorted_segment_min,
              math_ops.unsorted_segment_max,
              math_ops.unsorted_segment_prod,
          ),
          (
              [[0, 0, 2], [0, 1, 2], [2, 2, 2]],
              [[0, 0, 2, -1], [0, 1, 2, -1], [2, 2, 2, -1]],
          ),
      )
  )
  def test_unsorted_segment_reduction(self, reduction_op, indices):
    # [FUNCTION DOC]
    # Parameterized scenario: unsorted segment mean, sum, min, max and prod, with segment ids
    # with and without a negative (dropped) id. Data and segment ids are stacked tensors; segment
    # ids and number of segments are tried as int32 and int64. The loop mixes variant and
    # invariant data and ids. Expects pfor and loop results to match over 3 iterations.
    t = random_ops.random_uniform(constant_op.constant(indices).shape + [2])
    for segment_ids_dtype in (dtypes.int32, dtypes.int64):
      segment_ids = constant_op.constant(indices, dtype=segment_ids_dtype)
      for num_segments_dtype in (dtypes.int32, dtypes.int64):
        num_segments = constant_op.constant(3, dtype=num_segments_dtype)
        # pylint: disable=cell-var-from-loop
        def loop_fn(i):
          # [FUNCTION DOC]
          # HELPER: iteration body. Returns three segment reductions: variant data with variant
          # ids, invariant first data with variant ids, and variant data with invariant first ids.
          data = array_ops.gather(t, i)
          data_0 = array_ops.gather(t, 0)
          seg_ids = array_ops.gather(segment_ids, i)
          seg_ids_0 = array_ops.gather(segment_ids, 0)
          return (reduction_op(data, seg_ids, num_segments),
                  reduction_op(data_0, seg_ids, num_segments),
                  reduction_op(data, seg_ids_0, num_segments))

        # pylint: enable=cell-var-from-loop

        self._test_loop_fn(loop_fn, 3)

  @parameterized.parameters((math_ops.sparse_segment_sum_v2, True),
                            (math_ops.sparse_segment_mean_v2, True),
                            (math_ops.sparse_segment_sqrt_n_v2, True),
                            (math_ops.sparse_segment_sum_v2, False),
                            (math_ops.sparse_segment_mean_v2, False),
                            (math_ops.sparse_segment_sqrt_n_v2, False))
  def test_sparse_segment(self, op_func, with_num_segments):
    # [FUNCTION DOC]
    # Parameterized scenario: sparse segment sum, mean and sqrt-n, with and without an explicit
    # number of segments. Data, indices and segment ids are stacked and the loop mixes variant and
    # invariant arguments. Loop-variant segment ids are only exercised when the number of
    # segments is given. Expects pfor and loop results to match over 3 iterations.
    data = random_ops.random_uniform([3, 4, 2])
    indices = constant_op.constant([[1, 2, 3], [0, 1, 2], [0, 2, 3]])
    seg_ids = constant_op.constant([[0, 0, 2], [1, 1, 1], [0, 1, 1]])
    if with_num_segments:
      num_segments = 3
    else:
      num_segments = None

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Gathers slice i (and the invariant first slice) of data, indices
      # and segment ids and returns the op results for combinations of variant and invariant
      # arguments; adds more with variant segment ids when the number of segments is given.
      data_i = array_ops.gather(data, i)
      data_0 = array_ops.gather(data, 0)
      indices_i = array_ops.gather(indices, i)
      indices_0 = array_ops.gather(indices, 0)
      seg_ids_i = array_ops.gather(seg_ids, i)
      seg_ids_0 = array_ops.gather(seg_ids, 0)
      outputs = [
          op_func(data_0, indices_i, seg_ids_0, num_segments=num_segments),
          op_func(data_i, indices_i, seg_ids_0, num_segments=num_segments),
          op_func(data_0, indices_0, seg_ids_0, num_segments=num_segments),
          op_func(data_i, indices_0, seg_ids_0, num_segments=num_segments)
      ]
      if with_num_segments:
        # For this case, we support loop variant segment_ids as well.
        outputs += [
            op_func(data_0, indices_i, seg_ids_i, num_segments=num_segments),
            op_func(data_i, indices_i, seg_ids_i, num_segments=num_segments),
            op_func(data_0, indices_0, seg_ids_i, num_segments=num_segments),
            op_func(data_i, indices_0, seg_ids_i, num_segments=num_segments)
        ]
      return outputs

    self._test_loop_fn(loop_fn, 3)

  @parameterized.parameters(math_ops.sparse_segment_sum_grad,
                            math_ops.sparse_segment_mean_grad,
                            math_ops.sparse_segment_sqrt_n_grad)
  def test_sparse_segment_grad(self, op_func):
    # [FUNCTION DOC]
    # Parameterized scenario: gradient ops of sparse segment sum, mean and sqrt-n with a stacked
    # random gradient, fixed indices and segment ids, and output size 4. Expects pfor and loop
    # results to match over 3 iterations.
    grad = random_ops.random_uniform([3, 3, 2])
    indices = constant_op.constant([1, 2, 3])
    seg_ids = constant_op.constant([0, 0, 2])
    dim0 = 4

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Gathers slice i of the gradient and returns the segment gradient
      # op applied with the fixed indices, segment ids and output size.
      grad_i = array_ops.gather(grad, i)
      return op_func(grad_i, indices, seg_ids, dim0)

    self._test_loop_fn(loop_fn, 3)

  def test_cast(self):
    # [FUNCTION DOC]
    # Scenario: cast an integer row to float and a float row to integer. Expects pfor and loop
    # results to match.
    x = constant_op.constant([[1], [2]])
    y = constant_op.constant([[1.0], [2.0]])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Returns row i of the integer input cast to float32 and row i of
      # the float input cast to int32.
      return (math_ops.cast(array_ops.gather(x, i), dtypes.float32),
              math_ops.cast(array_ops.gather(y, i), dtypes.int32))

    self._test_loop_fn(loop_fn, 2)

  def test_tanh_axpy(self):
    # [FUNCTION DOC]
    # Scenario: tanh of a scalar times row i of x plus row i of y, where x has 4 rows and y has
    # 6. The number of iterations equals the row count of x, so only the first 4 rows of y are
    # used. Expects pfor and loop results to match.
    a = constant_op.constant(3.)
    x = random_ops.random_uniform([4, 5])
    y = random_ops.random_uniform([6, 5])
    n = x.shape[0]

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Returns tanh of the scalar times row i of x plus row i of y.
      return math_ops.tanh(a * array_ops.gather(x, i) + array_ops.gather(y, i))

    self._test_loop_fn(loop_fn, n)

  def test_select(self):
    # [FUNCTION DOC]
    # Scenario: elementwise select (legacy where) between two stacked 2x3x5 tensors with a
    # random boolean condition of rank one, two or three. Expects pfor and loop to match.
    a = random_ops.random_uniform([2, 3, 5])
    b = random_ops.random_uniform([2, 3, 5])
    for cond_shape in [2], [2, 3], [2, 3, 5]:
      cond = random_ops.random_uniform(cond_shape) > 0.5

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: iteration body. Gathers slice i of both branches and the condition and returns
        # the selected values.
        a_i = array_ops.gather(a, i)
        b_i = array_ops.gather(b, i)
        cond_i = array_ops.gather(cond, i)
        return array_ops.where(cond_i, a_i, b_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)

  def test_selectv2_cond_needs_broadcast(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting select (where version two) with two stacked 2x3x5 branches and a
    # stacked condition whose shape must be broadcast to the branches. Expects pfor and loop to
    # match.
    a = random_ops.random_uniform([2, 3, 5])
    b = random_ops.random_uniform([2, 3, 5])
    # wherev2 assumes all shapes are broadcastable with each other.
    # This means that we can only specify conditions that are
    # broadcastable with [3, 5].
    for cond_shape in [2], [2, 1], [2, 5], [2, 3, 1], [2, 3, 5]:
      cond = random_ops.random_uniform(cond_shape) > 0.5

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: iteration body. Gathers slice i of both branches and the condition and returns
        # the broadcasting select result.
        a_i = array_ops.gather(a, i)
        b_i = array_ops.gather(b, i)
        cond_i = array_ops.gather(cond, i)
        return array_ops.where_v2(cond_i, a_i, b_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)

  def test_selectv2_args_need_broadcast(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting select (where version two) where the first branch (2x5) has lower
    # rank than the second (2x3x5) and the condition shape varies. Expects pfor and loop to match.
    a = random_ops.random_uniform([2, 5])
    b = random_ops.random_uniform([2, 3, 5])
    # wherev2 assumes all shapes are broadcastable with each other.
    # This means that we can only specify conditions that are
    # broadcastable with [3, 5].
    for cond_shape in [2], [2, 1], [2, 5], [2, 3, 1], [2, 3, 5]:
      cond = random_ops.random_uniform(cond_shape) > 0.5

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: iteration body. Gathers slice i of both branches and the condition and returns
        # the broadcasting select result.
        a_i = array_ops.gather(a, i)
        b_i = array_ops.gather(b, i)
        cond_i = array_ops.gather(cond, i)
        return array_ops.where_v2(cond_i, a_i, b_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)

  def test_selectv2_cond_fixed(self):
    # [FUNCTION DOC]
    # Scenario: broadcasting select (where version two) with a loop-invariant 3x5 condition and
    # stacked branches, where the first branch has various broadcastable shapes. Expects pfor and
    # loop to match.
    cond = random_ops.random_uniform([3, 5]) > 0.5
    b = random_ops.random_uniform([2, 3, 5])
    # wherev2 assumes all shapes are broadcastable with each other.
    # This means that we can only specify conditions that are
    # broadcastable with [3, 5].
    for a_shape in [2], [2, 1], [2, 5], [2, 3, 1], [2, 3, 5]:
      a = random_ops.random_uniform(a_shape)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: iteration body. Gathers slice i of both branches and returns the broadcasting
        # select using the fixed (loop-invariant) condition.
        a_i = array_ops.gather(a, i)
        b_i = array_ops.gather(b, i)
        return array_ops.where_v2(cond, a_i, b_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)


@test_util.run_all_in_graph_and_eager_modes
class LinalgTest(PForTestCase):
  # [CLASS DOC]
  # Tests that pfor vectorization of linear algebra kernels (Cholesky, log determinant, inverse,
  # solve, triangular solve, self-adjoint eigendecomposition, einsum) matches a sequential loop.
  # Inherits the pfor test case helper; runs in graph and eager mode.

  def test_cholesky(self):
    # [FUNCTION DOC]
    # Scenario: Cholesky factorization of two random 3x3 positive definite, well-conditioned
    # matrices. Expects pfor and loop results to match.
    z = random_ops.random_normal([2, 3, 3])
    x = (
        math_ops.matmul(z, array_ops.matrix_transpose(z))  # Ensure pos. def.
        + linalg_ops.eye(3))  # Ensure well-conditioned.

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Returns the Cholesky factor of matrix i.
      return linalg_ops.cholesky(array_ops.gather(x, i))

    self._test_loop_fn(loop_fn, 2)

  def test_log_matrix_determinant(self):
    # [FUNCTION DOC]
    # Scenario: sign and log absolute determinant of random 2x2 matrices, with input shapes 3x4x2x2
    # and 3x2x2. Expects pfor and loop results to match over 3 iterations.
    for x_shape in ([3, 4, 2, 2], [3, 2, 2]):
      x = random_ops.random_normal(x_shape)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: iteration body. Returns the sign and log absolute determinant of slice i.
        return linalg_ops.log_matrix_determinant(array_ops.gather(x, i))

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3)

  def test_matrix_inverse(self):
    # [FUNCTION DOC]
    # Scenario: inverse of well-conditioned random 2x2 matrices (a stack of 3x4), with and without
    # the adjoint option. Expects pfor and loop results to match over 2 iterations.
    x = (random_ops.random_uniform([3, 4, 2, 2]) + 10 * linalg_ops.eye(2)
        )  # Ensure well-conditioned.

    for adjoint in (True, False):

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # HELPER: iteration body. Returns the (optionally adjoint) inverse of slice i.
        return linalg_ops.matrix_inverse(
            array_ops.gather(x, i), adjoint=adjoint)

      # pylint: enable=cell-var-from-loop
      self._test_loop_fn(loop_fn, 2)

  def test_matrix_solve(self):
    # [FUNCTION DOC]
    # Scenario: solve linear systems with well-conditioned 3x3 left-hand sides and 3x5 right-hand
    # sides, over combinations of adjoint and of stacking either side across the loop. Expects
    # pfor and loop results to match over 2 iterations.
    for adjoint in (True, False):
      for stack_a in (True, False):
        for stack_b in (True, False):
          shape_a = (2, 4, 3, 3) if stack_a else (4, 3, 3)
          shape_b = (2, 4, 3, 5) if stack_b else (4, 3, 5)
          x = (random_ops.random_uniform(shape_a) + 10 * linalg_ops.eye(3)
              )  # Ensure well-conditioned.
          y = random_ops.random_uniform(shape_b)

          # pylint: disable=cell-var-from-loop
          def loop_fn(i):
            # [FUNCTION DOC]
            # HELPER: iteration body. Takes slice i of a side only if it is stacked and returns
            # the solution of the system with the current adjoint setting.
            a = array_ops.gather(x, i) if stack_a else x
            b = array_ops.gather(y, i) if stack_b else y
            return linalg_ops.matrix_solve(a, b, adjoint=adjoint)

          # pylint: enable=cell-var-from-loop

          self._test_loop_fn(loop_fn, 2)

  def test_matrix_triangular_solve(self):
    # [FUNCTION DOC]
    # Scenario: triangular solve with lower or upper triangular, well-conditioned 3x3 left sides
    # (made triangular by zeroing the other half) and 3x5 right sides, over combinations of
    # lower, adjoint and stacking either side. Expects pfor and loop to match over 2 iterations.
    for lower in (True, False):
      for adjoint in (True, False):
        for stack_a in (True, False):
          for stack_b in (True, False):
            shape_a = (2, 4, 3, 3) if stack_a else (4, 3, 3)
            shape_b = (2, 4, 3, 5) if stack_b else (4, 3, 5)
            x = array_ops.matrix_band_part(
                random_ops.random_uniform(shape_a) +
                linalg_ops.eye(3),  # Ensure well-conditioned.
                *((-1, 0) if lower else (0, -1)))  # Ensure triangular.
            y = random_ops.random_uniform(shape_b)

            # pylint: disable=cell-var-from-loop
            def loop_fn(i):
              # [FUNCTION DOC]
              # HELPER: iteration body. Takes slice i of a side only if it is stacked and returns
              # the triangular solve with the current lower and adjoint settings.
              a = array_ops.gather(x, i) if stack_a else x
              b = array_ops.gather(y, i) if stack_b else y
              return linalg_ops.matrix_triangular_solve(
                  a, b, lower=lower, adjoint=adjoint)

            # pylint: enable=cell-var-from-loop

            self._test_loop_fn(loop_fn, 2)

  def test_self_adjoint_eig(self):
    # [FUNCTION DOC]
    # Scenario: eigendecomposition and eigenvalues of two random symmetric 3x3 matrices. Expects
    # pfor and loop results to match.
    z = random_ops.random_normal([2, 3, 3])
    x = z + array_ops.matrix_transpose(z)  # Ensure self-adjoint.

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Returns the eigendecomposition and the eigenvalues of matrix i.
      return (linalg_ops.self_adjoint_eig(array_ops.gather(x, i)),
              linalg_ops.self_adjoint_eigvals(array_ops.gather(x, i)))

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_without_tensor_float_32(
      "Calls einsum in parallel for-loop and compares result to calling einsum "
      "in sequential for-loop")
  def test_einsum(self):
    # [FUNCTION DOC]
    # Scenario: einsum with a sum reduction, matrix products and a transposing contraction over
    # combinations of loop-variant and loop-invariant operands (10 stacked 9x9 and 9x1 matrices),
    # with TF32 disabled. Expects pfor and loop results to match over 10 iterations.
    b = 10
    x_series = random_ops.random_uniform([b, 9, 9])
    y_series = random_ops.random_uniform([b, 9, 1])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: iteration body. Computes six einsums mixing the invariant first matrices and the
      # variant i-th matrices and returns them as a list.
      x = array_ops.gather(x_series, 0)  # invariant.
      y = array_ops.gather(y_series, 0)  # invariant.
      x_i = array_ops.gather(x_series, i)
      y_i = array_ops.gather(y_series, i)
      z0 = special_math_ops.einsum("ab->b", x_i)
      z1 = special_math_ops.einsum("ab,bc->ac", x_i, y)
      z2 = special_math_ops.einsum("ab,bc->ac", x, y_i)
      z3 = special_math_ops.einsum("ab,bc->ac", x, y)
      z4 = special_math_ops.einsum("ab,bc->ac", x_i, y_i)
      z5 = special_math_ops.einsum("cd,ce->de", y_i, x_i)  # Includes transpose.
      outputs = [z0, z1, z2, z3, z4, z5]
      return outputs

    self._test_loop_fn(loop_fn, b)


if __name__ == "__main__":
  test.main()
