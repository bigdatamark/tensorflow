# [MODULE DOC]
# HELPER: test module; the test cases here are run by the TensorFlow test runner, not imported.
# Tests for the parallel_for control flow API (pfor, for_loop and vectorized_map) in
# tensorflow/python/ops/parallel_for/control_flow_ops.py. Each test case builds a loop body (a
# function of the iteration index, optionally also a pfor configuration object) and has the
# PForTestCase harness (_test_loop_fn / run_and_assert_equal from the package test_util module) run
# it both vectorized with pfor and as a sequential for_loop, then compares results. Test classes
# cover reductions, bitwise, image, neural network, random, logging, TensorArray, TensorList, tensor
# scatter, optional, stack, while loops (v1 and v2), nested control flow, conditionals, RNNs, sparse
# and composite tensors, parsing, function calls, FFTs and variables. It also defines helper
# functions and classes (dynamic LSTM builders, a Particle composite tensor with its type spec) and
# a Benchmarks class. Importing it registers no global state; running it as a script starts the test
# main. Control flow v2 is toggled by some fixtures.
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
"""Tests for pfor and for_loop."""
# pylint: disable=g-direct-tensorflow-import

import functools
import sys
import time

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.signal import fft_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@test_util.run_all_in_graph_and_eager_modes
@test_util.with_control_flow_v2
class PForTest(PForTestCase):
  # [CLASS DOC]
  # Test case covering the core pfor/for_loop/vectorized_map entry points: fallback to while loops,
  # parallel_iterations validation, zero-iteration loops and vectorized_map shape handling. Runs in
  # both graph and eager modes with control flow v2 enabled; uses the PForTestCase fixture whose
  # _test_loop_fn compares vectorized and sequential results.

  def test_op_conversion_fallback_to_while_loop(self):
    # [FUNCTION DOC]
    # Checks that an op with no pfor converter (top_k is used as the example) is still handled when
    # fallback_to_while_loop is enabled. Setup: a random tensor of shape 3 by 2 by 4. Action: run
    # loop_fn over 3 iterations with fallback enabled. The test harness (_test_loop_fn) runs the
    # loop body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts
    # the results are close. Expected: results match. Only valid while top_k has no converter;
    # otherwise another op must be chosen.
    # Note that we used top_k op for this test. If a converter gets defined for
    # it, we will need to find another op for which a converter has not been
    # defined.
    x = random_ops.random_uniform([3, 2, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will take the i-th matrix
      # of the input and compute its top-k values and indices. Returns the per-iteration result or
      # tuple of results that pfor stacks along a new leading dimension; no side effects other than
      # creating graph operations.
      x_i = array_ops.gather(x, i)
      return nn.top_k(x_i)

    self._test_loop_fn(loop_fn, 3, fallback_to_while_loop=True)

  def test_parallel_iterations(self):
    # [FUNCTION DOC]
    # Checks that pfor works with several parallel_iterations chunk sizes (2, 3, 8, 10), with both a
    # static iteration count (8) and a tensor-valued count (4 times a constant 2). For each setting
    # a random 8 by 3 tensor is created and The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close. Expected: vectorized and sequential outputs agree for every chunk size.
    for parallel_iterations in [2, 3, 8, 10]:
      x = random_ops.random_uniform([8, 3])

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # Per-iteration body: returns the i-th row of the input x created in the enclosing loop over
        # parallel_iterations values. Takes the scalar iteration index tensor. Captures x from the
        # surrounding scope (the pylint comment notes the closure over a loop variable is
        # intentional since it is used before the next iteration).
        return array_ops.gather(x, i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 8, parallel_iterations=parallel_iterations)
      self._test_loop_fn(
          loop_fn,
          4 * constant_op.constant(2),
          parallel_iterations=parallel_iterations)

  def test_parallel_iterations_preserves_static_shape(self):
    # [FUNCTION DOC]
    # Verifies that pfor with parallel_iterations set still yields a result with fully known static
    # shape. Action: run pfor over 8 iterations of a function returning a random 2 by 3 tensor, for
    # each chunk size in 2, 3, 8, 10. Asserts the output static shape equals 8 by 2 by 3. Uses a
    # lambda, not the test harness.
    for parallel_iterations in [2, 3, 8, 10]:
      x = pfor_control_flow_ops.pfor(
          lambda _: random_ops.random_uniform([2, 3]),
          8,
          parallel_iterations=parallel_iterations)
      self.assertAllEqual(x.shape, [8, 2, 3])

  def test_parallel_iterations_zero(self):
    # [FUNCTION DOC]
    # Verifies argument validation: parallel_iterations of zero must be rejected. Calling pfor with
    # it expects a ValueError mentioning a positive integer, and calling for_loop with it expects a
    # TypeError with the same message text. Nothing else is asserted.
    with self.assertRaisesRegex(ValueError, "positive integer"):
      pfor_control_flow_ops.pfor(lambda i: 1, 8, parallel_iterations=0)
    with self.assertRaisesRegex(TypeError, "positive integer"):
      pfor_control_flow_ops.for_loop(
          lambda i: 1, dtypes.int32, 8, parallel_iterations=0)

  def test_parallel_iterations_one(self):
    # [FUNCTION DOC]
    # Verifies that pfor rejects parallel_iterations equal to one with a ValueError whose message
    # tells the caller to use for_loop instead.
    with self.assertRaisesRegex(ValueError, "Use `for_loop` instead"):
      pfor_control_flow_ops.pfor(lambda i: 1, 8, parallel_iterations=1)

  def test_zero_loop_iters_basic(self):
    # [FUNCTION DOC]
    # Runs the harness with zero iterations on a function that returns the constant one, checking
    # that an empty loop yields consistent (empty) outputs in the vectorized and sequential
    # versions. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.
    self._test_loop_fn(lambda i: 1, 0)

  def test_zero_loop_iters_tensor(self):
    # [FUNCTION DOC]
    # Runs the harness with zero iterations on a function returning a 10 by 3 int32 zeros tensor,
    # checking the zero-length stacked output has the right shape and dtype in both implementations.
    # The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially
    # with for_loop, evaluates both and asserts the results are close.
    self._test_loop_fn(
        lambda i: array_ops.zeros([10, 3], dtype=dtypes.int32), 0
    )

  def test_vectorized_map(self):
    # [FUNCTION DOC]
    # Checks vectorized_map on a batch of ten 5 by 3 ones tensors with a function that averages over
    # the first axis keeping dimensions. Asserts (via run_and_assert_equal) that the result equals a
    # 10 by 1 by 3 tensor of ones.

    def compute(x):
      # [FUNCTION DOC]
      # Mapped function: reduces its argument by mean over axis 0 keeping the reduced dimension.
      # Input: one element of the batch (tensor). Output: tensor with the leading dimension reduced
      # to size one.
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    result = pfor_control_flow_ops.vectorized_map(compute,
                                                  array_ops.ones((10, 5, 3)))
    self.run_and_assert_equal(result, array_ops.ones((10, 1, 3)))

  def test_vectorized_map_with_dynamic_shape(self):
    # [FUNCTION DOC]
    # Same as the basic vectorized_map test but the input is a placeholder-with-default of unknown
    # static shape, so vectorized_map must determine the batch size dynamically. Asserts the result
    # equals a 10 by 1 by 3 tensor of ones.

    def compute(x):
      # [FUNCTION DOC]
      # Mapped function: mean over axis 0 keeping dimensions, applied to one batch element of
      # dynamic shape. Returns a tensor with a leading dimension of one.
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    x = array_ops.placeholder_with_default(
        array_ops.ones((10, 5, 3)), shape=None)
    result = pfor_control_flow_ops.vectorized_map(compute, x)
    self.run_and_assert_equal(result, array_ops.ones((10, 1, 3)))

  def test_where_shape(self):
    # [FUNCTION DOC]
    # Regression test that vectorized_map over a where operation with a broadcast captured tensor
    # yields an inferred static shape. Defines a tf.function that maps a where(x > 0, x, b) function
    # over a 2 by 1 tensor and returns the result's shape; asserts calling it returns the shape 2 by
    # 1.
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Compiled helper: builds constant a (2 by 1 ones) and b (length 1), vectorizes a
      # where-selection over a, and returns the static shape of the result. Takes no arguments;
      # returns the shape object.
      a = constant_op.constant([[1.], [1.]])
      b = constant_op.constant([1.])
      result = pfor_control_flow_ops.vectorized_map(
          lambda x: array_ops.where(x > 0, x, b), a)
      return result.shape

    self.assertAllEqual([2, 1], f())

  def test_vectorized_map_broadcasts_unit_dimensions(self):
    # [FUNCTION DOC]
    # Checks that vectorized_map broadcasts inputs whose leading dimension is one against inputs
    # with a larger batch. Runs for inputs converted as static-shape tensors and as placeholders
    # with unknown shape. Cases: a single element plus one; two unit-batch elements summed; a
    # unit-batch element summed with a 3-element batch. Asserts each result equals the direct
    # broadcast addition.
    convert_with_static_shape = ops.convert_to_tensor
    convert_with_dynamic_shape = (
        lambda x: array_ops.placeholder_with_default(x, shape=None))

    for convert in (convert_with_static_shape, convert_with_dynamic_shape):
      a = convert([3.1])
      b = convert([-2., 6., 9.])

      # One elem with leading unit dimension.
      a_plus_1 = pfor_control_flow_ops.vectorized_map(lambda a: a + 1, a)
      self.assertAllEqual(*self.evaluate((a_plus_1, a + 1)))

      # Two elems, both with leading unit dimension.
      a_plus_a = pfor_control_flow_ops.vectorized_map(sum, (a, a))
      self.assertAllEqual(*self.evaluate((a_plus_a, a + a)))

      # Elem w/ unit dimension broadcast against elem with batch dim.
      a_plus_b = pfor_control_flow_ops.vectorized_map(sum, (a, b))
      self.assertAllEqual(*self.evaluate((a_plus_b, a + b)))

  def test_vectorized_map_example_1(self):
    # [FUNCTION DOC]
    # Documentation example check: vectorized_map computes an outer product (tensordot with zero
    # axes) for each of 100 32 by 32 ones matrices. Asserts the static result shape is 100 by 32 by
    # 32 by 32 by 32.

    def outer_product(a):
      # [FUNCTION DOC]
      # Mapped function: outer product of its argument with itself (tensordot with zero contraction
      # axes). Input: one matrix. Output: a tensor whose shape is the concatenation of both input
      # shapes.
      return math_ops.tensordot(a, a, 0)

    batch_size = 100
    a = array_ops.ones((batch_size, 32, 32))
    c = pfor_control_flow_ops.vectorized_map(outer_product, a)
    self.assertAllEqual((batch_size, 32, 32, 32, 32), c.shape)

  def test_disable_tf_function(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if an assertion fails the flag is not restored (no try/finally), which can leak
    # into later tests.
    # Verifies vectorized_map is unaffected by globally forcing functions to run eagerly. Side
    # effect: turns run-functions-eagerly on, asserts it is on, runs vectorized_map squaring
    # range(4) expecting 0, 1, 4, 9, asserts the flag is still on, then turns it off.
    def_function.run_functions_eagerly(True)
    # vectorized_map should ignore disabling tf.functions
    self.assertTrue(def_function.functions_run_eagerly())
    self.assertAllEqual([0, 1, 4, 9],
                        pfor_control_flow_ops.vectorized_map(
                            lambda x: x * x, math_ops.range(4)))
    self.assertTrue(def_function.functions_run_eagerly())
    def_function.run_functions_eagerly(False)


@test_util.run_all_in_graph_and_eager_modes
class IndexedSlicesTest(PForTestCase):
  # [CLASS DOC]
  # Test case checking that pfor can return IndexedSlices values from the loop body (as a whole or
  # as components). Runs in graph and eager modes.

  def test_indexed_slices(self):
    # [FUNCTION DOC]
    # Loop body returns an IndexedSlices built from the iteration index (indices), a one-element
    # reshaped value and dense shape 3 by 1; run for 2 iterations through the harness. The test
    # harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with
    # for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will build an
      # IndexedSlices whose index and value are the iteration index. Returns the per-iteration
      # result or tuple of results that pfor stacks along a new leading dimension; no side effects
      # other than creating graph operations.
      return indexed_slices.IndexedSlices(
          indices=i, values=array_ops.reshape(i, [1]), dense_shape=[3, 1])

    self._test_loop_fn(loop_fn, 2)

  def test_indexed_slices_components(self):
    # [FUNCTION DOC]
    # Same as the IndexedSlices test, but the loop body returns the values and indices components
    # rather than the object, which avoids densification. Run for 2 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will build an
      # IndexedSlices and return only its values and indices tensors. Returns the per-iteration
      # result or tuple of results that pfor stacks along a new leading dimension; no side effects
      # other than creating graph operations.
      slices = indexed_slices.IndexedSlices(
          indices=i, values=array_ops.reshape(i, [1]), dense_shape=[3, 1])
      # Note that returning the components inside the slice avoids
      # densification, which may be more efficient.
      return slices.values, slices.indices

    self._test_loop_fn(loop_fn, 2)


@test_util.run_all_in_graph_and_eager_modes
class ReductionTest(PForTestCase):
  # [CLASS DOC]
  # Test case for the pfor_config reduction helpers (reduce, reduce_concat, reduce_mean, reduce_sum)
  # that let loop bodies perform cross-iteration reductions. Runs in graph and eager modes.

  def test_reduce(self):
    # [FUNCTION DOC]
    # Tests pfor_config.reduce with a user reduction function (mean of the sum of two arguments over
    # axis 0) on two random 4 by 3 by 2 inputs. Loop body reduces slices across iterations and adds
    # its own slice. Asserts pfor output equals the same computation done on the full tensors,
    # comparing evaluated values.

    def reduce_fn(p, q):
      # [FUNCTION DOC]
      # Reduction callback: given two stacked tensors p and q returns the mean over axis 0 of their
      # sum. Pure function; used both by pfor_config.reduce and to compute the expected answer.
      return math_ops.reduce_mean(p + q, axis=0)

    x = random_ops.random_uniform([4, 3, 2])
    y = random_ops.random_uniform([4, 3, 2])

    def loop_fn(i, pfor_config):
      # [FUNCTION DOC]
      # Loop body taking the iteration index and the pfor configuration object. Gathers row i of x
      # and y, calls the config's reduce with reduce_fn on them, and returns the reduced value plus
      # x_i.
      x_i = array_ops.gather(x, i)
      y_i = array_ops.gather(y, i)
      reduced = pfor_config.reduce(reduce_fn, x_i, y_i)
      return reduced + x_i

    output = pfor_control_flow_ops.pfor(loop_fn, 4)
    ans = reduce_fn(x, y) + x
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_reduce_concat(self):
    # [FUNCTION DOC]
    # Tests pfor_config.reduce_concat: each iteration receives the concatenation of all iterations'
    # rows, takes its mean over axis 0 and subtracts it from its own row. Asserts pfor output equals
    # x minus the column mean of x for a random 8 by 3 input.
    x = random_ops.random_uniform([8, 3])

    def loop_fn(i, pfor_config):
      # [FUNCTION DOC]
      # Loop body with iteration index and pfor config. Gathers row i, obtains the concatenated rows
      # of all iterations, takes their mean over axis 0 and returns the row minus that mean.
      x_i = array_ops.gather(x, i)
      vectorized_value = pfor_config.reduce_concat(x_i)
      mean_value = math_ops.reduce_mean(vectorized_value, axis=0)
      return x_i - mean_value

    output = pfor_control_flow_ops.pfor(loop_fn, 8)
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_reduce_mean(self):
    # [FUNCTION DOC]
    # Tests pfor_config.reduce_mean: each iteration subtracts the mean over all iterations from its
    # own row of a random 8 by 3 tensor. Asserts equality with x minus its column-wise mean.
    x = random_ops.random_uniform([8, 3])

    def loop_fn(i, pfor_config):
      # [FUNCTION DOC]
      # Loop body with iteration index and pfor config: returns row i minus the cross-iteration mean
      # of the rows.
      x_i = array_ops.gather(x, i)
      return x_i - pfor_config.reduce_mean(x_i)

    output = pfor_control_flow_ops.pfor(loop_fn, 8)
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_reduce_sum(self):
    # [FUNCTION DOC]
    # Tests pfor_config.reduce_sum: each iteration subtracts the sum over all iterations from its
    # row of a random 8 by 3 tensor. Asserts equality with x minus its column-wise sum.
    x = random_ops.random_uniform([8, 3])

    def loop_fn(i, pfor_config):
      # [FUNCTION DOC]
      # Loop body with iteration index and pfor config: returns row i minus the cross-iteration sum
      # of the rows.
      x_i = array_ops.gather(x, i)
      return x_i - pfor_config.reduce_sum(x_i)

    output = pfor_control_flow_ops.pfor(loop_fn, 8)
    ans = x - math_ops.reduce_sum(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_reduce_class(self):
    # [FUNCTION DOC]
    # Checks that pfor accepts a callable object (instance of a local class defining a call method)
    # as the loop function when it takes a pfor config. Asserts the output equals x minus its column
    # mean for a random 8 by 3 input.
    x = random_ops.random_uniform([8, 3])

    class LoopFn:
      # [CLASS DOC]
      # Local callable class used as a loop function: has an empty constructor and a call method
      # taking the iteration index and pfor config, and returns the row minus the cross-iteration
      # mean. Captures x from the enclosing test.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor; does nothing (the instance holds no state).
        pass

      def __call__(self, i, pfor_config):
        # [FUNCTION DOC]
        # Loop body invoked by pfor with the iteration index and pfor config: gathers row i of the
        # captured x and returns it minus the cross-iteration mean computed by the config's
        # reduce_mean.
        x_i = array_ops.gather(x, i)
        return x_i - pfor_config.reduce_mean(x_i)

    output = pfor_control_flow_ops.pfor(LoopFn(), 8)
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_reduce_functools_partial(self):
    # [FUNCTION DOC]
    # Checks that pfor works when the loop function is a functools.partial that pre-binds a keyword
    # argument, with pfor config still passed positionally. Asserts the output equals x minus the
    # column mean of a random 8 by 3 input.
    x = random_ops.random_uniform([8, 3])

    def fn(i, pfor_config, dummy=None):
      # [FUNCTION DOC]
      # Loop function with parameters iteration index, pfor config and an optional dummy keyword
      # (ignored and deleted). Returns row i of captured x minus the cross-iteration mean.
      del dummy
      x_i = array_ops.gather(x, i)
      return x_i - pfor_config.reduce_mean(x_i)

    loop_fn = functools.partial(fn, dummy=1)
    output = pfor_control_flow_ops.pfor(loop_fn, 8)
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)

  def test_parallel_iterations(self):
    # [FUNCTION DOC]
    # Verifies that combining pfor_config reductions with parallel_iterations is rejected: calling
    # pfor with parallel_iterations of 2 and a loop body using reduce_sum must raise a ValueError
    # saying parallel_iterations is currently unsupported.
    x = random_ops.random_uniform([8, 3])

    def loop_fn(i, pfor_config):
      # [FUNCTION DOC]
      # Loop body with iteration index and pfor config: returns the cross-iteration sum of the
      # gathered rows; used only to trigger the unsupported-combination error.
      x_i = array_ops.gather(x, i)
      return pfor_config.reduce_sum(x_i)

    with self.assertRaisesRegex(ValueError,
                                "`parallel_iterations` currently unsupported"):
      pfor_control_flow_ops.pfor(loop_fn, 8, parallel_iterations=2)

  def test_var_loop_len(self):
    # [FUNCTION DOC]
    # Checks reductions work when the iteration count is a placeholder (unknown at graph build
    # time). Skipped under eager execution. Builds pfor with a placeholder count and runs it in a
    # cached session feeding 8; only asserts it runs without error.
    if context.executing_eagerly():
      self.skipTest("Variable length not possible under eager execution.")

    x = random_ops.random_uniform([8, 3])

    def loop_fn(i, pfor_config):
      # [FUNCTION DOC]
      # Loop body with iteration index and pfor config: returns the cross-iteration sum of row i of
      # x.
      return pfor_config.reduce_sum(array_ops.gather(x, i))

    num_iters = array_ops.placeholder(dtypes.int32)
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    with self.cached_session() as sess:
      sess.run(pfor, feed_dict={num_iters: 8})


@test_util.run_all_in_graph_and_eager_modes
class BitwiseTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of bitwise operations (unary and binary element-wise). Runs in
  # graph and eager modes.

  def test_unary_cwise(self):
    # [FUNCTION DOC]
    # Runs the harness for the unary bitwise invert op on a random int32 tensor of shape 7 by 3 by
    # 5, over 3 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with
    # pfor and sequentially with for_loop, evaluates both and asserts the results are close.
    for op in [bitwise_ops.invert]:
      x = random_ops.random_uniform([7, 3, 5], maxval=10, dtype=dtypes.int32)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # Per-iteration body: gathers slice i of x and applies the current unary op from the
        # enclosing loop (closure over a loop variable, intentionally). Returns the result.
        x1 = array_ops.gather(x, i)
        return op(x1)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 3)

  def test_binary_cwise(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the output_dtypes list is filled in the loop body but never read afterwards, so
    # it has no effect on the test.
    # Runs the harness for each binary bitwise op (and, or, xor, left shift, right shift) on random
    # int32 inputs, with each op applied to combinations of loop-invariant and loop-variant
    # operands. Over 3 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    binary_ops = [
        bitwise_ops.bitwise_and,
        bitwise_ops.bitwise_or,
        bitwise_ops.bitwise_xor,
        bitwise_ops.left_shift,
        bitwise_ops.right_shift,
    ]
    for op in binary_ops:
      x = random_ops.random_uniform([7, 3, 5], maxval=10, dtype=dtypes.int32)
      y = random_ops.random_uniform([3, 5], maxval=10, dtype=dtypes.int32)

      output_dtypes = []

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # Per-iteration body: gathers slice i of x and y and applies the current op to five operand
        # combinations (both invariant, each variant, both variant, variant with itself). Clears and
        # refills the enclosing output_dtypes list (side effect on a captured list). Returns the
        # list of five results.
        x1 = array_ops.gather(x, i)
        y1 = array_ops.gather(y, i)
        outputs = [op(x, y), op(x1, y), op(x, y1), op(x1, y1), op(x1, x1)]
        del output_dtypes[:]
        output_dtypes.extend(t.dtype for t in outputs)
        return outputs

      # pylint: enable=cell-var-from-loop
      self._test_loop_fn(loop_fn, 3)


@test_util.run_all_in_graph_and_eager_modes
class ImageTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of image adjustment ops (contrast, hue, saturation). Runs in graph
  # and eager modes.

  def test_adjust_contrast(self):
    # [FUNCTION DOC]
    # Runs the harness on adjust_contrast with factor 2.0 over a random 3 by 2 by 4 by 4 by 3 batch,
    # 3 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor
    # and sequentially with for_loop, evaluates both and asserts the results are close.
    images = random_ops.random_uniform([3, 2, 4, 4, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will apply
      # adjust_contrast with factor 2.0 to the i-th image batch. Returns the per-iteration result or
      # tuple of results that pfor stacks along a new leading dimension; no side effects other than
      # creating graph operations.
      image = array_ops.gather(images, i)
      return image_ops.adjust_contrast(image, 2.0)

    self._test_loop_fn(loop_fn, 3)

  def test_adjust_hue(self):
    # [FUNCTION DOC]
    # Runs the harness on adjust_hue with delta 0.25 over a random 3 by 2 by 4 by 4 by 3 batch, 3
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.
    images = random_ops.random_uniform([3, 2, 4, 4, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will apply adjust_hue
      # with delta 0.25 to the i-th image batch. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      image = array_ops.gather(images, i)
      return image_ops.adjust_hue(image, .25)

    self._test_loop_fn(loop_fn, 3)

  def test_adjust_saturation(self):
    # [FUNCTION DOC]
    # Runs the harness on adjust_saturation with factor 0.1 over a random 3 by 2 by 4 by 4 by 3
    # batch, 3 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with
    # pfor and sequentially with for_loop, evaluates both and asserts the results are close.
    images = random_ops.random_uniform([3, 2, 4, 4, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will apply
      # adjust_saturation with factor 0.1 to the i-th image batch. Returns the per-iteration result
      # or tuple of results that pfor stacks along a new leading dimension; no side effects other
      # than creating graph operations.
      image = array_ops.gather(images, i)
      return image_ops.adjust_saturation(image, 0.1)

    self._test_loop_fn(loop_fn, 3)


@test_util.run_all_in_graph_and_eager_modes
@test_util.run_all_without_tensor_float_32("Uses matmul")
class NNTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of neural-network ops: convolutions and their backprops, depthwise
  # convolutions, roll, pooling, batch norm, softmax and cross-entropy losses. Runs in graph and
  # eager modes with TF32 disabled for matmul-like ops.

  def test_conv2d(self):
    # [FUNCTION DOC]
    # Runs the harness on conv2d (stride 2, VALID, NHWC) with a loop-variant input slice from a
    # random 3 by 2 by 12 by 12 by 3 tensor and a loop-invariant 3 by 3 by 3 by 7 filter. 3
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    filt = random_ops.random_uniform([3, 3, 3, 7])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will convolve the i-th
      # input slice with the invariant filter. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      x1 = array_ops.gather(x, i)
      return nn.conv2d(
          x1, filt, strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_conv2d_backprop_input(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test immediately skips itself (nightly GPU build bug b/262851489), so the
    # remaining body never executes.
    # Intended to run the harness on conv2d_backprop_input with loop-variant output gradients.
    self.skipTest("b/262851489: Fix nightly build for GPU.")
    x_shape = [2, 12, 12, 3]
    filt = random_ops.random_uniform([3, 3, 3, 7])
    grad = random_ops.random_uniform([3, 2, 5, 5, 7])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute the conv2d
      # input gradient from the i-th output gradient. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      grad1 = array_ops.gather(grad, i)
      return nn.conv2d_backprop_input(
          x_shape,
          filt,
          grad1,
          strides=[1, 2, 2, 1],
          padding="VALID",
          data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_conv2d_backprop_filter(self):
    # [FUNCTION DOC]
    # Runs the harness on conv2d_backprop_filter for both a loop-variant input and a loop-invariant
    # input (slice 0) against loop-variant output gradients. 3 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    x_0 = array_ops.gather(x, 0)
    filter_sizes = [3, 3, 3, 7]
    grad = random_ops.random_uniform([3, 2, 5, 5, 7])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute filter
      # gradients for the i-th input slice and for the invariant slice 0. Returns the per-iteration
      # result or tuple of results that pfor stacks along a new leading dimension; no side effects
      # other than creating graph operations.
      x_i = array_ops.gather(x, i)
      grad_i = array_ops.gather(grad, i)
      return [
          nn.conv2d_backprop_filter(
              inp,
              filter_sizes,
              grad_i,
              strides=[1, 2, 2, 1],
              padding="VALID",
              data_format="NHWC") for inp in [x_i, x_0]
      ]

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native(self):
    # [FUNCTION DOC]
    # Runs the harness on depthwise_conv2d_native (NHWC, stride 2) with loop-variant input and
    # loop-variant filter. 3 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    filt = random_ops.random_uniform([3, 3, 3, 3, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will apply depthwise
      # convolution to the i-th input and i-th filter. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      x1 = array_ops.gather(x, i)
      filt1 = array_ops.gather(filt, i)
      return nn.depthwise_conv2d_native(
          x1, filt1, strides=[1, 2, 2, 1], padding="VALID", data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native_backprop_input(self):
    # [FUNCTION DOC]
    # Runs the harness on depthwise_conv2d_native_backprop_input (NHWC) with loop-variant filter and
    # output gradient and a fixed input shape. 3 iterations. The test harness (_test_loop_fn) runs
    # the loop body both vectorized with pfor and sequentially with for_loop, evaluates both and
    # asserts the results are close.
    x_shape = [2, 12, 12, 3]
    filt = random_ops.random_uniform([3, 3, 3, 3, 2])
    grad = random_ops.random_uniform([3, 2, 5, 5, 6])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute the
      # depthwise input gradient from the i-th filter and gradient. Returns the per-iteration result
      # or tuple of results that pfor stacks along a new leading dimension; no side effects other
      # than creating graph operations.
      grad1 = array_ops.gather(grad, i)
      filt1 = array_ops.gather(filt, i)
      return nn.depthwise_conv2d_native_backprop_input(
          x_shape,
          filt1,
          grad1,
          strides=[1, 2, 2, 1],
          padding="VALID",
          data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native_backprop_filter(self):
    # [FUNCTION DOC]
    # Runs the harness on depthwise_conv2d_native_backprop_filter (NHWC) with loop-variant input and
    # gradient and fixed filter sizes. 3 iterations. The test harness (_test_loop_fn) runs the loop
    # body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.
    x = random_ops.random_uniform([3, 2, 12, 12, 3])
    filter_sizes = [3, 3, 3, 2]
    grad = random_ops.random_uniform([3, 2, 5, 5, 6])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute the
      # depthwise filter gradient from the i-th input and gradient. Returns the per-iteration result
      # or tuple of results that pfor stacks along a new leading dimension; no side effects other
      # than creating graph operations.
      x_i = array_ops.gather(x, i)
      grad_i = array_ops.gather(grad, i)
      return nn.depthwise_conv2d_native_backprop_filter(
          x_i,
          filter_sizes,
          grad_i,
          strides=[1, 2, 2, 1],
          padding="VALID",
          data_format="NHWC")

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native_nchw(self):
    # [FUNCTION DOC]
    # Same as the depthwise convolution test but in NCHW layout; skipped when no GPU is available
    # because NCHW only works on GPU. 3 iterations. The test harness (_test_loop_fn) runs the loop
    # body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.
    if not test_util.is_gpu_available():
      self.skipTest("NCHW only works on GPU")
    x = random_ops.random_uniform([3, 2, 3, 12, 12])
    filt = random_ops.random_uniform([3, 3, 3, 3, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will apply NCHW depthwise
      # convolution to the i-th input and i-th filter. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      x1 = array_ops.gather(x, i)
      filt1 = array_ops.gather(filt, i)
      return nn.depthwise_conv2d_native(
          x1, filt1, strides=[1, 1, 2, 2], padding="VALID", data_format="NCHW")

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native_backprop_input_nchw(self):
    # [FUNCTION DOC]
    # NCHW variant of the depthwise input-gradient test; skipped without a GPU. 3 iterations. The
    # test harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially
    # with for_loop, evaluates both and asserts the results are close.
    if not test_util.is_gpu_available():
      self.skipTest("NCHW only works on GPU")
    x_shape = [2, 3, 12, 12]
    filt = random_ops.random_uniform([3, 3, 3, 3, 2])
    grad = random_ops.random_uniform([3, 2, 6, 5, 5])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute the NCHW
      # depthwise input gradient from the i-th filter and gradient. Returns the per-iteration result
      # or tuple of results that pfor stacks along a new leading dimension; no side effects other
      # than creating graph operations.
      grad1 = array_ops.gather(grad, i)
      filt1 = array_ops.gather(filt, i)
      return nn.depthwise_conv2d_native_backprop_input(
          x_shape,
          filt1,
          grad1,
          strides=[1, 1, 2, 2],
          padding="VALID",
          data_format="NCHW")

    self._test_loop_fn(loop_fn, 3)

  def test_depthwise_conv2d_native_backprop_filter_nchw(self):
    # [FUNCTION DOC]
    # NCHW variant of the depthwise filter-gradient test; skipped without a GPU. 3 iterations. The
    # test harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially
    # with for_loop, evaluates both and asserts the results are close.
    if not test_util.is_gpu_available():
      self.skipTest("NCHW only works on GPU")
    x = random_ops.random_uniform([3, 2, 3, 12, 12])
    filter_sizes = [3, 3, 3, 2]
    grad = random_ops.random_uniform([3, 2, 6, 5, 5])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute the NCHW
      # depthwise filter gradient from the i-th input and gradient. Returns the per-iteration result
      # or tuple of results that pfor stacks along a new leading dimension; no side effects other
      # than creating graph operations.
      x_i = array_ops.gather(x, i)
      grad_i = array_ops.gather(grad, i)
      return nn.depthwise_conv2d_native_backprop_filter(
          x_i,
          filter_sizes,
          grad_i,
          strides=[1, 1, 2, 2],
          padding="VALID",
          data_format="NCHW")

    self._test_loop_fn(loop_fn, 3)

  def test_roll(self):
    # [FUNCTION DOC]
    # Runs the harness on manip roll with a constant shift of 3 along axis 1 of a random 3 by 6 by 7
    # tensor. 3 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with
    # pfor and sequentially with for_loop, evaluates both and asserts the results are close.
    x = random_ops.random_uniform([3, 6, 7])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will roll slice i by 3
      # along axis 1. Returns the per-iteration result or tuple of results that pfor stacks along a
      # new leading dimension; no side effects other than creating graph operations.
      x_i = array_ops.gather(x, i)
      return manip_ops.roll(x_i, 3, axis=1)

    self._test_loop_fn(loop_fn, 3)

  def test_ensure_shape(self):
    # [FUNCTION DOC]
    # Runs the harness on ensure_shape asserting the per-iteration slice has shape 6 by 7, over a
    # random 3 by 6 by 7 tensor, 3 iterations. The test harness (_test_loop_fn) runs the loop body
    # both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.
    x = random_ops.random_uniform([3, 6, 7])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will assert that slice i
      # has static shape 6 by 7 and return it. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      x_i = array_ops.gather(x, i)
      return array_ops.ensure_shape(x_i, [6, 7])

    self._test_loop_fn(loop_fn, 3)

  def test_loop_variant_roll_shift(self):
    # [FUNCTION DOC]
    # Runs the harness on roll where the shift amounts depend on the iteration index (i minus 2,
    # minus 1, and i along axes 1, 2, 2) on a random 3 by 5 by 6 by 7 tensor. 3 iterations. The test
    # harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with
    # for_loop, evaluates both and asserts the results are close.
    x = random_ops.random_uniform([3, 5, 6, 7])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will roll slice i by
      # loop-dependent shifts along axes 1, 2, 2. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      x_i = array_ops.gather(x, i)
      return manip_ops.roll(x_i, [i - 2, -1, i], axis=[1, 2, 2])

    self._test_loop_fn(loop_fn, 3)

  def test_loop_variant_roll_scalar_shift(self):
    # [FUNCTION DOC]
    # Runs the harness on roll where a scalar shift equals the iteration index along axis 0 of a
    # random 5 by 5 by 6 tensor. 5 iterations. The test harness (_test_loop_fn) runs the loop body
    # both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.
    x = random_ops.random_uniform([5, 5, 6])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will roll slice i by i
      # along axis 0. Returns the per-iteration result or tuple of results that pfor stacks along a
      # new leading dimension; no side effects other than creating graph operations.
      x_i = array_ops.gather(x, i)
      return manip_ops.roll(x_i, i, axis=0)

    self._test_loop_fn(loop_fn, 5)

  def test_avg_pool(self):
    # [FUNCTION DOC]
    # Runs the harness on avg_pool (3 by 3 window, stride 2, VALID, NHWC) together with its
    # gradient, using a persistent gradient tape watching the input. Loop body returns the pooled
    # output and gradient of its l2 loss. 3 iterations. The test harness (_test_loop_fn) runs the
    # loop body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts
    # the results are close.
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([3, 2, 12, 12, 3])
      g.watch(x)
      ksize = [1, 3, 3, 1]

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will pool slice i inside
      # the persistent tape and return the output plus the gradient of its l2 loss. Returns the
      # per-iteration result or tuple of results that pfor stacks along a new leading dimension; no
      # side effects other than creating graph operations.
      with g:
        x1 = array_ops.gather(x, i)
        output = nn.avg_pool(
            x1,
            ksize,
            strides=[1, 2, 2, 1],
            padding="VALID",
            data_format="NHWC")
        loss = nn.l2_loss(output)
      return output, g.gradient(loss, x1)

    self._test_loop_fn(loop_fn, 3)

  def test_avg_pool3d(self):
    # [FUNCTION DOC]
    # Same as avg_pool but with 3-D average pooling (2 by 2 by 2 window, stride 2, NDHWC) on a
    # random 5 by 3 by 7 by 6 by 6 by 5 tensor; returns output and gradient. 3 iterations. The test
    # harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with
    # for_loop, evaluates both and asserts the results are close.
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([5, 3, 7, 6, 6, 5])
      g.watch(x)
      ksize = [1, 2, 2, 2, 1]
      strides = [1, 2, 2, 2, 1]

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will 3-D pool slice i
      # inside the persistent tape and return the output plus the gradient of its l2 loss. Returns
      # the per-iteration result or tuple of results that pfor stacks along a new leading dimension;
      # no side effects other than creating graph operations.
      with g:
        x1 = array_ops.gather(x, i)
        output = nn.avg_pool3d(
            x1, ksize, strides=strides, padding="VALID", data_format="NDHWC")
        loss = nn.l2_loss(output)
      return output, g.gradient(loss, x1)

    self._test_loop_fn(loop_fn, 3)

  def test_max_pool(self):
    # [FUNCTION DOC]
    # Runs the harness on max_pool with a second-order gradient: loop body computes the pooled
    # output, the gradient of its l2 loss with respect to the input using ones as output gradients,
    # and the gradient of that gradient with respect to the ones. Returns all three. 3 iterations.
    # The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially
    # with for_loop, evaluates both and asserts the results are close.
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([3, 2, 12, 12, 3])
      g.watch(x)
      ksize = [1, 3, 3, 1]
      strides = [1, 2, 2, 1]

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will max-pool slice i and
      # return output, gradient and gradient-of-gradient. Returns the per-iteration result or tuple
      # of results that pfor stacks along a new leading dimension; no side effects other than
      # creating graph operations.
      with g:
        x1 = array_ops.gather(x, i)
        output = nn.max_pool(
            x1, ksize, strides=strides, padding="VALID", data_format="NHWC")
        loss = nn.l2_loss(output)
        ones = array_ops.ones_like(output)
        g.watch(ones)
        grad = g.gradient(loss, x1, output_gradients=ones)
      grad_grad = g.gradient(grad, ones)
      return output, grad, grad_grad

    self._test_loop_fn(loop_fn, 3)

  def test_max_pool_v2(self):
    # [FUNCTION DOC]
    # Same as the max_pool test but calling the raw MaxPoolV2 op (ksize and strides passed as
    # inputs). Returns output, gradient and second-order gradient. 3 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([3, 2, 12, 12, 3])
      g.watch(x)
      ksize = [1, 3, 3, 1]
      strides = [1, 2, 2, 1]

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will max-pool slice i
      # with the V2 op and return output, gradient and gradient-of-gradient. Returns the
      # per-iteration result or tuple of results that pfor stacks along a new leading dimension; no
      # side effects other than creating graph operations.
      with g:
        x1 = array_ops.gather(x, i)
        output = gen_nn_ops.max_pool_v2(
            x1, ksize, strides=strides, padding="VALID", data_format="NHWC")
        loss = nn.l2_loss(output)
        ones = array_ops.ones_like(output)
        g.watch(ones)
        grad = g.gradient(loss, x1, output_gradients=ones)
      grad_grad = g.gradient(grad, ones)
      return output, grad, grad_grad

    self._test_loop_fn(loop_fn, 3)

  def test_max_pool3d(self):
    # [FUNCTION DOC]
    # Same as the max_pool test but with 3-D max pooling on a random 3 by 3 by 2 by 12 by 12 by 3
    # tensor (window 1 by 3 by 3, stride 2). Returns output, gradient and second-order gradient. 3
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.
    with backprop.GradientTape(persistent=True) as g:
      x = random_ops.random_uniform([3, 3, 2, 12, 12, 3])
      g.watch(x)
      ksize = [1, 1, 3, 3, 1]
      strides = [1, 1, 2, 2, 1]

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will 3-D max-pool slice i
      # and return output, gradient and gradient-of-gradient. Returns the per-iteration result or
      # tuple of results that pfor stacks along a new leading dimension; no side effects other than
      # creating graph operations.
      with g:
        x1 = array_ops.gather(x, i)
        output = nn.max_pool3d(
            x1, ksize, strides=strides, padding="VALID", data_format="NDHWC")
        loss = nn.l2_loss(output)
        ones = array_ops.ones_like(output)
        g.watch(ones)
        grad = g.gradient(loss, x1, output_gradients=ones)
      grad_grad = g.gradient(grad, ones)
      return output, grad, grad_grad

    self._test_loop_fn(loop_fn, 3)

  def test_fused_batch_norm(self):
    # [FUNCTION DOC]
    # Runs the harness on fused_batch_norm for training and inference modes and for NHWC (plus NCHW
    # when a GPU exists). Setup: watched random inputs, scale and offset, and for inference random
    # mean and variance. Loop body returns the outputs plus gradients wrt input, scale and offset
    # (zeros when not training). In inference mode the batch mean and variance outputs are replaced
    # by zeros because CPU and GPU differ. 3 iterations per combination. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.
    data_formats = ["NHWC"]
    if test.is_gpu_available():
      data_formats.append("NCHW")
    for is_training in (True, False):
      for data_format in data_formats:
        with backprop.GradientTape(persistent=True) as g:
          if data_format == "NCHW":
            x = random_ops.random_uniform([3, 1, 2, 5, 5])
          else:
            x = random_ops.random_uniform([3, 1, 5, 5, 2])
          g.watch(x)
          scale = random_ops.random_uniform([2])
          g.watch(scale)
          offset = random_ops.random_uniform([2])
          g.watch(offset)
          mean = None if is_training else random_ops.random_uniform([2])
          variance = None if is_training else random_ops.random_uniform([2])

        # pylint: disable=cell-var-from-loop
        def loop_fn(i):
          # [FUNCTION DOC]
          # Per-iteration body: runs fused batch norm (epsilon 0.01) on slice i with the enclosing
          # scale, offset and optional mean/variance inside the persistent tape; zeros out batch
          # statistics outputs when not training; returns outputs concatenated with gradients of the
          # l2 loss of the first output (or constant zeros when not training). Closes over the loop
          # variables on purpose.
          with g:
            x1 = array_ops.gather(x, i)
            outputs = nn.fused_batch_norm(
                x1,
                scale,
                offset,
                mean=mean,
                variance=variance,
                epsilon=0.01,
                data_format=data_format,
                is_training=is_training)
            outputs = list(outputs)
            # We only test the first value of outputs when is_training is
            # False. It looks like CPU and GPU have different outputs for
            # batch_mean and batch_variance for this case.
            if not is_training:
              outputs[1] = constant_op.constant(0.)
              outputs[2] = constant_op.constant(0.)
            loss = nn.l2_loss(outputs[0])
          if is_training:
            gradients = g.gradient(loss, [x1, scale, offset])
          else:
            gradients = [constant_op.constant(0.)] * 3
          return outputs + gradients

        # pylint: enable=cell-var-from-loop

        self._test_loop_fn(loop_fn, 3)

  def test_log_softmax(self):
    # [FUNCTION DOC]
    # Runs the harness on log_softmax with the default axis, axis 0 and axis -1 on a random 3 by 2
    # by 4 tensor. 3 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized
    # with pfor and sequentially with for_loop, evaluates both and asserts the results are close.
    logits = random_ops.random_uniform([3, 2, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute log_softmax
      # of slice i over default, 0 and last axes. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      logits_i = array_ops.gather(logits, i)
      return (nn.log_softmax(logits_i), nn.log_softmax(logits_i, axis=0),
              nn.log_softmax(logits_i, axis=-1))

    self._test_loop_fn(loop_fn, 3)

  def test_softmax(self):
    # [FUNCTION DOC]
    # Runs the harness on softmax with the default axis, axis 0 and axis -1 on a random 3 by 2 by 4
    # tensor. 3 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with
    # pfor and sequentially with for_loop, evaluates both and asserts the results are close.
    logits = random_ops.random_uniform([3, 2, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute softmax of
      # slice i over default, 0 and last axes. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      logits_i = array_ops.gather(logits, i)
      return (nn.softmax(logits_i), nn.softmax(logits_i, axis=0),
              nn.softmax(logits_i, axis=-1))

    self._test_loop_fn(loop_fn, 3)

  def test_softmax_cross_entropy_with_logits(self):
    # [FUNCTION DOC]
    # Runs the harness on softmax cross entropy with random logits and normalized random labels
    # (shape 3 by 2 by 4); body returns the loss and the gradient of its sum with respect to logits
    # (persistent tape). 3 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    with backprop.GradientTape(persistent=True) as g:
      logits = random_ops.random_uniform([3, 2, 4])
      g.watch(logits)
      labels = random_ops.random_uniform([3, 2, 4])
      labels /= math_ops.reduce_sum(labels, axis=[2], keepdims=True)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute cross
      # entropy for slice i and the gradient of its sum with respect to the logits. Returns the
      # per-iteration result or tuple of results that pfor stacks along a new leading dimension; no
      # side effects other than creating graph operations.
      with g:
        logits_i = array_ops.gather(logits, i)
        labels_i = array_ops.gather(labels, i)
        loss = nn.softmax_cross_entropy_with_logits(
            labels=labels_i, logits=logits_i)
        total_loss = math_ops.reduce_sum(loss)
      return loss, g.gradient(total_loss, logits_i)

    self._test_loop_fn(loop_fn, 3)

  def test_sparse_softmax_cross_entropy_with_logits(self):
    # [FUNCTION DOC]
    # Runs the harness on sparse softmax cross entropy with random logits (3 by 2 by 4) and random
    # int32 class labels (3 by 2). 3 iterations. The test harness (_test_loop_fn) runs the loop body
    # both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.
    logits = random_ops.random_uniform([3, 2, 4])
    labels = random_ops.random_uniform(
        shape=[3, 2], maxval=4, dtype=dtypes.int32)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute sparse cross
      # entropy for slice i. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      logits_i = array_ops.gather(logits, i)
      labels_i = array_ops.gather(labels, i)
      loss = nn.sparse_softmax_cross_entropy_with_logits(
          labels=labels_i, logits=logits_i)
      return loss

    self._test_loop_fn(loop_fn, 3)


class RandomTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of stateful random ops (uniform, normal, truncated normal, gamma,
  # poisson, categorical). Because the two implementations draw different random numbers,
  # run_and_assert_equal is overridden to compare shapes only.

  # The random values generated in the two implementations are not guaranteed to
  # match. So we only check the returned shapes.
  def run_and_assert_equal(self, targets1, targets2, rtol=1e-4, atol=1e-5):
    # [FUNCTION DOC]
    # Overrides the base comparison helper. Runs both targets and, rather than comparing values,
    # asserts the shape of each result in the first half equals the shape of the corresponding
    # result in the second half. Parameters rtol and atol are accepted for compatibility but
    # ignored. Returns nothing; raises an assertion failure on shape mismatch.
    outputs = self._run_targets(targets1, targets2)
    n = len(outputs) // 2
    for i in range(n):
      self.assertAllEqual(outputs[i].shape, outputs[i + n].shape)

  def test_random_uniform(self):
    # [FUNCTION DOC]
    # Runs the harness on a loop body that draws a random uniform vector of length 3, over 5
    # iterations; only shapes are compared. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will draw a length-3
      # uniform sample ignoring the index. Returns the per-iteration result or tuple of results that
      # pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      return random_ops.random_uniform([3])

    self._test_loop_fn(loop_fn, 5)

  def test_random_uniform_int(self):
    # [FUNCTION DOC]
    # Like the uniform test but draws int32 values with maxval 1. 5 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will draw a length-3
      # int32 uniform sample ignoring the index. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      return random_ops.random_uniform([3], maxval=1, dtype=dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  def test_random_standard_normal(self):
    # [FUNCTION DOC]
    # Harness run on a loop body drawing a length-3 standard normal sample, 5 iterations; shape-only
    # comparison. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will draw a length-3
      # normal sample ignoring the index. Returns the per-iteration result or tuple of results that
      # pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      return random_ops.random_normal([3])

    self._test_loop_fn(loop_fn, 5)

  def test_truncated_normal(self):
    # [FUNCTION DOC]
    # Harness run on a loop body drawing a length-3 truncated normal sample, 5 iterations;
    # shape-only comparison. The test harness (_test_loop_fn) runs the loop body both vectorized
    # with pfor and sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will draw a length-3
      # truncated normal sample ignoring the index. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      return random_ops.truncated_normal([3])

    self._test_loop_fn(loop_fn, 5)

  def test_random_gamma_invariant_alpha(self):
    # [FUNCTION DOC]
    # Harness run on random_gamma with loop-invariant alpha 0.5 and sample shape 3, 5 iterations;
    # shape-only comparison. The test harness (_test_loop_fn) runs the loop body both vectorized
    # with pfor and sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will draw gamma samples
      # with constant alpha. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      return random_ops.random_gamma([3], alpha=[0.5])

    self._test_loop_fn(loop_fn, 5)

  def test_random_gamma_varying_alpha(self):
    # [FUNCTION DOC]
    # Harness run on random_gamma where alpha is a loop-variant parameter slice from exp of normal
    # values (5 by 3 by 2); covers scalar and non-scalar alpha with empty and length-3 sample
    # shapes. 5 iterations; shape-only comparison. The test harness (_test_loop_fn) runs the loop
    # body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.
    alphas = math_ops.exp(random_ops.random_normal([5, 3, 2]))

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will draw four gamma
      # samples with scalar and tensor alpha and sample shapes empty and 3. Returns the
      # per-iteration result or tuple of results that pfor stacks along a new leading dimension; no
      # side effects other than creating graph operations.
      alphas_i = array_ops.gather(alphas, i)
      # Test both scalar and non-scalar params and shapes.
      return (random_ops.random_gamma(alpha=alphas_i[0, 0], shape=[]),
              random_ops.random_gamma(alpha=alphas_i, shape=[]),
              random_ops.random_gamma(alpha=alphas_i[0, 0], shape=[3]),
              random_ops.random_gamma(alpha=alphas_i, shape=[3]))

    self._test_loop_fn(loop_fn, 5)

  def test_random_poisson_v2_invariant_rate(self):
    # [FUNCTION DOC]
    # Harness run on random_poisson with constant rate 1.3 and shape 3, 5 iterations; shape-only
    # comparison. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will draw poisson samples
      # with constant rate. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      return random_ops.random_poisson(lam=[1.3], shape=[3])

    self._test_loop_fn(loop_fn, 5)

  def test_random_poisson_v2_varying_rate(self):
    # [FUNCTION DOC]
    # Harness run on random_poisson with loop-variant rates (5 by 3 by 2) for scalar and non-scalar
    # rates and empty and length-3 shapes; 5 iterations; shape-only comparison. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.
    rates = math_ops.exp(random_ops.random_normal([5, 3, 2]))

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will draw four poisson
      # samples with scalar and tensor rates and shapes empty and 3. Returns the per-iteration
      # result or tuple of results that pfor stacks along a new leading dimension; no side effects
      # other than creating graph operations.
      rates_i = array_ops.gather(rates, i)
      # Test both scalar and non-scalar params and shapes.
      return (random_ops.random_poisson(lam=rates_i[0, 0], shape=[]),
              random_ops.random_poisson(lam=rates_i, shape=[]),
              random_ops.random_poisson(lam=rates_i[0, 0], shape=[3]),
              random_ops.random_poisson(lam=rates_i, shape=[3]))

    self._test_loop_fn(loop_fn, 5)

  def test_random_multinomial_invariant_logits(self):
    # [FUNCTION DOC]
    # Harness run on categorical sampling with constant logits and 3 samples, 5 iterations;
    # shape-only comparison. The test harness (_test_loop_fn) runs the loop body both vectorized
    # with pfor and sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will draw 3 categorical
      # samples from constant logits. Returns the per-iteration result or tuple of results that pfor
      # stacks along a new leading dimension; no side effects other than creating graph operations.
      return random_ops.categorical(logits=[[1., -1.]], num_samples=3)

    self._test_loop_fn(loop_fn, 5)

  def test_random_multinomial_varying_logits(self):
    # [FUNCTION DOC]
    # Harness run on categorical sampling with loop-variant logits (random normal 5 by 3 by 2) and 3
    # samples, 5 iterations; shape-only comparison. The test harness (_test_loop_fn) runs the loop
    # body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.
    logits = random_ops.random_normal([5, 3, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will draw 3 categorical
      # samples from the i-th logits. Returns the per-iteration result or tuple of results that pfor
      # stacks along a new leading dimension; no side effects other than creating graph operations.
      logits_i = array_ops.gather(logits, i)
      return random_ops.categorical(logits_i, num_samples=3)

    self._test_loop_fn(loop_fn, 5)


class StatelessRandomTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of stateless random ops. Overrides run_and_assert_equal to compare
  # only shapes because stateless random values can differ under XLA compilation. Only multinomial
  # is covered so far (TODO notes other functions).

  # This test currently only tests that the vectorized and non-vectorized
  # outputs have same shapes. This is needed since under XLA compilation,
  # stateless random numbers can generate different random numbers.
  # TODO(agarwal): switch to checking for actual values matching once
  # b/149402339 is resolved.
  def run_and_assert_equal(self, targets1, targets2, rtol=1e-4, atol=1e-5):
    # [FUNCTION DOC]
    # Overrides the base comparison helper: runs both targets and asserts matching shapes between
    # corresponding outputs of the two halves; rtol and atol are accepted but ignored. Returns
    # nothing. Values are not compared (see pending bug b/149402339).
    outputs = self._run_targets(targets1, targets2)
    n = len(outputs) // 2
    for i in range(n):
      self.assertAllEqual(outputs[i].shape, outputs[i + n].shape)

  # TODO(agarwal): add tests for other random functions
  def test_multinomial(self):
    # [FUNCTION DOC]
    # Harness run on stateless_categorical covering every combination of loop-variant/invariant
    # logits and seeds (two seeds, random logits 2 by 3 by 4), 3 samples each, 2 iterations;
    # shape-only comparison. The test harness (_test_loop_fn) runs the loop body both vectorized
    # with pfor and sequentially with for_loop, evaluates both and asserts the results are close.
    seeds = [[1, 2], [3, 4]]
    logits = random_ops.random_uniform([2, 3, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body: gathers logits and seed 0 (invariant) and logits and seed i (variant)
      # and returns four stateless categorical sample tensors for all four combinations of logits
      # and seed variant/invariant.
      logits_0 = array_ops.gather(logits, 0)
      logits_i = array_ops.gather(logits, i)
      seeds_0 = array_ops.gather(seeds, 0)
      seeds_i = array_ops.gather(seeds, i)
      return (stateless_random_ops.stateless_categorical(
          logits=logits_i, num_samples=3, seed=seeds_i),
              stateless_random_ops.stateless_categorical(
                  logits=logits_i, num_samples=3, seed=seeds_0),
              stateless_random_ops.stateless_categorical(
                  logits=logits_0, num_samples=3, seed=seeds_i),
              stateless_random_ops.stateless_categorical(
                  logits=logits_0, num_samples=3, seed=seeds_0))

    self._test_loop_fn(loop_fn, 2)


class LoggingTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of logging ops: Print, print_v2 and Assert.

  def test_print(self):
    # [FUNCTION DOC]
    # Harness run on the legacy Print op that prints a slice with extra data tensors (including a
    # string and the shape), summarize 10, 3 iterations. The test harness (_test_loop_fn) runs the
    # loop body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts
    # the results are close.
    x = random_ops.random_uniform([3, 5])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will print slice i of the
      # input with extra data and return it. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      x1 = array_ops.gather(x, i)
      return logging_ops.Print(
          x1, [x1, "x1", array_ops.shape(x1)], summarize=10)

    self._test_loop_fn(loop_fn, 3)

  def test_print_v2(self):
    # [FUNCTION DOC]
    # Harness run on print_v2 inside a control dependency of an identity op for a constant vector, 3
    # iterations; afterwards captures stderr while evaluating pfor again and asserts the printed
    # text contains the stacked value line "[1 2 3] x1 []". Side effect: output written to standard
    # error is captured.
    x = constant_op.constant([1, 2, 3])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body: gathers element i of the constant, prints it with a label and its shape
      # via print_v2 and returns its identity after the print has run.
      x1 = array_ops.gather(x, i)
      with ops.control_dependencies([
          logging_ops.print_v2(
              x1, "x1", array_ops.shape(x1), summarize=10)]):
        return array_ops.identity(x1)

    self._test_loop_fn(loop_fn, 3)

    with self.captureWritesToStream(sys.stderr) as printed:
      self.evaluate(pfor_control_flow_ops.pfor(loop_fn, 3))
    self.assertIn("[1 2 3] x1 []", printed.contents())

  def test_assert(self):
    # [FUNCTION DOC]
    # Checks that Assert ops convert under pfor. Runs pfor (not for_loop, per a TODO) in a plain
    # session for the assert body, both as a one-argument function and wrapped to accept a pfor
    # config, with 3 iterations; asserts nothing beyond no error (assertion i < 10 always holds).

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body: asserts that i is below 10 with the data i, a constant list and i plus
      # one; returns the assert op.
      return control_flow_assert.Assert(i < 10, [i, [10], [i + 1]])

    # TODO(agarwal): make this work with for_loop.
    with session.Session() as sess:
      sess.run(pfor_control_flow_ops.pfor(loop_fn, 3))
      sess.run(pfor_control_flow_ops.pfor(
          lambda i, pfor_config: loop_fn(i), 3))


class TensorArrayTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of TensorArray operations under control flow v1 (the setUp
  # disables control flow v2 and tearDown restores it). Most tests are restricted to v1 graph mode.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: records whether control flow v2 is currently enabled, disables control
    # flow v2, then calls the parent setUp. Side effect: changes a process-wide toggle.
    self._enabled = control_flow_v2_toggles.control_flow_v2_enabled()
    control_flow_v2_toggles.disable_control_flow_v2()
    super(TensorArrayTest, self).setUp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: re-enables control flow v2 if it was enabled before setUp, then calls
    # the parent tearDown. Side effect: restores a process-wide toggle.
    if self._enabled:
      control_flow_v2_toggles.enable_control_flow_v2()
    super(TensorArrayTest, self).tearDown()

  @test_util.run_v1_only("b/122612051")
  def test_create_outside_and_read(self):
    # [FUNCTION DOC]
    # v1-only test. A TensorArray with two int32 elements (written 0 and 1, no clear after read) is
    # created outside the loop; body reads it at the loop-variant and fixed index. 2 iterations. The
    # test harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially
    # with for_loop, evaluates both and asserts the results are close.

    ta = tensor_array_ops.TensorArray(
        dtypes.int32, 2, clear_after_read=False).write(0, 0).write(1, 1)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will read the outer
      # TensorArray at index i and index 0. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      return ta.read(i), ta.read(0)

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
  def test_create_outside_and_gather(self):
    # [FUNCTION DOC]
    # v1-only test. Outer TensorArray with values 0 and 1; body gathers it at index list [i] and at
    # [0, 1]. 2 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with
    # pfor and sequentially with for_loop, evaluates both and asserts the results are close.

    ta = tensor_array_ops.TensorArray(
        dtypes.int32, 2, clear_after_read=False).write(0, 0).write(1, 1)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will gather the outer
      # TensorArray at [i] and [0, 1]. Returns the per-iteration result or tuple of results that
      # pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      return ta.gather([i]), ta.gather([0, 1])

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
  def test_create_outside_and_write_and_scatter(self):
    # [FUNCTION DOC]
    # v1-only test. A 10-element TensorArray is created outside; body writes at loop-variant indices
    # and scatters at loop-variant indices, returning the flow. The test builds both pfor and
    # for_loop versions, rebuilds a TensorArray from the original handle and each result flow,
    # stacks it, runs both and asserts they match. Note the sequential loop uses float32 as the
    # declared loop dtype for the flow.

    t = tensor_array_ops.TensorArray(dtypes.int32, 10, clear_after_read=False)
    handle = t.handle

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will write, then scatter
      # into the outer TensorArray and return its flow tensor. Returns the per-iteration result or
      # tuple of results that pfor stacks along a new leading dimension; no side effects other than
      # creating graph operations.
      ta = t.write(i + 2, 2 * i).write(i, 5)
      ta = ta.scatter([4 + i], [4]).scatter([6 + i, 8 + i], [6 + i, 8 + i])
      return ta.flow

    t1 = pfor_control_flow_ops.pfor(loop_fn, iters=2)
    out1 = tensor_array_ops.TensorArray(
        dtypes.int32, handle=handle, flow=t1[-1]).stack()
    output1 = self._run_targets(out1)

    t2 = pfor_control_flow_ops.for_loop(loop_fn, dtypes.float32, iters=2)
    out2 = tensor_array_ops.TensorArray(
        dtypes.int32, handle=handle, flow=t2[-1]).stack()
    output2 = self._run_targets(out2)
    self.assertAllClose(output2, output1)

  @test_util.run_v1_only("b/122612051")
  def test_create_inside_and_write(self):
    # [FUNCTION DOC]
    # v1-only test. Body creates two TensorArrays, writes loop-variant and constant values and
    # returns the stacked results. 3 iterations. The test harness (_test_loop_fn) runs the loop body
    # both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close. Source notes that changing the write order of the first array does not
    # work.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will create arrays inside
      # the loop, write values (one depends on i), stack both. Returns the per-iteration result or
      # tuple of results that pfor stacks along a new leading dimension; no side effects other than
      # creating graph operations.
      # TODO(agarwal): switching the order of writes to ta1 does not work.
      ta1 = tensor_array_ops.TensorArray(dtypes.int32, 2).write(0,
                                                                i).write(1, 1)
      ta2 = tensor_array_ops.TensorArray(dtypes.int32, 1).write(0, 1)
      return ta1.stack(), ta2.stack()

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_create_inside_and_scatter(self):
    # [FUNCTION DOC]
    # v1-only test. Body creates two TensorArrays inside the loop and fills them with scatter (one
    # loop-variant) then stacks them. 3 iterations. The test harness (_test_loop_fn) runs the loop
    # body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will create arrays inside
      # the loop, scatter values, stack both. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      # TODO(agarwal): switching the order of scatter to ta1 does not work.
      ta1 = tensor_array_ops.TensorArray(dtypes.int32,
                                         2).scatter([0],
                                                    [[i, 2]]).scatter([1],
                                                                      [[1, 2]])
      ta2 = tensor_array_ops.TensorArray(dtypes.int32,
                                         2).scatter([0], [3]).scatter([1], [4])
      return ta1.stack(), ta2.stack()

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_create_inside_and_read(self):
    # [FUNCTION DOC]
    # v1-only test. Body creates two arrays with write (clear-after-read off) and reads indices 0
    # and the loop-variant index (the other array only; reading the first array at a variant index
    # is unsupported per a TODO). 2 iterations. The test harness (_test_loop_fn) runs the loop body
    # both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will create arrays inside
      # the loop and read them at index 0 and i. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      ta1 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, i).write(1, 1)
      ta2 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, 1).write(1, 2)
      # TODO(agarwal): ta1.read(i) currently is not supported.
      return ta1.read(0), ta2.read(0), ta2.read(i)

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
  def test_create_inside_and_gather(self):
    # [FUNCTION DOC]
    # v1-only test. Body creates two arrays with write and gathers all elements and a loop-variant
    # single element. 2 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will create arrays inside
      # the loop and gather [0, 1] and [i]. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      ta1 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, i).write(1, 1)
      ta2 = tensor_array_ops.TensorArray(
          dtypes.int32, 2, clear_after_read=False).write(0, 1).write(1, 2)
      # TODO(agarwal): ta1.read(i) currently is not supported.
      return ta1.gather([0, 1]), ta2.gather([0, 1]), ta2.gather([i])

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
  def test_grad(self):
    # [FUNCTION DOC]
    # v1-only test of gradients through TensorArray. Setup: TensorArray unstacked from a random 3 by
    # 2 tensor, squared after stacking. Body computes the gradient of row i of y with respect to x
    # and returns row i of it. Runs pfor and compares with the analytic gradient 2 times x in a
    # session using close comparison.
    x = random_ops.random_uniform([3, 2])
    ta = tensor_array_ops.TensorArray(
        dtypes.float32, 3, clear_after_read=False).unstack(x)
    y = math_ops.square(ta.stack())

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will compute the gradient
      # of row i of the squared stacked array with respect to x and return row i. Returns the
      # per-iteration result or tuple of results that pfor stacks along a new leading dimension; no
      # side effects other than creating graph operations.
      y_i = array_ops.gather(y, i)
      grad = gradient_ops.gradients(y_i, x)[0]
      return array_ops.gather(grad, i)

    t1 = pfor_control_flow_ops.pfor(loop_fn, iters=3)
    # y = x * x. Hence dy/dx = 2 * x.
    actual_grad = 2.0 * x
    with session.Session() as sess:
      actual_grad, computed_grad = sess.run([t1, actual_grad])
      self.assertAllClose(actual_grad, computed_grad)


@test_util.run_all_in_graph_and_eager_modes
class TensorListTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of TensorList ops (reserve, set/get item, push/pop back, scatter,
  # gather, concat, stack, add_n, and while loops over lists). Runs in graph and eager modes.

  def test_create_outside_and_write(self):
    # [FUNCTION DOC]
    # Two reserved lists of two int32 scalars are created outside the loop; body sets items using
    # loop-variant and constant values on copies and returns both stacked. 3 iterations. The test
    # harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with
    # for_loop, evaluates both and asserts the results are close.
    handle1 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
    handle2 = list_ops.tensor_list_reserve([], 2, dtypes.int32)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will set items in copies
      # of two outer lists and stack both. Returns the per-iteration result or tuple of results that
      # pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      h1 = list_ops.tensor_list_set_item(handle1, 0, i)
      h1 = list_ops.tensor_list_set_item(h1, 1, 1)
      h2 = list_ops.tensor_list_set_item(handle2, 0, 1)
      return (list_ops.tensor_list_stack(h1, dtypes.int32),
              list_ops.tensor_list_stack(h2, dtypes.int32))

    self._test_loop_fn(loop_fn, 3)

  def _make_graph_def(self, text):
    # [FUNCTION DOC]
    # HELPER: builds a GraphDef for a test in this class.
    # Parses a text-format GraphDef description. Input: text (string of protocol buffer text).
    # Output: a new GraphDef message filled from it. Raises a parse error if the text is invalid.
    ret = graph_pb2.GraphDef()
    text_format.Parse(text, ret)
    return ret

  def test_no_fallback_with_internal_stacking(self):
    # [FUNCTION DOC]
    # Checks the error when pfor cannot vectorize a function-call op that internally needs a stacked
    # list handle. Setup: a tf.function fetching an item from a list handle; the loop body imports a
    # graph def invoking that function by name with a loop-variant list handle. Asserts a ValueError
    # ("No pfor vectorization") is raised both with fallback_to_while_loop disabled and enabled.
    # Assumes pfor does not look up function definitions for ops named like functions.

    # Create an op (really a function) that pfor definitely does not have a
    # converter for. Assumes pfor does not start looking up function definitions
    # for op-type-is-function-name calls.
    @def_function.function
    def opaque_list_fetch(x):
      # [FUNCTION DOC]
      # Compiled function used as an op pfor has no converter for: calls identity on its list-handle
      # argument (useless op) and returns item 0 of the list as int32.
      array_ops.identity(x)
      return list_ops.tensor_list_get_item(x, 0, dtypes.int32)

    external_handle = list_ops.tensor_list_reserve([], 2, dtypes.int32)
    opaque_list_fetch_concrete = opaque_list_fetch.get_concrete_function(
        external_handle)
    opaque_list_fetch_name = opaque_list_fetch_concrete.name

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body: creates a list with loop-variant item 0, registers the concrete function
      # in the graph, builds a graph definition with a placeholder and a node calling the function,
      # imports it with the list handle mapped to the placeholder, and returns the output of the
      # imported call node.
      h1 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      h1 = list_ops.tensor_list_set_item(h1, 0, i)
      opaque_list_fetch_concrete.add_to_graph()
      graph_def = self._make_graph_def("""
         node { name: 'x' op: 'Placeholder'
                attr { key: 'dtype' value { type: DT_FLOAT } }}
         node { name: 'fn' op: '""" + opaque_list_fetch_name.decode()
                                       + """' input: 'x:0' }""")
      return importer.import_graph_def(
          graph_def,
          input_map={"x:0": h1},
          return_elements=["fn"],
          name="import")[0].outputs[0]

    with self.assertRaisesRegex(ValueError, "No pfor vectorization"):
      self._test_loop_fn(loop_fn, 3, fallback_to_while_loop=False)
    with self.assertRaisesRegex(ValueError, "No pfor vectorization"):
      self._test_loop_fn(loop_fn, 3, fallback_to_while_loop=True)

  def test_create_inside_and_write(self):
    # [FUNCTION DOC]
    # Like create_outside_and_write but the lists are reserved inside the loop body. 3 iterations.
    # The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially
    # with for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will reserve two lists in
      # the loop, set items and stack both. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      h1 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      h1 = list_ops.tensor_list_set_item(h1, 0, i)
      h1 = list_ops.tensor_list_set_item(h1, 1, 1)
      h2 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      h2 = list_ops.tensor_list_set_item(h2, 0, 1)
      return (list_ops.tensor_list_stack(h1, dtypes.int32),
              list_ops.tensor_list_stack(h2, dtypes.int32))

    self._test_loop_fn(loop_fn, 3)

  def test_create_outside_and_read(self):
    # [FUNCTION DOC]
    # An outer list holds 0 and 1; the body gets item i, item 0, the length and the element shape as
    # int32 and int64. 2 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    handle = list_ops.tensor_list_reserve([], 2, dtypes.int32)
    handle = list_ops.tensor_list_set_item(handle, 0, 0)
    handle = list_ops.tensor_list_set_item(handle, 1, 1)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will read the outer list
      # by variant and fixed index and query length and element shape. Returns the per-iteration
      # result or tuple of results that pfor stacks along a new leading dimension; no side effects
      # other than creating graph operations.
      return (list_ops.tensor_list_get_item(handle, i, dtypes.int32),
              list_ops.tensor_list_get_item(handle, 0, dtypes.int32),
              list_ops.tensor_list_length(handle),
              list_ops.tensor_list_element_shape(handle, dtypes.int32),
              list_ops.tensor_list_element_shape(handle, dtypes.int64))

    self._test_loop_fn(loop_fn, 2)

  def test_create_outside_and_read_zero_loop_iters(self):
    # [FUNCTION DOC]
    # Same reads as the previous test but with zero loop iterations, to check empty-loop handling of
    # list read ops. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor
    # and sequentially with for_loop, evaluates both and asserts the results are close.
    handle = list_ops.tensor_list_reserve([], 2, dtypes.int32)
    handle = list_ops.tensor_list_set_item(handle, 0, 0)
    handle = list_ops.tensor_list_set_item(handle, 1, 1)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will read the outer list
      # and query length and element shape. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      return (
          list_ops.tensor_list_get_item(handle, i, dtypes.int32),
          list_ops.tensor_list_get_item(handle, 0, dtypes.int32),
          list_ops.tensor_list_length(handle),
          list_ops.tensor_list_element_shape(handle, dtypes.int32),
          list_ops.tensor_list_element_shape(handle, dtypes.int64),
      )

    self._test_loop_fn(loop_fn, 0)

  def test_create_inside_and_read(self):
    # [FUNCTION DOC]
    # The list is reserved and written inside the body (item 0 loop-variant) and then read at 0 and
    # i, plus length and element shapes. 2 iterations. The test harness (_test_loop_fn) runs the
    # loop body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts
    # the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will reserve, set, then
      # read the list and query length and shapes. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      handle = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      handle = list_ops.tensor_list_set_item(handle, 0, i)
      handle = list_ops.tensor_list_set_item(handle, 1, 1)
      return (list_ops.tensor_list_get_item(handle, 0, dtypes.int32),
              list_ops.tensor_list_get_item(handle, i, dtypes.int32),
              list_ops.tensor_list_length(handle),
              list_ops.tensor_list_element_shape(handle, dtypes.int32),
              list_ops.tensor_list_element_shape(handle, dtypes.int64))

    self._test_loop_fn(loop_fn, 2)

  def test_create_outside_and_push_back(self):
    # [FUNCTION DOC]
    # An outer reserved list is pushed to three times with loop-variant and constant vectors;
    # stacked result returned. 3 iterations. The test harness (_test_loop_fn) runs the loop body
    # both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.
    h = list_ops.tensor_list_reserve([2], 2, dtypes.int32)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will push three vectors
      # onto the outer list and stack it. Returns the per-iteration result or tuple of results that
      # pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      handle = list_ops.tensor_list_push_back(h, [i, 2])
      handle = list_ops.tensor_list_push_back(handle, [1, 2])
      handle = list_ops.tensor_list_push_back(handle, [1, 2])
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_create_inside_and_push_back(self):
    # [FUNCTION DOC]
    # The list is reserved inside the body, then two vectors (one loop-variant) are pushed and the
    # list is stacked. 3 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will reserve, push two
      # vectors and stack. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
      handle = list_ops.tensor_list_push_back(handle, [i, 2])
      handle = list_ops.tensor_list_push_back(handle, [1, 2])
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_pop_back_no_shape(self):
    # [FUNCTION DOC]
    # Pushes two vectors (second loop-variant) then pops back, returning the popped tensor and the
    # stacked remaining list. 3 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will reserve, push twice,
      # pop back and return popped value and stacked rest. Returns the per-iteration result or tuple
      # of results that pfor stacks along a new leading dimension; no side effects other than
      # creating graph operations.
      handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
      handle = list_ops.tensor_list_push_back(handle, [1, 2])
      handle = list_ops.tensor_list_push_back(handle, [i, 2])
      handle, tensor = list_ops.tensor_list_pop_back(handle, dtypes.int32)
      return tensor, list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_pop_back_no_shape_capture(self):
    # [FUNCTION DOC]
    # An outer list holds one vector; the body pops from it, pushes a loop-variant vector, and
    # returns the popped tensor and the stacked list. 3 iterations. The test harness (_test_loop_fn)
    # runs the loop body both vectorized with pfor and sequentially with for_loop, evaluates both
    # and asserts the results are close.
    h = list_ops.tensor_list_reserve([2], 1, dtypes.int32)
    h = list_ops.tensor_list_push_back(h, [1, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will pop from captured
      # list, push a loop-variant vector and stack. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      handle, tensor = list_ops.tensor_list_pop_back(h, dtypes.int32)
      handle = list_ops.tensor_list_push_back(handle, [1, i])
      return tensor, list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_pop_back_with_shape(self):
    # [FUNCTION DOC]
    # Checks gradient of push-back/stack produces a pop-back op under pfor. Body is a compiled
    # function taping a list reserve, push-back of the cast iteration index and stack, then taking
    # the gradient (which must be a TensorListPopBack op, asserted inside) and returning the
    # gradient, the stacked list and the pop op's second input. 3 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.

    @def_function.function
    def loop_fn(i):
      # [FUNCTION DOC]
      # Compiled per-iteration body: with a gradient tape, reserves a float list of unknown element
      # shape, pushes the cast index, stacks it; gets the gradient of the stacked value; asserts the
      # gradient op type is TensorListPopBack (using the enclosing test's assertion); returns
      # gradient, stacked value and the pop op's element shape input.
      with backprop.GradientTape() as tape:
        handle = list_ops.tensor_list_reserve(None, 1, dtypes.float32)
        x = math_ops.cast(i, dtypes.float32)[None]
        tape.watch(x)
        handle = list_ops.tensor_list_push_back(handle, x)
        stacked = list_ops.tensor_list_stack(handle, dtypes.float32)
      list_grad = tape.gradient(stacked, x, x)
      self.assertEqual("TensorListPopBack", list_grad.op.type)
      return list_grad, stacked, list_grad.op.inputs[1]

    self._test_loop_fn(loop_fn, 3)

  def test_create_outside_and_scatter(self):
    # [FUNCTION DOC]
    # Scatter into an outer reserved list at fixed indices three times with loop-variant data;
    # stacks result. 3 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    h = list_ops.tensor_list_reserve([2], 2, dtypes.int32)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will scatter into the
      # outer list and stack it. Returns the per-iteration result or tuple of results that pfor
      # stacks along a new leading dimension; no side effects other than creating graph operations.
      handle = list_ops.tensor_list_scatter([[i, 2]], [0], input_handle=h)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_create_inside_and_scatter(self):
    # [FUNCTION DOC]
    # Like the previous test but the list is reserved inside the body. 3 iterations. The test
    # harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with
    # for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will reserve, scatter
      # twice and stack. Returns the per-iteration result or tuple of results that pfor stacks along
      # a new leading dimension; no side effects other than creating graph operations.
      handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
      handle = list_ops.tensor_list_scatter([[i, 2]], [0], input_handle=handle)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 3)

  def test_loop_variant_scatter_indices(self):
    # [FUNCTION DOC]
    # Scatter with loop-dependent indices (i, i + 5) and values into a list reserved with 10 slots;
    # stack result. 5 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will scatter loop-variant
      # rows at loop-variant indices and stack. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      handle = list_ops.tensor_list_reserve([2], 10, dtypes.int32)
      handle = list_ops.tensor_list_scatter(
          [[1, i], [i + 1, 2]],
          [i, i + 5], input_handle=handle)
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  def test_loop_variant_scatter_duplicate_indices(self):
    # [FUNCTION DOC]
    # Scatter with duplicated loop-variant indices; skipped on GPU because of nondeterministic
    # scatter updates. 5 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will scatter rows
      # including a duplicate index and stack. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      handle = list_ops.tensor_list_reserve([2], 10, dtypes.int32)
      handle = list_ops.tensor_list_scatter(
          [[1, i], [1, i + 1], [i + 2, 3]],
          [i, i, i + 2], input_handle=handle)
      return list_ops.tensor_list_stack(handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  def test_create_outside_and_gather(self):
    # [FUNCTION DOC]
    # Outer list scattered with two rows; body gathers all indices and a loop-variant index. 2
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.
    handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
    handle = list_ops.tensor_list_scatter([[2, 3]], [0], input_handle=handle)
    handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will gather [0, 1] and
      # [i] from the outer list. Returns the per-iteration result or tuple of results that pfor
      # stacks along a new leading dimension; no side effects other than creating graph operations.
      return (list_ops.tensor_list_gather(handle, [0, 1], dtypes.int32),
              list_ops.tensor_list_gather(handle, [i], dtypes.int32))

    self._test_loop_fn(loop_fn, 2)

  def test_create_inside_and_gather(self):
    # [FUNCTION DOC]
    # List created and scattered inside the body (loop-variant row), then gathered at [0, 1] and
    # [i]. 2 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with
    # pfor and sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will reserve, scatter,
      # gather [0, 1] and [i]. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
      handle = list_ops.tensor_list_scatter([[i, 2]], [0], input_handle=handle)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      return (list_ops.tensor_list_gather(handle, [0, 1], dtypes.int32),
              list_ops.tensor_list_gather(handle, [i], dtypes.int32))

    self._test_loop_fn(loop_fn, 2)

  def test_create_inside_and_concat(self):
    # [FUNCTION DOC]
    # Scatters rows into a list created in the body and concatenates (concat v2) it. Runs pfor
    # directly and asserts the first output equals [[0, 2, 1, 2], [1, 2, 1, 2]] and the second
    # (lengths) equals [[2, 2], [2, 2]].

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will reserve, scatter two
      # rows and concat. Returns the per-iteration result or tuple of results that pfor stacks along
      # a new leading dimension; no side effects other than creating graph operations.
      handle = list_ops.tensor_list_reserve([2], 2, dtypes.int32)
      handle = list_ops.tensor_list_scatter([[i, 2]], [0], input_handle=handle)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      return gen_list_ops.tensor_list_concat_v2(
          handle,
          element_dtype=dtypes.int32,
          element_shape=[2],
          leading_dims=[])

    output = pfor_control_flow_ops.pfor(loop_fn, 2)
    self.assertAllClose([[0, 2, 1, 2], [1, 2, 1, 2]], output[0])
    self.assertAllClose([[2, 2], [2, 2]], output[1])

  def test_create_outside_and_concat(self):
    # [FUNCTION DOC]
    # Same as the previous test but scatters into a list created outside the loop. Same expected
    # values.
    h = list_ops.tensor_list_reserve([2], 2, dtypes.int32)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will scatter into the
      # outer list and concat. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      handle = list_ops.tensor_list_scatter([[i, 2]], [0], input_handle=h)
      handle = list_ops.tensor_list_scatter([[1, 2]], [1], input_handle=handle)
      return gen_list_ops.tensor_list_concat_v2(
          handle,
          element_dtype=dtypes.int32,
          element_shape=[2],
          leading_dims=[])

    output = pfor_control_flow_ops.pfor(loop_fn, 2)
    self.assertAllClose([[0, 2, 1, 2], [1, 2, 1, 2]], output[0])
    self.assertAllClose([[2, 2], [2, 2]], output[1])

  def test_tensor_list_from_tensor(self):
    # [FUNCTION DOC]
    # Body builds a list from slice i of a random 2 by 3 by 4 tensor (element shape 4) and stacks
    # it. 2 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with
    # pfor and sequentially with for_loop, evaluates both and asserts the results are close.
    t = random_ops.random_uniform([2, 3, 4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will create a list from
      # slice i and stack it. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      handle = list_ops.tensor_list_from_tensor(array_ops.gather(t, i), [4])
      return list_ops.tensor_list_stack(handle, t.dtype)

    self._test_loop_fn(loop_fn, 2)

  @test_util.enable_control_flow_v2
  def test_tensor_list_reserve_while_loop(self):
    # [FUNCTION DOC]
    # Control-flow-v2 test. A loop-invariant list captured by a while loop whose body inserts
    # loop-variant values, forcing stacking of the handle; handled by vectorizing the reserve op. 2
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.
    # Here a loop invariant TensorList is captured by a while_loop, which then
    # performs loop dependent operations on it, resulting in a loop variant
    # output. This forces stacking of the variant handle captured by the
    # while_loop.
    # We handle this particular case by forcing vectorization of
    # TensorListReserve operation.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will set items of a
      # reserved list inside a while loop and stack. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      handle = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      _, out_handle = while_loop.while_loop(
          lambda j, _: j < 2, lambda j, h:
          (j + 1, list_ops.tensor_list_set_item(h, j, i)), (0, handle))
      return list_ops.tensor_list_stack(out_handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 2)

  @test_util.enable_control_flow_v2
  def test_tensor_list_while_loop_stacked_cond_stacked_list(self):
    # [FUNCTION DOC]
    # Control-flow-v2 test. A while loop with a loop-variant condition (j below i) over a
    # loop-variant list. 5 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run a while loop
      # with variant condition updating a list built from a tensor. Returns the per-iteration result
      # or tuple of results that pfor stacks along a new leading dimension; no side effects other
      # than creating graph operations.
      handle = list_ops.tensor_list_from_tensor([20, 21, 22, 23, i], [])
      _, out_handle = while_loop.while_loop(
          lambda j, _: j < i,
          lambda j, h: (j + 1, list_ops.tensor_list_set_item(h, j, i)),
          (0, handle))
      return list_ops.tensor_list_stack(out_handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  @test_util.enable_control_flow_v2
  def test_tensor_list_while_loop_stacked_cond_stacked_list_unknown_shape(self):
    # [FUNCTION DOC]
    # Control-flow-v2 test. Fills a list of unknown element shape in a constant-count loop, then a
    # loop with variant condition updates it. 5 iterations. The test harness (_test_loop_fn) runs
    # the loop body both vectorized with pfor and sequentially with for_loop, evaluates both and
    # asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will fill a list, then
      # update it in a loop with variant condition. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      handle = list_ops.tensor_list_reserve(None, 5, dtypes.int32)
      _, handle = while_loop.while_loop(
          lambda j, _: j < 5,
          lambda j, h: (j + 1, list_ops.tensor_list_set_item(h, j, 0)),
          (0, handle))
      _, out_handle = while_loop.while_loop(
          lambda j, _: j < i,
          lambda j, h: (j + 1, list_ops.tensor_list_set_item(h, j, i)),
          (0, handle))
      return list_ops.tensor_list_stack(out_handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  @test_util.enable_control_flow_v2
  def test_tensor_list_while_loop_stacked_cond_unstacked_list(self):
    # [FUNCTION DOC]
    # Control-flow-v2 test. While loop with variant condition over a loop-invariant list. 5
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run a
      # variant-condition while loop on an invariant list. Returns the per-iteration result or tuple
      # of results that pfor stacks along a new leading dimension; no side effects other than
      # creating graph operations.
      handle = list_ops.tensor_list_from_tensor([20, 21, 22, 23, 24], [])
      _, out_handle = while_loop.while_loop(
          lambda j, _: j < i, lambda j, h:
          (j + 1, list_ops.tensor_list_set_item(h, j, i)), (0, handle))
      return list_ops.tensor_list_stack(out_handle, dtypes.int32)

    self._test_loop_fn(loop_fn, 5)

  def test_tensor_list_addn_already_stacked(self):
    # [FUNCTION DOC]
    # Adds two lists with add_n where both are already loop-variant, then stacks. 2 iterations. The
    # test harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially
    # with for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will build two variant
      # lists, add_n and stack. Returns the per-iteration result or tuple of results that pfor
      # stacks along a new leading dimension; no side effects other than creating graph operations.
      l1 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      l1 = list_ops.tensor_list_set_item(l1, 0, i)
      l2 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      l2 = list_ops.tensor_list_set_item(l2, 1, i)
      return list_ops.tensor_list_stack(math_ops.add_n([l1, l2]), dtypes.int32)

    self._test_loop_fn(loop_fn, 2)

  def test_tensor_list_addn_stacking_required(self):
    # [FUNCTION DOC]
    # Adds an outer invariant list with a loop-variant list via add_n, which forces stacking of the
    # invariant one. 2 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    l1 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
    l1 = list_ops.tensor_list_set_item(l1, 1, 1)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will add an invariant
      # list and a variant list and stack. Returns the per-iteration result or tuple of results that
      # pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      l2 = list_ops.tensor_list_reserve([], 2, dtypes.int32)
      l2 = list_ops.tensor_list_set_item(l2, 1, i)
      return list_ops.tensor_list_stack(
          math_ops.add_n([l1, l2]), dtypes.int32)

    self._test_loop_fn(loop_fn, 2)


@test_util.run_all_in_graph_and_eager_modes
class TensorTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of tensor scatter update with loop-variant indices and updates.
  # Runs in graph and eager modes; skipped on GPU due to scatter nondeterminism.

  def test_loop_variant_scatter_update_no_shape(self):
    # [FUNCTION DOC]
    # Scatter-update test through a function with fully unknown input shapes (input signature of
    # shape-less int32 tensors); skipped on GPU. 5 iterations. The test harness (_test_loop_fn) runs
    # the loop body both vectorized with pfor and sequentially with for_loop, evaluates both and
    # asserts the results are close.
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32),
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32),
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)
    ])
    def shapeless_func(tensor, indices, updates):
      # [FUNCTION DOC]
      # Compiled function with shape-less int32 input signature: applies tensor_scatter_nd_update of
      # updates into tensor at indices. Returns the updated tensor.
      return array_ops.tensor_scatter_nd_update(tensor, indices, updates)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will call the shapeless
      # function with loop-variant indices and updates. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      tensor = [0, 0, 0, 0, 0, 0, 0, 0]
      indices = [[i], [i + 1], [i + 3], [i + 2]]
      updates = [i, i - 10, i + 11, 12]
      return shapeless_func(tensor, indices, updates)

    self._test_loop_fn(loop_fn, 5)

  def test_loop_variant_scatter_update_singles(self):
    # [FUNCTION DOC]
    # Scatter-updates single elements at loop-dependent indices of an 8-element zero vector; skipped
    # on GPU. 5 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with
    # pfor and sequentially with for_loop, evaluates both and asserts the results are close.
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will scatter-update four
      # loop-variant single elements. Returns the per-iteration result or tuple of results that pfor
      # stacks along a new leading dimension; no side effects other than creating graph operations.
      tensor = [0, 0, 0, 0, 0, 0, 0, 0]
      indices = [[i], [i+1], [i+3], [i+2]]
      updates = [i, i-10, i+11, 12]
      return array_ops.tensor_scatter_nd_update(tensor, indices, updates)

    self._test_loop_fn(loop_fn, 5)

  def test_loop_variant_scatter_update_slices(self):
    # [FUNCTION DOC]
    # Scatter-updates two rows of a 10 by 3 zeros matrix with loop-dependent indices and values;
    # skipped on GPU. 5 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will scatter-update rows
      # of a matrix. Returns the per-iteration result or tuple of results that pfor stacks along a
      # new leading dimension; no side effects other than creating graph operations.
      tensor = array_ops.zeros([10, 3], dtype=dtypes.int32)
      indices = [[i+2], [4]]
      updates = [[1, i*2, 3], [i+4, i-5, 6]]
      return array_ops.tensor_scatter_nd_update(tensor, indices, updates)

    self._test_loop_fn(loop_fn, 5)

  def test_loop_variant_scatter_update_multi_dim_index(self):
    # [FUNCTION DOC]
    # Scatter-updates two elements with two-dimensional indices into a 10 by 3 matrix; skipped on
    # GPU. 5 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with
    # pfor and sequentially with for_loop, evaluates both and asserts the results are close.
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will scatter-update with
      # 2-D indices. Returns the per-iteration result or tuple of results that pfor stacks along a
      # new leading dimension; no side effects other than creating graph operations.
      tensor = array_ops.zeros([10, 3], dtype=dtypes.int32)
      indices = [[i+2, 1], [4, 2]]
      updates = [i, 5]
      return array_ops.tensor_scatter_nd_update(tensor, indices, updates)

    self._test_loop_fn(loop_fn, 5)

  def test_loop_variant_scatter_update_folded_indices(self):
    # [FUNCTION DOC]
    # Scatter-updates a 5 by 5 zeros matrix using a batch of two sets of diagonal indices ("folded"
    # index batches) with loop-dependent updates; skipped on GPU. 5 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.
    if test_util.is_gpu_available():
      self.skipTest(
          "Flaky in some GPU configurations due to TensorScatterNdUpdate "
          "nondeterminism.")

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will scatter-update with
      # batched index sets. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      tensor = array_ops.zeros([5, 5])
      indices = [
          [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
          [[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]],
      ]
      updates = [
          [1, i, 1, 1, 1],
          [1, 1, i+2, 1, i-5],
      ]
      return array_ops.tensor_scatter_nd_update(tensor, indices, updates)

    self._test_loop_fn(loop_fn, 5)


class OptionalTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of Optional ops.

  def test_optional_from_value(self):
    # [FUNCTION DOC]
    # Body wraps three values (two loop-variant, one constant) in an optional, creates and discards
    # an optional-none, then extracts the values with their dtypes and shapes. 2 iterations. The
    # test harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially
    # with for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will wrap values in an
      # optional and read them back. Returns the per-iteration result or tuple of results that pfor
      # stacks along a new leading dimension; no side effects other than creating graph operations.
      o = gen_optional_ops.optional_from_value(
          [i, i + 1, constant_op.constant(3)]
      )
      gen_optional_ops.optional_none()
      return gen_optional_ops.optional_get_value(
          o, [dtypes.int32, dtypes.int32, dtypes.int32], [[], [], []]
      )

    self._test_loop_fn(loop_fn, 2)


class StackTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of the stack_v2 ops (push, pop). Mostly restricted to v1 graph
  # mode.

  @test_util.run_v1_only("b/122612051")
  def test_stack_inside_loop_invariant(self):
    # [FUNCTION DOC]
    # v1-only. Stack created, pushed twice and popped twice inside the body with loop-invariant
    # values, returning the popped values. 2 iterations. The test harness (_test_loop_fn) runs the
    # loop body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts
    # the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will create a stack, push
      # 1 and 2, pop twice, return both pops. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      s = data_flow_ops.stack_v2(max_size=4, elem_type=dtypes.int32)
      op1 = data_flow_ops.stack_push_v2(s, 1)
      with ops.control_dependencies([op1]):
        op2 = data_flow_ops.stack_push_v2(s, 2)
      with ops.control_dependencies([op2]):
        e2 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      with ops.control_dependencies([e2]):
        e1 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      return e1, e2

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
  def test_stack_inside_push_loop_dependent(self):
    # [FUNCTION DOC]
    # v1-only. Same as the invariant test but the first pushed value is the iteration index. 2
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will create a stack, push
      # i and 2, pop twice, return both pops. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      s = data_flow_ops.stack_v2(max_size=4, elem_type=dtypes.int32)
      op1 = data_flow_ops.stack_push_v2(s, i)
      with ops.control_dependencies([op1]):
        op2 = data_flow_ops.stack_push_v2(s, 2)
      with ops.control_dependencies([op2]):
        e2 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      with ops.control_dependencies([e2]):
        e1 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      return e1, e2

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_v1_only("b/122612051")
  def test_stack_outside_pop(self):
    # [FUNCTION DOC]
    # v1-only. A stack created and pushed (5, 6, 7) outside the loop; the body pops twice per
    # iteration. After running pfor over 2 iterations a further pop is done. Asserts the pops return
    # [7, 7], [6, 6] and finally 5, showing pops are applied as a group. Runs targets without
    # re-initialisation.
    s = data_flow_ops.stack_v2(max_size=4, elem_type=dtypes.int32)
    op = data_flow_ops.stack_push_v2(s, 5)
    with ops.control_dependencies([op]):
      op = data_flow_ops.stack_push_v2(s, 6)
    with ops.control_dependencies([op]):
      op = data_flow_ops.stack_push_v2(s, 7)

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will pop twice from the
      # outer stack and return both. Returns the per-iteration result or tuple of results that pfor
      # stacks along a new leading dimension; no side effects other than creating graph operations.
      e1 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      with ops.control_dependencies([e1]):
        e2 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
      return e1, e2

    with ops.control_dependencies([op]):
      e1, e2 = pfor_control_flow_ops.pfor(loop_fn, iters=2)
    with ops.control_dependencies([e1, e2]):
      e3 = data_flow_ops.stack_pop_v2(s, elem_type=dtypes.int32)
    v1, v2, v3 = self._run_targets([e1, e2, e3], run_init=False)
    self.assertAllEqual([7, 7], v1)
    self.assertAllEqual([6, 6], v2)
    self.assertAllEqual(5, v3)

  @test_util.run_v1_only("b/122612051")
  def test_stack_outside_push(self):
    # [FUNCTION DOC]
    # v1-only. Pushing onto a stack defined outside the loop is not allowed: asserts pfor raises a
    # ValueError matching "StackPushV2 not allowed".
    s = data_flow_ops.stack_v2(max_size=4, elem_type=dtypes.int32)

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will push the constant 7
      # onto the outer stack. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      return data_flow_ops.stack_push_v2(s, 7)

    with self.assertRaisesRegex(ValueError, "StackPushV2 not allowed.*"):
      pfor_control_flow_ops.pfor(loop_fn, iters=2)


# TODO(agarwal): test nested while_loops. This currently requires converting a
# tf.cond.
class WhileV1Test(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of while loops under control flow v1 (setUp disables control flow
  # v2; tearDown restores). Has a TODO about nested while loops. Mostly v1 graph mode.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: remembers whether control flow v2 is enabled, disables it, calls the parent
    # setUp. Side effect: process-wide toggle changed.
    self._enabled = control_flow_v2_toggles.control_flow_v2_enabled()
    control_flow_v2_toggles.disable_control_flow_v2()
    super(WhileV1Test, self).setUp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture teardown: re-enables control flow v2 if it had been enabled, then calls the parent
    # tearDown.
    if self._enabled:
      control_flow_v2_toggles.enable_control_flow_v2()
    super(WhileV1Test, self).tearDown()

  def test_while_outside_loop(self):
    # [FUNCTION DOC]
    # A while loop (count to 4) is built outside; the loop body just adds the iteration index to its
    # result. 3 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with
    # pfor and sequentially with for_loop, evaluates both and asserts the results are close.

    x = while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will add i to the outer
      # while-loop result. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      return x + i

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_invariant_while(self):
    # [FUNCTION DOC]
    # v1-only. Body is an entirely loop-invariant while loop counting to 4. 3 iterations. The test
    # harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with
    # for_loop, evaluates both and asserts the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run an invariant
      # while loop. Returns the per-iteration result or tuple of results that pfor stacks along a
      # new leading dimension; no side effects other than creating graph operations.
      return while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_invariant_while_with_control_dependency(self):
    # [FUNCTION DOC]
    # v1-only. Invariant while loop built under a control dependency on the iteration index. 3
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run an invariant
      # while loop under a control dependency on i. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      with ops.control_dependencies([i]):
        return while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_while_with_stateful_ops(self):
    # [FUNCTION DOC]
    # v1-only. While loop whose body draws random numbers (stateful); returns the counter output. 3
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run a while loop
      # with a stateful body. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      return while_loop.while_loop(
          lambda j, x: j < 4, lambda j, x:
          (j + 1, x + random_ops.random_uniform([])), [0, 0.])[0]

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_while_unstacked_condition(self):
    # [FUNCTION DOC]
    # v1-only. While loop with a loop-invariant condition but a body adding the loop-variant i. 3
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run a while loop
      # accumulating i with invariant condition. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      return while_loop.while_loop(lambda j, x: j < 4, lambda j, x:
                                   (j + 1, x + i), [0, 0])

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_while(self):
    # [FUNCTION DOC]
    # v1-only. While loop whose trip count is loop-variant (lengths 4, 0, 2) summing prefix elements
    # of row i. 3 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized
    # with pfor and sequentially with for_loop, evaluates both and asserts the results are close.
    x = random_ops.random_uniform([3, 5])
    lengths = constant_op.constant([4, 0, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will sum the first
      # lengths_i elements of row i with a while loop. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      x_i = array_ops.gather(x, i)
      lengths_i = array_ops.gather(lengths, i)

      _, total = while_loop.while_loop(
          lambda j, _: j < lengths_i, lambda j, t:
          (j + 1, t + array_ops.gather(x_i, j)), [0, 0.])
      return total

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_while_jacobian(self):
    # [FUNCTION DOC]
    # v1-only. Computes a jacobian through a while loop: out = x times y four times. Body takes the
    # gradient of one output column with respect to x. Runs pfor and compares to a manually built
    # expected output (transposed power of y) in a session. Note the same computation does not work
    # with a plain loop.
    x = random_ops.random_uniform([1, 3])
    y = random_ops.random_uniform([3, 3])

    # out = x @ y @ y @ y @ y, where @ is matmul operator.
    _, out = while_loop.while_loop(
        lambda i, _: i < 4, lambda i, out: (i + 1, math_ops.matmul(out, y)),
        [0, x])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will take the gradient of
      # column i of the while-loop output with respect to x and flatten it. Returns the
      # per-iteration result or tuple of results that pfor stacks along a new leading dimension; no
      # side effects other than creating graph operations.
      out_i = array_ops.gather(out, i, axis=1)
      return array_ops.reshape(gradient_ops.gradients(out_i, x)[0], [-1])

    out = pfor_control_flow_ops.pfor(loop_fn, iters=3)

    # The above code does not work with tf.while_loop instead of pfor. So we
    # manually compute the expected output here.
    # Note that gradient of output w.r.t is (y @ y @ y @ y)^T.
    expected_output = y
    for _ in range(3):
      expected_output = math_ops.matmul(expected_output, y)
    expected_output = array_ops.transpose(expected_output, [1, 0])

    with session.Session() as sess:
      out, expected = sess.run([out, expected_output])
      self.assertAllClose(expected, out)

  @test_util.run_v1_only("b/122612051")
  def test_tensor_array_as_loop_variable(self):
    # [FUNCTION DOC]
    # v1-only. While loop carrying a TensorArray written with i plus j squared at index j for 4
    # steps, then stacked. 3 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body: runs a four-step while loop carrying a TensorArray, writing i plus j
      # squared at index j, and returns the stacked array.

      def body(j, ta):
        # [FUNCTION DOC]
        # While-loop body: writes i plus j squared into the array at index j and returns the next
        # counter and the updated array.
        ta = ta.write(j, i + j * j)
        return j + 1, ta

      _, ta = while_loop.while_loop(
          lambda j, _: j < 4, body,
          (0, tensor_array_ops.TensorArray(dtypes.int32, size=4)))
      return ta.stack()

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_read_tensor_array_partitioned_indices(self):
    # [FUNCTION DOC]
    # v1-only. TensorArray values depend on i and the while termination also depends on i; body sums
    # the first i elements. 3 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    # Note that tensor array values are pfor loop dependent, and the while loop
    # termination condition is also dependent on pfor iteration.
    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body: unstacks i plus a 5-element range into a TensorArray and runs a while
      # loop from 0 to i summing the read elements; returns the sum.
      ta = tensor_array_ops.TensorArray(dtypes.int32, size=6)
      ta = ta.unstack(i + list(range(5)))

      def body(j, s):
        # [FUNCTION DOC]
        # While-loop body: returns the next counter and the running sum plus array element j.
        return j + 1, s + ta.read(j)

      _, s = while_loop.while_loop(lambda j, _: j < i, body, (0, 0))
      return s

    self._test_loop_fn(loop_fn, 3)

  @test_util.run_v1_only("b/122612051")
  def test_external_while_loop_grad(self):
    # [FUNCTION DOC]
    # v1-only. An external while loop writes constant x to a TensorArray four times and stacks it.
    # Body takes the gradient of element i with respect to x; this extends the external loop from
    # inside pfor. Asserts in a session that pfor yields [1, 1, 1] and that the external loop is not
    # converted (otherwise all iterations would write the same indices).
    # Here we test that external while_loops that are extended from inside pfor
    # (due to gradient calls) are not actually converted. If the below was
    # converted all pfor iterations would write to the same tensor array
    # indices.
    x = constant_op.constant(1.)

    def body(j, ta):
      # [FUNCTION DOC]
      # External while-loop body: writes constant x at index j and returns next counter and array.
      ta = ta.write(j, x)
      return j + 1, ta

    _, ta = while_loop.while_loop(
        lambda j, _: j < 4, body,
        (0, tensor_array_ops.TensorArray(dtypes.float32, size=4)))
    out = ta.stack()

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will take the gradient of
      # element i of the external stacked output with respect to x. Returns the per-iteration result
      # or tuple of results that pfor stacks along a new leading dimension; no side effects other
      # than creating graph operations.
      out_i = array_ops.gather(out, i)
      return gradient_ops.gradients(out_i, x)[0]

    with session.Session() as sess:
      # out is [x, x, x]. Hence the gradients should be [1, 1, 1].
      self.assertAllEqual([1, 1, 1],
                          sess.run(pfor_control_flow_ops.pfor(loop_fn, 3)))

  @test_util.run_v1_only("b/122612051")
  def test_tensor_array_grad(self):
    # [FUNCTION DOC]
    # v1-only. TensorArray unstacked from a random 3 by 4 by 2 tensor; body runs a while loop
    # summing selected slices then takes the product and gradient with respect to the input.
    # Compares pfor outputs and the sum of per-iteration gradients with hand-built equivalent
    # computations (reduce sum then product, and its gradient) in a session.
    inp = constant_op.constant(np.random.rand(3, 4, 2), dtype=dtypes.float32)
    ta = tensor_array_ops.TensorArray(dtypes.float32, size=3)
    ta = ta.unstack(inp)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body: sums the row i of each array slice via a while loop, reduces by product
      # and returns the product and its gradient with respect to the input.

      def body(j, x):
        # [FUNCTION DOC]
        # While-loop body: gathers the array element at j, reshapes to 4 by 2, gathers row i and
        # adds it to the accumulator; returns next counter and accumulator.
        value = ta.gather([j])
        value = array_ops.gather(array_ops.reshape(value, [4, 2]), i)
        return j + 1, x + value

      _, out = while_loop.while_loop(lambda j, _: j < 3, body,
                                     (0, array_ops.zeros([2])))
      out = math_ops.reduce_prod(out)
      return out, gradient_ops.gradients(out, inp)[0]

    pfor_out, pfor_out_grad = pfor_control_flow_ops.pfor(loop_fn, 4)
    # Note that tf.while_loop does not work in the setup above. So we manually
    # construct the equivalent computation of the above loops here.
    real_out = math_ops.reduce_sum(inp, axis=[0])
    real_out = math_ops.reduce_prod(real_out, axis=[1])
    # Note that gradients of real_out will accumulate the gradients across the
    # output value. Hence we do the same aggregation on pfor_out_grad.
    real_out_grad = gradient_ops.gradients(real_out, inp)[0]
    sum_pfor_out_grad = math_ops.reduce_sum(pfor_out_grad, axis=[0])

    with session.Session() as sess:
      v1, v2, v1_grad, v2_grad = sess.run(
          [pfor_out, real_out, sum_pfor_out_grad, real_out_grad])
      self.assertAllClose(v1, v2)
      self.assertAllClose(v1_grad, v2_grad)


def dynamic_lstm_input_fn(batch_size, state_size, max_steps):
  # [FUNCTION DOC]
  # HELPER: builds the inputs for the dynamic RNN helpers and tests.
  # Creates constant random input data: a float32 tensor of shape batch_size by max_steps by
  # state_size and an int32 vector of random sequence lengths in the range 0 to max_steps inclusive.
  # Constants are used so repeated session runs give identical values. Parameters: batch_size,
  # state_size, max_steps (integers). Returns a pair (inputs, sequence_length).
  # We make inputs and sequence_length constant so that multiple session.run
  # calls produce the same result.
  inputs = constant_op.constant(
      np.random.rand(batch_size, max_steps, state_size), dtype=dtypes.float32)
  sequence_length = np.random.randint(0, size=[batch_size], high=max_steps + 1)
  sequence_length = constant_op.constant(sequence_length, dtype=dtypes.int32)
  return inputs, sequence_length


def create_dynamic_lstm(cell_fn, batch_size, state_size, max_steps):
  # [FUNCTION DOC]
  # HELPER: builds comparable pfor and dynamic_rnn graphs for RNN tests and benchmarks.
  # Builds a recurrent network two ways. Parameters: cell_fn (factory taking the state size and
  # returning an RNN cell), batch_size, state_size, max_steps. It makes inputs, unstacks them into a
  # time-major TensorArray, and defines a per-batch-element loop body that runs a while loop over
  # time, masking outputs and state past the sequence length, and returns the stacked outputs and
  # final state. It runs this with pfor over the batch, and also builds the standard dynamic_rnn on
  # the same cell and inputs. Returns a pair (pfor outputs, dynamic_rnn outputs). Creates graph ops
  # only.
  cell = cell_fn(state_size)
  inputs, sequence_length = dynamic_lstm_input_fn(batch_size, state_size,
                                                  max_steps)
  inputs_ta = tensor_array_ops.TensorArray(
      dtypes.float32, size=max_steps, element_shape=[batch_size, state_size])
  inputs_time_major = array_ops.transpose(inputs, [1, 0, 2])
  inputs_ta = inputs_ta.unstack(inputs_time_major)
  zeros = array_ops.zeros([state_size])

  def loop_fn(i):
    # [FUNCTION DOC]
    # Per-batch-element body: takes the example index, gets its sequence length, runs a while loop
    # over time steps that applies the cell to that example's input and masks output and state once
    # the step reaches the length, then returns the stacked outputs over time and the flattened
    # final state packed in the cell state structure.
    sequence_length_i = array_ops.gather(sequence_length, i)

    def body_fn(t, state, ta):
      # [FUNCTION DOC]
      # While body: reads the input of time t for example i, runs the cell on it with the current
      # state, flattens the output, zeros it after the sequence end (and keeps the old state),
      # writes it in the output array at t and returns next time, new state and array. A TODO notes
      # an unimplemented optimization avoiding the where op.
      inputs_t = array_ops.expand_dims(
          array_ops.gather(inputs_ta.read(t), i), 0)
      output, new_state = cell(inputs_t, state)
      output = array_ops.reshape(output, [-1])
      # TODO(agarwal): one optimization that dynamic_rnn uses is to avoid the
      # array_ops.where when t < min(sequence_length). Doing that requires
      # supporting tf.cond pfor conversion.
      done = t >= sequence_length_i
      output = array_ops.where(done, zeros, output)
      ta = ta.write(t, output)
      new_state = [
          array_ops.where(done, s, ns)
          for s, ns in zip(nest.flatten(state), nest.flatten(new_state))
      ]
      new_state = nest.pack_sequence_as(state, new_state)
      return t + 1, new_state, ta

    def condition_fn(t, _, unused):
      # [FUNCTION DOC]
      # While condition: returns whether the time index is below max_steps; ignores the state and
      # array arguments.
      del unused
      return t < max_steps

    initial_state = cell.zero_state(1, dtypes.float32)
    _, state, ta = while_loop.while_loop(condition_fn, body_fn, [
        0, initial_state,
        tensor_array_ops.TensorArray(dtypes.float32, max_steps)
    ])

    new_state = [array_ops.reshape(x, [-1]) for x in nest.flatten(state)]
    new_state = nest.pack_sequence_as(initial_state, new_state)
    return ta.stack(), new_state

  pfor_output = pfor_control_flow_ops.pfor(loop_fn, batch_size)
  tf_output = rnn.dynamic_rnn(
      cell,
      inputs,
      sequence_length=sequence_length,
      initial_state=cell.zero_state(batch_size, dtypes.float32))
  return pfor_output, tf_output


@test_util.run_all_in_graph_and_eager_modes
class WhileV2Test(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of while loops, scan and gradient jacobians under control flow v2
  # (setUp enables v2, tearDown restores). Runs in graph and eager modes.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: remembers if control flow v2 is enabled, enables it, calls the parent setUp.
    # Side effect: process-wide toggle.
    self._enabled = control_flow_v2_toggles.control_flow_v2_enabled()
    control_flow_v2_toggles.enable_control_flow_v2()
    super(WhileV2Test, self).setUp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture teardown: disables control flow v2 if it was not enabled originally, then calls the
    # parent tearDown.
    if not self._enabled:
      control_flow_v2_toggles.disable_control_flow_v2()
    super(WhileV2Test, self).tearDown()

  def test_while_outside_loop(self):
    # [FUNCTION DOC]
    # A helper function builds a counting while loop; loop body calls it and adds i. 3 iterations.
    # The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially
    # with for_loop, evaluates both and asserts the results are close.

    def _f():
      # [FUNCTION DOC]
      # HELPER: builds the invariant counting while loop for the test.
      # Runs a while loop from 0 to 4 and returns the final counter.
      return while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will add i to a
      # while-loop result computed by the helper. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      return _f() + i

    self._test_loop_fn(loop_fn, 3)

  def test_invariant_while(self):
    # [FUNCTION DOC]
    # Loop-invariant while loop counting to 4 inside the body. 3 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run an invariant
      # while loop. Returns the per-iteration result or tuple of results that pfor stacks along a
      # new leading dimension; no side effects other than creating graph operations.
      return while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    self._test_loop_fn(loop_fn, 3)

  def test_invariant_while_with_control_dependency(self):
    # [FUNCTION DOC]
    # Invariant while loop created under a control dependency on i. 3 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run an invariant
      # while loop under a control dependency on i. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      with ops.control_dependencies([i]):
        return while_loop.while_loop(lambda j: j < 4, lambda j: j + 1, [0])

    self._test_loop_fn(loop_fn, 3)

  def test_while_with_stateful_ops(self):
    # [FUNCTION DOC]
    # While loop with random draws in its body; returns the counter. 3 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run a stateful while
      # loop and return its counter. Returns the per-iteration result or tuple of results that pfor
      # stacks along a new leading dimension; no side effects other than creating graph operations.
      j, _ = while_loop.while_loop(
          lambda j, x: j < 4, lambda j, x:
          (j + 1, x + random_ops.random_uniform([])), [0, 0.])
      return j

    self._test_loop_fn(loop_fn, 3)

  def test_while_with_variable(self):
    # [FUNCTION DOC]
    # While loop adding a captured resource variable (value 5) four times; returns the sum. 3
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.
    v = resource_variable_ops.ResourceVariable(5.)

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will accumulate a
      # captured variable in a while loop. Returns the per-iteration result or tuple of results that
      # pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      _, output = while_loop.while_loop(lambda j, x: j < 4, lambda j, x:
                                        (j + 1, x + v), [0, 0.])
      return output

    self._test_loop_fn(loop_fn, 3)

  def test_while_unstacked_condition(self):
    # [FUNCTION DOC]
    # While loop with invariant condition, body adds loop-variant i; returns both loop variables. 3
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will accumulate i in a
      # while loop with invariant condition. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      return while_loop.while_loop(lambda j, x: j < 4, lambda j, x:
                                   (j + 1, x + i), [0, 0])

    self._test_loop_fn(loop_fn, 3)

  def test_while(self):
    # [FUNCTION DOC]
    # While loop with loop-variant trip count (lengths 4, 0, 2) summing prefix elements of row i;
    # returns both loop variables. 3 iterations. The test harness (_test_loop_fn) runs the loop body
    # both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.
    x = random_ops.random_uniform([3, 5])
    lengths = constant_op.constant([4, 0, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will sum the first
      # lengths_i elements of row i. Returns the per-iteration result or tuple of results that pfor
      # stacks along a new leading dimension; no side effects other than creating graph operations.
      x_i = array_ops.gather(x, i)
      lengths_i = array_ops.gather(lengths, i)

      return while_loop.while_loop(
          lambda j, _: j < lengths_i, lambda j, t:
          (j + 1, t + array_ops.gather(x_i, j)), [0, 0.])

    self._test_loop_fn(loop_fn, 3)

  def test_while_change_input_invariance(self):
    # [FUNCTION DOC]
    # Loop-invariant inputs get loop-dependent operations in the body, some inputs pass through
    # unchanged; trip count depends on i. 3 iterations. The test harness (_test_loop_fn) runs the
    # loop body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts
    # the results are close.
    # This tests cases where a loop invariant input to while has loop dependent
    # operations applied to it inside the while body.
    # It also test inputs that are passed through.
    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run a while loop
      # where invariant inputs become variant. Returns the per-iteration result or tuple of results
      # that pfor stacks along a new leading dimension; no side effects other than creating graph
      # operations.
      return while_loop.while_loop(
          lambda j, *_: j < i, lambda j, x, y, z, w:
          (j + 1, x + i, y + x, z, w), [
              0,
              constant_op.constant(0),
              constant_op.constant(1), i,
              constant_op.constant(2)
          ])

    self._test_loop_fn(loop_fn, 3)

  def test_while_shape_invariants(self):
    # [FUNCTION DOC]
    # While loop with explicit shape invariants for two 2-element vectors, adding i to one and 1 to
    # the other for 4 steps. 3 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will run a while loop
      # with shape invariants. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      return while_loop.while_loop(
          lambda j, *_: j < 4,
          lambda j, x, y: (j + 1, x + i, y + 1),
          [0, constant_op.constant([0, 1]),
           constant_op.constant([2, 3])],
          shape_invariants=[
              None,
              tensor_shape.TensorShape([2]),
              tensor_shape.TensorShape([2])
          ])

    self._test_loop_fn(loop_fn, 3)

  def test_while_jacobian(self):
    # [FUNCTION DOC]
    # Jacobian through a while loop computing x times y four times, inside a tf.function so the loop
    # is not run eagerly. Compares pfor and for_loop versions of per-column gradients; asserts they
    # are close for random float64 inputs (1 by 3 and 3 by 3).
    # Note that we wrap the code below in a tf.function since we don't want the
    # while_loop call to be evaluated eagerly using a python loop.
    @def_function.function
    def _f(x, y, use_pfor):
      # [FUNCTION DOC]
      # Compiled function with inputs x, y and a boolean use_pfor: builds the while loop, defines a
      # per-column gradient body, and returns pfor or for_loop of it (3 iterations) depending on the
      # flag.
      # out = x @ y @ y @ y @ y, where @ is matmul operator.
      _, out = while_loop.while_loop(
          lambda i, _: i < 4, lambda i, out: (i + 1, math_ops.matmul(out, y)),
          [0, x])

      def loop_fn(i):
        # [FUNCTION DOC]
        # Per-iteration body: gathers column i of the loop output and returns the flattened gradient
        # with respect to x.
        out_i = array_ops.gather(out, i, axis=1)
        grad = gradient_ops.gradients(out_i, x)
        return array_ops.reshape(grad[0], [-1])

      if use_pfor:
        return pfor_control_flow_ops.pfor(loop_fn, iters=3)
      else:
        return pfor_control_flow_ops.for_loop(
            loop_fn, iters=3, loop_fn_dtypes=out.dtype)

    x = constant_op.constant(np.random.uniform(size=(1, 3)))
    y = constant_op.constant(np.random.uniform(size=(3, 3)))
    self.assertAllClose(_f(x, y, True), _f(x, y, False))

  def test_scan(self):
    # [FUNCTION DOC]
    # Gradient check of vectorized_map over a function using scan: log_prob sums scan results of
    # squared differences against random data (seed 42). Checks theoretical vs numerical jacobian of
    # the vectorized function at a variable of ones, using delta 1e-3 and rtol 1e-2. Side effect:
    # seeds the numpy random generator.
    np.random.seed(seed=42)
    data = np.random.randn(3).astype(np.float32)

    def log_prob(x):
      # [FUNCTION DOC]
      # Mapped function: scans over the captured data array with initializer zero, producing squared
      # differences with its scalar input, and returns the sum.
      return math_ops.reduce_sum(functional_ops.scan_v2(
          lambda _, yi: (x - yi)**2,
          elems=data,
          initializer=constant_op.constant(0.)))

    x = variables.Variable(array_ops.ones([2]))
    self.evaluate(x.initializer)
    v_log_prob = lambda x: pfor_control_flow_ops.vectorized_map(log_prob, x)
    theoretical, numerical = gradient_checker_v2.compute_gradient(
        v_log_prob, (x,), delta=1e-3)
    self.assertAllClose(theoretical, numerical, rtol=1e-2)

  def test_scan_captured_variable(self):
    # [FUNCTION DOC]
    # Eager-only (skipped otherwise). Scan reading from a captured variable (range 10) inside a pfor
    # of 2 iterations under a gradient tape; asserts the gradient of the result with respect to the
    # variable is 2 for all ten entries.
    if not context.executing_eagerly():
      self.skipTest("Test only written for 2.x")
    v = variables.Variable(math_ops.range(10, dtype=dtypes.float32))

    def loop_fn(idx):
      # [FUNCTION DOC]
      # Per-iteration body: ignores the index and runs a scan over indices that gathers from the
      # captured variable, initializer 0.0; returns the scanned result.
      del idx
      return functional_ops.scan_v2(lambda _, i: array_ops.gather(v, i),
                                    elems=math_ops.range(v.shape[0]),
                                    initializer=0.0)
    with backprop.GradientTape() as tape:
      result = pfor_control_flow_ops.pfor(loop_fn, 2)
    self.assertAllClose([2.] * 10, tape.gradient(result, v))


@test_util.run_all_in_graph_and_eager_modes
class NestedControlFlowTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor on nested conditionals and while loops under control flow v2 (setUp enables
  # v2). Builds nested combinations from the _cond and _while helpers.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: remembers if control flow v2 is enabled, enables it, calls the parent setUp.
    self._enabled = control_flow_v2_toggles.control_flow_v2_enabled()
    control_flow_v2_toggles.enable_control_flow_v2()
    super(NestedControlFlowTest, self).setUp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture teardown: disables control flow v2 if it was off before, then calls the parent
    # tearDown.
    if not self._enabled:
      control_flow_v2_toggles.disable_control_flow_v2()
    super(NestedControlFlowTest, self).tearDown()

  def _cond(self, f=None, split=0):
    # [FUNCTION DOC]
    # HELPER: builds conditional bodies for nested control flow tests.
    # Parameters: f (optional function of x and y returning a pair; default identity pair) and split
    # (threshold, default 0). Returns a function of (x, y) that takes f(x, y) when y is above split
    # and otherwise returns (x plus one, y).
    if f is None:
      f = lambda x, y: (x, y)

    def _f(x, y):
      # [FUNCTION DOC]
      # HELPER: the conditional produced by _cond.
      # Applies the cond: if y exceeds the captured split returns the captured f of (x, y), else
      # returns x plus 1.0 and y.
      return cond.cond(y > split, lambda: f(x, y), lambda:
                       (x + 1., y))

    return _f

  def _while(self, f=None):
    # [FUNCTION DOC]
    # HELPER: builds while-loop bodies for nested control flow tests.
    # Parameter f (optional function of x and y returning a pair; default identity). Returns a
    # function of (x, y) that runs a while loop for y iterations accumulating element j of the first
    # output of f(x, y) onto x; returns the accumulated value and y.
    if f is None:
      f = lambda x, y: (x, y)

    def _f(x, y):
      # [FUNCTION DOC]
      # HELPER: the while body produced by _while.
      # Runs a loop from 0 to y adding gathered elements of f's first result to x; returns the
      # accumulator and y.
      return while_loop.while_loop(
          lambda j, _: j < y, lambda j, t:
          (j + 1, t + array_ops.gather(f(x, y)[0], j)), [0, x])[1], y

    return _f

  def _test_helper(self, f):
    # [FUNCTION DOC]
    # HELPER: shared test driver for the nested control flow tests.
    # Parameter f: function of two tensors returning a pair. Creates random 5 by 5 x and an int
    # vector y with positive and negative entries; runs the harness over 5 iterations with a body
    # applying f to row i of x and element i of y.
    x = random_ops.random_uniform([5, 5])
    y = constant_op.constant([4, -1, 2, -2, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body: gathers row i of x and element i of y and returns f applied to them.
      x_i = array_ops.gather(x, i)
      y_i = array_ops.gather(y, i)
      return f(x_i, y_i)

    self._test_loop_fn(loop_fn, 5)

  def test_cond_while(self):
    # [FUNCTION DOC]
    # Nested control flow test: a conditional whose true branch is a while loop. Uses the shared
    # driver.
    self._test_helper(self._cond(self._while()))

  def test_while_cond(self):
    # [FUNCTION DOC]
    # Nested control flow test: a while loop whose body is a conditional. Uses the shared driver.
    self._test_helper(self._while(self._cond()))

  def test_while_while(self):
    # [FUNCTION DOC]
    # Nested control flow test: a while loop whose body contains another while loop. Uses the shared
    # driver.
    self._test_helper(self._while(self._while()))

  def test_cond_cond(self):
    # [FUNCTION DOC]
    # Nested control flow test: a conditional whose true branch is another conditional. Uses the
    # shared driver.
    self._test_helper(self._cond(self._cond()))


@test_util.run_all_in_graph_and_eager_modes
@test_util.with_control_flow_v2
class StatelessIfTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of stateless conditionals (cond_v2) with loop-variant and
  # invariant predicates and branches. Control flow v2, graph and eager modes.

  def test_loop_variant_cond(self):
    # [FUNCTION DOC]
    # Cond with a loop-variant predicate (element i below 2.5) where outputs mix loop-variant and
    # invariant values across branches; 5 iterations. The test harness (_test_loop_fn) runs the loop
    # body both vectorized with pfor and sequentially with for_loop, evaluates both and asserts the
    # results are close.
    x = [1, 2, 3, 4, 5.]
    y = 2.5

    @def_function.function
    def loop_fn(i):
      # [FUNCTION DOC]
      # Compiled per-iteration body: gathers element i and chooses with cond_v2 between two branches
      # returning four values (mixes of variant and invariant ones).
      x_i = array_ops.gather(x, i)
      # Note that the output has a combination of then and else branches being
      # loop variant / invariant.
      return cond_v2.cond_v2(x_i < y, lambda: (y - x_i, y, 1., 2.), lambda:
                             (x_i - y, 0., y, 3.))

    self._test_loop_fn(loop_fn, iters=5)

  def test_loop_invariant_cond(self):
    # [FUNCTION DOC]
    # Cond with a loop-invariant random predicate and variant branch values; 5 iterations. The test
    # harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with
    # for_loop, evaluates both and asserts the results are close.
    x = [1, 2, 3, 4, 5.]
    y = 0.5
    z = random_ops.random_uniform([])

    @def_function.function
    def loop_fn(i):
      # [FUNCTION DOC]
      # Compiled per-iteration body: cond_v2 on an invariant predicate with branches returning four
      # values mixing variant and invariant ones.
      x_i = array_ops.gather(x, i)
      # Note that the output has a combination of then and else branches being
      # loop variant / invariant.
      return cond_v2.cond_v2(z < y, lambda: (y - x_i, y, 1., 2.), lambda:
                             (x_i - y, 0., y, 3.))

    self._test_loop_fn(loop_fn, iters=5)

  def test_empty_branch(self):
    # [FUNCTION DOC]
    # Cond where the predicate is always true (6.0 against elements up to 5), so one branch is never
    # taken (noted as empty); 5 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    x = [1, 2, 3, 4, 5.]
    y = 6.

    @def_function.function
    def loop_fn(i):
      # [FUNCTION DOC]
      # Compiled per-iteration body: cond_v2 on x_i below 6 with two branches returning four-value
      # tuples.
      x_i = array_ops.gather(x, i)
      return cond_v2.cond_v2(
          x_i < y,  # Note that else branch is empty.
          lambda: (y - x_i, y, 1., 2.),
          lambda: (x_i - y, 0., y, 3.))

    self._test_loop_fn(loop_fn, iters=5)


@test_util.run_all_in_graph_and_eager_modes
@test_util.with_control_flow_v2
class IfTest(PForTestCase):
  # [CLASS DOC]
  # Test case for conditionals that read variables under pfor (control flow v2, graph and eager
  # modes).

  def test_read_var(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: skipped unconditionally at the start (flaky, b/156438918), so the rest never
    # runs.
    # Intended: cond whose branches read a captured resource variable, 5 iterations.
    self.skipTest("b/156438918")  # Flaky

    x = [1, 2, 3, 4, 5.]
    y = 2.5
    z = resource_variable_ops.ResourceVariable(5.)

    @def_function.function
    def loop_fn(i):
      # [FUNCTION DOC]
      # Compiled per-iteration body: cond_v2 on x_i below 2.5 returning z minus x_i or z plus x_i
      # (never executed because the test skips).
      x_i = array_ops.gather(x, i)
      return cond_v2.cond_v2(x_i < y, lambda: z - x_i, lambda: z + x_i)

    self._test_loop_fn(loop_fn, iters=5)


class RNNTest(PForTestCase):
  # [CLASS DOC]
  # Test case comparing pfor-based dynamic RNN/LSTM implementations against the standard
  # dynamic_rnn. v1 graph mode only.

  @test_util.run_v1_only("b/122612051")
  def test_dynamic_rnn(self):
    # [FUNCTION DOC]
    # v1-only. Builds pfor and dynamic_rnn graphs for a basic RNN cell (batch 3, state 5, 7 steps)
    # via create_dynamic_lstm and asserts their outputs are equal.
    pfor_outputs, tf_outputs = create_dynamic_lstm(rnn_cell.BasicRNNCell, 3, 5,
                                                   7)
    self.run_and_assert_equal(pfor_outputs, tf_outputs)

  @test_util.run_v1_only("b/122612051")
  def test_dynamic_lstm(self):
    # [FUNCTION DOC]
    # v1-only. Same as the RNN test but with a basic LSTM cell (batch 3, state 5, 7 steps).
    pfor_outputs, tf_outputs = create_dynamic_lstm(rnn_cell.BasicLSTMCell, 3, 5,
                                                   7)
    self.run_and_assert_equal(pfor_outputs, tf_outputs)


# TODO(agarwal): benchmark numbers on GPU for graphs based on while_loop
# conversion don't look good. Some of it seems like lot of copies between host
# and device. Optimize that.
class Benchmarks(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class comparing pfor with manual vectorization, for_loop and map_fn on add, matmul,
  # map, while loops, dynamic RNN and reductions. Each benchmark reports timings.

  def _run(self, targets, iters, name=None):
    # [FUNCTION DOC]
    # HELPER: timing helper for the benchmark methods.
    # Parameters: targets (tensors or nested structure), iters (number of timed runs) and an
    # optional benchmark name. Reduces each target to its first element (so the device work is
    # finished), runs a new session with global variable initialization, creates a callable, warms
    # up once, times the given runs, converts to average milliseconds per run, reports it via the
    # benchmark reporter and returns it. Creates a session as a side effect.

    def _done(t):
      # [FUNCTION DOC]
      # HELPER: forces completion of device computation in timing.
      # Returns the first element of the flattened tensor; control dependencies would not guarantee
      # GPU completion. Assumes the tensor is not empty.
      # Note that we don't use tf.control_dependencies since that will not make
      # sure that the computation on GPU has actually finished. So we fetch the
      # first element of the output, and assume that this will not be called on
      # empty tensors.
      return array_ops.gather(array_ops.reshape(t, [-1]), 0)

    targets = [_done(x) for x in nest.flatten(targets)]
    sess = session.Session()
    with sess:
      init = variables.global_variables_initializer()
      sess.run(init)
      run_fn = sess.make_callable(targets)
      run_fn()  # Warm up
      begin = time.time()
      for _ in range(iters):
        run_fn()
      end = time.time()
    avg_time_ms = 1000 * (end - begin) / iters
    self.report_benchmark(iters=iters, wall_time=avg_time_ms, name=name)
    return avg_time_ms

  def benchmark_sess_run_overhead(self):
    # [FUNCTION DOC]
    # Measures the overhead of a session run of a trivial constant, 10000 runs.
    with ops.Graph().as_default():
      x = constant_op.constant(1.0)
      self._run(x, 10000, name="session_run_overhead")

  def benchmark_add(self):
    # [FUNCTION DOC]
    # Compares manual add, pfor and for_loop of adding two random 256 by 1000 matrices row by row;
    # runs 1000, 1000 and 100 times.
    with ops.Graph().as_default():
      n = 256
      params = 1000
      x = random_ops.random_normal([n, params])
      y = random_ops.random_normal([n, params])

      def loop_fn(i):
        # [FUNCTION DOC]
        # Per-row body: gathers row i of x and y and returns their sum.
        x_i = array_ops.gather(x, i)
        y_i = array_ops.gather(y, i)
        return x_i + y_i

      pfor_outputs = pfor_control_flow_ops.pfor(loop_fn, n)
      while_outputs = pfor_control_flow_ops.for_loop(loop_fn, dtypes.float32, n)
      manual = x + y

      self._run(manual, 1000, name="manual_add")
      self._run(pfor_outputs, 1000, name="pfor_add")
      self._run(while_outputs, 100, name="while_add")

  def benchmark_matmul(self):
    # [FUNCTION DOC]
    # Compares manual matmul, pfor and for_loop for multiplying each row of a 1024 by 1000 random
    # matrix with a 1000 by 1000 matrix; runs 1000, 1000 and 100 times.
    with ops.Graph().as_default():
      n = 1024
      params = 1000
      x = random_ops.random_normal([n, params])
      y = random_ops.random_normal([params, params])

      def loop_fn(i):
        # [FUNCTION DOC]
        # Per-row body: expands row i to a 1 by 1000 matrix and multiplies by y.
        x_i = array_ops.expand_dims(array_ops.gather(x, i), 0)
        return math_ops.matmul(x_i, y)

      pfor_outputs = pfor_control_flow_ops.pfor(loop_fn, n)
      while_outputs = pfor_control_flow_ops.for_loop(loop_fn, dtypes.float32, n)
      manual = math_ops.matmul(x, y)

      self._run(manual, 1000, name="manual_matmul")
      self._run(pfor_outputs, 1000, name="pfor_matmul")
      self._run(while_outputs, 100, name="while_matmul")

  def benchmark_map_fn(self):
    # [FUNCTION DOC]
    # Compares map_fn with a pfor-based map for squaring each row of a random 256 by 1000 tensor;
    # 100 runs each.
    with ops.Graph().as_default():
      b = 256
      params = 1000
      inp = random_ops.random_normal((b, params))
      fn = lambda x: x * x

      def pfor_map_fn(f, x):
        # [FUNCTION DOC]
        # HELPER: pfor equivalent of map_fn for the benchmark.
        # Parameters: f (function applied per row) and x (tensor). Returns pfor over the first
        # dimension of x applying f to each row.
        return pfor_control_flow_ops.pfor(lambda i: f(array_ops.gather(x, i)),
                                          array_ops.shape(x)[0])

      map_output = map_fn.map_fn(fn, inp)
      pfor_output = pfor_map_fn(fn, inp)

      self._run(map_output, 100, name="tf_map_fn")
      self._run(pfor_output, 100, name="pfor_map_fn")

  def benchmark_basic_while(self):
    # [FUNCTION DOC]
    # Compares pfor and for_loop on a while loop with variant condition for 50 iterations; 100 runs
    # each.
    with ops.Graph().as_default():

      def loop_fn(i):
        # [FUNCTION DOC]
        # Per-iteration body: runs a while loop incrementing a counter to i while accumulating i;
        # returns the sum.
        _, s = while_loop.while_loop(lambda t, x: t < i, lambda t, x:
                                     (t + 1, x + i), [0, 0])
        return s

      iters = 50
      pfor_output = pfor_control_flow_ops.pfor(loop_fn, iters)
      for_loop_output = pfor_control_flow_ops.for_loop(loop_fn, dtypes.int32,
                                                       iters)
      self._run(pfor_output, 100, name="pfor_basic")
      self._run(for_loop_output, 100, name="for_loop_basic")

  def benchmark_dynamic_rnn(self):
    # [FUNCTION DOC]
    # Compares the pfor dynamic RNN (batch 128, state 512, 16 steps) with the standard dynamic_rnn;
    # 100 runs each.
    with ops.Graph().as_default():
      pfor_outputs, tf_outputs = create_dynamic_lstm(rnn_cell.BasicRNNCell, 128,
                                                     512, 16)
      self._run(pfor_outputs, 100, name="pfor_rnn")
      self._run(tf_outputs, 100, name="tf_rnn")

  def benchmark_reduction(self):
    # [FUNCTION DOC]
    # Benchmarks pfor_config.reduce_concat followed by matmul and sum for a 1024 by 1024 input
    # against the plain computation, checking reduction adds no overhead (output treated as loop
    # invariant); 30 runs each.
    n = 1024
    with ops.Graph().as_default():
      x = random_ops.random_uniform([n, n])
      w = random_ops.random_uniform([n, n])

      def loop_fn(i, pfor_config):
        # [FUNCTION DOC]
        # Per-iteration body with iteration index and pfor config: concatenates all rows via
        # reduce_concat, multiplies by w and sums.
        x_i = array_ops.gather(x, i)
        return math_ops.reduce_sum(
            math_ops.matmul(pfor_config.reduce_concat(x_i), w))

      # Note that output_reduction will be tiled, so there may be some minor
      # overheads compared to output_no_reduction.
      output_reduction = pfor_control_flow_ops.pfor(loop_fn, n)
      output_no_reduction = math_ops.reduce_sum(math_ops.matmul(x, w))
      # Benchmark to test that reduction does not add overhead and its output is
      # treated as loop invariant.
      self._run(output_reduction, 30, name="matmul_reduction")
      self._run(output_no_reduction, 30, name="matmul_no_reduction")


class SparseTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor producing SparseTensor results with various stacked components. v1 graph mode
  # only.

  @test_util.run_v1_only("b/122612051")
  def test_var_loop_len(self):
    # [FUNCTION DOC]
    # v1-only. pfor with a placeholder iteration count and a body returning a constant sparse
    # tensor; runs in a session feeding 3, asserting only that it runs.
    num_iters = array_ops.placeholder(dtypes.int32)

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body returning the constant sparse vector with values 4, 5, 6 at indices 0, 1,
      # 2 and dense shape 3.
      return sparse_tensor.SparseTensor([[0], [1], [2]], [4, 5, 6],
                                        [3])  # [0, 2, 0]

    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    with self.cached_session() as sess:
      sess.run(pfor, feed_dict={num_iters: 3})

  @test_util.run_v1_only("b/122612051")
  def test_sparse_result_none_stacked(self):
    # [FUNCTION DOC]
    # v1-only. Body returns a loop-invariant sparse tensor; compares pfor over 10 iterations with
    # the manually built stacked sparse tensor via run_and_assert_equal.
    num_iters = 10

    def loop_fn(_):
      # [FUNCTION DOC]
      # Returns a loop-invariant sparse vector (values 4, 5, 6).
      return sparse_tensor.SparseTensor([[0], [1], [2]], [4, 5, 6],
                                        [3])  # [0, 2, 0]

    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)

    indices = [[i, j] for i in range(num_iters) for j in range(3)]
    values = [4, 5, 6] * num_iters
    dense_shapes = [num_iters, 3]
    # Expected result: [[4, 5, 6], [4, 5, 6], [4, 5, 6], ...]
    manual = sparse_tensor.SparseTensor(indices, values, dense_shapes)
    self.run_and_assert_equal(pfor, manual)

  @test_util.run_v1_only("b/122612051")
  def test_sparse_result_all_stacked(self):
    # [FUNCTION DOC]
    # v1-only. Body returns a sparse tensor whose indices, values and shape all depend on i;
    # compares to a manual diagonal sparse tensor over 10 iterations.
    num_iters = 10

    def loop_fn(i):
      # [FUNCTION DOC]
      # Casts i to int64 and builds a sparse tensor with index i, value i and shape i plus 1.
      i = array_ops.expand_dims(math_ops.cast(i, dtypes.int64), 0)
      indices = array_ops.expand_dims(i, 0)
      return sparse_tensor.SparseTensor(indices, i, i + 1)  # [0, ..., 0, i]

    # Expected result: [[0], [0, 1], [0, 0, 2], [0, 0, 0, 3], ...]
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    manual = sparse_tensor.SparseTensor([[i, i] for i in range(num_iters)],
                                        list(range(num_iters)),
                                        (num_iters, num_iters))
    self.run_and_assert_equal(pfor, manual)

  @test_util.run_v1_only("b/122612051")
  def test_sparse_result_indices_stacked(self):
    # [FUNCTION DOC]
    # v1-only. Only the indices depend on i (identity matrix result); 10 iterations compared with a
    # manual tensor.
    num_iters = 10

    def loop_fn(i):
      # [FUNCTION DOC]
      # Builds a sparse vector with loop-variant index, constant value 1 and constant shape.
      i = array_ops.expand_dims(math_ops.cast(i, dtypes.int64), 0)
      indices = array_ops.expand_dims(i, 0)
      return sparse_tensor.SparseTensor(indices, [1], [num_iters])

    # Expected result: identity matrix size num_iters * num_iters
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    manual = sparse_tensor.SparseTensor([[i, i] for i in range(num_iters)],
                                        [1] * num_iters, (num_iters, num_iters))
    self.run_and_assert_equal(pfor, manual)

  @test_util.run_v1_only("b/122612051")
  def test_sparse_result_values_stacked(self):
    # [FUNCTION DOC]
    # v1-only. Only values depend on i; 10 iterations compared with a manual tensor.
    num_iters = 10

    def loop_fn(i):
      # [FUNCTION DOC]
      # Builds a sparse vector with constant index 0, value i and constant shape.
      i = array_ops.expand_dims(math_ops.cast(i, dtypes.int64), 0)
      return sparse_tensor.SparseTensor([[0]], i, [num_iters])  # [i, 0, ..., 0]

    # Expected result: [[1, 0, ...], [2, 0, ...], [3, 0, ...], ...]
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    manual = sparse_tensor.SparseTensor([[i, 0] for i in range(num_iters)],
                                        list(range(num_iters)),
                                        (num_iters, num_iters))
    self.run_and_assert_equal(pfor, manual)

  @test_util.run_v1_only("b/122612051")
  def test_sparse_result_shapes_stacked(self):
    # [FUNCTION DOC]
    # v1-only. Only the dense shape depends on i; 10 iterations compared with a manual tensor.
    num_iters = 10

    def loop_fn(i):
      # [FUNCTION DOC]
      # Builds a sparse vector with constant index and value and dense shape i plus 1.
      i = array_ops.expand_dims(math_ops.cast(i, dtypes.int64), 0)
      return sparse_tensor.SparseTensor([[0]], [1], i + 1)  # [1, 0, ..., 0]

    # Expected result: [[1, 0, 0, ...], [1, 0, 0, ...], ...]
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    manual = sparse_tensor.SparseTensor([[i, 0] for i in range(num_iters)],
                                        [1] * num_iters, (num_iters, num_iters))
    self.run_and_assert_equal(pfor, manual)

  @test_util.run_v1_only("b/122612051")
  def test_sparse_result_shapes_stacked_2D(self):
    # [FUNCTION DOC]
    # v1-only. Two-dimensional result whose dense shape depends on i; 10 iterations compared with a
    # manual three-dimensional sparse tensor.
    num_iters = 10

    def loop_fn(i):
      # [FUNCTION DOC]
      # Builds a 2-D sparse tensor with one value at (0, 0) and a shape of (i plus 1) square.
      i = array_ops.expand_dims(math_ops.cast(i + 1, dtypes.int64), 0)
      shape = array_ops.concat([i, i], 0)
      return sparse_tensor.SparseTensor([[0, 0]], [1], shape)  # [1, 0, ..., 0]

    # Expected result: [[[1, 0, ...], [0, ..., 0], [0, ..., 0], ...], ...]
    pfor = pfor_control_flow_ops.pfor(loop_fn, num_iters)
    manual = sparse_tensor.SparseTensor([[i, 0, 0] for i in range(num_iters)],
                                        [1] * num_iters,
                                        (num_iters, num_iters, num_iters))
    self.run_and_assert_equal(pfor, manual)


# Dummy CompositeTensor to test CompositeTensor support.
class Particle(composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # Dummy CompositeTensor used by tests: a batch of particles with a mass tensor and a scalar
  # velocity tensor, whose static shape is the broadcast of the two component shapes. Provides a
  # type spec for batching.
  """A (batch of) particles each defined by a mass and a scalar velocity."""

  def __init__(self, mass, velocity):
    # [FUNCTION DOC]
    # Constructor: converts mass and velocity to tensors, stores them as attributes and sets shape
    # to their broadcast static shape.
    mass = ops.convert_to_tensor(mass)
    velocity = ops.convert_to_tensor(velocity)
    self.shape = array_ops.broadcast_static_shape(mass.shape, velocity.shape)
    self.mass = mass
    self.velocity = velocity

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Property returning a ParticleSpec built from the type specs of mass and velocity.
    return ParticleSpec(
        type_spec.type_spec_from_value(self.mass),
        type_spec.type_spec_from_value(self.velocity))


class ParticleSpec(type_spec.BatchableTypeSpec):
  # [CLASS DOC]
  # Batchable type spec for Particle used to test pfor's CompositeTensor support. Holds the
  # component specs (mass and velocity) and the broadcast shape; component shapes are padded to the
  # same rank when batching.

  def __init__(self, mass, velocity):
    # [FUNCTION DOC]
    # Constructor: parameters mass and velocity are component specs; stores them and the broadcast
    # static shape.
    self.shape = array_ops.broadcast_static_shape(
        mass.shape, velocity.shape)
    self.mass = mass
    self.velocity = velocity

  def _serialize(self):
    # [FUNCTION DOC]
    # Returns the pair (mass spec, velocity spec) describing this type spec.
    return (self.mass, self.velocity)

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # Property returning the Particle class as the value type.
    return Particle

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Property returning the pair of component specs.
    return (self.mass, self.velocity)

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Parameter value: a Particle. Returns (mass, velocity).
    return (value.mass, value.velocity)

  def _from_components(self, components):
    # [FUNCTION DOC]
    # Parameter components: sequence of mass and velocity. Returns a new Particle.
    return Particle(*components)

  def _pad_shape_to_full_rank(self, s):
    # [FUNCTION DOC]
    # HELPER: rank alignment for batching in ParticleSpec.
    # Pads a component shape with leading ones until it has the rank of the broadcast shape.
    # Parameter s: a shape. Returns the padded shape.
    """Pad component shapes with 1's so all components have the same rank."""
    return tensor_shape.TensorShape(
        [1] * (self.shape.ndims - s.ndims)).concatenate(s)

  def _batch(self, batch_size):
    # [FUNCTION DOC]
    # Parameter batch_size. Returns a spec whose mass and velocity have a leading batch dimension
    # followed by the rank-padded original shapes.
    return ParticleSpec(
        mass=tensor_spec.TensorSpec(
            dtype=self.mass.dtype,
            shape=tensor_shape.TensorShape([batch_size]).concatenate(
                self._pad_shape_to_full_rank(self.mass.shape))),
        velocity=tensor_spec.TensorSpec(
            dtype=self.velocity.dtype,
            shape=tensor_shape.TensorShape([batch_size]).concatenate(
                self._pad_shape_to_full_rank(self.velocity.shape))))

  def _unbatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: it does not undo the rank padding applied by _batch.
    # Returns a spec with the leading dimension removed from both components.
    return ParticleSpec(
                tensor_spec.TensorSpec(dtype=self.mass.dtype,
                                       shape=self.mass.shape[1:]),
                tensor_spec.TensorSpec(dtype=self.velocity.dtype,
                                       shape=self.velocity.shape[1:]))

  def _to_tensor_list(self, value):
    # [FUNCTION DOC]
    # Parameter value: a Particle. Returns a two-element list with mass and velocity each reshaped
    # to the padded full rank.
    return [array_ops.reshape(
                value.mass,
                self._pad_shape_to_full_rank(value.mass.shape)),
            array_ops.reshape(
                value.velocity,
                self._pad_shape_to_full_rank(value.velocity.shape))]


class CompositeTensorTest(PForTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for pfor and vectorized_map with CompositeTensor values (Particle and ragged tensors);
  # also parameterized.

  @parameterized.parameters((None,), (3,))
  def test_create_composite_inside_loop(self, parallel_iterations):
    # [FUNCTION DOC]
    # Parameterized over parallel_iterations of none and 3. Body builds a Particle with constant
    # mass 4 and gathered velocity for 10 particles; asserts the batched particle has mass 4
    # everywhere and velocities equal to the input.
    num_particles = 10
    velocities = random_ops.random_uniform([num_particles])
    particles = pfor_control_flow_ops.pfor(
        # Build a batch of particles all with the same mass.
        lambda i: Particle(mass=4., velocity=array_ops.gather(velocities, i)),
        num_particles,
        parallel_iterations=parallel_iterations)
    particles_mass, particles_velocity, velocities = self.evaluate(
        (particles.mass, particles.velocity, velocities))
    self.assertAllEqual(particles_mass, 4. * np.ones([num_particles]))
    self.assertAllEqual(particles_velocity, velocities)

  @parameterized.parameters((None,), (3,))
  def test_composite_is_converted_to_batched_tensor(
      self, parallel_iterations):
    # [FUNCTION DOC]
    # Parameterized over parallel_iterations none and 3. Body builds particles with mass shape 3 and
    # velocity shape 5 by 3 for 4 iterations; asserts batched shapes are 4 by 1 by 3 and 4 by 5 by 3
    # (padded for consistent broadcast).
    particles = pfor_control_flow_ops.pfor(
        lambda _: Particle(mass=random_ops.random_uniform([3]),  # pylint: disable=g-long-lambda
                           velocity=random_ops.random_uniform([5, 3])),
        4,
        parallel_iterations=parallel_iterations)
    # Naively batching the component shapes would give `[4, 3]` and `[4, 5, 3]`
    # which have no consistent broadcast shape.
    self.assertEqual(particles.mass.shape, [4, 1, 3])
    self.assertAllEqual(particles.velocity.shape, [4, 5, 3])

  def test_vectorized_map_gathers_composite_tensors(self):
    # [FUNCTION DOC]
    # vectorized_map over a Particle with five elements, product of mass and velocity; asserts it
    # equals the element-wise product.
    particles = Particle(mass=[1., 2., 3., 4., 5.],
                         velocity=[1., 2., 3., 4., 5.])
    self.assertAllEqual(
        pfor_control_flow_ops.vectorized_map(
            lambda x: x.mass * x.velocity, particles),
        particles.mass * particles.velocity)

  def test_vectorized_map_of_ragged_tensors(self):
    # [FUNCTION DOC]
    # vectorized_map over a uniform-row-length ragged tensor (shape 2 by 2 by 3) converting each
    # element to a dense tensor; asserts equality with the dense conversion of the whole.
    # Vmap should be able to handle ragged Tensors as long as they're not
    # *actually* ragged.
    ragged = ragged_tensor.RaggedTensor.from_uniform_row_length(
        ragged_tensor.RaggedTensor.from_row_lengths(
            values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            row_lengths=[3, 3, 3, 3]),
        uniform_row_length=2)  # Overall shape [2, 2, 3].
    self.assertAllEqual(
        pfor_control_flow_ops.vectorized_map(
            lambda x: x.to_tensor(shape=[2, 3]), ragged),
        ragged.to_tensor(shape=[2, 2, 3]))


class ParsingTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of parsing ops (decode_csv and parse_single_example).

  def test_decode_csv(self):
    # [FUNCTION DOC]
    # Harness run on decode_csv with colon delimiter and default values on three records. 3
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.
    csv_tensor = constant_op.constant([["1:2:3"], ["::"], ["7:8:9"]])
    kwargs = {"record_defaults": [[10], [20], [30]], "field_delim": ":"}

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will decode the i-th csv
      # line using default values and colon delimiter. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      line = array_ops.gather(csv_tensor, i)
      return parsing_ops.decode_csv(line, **kwargs)

    self._test_loop_fn(loop_fn, iters=3)

  @test_util.run_v1_only("b/122612051")
  def test_parse_single_example(self):
    # [FUNCTION DOC]
    # v1-only. Builds 10 serialized examples with dense and sparse int and string features; compares
    # pfor of parse_single_example over rows with parse_example on the whole batch using
    # run_and_assert_equal.

    def _int64_feature(*values):
      # [FUNCTION DOC]
      # HELPER: builds test example features.
      # Returns a Feature proto with an int64 list of the given values.
      return feature_pb2.Feature(int64_list=feature_pb2.Int64List(value=values))

    def _bytes_feature(*values):
      # [FUNCTION DOC]
      # HELPER: builds test example features.
      # Returns a Feature proto with a bytes list of the given strings encoded as UTF-8.
      return feature_pb2.Feature(
          bytes_list=feature_pb2.BytesList(
              value=[v.encode("utf-8") for v in values]))

    examples = constant_op.constant([
        example_pb2.Example(
            features=feature_pb2.Features(
                feature={
                    "dense_int": _int64_feature(i),
                    "dense_str": _bytes_feature(str(i)),
                    "sparse_int": _int64_feature(i, i * 2, i * 4, i * 8),
                    "sparse_str": _bytes_feature(*["abc"] * i)
                })).SerializeToString() for i in range(10)
    ])

    features = {
        "dense_int": parsing_ops.FixedLenFeature((), dtypes.int64, 0),
        "dense_str": parsing_ops.FixedLenFeature((), dtypes.string, ""),
        "sparse_int": parsing_ops.VarLenFeature(dtypes.int64),
        "sparse_str": parsing_ops.VarLenFeature(dtypes.string),
    }

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will parse the i-th
      # serialized example. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      example_proto = array_ops.gather(examples, i)
      f = parsing_ops.parse_single_example(example_proto, features)
      return f

    pfor = pfor_control_flow_ops.pfor(loop_fn, iters=10)
    manual = parsing_ops.parse_example(examples, features)
    self.run_and_assert_equal(pfor, manual)


class PartitionedCallTest(PForTestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of function calls (partitioned call ops) including nested
  # functions and gradients.

  def test_simple(self):
    # [FUNCTION DOC]
    # Harness run on a call to a compiled function (square plus one) with loop-variant input. 4
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # Compiled function: returns square of x plus one.
      return math_ops.square(x) + 1

    z = random_ops.random_uniform([4])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will call the compiled
      # function on element i. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      return f(array_ops.gather(z, i))

    self._test_loop_fn(loop_fn, 4)

  def test_nested_calls(self):
    # [FUNCTION DOC]
    # Harness run on an outer function calling an inner function; 4 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.

    @def_function.function
    def inner(x):
      # [FUNCTION DOC]
      # Compiled function returning x squared.
      return math_ops.square(x)

    @def_function.function
    def outer(y):
      # [FUNCTION DOC]
      # Compiled function returning the sum of inner(y) plus 2.
      return math_ops.reduce_sum(inner(y)) + 2

    z = random_ops.random_uniform([4, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will call the outer
      # function on row i. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      return outer(array_ops.gather(z, i))

    self._test_loop_fn(loop_fn, 4)

  def test_nested_calls_loop_fn_autograph(self):
    # [FUNCTION DOC]
    # Runs the harness inside a compiled function so autograph converts the loop function containing
    # a for loop over a tensor range; 4 iterations. A TODO asks whether coverage should be extended.
    #TODO (@bhack) Do we need to extend the coverage?

    def loop_fn(x):
      # [FUNCTION DOC]
      # Loop body: iterates over range of the constant 3 doing nothing, then returns the square of
      # x. Parameter x is the iteration index.
      for y in range(array_ops.constant(3)):
        pass
      return math_ops.square(x)

    @def_function.function
    def loop_fn_caller():
      # [FUNCTION DOC]
      # Compiled function with no arguments that runs the harness on loop_fn with 4 iterations;
      # returns nothing.
      self._test_loop_fn(loop_fn, 4)

    loop_fn_caller()


  def test_nested_definition(self):
    # [FUNCTION DOC]
    # Harness run where the inner function is defined inside the outer compiled function. 4
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.

    @def_function.function
    def outer(y):
      # [FUNCTION DOC]
      # Compiled function that defines an inner compiled function and returns sum of inner(y) plus
      # 2.

      @def_function.function
      def inner(x):
        # [FUNCTION DOC]
        # Compiled function returning x squared plus one.
        return math_ops.square(x) + 1

      return math_ops.reduce_sum(inner(y)) + 2

    z = random_ops.random_uniform([4, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will call the outer
      # function on row i. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      return outer(array_ops.gather(z, i))

    self._test_loop_fn(loop_fn, 4)

  def test_gradients(self):
    # [FUNCTION DOC]
    # Harness run where the body calls a function under a gradient tape; returns the output and
    # gradient. 4 iterations. The test harness (_test_loop_fn) runs the loop body both vectorized
    # with pfor and sequentially with for_loop, evaluates both and asserts the results are close.

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # Compiled function: square of x plus one.
      return math_ops.square(x) + 1

    z = random_ops.random_uniform([4, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will call f on row i
      # under a tape and return output and gradient. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      z_i = array_ops.gather(z, i)
      with backprop.GradientTape() as g:
        g.watch(z_i)
        out = f(z_i)
      return out, g.gradient(out, z_i)

    self._test_loop_fn(loop_fn, 4)

  def test_stateful_with_gradients(self):
    # [FUNCTION DOC]
    # Same as the gradient test but the function reads a captured variable. 4 iterations. The test
    # harness (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with
    # for_loop, evaluates both and asserts the results are close.

    z = random_ops.random_uniform([4, 2])
    v = variables.Variable(z[0])

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # Compiled function: square of x plus the captured variable plus one.
      return math_ops.square(x) + v + 1

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will call f on row i
      # under a tape and return output and gradient. Returns the per-iteration result or tuple of
      # results that pfor stacks along a new leading dimension; no side effects other than creating
      # graph operations.
      z_i = array_ops.gather(z, i)
      with backprop.GradientTape() as g:
        g.watch(z_i)
        out = f(z_i)
      return out, g.gradient(out, z_i)

    self._test_loop_fn(loop_fn, 4)


class SpectralTest(PForTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for pfor conversion of FFT ops (complex, real, inverse real), parameterized by op.

  @parameterized.parameters(
      (fft_ops.fft,),
      (fft_ops.fft2d,),
      (fft_ops.fft3d,),
      (fft_ops.ifft,),
      (fft_ops.ifft2d,),
      (fft_ops.ifft3d,),
  )
  def test_fft(self, op_func):
    # [FUNCTION DOC]
    # Parameterized over the six complex fft and ifft ops. Harness run on random complex input shape
    # 2 by 3 by 4 by 3 by 4, 2 iterations. The test harness (_test_loop_fn) runs the loop body both
    # vectorized with pfor and sequentially with for_loop, evaluates both and asserts the results
    # are close.
    shape = [2, 3, 4, 3, 4]
    x = np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will apply the op to
      # slice i. Returns the per-iteration result or tuple of results that pfor stacks along a new
      # leading dimension; no side effects other than creating graph operations.
      x_i = array_ops.gather(x, i)
      return op_func(x_i)

    self._test_loop_fn(loop_fn, 2)

  @parameterized.parameters(
      (fft_ops.rfft,),
      (fft_ops.rfft2d,),
      (fft_ops.rfft3d,),
  )
  def test_rfft(self, op_func):
    # [FUNCTION DOC]
    # Parameterized over the real FFT ops. For float32 and float64 inputs, harness run, 2
    # iterations. The test harness (_test_loop_fn) runs the loop body both vectorized with pfor and
    # sequentially with for_loop, evaluates both and asserts the results are close.
    for dtype in (dtypes.float32, dtypes.float64):
      x = random_ops.random_uniform([2, 3, 4, 3, 4], dtype=dtype)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
        # tensor; some tests also pass a pfor configuration object). Here it will apply the op to
        # slice i. Returns the per-iteration result or tuple of results that pfor stacks along a new
        # leading dimension; no side effects other than creating graph operations.
        x_i = array_ops.gather(x, i)
        return op_func(x_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)

  @parameterized.parameters(
      (fft_ops.irfft,),
      (fft_ops.irfft2d,),
      (fft_ops.irfft3d,),
  )
  def test_irfft(self, op_func):
    # [FUNCTION DOC]
    # Parameterized over the inverse real FFT ops. Skipped when a GPU is present (flaky). For
    # complex64 and complex128 random inputs, harness run, 2 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.
    if config.list_physical_devices("GPU"):
      # TODO(b/149957923): The test is flaky
      self.skipTest("b/149957923: irfft vectorization flaky")
    for dtype in (dtypes.complex64, dtypes.complex128):
      shape = [2, 3, 4, 3, 4]
      x = np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)
      x = math_ops.cast(x, dtype=dtype)

      # pylint: disable=cell-var-from-loop
      def loop_fn(i):
        # [FUNCTION DOC]
        # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
        # tensor; some tests also pass a pfor configuration object). Here it will apply the op to
        # slice i. Returns the per-iteration result or tuple of results that pfor stacks along a new
        # leading dimension; no side effects other than creating graph operations.
        x_i = array_ops.gather(x, i)
        return op_func(x_i)

      # pylint: enable=cell-var-from-loop

      self._test_loop_fn(loop_fn, 2)


class VariableTest(PForTestCase):
  # [CLASS DOC]
  # Test case for variable creation and variable use inside vectorized_map and pfor, and for
  # indexed_case.

  def test_create_variable_once(self):
    # [FUNCTION DOC]
    # Checks that vectorized_map allows a function to create a variable only on the first call: the
    # variable is stored in a list and reused. No assertions, only that no error is raised.
    x = array_ops.ones(shape=(3, 2, 2), dtype=dtypes.float32)
    y = array_ops.ones(shape=(2, 3), dtype=dtypes.float32)
    a_var = []

    def f(z):
      # [FUNCTION DOC]
      # Mapped function: on the first call creates a variable (initialized from y) and stores it in
      # the enclosing list; on every call multiplies z by the stored variable divided by 16 via
      # matmul. Returns the product. Side effect: appends to the captured list.
      if not a_var:
        a_var.append(variables.Variable(lambda: y, name="a"))
      return math_ops.matmul(z, a_var[0] / 16)

    pfor_control_flow_ops.vectorized_map(f, x)

  @test_util.run_v2_only
  def test_create_variable_repeated(self):
    # [FUNCTION DOC]
    # v2-only. Asserts vectorized_map raises ValueError (singleton variable on first call) when the
    # function creates a new variable every call.
    x = array_ops.ones(shape=(3, 2, 2), dtype=dtypes.float32)
    y = array_ops.ones(shape=(2, 3), dtype=dtypes.float32)

    def f(z):
      # [FUNCTION DOC]
      # Mapped function: creates a new variable on every call, divides it by 4, and matmuls z by it
      # divided by 16. Returns the product. Creating a variable per call is what triggers the
      # expected error.
      a_var = variables.Variable(lambda: y, name="a") / 4
      return math_ops.matmul(z, a_var / 16)

    # Note that this error is only raised under v2 behavior.
    with self.assertRaisesRegex(
        ValueError, "singleton tf.Variable.*on the first call"):
      pfor_control_flow_ops.vectorized_map(f, x)

  @test_util.run_all_in_graph_and_eager_modes
  def test_variable_shape(self):
    # [FUNCTION DOC]
    # Harness run on reading a variable's shape op via its handle. 2 iterations. The test harness
    # (_test_loop_fn) runs the loop body both vectorized with pfor and sequentially with for_loop,
    # evaluates both and asserts the results are close.
    v = resource_variable_ops.ResourceVariable([1, 2])

    def loop_fn(_):
      # [FUNCTION DOC]
      # Per-iteration body for the enclosing test. Takes the iteration index (a scalar integer
      # tensor; some tests also pass a pfor configuration object). Here it will read the variable
      # shape from its handle. Returns the per-iteration result or tuple of results that pfor stacks
      # along a new leading dimension; no side effects other than creating graph operations.
      return resource_variable_ops.variable_shape(v.handle)

    self._test_loop_fn(loop_fn, 2)

  @test_util.run_all_in_graph_and_eager_modes
  def test_variable_input(self):
    # [FUNCTION DOC]
    # Applies vectorized_map to a resource variable argument adding one; asserts the result equals
    # [2, 3].
    v = resource_variable_ops.ResourceVariable([1, 2])
    self.evaluate(v.initializer)

    def loop_fn(x):
      # [FUNCTION DOC]
      # Mapped function: adds one to its argument (an element of the variable) and returns it.
      return x + 1

    result = pfor_control_flow_ops.vectorized_map(loop_fn, v)
    expected_result = [2, 3]
    self.assertAllEqual(result, expected_result)

  @test_util.run_all_in_graph_and_eager_modes
  def testStatelessCase(self):
    # [FUNCTION DOC]
    # Checks vectorized_map over indexed_case with loop-variant branch index and fallback to while
    # loop disabled; asserts expected values.

    def branch1(x):
      # [FUNCTION DOC]
      # Branch function returning its argument unchanged.
      return x

    def branch2(x):
      # [FUNCTION DOC]
      # Branch function returning its argument plus one.
      return x + 1

    def branch3(x):
      # [FUNCTION DOC]
      # Branch function returning its argument plus two.
      return x + 2

    x = constant_op.constant(10)
    elems = constant_op.constant([1, 0, 0, 0, 2, 1, 0, 2, 0, 1])
    def loop_fn(z_i):
      # [FUNCTION DOC]
      # Mapped function: takes a branch index and calls indexed_case choosing among three lambdas
      # that apply the branch functions to the captured constant 10; returns the chosen result.
      return cond_v2.indexed_case(
          z_i, [lambda: branch1(x), lambda: branch2(x), lambda: branch3(x)])

    result = pfor_control_flow_ops.vectorized_map(
        loop_fn, elems, fallback_to_while_loop=False)

    expected_result = [11, 10, 10, 10, 12, 11, 10, 12, 10, 11]
    self.assertAllEqual(result, expected_result)

  @test_util.run_all_in_graph_and_eager_modes
  def testStatelessCaseUnstacked(self):
    # [FUNCTION DOC]
    # Checks vectorized_map over an indexed_case with an invariant branch index; fallback disabled;
    # asserts [2, 3, 3, 2, 3, 2, 3, 2, 2].

    def branch1(x):
      # [FUNCTION DOC]
      # Branch function returning its argument plus one.
      return x + 1

    def branch2(x):
      # [FUNCTION DOC]
      # Branch function returning its argument plus two.
      return x + 2

    # Unstacked case input
    case_input = constant_op.constant(1)
    @def_function.function
    def function(z_i):
      # [FUNCTION DOC]
      # Compiled mapped function: uses a captured (loop-invariant) index of 1 with indexed_case over
      # two branches applied to its argument; always takes the second branch, so returns the
      # argument plus two.
      return cond_v2.indexed_case(case_input,
                                  [lambda: branch1(z_i), lambda: branch2(z_i)])

    inputs = constant_op.constant([0, 1, 1, 0, 1, 0, 1, 0, 0])

    result = pfor_control_flow_ops.vectorized_map(
        function, inputs, fallback_to_while_loop=False)
    expected_result = [2, 3, 3, 2, 3, 2, 3, 2, 2]
    self.assertAllEqual(result, expected_result)

if __name__ == "__main__":
  test.main()
