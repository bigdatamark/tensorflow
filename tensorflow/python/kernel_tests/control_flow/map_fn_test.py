# [MODULE DOC]
# KNOWN ISSUE: the module docstring says it tests the functional ops kernels, but the tests cover the
# map function (map over tensors, sparse, ragged, nested structures) rather than functional ops.
# Tests for the map function in the ops package: simple maps, sparse and ragged inputs, variables,
# scoped variables, gradients, multi-input/output structures, autograph, shapes and error cases. Has a
# module-level scoped reducer helper that is unused here. Importing registers tensor array gradients.
# Run through the standard test main. Not a helper module.
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
# pylint: disable=anomalous-backslash-in-string

"""Tests for tensorflow.kernels.functional_ops."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


# pylint: disable=invalid-name
def simple_scoped_fn(a, x):
  # [FUNCTION DOC]
  # HELPER: reducer defined for scoped tests (unused in this file). Takes accumulator a and element x and
  # inside a variable scope named "body" gets or creates an integer scalar variable "two" initialized to
  # 2, returning (a plus x) times that variable. Side effect: creates or reuses a variable.
  # KNOWN ISSUE: this function is never called in this module; it is a leftover copy from the
  # functional ops tests.
  """Simple function: (a, x) -> 2(x+a), but with "2" as a variable in scope."""
  with variable_scope.variable_scope("body"):
    # Dummy variable, just to check that scoping works as intended.
    two = variable_scope.get_variable(
        "two", [],
        dtype=dtypes.int32,
        initializer=init_ops.constant_initializer(2))
    return math_ops.multiply(math_ops.add(a, x), two)


@test_util.with_control_flow_v2
class MapFnTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the map function (map over the first dimension of tensors, sparse and ragged tensors,
  # and nested structures). Decorated to also run with control flow version 2. Mixes graph-and-eager
  # tests with deprecated version 1 graph-only tests.

  @test_util.run_in_graph_and_eager_modes
  def testMap_Simple(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of mapping a function that adds 3 and doubles over a constant one to six.
    # Expects the equivalent computed list.
    nums = [1, 2, 3, 4, 5, 6]
    elems = constant_op.constant(nums, name="data")
    r = map_fn.map_fn(
        lambda x: math_ops.multiply(math_ops.add(x, 3), 2), elems)
    self.assertAllEqual(
        np.array([(x + 3) * 2 for x in nums]), self.evaluate(r))

  def testMapDtypeEager(self):
    # [FUNCTION DOC]
    # Test in eager mode that mapping over an empty float vector with a function returning an empty string
    # constant and declared output dtype string produces a result with dtype string.
    with context.eager_mode():
      dtype = map_fn.map_fn(lambda x: constant_op.constant(""),
                            constant_op.constant([]),
                            dtype=dtypes.string).dtype
      self.assertEqual(dtype, dtypes.string)

  def testMapSparseTensor(self):
    # [FUNCTION DOC]
    # Test that mapping the identity over a sparse tensor (indices, values, dense shape) returns a sparse
    # result whose indices, values and dense shape all equal the input.
    with self.cached_session():
      st = sparse_tensor.SparseTensor(
          indices=[[0, 0], [0, 1], [1, 0]],
          values=constant_op.constant([0, 1, 2]),
          dense_shape=[2, 2])
      result = map_fn.map_fn(lambda x: x, st)
      self.assertAllEqual(result.indices, st.indices)
      self.assertAllEqual(result.values, st.values)
      self.assertAllEqual(result.dense_shape, st.dense_shape)

  def testMapRaggedTensor(self):
    # [FUNCTION DOC]
    # Test of mapping over a ragged tensor with a function adding one and a ragged output signature.
    # Expects the result 2, 3 and 4 in the original ragged structure and static shape of two rows with
    # unknown row length.
    # Note: there are additional tests in ragged/ragged_map_fn_op_test.py
    with self.cached_session():
      rt = ragged_factory_ops.constant([[1, 2], [3]])
      result = map_fn.map_fn(
          lambda x: x + 1,
          rt,
          fn_output_signature=ragged_tensor.RaggedTensorSpec([None], rt.dtype))
      self.assertAllEqual([[2, 3], [4]], result)
      self.assertEqual([2, None], result.shape.as_list())

  @test_util.run_in_graph_and_eager_modes
  def testMapVariable(self):
    # [FUNCTION DOC]
    # Test (graph and eager) that map works on a resource variable input. Initializes a variable of 1 and
    # 2 and maps a function adding one; expects 2 and 3.
    v = resource_variable_ops.ResourceVariable([1, 2])
    self.evaluate(v.initializer)

    def loop_fn(x):
      # [FUNCTION DOC]
      # HELPER: mapped function for testMapVariable; returns the element plus one.
      return x + 1

    result = map_fn.map_fn(loop_fn, v)
    expected_result = [2, 3]
    self.assertAllEqual(result, expected_result)

  @test_util.run_in_graph_and_eager_modes
  def testMapOverScalarErrors(self):
    # [FUNCTION DOC]
    # Test (graph and eager) that mapping over a scalar raises a value error. Both a list of two numbers
    # and a bare number must raise a value error whose message says input must be a Tensor, not scalar.
    with self.assertRaisesRegex(ValueError, "must be .* Tensor.* not scalar"):
      map_fn.map_fn(lambda x: x, [1, 2])
    with self.assertRaisesRegex(ValueError, "must be .* Tensor.* not scalar"):
      map_fn.map_fn(lambda x: x, 1)

  @test_util.run_deprecated_v1
  def testMap_Scoped(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of mapping a function that creates a scoped variable. Checks
    # exactly one trainable variable root/body/two:0, results double the input, and after enabling variable
    # reuse another map still has one variable and the same results.
    with self.cached_session() as sess:

      def double_scoped(x):
        # [FUNCTION DOC]
        # HELPER: mapped function for testMap_Scoped. Takes x; in a variable scope "body" gets or creates an
        # integer variable "two" initialized to 2 and returns x times it. Side effect: creates or reuses the
        # variable.
        """2x with a dummy 2 that is scoped."""
        with variable_scope.variable_scope("body"):
          # Dummy variable, just to check that scoping works as intended.
          two = variable_scope.get_variable(
              "two", [],
              dtype=dtypes.int32,
              initializer=init_ops.constant_initializer(2))
          return math_ops.multiply(x, two)

      with variable_scope.variable_scope("root") as varscope:
        elems = constant_op.constant([1, 2, 3, 4, 5, 6], name="data")
        doubles = np.array([2 * x for x in [1, 2, 3, 4, 5, 6]])

        r = map_fn.map_fn(double_scoped, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertEqual(variables.trainable_variables()[0].name,
                         "root/body/two:0")
        sess.run([variables.global_variables_initializer()])
        self.assertAllEqual(doubles, self.evaluate(r))

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = map_fn.map_fn(double_scoped, elems)
        self.assertEqual(len(variables.trainable_variables()), 1)
        self.assertAllEqual(doubles, self.evaluate(r))

  @test_util.run_deprecated_v1
  def testMap_Grad(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) of gradients through map. Maps a function multiplying the
    # square of the element by a parameter 2 over one to six. Expects gradient with respect to the parameter
    # 91 and with respect to the elements 4, 8, 12, 16, 20, 24.
    with self.cached_session():
      param = constant_op.constant(2.0)
      elems = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="elems")
      y = map_fn.map_fn(
          lambda x: math_ops.multiply(math_ops.square(x), param), elems)
      r_param = gradients_impl.gradients(y, param)[0]
      r_elems = gradients_impl.gradients(y, elems)[0]
      self.assertAllEqual(91.0, self.evaluate(r_param))
      self.assertAllEqual([4.0, 8.0, 12.0, 16.0, 20.0, 24.0],
                          self.evaluate(r_elems))

  @test_util.run_in_graph_and_eager_modes
  def testMap_SimpleNotTensor(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of mapping the add-3-and-double function over a numpy array (not a tensor).
    # Expects the same values as the tensor version.
    nums = np.array([1, 2, 3, 4, 5, 6])
    r = map_fn.map_fn(
        lambda x: math_ops.multiply(math_ops.add(x, 3), 2), nums)
    self.assertAllEqual(
        np.array([(x + 3) * 2 for x in nums]), self.evaluate(r))

  @test_util.run_in_graph_and_eager_modes
  def testMap_SingleInputMultiOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of a mapped function returning a pair of values with a declared pair of 64-bit
    # integer dtypes. Expects two outputs with static shape of six each, equal to the doubled shifted and its
    # negation.
    nums = np.array([1, 2, 3, 4, 5, 6])
    r = map_fn.map_fn(
        lambda x: ((x + 3) * 2, -(x + 3) * 2),
        nums,
        dtype=(dtypes.int64, dtypes.int64))
    self.assertEqual(2, len(r))
    self.assertEqual((6,), r[0].get_shape())
    self.assertEqual((6,), r[1].get_shape())
    received = self.evaluate(r)
    self.assertAllEqual((nums + 3) * 2, received[0])
    self.assertAllEqual(-(nums + 3) * 2, received[1])

  @test_util.run_in_graph_and_eager_modes
  def testMap_MultiOutputMismatchedDtype(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of error handling: function returns a tuple but the declared dtype is a list.
    # Expects a type error about the two structures not having the same nested structure.
    nums = np.array([1, 2, 3, 4, 5, 6])
    with self.assertRaisesRegex(
        TypeError, r"two structures don't have the same nested structure"):
      # lambda emits tuple, but dtype is a list
      map_fn.map_fn(
          lambda x: ((x + 3) * 2, -(x + 3) * 2),
          nums,
          dtype=[dtypes.int64, dtypes.int64])

  @test_util.run_in_graph_and_eager_modes
  def testMap_MultiInputSingleOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of mapping over a nested input (numbers, (numbers, negated numbers)) with a
    # single 64-bit output computing first times second plus third. Checks static shape six and values.
    nums = np.array([1, 2, 3, 4, 5, 6])
    r = map_fn.map_fn(
        lambda x: x[0] * x[1][0] + x[1][1], (nums, (nums, -nums)),
        dtype=dtypes.int64)
    self.assertEqual((6,), r.get_shape())
    received = self.evaluate(r)
    self.assertAllEqual(nums * nums + (-nums), received)

  @test_util.run_in_graph_and_eager_modes
  def testMap_MultiInputSameStructureOutput(self):
    # [FUNCTION DOC]
    # Test (graph and eager) of mapping a function over a nested input that returns a nested output with the
    # same structure (second inner element, (third element, first)) without declaring dtype. Checks each of
    # three outputs has static shape six and equals twice the numbers, minus the numbers, the numbers.
    nums = np.array([1, 2, 3, 4, 5, 6])
    r = map_fn.map_fn(lambda x: (x[1][0], (x[1][1], x[0])),
                      (nums, (2 * nums, -nums)))
    r = [r[0], r[1][0], r[1][1]]
    self.assertEqual((6,), r[0].get_shape())
    self.assertEqual((6,), r[1].get_shape())
    self.assertEqual((6,), r[2].get_shape())
    received = self.evaluate(r)
    self.assertAllEqual(2 * nums, received[0])
    self.assertAllEqual(-nums, received[1])
    self.assertAllEqual(nums, received[2])

  @test_util.run_in_graph_and_eager_modes
  def testMap_autograph_indirect(self):
    # [FUNCTION DOC]
    # Test (graph and eager) that a mapped function containing Python-style conditionals on a tensor is
    # converted indirectly by autograph when mapped inside a compiled function. Expects the result list
    # containing 1.

    def test_function(x):
      # [FUNCTION DOC]
      # HELPER: mapped function for the autograph test. Builds constant minus one and branches on whether it
      # equals zero; both branches set the result to x, so it always returns x unchanged.
      # KNOWN ISSUE: both branches of the conditional assign the same value, so the conditional has no
      # effect beyond exercising autograph conversion.
      cond = constant_op.constant(-1)
      if cond == 0:
        result = x
      else:
        result = x
      return result

    @def_function.function
    def map_call(x):
      # [FUNCTION DOC]
      # HELPER: compiled function that maps the test function over its input x and returns the mapped result.
      return map_fn.map_fn(test_function, x)

    x = constant_op.constant([1])
    y = map_call(x)
    self.assertAllEqual([1], self.evaluate(y))

  @test_util.run_in_graph_and_eager_modes
  def testMapShape(self):
    # [FUNCTION DOC]
    # Test (graph and eager) that the static shape of a map over a two by three matrix with the identity
    # equals the evaluated shape.
    x = constant_op.constant([[1, 2, 3], [4, 5, 6]])
    y = map_fn.map_fn(lambda e: e, x)
    self.assertAllEqual(y.get_shape(), self.evaluate(y).shape)

  @test_util.run_deprecated_v1
  def testMapUnknownShape(self):
    # [FUNCTION DOC]
    # Test (deprecated version 1 graph mode) that mapping the identity over a placeholder of unknown shape
    # gives a result with fully unknown static shape.
    x = array_ops.placeholder(dtypes.float32)
    y = map_fn.map_fn(lambda e: e, x)
    self.assertIs(None, y.get_shape().dims)

  # TODO(b/124383826): this test fails in eager: the iterable is of length 0 so
  # so the body of the while loop never executes
  @test_util.run_v1_only("b/120545219")
  def testMapEmptyScalar(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the TODO above says this fails in eager mode because the body of the loop never runs for
    # an empty input, so it is restricted to version 1 graph mode.
    # Test of mapping a function returning one over an empty integer vector. Expects static and evaluated
    # result shape of a single zero dimension.
    map_return = map_fn.map_fn(lambda x: 1,
                               constant_op.constant([], dtype=dtypes.int32))
    self.assertAllEqual([0], map_return.get_shape().dims)
    self.assertAllEqual([0], self.evaluate(map_return).shape)

  # TODO(b/124383826): this test fails in eager: the iterable is of length 0 so
  # so the body of the while loop never executes
  @test_util.run_v1_only("b/120545219")
  def testMapEmptyTensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the TODO above says this fails in eager mode because the body of the loop never runs for
    # an empty input, so it is restricted to version 1 graph mode.
    # Test of mapping a function returning a three by two zeros tensor over an empty float vector. Expects
    # static and evaluated shape of zero by three by two.
    with self.cached_session():
      map_return = map_fn.map_fn(lambda x: array_ops.zeros([3, 2]),
                                 constant_op.constant([]))
      self.assertAllEqual([0, 3, 2], map_return.get_shape().dims)
      self.assertAllEqual([0, 3, 2], self.evaluate(map_return).shape)

  @test_util.run_in_graph_and_eager_modes
  def testMapEmptyList(self):
    # [FUNCTION DOC]
    # Test (graph and eager) that mapping over an empty Python list raises a value error saying that the
    # elems must be a Tensor or a nested structure of tensors.
    x = []
    with self.assertRaisesRegex(ValueError, r"elems must be a Tensor or"):
      _ = map_fn.map_fn(lambda e: e, x)


if __name__ == "__main__":
  test.main()

# pylint: enable=invalid-name
