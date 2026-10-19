# [MODULE DOC]
# Tests for the dataset scan transformation, which threads a state through a dataset with a
# user function returning a new state and an output element. Covers counting and Fibonacci
# sequences, sparse and tensor-array states, state shape and type validation, cardinality
# preservation, default-device placement, the name argument and checkpointing. Test classes only.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.Dataset.scan()`."""
import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import checkpoint_test_base
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import scan_op
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class ScanTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for scan: sequence generation, nested and sparse states, validation errors, device
  # placement and the name argument.

  def _counting_dataset(self, start, scan_fn):
    # [FUNCTION DOC]
    # HELPER: builds an endless dataset driven by a scan function.
    # Repeats a constant element forever and applies scan with the given initial state (start) and
    # scan function (scan_fn). Returns the resulting dataset.
    return dataset_ops.Dataset.from_tensors(0).repeat().scan(
        initial_state=start, scan_func=scan_fn)

  @combinations.generate(test_base.default_test_combinations())
  def testCount(self):
    # [FUNCTION DOC]
    # For several start, step and take values (including zero take and negative steps), builds a
    # scan that emits the state and adds the step. Asserts the first take outputs form the
    # arithmetic sequence and that reading past them raises an out-of-range error.

    def make_scan_fn(step):
      # [FUNCTION DOC]
      # HELPER: returns a scan function that outputs the current state and advances it by step.
      return lambda state, _: (state + step, state)

    def dataset_fn(start, step, take):
      # [FUNCTION DOC]
      # HELPER: returns the counting dataset for the given start state and step, limited to take
      # elements.
      return self._counting_dataset(start, make_scan_fn(step)).take(take)

    for start_val, step_val, take_val in [(0, 1, 10), (0, 1, 0), (10, 1, 10),
                                          (10, 2, 10), (10, -1, 10),
                                          (10, -2, 10)]:
      next_element = self.getNext(dataset_fn(start_val, step_val, take_val))
      for expected, _ in zip(
          itertools.count(start_val, step_val), range(take_val)):
        self.assertEqual(expected, self.evaluate(next_element()))
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testFibonacci(self):
    # [FUNCTION DOC]
    # Scans with a two-number state advancing like Fibonacci numbers and outputting the second
    # number. Asserts the first six outputs are 1, 1, 2, 3, 5, 8.
    data = dataset_ops.Dataset.from_tensors(1).repeat(None).scan(
        initial_state=[0, 1],
        scan_func=lambda a, _: ([a[1], a[0] + a[1]], a[1]))
    next_element = self.getNext(data)

    self.assertEqual(1, self.evaluate(next_element()))
    self.assertEqual(1, self.evaluate(next_element()))
    self.assertEqual(2, self.evaluate(next_element()))
    self.assertEqual(3, self.evaluate(next_element()))
    self.assertEqual(5, self.evaluate(next_element()))
    self.assertEqual(8, self.evaluate(next_element()))

  @combinations.generate(test_base.default_test_combinations())
  def testSparseCount(self):
    # [FUNCTION DOC]
    # Same as the counting test but the scan state is a sparse tensor holding the running value.
    # Asserts the first take outputs' single values follow the arithmetic sequence and that
    # reading further raises an out-of-range error.

    def _sparse(i):
      # [FUNCTION DOC]
      # HELPER: builds a one-by-one sparse tensor value with a single entry whose value is i.
      return sparse_tensor.SparseTensorValue(
          indices=np.array([[0, 0]]),
          values=(i * np.array([1])),
          dense_shape=np.array([1, 1]))

    def make_scan_fn(step):
      # [FUNCTION DOC]
      # HELPER: returns a scan function that outputs the sparse state and advances its value by
      # step.
      return lambda state, _: (_sparse(state.values[0] + step), state)

    def dataset_fn(start, step, take):
      # [FUNCTION DOC]
      # HELPER: returns the counting dataset with a sparse initial state, limited to take elements.
      return self._counting_dataset(_sparse(start),
                                    make_scan_fn(step)).take(take)

    for start_val, step_val, take_val in [(0, 1, 10), (0, 1, 0), (10, 1, 10),
                                          (10, 2, 10), (10, -1, 10),
                                          (10, -2, 10)]:
      next_element = self.getNext(dataset_fn(start_val, step_val, take_val))
      for expected, _ in zip(
          itertools.count(start_val, step_val), range(take_val)):
        self.assertEqual(expected, self.evaluate(next_element()).values[0])
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testTensorArraySimple(self):
    # [FUNCTION DOC]
    # Scans over a range of five with a growing tensor-array state that starts holding minus one.
    # Asserts outputs are the stacked arrays [-1], [-1,0], ... and that a second iteration pass
    # restarts from the initial state.

    def scan_fn(ta, x):
      # [FUNCTION DOC]
      # HELPER: appends x to the tensor array and returns the new array as state with its stacked
      # contents as the output.
      return (ta.write(ta.size(), x), ta.stack())

    start = tensor_array_ops.TensorArray(
        size=0, element_shape=[], dtype=dtypes.int64, dynamic_size=True)
    start = start.write(0, -1)

    ds = dataset_ops.Dataset.range(5).scan(
        initial_state=start, scan_func=scan_fn)

    self.assertDatasetProduces(
        ds,
        expected_output=[
            [-1],
            [-1, 0],
            [-1, 0, 1],
            [-1, 0, 1, 2],
            [-1, 0, 1, 2, 3],
        ],
        requires_initialization=True,
        num_test_iterations=2)

  @combinations.generate(test_base.default_test_combinations())
  def testTensorArrayWithCondReset(self):
    # [FUNCTION DOC]
    # Scans over a range of six with a tensor-array state that is reset to a fresh empty array
    # (built inside the function) whenever the element is divisible by three. Asserts the stacked
    # outputs match the expected sequence in two iteration passes.

    def empty():
      # [FUNCTION DOC]
      # HELPER: returns a new empty growable tensor array of 64-bit integers.
      return tensor_array_ops.TensorArray(
          size=0, element_shape=[], dtype=dtypes.int64, dynamic_size=True)

    def scan_fn(ta, x):
      # [FUNCTION DOC]
      # HELPER: appends x to the array; the next state is a fresh empty array when x is divisible
      # by three, else the updated array. Output is the stacked updated array.
      updated = ta.write(ta.size(), x)
      next_iter = cond.cond(
          math_ops.equal(x % 3, 0), empty, lambda: updated)
      return (next_iter, updated.stack())

    start = empty()
    start = start.write(0, -1)

    ds = dataset_ops.Dataset.range(6).scan(
        initial_state=start, scan_func=scan_fn)

    self.assertDatasetProduces(
        ds,
        expected_output=[
            [-1, 0],
            [1],
            [1, 2],
            [1, 2, 3],
            [4],
            [4, 5],
        ],
        requires_initialization=True,
        num_test_iterations=2)

  @combinations.generate(test_base.default_test_combinations())
  def testTensorArrayWithCondResetByExternalCaptureBreaks(self):
    # [FUNCTION DOC]
    # Skipped under control flow version 2. Like the reset test but the scan function resets to
    # an empty array captured from outside the function. Asserts building the scan raises a
    # not-implemented error telling the user to construct a new array inside the function.

    if control_flow_v2_toggles.control_flow_v2_enabled():
      self.skipTest("v1 only test")

    empty_ta = tensor_array_ops.TensorArray(
        size=0, element_shape=[], dtype=dtypes.int64, dynamic_size=True)

    def scan_fn(ta, x):
      # [FUNCTION DOC]
      # HELPER: appends x to the array and, when x is divisible by three, resets the state to the
      # externally captured empty array (the unsupported pattern); output is the stacked array.
      updated = ta.write(ta.size(), x)
      # Here, capture empty_ta from outside the function.  However, it may be
      # either a TF1-style TensorArray or an Eager-style TensorArray.
      next_iter = cond.cond(
          math_ops.equal(x % 3, 0), lambda: empty_ta, lambda: updated)
      return (next_iter, updated.stack())

    start = empty_ta
    start = start.write(0, -1)

    with self.assertRaisesRegex(
        NotImplementedError,
        r"construct a new TensorArray inside the function"):
      dataset_ops.Dataset.range(6).scan(initial_state=start, scan_func=scan_fn)

  @combinations.generate(test_base.default_test_combinations())
  def testChangingStateShape(self):
    # [FUNCTION DOC]
    # Uses a scan function that doubles a vector state's length and increases a scalar state's
    # rank each step, testing the shape fixed-point inference. Asserts the static shapes
    # (unknown-length vector, unknown rank, scalar output) and that over five steps the vector
    # has length two to the power i and the second state has i dimensions, then end of data.
    # Test the fixed-point shape invariant calculations: start with
    # initial values with known shapes, and use a scan function that
    # changes the size of the state on each element.
    def _scan_fn(state, input_value):
      # [FUNCTION DOC]
      # HELPER: state is a pair of a vector and a scalar. Returns a new state (vector concatenated
      # with itself, second part with a new leading dimension) and an output of the old state and
      # the input value.
      # Statically known rank, but dynamic length.
      ret_longer_vector = array_ops.concat([state[0], state[0]], 0)
      # Statically unknown rank.
      ret_larger_rank = array_ops.expand_dims(state[1], 0)
      return (ret_longer_vector, ret_larger_rank), (state, input_value)

    dataset = dataset_ops.Dataset.from_tensors(0).repeat(5).scan(
        initial_state=([0], 1), scan_func=_scan_fn)
    self.assertEqual(
        [None],
        dataset_ops.get_legacy_output_shapes(dataset)[0][0].as_list())
    self.assertIs(None,
                  dataset_ops.get_legacy_output_shapes(dataset)[0][1].ndims)
    self.assertEqual([],
                     dataset_ops.get_legacy_output_shapes(dataset)[1].as_list())

    next_element = self.getNext(dataset)

    for i in range(5):
      (longer_vector_val, larger_rank_val), _ = self.evaluate(next_element())
      self.assertAllEqual([0] * (2**i), longer_vector_val)
      self.assertAllEqual(np.array(1, ndmin=i), larger_rank_val)
    with self.assertRaises(errors.OutOfRangeError):
      self.evaluate(next_element())

  @combinations.generate(test_base.default_test_combinations())
  def testIncorrectStateType(self):
    # [FUNCTION DOC]
    # Starts with a 32-bit integer state while the scan function returns a 64-bit integer state.
    # Asserts a type error saying the new state's element types must match the initial state.

    def _scan_fn(state, _):
      # [FUNCTION DOC]
      # HELPER: returns a 64-bit integer constant as the new state and the old state as output.
      return constant_op.constant(1, dtype=dtypes.int64), state

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        TypeError,
        "The element types for the new state must match the initial state."):
      dataset.scan(
          initial_state=constant_op.constant(1, dtype=dtypes.int32),
          scan_func=_scan_fn)

  @combinations.generate(test_base.default_test_combinations())
  def testIncorrectReturnType(self):
    # [FUNCTION DOC]
    # The scan function returns a single value instead of a (state, output) pair. Asserts a type
    # error saying it should return a pair of new state and output value.

    def _scan_fn(unused_state, unused_input_value):
      # [FUNCTION DOC]
      # HELPER: ignores its inputs and wrongly returns a lone 64-bit integer constant.
      return constant_op.constant(1, dtype=dtypes.int64)

    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        TypeError,
        "`scan_func` should return a pair consisting of new state and the "
        "output value."):
      dataset.scan(
          initial_state=constant_op.constant(1, dtype=dtypes.int32),
          scan_func=_scan_fn)

  @combinations.generate(test_base.default_test_combinations())
  def testPreserveCardinality(self):
    # [FUNCTION DOC]
    # Scans a one-element dataset with a function calling a python function that raises a stop
    # iteration signal. Asserts fetching the element raises an invalid-argument error (scan keeps
    # cardinality and does not turn the signal into early end of data).

    def scan_fn(state, val):
      # [FUNCTION DOC]
      # HELPER: returns the state unchanged and, as output, the result of a wrapped python function
      # that raises a stop-iteration signal.

      def py_fn(_):
        # [FUNCTION DOC]
        # HELPER: ignores its input and always raises a stop-iteration signal.
        raise StopIteration()

      return state, script_ops.py_func(py_fn, [val], dtypes.int64)

    dataset = dataset_ops.Dataset.from_tensors(0).scan(
        initial_state=constant_op.constant(1), scan_func=scan_fn)
    get_next = self.getNext(dataset)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(get_next())

  @combinations.generate(
      combinations.combine(
          tf_api_version=2, mode="eager", use_default_device=[True, False]))
  def testUseDefaultDevice(self, use_default_device):
    # [FUNCTION DOC]
    # Eager, version 2 only; skipped without a GPU. Builds the scan dataset directly with the
    # parameterized use-default-device flag over a scan function that does a large matrix product
    # on GPU-resident variables and reports the device its colocated op ran on. Asserts the
    # reported device is the CPU when the flag is true and the GPU otherwise.
    if not test_util.is_gpu_available():
      self.skipTest("No GPUs available.")

    weights = variables.Variable(initial_value=array_ops.zeros((1000, 1000)))
    result = variables.Variable(initial_value=array_ops.zeros((1000, 1000)))

    def scan_fn(state, sample):
      # [FUNCTION DOC]
      # HELPER: multiplies the sample by the weights, accumulates into the result variable, and
      # returns the unchanged state with the device-placement string of an op colocated with the
      # product.
      product = math_ops.matmul(sample, weights)
      result.assign_add(product)
      with ops.colocate_with(product):
        device = test_ops.device_placement_op()
      return state, device

    data = variables.Variable(initial_value=array_ops.zeros((1, 1000, 1000)))
    dataset = dataset_ops.Dataset.from_tensor_slices(data)
    dataset = scan_op._ScanDataset(
        dataset, np.int64(1), scan_fn, use_default_device=use_default_device)
    get_next = self.getNext(dataset)

    if use_default_device:
      self.assertIn(b"CPU:0", self.evaluate(get_next()))
    else:
      self.assertIn(b"GPU:0", self.evaluate(get_next()))

  @combinations.generate(test_base.default_test_combinations())
  def testName(self):
    # [FUNCTION DOC]
    # Scans a single-element dataset with a custom name and a function outputting the element;
    # asserts the output is forty-two.
    dataset = dataset_ops.Dataset.from_tensors(42).scan(
        0, lambda x, y: (y, y), name="scan")
    self.assertDatasetProduces(dataset, [42])


class ScanCheckpointTest(checkpoint_test_base.CheckpointTestBase,
                         parameterized.TestCase):
  # [CLASS DOC]
  # Checkpoint save/restore tests for the scan dataset using the shared checkpoint base.

  def _build_dataset(self, num_elements, symbolic_checkpoint):
    # [FUNCTION DOC]
    # HELPER: builds the dataset under test for the checkpoint test.
    # Repeats a constant num_elements times, scans with a Fibonacci-style two-number state, and
    # sets the experimental symbolic checkpoint option from symbolic_checkpoint (boolean). Returns
    # the dataset with the options applied.
    dataset = dataset_ops.Dataset.from_tensors(1).repeat(num_elements)
    dataset = dataset.scan(
        initial_state=[0, 1],
        scan_func=lambda a, _: ([a[1], a[0] + a[1]], a[1]))
    options = options_lib.Options()
    options.experimental_symbolic_checkpoint = symbolic_checkpoint
    return dataset.with_options(options)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          checkpoint_test_base.default_test_combinations(),
          combinations.combine(symbolic_checkpoint=[False, True])))
  def test(self, verify_fn, symbolic_checkpoint):
    # [FUNCTION DOC]
    # Runs the verification callback on the scan dataset built with ten elements and the
    # parameterized symbolic-checkpoint option, expecting ten outputs and correct checkpoint
    # round-tripping.
    num_outputs = 10
    verify_fn(
        self,
        lambda: self._build_dataset(num_outputs, symbolic_checkpoint),
        num_outputs=num_outputs)


if __name__ == "__main__":
  test.main()
