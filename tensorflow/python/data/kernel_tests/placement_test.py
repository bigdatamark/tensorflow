# [MODULE DOC]
# Kernel tests for device placement of tf.data datasets and iterators inside traced functions.
# Dataset tensors (variant tensors) cannot be copied between devices, so these tests check that
# datasets, iterators and related ops are placed or colocated so no cross-device copy is needed:
# loops and conditionals over captured or locally created datasets, explicit device scopes,
# colocation, identity ops, nested function calls, creating an iterator inside a function on a
# GPU, and iterators fed by prefetch-to-device in eager and graph modes. Several tests are
# GPU-only. The test fixture disables the graph meta-optimizer so placement problems are not
# masked by rewrites. No import-time side effects other than the test entry point.
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
"""Tests for tf.data placement within tf.functions."""
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class PlacementTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case verifying that dataset ops, iterators and surrounding control flow are placed on
  # devices so that no dataset tensor has to be copied across devices. Fixture disables the
  # graph meta-optimizer globally (a process-wide experimental option) in setUp.
  """Tests for tf.data placement within tf.functions.

  Specifically, tf.data dataset tensors cannot be copied between devices. These
  tests verify the ops are placed in a way that avoids this.
  """

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the parent setUp, then turns off the graph meta-optimizer through the
    # global experimental optimizer options so that rewrites cannot hide cross-device copies.
    # Side effect: changes process-wide optimizer configuration. Takes and returns nothing.
    super(PlacementTest, self).setUp()
    # Grappler optimizations can affect whether the placement issues occur,
    # since they may inadvertently rewrite nodes and edges in a way that removes
    # cross-device copies.
    config.set_optimizer_experimental_options({"disable_meta_optimizer": True})

  @combinations.generate(test_base.eager_only_combinations())
  def testWhileWithCapturedDataset(self):
    # [FUNCTION DOC]
    # Eager-only scenario: a dataset created outside a traced function is captured and iterated
    # inside a loop within the function. Expected: the function returns the sum 45.
    dataset = dataset_ops.Dataset.range(10)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function; sums all elements of the captured dataset inside a one-iteration
      # loop and returns the 64-bit total.
      total = constant_op.constant(0, dtypes.int64)
      for _ in math_ops.range(1):
        for elem in dataset:
          total += elem
      return total

    self.assertEqual(f().numpy(), 45)

  @combinations.generate(test_base.eager_only_combinations())
  def testWhile(self):
    # [FUNCTION DOC]
    # Eager-only scenario: a dataset created inside a traced function is iterated inside a loop
    # in that function. Expected: the function returns the sum 45.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function; creates a range dataset, sums its elements inside a
      # one-iteration loop and returns the 64-bit total.
      dataset = dataset_ops.Dataset.range(10)
      total = constant_op.constant(0, dtypes.int64)
      for _ in math_ops.range(1):
        for elem in dataset:
          total += elem
      return total

    self.assertEqual(f().numpy(), 45)

  @combinations.generate(test_base.eager_only_combinations())
  def testCondWithPlacement(self):
    # [FUNCTION DOC]
    # Eager-only scenario: inside a traced function a conditional op whose branches both map a
    # dataset is explicitly placed on the CPU device and the first element is read. Expected: no
    # cross-device copy error and the first element is 1.
    # When the cond op is explicitly placed, there shouldn't be cross-device
    # copies.
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function; builds a range dataset, evaluates a conditional choosing
      # between two identical map branches under a CPU device scope, and returns the first
      # element of the resulting dataset.
      dataset = dataset_ops.Dataset.range(10)

      def fn():
        # [FUNCTION DOC]
        # HELPER: conditional branch; returns the captured dataset with one added to every
        # element.
        return dataset.map(lambda x: x+1)

      c = constant_op.constant(2)
      with ops.device("/cpu:0"):
        a = cond.cond(math_ops.equal(c, 2), fn, fn)
        iterator = iter(a)
        nxt = next(iterator)
      return nxt

    self.assertEqual(f().numpy(), 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testCondWithColocation(self):
    # [FUNCTION DOC]
    # Eager-only scenario: same as the explicit-placement test, but the conditional op is
    # colocated with the dataset's variant tensor instead. Expected: no cross-device copy error
    # and the first element is 1.
    # When the cond op is colocated with the dataset, there shouldn't be
    # cross-device copies.
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function; builds a range dataset, evaluates a conditional choosing
      # between two identical map branches colocated with the dataset tensor, and returns the
      # first element of the result.
      dataset = dataset_ops.Dataset.range(8)

      def fn():
        # [FUNCTION DOC]
        # HELPER: conditional branch; returns the captured dataset with one added to every
        # element.
        return dataset.map(lambda x: x+1)

      c = constant_op.constant(2)
      with ops.colocate_with(dataset._variant_tensor):  # pylint:disable=protected-access
        a = cond.cond(math_ops.equal(c, 2), fn, fn)
        iterator = iter(a)
        nxt = next(iterator)
      return nxt

    self.assertEqual(f().numpy(), 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testCond(self):
    # [FUNCTION DOC]
    # Eager-only scenario: no explicit placement; a conditional with two different map branches
    # over a dataset is built in a traced function and its first element read. Expected: the
    # result is 1 (the true branch adds one) with no placement error.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function; creates a range dataset, picks between an add-one map and an
      # add-two map via a conditional on a constant, and returns the first element.
      dataset = dataset_ops.Dataset.range(8)
      c = constant_op.constant(2)
      a = cond.cond(
          math_ops.equal(c, 2),
          lambda: dataset.map(lambda x: x + 1),
          lambda: dataset.map(lambda x: x + 2),
      )
      return next(iter(a))

    self.assertEqual(f().numpy(), 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testId(self):
    # [FUNCTION DOC]
    # Eager-only scenario: an identity op applied to a dataset inside a traced function.
    # Expected: the call completes without a placement error (ideally the placer keeps the
    # identity on the dataset's device).
    # Ideally, placer should know that Identity(dataset) should be on the same
    # device as the dataset.
    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced function; creates a range dataset, passes it through an identity op and
      # returns it.
      dataset = dataset_ops.Dataset.range(10)
      dataset = array_ops.identity(dataset)
      return dataset
    f()

  @combinations.generate(test_base.default_test_combinations())
  @test_util.run_gpu_only
  def testFunctionCall(self):
    # [FUNCTION DOC]
    # GPU-only scenario: an inner traced function that reduces a dataset argument (could be
    # placed on GPU) is called from an outer function that creates the dataset (which cannot be
    # on GPU). Expected: the outer function evaluates to 10 with no placement error.
    # Ideally, placer should know that Call(dataset) should be on the same
    # device as the dataset. Create a function that could be place don the GPU,
    # but a Dataset that cannot.
    @def_function.function
    def test_call(dataset):
      # [FUNCTION DOC]
      # HELPER: inner traced function; counts the elements of the given dataset by reduction and
      # returns the count.
      return dataset.reduce(0, lambda s, _: s + 1)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: outer traced function; creates a ten-element range dataset and returns the
      # result of the inner function applied to it.
      dataset = dataset_ops.Dataset.range(10)
      return test_call(dataset)

    self.assertEqual(self.evaluate(f()), 10)

  @combinations.generate(test_base.eager_only_combinations())
  @test_util.run_gpu_only
  def testIteratorOnDeviceEagerMode(self):
    # [FUNCTION DOC]
    # GPU-only eager scenario: a dataset prefetched to the first GPU is iterated, taking one
    # element and one optional. Expected: the dataset tensor, iterator resource, element and the
    # optional's value and has-value tensors all report a device name containing the first GPU.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(prefetching_ops.prefetch_to_device("/gpu:0"))
    iterator = iter(dataset)
    data = next(iterator)
    optional_data = iterator.get_next_as_optional()

    self.assertIn("gpu:0", dataset._variant_tensor.device.lower())
    self.assertIn("gpu:0", iterator._iterator_resource.device.lower())
    self.assertIn("gpu:0", data.device.lower())
    self.assertIn("gpu:0", optional_data.get_value().device.lower())
    self.assertIn("gpu:0", optional_data.has_value().device.lower())

  # There are HostMemory constraints on AnonymousIteratorV2 and
  # DeleteIterator kernels on TPU but not on GPU. This is intentional because
  # when running AnonymousIteratorV2 in a function
  #
  # - If the op is placed on GPU, the variant _Retval is placed on GPU.
  # - However, if the op is placed on TPU, the variant _Retval is placed on
  #   CPU.
  #
  # So if were to add HostMemory constraints to the GPU kernels it would lead
  # to variant device copy errors.
  #
  # TODO(b/204231062): Unify behavior across GPU and TPU.
  @combinations.generate(test_base.eager_only_combinations())
  @test_util.run_gpu_only
  def testCreateIteratorInFuncOnGpu(self):
    # [FUNCTION DOC]
    # GPU-only eager scenario: a traced function creates an anonymous iterator resource (version
    # two) with a scalar float output. Expected: the call succeeds without variant device copy
    # errors when a GPU is present.

    @def_function.function
    def create_iter():
      # [FUNCTION DOC]
      # HELPER: traced function; creates and returns an anonymous iterator resource for scalar
      # 32-bit float elements.
      return gen_dataset_ops.anonymous_iterator_v2(
          output_types=[dtypes.float32], output_shapes=[[]])

    create_iter()

  @combinations.generate(test_base.graph_only_combinations())
  @test_util.run_gpu_only
  def testIteratorOnDeviceGraphModeOneShotIterator(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test skips itself immediately (one-shot iterators do not support GPU
    # placement, tracked by a bug reference), so all code after the skip is dead.
    # GPU-only graph scenario: dataset prefetched to the GPU, read via a one-shot iterator; the
    # intended assertion is that colocating a device-query op with the dataset tensor, iterator
    # resource, element, optional value and optional has-value gives the first GPU.
    self.skipTest("TODO(b/169429285): tf.data.Dataset.make_one_shot_iterator "
                  "does not support GPU placement.")

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(prefetching_ops.prefetch_to_device("/gpu:0"))
    iterator = dataset_ops.make_one_shot_iterator(dataset)
    data = iterator.get_next()
    optional_data = iterator.get_next_as_optional()

    with ops.colocate_with(dataset._variant_tensor):
      dataset_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(dataset_device))

    with ops.colocate_with(iterator._iterator_resource):
      iterator_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(iterator_device))

    with ops.colocate_with(data):
      data_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(data_device))

    with ops.colocate_with(optional_data.get_value()):
      get_value_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(get_value_device))

    with ops.colocate_with(optional_data.has_value()):
      has_value_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(has_value_device))

  @combinations.generate(test_base.graph_only_combinations())
  @test_util.run_gpu_only
  def testIteratorOnDeviceGraphModeInitializableIterator(self):
    # [FUNCTION DOC]
    # GPU-only graph scenario: a dataset prefetched to the first GPU is read via an
    # initializable iterator. Expected: a device-placement op colocated with the dataset tensor,
    # the iterator resource, the element, the optional's value and its has-value tensor each
    # reports the first GPU.
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(prefetching_ops.prefetch_to_device("/gpu:0"))
    iterator = dataset_ops.make_initializable_iterator(dataset)
    data = iterator.get_next()
    optional_data = iterator.get_next_as_optional()

    with ops.colocate_with(dataset._variant_tensor):
      dataset_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(dataset_device))

    with ops.colocate_with(iterator._iterator_resource):
      iterator_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(iterator_device))

    with ops.colocate_with(data):
      data_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(data_device))

    with ops.colocate_with(optional_data.get_value()):
      get_value_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(get_value_device))

    with ops.colocate_with(optional_data.has_value()):
      has_value_device = test_ops.device_placement_op()
    self.assertIn(b"GPU:0", self.evaluate(has_value_device))

  @combinations.generate(test_base.eager_only_combinations())
  @test_util.run_gpu_only
  def testIterDatasetEagerModeWithExplicitDevice(self):
    # [FUNCTION DOC]
    # GPU-only eager scenario: a traced function that iterates a dataset created inside it is
    # called under an explicit first-GPU device scope. Expected: the sum is 45.

    @def_function.function
    def comp():
      # [FUNCTION DOC]
      # HELPER: traced function; iterates a ten-element range dataset created inside it and
      # returns the 64-bit sum.
      value = constant_op.constant(0, dtype=dtypes.int64)
      for d in iter(dataset_ops.Dataset.range(10)):
        value += d
      return value

    with ops.device("/gpu:0"):
      result = comp()
    self.assertEqual(result.numpy(), 45)

  @combinations.generate(test_base.eager_only_combinations())
  @test_util.run_gpu_only
  def testFunctionInliningColocation(self):
    # [FUNCTION DOC]
    # GPU-only eager scenario: an outer function creates a dataset and passes it to an inner
    # function that creates an iterator and reads the first element, executed under a GPU device
    # scope. Expected: the result is 0, i.e. inlining keeps iterator ops colocated.

    @def_function.function
    def f(ds):
      # [FUNCTION DOC]
      # HELPER: inner traced function; creates an iterator over the given dataset and returns
      # its next element.
      return next(iter(ds))

    @def_function.function
    def g():
      # [FUNCTION DOC]
      # HELPER: outer traced function; creates a ten-element range dataset and returns the result
      # of the inner function applied to it.
      dataset = dataset_ops.Dataset.range(10)
      return f(dataset)

    with ops.device("/gpu:0"):
      self.assertEqual(self.evaluate(g()), 0)


if __name__ == "__main__":
  test.main()
