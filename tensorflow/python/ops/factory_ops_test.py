# [MODULE DOC]
# Tests that sparse tensors (integer and string valued) work with a distributed dataset across
# two GPUs. String sparse tensors cannot live on GPU, so they must be placed on CPU. Defines
# two small sparse tensor factories and one parameterized GPU-only test case.
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
"""Tests that sparse tensors work with GPU, such as placement of int and string.

Test using sparse tensors with distributed dataset. Since GPU does
not support strings, sparse tensors containing string should always be placed
on CPU.
"""

from absl.testing import parameterized
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def sparse_int64():
  # [FUNCTION DOC]
  # HELPER: test data factory.
  # Returns a sparse tensor with dense shape 8 by 4 with eight entries on a diagonal pattern
  # holding the 64-bit integers one through eight.
  return sparse_tensor.SparseTensor(
      indices=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 0], [5, 1], [6, 2], [7, 3]],
      values=constant_op.constant([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtypes.int64),
      dense_shape=[8, 4])


def sparse_str():
  # [FUNCTION DOC]
  # HELPER: test data factory.
  # Returns a sparse tensor with the same indices and dense shape (8 by 4) as the integer
  # factory but with string values "1" through "8".
  return sparse_tensor.SparseTensor(
      indices=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 0], [5, 1], [6, 2], [7, 3]],
      values=constant_op.constant(['1', '2', '3', '4', '5', '6', '7', '8']),
      dense_shape=[8, 4])


class FactoryOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case checking that sparse tensor factories work with distributed
  # datasets on a mirrored strategy over two GPUs. Holds no state.

  @parameterized.parameters(
      (sparse_int64,),
      (sparse_str,),
  )
  @test_util.run_gpu_only
  def testSparseWithDistributedDataset(self, sparse_factory):
    # [FUNCTION DOC]
    # Parameterized by a sparse tensor factory (integer or string valued); runs only on GPU.
    # Creates the sparse tensor, then runs a compiled function that distributes a batched
    # dataset of its slices over a two-GPU mirrored strategy, reads the first local result and
    # drains the iterator. Asserts the dense form of the first row of the result equals that of
    # the original tensor. Returns nothing.

    @def_function.function
    def distributed_dataset_producer(t):
      # [FUNCTION DOC]
      # Compiled helper: given a sparse tensor t, builds a mirrored strategy on GPU 0 and 1,
      # batches slices of t by two, distributes the dataset, takes the first batch's local
      # result, iterates through the rest of the iterator, and returns that first result.
      strategy = mirrored_strategy.MirroredStrategy(['GPU:0', 'GPU:1'])
      sparse_ds = dataset_ops.Dataset.from_tensor_slices(t).batch(2)
      dist_dataset = strategy.experimental_distribute_dataset(sparse_ds)
      ds = iter(dist_dataset)
      result = strategy.experimental_local_results(next(ds))[0]
      # Reach the end of the iterator
      for ignore in ds:  # pylint: disable=unused-variable
        pass
      return result

    t = sparse_factory()

    result = distributed_dataset_producer(t)
    self.assertAllEqual(
        self.evaluate(sparse_ops.sparse_tensor_to_dense(t)[0]),
        self.evaluate(sparse_ops.sparse_tensor_to_dense(result)[0]))


if __name__ == '__main__':
  test.main()
