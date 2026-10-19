# [MODULE DOC]
# Tests for the DTensor input pipeline utilities: the dataset wrapper that shards batches of a
# regular dataset across a mesh according to layouts, its iterators (eager and compiled use),
# prefetching, validation errors, a shard-count helper, and the iterator type-spec conversion.
# Defines three test classes plus module constants for mesh dimension names and sizes (4x2x2
# CPU-only mesh in the main class) and aliases for layout types. Uses private members of the
# input utility module.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for DTensor input pipeline utilities."""

import contextlib
import threading

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import mesh_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as tf_test

MESH_DIM_BATCH = 'batch'
MESH_DIM_HEIGHT = 'height'
MESH_DIM_WIDTH = 'width'
MESH_SIZE_BATCH = 4
MESH_SIZE_HEIGHT = 2
MESH_SIZE_WIDTH = 2

Layout = layout_lib.Layout
Mesh = layout_lib.Mesh
UNSHARDED = layout_lib.UNSHARDED


class DTensorDatasetTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case for the distributed dataset wrapper and its iterators. Fixtures: a 4x2x2 mesh
  # (batch, height, width; 16 CPU devices) stored as mesh, a random 8x8x3 image tensor and a
  # random single-element label tensor. Tests compare DTensor iteration output against ordinary
  # dataset iteration and check input validation errors.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Creates a 16-device CPU mesh with batch (4), height (2) and width (2)
    # dimensions, lets the base class configure it as the test mesh, and stores sample images
    # (8x8x3) and labels (shape [1]) as attributes. Returns nothing.
    super().setUp()
    self._num_devices = MESH_SIZE_BATCH * MESH_SIZE_HEIGHT * MESH_SIZE_WIDTH
    self.mesh = mesh_util.create_mesh(
        devices=['CPU:%d' % i for i in range(self._num_devices)],
        mesh_dims=[(MESH_DIM_BATCH, MESH_SIZE_BATCH),
                   (MESH_DIM_HEIGHT, MESH_SIZE_HEIGHT),
                   (MESH_DIM_WIDTH, MESH_SIZE_WIDTH)])

    self.mesh = self.configTestMesh({'CPU': self.mesh})

    self.images = self._images([8, 8, 3])
    self.labels = self._labels([1])

  def _images(self, shape):
    # [FUNCTION DOC]
    # HELPER: produces deterministic sample image data. Takes a shape and returns a tensor of
    # that shape with values from a stateless uniform distribution between 0 and 255 (fixed seed).
    return stateless_random_ops.stateless_random_uniform(
        shape, seed=(1, 2), minval=0, maxval=255)

  def _labels(self, shape):
    # [FUNCTION DOC]
    # HELPER: produces deterministic sample label data. Takes a shape and returns a float32
    # tensor of that shape with values from a stateless uniform distribution between 0 and 10
    # (fixed seed).
    return stateless_random_ops.stateless_random_uniform(
        shape, seed=(1, 2), minval=0, maxval=10, dtype=dtypes.float32)

  def testIterableFailsWithUnknownShapeDatasetSpec(self):
    # [FUNCTION DOC]
    # Builds a generator-based dataset whose element spec has unknown rank and tries to wrap it
    # in the DTensor dataset with a replicated layout. Expects a value error saying the element
    # shape must have a valid rank.
    def gen():
      # [FUNCTION DOC]
      # HELPER: generator for the test dataset; yields a single int32 vector of two elements.
      yield constant_op.constant([1, 2], dtype=dtypes.int32)

    dataset = dataset_ops.DatasetV2.from_generator(
        gen,
        output_signature=tensor_spec.TensorSpec(
            tensor_shape.TensorShape(None), dtype=dtypes.int32))

    with self.assertRaisesRegex(
        ValueError, 'Dataset element shape must have a valid rank'):
      input_util.DTensorDataset(
          dataset=dataset,
          global_batch_size=8,
          mesh=self.mesh,
          layouts=Layout.replicated(self.mesh, rank=2))

  def testIterMismatchedLayoutFails(self):
    # [FUNCTION DOC]
    # Wraps a repeated image dataset with a rank-3 layout although batched elements are rank 4.
    # Expects a value error saying a layout with rank 4 was expected.
    dataset = dataset_ops.DatasetV2.from_tensors(self.images).repeat()

    # Mismatched rank-3 layout for rank-4 input (after batching)
    images_layout = Layout(
        [MESH_DIM_BATCH, MESH_DIM_HEIGHT, MESH_DIM_WIDTH], self.mesh)

    with self.assertRaisesRegex(ValueError, 'Expected layout with rank 4'):
      _ = input_util.DTensorDataset(
          dataset=dataset,
          global_batch_size=32,
          mesh=self.mesh,
          layouts=images_layout,
          batch_dim=MESH_DIM_BATCH)

  @parameterized.named_parameters(('Eager', False), ('Graph', True))
  def testRangeIteration(self, is_graph):
    # [FUNCTION DOC]
    # Parameterized over eager (false) and graph (true) mode. Builds a dataset of 32 images, and
    # sums four batches of 8 via a loop over a range using the next call, for both a plain
    # batched iterator and the batch-sharded DTensor dataset iterator. Asserts the number of
    # iterations and summed output match and layout is batch-sharded, then asserts that one more
    # step on the exhausted DTensor iterator raises the end-of-data error (out-of-range in graph
    # mode, stop-iteration in eager mode). Uses a barrier across the mesh to flush async work.
    batch_size = 8
    num_batches = 4
    dataset = dataset_ops.DatasetV2.from_tensor_slices(
        self._images([batch_size * num_batches, 8, 8, 3]))
    images_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=4)

    d_dataset = input_util.DTensorDataset(
        dataset=dataset,
        global_batch_size=batch_size,
        mesh=self.mesh,
        layouts=images_layout,
        batch_dim=MESH_DIM_BATCH)

    def train(iterator, steps):
      # [FUNCTION DOC]
      # Helper loop: takes an iterator and a step count. Reads the first element, then adds the
      # next steps minus one elements, counting iterations (with a mesh barrier per step in
      # eager mode). Returns the summed output and the iteration count.
      iters = 1
      output = next(iterator)
      for _ in math_ops.range(steps - 1):
        output += next(iterator)
        iters += 1
        if not is_graph:
          mesh_util.barrier(self.mesh)
      return output, iters

    train_fn = polymorphic_function.function(train) if is_graph else train
    exception = errors_impl.OutOfRangeError if is_graph else StopIteration

    iterator = iter(dataset.batch(batch_size, drop_remainder=True))
    output, iters = train_fn(iterator, num_batches)

    d_iterator = iter(d_dataset)
    d_output, d_iters = train_fn(d_iterator, num_batches)

    mesh_util.barrier(self.mesh)
    # Try one more iteration which will raise an exception since the iterator is
    # exhausted.
    with self.assertRaises(exception):
      if is_graph:
        # FIXME(b/285884302): This flakily raises error
        # "Cannot add 'while_cond' function, because a different function"
        # Since num_batches is changed to 1, it retriggers SPMD expansion.
        # Recreating polymorphic function to avoid running into the error.
        train_fn = polymorphic_function.function(train)
      train_fn(d_iterator, 1)
      # In the graph case, we need to wait for the executor to finish all async
      # calls after invoking the tf.function to ensure any pending error is
      # raised.
      mesh_util.barrier(self.mesh)

    self.assertEqual(iters, d_iters)
    self.assertDTensorEqual(output, images_layout, d_output)

  @parameterized.named_parameters(('Eager', False), ('Graph', True))
  def testForInIteration(self, is_graph):
    # [FUNCTION DOC]
    # Parameterized over eager and graph mode. Same data as the range test, but sums all batches
    # by iterating directly over the iterator. Asserts the iteration counts match and the DTensor
    # output equals the plain output with the batch-sharded layout.
    batch_size = 8
    num_batches = 4
    dataset = dataset_ops.DatasetV2.from_tensor_slices(
        self._images([batch_size * num_batches, 8, 8, 3]))
    images_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=4)

    d_dataset = input_util.DTensorDataset(
        dataset=dataset,
        global_batch_size=batch_size,
        mesh=self.mesh,
        layouts=images_layout,
        batch_dim=MESH_DIM_BATCH)

    def train(iterator):
      # [FUNCTION DOC]
      # Helper loop: takes an iterator, reads its first element, then adds every remaining
      # element by for-in iteration, counting iterations (with a mesh barrier per step in eager
      # mode). Returns the summed output and the iteration count.
      iters = 1
      output = next(iterator)
      for img in iterator:
        output += img
        iters += 1
        if not is_graph:
          mesh_util.barrier(self.mesh)
      return output, iters

    train_fn = polymorphic_function.function(train) if is_graph else train

    iterator = iter(dataset.batch(batch_size, drop_remainder=True))
    output, iters = train_fn(iterator)

    d_iterator = iter(d_dataset)
    d_output, d_iters = train_fn(d_iterator)

    self.assertEqual(iters, d_iters)
    self.assertDTensorEqual(output, images_layout, d_output)

  @parameterized.named_parameters(('Eager', False), ('Graph', True))
  def testIterSingleInput(self, is_graph):
    # [FUNCTION DOC]
    # Parameterized over eager and graph mode. Wraps a repeated image dataset with global batch
    # 32 and a batch-sharded rank-4 layout. Asserts the dataset and iterator element specs have
    # shape 32x8x8x3, fetches one element and asserts it equals the first plain batch with the
    # expected layout.
    dataset = dataset_ops.DatasetV2.from_tensors(self.images).repeat()
    batch_size = 32
    images_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=4)

    d_dataset = input_util.DTensorDataset(
        dataset=dataset,
        global_batch_size=batch_size,
        mesh=self.mesh,
        layouts=images_layout,
        batch_dim=MESH_DIM_BATCH)

    self.assertEqual(d_dataset.element_spec.shape, [batch_size, 8, 8, 3])

    def train(iterator):
      # [FUNCTION DOC]
      # Helper: returns the next element of the given iterator.
      it = next(iterator)
      return it

    train_fn = polymorphic_function.function(train) if is_graph else train

    d_iterator = iter(d_dataset)
    self.assertEqual(d_iterator.element_spec.shape, [batch_size, 8, 8, 3])

    d_images = train_fn(d_iterator)
    mesh_util.barrier(self.mesh)
    expected = next(iter(dataset.batch(batch_size, drop_remainder=True)))
    mesh_util.barrier(self.mesh)
    self.assertDTensorEqual(expected, images_layout, d_images)

  @parameterized.named_parameters(('Eager', False), ('Graph', True))
  def testIterTupleInputs(self, is_graph):
    # [FUNCTION DOC]
    # Parameterized over eager and graph mode. Dataset elements are (image, label) pairs and the
    # layouts are a tuple of batch-sharded layouts (rank 4 and rank 2). Fetches one element and
    # asserts both parts equal the plain batched values with their layouts.
    dataset = dataset_ops.DatasetV2.from_tensors(
        (self.images, self.labels)
    ).repeat()
    batch_size = 32

    images_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=4)
    labels_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=2)
    layouts = (images_layout, labels_layout)

    d_dataset = input_util.DTensorDataset(
        dataset=dataset,
        global_batch_size=batch_size,
        mesh=self.mesh,
        layouts=layouts,
        batch_dim=MESH_DIM_BATCH)

    def train(iterator):
      # [FUNCTION DOC]
      # Helper: returns the next element of the given iterator.
      return next(iterator)

    train_fn = polymorphic_function.function(train) if is_graph else train

    d_iterator = iter(d_dataset)
    d_images, d_labels = train_fn(d_iterator)
    expected_images, expected_labels = next(
        iter(dataset.batch(batch_size, drop_remainder=True)))
    self.assertDTensorEqual(expected_images, images_layout, d_images)
    self.assertDTensorEqual(expected_labels, labels_layout, d_labels)

  @parameterized.named_parameters(('Eager', False), ('Graph', True))
  def testIterDictInputs(self, is_graph):
    # [FUNCTION DOC]
    # Parameterized over eager and graph mode. Dataset elements are mappings with image and
    # label entries, and the layouts are a mapping with the same keys. Fetches one element and
    # asserts each entry equals the plain batched value with its batch-sharded layout.
    dataset = dataset_ops.DatasetV2.from_tensors({
        'images': self.images,
        'labels': self.labels,
    }).repeat()
    batch_size = 32

    images_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=4)
    labels_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=2)
    layouts = {'images': images_layout, 'labels': labels_layout}

    d_dataset = input_util.DTensorDataset(
        dataset=dataset,
        global_batch_size=batch_size,
        mesh=self.mesh,
        layouts=layouts,
        batch_dim=MESH_DIM_BATCH)

    def train(iterator):
      # [FUNCTION DOC]
      # Helper: returns the next element of the given iterator.
      return next(iterator)

    train_fn = polymorphic_function.function(train) if is_graph else train

    d_iterator = iter(d_dataset)
    d_element = train_fn(d_iterator)

    expected = next(iter(dataset.batch(batch_size, drop_remainder=True)))
    self.assertDTensorEqual(expected['images'], images_layout,
                            d_element['images'])
    self.assertDTensorEqual(expected['labels'], labels_layout,
                            d_element['labels'])

  @parameterized.named_parameters(('Eager', False), ('Graph', True))
  def testIterOnBatchedDataset(self, is_graph):
    # [FUNCTION DOC]
    # Parameterized over eager and graph mode. The input dataset is already batched per replica
    # (global batch 32 divided by the batch mesh size 4, dropping remainders) and the wrapper is
    # told so. Asserts one fetched element equals the plain batch of 32 with batch-sharded layouts.
    dataset = dataset_ops.DatasetV2.from_tensors({
        'images': self.images,
        'labels': self.labels,
    }).repeat()

    images_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=4)
    labels_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=2)
    layouts = {'images': images_layout, 'labels': labels_layout}

    global_batch_size = 32
    per_replica_batch_size = global_batch_size // MESH_SIZE_BATCH
    batched_dataset = dataset.batch(per_replica_batch_size, drop_remainder=True)

    d_dataset = input_util.DTensorDataset(
        dataset=batched_dataset,
        global_batch_size=global_batch_size,
        dataset_already_batched=True,
        mesh=self.mesh,
        layouts=layouts,
        batch_dim=MESH_DIM_BATCH)

    def train(iterator):
      # [FUNCTION DOC]
      # Helper: returns the next element of the given iterator.
      return next(iterator)

    train_fn = polymorphic_function.function(train) if is_graph else train

    d_iterator = iter(d_dataset)
    d_element = train_fn(d_iterator)

    expected = next(iter(dataset.batch(global_batch_size, drop_remainder=True)))
    self.assertDTensorEqual(expected['images'], images_layout,
                            d_element['images'])
    self.assertDTensorEqual(expected['labels'], labels_layout,
                            d_element['labels'])

  def testIterOnBatchedDatasetFailsOnIncorrectBatchSize(self):
    # [FUNCTION DOC]
    # Passes an already-batched dataset with per-replica batch 16 where 8 is expected (32 divided
    # by 4). Expects a value error stating the per-replica batch size does not match, got 16 but
    # expected 8.
    dataset = dataset_ops.DatasetV2.from_tensors({
        'images': self.images,
        'labels': self.labels,
    }).repeat()

    images_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=4)
    labels_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=2)
    layouts = {'images': images_layout, 'labels': labels_layout}

    global_batch_size = 32
    per_replica_batch_size = 16  # correct value would be: 32 // 4 = 8
    batched_dataset = dataset.batch(
        per_replica_batch_size, drop_remainder=True)

    with self.assertRaisesRegex(
        ValueError,
        ('per_replica_batch_size does not matched expected size based on the '
         'mesh, got 16 but expected 8.')):
      _ = input_util.DTensorDataset(
          dataset=batched_dataset,
          global_batch_size=global_batch_size,
          dataset_already_batched=True,
          mesh=self.mesh,
          layouts=layouts,
          batch_dim=MESH_DIM_BATCH)

  def testIterOnBatchedDatasetFailsNoDropLastBatch(self):
    # [FUNCTION DOC]
    # Passes an already-batched dataset created without dropping the remainder. Expects a value
    # error telling the user to ensure remainders are dropped when batching.
    dataset = dataset_ops.DatasetV2.from_tensors({
        'images': self.images,
        'labels': self.labels,
    }).repeat()

    images_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=4)
    labels_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=2)
    layouts = {'images': images_layout, 'labels': labels_layout}

    global_batch_size = 32
    per_replica_batch_size = global_batch_size // MESH_SIZE_BATCH
    batched_dataset = dataset.batch(
        per_replica_batch_size, drop_remainder=False)

    with self.assertRaisesRegex(
        ValueError, 'Ensure drop_remainder=True when batching the dataset.'):
      _ = input_util.DTensorDataset(
          dataset=batched_dataset,
          global_batch_size=global_batch_size,
          dataset_already_batched=True,
          mesh=self.mesh,
          layouts=layouts,
          batch_dim=MESH_DIM_BATCH)

  @parameterized.named_parameters(('Disabled', False), ('Enabled', True))
  def testIterPrefetch(self, prefetch):
    # [FUNCTION DOC]
    # Parameterized over prefetch disabled or enabled (buffer size 2). A mapped dataset counts
    # elements it produces in a variable. Asserts nothing is fetched on dataset creation or
    # iterator creation, then for each of 8 batches checks (waiting up to 5 seconds on a
    # condition) that the counter reaches the number of elements expected with the prefetch
    # buffer, and that each batch equals the matching slice of the inputs.
    # KNOWN ISSUE: the local variable named count is rebound to a number inside the loop,
    # shadowing the helper function of the same name (harmless since the dataset already mapped).
    condition = threading.Condition()
    counter = variables.Variable(0)

    def count(x):
      # [FUNCTION DOC]
      # HELPER: map function for the test dataset; increments the shared counter variable by one
      # and returns the element unchanged (side effect: mutates the counter).
      counter.assign_add(1)
      return x

    num_batches = 8
    batch_size = 4
    total_elems = num_batches * batch_size
    prefetch_buffer_size = 2 if prefetch else 0

    inputs = np.arange(total_elems)
    dataset = dataset_ops.DatasetV2.from_tensor_slices(inputs)
    dataset = dataset.map(count)

    inputs_layout = Layout.batch_sharded(
        self.mesh, batch_dim=MESH_DIM_BATCH, rank=1)

    d_dataset = input_util.DTensorDataset(
        dataset=dataset,
        global_batch_size=batch_size,
        mesh=self.mesh,
        layouts=inputs_layout,
        batch_dim=MESH_DIM_BATCH,
        prefetch=prefetch_buffer_size if prefetch else None)

    # Check nothing was prefetched before iterators were created.
    self.assertEqual(counter.numpy(), 0)

    # Check nothing was prefetched before the first iteration.
    d_iterator = iter(d_dataset)
    self.assertEqual(counter.numpy(), 0)

    # The number of elements that are expected to be fetched in each iteration.
    multiple = batch_size * (self.mesh.size // MESH_SIZE_BATCH)

    # Check the number of elements fetched upon for each batch.
    for i in range(num_batches):
      elem = next(d_iterator)

      with condition:
        count = min((i + prefetch_buffer_size) * multiple,
                    num_batches * multiple)
        result = condition.wait_for(lambda: counter.numpy() >= count, timeout=5)
      self.assertTrue(result)

      start_idx, end_idx = i * batch_size, (i + 1) * batch_size
      self.assertDTensorEqual(inputs[start_idx:end_idx], inputs_layout, elem)

  @parameterized.product(
      (
          dict(
              images_sharding=[UNSHARDED, UNSHARDED, UNSHARDED, UNSHARDED],
              labels_sharding=[UNSHARDED, UNSHARDED],
          ),
          dict(
              images_sharding=[MESH_DIM_BATCH, UNSHARDED, UNSHARDED, UNSHARDED],
              labels_sharding=[MESH_DIM_BATCH, UNSHARDED],
          ),
          dict(
              images_sharding=[
                  UNSHARDED,
                  MESH_DIM_HEIGHT,
                  MESH_DIM_WIDTH,
                  UNSHARDED,
              ],
              labels_sharding=[UNSHARDED, UNSHARDED],
          ),
          dict(
              images_sharding=[
                  UNSHARDED,
                  MESH_DIM_WIDTH,
                  MESH_DIM_HEIGHT,
                  UNSHARDED,
              ],
              labels_sharding=[UNSHARDED, UNSHARDED],
          ),
          dict(
              images_sharding=[
                  MESH_DIM_BATCH,
                  MESH_DIM_HEIGHT,
                  MESH_DIM_WIDTH,
                  UNSHARDED,
              ],
              labels_sharding=[MESH_DIM_BATCH, UNSHARDED],
          ),
          dict(
              images_sharding=[
                  MESH_DIM_BATCH,
                  MESH_DIM_WIDTH,
                  MESH_DIM_HEIGHT,
                  UNSHARDED,
              ],
              labels_sharding=[MESH_DIM_BATCH, UNSHARDED],
          ),
      ),
      is_graph=[False, True],
      through_dtensor=[False, True],
  )
  def testIterWithLayouts(
      self, images_sharding, labels_sharding, is_graph, through_dtensor
  ):
    # [FUNCTION DOC]
    # Parameterized over six pairs of image/label sharding specs (unsharded, batch, spatial, and
    # combinations), eager vs graph mode, and whether the dataset is built under a default mesh
    # scope. Builds image-label pairs, creates the DTensor dataset (using the batch dimension only
    # if some spec uses it), fetches one element and compares with the plain batched element,
    # asserting the specified layouts.
    if through_dtensor:
      scope = api.default_mesh(self.mesh)
    else:
      scope = contextlib.nullcontext()

    with scope:
      batch_size = 32
      dataset = dataset_ops.DatasetV2.from_tensors(
          (self.images, self.labels)
      ).repeat()
      batched_dataset = dataset.batch(batch_size, drop_remainder=True)

      images_layout = Layout(images_sharding, self.mesh)
      labels_layout = Layout(labels_sharding, self.mesh)
      layouts = (images_layout, labels_layout)
      batch_dim = None
      if MESH_DIM_BATCH in images_sharding or MESH_DIM_BATCH in labels_sharding:
        batch_dim = MESH_DIM_BATCH

      d_dataset = input_util.DTensorDataset(
          dataset=dataset,
          global_batch_size=batch_size,
          mesh=self.mesh,
          layouts=layouts,
          batch_dim=batch_dim,
      )

      def train(iterator):
        # [FUNCTION DOC]
        # Helper: returns the next element of the given iterator.
        return next(iterator)

      train_fn = polymorphic_function.function(train) if is_graph else train

      d_iterator = iter(d_dataset)
      d_images, d_labels = train_fn(d_iterator)

    iterator = iter(batched_dataset)
    images, labels = train_fn(iterator)

    self.assertDTensorEqual(images, images_layout, d_images)
    self.assertDTensorEqual(labels, labels_layout, d_labels)

  def testMixedLayoutsFails(self):
    # [FUNCTION DOC]
    # Passes layouts where only one of two uses the batch dimension while the batch dimension is
    # specified. Expects a value error saying the batch dimension was specified but at least one
    # layout did not contain it.
    dataset = dataset_ops.DatasetV2.from_tensors(
        (self.images, self.labels)
    ).repeat()

    images_layout = Layout(
        [UNSHARDED, MESH_DIM_HEIGHT, MESH_DIM_WIDTH, UNSHARDED], self.mesh)
    labels_layout = Layout([MESH_DIM_BATCH, UNSHARDED], self.mesh)
    layouts = (images_layout, labels_layout)

    with self.assertRaisesRegex(ValueError, (
        f'batch_dim {MESH_DIM_BATCH} was specified but at least one layout did '
        'not contain it')):
      input_util.DTensorDataset(
          dataset=dataset,
          global_batch_size=32,
          mesh=self.mesh,
          layouts=layouts,
          batch_dim=MESH_DIM_BATCH)


class InputUtilHelpersTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case for small helper functions of the input utility module; currently the helper that
  # computes how many shards a layout splits each dimension into.

  @parameterized.parameters(
      {
          'mesh_dims': [(MESH_DIM_BATCH, 8)],
          'layout_specs': [UNSHARDED],
          'batch_dim': None,
          'counts': [1],
      }, {
          'mesh_dims': [(MESH_DIM_BATCH, 8)],
          'layout_specs': [MESH_DIM_BATCH],
          'batch_dim': None,
          'counts': [8],
      }, {
          'mesh_dims': [(MESH_DIM_BATCH, 8)],
          'layout_specs': [MESH_DIM_BATCH],
          'batch_dim': MESH_DIM_BATCH,
          'counts': [1],
      }, {
          'mesh_dims': [(MESH_DIM_BATCH, 2),
                        (MESH_DIM_HEIGHT, 4),
                        (MESH_DIM_WIDTH, 2)],
          'layout_specs': [UNSHARDED, MESH_DIM_HEIGHT],
          'batch_dim': None,
          'counts': [1, 4],
      }, {
          'mesh_dims': [(MESH_DIM_BATCH, 2),
                        (MESH_DIM_HEIGHT, 4),
                        (MESH_DIM_WIDTH, 2)],
          'layout_specs': [MESH_DIM_BATCH, MESH_DIM_WIDTH, MESH_DIM_HEIGHT],
          'batch_dim': None,
          'counts': [2, 2, 4],
      }, {
          'mesh_dims': [(MESH_DIM_BATCH, 2),
                        (MESH_DIM_HEIGHT, 4),
                        (MESH_DIM_WIDTH, 2)],
          'layout_specs': [MESH_DIM_BATCH, MESH_DIM_WIDTH, MESH_DIM_HEIGHT],
          'batch_dim': MESH_DIM_BATCH,
          'counts': [1, 2, 4],
      })
  def testShardCounts(self, mesh_dims, layout_specs, batch_dim, counts):
    # [FUNCTION DOC]
    # Parameterized over mesh dimension lists, layout specs, optional batch dimension and the
    # expected shard counts. Builds a CPU mesh of the product size, makes the layout, and asserts
    # the private shard-count helper returns the expected list (the batch dimension, if named,
    # counts as one shard).
    num_devices = np.prod([size for _, size in mesh_dims])
    mesh = mesh_util.create_mesh(
        mesh_dims=mesh_dims, devices=['CPU:%d' % i for i in range(num_devices)])
    layout = Layout(layout_specs, mesh)

    self.assertEqual(input_util._shard_counts(layout, batch_dim), counts)


class DTensorIteratorSpecTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case for the type specification of DTensor iterators, specifically conversion to and
  # from a list of tensors. Fixtures: an 8-device CPU mesh with one batch dimension and a random
  # 8x8x3 image tensor.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Creates an 8-device CPU mesh with a single batch dimension, configures it
    # as the test mesh, and stores a random 8x8x3 image tensor as images. Returns nothing.
    super().setUp()
    mesh = mesh_util.create_mesh(
        devices=['CPU:%d' % i for i in range(8)],
        mesh_dims=[(MESH_DIM_BATCH, 8)])
    self.mesh = self.configTestMesh({'CPU': mesh})

    self.images = stateless_random_ops.stateless_random_uniform(
        [8, 8, 3], seed=(1, 2), minval=0, maxval=255)

  def testToTensorList(self):
    # [FUNCTION DOC]
    # Creates a DTensor dataset iterator over eight replicated images (global batch 4) and a
    # matching iterator spec from its element spec and layout string. Asserts converting the
    # iterator to a tensor list gives exactly one tensor, the iterator's resource DTensor.
    dataset = dataset_ops.DatasetV2.from_tensors(self.images).repeat(8)
    images_layout = Layout.replicated(self.mesh, rank=4)
    d_dataset = input_util.DTensorDataset(
        dataset=dataset,
        global_batch_size=4,
        mesh=self.mesh,
        layouts=images_layout)
    d_iterator = iter(d_dataset)

    spec = input_util._DTensorIteratorSpec(
        global_element_spec=d_iterator._global_element_spec,
        layouts_str=d_iterator._layouts_str)

    value = d_iterator
    tensor_list = spec._to_tensor_list(value)
    self.assertListEqual(tensor_list, [d_iterator._iterator_resource_dtensor])

  def testFromTensorList(self):
    # [FUNCTION DOC]
    # Creates a DTensor iterator and a matching spec as in the previous test, then rebuilds an
    # iterator from a list holding the resource DTensor. Asserts the result is a DTensor
    # iterator that shares the same element spec object and has equal layouts.
    dataset = dataset_ops.DatasetV2.from_tensors(self.images).repeat(8)
    images_layout = Layout.replicated(self.mesh, rank=4)
    d_dataset = input_util.DTensorDataset(
        dataset=dataset,
        global_batch_size=4,
        mesh=self.mesh,
        layouts=images_layout)
    d_iterator = iter(d_dataset)

    spec = input_util._DTensorIteratorSpec(
        global_element_spec=d_iterator._global_element_spec,
        layouts_str=d_iterator._layouts_str)

    tensor_list = [d_iterator._iterator_resource_dtensor]
    value = spec._from_tensor_list(tensor_list)
    self.assertIsInstance(value, input_util._DTensorIterator)
    self.assertIs(value._global_element_spec, d_iterator._global_element_spec)
    self.assertEqual(value._layouts, d_iterator._layouts)


if __name__ == '__main__':
  tf_test.main()
