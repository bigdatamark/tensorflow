# [MODULE DOC]
# Cardinality utilities for tf.data datasets. Defines the INFINITE and UNKNOWN cardinality
# constants (exported publicly at import time), a function to query a dataset cardinality, and a
# transformation that asserts the cardinality at runtime, backed by a private dataset class.
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
"""Cardinality analysis of `Dataset` objects."""
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.util.tf_export import tf_export


INFINITE = -1
UNKNOWN = -2
tf_export("data.experimental.INFINITE_CARDINALITY").export_constant(
    __name__, "INFINITE")
tf_export("data.experimental.UNKNOWN_CARDINALITY").export_constant(
    __name__, "UNKNOWN")


# TODO(b/157691652): Deprecate this method after migrating users to the new API.
@tf_export("data.experimental.cardinality")
def cardinality(dataset):
  # [FUNCTION DOC]
  # Parameter: dataset (a tf.data dataset). Returns a scalar 64-bit integer tensor with the
  # dataset's cardinality, or the infinite (-1) or unknown (-2) constants. Accesses the dataset's
  # internal variant tensor directly. Adds a cardinality op to the graph (or runs it eagerly).
  """Returns the cardinality of `dataset`, if known.

  The operation returns the cardinality of `dataset`. The operation may return
  `tf.data.experimental.INFINITE_CARDINALITY` if `dataset` contains an infinite
  number of elements or `tf.data.experimental.UNKNOWN_CARDINALITY` if the
  analysis fails to determine the number of elements in `dataset` (e.g. when the
  dataset source is a file).

  >>> dataset = tf.data.Dataset.range(42)
  >>> print(tf.data.experimental.cardinality(dataset).numpy())
  42
  >>> dataset = dataset.repeat()
  >>> cardinality = tf.data.experimental.cardinality(dataset)
  >>> print((cardinality == tf.data.experimental.INFINITE_CARDINALITY).numpy())
  True
  >>> dataset = dataset.filter(lambda x: True)
  >>> cardinality = tf.data.experimental.cardinality(dataset)
  >>> print((cardinality == tf.data.experimental.UNKNOWN_CARDINALITY).numpy())
  True

  Args:
    dataset: A `tf.data.Dataset` for which to determine cardinality.

  Returns:
    A scalar `tf.int64` `Tensor` representing the cardinality of `dataset`. If
    the cardinality is infinite or unknown, the operation returns the named
    constant `INFINITE_CARDINALITY` and `UNKNOWN_CARDINALITY` respectively.
  """

  return gen_dataset_ops.dataset_cardinality(dataset._variant_tensor)  # pylint: disable=protected-access


@tf_export("data.experimental.assert_cardinality")
def assert_cardinality(expected_cardinality):
  # [FUNCTION DOC]
  # Parameter: expected cardinality (integer). Returns a transformation function that wraps a
  # dataset in a cardinality-asserting dataset. The check happens at iteration time and fails
  # with a failed-precondition error if the actual count differs.
  """Asserts the cardinality of the input dataset.

  NOTE: The following assumes that "examples.tfrecord" contains 42 records.

  >>> dataset = tf.data.TFRecordDataset("examples.tfrecord")
  >>> cardinality = tf.data.experimental.cardinality(dataset)
  >>> print((cardinality == tf.data.experimental.UNKNOWN_CARDINALITY).numpy())
  True
  >>> dataset = dataset.apply(tf.data.experimental.assert_cardinality(42))
  >>> print(tf.data.experimental.cardinality(dataset).numpy())
  42

  Args:
    expected_cardinality: The expected cardinality of the input dataset.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    FailedPreconditionError: The assertion is checked at runtime (when iterating
      the dataset) and an error is raised if the actual and expected cardinality
      differ.
  """
  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: the returned transformation; wraps the given dataset with the captured expected
    # cardinality and returns the new dataset.
    return _AssertCardinalityDataset(dataset, expected_cardinality)

  return _apply_fn


class _AssertCardinalityDataset(dataset_ops.UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # HELPER: dataset passing through its input elements unchanged (same structure) while verifying
  # at runtime that the total count equals the expected cardinality. Holds the input dataset and
  # the expected cardinality as a 64-bit integer tensor.
  """A `Dataset` that assert the cardinality of its input."""

  def __init__(self, input_dataset, expected_cardinality):
    # [FUNCTION DOC]
    # Constructor. Parameters: input dataset; expected cardinality (integer). Stores the input,
    # converts the expectation to a tensor, builds the assert-cardinality dataset operation, and
    # initializes the base class. A stray comment about re-enabling a lint check appears with no
    # matching disable; it has no effect.
    self._input_dataset = input_dataset
    self._expected_cardinality = ops.convert_to_tensor(
        expected_cardinality, dtype=dtypes.int64, name="expected_cardinality")

    # pylint: enable=protected-access
    variant_tensor = ged_ops.assert_cardinality_dataset(
        self._input_dataset._variant_tensor,  # pylint: disable=protected-access
        self._expected_cardinality,
        **self._flat_structure)
    super(_AssertCardinalityDataset, self).__init__(input_dataset,
                                                    variant_tensor)
