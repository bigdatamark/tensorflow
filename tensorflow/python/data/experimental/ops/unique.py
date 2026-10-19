# [MODULE DOC]
# Deprecated experimental module exposing the unique-element transformation factory, which
# delegates to the dataset's own unique method. Importing it registers the public export name.
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
"""Unique element dataset transformations."""
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@deprecation.deprecated(None, "Use `tf.data.Dataset.unique(...)")
@tf_export("data.experimental.unique")
def unique():
  # [FUNCTION DOC]
  # Deprecated factory with no parameters returning a transformation that drops duplicate
  # elements. Returns a callable taking a dataset and returning the deduplicated dataset. Emits a
  # deprecation warning when called. Elements must be of a type supported by the native op.
  """Creates a `Dataset` from another `Dataset`, discarding duplicates.

  Use this transformation to produce a dataset that contains one instance of
  each unique element in the input. For example:

  ```python
  dataset = tf.data.Dataset.from_tensor_slices([1, 37, 2, 37, 2, 1])

  # Using `unique()` will drop the duplicate elements.
  dataset = dataset.apply(tf.data.experimental.unique())  # ==> { 1, 37, 2 }
  ```

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: inner transformation returned by the unique factory.
    # Returns the result of the dataset's unique method on the given dataset.
    return dataset.unique()

  return _apply_fn
