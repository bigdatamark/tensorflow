# [MODULE DOC]
# Deprecated experimental module exposing the take-while transformation factory, which delegates
# to the dataset's own take-while method. Importing it registers the public export name.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""take-while dataset transformation."""
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@deprecation.deprecated(None, "Use `tf.data.Dataset.take_while(...)")
@tf_export("data.experimental.take_while")
def take_while(predicate):
  # [FUNCTION DOC]
  # Deprecated factory returning a transformation that stops iteration at the first element for
  # which the predicate is false. The predicate maps an element to a scalar boolean tensor.
  # Returns a callable taking a dataset and returning the truncated dataset. Emits a deprecation
  # warning when called.
  """A transformation that stops dataset iteration based on a `predicate`.

  Args:
    predicate: A function that maps a nested structure of tensors (having shapes
      and types defined by `self.output_shapes` and `self.output_types`) to a
      scalar `tf.bool` tensor.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: inner transformation returned by the take-while factory.
    # Applies the dataset's take-while method with the captured predicate and returns the result.
    return dataset.take_while(predicate=predicate)

  return _apply_fn
