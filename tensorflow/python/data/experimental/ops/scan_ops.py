# [MODULE DOC]
# Deprecated experimental module exposing the scan transformation factory, which delegates to the
# dataset's own scan method. Importing it registers the public export name; no other side effects.
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
"""Scan dataset transformation."""
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@deprecation.deprecated(None, "Use `tf.data.Dataset.scan(...) instead")
@tf_export("data.experimental.scan")
def scan(initial_state, scan_func):
  # [FUNCTION DOC]
  # Deprecated factory returning a transformation that threads accumulator state through a
  # dataset. initial_state is a nested structure of tensors giving the starting state; scan_func
  # maps the old state and an input element to a new state and an output element, and the new
  # state must match the structure of the initial state. Returns a callable taking a dataset and
  # returning the scanned dataset. Emits a deprecation warning when called.
  """A transformation that scans a function across an input dataset.

  This transformation is a stateful relative of `tf.data.Dataset.map`.
  In addition to mapping `scan_func` across the elements of the input dataset,
  `scan()` accumulates one or more state tensors, whose initial values are
  `initial_state`.

  Args:
    initial_state: A nested structure of tensors, representing the initial state
      of the accumulator.
    scan_func: A function that maps `(old_state, input_element)` to
      `(new_state, output_element)`. It must take two arguments and return a
      pair of nested structures of tensors. The `new_state` must match the
      structure of `initial_state`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    # [FUNCTION DOC]
    # HELPER: inner transformation returned by the scan factory.
    # Takes a dataset and returns the result of its scan method using the captured initial state
    # and scan function.
    return dataset.scan(initial_state=initial_state, scan_func=scan_func)

  return _apply_fn
