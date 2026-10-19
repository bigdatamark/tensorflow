# [MODULE DOC]
# Implementation of the dataset counter operation: a private function building an unbounded dataset
# of evenly spaced numbers. Used by the public dataset class. No import-time side effects.
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
"""The implementation of `tf.data.Dataset.counter`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import ops


def _counter(start, step, dtype, name=None):
  # [FUNCTION DOC]
  # HELPER: implementation behind the public counter function.
  # Inputs: start and step (numbers, converted to tensors of the given dtype), dtype, and an
  # optional op name. Returns an infinite dataset whose elements are start, start+step,
  # start+2*step and so on, built by scanning over an endlessly repeated single element.
  # Creates ops under a counter name scope.
  with ops.name_scope("counter"):
    start = ops.convert_to_tensor(start, dtype=dtype, name="start")
    step = ops.convert_to_tensor(step, dtype=dtype, name="step")
    return (dataset_ops.Dataset.from_tensors(0, name=name).repeat(None).scan(
        start, lambda state, _: (state + step, state)))
