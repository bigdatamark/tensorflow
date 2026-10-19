# [MODULE DOC]
# Implementation module for the dataset range transformation: a source dataset of evenly spaced
# integers following the rules of the built-in range. Exposes a private factory function and a
# dataset source class, used by the main dataset module. No import-time side effects.
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
"""The implementation of `tf.data.Dataset.range`."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import gen_dataset_ops


def _range(*args, **kwargs):  # pylint: disable=unused-private-name
  # [FUNCTION DOC]
  # HELPER: factory used by the public dataset range method.
  # Forwards all positional and keyword arguments (one to three bounds, optional output type
  # and name) to the range dataset constructor and returns the new dataset.
  return _RangeDataset(*args, **kwargs)


class _RangeDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # Source dataset of scalar integers from start up to (excluding) stop by step. Holds start,
  # stop and step as 64-bit tensors, the output type, the name and the element structure.
  """A `Dataset` of a step separated range of values."""

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Parses the arguments, sets the element spec to a scalar of the chosen output type, creates
    # the range op from start, stop and step, and calls the parent constructor with it.
    # Raises a value error (via argument parsing) when the number of positional arguments is
    # not between one and three.
    """See `Dataset.range()` for details."""
    self._parse_args(*args, **kwargs)
    self._structure = tensor_spec.TensorSpec([], self._output_type)
    variant_tensor = gen_dataset_ops.range_dataset(
        start=self._start,
        stop=self._stop,
        step=self._step,
        **self._common_args)
    super().__init__(variant_tensor)

  def _parse_args(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: argument parser for the constructor.
    # KNOWN ISSUE: unknown keyword arguments are silently ignored.
    # One positional argument means stop (start 0, step 1); two mean start and stop (step 1);
    # three mean start, stop and step. Each is converted to a 64-bit integer tensor stored on the
    # instance. Optional keyword output_type (default 64-bit integer) and name (default nothing)
    # are read. Raises a value error for any other number of positional arguments.
    """Parses arguments according to the same rules as the `range()` builtin."""
    if len(args) == 1:
      self._start = self._build_tensor(0, "start")
      self._stop = self._build_tensor(args[0], "stop")
      self._step = self._build_tensor(1, "step")
    elif len(args) == 2:
      self._start = self._build_tensor(args[0], "start")
      self._stop = self._build_tensor(args[1], "stop")
      self._step = self._build_tensor(1, "step")
    elif len(args) == 3:
      self._start = self._build_tensor(args[0], "start")
      self._stop = self._build_tensor(args[1], "stop")
      self._step = self._build_tensor(args[2], "step")
    else:
      raise ValueError(f"Invalid `args`. The length of `args` should be "
                       f"between 1 and 3 but was {len(args)}.")
    if "output_type" in kwargs:
      self._output_type = kwargs["output_type"]
    else:
      self._output_type = dtypes.int64
    self._name = kwargs["name"] if "name" in kwargs else None

  def _build_tensor(self, int64_value, name):
    # [FUNCTION DOC]
    # HELPER: converts a value to a 64-bit integer tensor with the given op name and returns it.
    return ops.convert_to_tensor(int64_value, dtype=dtypes.int64, name=name)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec: a scalar tensor spec of the configured output type.
    return self._structure
