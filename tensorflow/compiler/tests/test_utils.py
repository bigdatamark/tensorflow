# [MODULE DOC]
# HELPER: shared utility module for the XLA compiler op tests.
# Provides numpy-based helpers to convert arrays and shape lists between image/volume data layouts
# (such as channels-first and channels-last) and a helper that runs a session graph repeatedly so
# JIT clusters are compiled before the measured run. No import-time side effects.
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
"""Utilities for helping test ops."""

import numpy as np
from six.moves import range


def ConvertBetweenDataFormats(x, data_format_src, data_format_dst):
  # [FUNCTION DOC]
  # HELPER: test utility for ops tests that must feed data in different layouts.
  # Transposes a 4-D or 5-D array from one data layout to another. Inputs: x, a numpy array;
  # data_format_src, the layout string describing x; data_format_dst, the desired layout string.
  # Both layouts must be one of the supported names and have equal length matching the rank of x.
  # Returns x itself (no copy) when the layouts are identical, otherwise a transposed view.
  # Raises a value error if layouts differ in length, are unsupported, x is not rank 4 or 5, or
  # the rank of x does not match the layout length.
  """Converts 4D/5D tensor between data formats."""

  valid_data_formats = ["NHWC", "NCHW", "HWNC", "HWCN", "NDHWC", "NCDHW"]
  if len(data_format_src) != len(data_format_dst):
    raise ValueError(
        "data_format_src and data_format_dst must have the same dimension, got"
        " %s and %s." % (len(data_format_src), len(data_format_dst))
    )
  if data_format_src not in valid_data_formats:
    raise ValueError("data_format_src must be of %s, got %s." %
                     (valid_data_formats, data_format_src))
  if data_format_dst not in valid_data_formats:
    raise ValueError("data_format_dst must be of %s, got %s." %
                     (valid_data_formats, data_format_dst))
  if len(x.shape) != 4 and len(x.shape) != 5:
    raise ValueError("x must be 4D or 5D, got shape %s." % x.shape)
  if len(x.shape) != len(data_format_src):
    raise ValueError(
        "x must be the same dimensions as data_format_src (%s), got shape %s."
        % (len(data_format_src), x.shape)
    )

  if data_format_src == data_format_dst:
    return x

  dim_map = {d: i for i, d in enumerate(data_format_src)}
  transpose_dims = [dim_map[d] for d in data_format_dst]
  return np.transpose(x, transpose_dims)


def PermuteDimsBetweenDataFormats(dims, data_format_src, data_format_dst):
  # [FUNCTION DOC]
  # HELPER: test utility companion of the array layout converter, working on shape lists.
  # Reorders a list of dimension sizes (length 4 or 5) from the source layout to the destination
  # layout. Inputs: dims, a sequence of sizes; data_format_src and data_format_dst, layout names
  # of equal length chosen from the supported set. Returns dims unchanged when layouts are equal,
  # otherwise a new list permuted into the destination order.
  # Raises a value error for mismatched or unsupported layouts, or when dims has the wrong length.
  """Get new shape for converting between data formats."""

  valid_data_formats = ["NHWC", "NCHW", "HWNC", "HWCN", "NDHWC", "NCDHW"]
  if len(data_format_src) != len(data_format_dst):
    raise ValueError(
        "data_format_src and data_format_dst must have the same dimension, got"
        " %s and %s." % (len(data_format_src), len(data_format_dst))
    )
  if data_format_src not in valid_data_formats:
    raise ValueError("data_format_src must be of %s, got %s." %
                     (valid_data_formats, data_format_src))
  if data_format_dst not in valid_data_formats:
    raise ValueError("data_format_dst must be of %s, got %s." %
                     (valid_data_formats, data_format_dst))
  if len(dims) != 4 and len(dims) != 5:
    raise ValueError("dims must be of length 4 or 5, got %s." % dims)
  if len(dims) != len(data_format_src):
    raise ValueError(
        "dims must be the same dimensions as data_format_src (%s), got %s."
        % (len(data_format_src), dims)
    )

  if data_format_src == data_format_dst:
    return dims

  dim_map = {d: i for i, d in enumerate(data_format_src)}
  permuted_dims = [dims[dim_map[d]] for d in data_format_dst]
  return permuted_dims


_JIT_WARMUP_ITERATIONS = 10


def RunWithWarmup(sess, op_to_run, feed_dict, options=None, run_metadata=None):
  # [FUNCTION DOC]
  # HELPER: lets tests exercise JIT-compiled clusters after compilation has happened.
  # Runs the given operation in the session ten times (results discarded) and then once more,
  # returning that final result. Inputs: sess, a session; op_to_run, the tensor/op to fetch;
  # feed_dict, input values; options and run_metadata, optional run options and a metadata
  # object that is filled only during the final run. Side effects: executes the graph eleven times.
  """Runs a graph a few times to ensure that its clusters are compiled."""
  for _ in range(0, _JIT_WARMUP_ITERATIONS):
    sess.run(op_to_run, feed_dict, options=options)
  return sess.run(
      op_to_run, feed_dict, options=options, run_metadata=run_metadata)
