# [MODULE DOC]
# Thin Python wrapper around the native Grappler model analyzer extension module. It exposes a
# single function that serializes a MetaGraphDef and returns a textual report of statically
# known per-node information. No import-time side effects beyond importing the native extension.
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
# =============================================================================
"""Provides a proper python API for the symbols exported through swig."""

from tensorflow.python.grappler import _pywrap_model_analyzer as tf_wrap


def GenerateModelReport(metagraph, assume_valid_feeds=True, debug=False):
  # [FUNCTION DOC]
  # Produces a human-readable text report of what Grappler can statically infer about every node
  # in a MetaGraphDef (shapes, types and similar properties).
  # Inputs: metagraph is a MetaGraphDef protocol buffer message; assume_valid_feeds is a boolean
  # (default true) saying whether fed node shapes are assumed to be valid; debug is a boolean
  # (default false) that adds extra debugging information to the report.
  # Output: a string holding the report. The metagraph is serialized to bytes and handed to the
  # native model analyzer extension; no arguments are mutated and there are no other side effects.
  # Exceptions: whatever the native extension raises for malformed input; none are raised here.
  """Report what's known statically about each node in the provided metagraph.

  Args:
    metagraph: A TensorFlow MetaGraphDef.
    assume_valid_feeds: If True, assume that the shape of the fed nodes is valid
    debug: Add some information useful for debugging.

  Returns:
    A string containing the report.
  """
  return tf_wrap.GenerateModelReport(
      metagraph.SerializeToString(), assume_valid_feeds, debug)
