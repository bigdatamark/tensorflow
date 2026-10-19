# [MODULE DOC]
# HELPER: Shared base class and model-building utilities for the TF quantization integration tests.
# Defines a type alias for supported op-attribute values (list of ints, bool, string or nothing)
# and the QuantizedModelTest base class, which combines the TensorFlow test case with the
# parameterized test mixin. The class offers helpers that build small TF1 (graph mode) and TF2
# (module based) models (convolution, depthwise convolution, matmul, einsum, gather, while loop,
# vocabulary-table lookup, with optional quantization-aware-training fake-quant ops), save them as
# saved models, create representative datasets, and inspect GraphDefs (check for or count ops and
# quantized/composite functions). Importing the module has no side effects beyond importing
# TensorFlow itself; it is meant to be subclassed by the actual quantize_model test files.
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
"""Base test class for quantize_model Tests."""
import os
import re
from typing import Collection, Iterable, Mapping, Sequence, Tuple, Optional, Union, List

from absl.testing import parameterized
import numpy as np
import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop as while_loop_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging  # pylint: disable=unused-import
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.saved_model import signature_def_utils_impl
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.types import core

# Type aliases for supported attribute types.
_AttrValType = Union[List[int], bool, str, None]


class QuantizedModelTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # HELPER: Base class that quantization integration tests derive from.
  # Test case base (TensorFlow test case plus parameterized tests) that holds three temporary
  # directory paths (input model, output model, second output model) created per test, and offers
  # many private helpers to build and save float or QAT models and to inspect graph definitions.
  # Not meant to contain test methods itself.
  """Base test class for TF-quant tests."""

  def setUp(self) -> None:
    # [FUNCTION DOC]
    # Test fixture run before each test. Calls the parent setUp, then creates three temporary
    # directories (named input, output, output2) and stores their full paths in the attributes for
    # the input saved model path, output saved model path and second output saved model path.
    # Takes no arguments and returns nothing. The directories are cleaned up by the test framework.
    super().setUp()

    # Many test cases for quantization involve creating and saving the input
    # model and saving the output quantized model. These two member
    # attributes can be used to specify the paths for such models,
    # respectively. These paths will be cleaned up after each test case.
    self._input_saved_model_path = self.create_tempdir('input').full_path
    self._output_saved_model_path = self.create_tempdir('output').full_path
    # Extra output path occasionally used for comparing two different
    # quantized models.
    self._output_saved_model_path_2 = self.create_tempdir('output2').full_path

  def _get_dir_size(self, path: str = '.'):
    # [FUNCTION DOC]
    # HELPER: Measures saved-model size in tests.
    # Walks the directory tree under path (default: current directory) and sums the byte sizes of
    # all files found. Returns the total as an integer. A plain file path yields zero because the
    # walk finds no files there, contrary to the docstring that claims a file is supported.
    # KNOWN ISSUE: the docstring says path may be a file, but a file path returns zero.
    """Get the total size of files and sub-directories under the path.

    Args:
      path: Path of a directory or a file to calculate the total size.

    Returns:
      Total size of the directory or a file.
    """
    total = 0
    for root, _, files in os.walk(path):
      for filename in files:
        total += os.path.getsize(os.path.join(root, filename))
    return total

  def _any_log_contains(
      self, substring: str, log_record_list: List['logging.LogRecord']
  ) -> bool:
    # [FUNCTION DOC]
    # HELPER: Assertion helper for checking captured log output.
    # Inputs: substring (text to look for) and log_record_list (list of log records). Returns true
    # if the message of any record contains the substring, false otherwise (also for an empty list).
    """Returns True if any of the log contains a given substring.

    Args:
      substring: A piece of string to check whether it exists in the log
        message.
      log_record_list: A list of `absl.logging.LogRecord`s.

    Returns:
      True if and only if the substring exists in any of the log in
      `log_record_list`.
    """
    return any(
        map(
            lambda log_record: substring in str(log_record.message),
            log_record_list,
        )
    )

  def _is_quantized_function(self, func: function_pb2.FunctionDef) -> bool:
    # [FUNCTION DOC]
    # HELPER: Predicate used by the graph-inspection helpers.
    # Input: a function definition message. Returns true if the function's signature name starts
    # with the prefix "quantized_", false otherwise.
    """Determine whether a FunctionDef is quantized.

    Args:
      func: A FunctionDef object.

    Returns:
      True iff `func` is quantized.
    """
    return func.signature.name.startswith('quantized_')

  def _is_composite_function(self, func: function_pb2.FunctionDef) -> bool:
    # [FUNCTION DOC]
    # HELPER: Predicate used by the graph-inspection helpers.
    # Input: a function definition message. Returns true if the function's signature name starts
    # with the prefix "composite_", false otherwise.
    """Determine whether a FunctionDef is composite function.

    Args:
      func: A FunctionDef object.

    Returns:
      True iff `func` is composte function.
    """
    return func.signature.name.startswith('composite_')

  def _contains_op_with_name_and_attribute(
      self,
      nodes: Iterable[node_def_pb2.NodeDef],
      op_name: str,
      attr_name: str,
      attr_val: _AttrValType,
      node_name: str = '',
  ) -> bool:
    # [FUNCTION DOC]
    # HELPER: Graph-inspection helper for the op-existence check.
    # KNOWN ISSUE: the attribute comparison is always applied (not only when attr_name is given);
    # with the default empty name and nothing value it passes only because a missing attribute
    # compares equal to nothing.
    # Inputs: nodes (iterable of node definitions), op_name, attr_name, attr_val, and node_name
    # (optional regular expression that must match the whole node name; empty means any node).
    # Returns true if some node has the given op type, a matching name, and an attribute value equal
    # to attr_val. Contains a nested function that applies the full-match regular expression check.
    """Determine whether there is a node whose operation name matches `op_name`.

    If `attr_name` is given, additionally check if the `attr_val` matches with
    the attribute value of the op.

    Args:
      nodes: Iterable of NodeDefs.
      op_name: Name of the op to match.
      attr_name: Name of the attribute of the op to match.
      attr_val: Value of the attr_name to check.
      node_name: Name of the node to match. Accepts regex2 format.

    Returns:
      True if there exists a node whose name matches `op_name` and 'attr_val' if
      'attr_name' is given.
    """

    def match_node_name(name):
      # [FUNCTION DOC]
      # HELPER: Name filter for the enclosing node search.
      # Takes a node name. Returns true when no node_name pattern was given, otherwise true only if
      # the whole name fully matches the regular expression from the enclosing scope.
      if not node_name:
        return True
      compiled_regex = re.compile(node_name)
      match = re.fullmatch(compiled_regex, name)
      return match is not None

    return any(
        node.attr.get(attr_name) == attr_val
        for node in nodes
        if node.op == op_name and match_node_name(node.name)
    )

  def _contains_quantized_function_call(
      self, graphdef: graph_pb2.GraphDef
  ) -> bool:
    # [FUNCTION DOC]
    # HELPER: Assertion helper for quantized output graphs.
    # Input: a graph definition. Returns true if any function in its function library is a
    # quantized function (name starts with "quantized_").
    """Determines if the graph def has quantized function call.

    Args:
      graphdef: A GraphDef object.

    Returns:
      True if and only if the graph def contains a quantized function call.
    """
    return any(map(self._is_quantized_function, graphdef.library.function))

  def _contains_composite_function_call(
      self, graphdef: graph_pb2.GraphDef
  ) -> bool:
    # [FUNCTION DOC]
    # HELPER: Assertion helper for quantized output graphs.
    # Input: a graph definition. Returns true if any function in its function library is a
    # composite function (name starts with "composite_").
    """Determines if the graph def has composite function call.

    Args:
      graphdef: A GraphDef object.

    Returns:
      True if and only if the graph def contains a composite function call.
    """
    return any(map(self._is_composite_function, graphdef.library.function))

  def _contains_op(
      self,
      graphdef: graph_pb2.GraphDef,
      op_name: str,
      attr_name: str = '',
      attr_val: _AttrValType = None,
      node_name: str = '',
  ) -> bool:
    # [FUNCTION DOC]
    # HELPER: Assertion helper that searches a graph for an op.
    # Inputs: graphdef, op_name, optional attr_name (default empty), attr_val (default nothing) and
    # node_name (optional full-match regular expression, default empty). Searches the main graph
    # nodes first, then the nodes of every function in the library. Returns true on the first
    # match, false otherwise. Delegates to the node-level matching helper.
    """Determines if the graph def contains the given op.

    Args:
      graphdef: A GraphDef object.
      op_name: Name of the operation to find within the graph.
      attr_name: Name of the attribute of the op to match.
      attr_val: Value of the attr_name to check.
      node_name: Name of the node to match. Accepts regex2 format.

    Returns:
      True if and only if the graph def contains an op named `op_name`. If
      `attr_name` is given, check if the `attr_val` matches with the attribute
      value of the op.
    """
    # Check the main graph
    if self._contains_op_with_name_and_attribute(
        nodes=graphdef.node,
        op_name=op_name,
        attr_name=attr_name,
        attr_val=attr_val,
        node_name=node_name,
    ):
      return True

    # Check the graph genederated from user defined functions
    for func in graphdef.library.function:
      if self._contains_op_with_name_and_attribute(
          nodes=func.node_def,
          op_name=op_name,
          attr_name=attr_name,
          attr_val=attr_val,
          node_name=node_name,
      ):
        return True
    return False

  def _count_ops(
      self,
      graphdef: graph_pb2.GraphDef,
      op_names: Collection[str],
      attr_name: str = '',
      attr_val: _AttrValType = None,
      get_op_name: bool = False,
  ) -> int:
    # [FUNCTION DOC]
    # HELPER: Assertion helper that counts ops in a graph.
    # KNOWN ISSUE: the docstring says ops are counted only if the attribute value matches, but the
    # delegate counts every node matching the name regardless of attribute value.
    # Inputs: graphdef, op_names (collection of names), attr_name and attr_val (defaults empty and
    # nothing), get_op_name (when true compare node names instead of op types). For each name,
    # sums matches in the main graph and in every library function. Returns the integer total.
    """Returns the number of given ops in a graph def.

    Args:
      graphdef: A GraphDef object.
      op_names: Names of the operations to find within the graph.
      attr_name: Name of the attribute of the ops to match.
      attr_val: Value of the attr_name to check.
      get_op_name: If set True, checks node.name rather than node.op.

    Returns:
      The number of occurrences of the given ops in a graph. The ops will be
      counted only if the ops are named 'op_name' and has 'attr_val' if
      'attr_name' is specified.
    """
    op_count = 0
    for op_name in op_names:
      # Check the main graph
      op_count += self._count_op_with_name_and_attribute(
          nodes=graphdef.node,
          op_name=op_name,
          attr_name=attr_name,
          attr_val=attr_val,
          get_op_name=get_op_name,
      )

      # Check the graph genederated from user defined functions
      for func in graphdef.library.function:
        op_count += self._count_op_with_name_and_attribute(
            nodes=func.node_def,
            op_name=op_name,
            attr_name=attr_name,
            attr_val=attr_val,
            get_op_name=get_op_name,
        )
    return op_count

  def _count_op_with_name_and_attribute(
      self,
      nodes: Iterable[node_def_pb2.NodeDef],
      op_name: str,
      attr_name: str,
      attr_val: _AttrValType,
      get_op_name: bool = False,
  ) -> int:
    # [FUNCTION DOC]
    # HELPER: Node-level counter for the op-count helper.
    # KNOWN ISSUE: builds a list of attribute-equality booleans and returns its length, so the
    # attribute comparison has no effect; every node whose op type (or node name when get_op_name
    # is set) equals op_name is counted, contradicting the docstring.
    # Inputs: nodes, op_name, attr_name, attr_val, get_op_name (default false). Returns the count.
    """Determine the number of nodes whose operation name matches `op_name`.

    If `attr_name` is given, additionally check if the `attr_val` matches with
    the attribute value of the op.

    Args:
      nodes: Iterable of NodeDefs.
      op_name: Name of the op to match.
      attr_name: Name of the attribute of the op to match.
      attr_val: Value of the attr_name to check.
      get_op_name: If set True, checks node.name rather than node.op.

    Returns:
      The number of occurrences of nodes whose name match `op_name` and
      'attr_val' if 'attr_name' is given.
    """
    if get_op_name:
      return len(
          [
              node.attr.get(attr_name) == attr_val
              for node in nodes
              if node.name == op_name
          ]
      )
    else:
      return len(
          [
              node.attr.get(attr_name) == attr_val
              for node in nodes
              if node.op == op_name
          ]
      )

  def _create_simple_tf1_conv_model(
      self,
      input_shape: Sequence[int] = (1, 3, 4, 3),
      filter_shape: Sequence[int] = (2, 3, 3, 2),
      use_variable_for_filter=False,
  ) -> Tuple[core.Tensor, core.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Builds a small TF1 graph-mode model for tests.
    # Inputs: input_shape (default 1x3x4x3), filter_shape (default 2x3x3x2), use_variable_for_filter
    # (default false). Adds a float placeholder, a random uniform filter in minus one to one
    # (wrapped in a variable if requested) and a same-padded channels-last 2D convolution with
    # strides 1,1,2,1 into the current default graph. Returns the placeholder and output tensor.
    # Side effects: adds ops to the current graph; the variable is not initialized here.
    """Creates a basic convolution model.

    This is intended to be used for TF1 (graph mode) tests.

    Args:
      input_shape: Shape of the input tensor.
      filter_shape: Shape of the filter.
      use_variable_for_filter: Setting this to `True` makes the filter for the
        conv operation a `tf.Variable`.

    Returns:
      in_placeholder: Input tensor placeholder.
      output_tensor: The resulting tensor of the convolution operation.
    """
    in_placeholder = array_ops.placeholder(dtypes.float32, shape=input_shape)

    filters = random_ops.random_uniform(
        shape=filter_shape, minval=-1.0, maxval=1.0
    )
    if use_variable_for_filter:
      filters = variables.Variable(filters)

    output_tensor = nn_ops.conv2d(
        in_placeholder,
        filters,
        strides=[1, 1, 2, 1],
        dilations=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC',
    )

    return in_placeholder, output_tensor

  def _create_simple_tf1_gather_model(
      self, input_type: dtypes.DType, use_variable_for_filter=False
  ) -> Tuple[core.Tensor, core.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Builds a small TF1 graph-mode model for tests.
    # Inputs: input_type (dtype of the index placeholder), use_variable_for_filter (default false).
    # Creates a one-dimensional index placeholder of length six, a random normal 128x32 float
    # table (wrapped in a variable if requested) and gathers rows of the table by the indices.
    # Returns the placeholder and the gather output tensor. Adds ops to the current graph.
    """Creates a basic gather model.

    This is intended to be used for TF1 (graph mode) tests.

    Args:
      input_type: type of the input index tensor for gather operation.
      use_variable_for_filter: Setting this to `True` makes the filter for the
        gather operation a `tf.Variable`.

    Returns:
      in_placeholder: Input tensor placeholder.
      output_tensor: The resulting tensor of the gather operation.
    """
    in_placeholder = array_ops.placeholder(input_type, shape=(6))

    filters = np.random.randn(128, 32).astype(np.float32)
    if use_variable_for_filter:
      filters = variables.Variable(filters)

    output_tensor = array_ops.gather_v2(filters, in_placeholder)

    return in_placeholder, output_tensor

  def _create_and_save_vocab_table_lookup_model_tf1(
      self,
      output_path: str,
      tags: Collection[str],
      signature_def_key: str,
  ) -> Tuple[Mapping[str, core.Tensor], Mapping[str, core.Tensor]]:
    # [FUNCTION DOC]
    # HELPER: Creates and saves a TF1 vocabulary-table test model.
    # Inputs: output_path (directory to save into), tags (collection of meta graph tags),
    # signature_def_key (name of the signature). Builds the vocab-table lookup model in a fresh
    # graph and session, then saves it with the table initializer as init op and the asset file
    # paths collection as assets. Returns two mappings: inputs (key "input_vocabs") and outputs
    # (keys "lookup" and "output"). Side effects: writes a saved model to disk and an asset file.
    """Creates and saves a simple model that uses a vocab table.

    Args:
      output_path: Path to the directory to save the created model.
      tags: Set of strings that identifies the saved meta graph.
      signature_def_key: Name of the SignatureDef. Used to identify the
        SignatureDef within the meta graph.

    Returns:
      inputs: A mapping of input_key -> input_tensor (placeholder). The input
        key is "input_vocabs".
      outputs: A mapping of output_key -> output_tensor. The output keys are
        "lookup" and "output".
    """
    with session.Session(graph=ops.Graph()) as sess:
      input_vocabs_placeholder, lookup_tensor, output_tensor = (
          self._create_vocab_table_lookup_model_tf1(sess)
      )

      inputs = {'input_vocabs': input_vocabs_placeholder}
      outputs = {
          'lookup': lookup_tensor,
          'output': output_tensor,
      }

      self._save_tf1_model(
          sess,
          output_path,
          signature_def_key,
          tags,
          inputs=inputs,
          outputs=outputs,
          init_op=lookup_ops.tables_initializer(),
          assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
      )

    return inputs, outputs

  def _create_and_save_file_init_hash_table_model_tf1(
      self,
      output_path: str,
      tags: Collection[str],
      signature_def_key: str,
  ) -> Tuple[Mapping[str, core.Tensor], Mapping[str, core.Tensor]]:
    # [FUNCTION DOC]
    # HELPER: Creates and saves a TF1 file-initialized hash table test model.
    # Inputs: output_path, tags, signature_def_key as in the vocab-table variant. Builds the
    # table-from-file model in a fresh graph and session and saves it with the table initializer
    # as init op and the asset collection. Returns inputs (key "input_vocabs") and outputs (keys
    # "lookup" and "output"). Side effects: writes a saved model and an asset file to disk.
    """Creates and saves a model that uses a file-initialized table.

    The asset file "vocab_file.txt" is used to initialize a hash table.

    Args:
      output_path: Path to the directory to save the created model.
      tags: Set of strings that identifies the saved meta graph.
      signature_def_key: Name of the SignatureDef. Used to identify the
        SignatureDef within the meta graph.

    Returns:
      inputs: A mapping of input_key -> input_tensor (placeholder). The input
        key is "input_vocabs".
      outputs: A mapping of output_key -> output_tensor. The output keys are
        "lookup" and "output".
    """
    with session.Session(graph=ops.Graph()) as sess:
      input_vocabs_placeholder, lookup_tensor, output_tensor = (
          self._create_table_init_from_file_model_tf1(sess)
      )

      inputs = {'input_vocabs': input_vocabs_placeholder}
      outputs = {
          'lookup': lookup_tensor,
          'output': output_tensor,
      }

      self._save_tf1_model(
          sess,
          output_path,
          signature_def_key,
          tags,
          inputs=inputs,
          outputs=outputs,
          init_op=lookup_ops.tables_initializer(),
          assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
      )

    return inputs, outputs

  def _create_table_init_from_file_model_tf1(
      self, sess: session.Session
  ) -> Tuple[core.Tensor, core.Tensor, core.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Builds the graph for the file-initialized table test model.
    # Input: sess (session, unused except as context). Writes a three-line vocabulary file
    # ("static", "range", "quantization") in a temp directory, creates a static hash table from it
    # (line number as value, default minus one), a string placeholder named input_vocabs, casts
    # the lookups to float, stacks them twice, multiplies by an all-ones weight matrix sized from
    # the input length. Returns the placeholder, lookup values and matmul output (a 2x2 matrix).
    """Creates a simple model that initializes a table from an asset file.

    This model creates an asset file at "vocab_file.txt" containing
    comma-separated vocabularies and uses it to initialize a
    `StaticVocabularyTable`. For inference, the model performs a lookup with a
    1D string tensor input vocabs.

    Args:
      sess: Tensorflow Session to create the model in.

    Returns:
      (input_vocabs_placeholder, lookup_vals, output_tensor), where
      * input_vocabs_placeholder is a placeholder tensor of 1D strings
      * lookup_vals is an output tensor that is a direct result of table lookup
      * output_tensor is a float 2x2 matrix
    """
    # Creates and populates an asset file.
    asset_dir = self.create_tempdir('assets').full_path
    asset_file = os.path.join(asset_dir, 'vocab_file.txt')
    content = '\n'.join(['static', 'range', 'quantization'])
    file_io.write_string_to_file(filename=asset_file, file_content=content)

    # The resulting table looks like:
    # "static" -> 0
    # "range" -> 1
    # "quantization" -> 2
    # default -> -1
    init = lookup_ops.TextFileInitializer(
        filename=asset_file,
        key_dtype=dtypes.string,
        key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
        value_dtype=dtypes.int64,
        value_index=lookup_ops.TextFileIndex.LINE_NUMBER,
    )
    table = lookup_ops.StaticHashTable(init, default_value=-1)

    input_vocabs_placeholder = array_ops.placeholder(
        dtypes.string, shape=(None,), name='input_vocabs'
    )

    # Introduce a matmul op that takes the lookup values to observe the
    # effects of quantization.
    lookup_vals = math_ops.cast(
        table.lookup(input_vocabs_placeholder), dtypes.float32
    )
    # shape: (2, ?)
    matmul_input = array_ops_stack.stack([lookup_vals, lookup_vals])

    # Create a dummy weight matrix filled with ones.
    weight_row = array_ops.ones(
        shape=array_ops.shape(input_vocabs_placeholder), dtype=dtypes.float32
    )
    # shape: (?, 2)
    weight = array_ops.transpose_v2(
        array_ops_stack.stack([weight_row, weight_row])
    )
    # shape: (2, 2)
    output_tensor = math_ops.matmul(matmul_input, weight)

    return input_vocabs_placeholder, lookup_vals, output_tensor

  def _create_vocab_table_lookup_model_tf1(
      self, sess: session.Session
  ) -> Tuple[core.Tensor, core.Tensor, core.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Builds the graph for the vocabulary-table test model.
    # KNOWN ISSUE: the docstring says comma-separated vocabularies are written, which matches the
    # file content, but the table keys come from splitting that single line, so three keys map to
    # ids zero to two with five out-of-vocabulary buckets.
    # Input: sess (unused except as context). Writes an asset file with the text
    # "hello,model,quantization", reads and splits it on commas into keys, builds a static
    # vocabulary table with five OOV buckets, looks up a string placeholder named input_vocabs,
    # casts to float, stacks twice and multiplies by an all-ones weight matrix. Returns the
    # placeholder, lookup values and the 2x2 matmul output.
    """Creates a simple model that initializes and lookups a vocab table.

    This model creates an asset file at "vocab_file.txt" containing
    comma-separated vocabularies.  It also initializes a `StaticVocabularyTable`
    and performs a lookup with the input vocabs, which is a 1D tensor of
    strings.

    Args:
      sess: Tensorflow Session to create the model in.

    Returns:
      (input_vocabs_placeholder, lookup_vals, output_tensor), where
      * input_vocabs_placeholder is a placeholder tensor of 1D strings
      * lookup_vals is an output tensor that is a direct result of table lookup
      * output_tensor is a float 2x2 matrix
    """
    # Creates and populates an asset file.
    asset_dir = self.create_tempdir('assets').full_path
    asset_file = os.path.join(asset_dir, 'vocab_file.txt')
    file_io.write_string_to_file(
        filename=asset_file, file_content='hello,model,quantization\n'
    )

    vocab_file = asset.Asset(asset_file)

    raw_vocab = io_ops.read_file(vocab_file)
    vocabs = ragged_string_ops.string_split_v2(
        string_ops.string_strip(raw_vocab), sep=','
    )

    # Initialize the vocab table. Each comma-separated word in vocab_file.txt
    # corresponds to the numeric identifiers in `values`.
    kv_init = lookup_ops.KeyValueTensorInitializer(
        keys=vocabs, values=np.array([0, 1, 2]), value_dtype=dtypes.int64
    )
    table = lookup_ops.StaticVocabularyTable(kv_init, num_oov_buckets=5)

    input_vocabs_placeholder = array_ops.placeholder(
        dtypes.string, shape=(None,), name='input_vocabs'
    )

    # Introduce a matmul op that takes the lookup values to observe the
    # effects of quantization.
    lookup_vals = math_ops.cast(
        table.lookup(input_vocabs_placeholder), dtypes.float32
    )
    # shape: (2, ?)
    matmul_input = array_ops_stack.stack([lookup_vals, lookup_vals])

    # Create a dummy weight matrix filled with ones.
    weight_row = array_ops.ones(
        shape=array_ops.shape(input_vocabs_placeholder), dtype=dtypes.float32
    )
    # shape: (?, 2)
    weight = array_ops.transpose_v2(
        array_ops_stack.stack([weight_row, weight_row])
    )
    # shape: (2, 2)
    output_tensor = math_ops.matmul(matmul_input, weight)

    return input_vocabs_placeholder, lookup_vals, output_tensor

  def _create_and_save_vocab_table_lookup_qat_model_tf1(
      self,
      output_path: str,
      tags: Collection[str],
      signature_def_key: str,
  ) -> Tuple[Mapping[str, core.Tensor], Mapping[str, core.Tensor]]:
    # [FUNCTION DOC]
    # HELPER: Creates and saves a TF1 quantization-aware-training vocabulary-table test model.
    # Inputs: output_path, tags, signature_def_key. Builds the QAT vocab-table model in a fresh
    # graph and session and saves it with the table initializer and asset collection. Returns inputs
    # (key "input_vocabs") and outputs (keys "lookup" and "output"). Writes files to disk.
    """Creates and saves a simple QAT model that uses a vocab table.

    Args:
      output_path: Path to the directory to save the created model.
      tags: Set of strings that identifies the saved meta graph.
      signature_def_key: Name of the SignatureDef. Used to identify the
        SignatureDef within the meta graph.

    Returns:
      inputs: A mapping of input_key -> input_tensor (placeholder). The input
        key is "input_vocabs".
      outputs: A mapping of output_key -> output_tensor. The output keys are
        "lookup" and "output".
    """
    with session.Session(graph=ops.Graph()) as sess:
      input_vocabs_placeholder, lookup_tensor, output_tensor = (
          self._create_vocab_table_lookup_qat_model_tf1(sess)
      )

      inputs = {'input_vocabs': input_vocabs_placeholder}
      outputs = {
          'lookup': lookup_tensor,
          'output': output_tensor,
      }

      self._save_tf1_model(
          sess,
          output_path,
          signature_def_key,
          tags,
          inputs=inputs,
          outputs=outputs,
          init_op=lookup_ops.tables_initializer(),
          assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
      )

    return inputs, outputs

  def _create_vocab_table_lookup_qat_model_tf1(
      self, sess: session.Session
  ) -> Tuple[core.Tensor, core.Tensor, core.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Builds the graph for the QAT vocabulary-table test model.
    # Same as the non-QAT vocabulary-table graph (asset file "hello,model,quantization", static
    # vocabulary table with five OOV buckets, 2x2 matmul with an all-ones weight) but inserts
    # eight-bit fake-quant ops on the matmul input (range -0.3 to 0.3), the weight (-0.1 to 0.2)
    # and the output (-0.2 to 0.2). Input: sess. Returns placeholder, lookup values, output tensor.
    """Creates a simple QAT model that initializes and lookups a vocab table.

    This model creates an asset file at "vocab_file.txt" containing
    comma-separated vocabularies.  It also initializes a `StaticVocabularyTable`
    and performs a lookup with the input vocabs, which is a 1D tensor of
    strings.

    Args:
      sess: Tensorflow Session to create the model in.

    Returns:
      (input_vocabs_placeholder, lookup_vals, output_tensor), where
      * input_vocabs_placeholder is a placeholder tensor of 1D strings
      * lookup_vals is an output tensor that is a direct result of table lookup
      * output_tensor is a float 2x2 matrix
    """
    # Creates and populates an asset file.
    asset_dir = self.create_tempdir('assets').full_path
    asset_file = os.path.join(asset_dir, 'vocab_file.txt')
    file_io.write_string_to_file(
        filename=asset_file, file_content='hello,model,quantization\n'
    )

    vocab_file = asset.Asset(asset_file)

    raw_vocab = io_ops.read_file(vocab_file)
    vocabs = ragged_string_ops.string_split_v2(
        string_ops.string_strip(raw_vocab), sep=','
    )

    # Initialize the vocab table. Each comma-separated word in vocab_file.txt
    # corresponds to the numeric identifiers in `values`.
    kv_init = lookup_ops.KeyValueTensorInitializer(
        keys=vocabs, values=np.array([0, 1, 2]), value_dtype=dtypes.int64
    )
    table = lookup_ops.StaticVocabularyTable(kv_init, num_oov_buckets=5)

    input_vocabs_placeholder = array_ops.placeholder(
        dtypes.string, shape=(None,), name='input_vocabs'
    )

    # Introduce a matmul op that takes the lookup values to observe the
    # effects of quantization.
    lookup_vals = math_ops.cast(
        table.lookup(input_vocabs_placeholder), dtypes.float32
    )

    # shape: (2, ?)
    matmul_input = array_ops_stack.stack([lookup_vals, lookup_vals])
    # Insert fake quant to simulate a QAT model.
    matmul_input = array_ops.fake_quant_with_min_max_args(
        matmul_input, min=-0.3, max=0.3, num_bits=8, narrow_range=False
    )

    # Create a dummy weight matrix filled with ones.
    weight_row = array_ops.ones(
        shape=array_ops.shape(input_vocabs_placeholder), dtype=dtypes.float32
    )

    # shape: (?, 2)
    weight = array_ops.transpose_v2(
        array_ops_stack.stack([weight_row, weight_row])
    )
    # Insert fake quant to simulate a QAT model.
    weight = array_ops.fake_quant_with_min_max_args(
        weight, min=-0.1, max=0.2, num_bits=8, narrow_range=False
    )

    # shape: (2, 2)
    output_tensor = math_ops.matmul(matmul_input, weight)
    # Insert fake quant to simulate a QAT model.
    output_tensor = array_ops.fake_quant_with_min_max_args(
        output_tensor, min=-0.2, max=0.2, num_bits=8, narrow_range=False
    )

    return input_vocabs_placeholder, lookup_vals, output_tensor

  def _create_table_init_from_file_qat_model_tf1(
      self, sess: session.Session
  ) -> Tuple[core.Tensor, core.Tensor, core.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Builds the graph for the QAT file-initialized table test model.
    # Same as the non-QAT file-initialized table graph (three-line vocabulary file, static hash
    # table with default minus one, 2x2 matmul with an all-ones weight) but with eight-bit fake-quant
    # ops on the matmul input (-0.3 to 0.3), the weight (-0.1 to 0.2) and the output (-0.2 to 0.2).
    # Input: sess. Returns placeholder, lookup values, output tensor.
    """Creates a simple QAT model that initializes a table from an asset file.

    This model creates an asset file at "vocab_file.txt" containing
    comma-separated vocabularies and uses it to initialize a
    `StaticVocabularyTable`. For inference, the model performs a lookup with a
    1D string tensor input vocabs.

    Args:
      sess: Tensorflow Session to create the model in.

    Returns:
      (input_vocabs_placeholder, lookup_vals, output_tensor), where
      * input_vocabs_placeholder is a placeholder tensor of 1D strings
      * lookup_vals is an output tensor that is a direct result of table lookup
      * output_tensor is a float 2x2 matrix
    """
    # Creates and populates an asset file.
    asset_dir = self.create_tempdir('assets').full_path
    asset_file = os.path.join(asset_dir, 'vocab_file.txt')
    content = '\n'.join(['static', 'range', 'quantization'])
    file_io.write_string_to_file(filename=asset_file, file_content=content)

    # The resulting table looks like:
    # "static" -> 0
    # "range" -> 1
    # "quantization" -> 2
    # default -> -1
    init = lookup_ops.TextFileInitializer(
        filename=asset_file,
        key_dtype=dtypes.string,
        key_index=lookup_ops.TextFileIndex.WHOLE_LINE,
        value_dtype=dtypes.int64,
        value_index=lookup_ops.TextFileIndex.LINE_NUMBER,
    )
    table = lookup_ops.StaticHashTable(init, default_value=-1)

    input_vocabs_placeholder = array_ops.placeholder(
        dtypes.string, shape=(None,), name='input_vocabs'
    )

    # Introduce a matmul op that takes the lookup values to observe the
    # effects of quantization.
    lookup_vals = math_ops.cast(
        table.lookup(input_vocabs_placeholder), dtypes.float32
    )
    # shape: (2, ?)
    matmul_input = array_ops_stack.stack([lookup_vals, lookup_vals])
    matmul_input = array_ops.fake_quant_with_min_max_args(
        matmul_input, min=-0.3, max=0.3, num_bits=8, narrow_range=False
    )

    # Create a dummy weight matrix filled with ones.
    weight_row = array_ops.ones(
        shape=array_ops.shape(input_vocabs_placeholder), dtype=dtypes.float32
    )
    # shape: (?, 2)
    weight = array_ops.transpose_v2(
        array_ops_stack.stack([weight_row, weight_row])
    )
    weight = array_ops.fake_quant_with_min_max_args(
        weight, min=-0.1, max=0.2, num_bits=8, narrow_range=False
    )

    # shape: (2, 2)
    output_tensor = math_ops.matmul(matmul_input, weight)
    output_tensor = array_ops.fake_quant_with_min_max_args(
        output_tensor, min=-0.2, max=0.2, num_bits=8, narrow_range=False
    )

    return input_vocabs_placeholder, lookup_vals, output_tensor

  def _create_and_save_file_init_hash_table_qat_model_tf1(
      self,
      output_path: str,
      tags: Collection[str],
      signature_def_key: str,
  ) -> Tuple[Mapping[str, core.Tensor], Mapping[str, core.Tensor]]:
    # [FUNCTION DOC]
    # HELPER: Creates and saves a TF1 QAT file-initialized hash table test model.
    # Inputs: output_path, tags, signature_def_key. Builds the QAT table-from-file model in a fresh
    # graph and session and saves it with the table initializer and asset collection. Returns inputs
    # (key "input_vocabs") and outputs (keys "lookup" and "output"). Writes files to disk.
    """Creates and saves a QAT model that uses a file-initialized table.

    The asset file "vocab_file.txt" is used to initialize a hash table.

    Args:
      output_path: Path to the directory to save the created model.
      tags: Set of strings that identifies the saved meta graph.
      signature_def_key: Name of the SignatureDef. Used to identify the
        SignatureDef within the meta graph.

    Returns:
      inputs: A mapping of input_key -> input_tensor (placeholder). The input
        key is "input_vocabs".
      outputs: A mapping of output_key -> output_tensor. The output keys are
        "lookup" and "output".
    """
    with session.Session(graph=ops.Graph()) as sess:
      input_vocabs_placeholder, lookup_tensor, output_tensor = (
          self._create_table_init_from_file_qat_model_tf1(sess)
      )

      inputs = {'input_vocabs': input_vocabs_placeholder}
      outputs = {
          'lookup': lookup_tensor,
          'output': output_tensor,
      }

      self._save_tf1_model(
          sess,
          output_path,
          signature_def_key,
          tags,
          inputs=inputs,
          outputs=outputs,
          init_op=lookup_ops.tables_initializer(),
          assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS),
      )

    return inputs, outputs

  def _create_data_generator(
      self,
      input_key: str,
      shape: Sequence[int],
      minval: float = -1.0,
      maxval: float = 1.0,
      dtype: dtypes.DType = dtypes.float32,
      num_examples: int = 8,
  ) -> repr_dataset.RepresentativeDataset:
    # [FUNCTION DOC]
    # HELPER: Produces representative calibration data for quantization tests.
    # KNOWN ISSUE: annotated as returning a dataset, but it is a generator function; the docstring
    # names the yielded item a data_gen, whereas each yielded item is a mapping from input_key to
    # a random uniform tensor.
    # Inputs: input_key (name of the input), shape, minval and maxval (defaults minus one and one),
    # dtype (default float32), num_examples (default 8). Yields num_examples mappings, each with a
    # freshly generated random tensor of the given shape and type in the given range.
    """Creates a data generator to be used as representative dataset.

    Supports generating random value input tensors mapped by the `input_key`.

    Args:
      input_key: The string key that identifies the created tensor as an input.
      shape: Shape of the tensor data.
      minval: The lower bound of the generated input
      maxval: The upper bound of the generated input
      dtype: The type of the generated input - usually dtypes.float32 for float
        and dtypes.int64 for int
      num_examples: Number of examples in the representative dataset.

    Yields:
      data_gen: A `quantize_model._RepresentativeSample` filled with random
        values.
    """
    for _ in range(num_examples):
      yield {input_key: random_ops.random_uniform(shape, minval, maxval, dtype)}

  def _save_tf1_model(
      self,
      sess: session.Session,
      saved_model_path: str,
      signature_key: str,
      tags: Collection[str],
      inputs: Mapping[str, core.Tensor],
      outputs: Mapping[str, core.Tensor],
      init_op: Optional[ops.Operation] = None,
      assets_collection: Optional[Sequence[core.Symbol]] = None,
  ) -> None:
    # [FUNCTION DOC]
    # HELPER: Saves TF1 sessions as saved models for the model-building helpers.
    # Inputs: sess (session holding the graph and variables), saved_model_path (output directory),
    # signature_key, tags, inputs and outputs (mappings from names to tensors), init_op (optional
    # main op), assets_collection (optional asset file tensors). Builds a predict signature,
    # adds the meta graph and variables with that signature, and writes the saved model.
    # Returns nothing; writes to disk.
    """Saves a TF1 model.

    Args:
      sess: Current tf.Session object.
      saved_model_path: Directory to save the model.
      signature_key: The key to the SignatureDef that inputs & outputs
        correspond to.
      tags: Set of tags associated with the model.
      inputs: Input name -> input tensor mapping.
      outputs: Output name -> output tensor mapping.
      init_op: Op for initialization.
      assets_collection: Assets collection. This collection is a list of string
        tensors. Each tensor contains the asset file names.
    """
    v1_builder = builder.SavedModelBuilder(saved_model_path)
    sig_def = signature_def_utils_impl.predict_signature_def(
        inputs=inputs, outputs=outputs
    )

    v1_builder.add_meta_graph_and_variables(
        sess,
        tags,
        signature_def_map={signature_key: sig_def},
        main_op=init_op,
        assets_collection=assets_collection,
    )
    v1_builder.save()

  def _create_simple_gather_and_conv_model(
      self,
      input_type: dtypes.DType,
      filter_shape: Sequence[int],
      is_qat_model: bool = False,
  ) -> module.Module:
    # [FUNCTION DOC]
    # HELPER: Factory for a gather-then-two-convolutions test module.
    # Inputs: input_type (dtype of the index input), filter_shape (shape of the first conv filter),
    # is_qat_model (default false; adds fake-quant ops). Defines a local module class and returns
    # a new instance of it. Nothing is saved here.
    class SimpleGatherAndConvModel(module.Module):
      # [CLASS DOC]
      # HELPER: Local test module used by the gather-and-conv factory.
      # Holds random constant weights: a clipped 1024x3x4x3 embedding table, and two conv filters.
      # Exposes a traced model function taking an index vector.
      """A simple model with a single gather and a conv2d."""

      def __init__(self):
        # [FUNCTION DOC]
        # Creates numpy weights: an embedding table of random normal values clipped to minus four
        # to four, a first conv filter uniform in minus ten to ten with the captured filter_shape,
        # and a second 3x3 filter whose input channels equal the last dimension of the first
        # filter, with one output channel. Takes no arguments, returns nothing.
        """Initializes a SimpleGatherAndConvModel."""
        self.embedding_w = np.random.randn(1024, 3, 4, 3).astype('f4')
        self.embedding_w = np.minimum(np.maximum(self.embedding_w, -4), 4)

        self.conv_filters = np.random.uniform(
            low=-10, high=10, size=filter_shape
        ).astype('f4')

        second_conv_filter_shape = (3, 3, filter_shape[-1], 1)
        self.second_conv_filters = np.random.uniform(
            low=-10, high=10, size=second_conv_filter_shape
        ).astype('f4')

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  shape=[None], dtype=input_type, name='input_tensor'
              )
          ]
      )
      def model(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # KNOWN ISSUE: the docstring says gather and a 2D convolution, but it performs two
        # convolutions after the gather.
        # Traced function with a fixed signature (rank-one tensor of the captured index dtype,
        # named input_tensor). Gathers embedding rows by the input, applies a same-padded conv with
        # the first filter, then a second conv with the second filter. In QAT mode inserts
        # eight-bit fake-quant ops after the first conv, on the second filter (narrow range) and on
        # the final output. Returns a mapping with the single key "output".
        """Performs a gather and a 2D convolution operation.

        Args:
          input_tensor: Input tensor to perform operation on.

        Returns:
          A map of: output key -> output result.
        """

        out = array_ops.gather_v2(self.embedding_w, input_tensor)

        # One pure conv
        out = nn_ops.conv2d(
            out,
            self.conv_filters,
            strides=(1, 1, 2, 1),
            dilations=(1, 1, 1, 1),
            padding='SAME',
            data_format='NHWC',
        )

        # One fakequant attached conv
        if is_qat_model:
          out = array_ops.fake_quant_with_min_max_args(
              out, min=-0.1, max=0.2, num_bits=8, narrow_range=False
          )
          second_conv_filters = array_ops.fake_quant_with_min_max_args(
              self.second_conv_filters,
              min=-0.1,
              max=0.2,
              num_bits=8,
              narrow_range=True,
          )
        else:
          second_conv_filters = self.second_conv_filters

        out = nn_ops.conv2d(
            out,
            second_conv_filters,
            strides=(1, 1, 2, 1),
            dilations=(1, 1, 1, 1),
            padding='SAME',
            data_format='NHWC',
        )
        if is_qat_model:
          out = array_ops.fake_quant_with_min_max_args(
              out, min=-0.1, max=0.2, num_bits=8, narrow_range=False
          )
        return {'output': out}

    return SimpleGatherAndConvModel()

  def _create_and_save_tf1_gather_model(
      self,
      saved_model_path: str,
      signature_key: str,
      tags: Collection[str],
      input_key: str,
      output_key: str,
      input_type: dtypes.DType,
      use_variable=False,
  ) -> core.Tensor:
    # [FUNCTION DOC]
    # HELPER: Creates and saves a TF1 gather test model.
    # Inputs: saved_model_path, signature_key, tags, input_key, output_key, input_type (index
    # dtype), use_variable (default false). In a new graph and session builds the gather model,
    # initializes variables if used, and saves it with the given keys. Returns the input
    # placeholder. Writes the saved model to disk.
    """Creates and saves a simple gather model.

    This is intended to be used for TF1 (graph mode) tests.

    Args:
      saved_model_path: Directory to save the model.
      signature_key: The key to the SignatureDef that inputs & outputs
        correspond to.
      tags: Set of tags associated with the model.
      input_key: The key to the input tensor.
      output_key: The key to the output tensor.
      input_type: type of the input index tensor for gather operation.
      use_variable: Setting this to `True` makes the filter for the gather
        operation a `tf.Variable`.

    Returns:
      in_placeholder: The placeholder tensor used as an input to the model.
    """
    with ops.Graph().as_default(), session.Session() as sess:
      in_placeholder, output_tensor = self._create_simple_tf1_gather_model(
          input_type=input_type, use_variable_for_filter=use_variable
      )

      if use_variable:
        sess.run(variables.global_variables_initializer())

      self._save_tf1_model(
          sess,
          saved_model_path,
          signature_key,
          tags,
          inputs={input_key: in_placeholder},
          outputs={output_key: output_tensor},
      )

      return in_placeholder

  def _create_gather_model(self, input_type, use_variable):
    # [FUNCTION DOC]
    # HELPER: Factory for a single-gather test model.
    # Inputs: input_type (dtype of the index input), use_variable (whether the weight is a
    # variable). Returns a new instance of a local trackable class. Nothing is saved.
    class GatherModel(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: Local test model used by the gather factory.
      # Holds a random 128x32 weight (variable or numpy array) and is callable with a length-six
      # index vector.
      """A simple model with a single gather."""

      def __init__(self, use_variable):
        # [FUNCTION DOC]
        # Calls the parent constructor and creates a random normal 128x32 float weight, stored as
        # a variable when use_variable is true, otherwise as a plain array. Returns nothing.
        """Initializes a GatherModel.

        Args:
          use_variable: If True, creates a variable for weight.
        """
        super(GatherModel, self).__init__()
        w_val = np.random.randn(128, 32).astype('f4')
        if use_variable:
          self.w = variables.Variable(w_val)
        else:
          self.w = w_val

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  shape=[6], dtype=input_type, name='input_tensor'
              )
          ]
      )
      def __call__(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced call with a fixed signature (length-six tensor named input_tensor). Gathers weight
        # rows by the indices and returns a mapping with the single key "output".
        """Performs a gather operation."""
        out = array_ops.gather_v2(self.w, input_tensor)
        return {'output': out}

    return GatherModel(use_variable)

  def _create_depthwise_conv2d_model(
      self,
      input_shape: Sequence[int],
      filter_shape: Sequence[int],
      has_bias: bool = False,
      has_batch_norm: bool = False,
      activation_fn: Optional[ops.Operation] = None,
      strides: Sequence[int] = (1, 2, 2, 1),
      dilations: Sequence[int] = (1, 1, 1, 1),
      padding: str = 'SAME',
  ):
    # [FUNCTION DOC]
    # HELPER: Factory for a depthwise-convolution test module.
    # KNOWN ISSUE: the strides, dilations and padding parameters are accepted but ignored; the
    # traced function hardcodes strides 1,2,2,1, dilations all ones and same padding.
    # Inputs: input_shape, filter_shape (height, width, in channels, multiplier), has_bias,
    # has_batch_norm (both default false), activation_fn (optional callable). Returns an instance
    # of a local module class; nothing is saved.
    class DepthwiseConvModel(module.Module):
      # [CLASS DOC]
      # HELPER: Local test module used by the depthwise factory.
      # Holds per-channel random filters and bias; exposes a traced depthwise conv function.
      """A simple model with a single depthwise conv2d, bias and relu."""

      def __init__(self):
        # [FUNCTION DOC]
        # Builds filters whose value range grows with the output channel index (channel i is
        # uniform in minus (i+1) to (i+1)), reshaped to the filter shape, plus a random bias in
        # zero to ten. The output channel count is the product of the last two filter dims.
        self.out_channel_size = filter_shape[2] * filter_shape[3]

        # This ensures filters will have different value range per out channel
        self.filters = np.stack(
            [
                np.random.uniform(
                    low=-(i + 1), high=(i + 1), size=filter_shape[:-2]
                ).astype('f4')
                for i in range(self.out_channel_size)
            ],
            axis=-1,
        )
        self.filters = self.filters.reshape(filter_shape)

        self.bias = np.random.uniform(
            low=0, high=10, size=(self.out_channel_size)
        ).astype('f4')

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32)
          ]
      )
      def depthwise_conv(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function (signature: float tensor of input_shape). Applies a depthwise conv with
        # fixed strides and same padding, then optionally a bias add, a non-training fused batch
        # norm, and the activation. Returns a mapping with the single key "output".
        """Performs a 2D depthwise convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        scale = [1.0] * self.out_channel_size
        offset = [0.5] * self.out_channel_size
        mean, variance = scale, offset
        out = nn_ops.depthwise_conv2d_native(
            input_tensor,
            self.filters,
            strides=[1, 2, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC',
        )
        if has_bias:
          out = nn_ops.bias_add(out, self.bias)
        if has_batch_norm:
          # Fusing is supported for non-training case.
          out, _, _, _, _, _ = nn_ops.fused_batch_norm_v3(
              out, scale, offset, mean, variance, is_training=False
          )
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    return DepthwiseConvModel()

  def _create_conv2d_model(
      self,
      input_shape: Sequence[int],
      filter_shape: Sequence[int],
      has_bias: bool = False,
      has_batch_norm: bool = False,
      activation_fn: Optional[ops.Operation] = None,
      strides: Sequence[int] = (1, 2, 2, 1),
      dilations: Sequence[int] = (1, 1, 1, 1),
      padding: str = 'SAME',
  ):
    # [FUNCTION DOC]
    # HELPER: Factory for a convolution test module.
    # Inputs: input_shape, filter_shape, has_bias, has_batch_norm (default false), activation_fn
    # (optional callable), strides, dilations, padding (defaults 1,2,2,1; all ones; same). Returns
    # an instance of a local module class; nothing is saved.
    class ConvModel(module.Module):
      # [CLASS DOC]
      # HELPER: Local test module used by the conv factory.
      # Holds per-channel random filters and bias; exposes a traced conv function.
      """A simple model with a single conv2d, bias and relu."""

      def __init__(self):
        # [FUNCTION DOC]
        # Builds filters whose value range grows with the output channel index (the last filter
        # dimension) and a random bias in zero to ten.
        self.out_channel_size = filter_shape[-1]

        # This ensures filters will have different value range per out channel
        self.filters = np.stack(
            [
                np.random.uniform(
                    low=-(i + 1), high=(i + 1), size=filter_shape[:-1]
                ).astype('f4')
                for i in range(self.out_channel_size)
            ],
            axis=-1,
        )

        self.bias = np.random.uniform(
            low=0, high=10, size=(self.out_channel_size)
        ).astype('f4')

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32)
          ]
      )
      def conv(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function (signature: float tensor of input_shape). Applies a channels-last conv
        # with the captured strides, dilations and padding, then optionally a bias add, a
        # non-training fused batch norm, and the activation. Returns a mapping with key "output".
        """Performs a 2D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        scale = [1.0] * self.out_channel_size
        offset = [0.5] * self.out_channel_size
        mean, variance = scale, offset
        out = nn_ops.conv2d(
            input_tensor,
            self.filters,
            strides=strides,
            dilations=dilations,
            padding=padding,
            data_format='NHWC',
        )
        if has_bias:
          out = nn_ops.bias_add(out, self.bias, data_format='NHWC')
        if has_batch_norm:
          # Fusing is supported for non-training case.
          out, _, _, _, _, _ = nn_ops.fused_batch_norm_v3(
              out, scale, offset, mean, variance, is_training=False
          )
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    return ConvModel()

  def _create_matmul_model(
      self,
      input_shape: Sequence[int],
      weight_shape: Sequence[int],
      saved_model_path: str,
      has_bias: bool = False,
      activation_fn: Optional[ops.Operation] = None,
      bias_size: Optional[int] = None,
      use_biasadd: bool = True,
  ) -> module.Module:
    # [FUNCTION DOC]
    # HELPER: Factory that builds and saves a matmul test model.
    # KNOWN ISSUE: use_biasadd is not forwarded to the model constructor (the default of true is
    # always used), and the activation is passed positionally in the slot after the bias size.
    # Inputs: input_shape, weight_shape, saved_model_path, has_bias (default false), activation_fn
    # (optional), bias_size (optional; defaults to the last weight dimension when has_bias is
    # set), use_biasadd. Asserts that bias size presence agrees with has_bias and that the bias
    # size divides the output element count (assertion errors otherwise). Creates the model, saves
    # it as a saved model with the matmul function's concrete signature, and returns the model.
    class MatmulModel(module.Module):
      # [CLASS DOC]
      # HELPER: Local test module used by the matmul factory.
      # Holds random uniform weights, an optional bias, an optional activation and a flag choosing
      # bias-add versus plain add. Exposes helper predicates and a traced matmul function.
      """A simple model with a single matmul.

      Bias and activation function are optional.
      """

      def __init__(
          self,
          weight_shape: Sequence[int],
          bias_size: Optional[int] = None,
          activation_fn: Optional[ops.Operation] = None,
          use_biasadd: bool = True,
      ) -> None:
        # [FUNCTION DOC]
        # Stores the bias size, activation and bias-add flag, creates random uniform weights in
        # minus one to one of weight_shape, and a random bias of bias_size if given. Returns nothing.
        """Initializes a MatmulModel.

        Args:
          weight_shape: Shape of the weight tensor.
          bias_size: If None, do not use bias. Else, use given size as bias.
          activation_fn: The activation function to be used. No activation
            function if None.
          use_biasadd: If True, use BiasAdd for adding bias, else use AddV2.
        """
        self.bias_size = bias_size
        self.activation_fn = activation_fn
        self.use_biasadd = use_biasadd
        self.filters = np.random.uniform(low=-1.0, high=1.0, size=weight_shape)

        if bias_size is not None:
          self.bias = np.random.uniform(low=-1.0, high=1.0, size=bias_size)

      def has_bias(self) -> bool:
        # [FUNCTION DOC]
        # Returns true when a bias size was provided.
        return self.bias_size is not None

      def has_reshape(self) -> bool:
        # [FUNCTION DOC]
        # Returns true when a bias exists and its size differs from the last weight dimension, in
        # which case the matmul output is reshaped before adding the bias.
        return self.has_bias() and self.bias_size != self.filters.shape[-1]

      @def_function.function
      def matmul(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function (no fixed signature). Multiplies the input by the weights (op named
        # sample/matmul), reshapes to rank two or three with the bias size as last dimension when
        # needed, adds the bias (bias-add or plain add per the flag) and applies the activation if
        # any. Returns a mapping with the single key "output".
        """Performs a matrix multiplication.

        Depending on self.has_bias and self.activation_fn, it may add a bias
        term or
        go through the activaction function.

        Args:
          input_tensor: Input tensor to matmul with the filter.

        Returns:
          A map of: output key -> output result.
        """
        out = math_ops.matmul(input_tensor, self.filters, name='sample/matmul')

        if self.has_reshape():
          input_shape = input_tensor.shape
          if len(input_shape) == 3:
            reshape_shape = (input_shape[0], -1, self.bias_size)
          else:
            reshape_shape = (-1, self.bias_size)

          out = array_ops.reshape(out, reshape_shape)

        if self.has_bias():
          if self.use_biasadd:
            out = nn_ops.bias_add(out, self.bias)
          else:
            out = math_ops.add_v2(out, self.bias)

        if self.activation_fn is not None:
          out = self.activation_fn(out)

        return {'output': out}

    # If bias_size is not explicitly given, it should default to width of weight.
    if bias_size is None and has_bias:
      bias_size = weight_shape[-1]

    # Verify that when bias_size is not None, has_bias should be True.
    # And if bias_size is None, has_bias should be False using XNOR
    assert not ((bias_size is not None) ^ has_bias)

    # Verify that bias size is correct
    if bias_size:
      input_height = input_shape[0] if len(input_shape) == 2 else input_shape[1]
      assert input_height * weight_shape[-1] % bias_size == 0

    model = MatmulModel(weight_shape, bias_size, activation_fn)
    saved_model_save.save(
        model,
        saved_model_path,
        signatures=model.matmul.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=input_shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
    )
    return model

  # Prepares sample einsum input data shapes.
  # This function returns:
  # 1. Shape for input 1
  # 2. Shape for input 2
  # 3. Shape for bias
  # 4. Signature for input 1 (Could contain None dimension)
  # 5. Signature for input 2 (Could contain None dimension)
  def _prepare_sample_einsum_datashapes(
      self,
      equation: str,
      generate_unknown_shape_signature: bool = False,
      use_bias: bool = False,
  ) -> Tuple[
      List[Optional[int]],
      List[Optional[int]],
      Optional[List[Optional[int]]],
      List[Optional[int]],
      List[Optional[int]],
  ]:
    # [FUNCTION DOC]
    # HELPER: Computes sample tensor shapes for einsum test models.
    # Inputs: equation (two-operand einsum text with a comma and an arrow), generate_unknown_shape_
    # signature (default false), use_bias (default false). Maps labels a,b,c,d,e to sizes 4,32,64,
    # 128,8 (an unknown label gives a null entry). Returns the first shape, second shape, bias shape
    # (last output label size only, or nothing), and the two signatures. When unknown signatures
    # are requested, dimensions not shared by both operands become null.
    # 1. Parse equation.
    comma_pos = equation.find(',')
    arrow_pos = equation.find('->')
    x_labels = equation[0:comma_pos]
    y_labels = equation[comma_pos + 1 : arrow_pos]
    out_labels = equation[arrow_pos + 1 :]

    # 2. Create sample shapes.
    label_to_size = {'a': 4, 'b': 32, 'c': 64, 'd': 128, 'e': 8}
    x_shape = [label_to_size.get(x_label) for x_label in x_labels]
    y_shape = [label_to_size.get(y_label) for y_label in y_labels]
    bias_shape = None
    if use_bias:
      bias_shape = [label_to_size.get(out_label) for out_label in out_labels]
      bias_shape = bias_shape[-1:]
    contracting_dims = set()

    x_signature = list(x_shape)
    y_signature = list(y_shape)
    if generate_unknown_shape_signature:
      for c in x_labels:
        if c in y_labels:
          contracting_dims.add(c)
      x_signature = [
          None if c not in contracting_dims else x_shape[cidx]
          for cidx, c in enumerate(x_labels)
      ]
      y_signature = [
          None if c not in contracting_dims else y_shape[cidx]
          for cidx, c in enumerate(y_labels)
      ]
    return x_shape, y_shape, bias_shape, x_signature, y_signature

  def _create_einsum_model(
      self,
      equation: str,
      y_shape: Sequence[int],
      x_signature: Sequence[Optional[int]],
      y_signature: Sequence[Optional[int]],
      bias_shape: Optional[Sequence[int]] = None,
      activation_fn: Optional[ops.Operation] = None,
      is_qat_model: bool = False,
  ) -> module.Module:
    # [FUNCTION DOC]
    # HELPER: Factory for an einsum test module.
    # Inputs: equation, y_shape (kernel shape), x_signature and y_signature (input specs, may
    # contain nulls), bias_shape (optional), activation_fn (optional), is_qat_model (default false,
    # adds fake-quant ops). Returns an instance of a local module class; nothing is saved.
    class EinsumModel(module.Module):
      # [CLASS DOC]
      # HELPER: Local test module used by the einsum factory.
      # Holds an optional bias, a random kernel and fake-quant min/max ranges; exposes traced
      # functions computing the einsum with the stored kernel or with a second input.
      """Einsum class."""

      def __init__(self):
        # [FUNCTION DOC]
        # Creates an optional constant bias (random uniform) when bias_shape is given, a random
        # float kernel of y_shape, and fixed min and max triples used by QAT fake-quant ops.
        self._bias = None
        if bias_shape is not None:
          self._bias = array_ops.constant(
              np.random.uniform(size=bias_shape), dtype=dtypes.float32
          )

        self._kernel = np.random.uniform(size=y_shape).astype('f4')
        self._min = (-0.8, -0.8, -0.9)
        self._max = (0.9, 0.9, 1.0)

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='x', shape=x_signature, dtype=dtypes.float32
              )
          ]
      )
      def einsum_with_kernel(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function taking one float input x; computes the einsum of x with the stored kernel
        # and returns a mapping with key "output".
        return self._einsum(x, self._kernel)

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='x', shape=x_signature, dtype=dtypes.float32
              ),
              tensor_spec.TensorSpec(
                  name='y', shape=y_signature, dtype=dtypes.float32
              ),
          ]
      )
      def einsum_without_kernel(
          self, x: core.Tensor, y: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function taking two float inputs x and y; computes the einsum of both and returns
        # a mapping with key "output".
        return self._einsum(x, y)

      def _einsum(self, x, y):
        # [FUNCTION DOC]
        # HELPER: Shared body of the two traced einsum functions.
        # Inputs: x and y tensors. In QAT mode applies eight-bit fake-quant with variable ranges to
        # both, runs the einsum with the captured equation, optionally adds the bias and applies
        # the activation, and in QAT mode fake-quants the result. Returns a mapping with "output".
        if is_qat_model:
          x = array_ops.fake_quant_with_min_max_vars(
              x,
              min=ops.convert_to_tensor(self._min[0]),
              max=ops.convert_to_tensor(self._max[0]),
              num_bits=8,
              narrow_range=False,
          )
          y = array_ops.fake_quant_with_min_max_vars(
              y,
              min=ops.convert_to_tensor(self._min[1]),
              max=ops.convert_to_tensor(self._max[1]),
              num_bits=8,
              narrow_range=False,
          )

        out = tensorflow.einsum(equation, x, y)
        if self._bias is not None:
          out = nn_ops.bias_add(out, self._bias)
        if activation_fn is not None:
          out = activation_fn(out)
        if is_qat_model:
          out = array_ops.fake_quant_with_min_max_vars(
              out,
              min=ops.convert_to_tensor(self._min[2]),
              max=ops.convert_to_tensor(self._max[2]),
              num_bits=8,
              narrow_range=False,
          )
        return {'output': out}

    return EinsumModel()

  def _create_and_save_tf1_conv_model(
      self,
      saved_model_path: str,
      signature_key: str,
      tags: Collection[str],
      input_key: str,
      output_key: str,
      *,
      input_shape: Sequence[int] = (1, 3, 4, 3),
      filter_shape: Sequence[int] = (2, 3, 3, 2),
      use_variable: bool = False,
  ) -> core.Tensor:
    # [FUNCTION DOC]
    # HELPER: Creates and saves a TF1 convolution test model.
    # Inputs: saved_model_path, signature_key, tags, input_key, output_key, and keyword-only
    # input_shape, filter_shape and use_variable (default false). In a new graph and session
    # builds the conv model, initializes variables if used, and saves it. Returns the input
    # placeholder. Writes the saved model to disk.
    """Creates and saves a simple convolution model.

    This is intended to be used for TF1 (graph mode) tests.

    Args:
      saved_model_path: Directory to save the model.
      signature_key: The key to the SignatureDef that inputs & outputs
        correspond to.
      tags: Set of tags associated with the model.
      input_key: The key to the input tensor.
      output_key: The key to the output tensor.
      input_shape: Shape of the input tensor.
      filter_shape: Shape of the filter.
      use_variable: Setting this to `True` makes the filter for the conv
        operation a `tf.Variable`.

    Returns:
      in_placeholder: The placeholder tensor used as an input to the model.
    """
    with ops.Graph().as_default(), session.Session() as sess:
      in_placeholder, output_tensor = self._create_simple_tf1_conv_model(
          input_shape=input_shape,
          filter_shape=filter_shape,
          use_variable_for_filter=use_variable,
      )

      if use_variable:
        sess.run(variables.global_variables_initializer())

      self._save_tf1_model(
          sess,
          saved_model_path,
          signature_key,
          tags,
          inputs={input_key: in_placeholder},
          outputs={output_key: output_tensor},
      )

    return in_placeholder

  def _create_while_model(self, input_shape: Sequence[int] = (1, 32, 32, 512)):
    # [FUNCTION DOC]
    # HELPER: Factory for a test module containing a while loop.
    # Input: input_shape (default 1x32x32x512). Returns an instance of a local module class.
    class WhileModel(module.Module):
      # [CLASS DOC]
      # HELPER: Local test module used by the while-model factory.
      # Holds a random 3x3 conv weight with equal in and out channels (last input dimension) and
      # traced functions for the loop condition, body and the main entry point.
      """A model with a while op."""

      def __init__(self):
        # [FUNCTION DOC]
        # Creates the conv weight: uniform in minus two to two, shape 3x3xCxC where C is the last
        # input dimension.
        w_shape = [3, 3] + [input_shape[-1], input_shape[-1]]
        self.w = np.random.uniform(low=-2, high=2, size=w_shape).astype('f4')

      @def_function.function
      def condition(self, x, w):
        # [FUNCTION DOC]
        # Loop condition: returns whether the sum of all elements of x is below one hundred. The
        # weight argument is unused.
        return math_ops.reduce_sum(x, keepdims=False) < 100

      @def_function.function
      def body(self, x, w):
        # [FUNCTION DOC]
        # Loop body: applies a same-padded convolution of x with w and returns the result together
        # with the unchanged weight.
        z = nn_ops.conv2d(x, w, padding='SAME')
        return z, w

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  shape=input_shape, dtype=dtypes.float32, name='input_tensor'
              )
          ]
      )
      def main(self, x):
        # [FUNCTION DOC]
        # Entry point (signature: float tensor of input_shape named input_tensor). Convolves x with
        # the weight, runs a while loop of the condition and body on x and the weight, and returns
        # a mapping with key "output" holding the sum of the first conv result and the loop result.
        x1 = nn_ops.conv2d(x, self.w, padding='SAME')
        x2, _ = while_loop_ops.while_loop(
            self.condition, self.body, [x, self.w]
        )
        result = x1 + x2
        return {'output': result}

    return WhileModel()
