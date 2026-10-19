# [MODULE DOC]
# Sanity tests for the TFLite python conversion layer. Covers graph definition conversion
# (plain, quantized, and invalid quantization arguments), the retry behavior when the converter
# reports a stateful-partitioned-call error, and the operation hint to stub transformation
# (hint indices, tags, aggregation, hinted output discovery). Includes two module-level mock
# helpers that replace the native converter and error retrieval. Run via the TensorFlow test main.
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
"""TensorFlow Lite Python Interface: Sanity check."""
from unittest import mock

import numpy as np

from tensorflow.compiler.mlir.lite import converter_flags_pb2 as _conversion_flags_pb2
from tensorflow.compiler.mlir.lite.metrics import converter_error_data_pb2
from tensorflow.lite.python import convert
from tensorflow.lite.python import op_hint
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.framework.graph_util_impl import _node_name
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _mock_wrapped_convert(
    unused_model_flags_str="",
    conversion_flags_str="",
    unused_input_data_str="",
    unused_debug_info_str="",
):
  # [FUNCTION DOC]
  # HELPER: fake replacement for the native wrapped converter, used with a patch in the retry
  # test. Takes model flags, conversion flags (serialized message string), input data and debug
  # info strings (all but the conversion flags ignored). Parses the conversion flags; if the
  # flag guaranteeing all functions have one use is not set it raises a generic exception,
  # otherwise returns the bytes of a fake model text.
  # Simulate the converter throwing and error when
  # `guarantee_all_funcs_one_use` is not set.
  if not _conversion_flags_pb2.ConverterFlags.FromString(
      conversion_flags_str
  ).guarantee_all_funcs_one_use:
    raise Exception()
  else:
    return bytes("A model", encoding="utf-8")


def _mock_retrieve_errors():
  # [FUNCTION DOC]
  # HELPER: fake replacement for the metrics wrapper's collected-error retrieval. Takes no
  # arguments and returns a list holding one converter error record with the error code for a
  # stateful partitioned call remaining in the final IR, which triggers the converter retry.
  err_data = converter_error_data_pb2.ConverterErrorData(
      error_code=converter_error_data_pb2.ConverterErrorData.ERROR_STATEFUL_PARTITIONED_CALL_IN_FINAL_IR
  )
  return [err_data]


class ConvertTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the graph definition conversion functions of the convert module: basic
  # conversion, retry on a specific converter error, quantization and argument validation.

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: builds a graph adding a 1x16x16x3 float input to itself and converts the graph
    # definition. Expects a non-empty model.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32
      )
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    # Try running on valid graph
    tflite_model = convert.convert_graphdef(
        sess.graph_def, input_tensors=[in_tensor], output_tensors=[out_tensor]
    )
    self.assertTrue(tflite_model)

  @mock.patch.object(
      convert.wrap_converter, "wrapped_convert", new=_mock_wrapped_convert
  )
  @mock.patch.object(
      metrics_wrapper, "retrieve_collected_errors", new=_mock_retrieve_errors
  )
  # This test wants to check that in the case of the converter throwing an
  # `ERROR_STATEFUL_PARTITIONED_CALL_IN_FINAL_IR` error, it will
  # retry conversion with the `guarantee_all_funcs_one_use` flag.
  # We can wrap the convert call in order to assert it is called appropriately.
  @mock.patch.object(convert, "convert", wraps=convert.convert)
  def testConversionStatefulPartitionRetry(self, mock_convert):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the model-content check uses the truthiness assertion with the expected text
    # as the failure message, so it passes for any non-empty value and does not compare content.
    # Scenario: the native converter and error retrieval are patched with fakes (the fake
    # converter fails unless the one-use-functions flag is set; retrieved errors report a
    # stateful partitioned call). The convert function is wrapped to count calls (mock_convert).
    # Converts a simple add graph with that flag false. Expects conversion to succeed via a
    # retry, with the convert function called exactly twice.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32
      )
      out_tensor = in_tensor + in_tensor
      sess = session.Session()

    model = convert.convert_graphdef(
        sess.graph_def,
        input_tensors=[in_tensor],
        output_tensors=[out_tensor],
        guarantee_all_funcs_one_use=False,
    )
    self.assertTrue(str(model, encoding="utf-8"), "A model")
    self.assertEqual(mock_convert.call_count, 2)

  def testQuantization(self):
    # [FUNCTION DOC]
    # Scenario: graph with a fake-quantization node (range 0 to 1) after an add, converted with
    # unsigned 8-bit inference type and one input stat pair (mean 0, std 1). Expects a non-empty
    # model.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32
      )
      out_tensor = array_ops.fake_quant_with_min_max_args(
          in_tensor + in_tensor, min=0.0, max=1.0
      )
      sess = session.Session()

    tflite_model = convert.convert_graphdef(
        sess.graph_def,
        input_tensors=[in_tensor],
        output_tensors=[out_tensor],
        inference_type=dtypes.uint8,
        quantized_input_stats=[(0.0, 1.0)],
    )
    self.assertTrue(tflite_model)

  def testGraphDefBasic(self):
    # [FUNCTION DOC]
    # Scenario: add graph converted by array names (input with shape, output add, float32).
    # Loads the result in the interpreter and expects one input named input, float32, shape
    # 1x16x16x3, quantization (0,0); and one output named add with the same type, shape, and
    # quantization.
    with ops.Graph().as_default():
      in_tensor = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name="input"
      )
      _ = in_tensor + in_tensor
      sess = session.Session()

    tflite_model = convert.convert_graphdef_with_arrays(
        sess.graph_def,
        input_arrays_with_shape=[("input", [1, 16, 16, 3])],
        output_arrays=["add"],
        control_output_arrays=None,
        inference_type=dtypes.float32,
    )
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(1, len(input_details))
    self.assertEqual("input", input_details[0]["name"])
    self.assertEqual(np.float32, input_details[0]["dtype"])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]["shape"]).all())  # type: ignore
    self.assertEqual((0.0, 0.0), input_details[0]["quantization"])

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual("add", output_details[0]["name"])
    self.assertEqual(np.float32, output_details[0]["dtype"])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]["shape"]).all())  # type: ignore
    self.assertEqual((0.0, 0.0), output_details[0]["quantization"])

  def testGraphDefQuantization(self):
    # [FUNCTION DOC]
    # Scenario: two inputs added and passed through fake quantization (range 0 to 1), converted
    # by array names with unsigned 8-bit inference and input stats (0,1) for each. Expects two
    # uint8 inputs inputA and inputB of shape 1x16x16x3 with quantization (scale 1, zero point 0),
    # and one uint8 output named output with the same shape and a positive scale.
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name="inputA"
      )
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name="inputB"
      )
      _ = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0.0, max=1.0, name="output"
      )
      sess = session.Session()

    tflite_model = convert.convert_graphdef_with_arrays(
        sess.graph_def,
        input_arrays_with_shape=[
            ("inputA", [1, 16, 16, 3]),
            ("inputB", [1, 16, 16, 3]),
        ],
        output_arrays=["output"],
        control_output_arrays=None,
        inference_type=dtypes.uint8,
        quantized_input_stats=[(0.0, 1.0), (0.0, 1.0)],
    )
    self.assertTrue(tflite_model)

    # Check values from converted model.
    interpreter = Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    self.assertEqual(2, len(input_details))
    self.assertEqual("inputA", input_details[0]["name"])
    self.assertEqual(np.uint8, input_details[0]["dtype"])
    self.assertTrue(([1, 16, 16, 3] == input_details[0]["shape"]).all())  # type: ignore
    self.assertEqual(
        (1.0, 0.0), input_details[0]["quantization"]
    )  # scale, zero_point

    self.assertEqual("inputB", input_details[1]["name"])
    self.assertEqual(np.uint8, input_details[1]["dtype"])
    self.assertTrue(([1, 16, 16, 3] == input_details[1]["shape"]).all())  # type: ignore
    self.assertEqual(
        (1.0, 0.0), input_details[1]["quantization"]
    )  # scale, zero_point

    output_details = interpreter.get_output_details()
    self.assertEqual(1, len(output_details))
    self.assertEqual("output", output_details[0]["name"])
    self.assertEqual(np.uint8, output_details[0]["dtype"])
    self.assertTrue(([1, 16, 16, 3] == output_details[0]["shape"]).all())  # type: ignore
    self.assertGreater(output_details[0]["quantization"][0], 0)  # scale

  def testGraphDefQuantizationInvalid(self):
    # [FUNCTION DOC]
    # Scenario: same two-input quantized graph, converted with unsigned 8-bit inference type but
    # without input quantization stats. Expects a value error saying the quantized input stats
    # flag must be defined when the inference type or input type is 8-bit integer.
    with ops.Graph().as_default():
      in_tensor_1 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name="inputA"
      )
      in_tensor_2 = array_ops.placeholder(
          shape=[1, 16, 16, 3], dtype=dtypes.float32, name="inputB"
      )
      _ = array_ops.fake_quant_with_min_max_args(
          in_tensor_1 + in_tensor_2, min=0.0, max=1.0, name="output"
      )
      sess = session.Session()

    with self.assertRaises(ValueError) as error:
      convert.convert_graphdef_with_arrays(
          sess.graph_def,
          input_arrays_with_shape=[
              ("inputA", [1, 16, 16, 3]),
              ("inputB", [1, 16, 16, 3]),
          ],
          output_arrays=["output"],
          control_output_arrays=None,
          inference_type=dtypes.uint8,
      )
    self.assertEqual(
        "The `quantized_input_stats` flag must be defined when either "
        "`inference_type` flag or `inference_input_type` flag is set to "
        "tf.int8 or tf.uint8.",
        str(error.exception),
    )


class ConvertTestOpHint(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for operation hints: wrapping inputs and outputs of a user function with hint
  # identity nodes and converting hinted regions into single stub operations. Provides helpers
  # to count identities, collect reachable op types and read hint index attributes.
  """Test the hint to stub functionality."""

  def _getGraphOpTypes(self, graphdef, output_nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Given a graph definition and output node names, returns the set
    # of operation type strings of all nodes reachable backward from those outputs.
    """Returns used op types in `graphdef` reachable from `output_nodes`.

    This is used to check that after the stub transformation the expected
    nodes are there.

    NOTE: this is not a exact test that the graph is the correct output, but
      it balances compact expressibility of test with sanity checking.

    Args:
      graphdef: TensorFlow proto graphdef.
      output_nodes: A list of output node names that we need to reach.

    Returns:
      A set of node types reachable from `output_nodes`.
    """
    name_to_input_name, name_to_node, _ = _extract_graph_summary(graphdef)
    # Find all nodes that are needed by the outputs
    used_node_names = _bfs_for_reachable_nodes(output_nodes, name_to_input_name)
    return set([name_to_node[node_name].op for node_name in used_node_names])

  def _countIdentities(self, nodes):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Returns how many nodes in the given node list have operation
    # type Identity.
    """Count the number of "Identity" op types in the list of proto nodes.

    Args:
      nodes: NodeDefs of the graph.

    Returns:
      The number of nodes with op type "Identity" found.
    """
    return len([x for x in nodes if x.op == "Identity"])

  def testSwishLiteHint(self):
    # [FUNCTION DOC]
    # Scenario: a swish-like activation built from a sigmoid, multiply and a scale, with its two
    # inputs and one output wrapped in a named hint, followed by a final identity. Expects 4
    # identity nodes in the graph, and after converting hints to stubs, the reachable op types
    # from the final output are exactly the hint name, constant and identity.
    # Nested helper builds the hinted activation.
    """Makes a custom op swish and makes sure it gets converted as a unit."""
    with ops.Graph().as_default():
      image = array_ops.constant([1.0, 2.0, 3.0, 4.0])
      swish_scale = array_ops.constant(1.0)

      def _swish(input_tensor, scale):
        # [FUNCTION DOC]
        # HELPER: nested builder. Marks the input tensor and scale as hint inputs, computes
        # sigmoid of the input times input times scale, marks the result as the hint output and
        # returns it.
        custom = op_hint.OpHint("cool_activation")
        input_tensor, scale = custom.add_inputs(input_tensor, scale)
        output = math_ops.sigmoid(input_tensor) * input_tensor * scale
        (output,) = custom.add_outputs(output)
        return output

      output = array_ops.identity(
          _swish(image, swish_scale), name="ModelOutput"
      )

      with self.cached_session() as sess:
        # check if identities have been put into the graph (2 input, 1 output,
        # and 1 final output).
        self.assertEqual(self._countIdentities(sess.graph_def.node), 4)

        stubbed_graphdef = op_hint.convert_op_hints_to_stubs(
            graph_def=sess.graph_def
        )

        self.assertEqual(
            self._getGraphOpTypes(
                stubbed_graphdef,
                output_nodes=[op_hint._tensor_name_base(output.name)],
            ),
            set(["cool_activation", "Const", "Identity"]),
        )

  def testScaleAndBiasAndIdentity(self):
    # [FUNCTION DOC]
    # Scenario: a hinted function with three inputs (scalar, vector, vector) and two outputs
    # (a times x plus b, and x), followed by a final identity. Expects 6 identity nodes (3 + 2 +
    # 1), and after stub conversion the reachable op types are the hint name, constant, identity
    # and pack.
    """This tests a scaled add which has 3 inputs and 2 outputs."""
    with ops.Graph().as_default():
      a = array_ops.constant(1.0)
      x = array_ops.constant([2.0, 3.0])
      b = array_ops.constant([4.0, 5.0])

      def _scaled_and_bias_and_identity(a, x, b):
        # [FUNCTION DOC]
        # HELPER: nested builder. Registers a, x, b as hint inputs and returns the hint outputs
        # for the expression a times x plus b and for x itself.
        custom = op_hint.OpHint("scale_and_bias_and_identity")
        a, x, b = custom.add_inputs(a, x, b)
        return custom.add_outputs(a * x + b, x)

      output = array_ops.identity(
          _scaled_and_bias_and_identity(a, x, b), name="ModelOutput"
      )

      with self.cached_session() as sess:
        # make sure one identity for each input (3) and output (2) => 3 + 2 = 5
        # +1 for the final output
        self.assertEqual(self._countIdentities(sess.graph_def.node), 6)

        stubbed_graphdef = op_hint.convert_op_hints_to_stubs(
            graph_def=sess.graph_def
        )

        self.assertEqual(
            self._getGraphOpTypes(
                stubbed_graphdef,
                output_nodes=[op_hint._tensor_name_base(output.name)],
            ),
            set(["scale_and_bias_and_identity", "Const", "Identity", "Pack"]),
        )

  def testTwoFunctions(self):
    # [FUNCTION DOC]
    # Scenario: the same hinted squaring function applied to two different constants and the
    # results added, then a final identity. Expects 5 identity nodes before conversion and, after
    # stub conversion, reachable op types of the hint name, constant, identity and add.
    """Tests if two functions are converted correctly."""
    with ops.Graph().as_default():
      a = array_ops.constant([1.0])
      b = array_ops.constant([1.0])

      def _double_values(x):
        # [FUNCTION DOC]
        # HELPER: nested builder. Wraps its single input and its output of a multiply of x by
        # itself in a hint named add_test and returns the hinted output.
        custom = op_hint.OpHint("add_test")
        (x,) = custom.add_inputs(x)
        output = math_ops.multiply(x, x)
        (output,) = custom.add_outputs(output)
        return output

      output = array_ops.identity(
          math_ops.add(_double_values(a), _double_values(b)), name="ModelOutput"
      )

      with self.cached_session() as sess:
        # make sure one identity for each input (2) and output (2) => 2 + 2
        # +1 for the final output
        self.assertEqual(self._countIdentities(sess.graph_def.node), 5)
        stubbed_graphdef = op_hint.convert_op_hints_to_stubs(
            graph_def=sess.graph_def
        )
        self.assertEqual(
            self._getGraphOpTypes(
                stubbed_graphdef,
                output_nodes=[op_hint._tensor_name_base(output.name)],
            ),
            set(["add_test", "Const", "Identity", "AddV2"]),
        )

  def _get_input_index(self, x):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Returns the integer hint input index attribute stored on the
    # operation that produced tensor x.
    return x.op.node_def.attr[op_hint.OpHint.FUNCTION_INPUT_INDEX_ATTR].i

  def _get_output_index(self, x):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Returns the integer hint output index attribute of the operation
    # that produced tensor x. Currently unused by the tests in this file.
    return x.op.node_def.attr[op_hint.OpHint.FUNCTION_OUTPUT_INDEX_ATTR].i

  def _get_sort_index(self, x):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Returns the integer hint sort index attribute (position within
    # an aggregated group) of the operation that produced tensor x.
    return x.op.node_def.attr[op_hint.OpHint.FUNCTION_SORT_INDEX_ATTR].i

  def testTags(self):
    # [FUNCTION DOC]
    # Scenario: one hint with inputs a and c under the same tag and stack aggregation, b without
    # a tag, and d under a second tag. Expects a to have input index 0 and sort index 0, b input
    # index 1 and sort index 0, c input index 0 and sort index 1, i.e. same-tag inputs share an
    # index and are ordered by sort index.
    """Test if multiple args with the same tag are grouped."""
    with ops.Graph().as_default():
      a = array_ops.constant([1.0])
      b = array_ops.constant([2.0])
      c = array_ops.constant([3.0])
      d = array_ops.constant([4.0])
      custom = op_hint.OpHint("test_tag")
      a = custom.add_input(
          a, tag="mytag", aggregate=op_hint.OpHint.AGGREGATE_STACK
      )
      (b,) = custom.add_inputs(b)
      c = custom.add_input(
          c, tag="mytag", aggregate=op_hint.OpHint.AGGREGATE_STACK
      )
      d = custom.add_input(
          d, tag="mytag2", aggregate=op_hint.OpHint.AGGREGATE_STACK
      )
      res = math_ops.add(math_ops.mul(a, b), math_ops.mul(c, b))
      custom.add_outputs([res])
      with self.cached_session():
        self.assertEqual(self._get_input_index(a), 0)
        self.assertEqual(self._get_sort_index(a), 0)
        self.assertEqual(self._get_input_index(b), 1)
        self.assertEqual(self._get_sort_index(b), 0)
        self.assertEqual(self._get_input_index(c), 0)
        self.assertEqual(self._get_sort_index(c), 1)

  def testOverrideIndex(self):
    # [FUNCTION DOC]
    # Scenario: adds hint inputs b (auto index), a (explicit index override 1) and c (auto).
    # Expects indices a=1, b=0, c=2, i.e. automatic assignment skips the overridden slot.
    with ops.Graph().as_default():
      a = array_ops.constant([1.0])
      b = array_ops.constant([2.0])
      c = array_ops.constant([3.0])
      custom = op_hint.OpHint("test_override")
      b = custom.add_input(b)  # should auto assign 0
      a = custom.add_input(a, index_override=1)
      c = custom.add_input(c)  # should auto assign 2
      with self.cached_session():
        self.assertEqual(self._get_input_index(a), 1)
        self.assertEqual(self._get_input_index(b), 0)
        self.assertEqual(self._get_input_index(c), 2)

  def testAggregate(self):
    # [FUNCTION DOC]
    # Scenario: two vectors are unstacked and elements are added to inputs of one hint with two
    # tags and stack aggregation; the two sums are hinted outputs with one tag and stack
    # aggregation, then restacked and passed through a final identity. After converting hints to
    # stubs, expects reachable op types of exactly the hint name, constant and identity.
    with ops.Graph().as_default():
      a = array_ops.constant([3.0, 4.0])
      b = array_ops.constant([5.0, 6.0])
      hint = op_hint.OpHint("agg")
      a0, a1 = array_ops_stack.unstack(a)
      b0, b1 = array_ops_stack.unstack(b)

      a0 = hint.add_input(a0, tag="c", aggregate=op_hint.OpHint.AGGREGATE_STACK)
      b0 = hint.add_input(b0, tag="n", aggregate=op_hint.OpHint.AGGREGATE_STACK)
      a1 = hint.add_input(a1, tag="c", aggregate=op_hint.OpHint.AGGREGATE_STACK)
      b1 = hint.add_input(b1, tag="n", aggregate=op_hint.OpHint.AGGREGATE_STACK)

      c0 = math_ops.add(a0, b0, name="addleft")
      c1 = math_ops.add(a1, b1, name="addright")
      c0 = hint.add_output(
          c0, tag="out", aggregate=op_hint.OpHint.AGGREGATE_STACK
      )
      c1 = hint.add_output(
          c1, tag="out", aggregate=op_hint.OpHint.AGGREGATE_STACK
      )

      curr = array_ops_stack.stack([c0, c1])
      output = array_ops.identity(curr, name="FINAL_OUTPUT")
      with self.cached_session() as sess:
        stubbed_graphdef = op_hint.convert_op_hints_to_stubs(
            graph_def=sess.graph_def
        )
        self.assertEqual(
            self._getGraphOpTypes(
                stubbed_graphdef,
                output_nodes=[op_hint._tensor_name_base(output.name)],
            ),
            set(["agg", "Const", "Identity"]),
        )

  def testFindHintedOutputNodes(self):
    # [FUNCTION DOC]
    # Scenario: two separately named hinted multiply ops are built in one graph. Searching the
    # session for hinted output nodes must return as many nodes as the two outputs (only the
    # count is compared, not the node names).
    """Test if all hinted output nodes are correctly found."""
    with ops.Graph().as_default():

      def _build_ophinted_op(name, input1, input2):
        # [FUNCTION DOC]
        # HELPER: nested builder. Creates a hint with the given name, registers both inputs,
        # multiplies them and returns the registered hint output.
        custom_op = op_hint.OpHint(name)
        input1 = custom_op.add_input(input1)
        input2 = custom_op.add_input(input2)
        output = math_ops.mul(input1, input2)
        return custom_op.add_output(output)

      output_1 = _build_ophinted_op(
          "custom_op_1", array_ops.constant([1.0]), array_ops.constant([2.0])
      )
      output_2 = _build_ophinted_op(
          "custom_op_2", array_ops.constant([3.0]), array_ops.constant([4.0])
      )
      with self.cached_session() as sess:
        hinted_outputs_nodes = op_hint.find_all_hinted_output_nodes(sess)
        expected_hinted_output_nodes = [
            _node_name(output_1.name),
            _node_name(output_2.name),
        ]
        self.assertEqual(
            len(hinted_outputs_nodes), len(expected_hinted_output_nodes)
        )


if __name__ == "__main__":
  test.main()
