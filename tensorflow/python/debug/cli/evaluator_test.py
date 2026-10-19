# [MODULE DOC]
# Unit tests for the tfdbg expression evaluator module. Covers parsing of debug tensor names
# (with or without device prefix, debug op suffix and execution index, plus malformed names) and
# evaluation of expressions containing backtick-quoted tensor references against a mocked debug
# dump. No import-time side effects; runs the TensorFlow test main when executed directly.
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
"""Tests for arbitrary expression evaluator."""
import numpy as np

from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class ParseDebugTensorNameTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the private debug tensor name parser of the evaluator module. Pure string parsing,
  # no shared fixtures.

  def testParseNamesWithoutPrefixOrSuffix(self):
    # [FUNCTION DOC]
    # Parses names with only node name and output slot. Expects no device name, the given node
    # and slot, the default identity debug op and execution index zero.
    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name("foo:1"))
    self.assertIsNone(device_name)
    self.assertEqual("foo", node_name)
    self.assertEqual(1, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(0, exec_index)

    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name("hidden_0/Weights:0"))
    self.assertIsNone(device_name)
    self.assertEqual("hidden_0/Weights", node_name)
    self.assertEqual(0, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(0, exec_index)

  def testParseNamesWithoutPrefixWithDebugOpSuffix(self):
    # [FUNCTION DOC]
    # Parses names with a debug op suffix but no device prefix. Expects no device name, the
    # explicit debug op returned, and execution index zero.
    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name("foo:1:DebugNanCount"))
    self.assertIsNone(device_name)
    self.assertEqual("foo", node_name)
    self.assertEqual(1, output_slot)
    self.assertEqual("DebugNanCount", debug_op)
    self.assertEqual(0, exec_index)

    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name(
            "hidden_0/Weights:0:DebugNumericSummary"))
    self.assertIsNone(device_name)
    self.assertEqual("hidden_0/Weights", node_name)
    self.assertEqual(0, output_slot)
    self.assertEqual("DebugNumericSummary", debug_op)
    self.assertEqual(0, exec_index)

  def testParseNamesWithDeviceNamePrefixWithoutDebugOpSuffix(self):
    # [FUNCTION DOC]
    # Parses names with a full device prefix but no debug op. Expects the device name returned
    # without its trailing colon, the node and slot, the default identity debug op and index zero.
    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name(
            "/job:ps/replica:0/task:2/cpu:0:foo:1"))
    self.assertEqual("/job:ps/replica:0/task:2/cpu:0", device_name)
    self.assertEqual("foo", node_name)
    self.assertEqual(1, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(0, exec_index)

    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name(
            "/job:worker/replica:0/task:3/gpu:0:hidden_0/Weights:0"))
    self.assertEqual("/job:worker/replica:0/task:3/gpu:0", device_name)
    self.assertEqual("hidden_0/Weights", node_name)
    self.assertEqual(0, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(0, exec_index)

  def testParseNamesWithDeviceNamePrefixWithDebugOpSuffix(self):
    # [FUNCTION DOC]
    # Parses names carrying both a device prefix and a debug op suffix (one given as a single
    # string, one split across concatenated literals). Expects all parts extracted and index zero.
    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name(
            "/job:ps/replica:0/task:2/cpu:0:foo:1:DebugNanCount"))
    self.assertEqual("/job:ps/replica:0/task:2/cpu:0", device_name)
    self.assertEqual("foo", node_name)
    self.assertEqual(1, output_slot)
    self.assertEqual("DebugNanCount", debug_op)
    self.assertEqual(0, exec_index)

    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name(
            "/job:worker/replica:0/task:3/gpu:0:"
            "hidden_0/Weights:0:DebugNumericSummary"))
    self.assertEqual("/job:worker/replica:0/task:3/gpu:0", device_name)
    self.assertEqual("hidden_0/Weights", node_name)
    self.assertEqual(0, output_slot)
    self.assertEqual("DebugNumericSummary", debug_op)
    self.assertEqual(0, exec_index)

  def testParseMalformedDebugTensorName(self):
    # [FUNCTION DOC]
    # Feeds malformed names (too many colon items, an incomplete device prefix, empty or
    # non-numeric bracket suffixes). Expects a value error each time; the first two cases must
    # carry the "malformed" message about the debug tensor name.
    with self.assertRaisesRegex(
        ValueError,
        r"The debug tensor name in the to-be-evaluated expression is "
        r"malformed:"):
      evaluator._parse_debug_tensor_name(
          "/job:ps/replica:0/task:2/cpu:0:foo:1:DebugNanCount:1337")

    with self.assertRaisesRegex(
        ValueError,
        r"The debug tensor name in the to-be-evaluated expression is "
        r"malformed:"):
      evaluator._parse_debug_tensor_name(
          "/job:ps/replica:0/cpu:0:foo:1:DebugNanCount")

    with self.assertRaises(ValueError):
      evaluator._parse_debug_tensor_name(
          "foo:1:DebugNanCount[]")

    with self.assertRaises(ValueError):
      evaluator._parse_debug_tensor_name(
          "foo:1[DebugNanCount]")

  def testParseNamesWithExecIndex(self):
    # [FUNCTION DOC]
    # Parses names ending in a bracketed execution index. Expects the integer index returned
    # (20 and 3), with the node, slot and default debug op unaffected and no device name.
    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name("foo:1[20]"))
    self.assertIsNone(device_name)
    self.assertEqual("foo", node_name)
    self.assertEqual(1, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(20, exec_index)

    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name("hidden_0/Weights:0[3]"))
    self.assertIsNone(device_name)
    self.assertEqual("hidden_0/Weights", node_name)
    self.assertEqual(0, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(3, exec_index)


class EvaluatorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for ExpressionEvaluator.evaluate. Each test builds a mock debug dump whose tensor
  # lookup method is patched with a fake returning canned numpy arrays, then evaluates
  # expressions against it.

  def testEvaluateSingleTensor(self):
    # [FUNCTION DOC]
    # Evaluates an expression taking the size of one tensor, expecting 3, and checks that
    # whitespace inside the backticks is tolerated.
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      # [FUNCTION DOC]
      # HELPER: fake dump lookup for this test. Ignores its arguments and always returns a list
      # holding one one-by-three array.
      del node_name, output_slot, debug_op, device_name  # Unused.
      return [np.array([[1.0, 2.0, 3.0]])]

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      self.assertEqual(3, ev.evaluate("np.size(`a:0`)"))

      # Whitespace in backticks should be tolerated.
      self.assertEqual(3, ev.evaluate("np.size(` a:0 `)"))

  def testEvaluateTwoTensors(self):
    # [FUNCTION DOC]
    # Evaluates expressions referencing two different tensors (matrix product, and product plus
    # the second tensor) and expects the numerically correct results.
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      # [FUNCTION DOC]
      # HELPER: fake dump lookup. Returns a two-by-two matrix for node a slot 0 and a column for
      # node b slot 0; returns nothing for anything else.
      del debug_op, device_name  # Unused.
      if node_name == "a" and output_slot == 0:
        return [np.array([[1.0, -2.0], [0.0, 1.0]])]
      elif node_name == "b" and output_slot == 0:
        return [np.array([[-1.0], [1.0]])]

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      self.assertAllClose([[-3.0], [1.0]],
                          ev.evaluate("np.matmul(`a:0`, `b:0`)"))
      self.assertAllClose(
          [[-4.0], [2.0]], ev.evaluate("np.matmul(`a:0`, `b:0`) + `b:0`"))

  def testEvaluateNoneExistentTensorGeneratesError(self):
    # [FUNCTION DOC]
    # The fake lookup always signals that the watch key is missing; evaluating an expression is
    # expected to raise a value error saying the evaluation failed because the value is
    # unavailable.
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      # [FUNCTION DOC]
      # HELPER: fake dump lookup that always raises the watch-key-does-not-exist error.
      del node_name, output_slot, debug_op, device_name  # Unused.
      raise debug_data.WatchKeyDoesNotExistInDebugDumpDirError()

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      with self.assertRaisesRegex(
          ValueError, "Eval failed due to the value of .* being unavailable"):
        ev.evaluate("np.matmul(`a:0`, `b:0`)")

  def testEvaluateWithMultipleDevicesContainingTheSameTensorName(self):
    # [FUNCTION DOC]
    # Two devices hold a node with the same name. Without a device prefix the evaluation is
    # expected to raise the ambiguity value error; with explicit device prefixes in the
    # expression it must return the sum of the two device-specific values (30).
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      # [FUNCTION DOC]
      # HELPER: fake dump lookup. Raises a multiple-devices error when no device is given for
      # node a; returns 10 for the worker task 0 cpu and 20 for the task 1 cpu.
      del output_slot, debug_op  # Unused.
      if node_name == "a" and device_name is None:
        raise ValueError(
            "There are multiple (2) devices with nodes named 'a' but "
            "device_name is not specified")
      elif (node_name == "a" and
            device_name == "/job:worker/replica:0/task:0/cpu:0"):
        return [np.array(10.0)]
      elif (node_name == "a" and
            device_name == "/job:worker/replica:0/task:1/cpu:0"):
        return [np.array(20.0)]

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      with self.assertRaisesRegex(ValueError, r"multiple \(2\) devices"):
        ev.evaluate("`a:0` + `a:0`")

      self.assertAllClose(
          30.0,
          ev.evaluate("`/job:worker/replica:0/task:0/cpu:0:a:0` + "
                      "`/job:worker/replica:0/task:1/cpu:0:a:0`"))

  def testEvaluateWithNonDefaultDebugOp(self):
    # [FUNCTION DOC]
    # Evaluates a product of the same tensor read under two different debug ops (a custom one and
    # the identity one) and expects the lookup to honor the debug op, giving 4.
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      # [FUNCTION DOC]
      # HELPER: fake dump lookup keyed on debug op: a column for the identity op and a row for the
      # custom op on node a slot 0; returns nothing otherwise.
      del device_name  # Unused.
      if node_name == "a" and output_slot == 0 and debug_op == "DebugIdentity":
        return [np.array([[-1.0], [1.0]])]
      elif node_name == "a" and output_slot == 0 and debug_op == "DebugFoo":
        return [np.array([[-2.0, 2.0]])]

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      self.assertAllClose(
          [[4.0]],
          ev.evaluate("np.matmul(`a:0:DebugFoo`, `a:0:DebugIdentity`)"))

  def testEvaluateWithMultipleExecIndexes(self):
    # [FUNCTION DOC]
    # The fake lookup returns two executions of one tensor. An expression using execution indexes
    # one and zero (transposing the first) is expected to pick the right entries, giving 4.
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      # [FUNCTION DOC]
      # HELPER: fake dump lookup returning a list of two column arrays for node a slot 0.
      del debug_op, device_name  # Unused.
      if node_name == "a" and output_slot == 0:
        return [np.array([[-1.0], [1.0]]), np.array([[-2.0], [2.0]])]

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      self.assertAllClose(
          [[4.0]], ev.evaluate("np.matmul(`a:0[1]`.T, `a:0[0]`)"))

  def testEvaluateExpressionWithUnmatchedBacktick(self):
    # [FUNCTION DOC]
    # An expression with an odd number of backticks leaves a stray backtick after rewriting;
    # evaluation is expected to raise a syntax error.
    dump = test.mock.MagicMock()
    ev = evaluator.ExpressionEvaluator(dump)
    with self.assertRaises(SyntaxError):
      ev.evaluate("np.matmul(`a:0`, `b:0`) + `b:0")

  def testEvaluateExpressionWithInvalidDebugTensorName(self):
    # [FUNCTION DOC]
    # Expressions with malformed backtick names (no slot, too many items, empty bracket index)
    # are expected to raise value errors; the first two must mention the malformed tensor name.
    dump = test.mock.MagicMock()
    ev = evaluator.ExpressionEvaluator(dump)
    with self.assertRaisesRegex(ValueError,
                                r".* tensor name .* expression .* malformed"):
      ev.evaluate("np.matmul(`a`, `b`)")

    with self.assertRaisesRegex(ValueError,
                                r".* tensor name .* expression .* malformed"):
      ev.evaluate("np.matmul(`a:0:DebugIdentity:0`, `b:1:DebugNanCount:2`)")

    with self.assertRaises(ValueError):
      ev.evaluate("np.matmul(`a:0[]`, `b:0[]`)")


if __name__ == "__main__":
  test.main()
