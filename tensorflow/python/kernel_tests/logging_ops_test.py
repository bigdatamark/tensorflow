# [MODULE DOC]
# Tests for the logging ops: the assert op's multi-tensor failure message, the version-2 print op
# (summarization, separators, end strings, structures, sparse tensors, output streams such as
# standard output and files, op naming, behavior inside compiled functions) and the older print
# op's shape and gradient pass-through. Defines three test classes and runs the TensorFlow test
# main when executed as a script.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.kernels.logging_ops."""

import os
import string
import sys
import tempfile

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class LoggingOpsTest(test.TestCase):
  # [CLASS DOC]
  # Graph-mode test case for the assert op used as a control dependency. No shared fixtures.

  @test_util.run_deprecated_v1
  def testAssertDivideByZero(self):
    # [FUNCTION DOC]
    # Graph-mode only. First guards a division with an assert that passes and checks the result is
    # two. Then guards a division with an assert that fails (a tiny epsilon is not less than zero)
    # carrying two message items, and asserts evaluating raises an op error containing the second
    # message. Returns nothing.
    with self.cached_session() as sess:
      epsilon = ops.convert_to_tensor(1e-20)
      x = ops.convert_to_tensor(0.0)
      y = ops.convert_to_tensor(1.0)
      z = ops.convert_to_tensor(2.0)
      # assert(epsilon < y)
      # z / y
      with sess.graph.control_dependencies([
          control_flow_assert.Assert(
              math_ops.less(epsilon, y), ["Divide-by-zero"])
      ]):
        out = math_ops.div(z, y)
      self.assertAllEqual(2.0, self.evaluate(out))
      # assert(epsilon < x)
      # z / x
      #
      # This tests printing out multiple tensors
      with sess.graph.control_dependencies([
          control_flow_assert.Assert(
              math_ops.less(epsilon, x), ["Divide-by-zero", "less than x"])
      ]):
        out = math_ops.div(z, x)
      with self.assertRaisesOpError("less than x"):
        self.evaluate(out)


@test_util.run_all_in_graph_and_eager_modes
class PrintV2Test(test.TestCase):
  # [CLASS DOC]
  # Test case for the version-2 print op, run in both graph and eager modes by the class
  # decorator. Each test captures writes to a standard stream (or a file) and compares the printed
  # text to the expected formatting. No shared fixtures.

  def testPrintOneTensor(self):
    # [FUNCTION DOC]
    # Prints a range of ten numbers to standard error and asserts the output is the abbreviated
    # form with first and last three items, followed by a newline. Returns nothing.
    tensor = math_ops.range(10)
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(tensor)
      self.evaluate(print_op)

    expected = "[0 1 2 ... 7 8 9]"
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintOneStringTensor(self):
    # [FUNCTION DOC]
    # Prints a string tensor of the 26 lowercase letters and asserts the abbreviated output shows
    # the first three and last three letters in double quotes. Returns nothing.
    tensor = ops.convert_to_tensor([char for char in string.ascii_lowercase])
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(tensor)
      self.evaluate(print_op)

    expected = "[\"a\" \"b\" \"c\" ... \"x\" \"y\" \"z\"]"
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintOneTensorVarySummarize(self):
    # [FUNCTION DOC]
    # Prints a range of ten numbers four times with summarize set to one, two, three and negative
    # one; asserts the printed edge-item counts are one, two, three and all ten respectively.
    tensor = math_ops.range(10)
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(tensor, summarize=1)
      self.evaluate(print_op)

    expected = "[0 ... 9]"
    self.assertIn((expected + "\n"), printed.contents())

    tensor = math_ops.range(10)
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(tensor, summarize=2)
      self.evaluate(print_op)

    expected = "[0 1 ... 8 9]"
    self.assertIn((expected + "\n"), printed.contents())

    tensor = math_ops.range(10)
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(tensor, summarize=3)
      self.evaluate(print_op)

    expected = "[0 1 2 ... 7 8 9]"
    self.assertIn((expected + "\n"), printed.contents())

    tensor = math_ops.range(10)
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(tensor, summarize=-1)
      self.evaluate(print_op)

    expected = "[0 1 2 3 4 5 6 7 8 9]"
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintOneVariable(self):
    # [FUNCTION DOC]
    # Prints a variable holding a range of ten numbers (initializing it first in graph mode) and
    # asserts the abbreviated values are printed. Returns nothing.
    var = variables.Variable(math_ops.range(10))
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(var)
      self.evaluate(print_op)
    expected = "[0 1 2 ... 7 8 9]"
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintTwoVariablesInStructWithAssignAdd(self):
    # [FUNCTION DOC]
    # Increments a scalar variable (2.14 plus one), then prints it together with a mapping holding
    # a second variable; asserts the output shows the updated value 3.14 and the mapping with the
    # abbreviated second variable. Initializes variables first in graph mode.
    var_one = variables.Variable(2.14)
    plus_one = var_one.assign_add(1.0)
    var_two = variables.Variable(math_ops.range(10))
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
    with self.captureWritesToStream(sys.stderr) as printed:
      self.evaluate(plus_one)
      print_op = logging_ops.print_v2(var_one, {"second": var_two})
      self.evaluate(print_op)
    expected = "3.14 {'second': [0 1 2 ... 7 8 9]}"
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintTwoTensors(self):
    # [FUNCTION DOC]
    # Prints two tensors in one call and asserts they are joined by the default single space.
    tensor = math_ops.range(10)
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(tensor, tensor * 10)
      self.evaluate(print_op)
    expected = "[0 1 2 ... 7 8 9] [0 10 20 ... 70 80 90]"
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintTwoTensorsDifferentSep(self):
    # [FUNCTION DOC]
    # Prints two tensors with a custom separator string and asserts the output joins them with
    # exactly that separator, followed by a newline.
    tensor = math_ops.range(10)
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(tensor, tensor * 10, sep="<separator>")
      self.evaluate(print_op)
    expected = "[0 1 2 ... 7 8 9]<separator>[0 10 20 ... 70 80 90]"
    self.assertIn(expected + "\n", printed.contents())

  def testPrintPlaceholderGeneration(self):
    # [FUNCTION DOC]
    # Prints a string containing a literal placeholder-like token together with a mapping whose
    # key is the same token and whose value is a tensor; asserts the literal text is preserved
    # (not treated as a format placeholder) and the tensor is abbreviated.
    tensor = math_ops.range(10)
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2("{}6", {"{}": tensor * 10})
      self.evaluate(print_op)
    expected = "{}6 {'{}': [0 10 20 ... 70 80 90]}"
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintNoTensors(self):
    # [FUNCTION DOC]
    # Prints plain Python values (an integer, a list and a mapping) containing no tensors and
    # asserts they are rendered in their ordinary text form separated by spaces.
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(23, [23, 5], {"6": 12})
      self.evaluate(print_op)
    expected = "23 [23, 5] {'6': 12}"
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintFloatScalar(self):
    # [FUNCTION DOC]
    # For each of bfloat16, half, float32 and float64, prints the scalar 43.5 and asserts the
    # output is exactly 43.5 plus a newline.
    for dtype in [dtypes.bfloat16, dtypes.half, dtypes.float32, dtypes.float64]:
      tensor = ops.convert_to_tensor(43.5, dtype=dtype)
      with self.captureWritesToStream(sys.stderr) as printed:
        print_op = logging_ops.print_v2(tensor)
        self.evaluate(print_op)
      expected = "43.5"
      self.assertIn((expected + "\n"), printed.contents())

  def testPrintStringScalar(self):
    # [FUNCTION DOC]
    # Prints a scalar string tensor and asserts the raw text (no quotes) plus a newline appears.
    tensor = ops.convert_to_tensor("scalar")
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(tensor)
      self.evaluate(print_op)
    expected = "scalar"
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintStringScalarDifferentEnd(self):
    # [FUNCTION DOC]
    # Prints a scalar string tensor with a custom end string and asserts the output contains the
    # text followed by that custom end instead of a newline.
    tensor = ops.convert_to_tensor("scalar")
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(tensor, end="<customend>")
      self.evaluate(print_op)
    expected = "scalar<customend>"
    self.assertIn(expected, printed.contents())

  def testPrintComplexTensorStruct(self):
    # [FUNCTION DOC]
    # Prints a mix of strings, a tensor, a mapping of two tensors, an integer and a list of
    # tensors in a single call; asserts the nested structures are rendered with mapping keys in
    # sorted order (capitalized key first) and tensors abbreviated.
    tensor = math_ops.range(10)
    small_tensor = constant_op.constant([0.3, 12.4, -16.1])
    big_tensor = math_ops.mul(tensor, 10)
    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(
          "first:", tensor, "middle:",
          {"small": small_tensor, "Big": big_tensor}, 10,
          [tensor * 2, tensor])
      self.evaluate(print_op)
    # Note that the keys in the dict will always be sorted,
    # so 'Big' comes before 'small'
    expected = ("first: [0 1 2 ... 7 8 9] "
                "middle: {'Big': [0 10 20 ... 70 80 90], "
                "'small': [0.3 12.4 -16.1]} "
                "10 [[0 2 4 ... 14 16 18], [0 1 2 ... 7 8 9]]")
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintSparseTensor(self):
    # [FUNCTION DOC]
    # Builds a 5-by-6 sparse tensor with seven entries and prints it; asserts the output is the
    # quoted sparse-tensor text with abbreviated indices, values and the dense shape.
    ind = [[0, 0], [1, 0], [1, 3], [4, 1], [1, 4], [3, 2], [3, 3]]
    val = [0, 10, 13, 4, 14, 32, 33]
    shape = [5, 6]

    sparse = sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int64),
        constant_op.constant(shape, dtypes.int64))

    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2(sparse)
      self.evaluate(print_op)
    expected = ("'SparseTensor(indices=[[0 0]\n"
                " [1 0]\n"
                " [1 3]\n"
                " ...\n"
                " [1 4]\n"
                " [3 2]\n"
                " [3 3]], values=[0 10 13 ... 14 32 33], shape=[5 6])'")
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintSparseTensorInDataStruct(self):
    # [FUNCTION DOC]
    # Same sparse tensor as the previous test but printed inside a list; asserts the list output
    # contains the quoted sparse-tensor text with abbreviated indices, values and shape.
    ind = [[0, 0], [1, 0], [1, 3], [4, 1], [1, 4], [3, 2], [3, 3]]
    val = [0, 10, 13, 4, 14, 32, 33]
    shape = [5, 6]

    sparse = sparse_tensor.SparseTensor(
        constant_op.constant(ind, dtypes.int64),
        constant_op.constant(val, dtypes.int64),
        constant_op.constant(shape, dtypes.int64))

    with self.captureWritesToStream(sys.stderr) as printed:
      print_op = logging_ops.print_v2([sparse])
      self.evaluate(print_op)
    expected = ("['SparseTensor(indices=[[0 0]\n"
                " [1 0]\n"
                " [1 3]\n"
                " ...\n"
                " [1 4]\n"
                " [3 2]\n"
                " [3 3]], values=[0 10 13 ... 14 32 33], shape=[5 6])']")
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintOneTensorStdout(self):
    # [FUNCTION DOC]
    # Prints a range of ten numbers with standard output as the output stream and asserts the
    # abbreviated text arrives on captured standard output.
    tensor = math_ops.range(10)
    with self.captureWritesToStream(sys.stdout) as printed:
      print_op = logging_ops.print_v2(
          tensor, output_stream=sys.stdout)
      self.evaluate(print_op)
    expected = "[0 1 2 ... 7 8 9]"
    self.assertIn((expected + "\n"), printed.contents())

  def testPrintTensorsToFile(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the temporary file descriptor is closed and the file removed only when every
    # assertion passes; on failure (or a non-IO exception) both the descriptor and file leak.
    # Creates a temporary file, prints two tensors to it through a file-scheme output stream in
    # two separate calls, then reads two lines and asserts each contains the expected abbreviated
    # tensor text. Closes and deletes the file; an IO error fails the test. Returns nothing.
    fd, tmpfile_name = tempfile.mkstemp(".printv2_test")
    tensor_0 = math_ops.range(0, 10)
    print_op_0 = logging_ops.print_v2(tensor_0,
                                      output_stream="file://"+tmpfile_name)
    self.evaluate(print_op_0)
    tensor_1 = math_ops.range(11, 20)
    print_op_1 = logging_ops.print_v2(tensor_1,
                                      output_stream="file://"+tmpfile_name)
    self.evaluate(print_op_1)
    try:
      f = os.fdopen(fd, "r")
      line_0 = f.readline()
      expected_0 = "[0 1 2 ... 7 8 9]"
      self.assertTrue(expected_0 in line_0)
      line_1 = f.readline()
      expected_1 = "[11 12 13 ... 17 18 19]"
      self.assertTrue(expected_1 in line_1)
      os.close(fd)
      os.remove(tmpfile_name)
    except IOError as e:
      self.fail(e)

  def testInvalidOutputStreamRaisesError(self):
    # [FUNCTION DOC]
    # Asserts that using an unrecognized output stream string raises a value error.
    tensor = math_ops.range(10)
    with self.assertRaises(ValueError):
      print_op = logging_ops.print_v2(
          tensor, output_stream="unknown")
      self.evaluate(print_op)

  @test_util.run_deprecated_v1
  def testPrintOpName(self):
    # [FUNCTION DOC]
    # Graph-mode only. Asserts the print op created with an explicit name carries exactly that name.
    tensor = math_ops.range(10)
    print_op = logging_ops.print_v2(tensor, name="print_name")
    self.assertEqual(print_op.name, "print_name")

  @test_util.run_deprecated_v1
  def testNoDuplicateFormatOpGraphModeAfterExplicitFormat(self):
    # [FUNCTION DOC]
    # Graph-mode only. Formats a tensor into a string explicitly, prints the string, and asserts
    # the default graph contains exactly one string-format operation (print did not add another).
    tensor = math_ops.range(10)
    formatted_string = string_ops.string_format("{}", tensor)
    print_op = logging_ops.print_v2(formatted_string)
    self.evaluate(print_op)
    graph_ops = ops.get_default_graph().get_operations()
    format_ops = [op for op in graph_ops if op.type == "StringFormat"]
    # Should be only 1 format_op for graph mode.
    self.assertEqual(len(format_ops), 1)

  def testPrintOneTensorEagerOnOpCreate(self):
    # [FUNCTION DOC]
    # In eager mode, calls print without evaluating the result and asserts the abbreviated text is
    # already written to captured standard error (printing happens at op creation).
    with context.eager_mode():
      tensor = math_ops.range(10)
      expected = "[0 1 2 ... 7 8 9]"
      with self.captureWritesToStream(sys.stderr) as printed:
        logging_ops.print_v2(tensor)
      self.assertIn((expected + "\n"), printed.contents())

  def testPrintsOrderedInDefun(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the final check applies a truthiness assertion to a non-empty string literal
    # (the second argument is only the failure message), so it always passes and does not verify
    # the order of printed output.
    # In eager mode, calls a compiled function that prints "A", "B", "C" in sequence while
    # capturing standard error. Returns nothing.
    with context.eager_mode():

      @def_function.function
      def prints():
        # [FUNCTION DOC]
        # HELPER: compiled function printing three strings in order; takes no inputs, returns
        # nothing.
        logging_ops.print_v2("A")
        logging_ops.print_v2("B")
        logging_ops.print_v2("C")

      with self.captureWritesToStream(sys.stderr) as printed:
        prints()
      self.assertTrue(("A\nB\nC\n"), printed.contents())

  def testPrintInDefunWithoutExplicitEvalOfPrint(self):
    # [FUNCTION DOC]
    # Calls a compiled function twice that prints its tensor argument without the print result
    # being returned or evaluated; asserts the abbreviated text is printed on each call, not only
    # on the first.
    tensor = math_ops.range(10)

    @def_function.function
    def f(tensor):
      # [FUNCTION DOC]
      # HELPER: compiled function that prints its tensor argument as a side effect and returns the
      # same tensor.
      logging_ops.print_v2(tensor)
      return tensor

    expected = "[0 1 2 ... 7 8 9]"
    with self.captureWritesToStream(sys.stderr) as printed_one:
      x = f(tensor)
      self.evaluate(x)
    self.assertIn((expected + "\n"), printed_one.contents())

    # We execute the function again to make sure it doesn't only print on the
    # first call.
    with self.captureWritesToStream(sys.stderr) as printed_two:
      y = f(tensor)
      self.evaluate(y)
    self.assertIn((expected + "\n"), printed_two.contents())


class PrintGradientTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the older print op as an identity-like op: it must preserve static shape and
  # pass gradients through. No shared fixtures.

  @test_util.run_in_graph_and_eager_modes
  def testPrintShape(self):
    # [FUNCTION DOC]
    # Applies the print op to a 100-by-32 tensor with the tensor as the printed data and asserts
    # the output keeps the same static shape. Graph and eager modes.
    inp = constant_op.constant(2.0, shape=[100, 32])
    inp_printed = logging_ops.Print(inp, [inp])
    self.assertEqual(inp.get_shape(), inp_printed.get_shape())

  def testPrintString(self):
    # [FUNCTION DOC]
    # Applies the print op to a 100-by-32 tensor with a string message as printed data and
    # asserts the output keeps the same static shape. Runs in the default mode only.
    inp = constant_op.constant(2.0, shape=[100, 32])
    inp_printed = logging_ops.Print(inp, ["hello"])
    self.assertEqual(inp.get_shape(), inp_printed.get_shape())

  @test_util.run_deprecated_v1
  def testPrintGradient(self):
    # [FUNCTION DOC]
    # Graph-mode only. Computes a matrix product of a weight and an input, wraps it in the print
    # op, and asserts the gradient with respect to the weight is the same with and without the
    # print wrapper (gradients pass through).
    inp = constant_op.constant(2.0, shape=[100, 32], name="in")
    w = constant_op.constant(4.0, shape=[10, 100], name="w")
    wx = math_ops.matmul(w, inp, name="wx")
    wx_print = logging_ops.Print(wx, [w, w, w])
    wx_grad = gradients_impl.gradients(wx, w)[0]
    wx_print_grad = gradients_impl.gradients(wx_print, w)[0]
    wxg = self.evaluate(wx_grad)
    wxpg = self.evaluate(wx_print_grad)
    self.assertAllEqual(wxg, wxpg)


if __name__ == "__main__":
  test.main()
