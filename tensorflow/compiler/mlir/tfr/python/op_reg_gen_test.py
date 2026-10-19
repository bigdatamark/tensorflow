# [MODULE DOC]
# Unit test for the op registration generator. Defines two composite op functions in this module
# (one with only derived attributes and an output, one with inputs, attributes, derived attributes
# and two outputs) and checks the generated C++ registration text against expected lines using a
# FileCheck-style helper. Runs the standard test main when executed.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `op_reg_gen` module."""

# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=g-direct-tensorflow-import

import sys

from tensorflow.compiler.mlir.python.mlir_wrapper import filecheck_wrapper as fw
from tensorflow.compiler.mlir.tfr.python import composite
from tensorflow.compiler.mlir.tfr.python.op_reg_gen import gen_register_op
from tensorflow.python.platform import test


Composite = composite.Composite


@composite.Composite(
    'TestNoOp', derived_attrs=['T: numbertype'], outputs=['o1: T'])
def _composite_no_op():
  # [FUNCTION DOC]
  # HELPER: Test fixture composite function for the op named test no op (derived numeric type
  # attribute, one output). Takes no arguments, empty body, returns nothing.
  pass


@Composite(
    'TestCompositeOp',
    inputs=['x: T', 'y: T'],
    attrs=['act: {"", "relu"}', 'trans: bool = true'],
    derived_attrs=['T: numbertype'],
    outputs=['o1: T', 'o2: T'])
def _composite_op(x, y, act, trans):
  # [FUNCTION DOC]
  # HELPER: Test fixture composite function for the op named test composite op, with two inputs,
  # two attributes (an activation string and a boolean defaulting to true), a derived type
  # attribute and two outputs. Takes four parameters matching the declared inputs plus attributes.
  # Returns a pair: x plus act and y plus trans (meaningless values; the body is never executed).
  return x + act, y + trans


class TFRGenTensorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the op registration generator. Uses the composite functions defined at module
  # level as fixtures.
  # KNOWN ISSUE: Docstring mentions MLIR generation, but the tests check generated C++ op
  # registration code.
  """MLIR Generation Tests for MLIR TFR Program."""

  def test_op_reg_gen(self):
    # [FUNCTION DOC]
    # Scenario: generates registration code for all functions of this module (no name prefix
    # filter). Expects the include of the op header, the namespace opening, a registration for the
    # test no op (attribute and output), and a registration for the test composite op (two inputs,
    # two attributes with double quotes converted to single quotes, the derived attribute and two
    # outputs), then the namespace closing. Asserts that the text matches, showing the generated
    # code on failure.
    cxx_code = gen_register_op(sys.modules[__name__])
    cxx_code_exp = r"""
      CHECK: #include "tensorflow/core/framework/op.h"
      CHECK-EMPTY
      CHECK: namespace tensorflow {
      CHECK-EMPTY
      CHECK-LABEL: REGISTER_OP("TestNoOp")
      CHECK-NEXT:      .Attr("T: numbertype")
      CHECK-NEXT:      .Output("o1: T");
      CHECK-EMPTY
      CHECK-LABEL: REGISTER_OP("TestCompositeOp")
      CHECK-NEXT:      .Input("x: T")
      CHECK-NEXT:      .Input("y: T")
      CHECK-NEXT:      .Attr("act: {'', 'relu'}")
      CHECK-NEXT:      .Attr("trans: bool = true")
      CHECK-NEXT:      .Attr("T: numbertype")
      CHECK-NEXT:      .Output("o1: T")
      CHECK-NEXT:      .Output("o2: T");
      CHECK-EMPTY
      CHECK:  }  // namespace tensorflow
    """
    self.assertTrue(fw.check(str(cxx_code), cxx_code_exp), str(cxx_code))


if __name__ == '__main__':
  test.main()
