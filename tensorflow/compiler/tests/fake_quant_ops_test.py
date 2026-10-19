# [MODULE DOC]
# Test module for the fake-quantization ops on XLA test devices. Six test classes cover the
# min/max "args" (attributes), "vars" (tensor inputs) and per-channel variants, each in a forward
# and a gradient flavor. Each class sweeps 8-bit and 7-bit quantization, wide and narrow range,
# and minimum/maximum values that need no nudging or get nudged down, up or in between, using
# generated inputs around the expected nudged range and comparing with expected outputs or
# gradients. No import-time side effects; runs the test runner when executed as a script.
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
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.platform import googletest


class FakeQuantWithMinMaxArgsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the forward fake-quantization op whose minimum, maximum, bit width and
  # narrow-range flag are op attributes. Test methods pass range parameters and the expected
  # nudged range and step to the shared _TestOp helper.
  """Test cases for FakeQuantWithMinMaxArgs operation."""

  # 8 bits, wide range.
  def testOp_with8BitsNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum 0 and maximum 255: no scaling or nudging; expected nudged
    # range 0 to 255 with step 1.
    self._TestOp(0.0, 255.0, 8, False, 0.0, 255.0, 1.0)

  def testOp_with8BitsScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum 0.5 and maximum 128: minimum is nudged down to zero; expected
    # nudged range 0 to 127.5 with step 0.5.
    self._TestOp(0.5, 128.0, 8, False, 0.0, 127.5, 0.5)

  def testOp_with8BitsScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum -128 and maximum -0.5: maximum is nudged up to zero; expected
    # nudged range -127.5 to 0 with step 0.5.
    self._TestOp(-128.0, -0.5, 8, False, -127.5, 0.0, 0.5)

  def testOp_with8BitsScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum -0.1 and maximum 127.4: range is nudged so zero is
    # representable; expected nudged range 0 to 127.5 with step 0.5.
    self._TestOp(-0.1, 127.4, 8, False, 0.0, 127.5, 0.5)

  # 8 bits, narrow range.
  def testOp_with8BitsNarrowRangeNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum 0 and maximum 254: no nudging; expected range 0 to 254
    # with step 1.
    self._TestOp(0.0, 254.0, 8, True, 0.0, 254.0, 1.0)

  def testOp_with8BitsNarrowRangeScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum 0.1 and maximum 127.1: minimum nudged down; expected
    # nudged range 0 to 127 with step 0.5.
    self._TestOp(0.1, 127.1, 8, True, 0.0, 127.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum -127.1 and maximum -0.1: maximum nudged up; expected
    # nudged range -127 to 0 with step 0.5.
    self._TestOp(-127.1, -0.1, 8, True, -127.0, 0.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum -0.1 and maximum 126.9: nudged so zero is representable;
    # expected nudged range 0 to 127 with step 0.5.
    self._TestOp(-0.1, 126.9, 8, True, 0.0, 127.0, 0.5)

  # 7 bits, wide range.
  def testOp_with7BitsNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum 0 and maximum 127: no nudging; expected range 0 to 127 with
    # step 1.
    self._TestOp(0.0, 127.0, 7, False, 0.0, 127.0, 1.0)

  def testOp_with7BitsScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum 0.5 and maximum 64: minimum nudged down; expected nudged
    # range 0 to 63.5 with step 0.5.
    self._TestOp(0.5, 64.0, 7, False, 0.0, 63.5, 0.5)

  def testOp_with7BitsScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum -64 and maximum -0.5: maximum nudged up; expected nudged
    # range -63.5 to 0 with step 0.5.
    self._TestOp(-64.0, -0.5, 7, False, -63.5, 0.0, 0.5)

  def testOp_with7BitsScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum -0.1 and maximum 63.4: nudged so zero is representable;
    # expected nudged range 0 to 63.5 with step 0.5.
    self._TestOp(-0.1, 63.4, 7, False, 0.0, 63.5, 0.5)

  # 7 bits, narrow range.
  def testOp_with7BitsNarrowRangeNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum 0 and maximum 126: no nudging; expected range 0 to 126
    # with step 1.
    self._TestOp(0.0, 126.0, 7, True, 0.0, 126.0, 1.0)

  def testOp_with7BitsNarrowRangeScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum 0.1 and maximum 63.1: minimum nudged down; expected
    # nudged range 0 to 63 with step 0.5.
    self._TestOp(0.1, 63.1, 7, True, 0.0, 63.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum -63.1 and maximum -0.1: maximum nudged up; expected
    # nudged range -63 to 0 with step 0.5.
    self._TestOp(-63.1, -0.1, 7, True, -63.0, 0.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum -0.1 and maximum 62.9: nudged so zero is representable;
    # expected nudged range 0 to 63 with step 0.5.
    self._TestOp(-0.1, 62.9, 7, True, 0.0, 63.0, 0.5)

  def _TestOp(self, input_min, input_max, num_bits, narrow_range,
              expected_nudged_input_min, expected_nudged_input_max,
              expected_step):
    # [FUNCTION DOC]
    # HELPER: shared verification for all tests in this class.
    # Takes the op minimum and maximum, bit count, narrow-range flag, and the expected nudged
    # minimum, nudged maximum and quantization step. Builds eleven float inputs around the nudged
    # range (below, at, just above and just below step boundaries, at and above the maximum) and
    # the expected quantized outputs (clamped to the nudged ends or snapped to the step grid).
    # Runs the fake-quant op on the test device through a placeholder and asserts the result is
    # close to the expected values with type-dependent tolerances. Returns nothing.
    inputs = np.array(
        [
            expected_nudged_input_min - expected_step,
            expected_nudged_input_min - 0.01, expected_nudged_input_min,
            expected_nudged_input_min + 0.01,
            expected_nudged_input_min + expected_step - 0.01,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step + 0.01,
            expected_nudged_input_max - 0.01, expected_nudged_input_max,
            expected_nudged_input_max + 0.01,
            expected_nudged_input_max + expected_step
        ],
        dtype=np.float32)
    expected = np.array(
        [
            expected_nudged_input_min, expected_nudged_input_min,
            expected_nudged_input_min, expected_nudged_input_min,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_max, expected_nudged_input_max,
            expected_nudged_input_max, expected_nudged_input_max
        ],
        dtype=np.float32)

    with self.session() as session:
      with self.test_scope():
        input_placeholder = array_ops.placeholder(
            dtypes.float32, inputs.shape, name="inputs")
        outputs = array_ops.fake_quant_with_min_max_args(
            input_placeholder,
            min=input_min,
            max=input_max,
            num_bits=num_bits,
            narrow_range=narrow_range)
      result = session.run(outputs, {input_placeholder: inputs})
      self.assertAllCloseAccordingToType(
          result, expected, rtol=1e-3, atol=1e-5, bfloat16_rtol=0.03)


class FakeQuantWithMinMaxArgsGradientTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the gradient of the attribute-based fake-quantization op. Test methods pass
  # range parameters and the expected nudged range and step to the shared _TestOp helper.
  """Test cases for FakeQuantWithMinMaxArgsGradient operation."""

  # 8 bits, wide range.
  def testOp_with8BitsNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum 0 and maximum 255: no scaling or nudging; expected nudged
    # range 0 to 255 with step 1.
    self._TestOp(0.0, 255.0, 8, False, 0.0, 255.0, 1.0)

  def testOp_with8BitsScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum 0.5 and maximum 128: minimum is nudged down to zero; expected
    # nudged range 0 to 127.5 with step 0.5.
    self._TestOp(0.5, 128.0, 8, False, 0.0, 127.5, 0.5)

  def testOp_with8BitsScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum -128 and maximum -0.5: maximum is nudged up to zero; expected
    # nudged range -127.5 to 0 with step 0.5.
    self._TestOp(-128.0, -0.5, 8, False, -127.5, 0.0, 0.5)

  def testOp_with8BitsScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum -0.1 and maximum 127.4: range is nudged so zero is
    # representable; expected nudged range 0 to 127.5 with step 0.5.
    self._TestOp(-0.1, 127.4, 8, False, 0.0, 127.5, 0.5)

  # 8 bits, narrow range.
  def testOp_with8BitsNarrowRangeNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum 0 and maximum 254: no nudging; expected range 0 to 254
    # with step 1.
    self._TestOp(0.0, 254.0, 8, True, 0.0, 254.0, 1.0)

  def testOp_with8BitsNarrowRangeScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum 0.1 and maximum 127.1: minimum nudged down; expected
    # nudged range 0 to 127 with step 0.5.
    self._TestOp(0.1, 127.1, 8, True, 0.0, 127.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum -127.1 and maximum -0.1: maximum nudged up; expected
    # nudged range -127 to 0 with step 0.5.
    self._TestOp(-127.1, -0.1, 8, True, -127.0, 0.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum -0.1 and maximum 126.9: nudged so zero is representable;
    # expected nudged range 0 to 127 with step 0.5.
    self._TestOp(-0.1, 126.9, 8, True, 0.0, 127.0, 0.5)

  # 7 bits, wide range.
  def testOp_with7BitsNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum 0 and maximum 127: no nudging; expected range 0 to 127 with
    # step 1.
    self._TestOp(0.0, 127.0, 7, False, 0.0, 127.0, 1.0)

  def testOp_with7BitsScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum 0.5 and maximum 64: minimum nudged down; expected nudged
    # range 0 to 63.5 with step 0.5.
    self._TestOp(0.5, 64.0, 7, False, 0.0, 63.5, 0.5)

  def testOp_with7BitsScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum -64 and maximum -0.5: maximum nudged up; expected nudged
    # range -63.5 to 0 with step 0.5.
    self._TestOp(-64.0, -0.5, 7, False, -63.5, 0.0, 0.5)

  def testOp_with7BitsScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum -0.1 and maximum 63.4: nudged so zero is representable;
    # expected nudged range 0 to 63.5 with step 0.5.
    self._TestOp(-0.1, 63.4, 7, False, 0.0, 63.5, 0.5)

  # 7 bits, narrow range.
  def testOp_with7BitsNarrowRangeNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum 0 and maximum 126: no nudging; expected range 0 to 126
    # with step 1.
    self._TestOp(0.0, 126.0, 7, True, 0.0, 126.0, 1.0)

  def testOp_with7BitsNarrowRangeScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum 0.1 and maximum 63.1: minimum nudged down; expected
    # nudged range 0 to 63 with step 0.5.
    self._TestOp(0.1, 63.1, 7, True, 0.0, 63.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum -63.1 and maximum -0.1: maximum nudged up; expected
    # nudged range -63 to 0 with step 0.5.
    self._TestOp(-63.1, -0.1, 7, True, -63.0, 0.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum -0.1 and maximum 62.9: nudged so zero is representable;
    # expected nudged range 0 to 63 with step 0.5.
    self._TestOp(-0.1, 62.9, 7, True, 0.0, 63.0, 0.5)

  def _TestOp(self, input_min, input_max, num_bits, narrow_range,
              expected_nudged_input_min, expected_nudged_input_max,
              expected_step):
    # [FUNCTION DOC]
    # HELPER: shared verification for all tests in this class.
    # Takes the op minimum and maximum, bit count, narrow-range flag, and the expected nudged
    # minimum, nudged maximum and step. Builds eleven float inputs around the nudged range and
    # upstream gradients 1 to 11. Runs the gradient op on the test device and asserts the
    # backpropagated values pass the gradient through only for inputs inside the nudged range
    # (third through ninth entries) and are zero outside. Returns nothing.
    inputs = np.array(
        [
            expected_nudged_input_min - expected_step,
            expected_nudged_input_min - 0.01, expected_nudged_input_min,
            expected_nudged_input_min + 0.01,
            expected_nudged_input_min + expected_step - 0.01,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step + 0.01,
            expected_nudged_input_max - 0.01, expected_nudged_input_max,
            expected_nudged_input_max + 0.01,
            expected_nudged_input_max + expected_step
        ],
        dtype=np.float32)
    gradients = np.arange(1, len(inputs) + 1, dtype=np.float32)
    expected_backprops = np.array(
        [0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0],
        dtype=np.float32)

    with self.session() as session:
      with self.test_scope():
        gradient_placeholder = array_ops.placeholder(
            dtypes.float32, gradients.shape, name="gradients")
        input_placeholder = array_ops.placeholder(
            dtypes.float32, inputs.shape, name="inputs")
        outputs = gen_array_ops.fake_quant_with_min_max_args_gradient(
            gradient_placeholder,
            input_placeholder,
            min=input_min,
            max=input_max,
            num_bits=num_bits,
            narrow_range=narrow_range)
      backprops = session.run(outputs, {
          gradient_placeholder: gradients,
          input_placeholder: inputs
      })
      self.assertAllCloseAccordingToType(
          backprops,
          expected_backprops,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)


class FakeQuantWithMinMaxVarsTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the forward fake-quantization op whose minimum and maximum are scalar tensor
  # inputs. Test methods pass range parameters to the shared _TestOp helper.
  """Test cases for FakeQuantWithMinMaxVars operation."""

  # 8 bits, wide range.
  def testOp_with8BitsNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum 0 and maximum 255: no scaling or nudging; expected nudged
    # range 0 to 255 with step 1.
    self._TestOp(0.0, 255.0, 8, False, 0.0, 255.0, 1.0)

  def testOp_with8BitsScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum 0.5 and maximum 128: minimum is nudged down to zero; expected
    # nudged range 0 to 127.5 with step 0.5.
    self._TestOp(0.5, 128.0, 8, False, 0.0, 127.5, 0.5)

  def testOp_with8BitsScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum -128 and maximum -0.5: maximum is nudged up to zero; expected
    # nudged range -127.5 to 0 with step 0.5.
    self._TestOp(-128.0, -0.5, 8, False, -127.5, 0.0, 0.5)

  def testOp_with8BitsScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum -0.1 and maximum 127.4: range is nudged so zero is
    # representable; expected nudged range 0 to 127.5 with step 0.5.
    self._TestOp(-0.1, 127.4, 8, False, 0.0, 127.5, 0.5)

  # 8 bits, narrow range.
  def testOp_with8BitsNarrowRangeNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum 0 and maximum 254: no nudging; expected range 0 to 254
    # with step 1.
    self._TestOp(0.0, 254.0, 8, True, 0.0, 254.0, 1.0)

  def testOp_with8BitsNarrowRangeScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum 0.1 and maximum 127.1: minimum nudged down; expected
    # nudged range 0 to 127 with step 0.5.
    self._TestOp(0.1, 127.1, 8, True, 0.0, 127.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum -127.1 and maximum -0.1: maximum nudged up; expected
    # nudged range -127 to 0 with step 0.5.
    self._TestOp(-127.1, -0.1, 8, True, -127.0, 0.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum -0.1 and maximum 126.9: nudged so zero is representable;
    # expected nudged range 0 to 127 with step 0.5.
    self._TestOp(-0.1, 126.9, 8, True, 0.0, 127.0, 0.5)

  # 7 bits, wide range.
  def testOp_with7BitsNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum 0 and maximum 127: no nudging; expected range 0 to 127 with
    # step 1.
    self._TestOp(0.0, 127.0, 7, False, 0.0, 127.0, 1.0)

  def testOp_with7BitsScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum 0.5 and maximum 64: minimum nudged down; expected nudged
    # range 0 to 63.5 with step 0.5.
    self._TestOp(0.5, 64.0, 7, False, 0.0, 63.5, 0.5)

  def testOp_with7BitsScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum -64 and maximum -0.5: maximum nudged up; expected nudged
    # range -63.5 to 0 with step 0.5.
    self._TestOp(-64.0, -0.5, 7, False, -63.5, 0.0, 0.5)

  def testOp_with7BitsScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum -0.1 and maximum 63.4: nudged so zero is representable;
    # expected nudged range 0 to 63.5 with step 0.5.
    self._TestOp(-0.1, 63.4, 7, False, 0.0, 63.5, 0.5)

  # 7 bits, narrow range.
  def testOp_with7BitsNarrowRangeNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum 0 and maximum 126: no nudging; expected range 0 to 126
    # with step 1.
    self._TestOp(0.0, 126.0, 7, True, 0.0, 126.0, 1.0)

  def testOp_with7BitsNarrowRangeScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum 0.1 and maximum 63.1: minimum nudged down; expected
    # nudged range 0 to 63 with step 0.5.
    self._TestOp(0.1, 63.1, 7, True, 0.0, 63.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum -63.1 and maximum -0.1: maximum nudged up; expected
    # nudged range -63 to 0 with step 0.5.
    self._TestOp(-63.1, -0.1, 7, True, -63.0, 0.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum -0.1 and maximum 62.9: nudged so zero is representable;
    # expected nudged range 0 to 63 with step 0.5.
    self._TestOp(-0.1, 62.9, 7, True, 0.0, 63.0, 0.5)

  def _TestOp(self, input_min, input_max, num_bits, narrow_range,
              expected_nudged_input_min, expected_nudged_input_max,
              expected_step):
    # [FUNCTION DOC]
    # HELPER: shared verification for all tests in this class.
    # Takes the minimum and maximum, bit count, narrow-range flag, and expected nudged minimum,
    # nudged maximum and step. Builds the same eleven inputs and expected quantized outputs as the
    # attribute-based variant, but feeds minimum and maximum as scalar placeholders. Asserts the
    # op output is close to the expected values. Returns nothing.
    inputs = np.array(
        [
            expected_nudged_input_min - expected_step,
            expected_nudged_input_min - 0.01, expected_nudged_input_min,
            expected_nudged_input_min + 0.01,
            expected_nudged_input_min + expected_step - 0.01,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step + 0.01,
            expected_nudged_input_max - 0.01, expected_nudged_input_max,
            expected_nudged_input_max + 0.01,
            expected_nudged_input_max + expected_step
        ],
        dtype=np.float32)
    expected = np.array(
        [
            expected_nudged_input_min, expected_nudged_input_min,
            expected_nudged_input_min, expected_nudged_input_min,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_max, expected_nudged_input_max,
            expected_nudged_input_max, expected_nudged_input_max
        ],
        dtype=np.float32)

    with self.session() as session:
      with self.test_scope():
        input_placeholder = array_ops.placeholder(
            dtypes.float32, inputs.shape, name="inputs")
        min_placeholder = array_ops.placeholder(dtypes.float32, (), name="min")
        max_placeholder = array_ops.placeholder(dtypes.float32, (), name="max")
        outputs = array_ops.fake_quant_with_min_max_vars(
            input_placeholder,
            min_placeholder,
            max_placeholder,
            num_bits=num_bits,
            narrow_range=narrow_range)
      result = session.run(
          outputs, {
              input_placeholder: inputs,
              min_placeholder: input_min,
              max_placeholder: input_max
          })
      self.assertAllCloseAccordingToType(
          result, expected, rtol=1e-3, atol=1e-5, bfloat16_rtol=0.03)


class FakeQuantWithMinMaxVarsGradientTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the gradient of the tensor-range fake-quantization op, which returns
  # gradients with respect to the input, the minimum and the maximum. Test methods pass range
  # parameters to the shared _TestOp helper.
  """Test cases for FakeQuantWithMinMaxVarsGradient operation."""

  # 8 bits, wide range.
  def testOp_with8BitsNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum 0 and maximum 255: no scaling or nudging; expected nudged
    # range 0 to 255 with step 1.
    self._TestOp(0.0, 255.0, 8, False, 0.0, 255.0, 1.0)

  def testOp_with8BitsScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum 0.5 and maximum 128: minimum is nudged down to zero; expected
    # nudged range 0 to 127.5 with step 0.5.
    self._TestOp(0.5, 128.0, 8, False, 0.0, 127.5, 0.5)

  def testOp_with8BitsScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum -128 and maximum -0.5: maximum is nudged up to zero; expected
    # nudged range -127.5 to 0 with step 0.5.
    self._TestOp(-128.0, -0.5, 8, False, -127.5, 0.0, 0.5)

  def testOp_with8BitsScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Eight bits, wide range, minimum -0.1 and maximum 127.4: range is nudged so zero is
    # representable; expected nudged range 0 to 127.5 with step 0.5.
    self._TestOp(-0.1, 127.4, 8, False, 0.0, 127.5, 0.5)

  # 8 bits, narrow range.
  def testOp_with8BitsNarrowRangeNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum 0 and maximum 254: no nudging; expected range 0 to 254
    # with step 1.
    self._TestOp(0.0, 254.0, 8, True, 0.0, 254.0, 1.0)

  def testOp_with8BitsNarrowRangeScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum 0.1 and maximum 127.1: minimum nudged down; expected
    # nudged range 0 to 127 with step 0.5.
    self._TestOp(0.1, 127.1, 8, True, 0.0, 127.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum -127.1 and maximum -0.1: maximum nudged up; expected
    # nudged range -127 to 0 with step 0.5.
    self._TestOp(-127.1, -0.1, 8, True, -127.0, 0.0, 0.5)

  def testOp_with8BitsNarrowRangeScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Eight bits, narrow range, minimum -0.1 and maximum 126.9: nudged so zero is representable;
    # expected nudged range 0 to 127 with step 0.5.
    self._TestOp(-0.1, 126.9, 8, True, 0.0, 127.0, 0.5)

  # 7 bits, wide range.
  def testOp_with7BitsNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum 0 and maximum 127: no nudging; expected range 0 to 127 with
    # step 1.
    self._TestOp(0.0, 127.0, 7, False, 0.0, 127.0, 1.0)

  def testOp_with7BitsScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum 0.5 and maximum 64: minimum nudged down; expected nudged
    # range 0 to 63.5 with step 0.5.
    self._TestOp(0.5, 64.0, 7, False, 0.0, 63.5, 0.5)

  def testOp_with7BitsScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum -64 and maximum -0.5: maximum nudged up; expected nudged
    # range -63.5 to 0 with step 0.5.
    self._TestOp(-64.0, -0.5, 7, False, -63.5, 0.0, 0.5)

  def testOp_with7BitsScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Seven bits, wide range, minimum -0.1 and maximum 63.4: nudged so zero is representable;
    # expected nudged range 0 to 63.5 with step 0.5.
    self._TestOp(-0.1, 63.4, 7, False, 0.0, 63.5, 0.5)

  # 7 bits, narrow range.
  def testOp_with7BitsNarrowRangeNoScalingNoNudging(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum 0 and maximum 126: no nudging; expected range 0 to 126
    # with step 1.
    self._TestOp(0.0, 126.0, 7, True, 0.0, 126.0, 1.0)

  def testOp_with7BitsNarrowRangeScalingAndNudgingDown(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum 0.1 and maximum 63.1: minimum nudged down; expected
    # nudged range 0 to 63 with step 0.5.
    self._TestOp(0.1, 63.1, 7, True, 0.0, 63.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingUp(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum -63.1 and maximum -0.1: maximum nudged up; expected
    # nudged range -63 to 0 with step 0.5.
    self._TestOp(-63.1, -0.1, 7, True, -63.0, 0.0, 0.5)

  def testOp_with7BitsNarrowRangeScalingAndNudgingBetween(self):
    # [FUNCTION DOC]
    # Seven bits, narrow range, minimum -0.1 and maximum 62.9: nudged so zero is representable;
    # expected nudged range 0 to 63 with step 0.5.
    self._TestOp(-0.1, 62.9, 7, True, 0.0, 63.0, 0.5)

  def _TestOp(self, input_min, input_max, num_bits, narrow_range,
              expected_nudged_input_min, expected_nudged_input_max,
              expected_step):
    # [FUNCTION DOC]
    # HELPER: shared verification for all tests in this class.
    # Takes the minimum and maximum, bit count, narrow-range flag, and expected nudged range and
    # step. Builds eleven inputs and upstream gradients 1 to 11, feeds minimum and maximum as
    # scalar placeholders, runs the gradient op and asserts three results: input gradient passed
    # through only inside the nudged range, minimum gradient equal to the sum of gradients of the
    # two inputs below the range (3), and maximum gradient equal to the sum for the two above (21).
    # Returns nothing.
    inputs = np.array(
        [
            expected_nudged_input_min - expected_step,
            expected_nudged_input_min - 0.01, expected_nudged_input_min,
            expected_nudged_input_min + 0.01,
            expected_nudged_input_min + expected_step - 0.01,
            expected_nudged_input_min + expected_step,
            expected_nudged_input_min + expected_step + 0.01,
            expected_nudged_input_max - 0.01, expected_nudged_input_max,
            expected_nudged_input_max + 0.01,
            expected_nudged_input_max + expected_step
        ],
        dtype=np.float32)
    gradients = np.arange(1, len(inputs) + 1, dtype=np.float32)
    expected_backprops_wrt_input = np.array(
        [0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0],
        dtype=np.float32)
    expected_backprops_wrt_min = 1.0 + 2.0
    expected_backprops_wrt_max = 10.0 + 11.0

    with self.session() as session:
      with self.test_scope():
        gradient_placeholder = array_ops.placeholder(
            dtypes.float32, gradients.shape, name="gradients")
        input_placeholder = array_ops.placeholder(
            dtypes.float32, inputs.shape, name="inputs")
        min_placeholder = array_ops.placeholder(dtypes.float32, (), name="min")
        max_placeholder = array_ops.placeholder(dtypes.float32, (), name="max")
        outputs = array_ops.fake_quant_with_min_max_vars_gradient(
            gradient_placeholder,
            input_placeholder,
            min_placeholder,
            max_placeholder,
            num_bits=num_bits,
            narrow_range=narrow_range)
      backprops_wrt_input, backprops_wrt_min, backprops_wrt_max = session.run(
          outputs, {
              gradient_placeholder: gradients,
              input_placeholder: inputs,
              min_placeholder: input_min,
              max_placeholder: input_max
          })
      self.assertAllCloseAccordingToType(
          backprops_wrt_input,
          expected_backprops_wrt_input,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)
      self.assertAllCloseAccordingToType(
          backprops_wrt_min,
          expected_backprops_wrt_min,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)
      self.assertAllCloseAccordingToType(
          backprops_wrt_max,
          expected_backprops_wrt_max,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)


class FakeQuantWithMinMaxVarsPerChannelTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the forward per-channel fake-quantization op, where minimum and maximum are
  # vectors with one entry per channel (four channels). Test methods pass per-channel parameter
  # lists to the shared _TestOp helper.
  """Test cases for FakeQuantWithMinMaxVarsPerChannel operation."""

  # 8 bits, wide range.
  def testOp_with8Bits(self):
    # [FUNCTION DOC]
    # Four channels, eight bits, wide range; each channel has its own minimum and maximum covering
    # the no-nudging and nudging cases, with the matching expected nudged ranges and steps.
    self._TestOp(
        [0.0, 0.5, -128.0, -0.1],
        [255.0, 128.0, -0.5, 127.4],
        8,
        False,
        [0.0, 0.0, -127.5, 0.0],
        [255.0, 127.5, 0.0, 127.5],
        [1.0, 0.5, 0.5, 0.5])

  # 8 bits, narrow range.
  def testOp_with8BitsNarrowRange(self):
    # [FUNCTION DOC]
    # Four channels, eight bits, narrow range; per-channel minimums and maximums with expected
    # nudged ranges and steps.
    self._TestOp(
        [0.0, 0.1, -127.1, -0.1],
        [254.0, 127.1, -0.1, 126.9],
        8,
        True,
        [0.0, 0.0, -127.0, 0.0],
        [254.0, 127.0, 0.0, 127.0],
        [1.0, 0.5, 0.5, 0.5])

  # 7 bits, wide range.
  def testOp_with7Bits(self):
    # [FUNCTION DOC]
    # Four channels, seven bits, wide range; per-channel minimums and maximums with expected
    # nudged ranges and steps.
    self._TestOp(
        [0.0, 0.5, -64.0, -0.1],
        [127.0, 64.0, -0.5, 63.4],
        7,
        False,
        [0.0, 0.0, -63.5, 0.0],
        [127.0, 63.5, 0.0, 63.5],
        [1.0, 0.5, 0.5, 0.5])

  # 7 bits, narrow range.
  def testOp_with7BitsNarrowRange(self):
    # [FUNCTION DOC]
    # Four channels, seven bits, narrow range; per-channel minimums and maximums with expected
    # nudged ranges and steps.
    self._TestOp(
        [0.0, 0.1, -63.1, -0.1],
        [126.0, 63.1, -0.1, 62.9],
        7,
        True,
        [0.0, 0.0, -63.0, 0.0],
        [126.0, 63.0, 0.0, 63.0],
        [1.0, 0.5, 0.5, 0.5])

  def _TestOp(self, input_mins, input_maxs, num_bits, narrow_range,
              expected_nudged_input_mins, expected_nudged_input_maxs,
              expected_steps):
    # [FUNCTION DOC]
    # HELPER: shared verification for all tests in this class.
    # Takes lists of per-channel minimums and maximums, bit count, narrow-range flag, and lists of
    # expected per-channel nudged minimums, maximums and steps. For each channel builds eleven
    # inputs and expected quantized outputs, transposes them so channels are the last dimension,
    # runs the per-channel op with vector placeholders for minimum and maximum, and asserts the
    # output is close to the expected values. Returns nothing.
    num_channels = len(input_mins)
    inputs_list = []
    expected_list = []
    for i in range(num_channels):
      expected_nudged_input_min = expected_nudged_input_mins[i]
      expected_nudged_input_max = expected_nudged_input_maxs[i]
      expected_step = expected_steps[i]

      inputs_list.append(
          [
              expected_nudged_input_min - expected_step,
              expected_nudged_input_min - 0.01, expected_nudged_input_min,
              expected_nudged_input_min + 0.01,
              expected_nudged_input_min + expected_step - 0.01,
              expected_nudged_input_min + expected_step,
              expected_nudged_input_min + expected_step + 0.01,
              expected_nudged_input_max - 0.01, expected_nudged_input_max,
              expected_nudged_input_max + 0.01,
              expected_nudged_input_max + expected_step
          ])
      expected_list.append(
          [
              expected_nudged_input_min, expected_nudged_input_min,
              expected_nudged_input_min, expected_nudged_input_min,
              expected_nudged_input_min + expected_step,
              expected_nudged_input_min + expected_step,
              expected_nudged_input_min + expected_step,
              expected_nudged_input_max, expected_nudged_input_max,
              expected_nudged_input_max, expected_nudged_input_max
          ])
    inputs = np.transpose(np.array(inputs_list, dtype=np.float32))
    expected = np.transpose(np.array(expected_list, dtype=np.float32))

    with self.session() as session:
      with self.test_scope():
        input_placeholder = array_ops.placeholder(
            dtypes.float32, inputs.shape, name="inputs")
        min_placeholder = array_ops.placeholder(
            dtypes.float32, (num_channels), name="min")
        max_placeholder = array_ops.placeholder(
            dtypes.float32, (num_channels), name="max")
        outputs = array_ops.fake_quant_with_min_max_vars_per_channel(
            input_placeholder,
            min_placeholder,
            max_placeholder,
            num_bits=num_bits,
            narrow_range=narrow_range)
      result = session.run(
          outputs, {
              input_placeholder: inputs,
              min_placeholder: input_mins,
              max_placeholder: input_maxs
          })
      self.assertAllCloseAccordingToType(
          result, expected, rtol=1e-3, atol=1e-5, bfloat16_rtol=0.03)


class FakeQuantWithMinMaxVarsPerChannelGradientTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the gradient of the per-channel fake-quantization op, checking gradients with
  # respect to input, per-channel minimum and per-channel maximum. Test methods pass per-channel
  # parameter lists to the shared _TestOp helper.
  """Test cases for FakeQuantWithMinMaxVarsPerChannelGradient operation."""

  # 8 bits, wide range.
  def testOp_with8Bits(self):
    # [FUNCTION DOC]
    # Four channels, eight bits, wide range; each channel has its own minimum and maximum covering
    # the no-nudging and nudging cases, with the matching expected nudged ranges and steps.
    self._TestOp(
        [0.0, 0.5, -128.0, -0.1],
        [255.0, 128.0, -0.5, 127.4],
        8,
        False,
        [0.0, 0.0, -127.5, 0.0],
        [255.0, 127.5, 0.0, 127.5],
        [1.0, 0.5, 0.5, 0.5])

  # 8 bits, narrow range.
  def testOp_with8BitsNarrowRange(self):
    # [FUNCTION DOC]
    # Four channels, eight bits, narrow range; per-channel minimums and maximums with expected
    # nudged ranges and steps.
    self._TestOp(
        [0.0, 0.1, -127.1, -0.1],
        [254.0, 127.1, -0.1, 126.9],
        8,
        True,
        [0.0, 0.0, -127.0, 0.0],
        [254.0, 127.0, 0.0, 127.0],
        [1.0, 0.5, 0.5, 0.5])

  # 7 bits, wide range.
  def testOp_with7Bits(self):
    # [FUNCTION DOC]
    # Four channels, seven bits, wide range; per-channel minimums and maximums with expected
    # nudged ranges and steps.
    self._TestOp(
        [0.0, 0.5, -64.0, -0.1],
        [127.0, 64.0, -0.5, 63.4],
        7,
        False,
        [0.0, 0.0, -63.5, 0.0],
        [127.0, 63.5, 0.0, 63.5],
        [1.0, 0.5, 0.5, 0.5])

  # 7 bits, narrow range.
  def testOp_with7BitsNarrowRange(self):
    # [FUNCTION DOC]
    # Four channels, seven bits, narrow range; per-channel minimums and maximums with expected
    # nudged ranges and steps.
    self._TestOp(
        [0.0, 0.1, -63.1, -0.1],
        [126.0, 63.1, -0.1, 62.9],
        7,
        True,
        [0.0, 0.0, -63.0, 0.0],
        [126.0, 63.0, 0.0, 63.0],
        [1.0, 0.5, 0.5, 0.5])

  def _TestOp(self, input_mins, input_maxs, num_bits, narrow_range,
              expected_nudged_input_mins, expected_nudged_input_maxs,
              expected_steps):
    # [FUNCTION DOC]
    # HELPER: shared verification for all tests in this class.
    # Takes lists of per-channel minimums and maximums, bit count, narrow-range flag, and lists of
    # expected per-channel nudged ranges and steps. For each channel builds eleven inputs, upstream
    # gradients 1 to 11 and the expected gradients (input gradient passed inside the nudged range,
    # minimum gradient 3, maximum gradient 21), transposes as needed, runs the per-channel
    # gradient op and asserts all three outputs are close to expectations. Returns nothing.
    num_channels = len(input_mins)
    inputs_list = []
    gradients_list = []
    expected_backprops_wrt_input_list = []
    expected_backprops_wrt_min_list = []
    expected_backprops_wrt_max_list = []
    for i in range(num_channels):
      expected_nudged_input_min = expected_nudged_input_mins[i]
      expected_nudged_input_max = expected_nudged_input_maxs[i]
      expected_step = expected_steps[i]
      inputs = [
          expected_nudged_input_min - expected_step,
          expected_nudged_input_min - 0.01, expected_nudged_input_min,
          expected_nudged_input_min + 0.01,
          expected_nudged_input_min + expected_step - 0.01,
          expected_nudged_input_min + expected_step,
          expected_nudged_input_min + expected_step + 0.01,
          expected_nudged_input_max - 0.01, expected_nudged_input_max,
          expected_nudged_input_max + 0.01,
          expected_nudged_input_max + expected_step
      ]
      inputs_list.append(inputs)
      gradients_list.append(list(range(1, len(inputs) + 1)))
      expected_backprops_wrt_input_list.append(
          [0.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 0.0])
      expected_backprops_wrt_min_list.append(1.0 + 2.0)
      expected_backprops_wrt_max_list.append(10.0 + 11.0)

    inputs = np.transpose(np.array(inputs_list, dtype=np.float32))
    gradients = np.transpose(np.array(gradients_list, dtype=np.float32))
    expected_backprops_wrt_input = np.transpose(np.array(
        expected_backprops_wrt_input_list, dtype=np.float32))
    expected_backprops_wrt_min = np.array(
        expected_backprops_wrt_min_list, dtype=np.float32)
    expected_backprops_wrt_max = np.array(
        expected_backprops_wrt_max_list, dtype=np.float32)

    with self.session() as session:
      with self.test_scope():
        gradient_placeholder = array_ops.placeholder(
            dtypes.float32, gradients.shape, name="gradients")
        input_placeholder = array_ops.placeholder(
            dtypes.float32, inputs.shape, name="inputs")
        min_placeholder = array_ops.placeholder(
            dtypes.float32, (num_channels), name="min")
        max_placeholder = array_ops.placeholder(
            dtypes.float32, (num_channels), name="max")
        outputs = array_ops.fake_quant_with_min_max_vars_per_channel_gradient(
            gradient_placeholder,
            input_placeholder,
            min_placeholder,
            max_placeholder,
            num_bits=num_bits,
            narrow_range=narrow_range)
      backprops_wrt_input, backprops_wrt_min, backprops_wrt_max = session.run(
          outputs, {
              gradient_placeholder: gradients,
              input_placeholder: inputs,
              min_placeholder: input_mins,
              max_placeholder: input_maxs
          })
      self.assertAllCloseAccordingToType(
          backprops_wrt_input,
          expected_backprops_wrt_input,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)
      self.assertAllCloseAccordingToType(
          backprops_wrt_min,
          expected_backprops_wrt_min,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)
      self.assertAllCloseAccordingToType(
          backprops_wrt_max,
          expected_backprops_wrt_max,
          rtol=1e-3,
          atol=1e-5,
          bfloat16_rtol=0.03)


if __name__ == "__main__":
  googletest.main()
