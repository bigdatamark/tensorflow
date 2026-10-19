# [MODULE DOC]
# Tests for the XLA custom-call op wrappers (the original and the second-generation variant).
# Each test compiles a small function with jit and inspects the generated HLO text for the expected
# custom-call target, shapes, side-effect flag, API version and backend configuration. When run as a
# script it enables eager execution before starting the tests. No other import-time side effects.
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
"""Tests for XLA custom call op wrapper."""

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class XlaCustomCallOpTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case for the XLA custom-call ops. Inherits device selection from the XLA test base; each
  # test places work on the XLA device under test, compiles with jit and examines the HLO text.

  def testXlaCustomCallOp(self):
    # [FUNCTION DOC]
    # Builds a function calling the custom-call op with two inputs, a target name, an int32 result
    # of shape three by four by five and a backend configuration string. Compiles it with jit on
    # the test device, retrieves the HLO text and asserts it contains the custom call with the
    # expected result and operand types, the target name and the backend configuration.
    with ops.device('device:{}:0'.format(self.device)):

      def f(x, y):
        # [FUNCTION DOC]
        # Calls the custom-call op on both inputs with fixed target, type, shape and backend
        # configuration; returns the resulting tensor.
        return xla.custom_call(
            args=(x, y),
            target_name='my_call',
            dtype=dtypes.int32,
            shape=(3, 4, 5),
            backend_config='my_backend_config')

      compiled_f = def_function.function(f, jit_compile=True)

      x = random_ops.random_normal([1, 2, 3], dtype=dtypes.float32)
      y = random_ops.random_normal([], dtype=dtypes.float32)
      hlo = compiled_f.experimental_get_compiler_ir(x, y)(stage='hlo')
      self.assertIn('s32[3,4,5]{2,1,0} custom-call(f32[1,2,3]{2,1,0}', hlo)
      self.assertIn('custom_call_target="my_call"', hlo)
      self.assertIn('backend_config="my_backend_config"', hlo)

  def testXlaCustomCallOpDoesntExist(self):
    # [FUNCTION DOC]
    # Uses the custom-call op with a target name that is not registered anywhere. Compiling and
    # running it with jit on the test device must raise a TensorFlow operation error, which the
    # test asserts.
    with ops.device('device:{}:0'.format(self.device)):

      def f():
        # [FUNCTION DOC]
        # Calls the custom-call op with two constant arguments and a non-existent target name,
        # an int32 scalar result and a backend configuration; returns the resulting tensor.
        return xla.custom_call(
            args=(1, 2),
            target_name='my_non_existing_call_target',
            dtype=dtypes.int32,
            shape=(),
            backend_config='my_backend_config',
        )

      with self.assertRaises(errors_impl.OpError):
        compiled_f = def_function.function(f, jit_compile=True)
        compiled_f()

  def testXlaCustomCallV2Op(self):
    # [FUNCTION DOC]
    # Builds a function calling the second-generation custom-call op with two inputs, two result
    # specs (int32 two by three and float32 vector of five), side-effect flag set and a backend
    # configuration. Compiles with jit, gets the HLO text and asserts, in order, the tuple result
    # type, operand types, target name, side-effect flag, status-returning unified API version and
    # backend configuration.
    with ops.device('device:{}:0'.format(self.device)):

      def f(x, y):
        # [FUNCTION DOC]
        # Calls the second-generation custom-call op on both inputs; returns its two result
        # tensors.
        return xla.custom_call_v2(
            'my_call',
            (x, y),
            (
                tensor_spec.TensorSpec((2, 3), dtypes.int32),
                tensor_spec.TensorSpec((5,), dtypes.float32),
            ),
            has_side_effect=True,
            backend_config='my_backend_config',
        )

      compiled_f = def_function.function(f, jit_compile=True)

      x = random_ops.random_normal([7, 11], dtype=dtypes.float32)
      y = random_ops.random_normal([13, 17, 19], dtype=dtypes.float32)
      hlo = compiled_f.experimental_get_compiler_ir(x, y)(stage='hlo')
      self.assertContainsInOrder([
          '= (s32[2,3]{1,0}, f32[5]{0}) custom-call(',
          'f32[7,11]{1,0}',
          'f32[13,17,19]{2,1,0}',
          'custom_call_target="my_call"',
          'custom_call_has_side_effect=true',
          'api_version=API_VERSION_STATUS_RETURNING_UNIFIED',
          'backend_config="my_backend_config"',
      ], hlo)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
