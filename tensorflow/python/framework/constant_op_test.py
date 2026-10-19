# [MODULE DOC]
# Unit tests for the constant creation function in the framework constant module. Covers
# rejection of string input for numeric dtypes, copy semantics for numpy array input, and the
# internal eager-constant operation under XLA compilation, gradient computation and vectorized
# mapping. Run as a test binary; the main entry enables eager execution first.
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
"""Tests for tensorflow.python.framework.constant_op."""

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.platform import test


class ConstantOpTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (with parameterized support) covering the constant creation function: dtype
  # validation, numpy memory isolation and the internal eager-constant graph operation. It holds
  # no state of its own beyond a helper that parses text-format graph definitions.

  @parameterized.parameters(
      dtypes.bfloat16,
      dtypes.complex128,
      dtypes.complex64,
      dtypes.double,
      dtypes.float16,
      dtypes.float32,
      dtypes.float64,
      dtypes.half,
      dtypes.int16,
      dtypes.int32,
      dtypes.int64,
      dtypes.int8,
      dtypes.qint16,
      dtypes.qint32,
      dtypes.qint8,
      dtypes.quint16,
      dtypes.quint8,
      dtypes.uint16,
      dtypes.uint32,
      dtypes.uint64,
      dtypes.uint8,
  )
  def test_convert_string_to_number(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over many numeric and quantized dtypes. Scenario: build a constant from the
    # string "hello" with the given dtype. Expected: a type error is raised for every dtype.
    with self.assertRaises(TypeError):
      constant_op.constant("hello", dtype)

  def _make_graph_def(self, text):
    # [FUNCTION DOC]
    # HELPER: builds graph definitions for the eager-constant tests in this class.
    # Parses the given text-format graph definition string into a new graph definition message
    # and returns it. Raises a parse error from the protobuf text parser if the text is invalid.
    ret = graph_pb2.GraphDef()
    text_format.Parse(text, ret)
    return ret

  def test_eager_const_xla(self):
    # [FUNCTION DOC]
    # Scenario: inside an XLA-compiled function, import a graph containing a NaN constant and an
    # internal eager-constant node whose input is mapped to the function argument. Action: call
    # the function with the constant 3.14. Expected: the result is close to 3.14, i.e. the
    # eager-constant node passes its input through under XLA compilation.

    @def_function.function(jit_compile=True)
    def f_using_eagerconst(x):
      # [FUNCTION DOC]
      # Nested compiled function: imports a graph def with an eager-constant node, mapping its
      # input to argument x, and returns the imported node's output tensor.
      graph_def = self._make_graph_def("""
         node { name: 'x' op: 'Const'
           attr { key: 'dtype' value { type: DT_FLOAT } }
           attr { key: 'value' value { tensor {
             dtype: DT_FLOAT tensor_shape {} float_val: NaN } } } }
         node { name: 'const' op: '_EagerConst' input: 'x:0'
                attr { key: 'T' value { type: DT_FLOAT } }}""")
      x_id = importer.import_graph_def(
          graph_def,
          input_map={"x:0": x},
          return_elements=["const"],
          name="import")[0].outputs[0]
      return x_id

    self.assertAllClose(3.14, f_using_eagerconst(constant_op.constant(3.14)))

  def test_np_array_memory_not_shared(self):
    # [FUNCTION DOC]
    # Scenario: repeated many times (10000), create a numpy range array, make a constant from it,
    # then mutate one element of the array. Expected: the constant's value at that index keeps
    # its original value, showing the constant does not share memory with the input array.
    # An arbitrarily large loop number to test memory sharing
    for _ in range(10000):
      x = np.arange(10)
      xt = constant_op.constant(x)
      x[3] = 42
      # Changing the input array after `xt` is created should not affect `xt`
      self.assertEqual(xt.numpy()[3], 3)

  def test_eager_const_grad_error(self):
    # [FUNCTION DOC]
    # Scenario: inside a traced function, import a graph with an eager-constant node fed by a
    # constant and then request gradients of its output with respect to that constant. Expected:
    # calling the function raises an assertion error whose message asks to file a bug, because
    # the eager-constant operation has no gradient defined.

    @def_function.function
    def f_using_eagerconst():
      # [FUNCTION DOC]
      # Nested traced function: creates a constant, imports an eager-constant node reading it,
      # computes gradients of the node output with respect to the constant (which is expected to
      # fail), and otherwise returns the node output.
      x = constant_op.constant(1.)
      graph_def = self._make_graph_def("""
         node { name: 'x' op: 'Placeholder'
                attr { key: 'dtype' value { type: DT_FLOAT } }}
         node { name: 'const' op: '_EagerConst' input: 'x:0'
                attr { key: 'T' value { type: DT_FLOAT } }}""")
      x_id = importer.import_graph_def(
          graph_def,
          input_map={"x:0": x},
          return_elements=["const"],
          name="import")[0].outputs[0]
      gradients_impl.gradients(x_id, x)
      return x_id

    with self.assertRaisesRegex(AssertionError, "Please file a bug"):
      f_using_eagerconst()

  def test_eager_const_pfor(self):
    # [FUNCTION DOC]
    # Scenario: inside a traced function, vectorize (without falling back to a while loop) a
    # per-element function that imports an eager-constant node over a constant 3.14, applied to
    # the vector of one and two. Expected: the result is close to the vector one, two, meaning the
    # eager-constant node is vectorizable and passes through its mapped input.

    @def_function.function
    def f_using_eagerconst():
      # [FUNCTION DOC]
      # Nested traced function: runs the vectorized map of the per-element function over a
      # two-element constant vector, disallowing the while-loop fallback, and returns the result.

      def vec_fn(x):
        # [FUNCTION DOC]
        # Per-element function for the vectorized map: imports a graph with a constant and an
        # eager-constant node whose input is mapped to x, and returns the imported node output.
        graph_def = self._make_graph_def("""
           node { name: 'x' op: 'Const'
             attr { key: 'dtype' value { type: DT_FLOAT } }
             attr { key: 'value' value { tensor {
               dtype: DT_FLOAT tensor_shape {} float_val: 3.14 } } } }
           node { name: 'const' op: '_EagerConst' input: 'x:0'
                  attr { key: 'T' value { type: DT_FLOAT } }}""")
        return importer.import_graph_def(
            graph_def,
            input_map={"x:0": x},
            return_elements=["const"],
            name="import")[0].outputs[0]

      return control_flow_ops.vectorized_map(
          vec_fn, constant_op.constant([1., 2.]), fallback_to_while_loop=False)

    self.assertAllClose([1., 2.], f_using_eagerconst())


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
