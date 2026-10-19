# [MODULE DOC]
# Tests for the jit_compile option of the function decorator: basic compilation, gradients,
# integer inputs and error reporting for ops XLA cannot compile. Test module; no side effects.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class JitCompileTest(test.TestCase):
  # [CLASS DOC]
  # Tests that the jit-compile option of function wrapping marks functions for mandatory XLA
  # compilation, propagates to derivatives, and surfaces compiler errors. No shared fixtures.

  def testBasic(self):
    # [FUNCTION DOC]
    # In a fresh graph, wraps an add function with jit compilation enabled, feeds a float vector
    # of five values through a session with the constant one, and asserts the output is each
    # element plus one and that the function definition carries the must-compile attribute.
    with ops.Graph().as_default() as g:

      def fn(x, a):
        # [FUNCTION DOC]
        # HELPER: function under test; returns the sum of its two arguments.
        return x + a

      xla_func = def_function.function(fn, jit_compile=True)
      inputs = array_ops.placeholder(dtypes.float32, [5])
      x = xla_func(inputs, 1)
      with session.Session(graph=g) as sess:
        y = sess.run(x, feed_dict={inputs: [1, 2, 2, 3, 3]})
        self.assertTrue(x.graph.as_graph_def().library.function[0]
                        .attr["_XlaMustCompile"].b)
        self.assertAllClose([2, 3, 3, 4, 4], y)

  def testDerivative(self):
    # [FUNCTION DOC]
    # Builds a jit-compiled function computing two times x plus a, takes its gradient with respect
    # to a placeholder with a gradient tape, and asserts the gradient is two for every element.
    # It then fetches the forward and backward functions and asserts the must-compile attribute
    # is present on both (the check on the attribute mapping is only a truthiness check).
    def fn(x, a):
      # [FUNCTION DOC]
      # HELPER: function under test; returns twice the first argument plus the second.
      return 2 * x + a

    with ops.Graph().as_default() as g:
      xla_func = def_function.function(fn, jit_compile=True)
      with backprop.GradientTape() as tape:
        inputs = array_ops.placeholder(dtypes.float32, [5])
        tape.watch(inputs)
        outputs = xla_func(inputs, 1)
      grads = tape.gradient(outputs, inputs)

    with session.Session(graph=g) as sess:
      grads_tensor = sess.run(grads, feed_dict={inputs: [1, 2, 2, 3, 3]})
      self.assertAllClose([2, 2, 2, 2, 2], grads_tensor)
      (forward, backward) = xla_func.get_concrete_function(
          inputs, 1)._delayed_rewrite_functions.forward_backward()

      # Check that the must-compile attribute gets correctly propagated to the
      # created derivatives.
      self.assertTrue(forward.cached_definition.attr["_XlaMustCompile"])
      self.assertTrue(backward.function_def.attr["_XlaMustCompile"])

  def testBasicInt32(self):
    # [FUNCTION DOC]
    # Same scenario as the basic test but with a 32-bit integer placeholder: asserts the jit
    # compiled add yields each element plus one and the must-compile attribute is set.
    with ops.Graph().as_default() as g:

      def fn(x, a):
        # [FUNCTION DOC]
        # HELPER: function under test; returns the sum of its two arguments.
        return x + a

      xla_func = def_function.function(fn, jit_compile=True)
      inputs = array_ops.placeholder(dtypes.int32, [5])
      x = xla_func(inputs, 1)
      with session.Session(graph=g) as sess:
        y = sess.run(x, feed_dict={inputs: [1, 2, 2, 3, 3]})
        self.assertTrue(x.graph.as_graph_def().library.function[0]
                        .attr["_XlaMustCompile"].b)
        self.assertAllClose([2, 3, 3, 4, 4], y)

  # Checking that we crash on an unsupported operation lets us test that the XLA
  # compiler was actually invoked.
  def testUnsupportedOps(self):
    # [FUNCTION DOC]
    # Jit compiles a function that formats a tensor into a string and takes its length (string ops
    # unsupported by XLA) and asserts that running it raises an invalid-argument error mentioning
    # unsupported operations, proving the XLA compiler was invoked.
    with ops.Graph().as_default() as g:

      def fn(x):
        # [FUNCTION DOC]
        # HELPER: function under test; formats the input as a string and returns its length.
        return string_ops.string_length(
            string_ops.string_format('{}', x))

      xla_func = def_function.function(fn, jit_compile=True)
      inputs = array_ops.placeholder(dtypes.float32, [5])
      x = xla_func(inputs)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Detected unsupported operations"):
        with session.Session(graph=g) as sess:
          sess.run(x, feed_dict={inputs: [1, 2, 2, 3, 3]})


if __name__ == "__main__":
  test.main()
