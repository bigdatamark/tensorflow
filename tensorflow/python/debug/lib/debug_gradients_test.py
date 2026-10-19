# [MODULE DOC]
# Unit tests for the debug_gradients module (GradientsDebugger and gradient_values_from_dump).
# Covers identifying gradients with and without a context manager, multiple losses and debuggers,
# error cases, watching gradients by tensor names or tensor objects, reference-typed variables,
# and reading gradient values from a debug dump. Graph/session based, so run only under TF1
# behavior. Test module only.
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
"""Unit tests for debug_gradients module."""

import tempfile

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_gradients
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent


@test_util.run_v1_only("Sessions are not available in TF 2.x")
class IdentifyGradientTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (TF1 only, uses sessions) for GradientsDebugger. The fixture builds a session with
  # model pruning and dependency optimization disabled and a graph with variables u=2, v=3 and
  # w=u*v; tests differentiate expressions of w and check the gradient tensors the debugger holds.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: creates a session whose rewriter config disables model pruning and dependency
    # optimization, then (in that session) creates variables u (2.0), v (3.0) and w, the product
    # of their values. Stores the session and tensors on the test instance.
    rewriter_config = rewriter_config_pb2.RewriterConfig(
        disable_model_pruning=True,
        dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    config = config_pb2.ConfigProto(graph_options=graph_options)
    self.sess = session.Session(config=config)
    with self.sess.as_default():
      self.u = variables.Variable(2.0, name="u")
      self.v = variables.Variable(3.0, name="v")
      self.w = math_ops.multiply(self.u.value(), self.v.value(), name="w")

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: resets the default graph and clears the global registry of gradient
    # debuggers so tests do not interfere with each other.
    ops.reset_default_graph()
    debug_gradients.clear_gradient_debuggers()

  def testIdentifyGradientGivesCorrectTensorObjectWithoutContextManager(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the last two checks both fetch by name, so the third repeats the second.
    # Scenario: wraps w with identify_gradient, builds y from it, takes gradients without entering
    # the debugger as a context (allowed because only one debugger exists). Asserts values of y
    # and the gradients with respect to u and v, and that the gradient of w is fetchable by tensor
    # and by name with value 1.
    grad_debugger = debug_gradients.GradientsDebugger()
    id_grad_w = grad_debugger.identify_gradient(self.w)
    y = math_ops.add(id_grad_w, -1.0, name="y")

    grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]
    v_grad = grads[1]

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0, self.sess.run(y))
    self.assertAllClose(3.0, self.sess.run(u_grad))
    self.assertAllClose(2.0, self.sess.run(v_grad))

    # Fetch the gradient tensor with the x-tensor object.
    w_grad = grad_debugger.gradient_tensor(self.w)
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    # Fetch the gradient tensor with the x-tensor's name.
    w_grad = grad_debugger.gradient_tensor(self.w.name)
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    # Fetch the gradient tensor with the x-tensor name.
    w_grad = grad_debugger.gradient_tensor(self.w.name)
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

  def testIdentifyGradientGivesCorrectTensorObjectWithTfGradients(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the last two checks both fetch by name, so the third repeats the second.
    # Scenario: same as the no-context-manager test but the gradient call is made inside the
    # debugger's context. Asserts the same values and that the gradient of w is retrievable by
    # tensor object and by name.
    grad_debugger = debug_gradients.GradientsDebugger()
    id_grad_w = grad_debugger.identify_gradient(self.w)
    y = math_ops.add(id_grad_w, -1.0, name="y")

    with grad_debugger:
      grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]
    v_grad = grads[1]

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0, self.sess.run(y))
    self.assertAllClose(3.0, self.sess.run(u_grad))
    self.assertAllClose(2.0, self.sess.run(v_grad))

    # Fetch the gradient tensor with the x-tensor object.
    w_grad = grad_debugger.gradient_tensor(self.w)
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    # Fetch the gradient tensor with the x-tensor's name.
    w_grad = grad_debugger.gradient_tensor(self.w.name)
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    # Fetch the gradient tensor with the x-tensor name.
    w_grad = grad_debugger.gradient_tensor(self.w.name)
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

  def testCallingIdentifyGradientTwiceWithTheSameGradientsDebuggerErrors(self):
    # [FUNCTION DOC]
    # Scenario: identify_gradient is called twice on the same tensor with one debugger. Asserts the
    # second call raises a value error saying the graph already contains an op with that name.
    grad_debugger = debug_gradients.GradientsDebugger()
    grad_debugger.identify_gradient(self.w)
    with self.assertRaisesRegex(ValueError,
                                "The graph already contains an op named .*"):
      grad_debugger.identify_gradient(self.w)

  def testIdentifyGradientWorksOnMultipleLosses(self):
    # [FUNCTION DOC]
    # Scenario: two debuggers each wrap the same tensor y; two losses (square and square root) are
    # minimized, each inside its own debugger context. Asserts each debugger holds a distinct
    # gradient tensor for y, with loss values and gradients (2*5 and 0.5*5^-0.5) as expected.
    grad_debugger_1 = debug_gradients.GradientsDebugger()
    grad_debugger_2 = debug_gradients.GradientsDebugger()

    y = math_ops.add(self.w, -1.0, name="y")
    debug_y = grad_debugger_1.identify_gradient(y)
    z1 = math_ops.square(debug_y, name="z1")

    debug_y = grad_debugger_2.identify_gradient(y)
    z2 = math_ops.sqrt(debug_y, name="z2")

    with grad_debugger_1:
      gradient_descent.GradientDescentOptimizer(0.1).minimize(z1)
    with grad_debugger_2:
      gradient_descent.GradientDescentOptimizer(0.1).minimize(z2)

    dz1_dy = grad_debugger_1.gradient_tensor(y)
    dz2_dy = grad_debugger_2.gradient_tensor(y)
    self.assertIsInstance(dz1_dy, tensor.Tensor)
    self.assertIsInstance(dz2_dy, tensor.Tensor)
    self.assertIsNot(dz1_dy, dz2_dy)

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0**2, self.sess.run(z1))
    self.assertAllClose(5.0**0.5, self.sess.run(z2))
    self.assertAllClose(2.0 * 5.0, self.sess.run(dz1_dy))
    self.assertAllClose(0.5 * (5.0**-0.5), self.sess.run(dz2_dy))

  def testIdentifyGradientRaisesLookupErrorForUnknownXTensor(self):
    # [FUNCTION DOC]
    # Scenario: two debuggers exist, one wraps w, but gradients are computed outside any debugger
    # context so none are registered. Asserts a lookup error (no gradient received) is raised by
    # both debuggers when asked for the gradient of w.
    grad_debugger_1 = debug_gradients.GradientsDebugger()
    grad_debugger_2 = debug_gradients.GradientsDebugger()
    id_grad_w = grad_debugger_1.identify_gradient(self.w)
    y = math_ops.add(id_grad_w, -1.0, name="y")

    # There are >1 gradient debuggers registered, and grad_debugger is not used
    # as a context manager here, so the gradient w.r.t. self.w will not be
    # registered.
    gradients_impl.gradients(y, [self.u, self.v])

    with self.assertRaisesRegex(
        LookupError,
        r"This GradientsDebugger has not received any gradient tensor for "):
      grad_debugger_1.gradient_tensor(self.w)
    with self.assertRaisesRegex(
        LookupError,
        r"This GradientsDebugger has not received any gradient tensor for "):
      grad_debugger_2.gradient_tensor(self.w)

  def testIdentifyGradientRaisesTypeErrorForNonTensorOrTensorNameInput(self):
    # [FUNCTION DOC]
    # Scenario: asks for a gradient tensor using an operation (the variables initializer) as the
    # key. Asserts a type error whose message says the argument must be a string, tensor or
    # variable and names the operation type.
    grad_debugger = debug_gradients.GradientsDebugger()
    with self.assertRaisesRegex(
        TypeError,
        r"x_tensor must be a str or tf\.Tensor or tf\.Variable, but instead "
        r"has type .*Operation.*"):
      grad_debugger.gradient_tensor(variables.global_variables_initializer())

  def testIdentifyGradientTensorWorksWithGradientDescentOptimizer(self):
    # [FUNCTION DOC]
    # Scenario: wraps w, then minimizes y with a gradient descent optimizer inside the debugger
    # context. Asserts the gradient tensor of w can be fetched and evaluates to 1.
    grad_debugger = debug_gradients.GradientsDebugger()
    id_grad_w = grad_debugger.identify_gradient(self.w)
    y = math_ops.add(id_grad_w, -1.0, name="y")

    with grad_debugger:
      gradient_descent.GradientDescentOptimizer(0.1).minimize(y)

    self.sess.run(variables.global_variables_initializer())

    # Fetch the gradient tensor with the x-tensor object.
    w_grad = grad_debugger.gradient_tensor(self.w)
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

  def testWatchGradientsByXTensorNamesWorks(self):
    # [FUNCTION DOC]
    # Scenario: after the forward graph is built, watches tensors matching the pattern for w using
    # the by-names method as a context manager while taking gradients. Asserts y and the gradients
    # with respect to u and v, and that the gradient of w is retrievable by tensor and by name.
    y = math_ops.add(self.w, -1.0, name="y")

    # The constructrion of the forward graph has completed.
    # But we can still get the gradient tensors by using
    # watch_gradients_by_tensor_names().
    grad_debugger = debug_gradients.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensor_names(self.sess.graph, "w:0$"):
      grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]
    v_grad = grads[1]

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0, self.sess.run(y))
    self.assertAllClose(3.0, self.sess.run(u_grad))
    self.assertAllClose(2.0, self.sess.run(v_grad))

    w_grad = grad_debugger.gradient_tensor(self.w)
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    w_grad = grad_debugger.gradient_tensor("w:0")
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

  def testWatchGradientsByXTensorNamesWorksWithoutContextManager(self):
    # [FUNCTION DOC]
    # Scenario: like the by-names test but the watch call is made without a with-statement (only
    # one debugger exists, so registration still happens). Asserts the same values and lookups.
    y = math_ops.add(self.w, -1.0, name="y")

    # The constructrion of the forward graph has completed.
    # But we can still get the gradient tensors by using
    # watch_gradients_by_tensor_names().
    grad_debugger = debug_gradients.GradientsDebugger()
    grad_debugger.watch_gradients_by_tensor_names(self.sess.graph, "w:0$")
    grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]
    v_grad = grads[1]

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0, self.sess.run(y))
    self.assertAllClose(3.0, self.sess.run(u_grad))
    self.assertAllClose(2.0, self.sess.run(v_grad))

    w_grad = grad_debugger.gradient_tensor(self.w)
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

    w_grad = grad_debugger.gradient_tensor("w:0")
    self.assertIsInstance(w_grad, tensor.Tensor)
    self.assertAllClose(1.0, self.sess.run(w_grad))

  def testWatchGradientsWorksOnRefTensor(self):
    # [FUNCTION DOC]
    # Scenario: watches the reference-typed variable u by name pattern while taking gradients.
    # Asserts the gradient of u returned by tf gradients is the very object the debugger holds,
    # and that u and v gradients evaluate to 3 and 2.
    y = math_ops.add(self.w, -1.0, name="y")

    grad_debugger = debug_gradients.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensor_names(self.sess.graph, "u:0$"):
      grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]
    v_grad = grads[1]

    self.assertIs(u_grad, grad_debugger.gradient_tensor("u:0"))

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(3.0, self.sess.run(u_grad))
    self.assertAllClose(2.0, self.sess.run(v_grad))
    self.assertAllClose(3.0, self.sess.run(
        grad_debugger.gradient_tensor("u:0")))

  def testWatchGradientsWorksOnMultipleTensors(self):
    # [FUNCTION DOC]
    # Scenario: a pattern matching both u and w is watched. Asserts exactly two gradient tensors
    # are recorded, the one for u is the object tf gradients returned, and values are 3 (u) and
    # 1 (w).
    y = math_ops.add(self.w, -1.0, name="y")

    grad_debugger = debug_gradients.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensor_names(self.sess.graph,
                                                       "(u|w):0$"):
      grads = gradients_impl.gradients(y, [self.u, self.v])
    self.assertEqual(2, len(grads))
    u_grad = grads[0]

    self.assertEqual(2, len(grad_debugger.gradient_tensors()))
    self.assertIs(u_grad, grad_debugger.gradient_tensor("u:0"))
    self.assertIsInstance(grad_debugger.gradient_tensor("w:0"), tensor.Tensor)

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(1.0, self.sess.run(
        grad_debugger.gradient_tensor("w:0")))
    self.assertAllClose(3.0, self.sess.run(
        grad_debugger.gradient_tensor("u:0")))

  def testWatchGradientsByXTensorsWorks(self):
    # [FUNCTION DOC]
    # Scenario: after building a graph with scoped op names, watches a list of tensors (w, the
    # variable u, and y) by object and minimizes the squared loss. Asserts three gradient tensors
    # are recorded with values 10 (y), 10 (w) and 30 (u).
    y = math_ops.add(self.w, -1.0, name="foo/y")
    z = math_ops.square(y, name="foo/z")

    # The constructrion of the forward graph has completed.
    # But we can still get the gradient tensors by using
    # watch_gradients_by_x_tensors().
    grad_debugger = debug_gradients.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensors(self.sess.graph,
                                                  [self.w, self.u, y]):
      gradient_descent.GradientDescentOptimizer(0.1).minimize(z)

    self.assertEqual(3, len(grad_debugger.gradient_tensors()))
    u_grad = grad_debugger.gradient_tensor(self.u)
    w_grad = grad_debugger.gradient_tensor(self.w)
    y_grad = grad_debugger.gradient_tensor(y)

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(10.0, self.sess.run(y_grad))
    self.assertAllClose(10.0, self.sess.run(w_grad))
    self.assertAllClose(30.0, self.sess.run(u_grad))

  def testWatchGradientsByTensorCanWorkOnMultipleLosses(self):
    # [FUNCTION DOC]
    # Scenario: two debuggers each watch the single tensor y by object, each for its own loss
    # (square, square root). Asserts distinct gradient tensors and the expected loss values and
    # gradients.
    y = math_ops.add(self.w, -1.0, name="y")
    z1 = math_ops.square(y, name="z1")
    z2 = math_ops.sqrt(y, name="z2")

    grad_debugger_1 = debug_gradients.GradientsDebugger()
    with grad_debugger_1.watch_gradients_by_tensors(self.sess.graph, y):
      gradient_descent.GradientDescentOptimizer(0.1).minimize(z1)

    grad_debugger_2 = debug_gradients.GradientsDebugger()
    with grad_debugger_2.watch_gradients_by_tensors(self.sess.graph, y):
      gradient_descent.GradientDescentOptimizer(0.1).minimize(z2)

    dz1_dy = grad_debugger_1.gradient_tensor(y)
    dz2_dy = grad_debugger_2.gradient_tensor(y)
    self.assertIsInstance(dz1_dy, tensor.Tensor)
    self.assertIsInstance(dz2_dy, tensor.Tensor)
    self.assertIsNot(dz1_dy, dz2_dy)

    self.sess.run(variables.global_variables_initializer())
    self.assertAllClose(5.0**2, self.sess.run(z1))
    self.assertAllClose(5.0**0.5, self.sess.run(z2))
    self.assertAllClose(2.0 * 5.0, self.sess.run(dz1_dy))
    self.assertAllClose(0.5 * (5.0**-0.5), self.sess.run(dz2_dy))

  def testGradientsValuesFromDumpWorks(self):
    # [FUNCTION DOC]
    # Scenario: watches w, u and y for a squared loss, runs one training step with debug watches
    # dumping to a temporary directory, and builds a dump object with the partition graphs and
    # the Python graph. Asserts gradient_values_from_dump returns one value each for y (10), w (10)
    # and u (30), that u was updated to -1, and that asking for v raises a lookup error. Removes
    # the temporary directory at the end.
    y = math_ops.add(self.w, -1.0, name="y")
    z = math_ops.square(y, name="z")

    grad_debugger = debug_gradients.GradientsDebugger()
    with grad_debugger.watch_gradients_by_tensors(self.sess.graph,
                                                  [self.w, self.u, y]):
      train_op = gradient_descent.GradientDescentOptimizer(0.1).minimize(z)

    self.sess.run(variables.global_variables_initializer())

    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    dump_dir = tempfile.mkdtemp()
    debug_url = "file://" + dump_dir
    debug_utils.watch_graph(run_options, self.sess.graph, debug_urls=debug_url)
    run_metadata = config_pb2.RunMetadata()
    self.assertAllClose(2.0, self.sess.run(self.u))
    self.sess.run(train_op, options=run_options, run_metadata=run_metadata)
    self.assertAllClose(-1.0, self.sess.run(self.u))

    dump = debug_data.DebugDumpDir(
        dump_dir, partition_graphs=run_metadata.partition_graphs)
    dump.set_python_graph(self.sess.graph)

    y_grad_values = debug_gradients.gradient_values_from_dump(
        grad_debugger, y, dump)
    self.assertEqual(1, len(y_grad_values))
    self.assertAllClose(10.0, y_grad_values[0])

    w_grad_values = debug_gradients.gradient_values_from_dump(
        grad_debugger, self.w, dump)
    self.assertEqual(1, len(w_grad_values))
    self.assertAllClose(10.0, w_grad_values[0])

    u_grad_values = debug_gradients.gradient_values_from_dump(
        grad_debugger, self.u, dump)
    self.assertEqual(1, len(u_grad_values))
    self.assertAllClose(30.0, u_grad_values[0])

    with self.assertRaisesRegex(
        LookupError,
        r"This GradientsDebugger has not received any gradient tensor for "
        r"x-tensor v:0"):
      debug_gradients.gradient_values_from_dump(grad_debugger, self.v, dump)

    # Cleanup.
    file_io.delete_recursively(dump_dir)


if __name__ == "__main__":
  googletest.main()
