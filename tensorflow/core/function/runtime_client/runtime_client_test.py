# [MODULE DOC]
# Tests for the runtime client: creating functions from protobuf definitions, reading back and
# editing the definitions of traced functions, running native compiler passes (graph dialect,
# TF dialect and mixed) on functions, and transforming functions in a multi-worker local
# cluster. Most tests are skipped when TF2 behavior is disabled. The main block disables soft
# device placement and registers the test passes before running.
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
"""Tests for runtime_client."""

from google.protobuf import text_format
from tensorflow.core.framework import function_pb2
from tensorflow.core.function.runtime_client import runtime_client
from tensorflow.core.function.testing import test_pass
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class RuntimeClientTest(test.TestCase):
  # [CLASS DOC]
  # Single-process tests for the Runtime client; no shared fixtures.

  def test_create_nullary(self):
    # [FUNCTION DOC]
    # Parses a text-format definition of a function with no inputs returning constant 1,
    # creates it through a runtime built on the global eager context; asserts only that
    # creation succeeds without error.
    fndef = text_format.Parse(
        """
            signature {
               name: 'NullaryFunction'
               output_arg { name: 'o' type: DT_INT32 }
             }
             node_def {
               name: 'retval'
               op: 'Const'
               attr {
                 key: 'dtype'
                 value { type: DT_INT32 }
               }
               attr {
                 key: 'value'
                 value {
                   tensor {
                     dtype: DT_INT32
                     tensor_shape {}
                     int_val: 1
                   }
                 }
               }
             }
             ret { key: 'o' value: 'retval:output' }
         """,
        function_pb2.FunctionDef(),
    )

    ctx = runtime_client.GlobalEagerContext()
    rt = runtime_client.Runtime(ctx)
    rt.CreateFunction(fndef)

  def test_create_function_called_by_py_runtime(self):
    # [FUNCTION DOC]
    # Skipped unless TF2 is enabled. Creates the constant-1 function definition via a runtime
    # on the Python eager context, then executes it by name through the Python eager execute
    # API and asserts the result equals 1.
    if not tf2.enabled():
      self.skipTest("TF2 test")

    fndef = text_format.Parse(
        """
            signature {
               name: 'NullaryFunction'
               output_arg { name: 'o' type: DT_INT32 }
             }
             node_def {
               name: 'retval'
               op: 'Const'
               attr {
                 key: 'dtype'
                 value { type: DT_INT32 }
               }
               attr {
                 key: 'value'
                 value {
                   tensor {
                     dtype: DT_INT32
                     tensor_shape {}
                     int_val: 1
                   }
                 }
               }
             }
             ret { key: 'o' value: 'retval:output' }
         """,
        function_pb2.FunctionDef(),
    )

    ctx = runtime_client.GlobalPythonEagerContext()
    rt = runtime_client.Runtime(ctx)
    rt.CreateFunction(fndef)

    ret, = execute.execute("NullaryFunction", 1, [], (), context.context())
    self.assertAllEqual(ret, 1)

  def test_get_function_proto_from_py_runtime_function(self):
    # [FUNCTION DOC]
    # Skipped unless TF2 is enabled. Traces a constant function, fetches its definition by
    # name through the runtime and asserts the returned signature name matches the traced one.
    if not tf2.enabled():
      self.skipTest("TF2 test")

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced sample function returning the constant 1.
      return 1

    cf = f.get_concrete_function()

    ctx = runtime_client.GlobalPythonEagerContext()
    rt = runtime_client.Runtime(ctx)
    fndef = rt.GetFunctionProto(cf.function_def.signature.name)

    self.assertEqual(fndef.signature.name, cf.function_def.signature.name)

  def test_concrete_function_editing_proto_executed_directly(self):
    # [FUNCTION DOC]
    # Skipped unless TF2 is enabled. Fetches the definition of a traced constant-1 function,
    # edits its constant to 2, re-creates it in the runtime, executes it by name through the
    # eager execute API and asserts the result is 2.
    if not tf2.enabled():
      self.skipTest("TF2 test")

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced sample function returning the constant 1.
      return 1

    cf = f.get_concrete_function()

    ctx = runtime_client.GlobalPythonEagerContext()
    rt = runtime_client.Runtime(ctx)
    fndef = rt.GetFunctionProto(cf.function_def.signature.name)

    fndef.node_def[0].attr["value"].tensor.int_val[0] = 2

    rt.CreateFunction(fndef)

    ret, = execute.execute(fndef.signature.name, 1, [], (), context.context())
    self.assertAllEqual(ret, 2)

  def test_concrete_function_editing_proto(self):
    # [FUNCTION DOC]
    # Skipped unless TF2 is enabled. Traces a constant-1 function, edits its definition to
    # return 2 and re-creates it in the runtime, then calls the original Python function and
    # asserts it now evaluates to 2.
    if not tf2.enabled():
      self.skipTest("TF2 test")

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced sample function returning the constant 1.
      return 1

    cf = f.get_concrete_function()

    ctx = runtime_client.GlobalPythonEagerContext()
    rt = runtime_client.Runtime(ctx)
    fndef = rt.GetFunctionProto(cf.function_def.signature.name)

    fndef.node_def[0].attr["value"].tensor.int_val[0] = 2

    rt.CreateFunction(fndef)

    self.assertAllEqual(self.evaluate(f()), 2)

  def test_concrete_function_editing_proto_after_instantiation(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests editing after the function was already instantiated
    # (called), but the body is identical to the previous test: it never calls the function
    # before editing it.
    # Skipped unless TF2 is enabled. Traces a constant-1 function, edits its definition to
    # return 2, re-creates it in the runtime and asserts calling the function yields 2.
    if not tf2.enabled():
      self.skipTest("TF2 test")

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced sample function returning the constant 1.
      return 1

    cf = f.get_concrete_function()

    ctx = runtime_client.GlobalPythonEagerContext()
    rt = runtime_client.Runtime(ctx)
    fndef = rt.GetFunctionProto(cf.function_def.signature.name)

    fndef.node_def[0].attr["value"].tensor.int_val[0] = 2

    rt.CreateFunction(fndef)

    self.assertAllEqual(self.evaluate(f()), 2)

  def test_concrete_function_editing_via_mlir_pass_tfg_dialect(self):
    # [FUNCTION DOC]
    # Skipped unless TF2 is enabled. Traces an addition function on integer one, applies the
    # test pass registered under the name for the graph dialect (which turns addition into
    # multiplication) through the runtime, and asserts calling the function now gives 1
    # instead of 2.
    if not tf2.enabled():
      self.skipTest("TF2 test")

    @def_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Traced sample function adding its two inputs, with the add node named x_plus_y.
      return math_ops.add(x, y, name="x_plus_y")

    one = constant_op.constant(1)
    cf = f.get_concrete_function(one, one)

    ctx = runtime_client.GlobalPythonEagerContext()
    rt = runtime_client.Runtime(ctx)
    rt.TransformFunction(cf.function_def.signature.name, "test-pass")

    # 1 + 1 = 2. But the pass changes it to 1 * 1.
    self.assertAllEqual(self.evaluate(f(one, one)), 1)

  def test_concrete_function_editing_via_mlir_pass_tf_dialect(self):
    # [FUNCTION DOC]
    # Skipped unless TF2 is enabled. Traces a multiplication function on integer one, applies
    # the test pass for the TF dialect (which turns multiplication into addition) and asserts
    # calling the function now gives 2 instead of 1.
    if not tf2.enabled():
      self.skipTest("TF2 test")

    @def_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Traced sample function multiplying its inputs, with the node named x_times_y.
      return math_ops.multiply(x, y, name="x_times_y")

    one = constant_op.constant(1)
    cf = f.get_concrete_function(one, one)
    fname = cf.function_def.signature.name
    ctx = runtime_client.GlobalPythonEagerContext()
    rt = runtime_client.Runtime(ctx)

    # 1 * 1 = 1 -> 1 + 1 = 2
    rt.TransformFunction(
        fname, "test-pass-tf-dialect", runtime_client.Runtime.Dialect.TF
    )

    self.assertAllEqual(f(one, one), 2)

  def test_concrete_function_editing_via_mlir_pass_mixed_dialects(self):
    # [FUNCTION DOC]
    # Skipped unless TF2 is enabled. Traces an addition function, applies the graph-dialect
    # test pass (add becomes multiply, result 1), then the TF-dialect test pass (multiply
    # becomes add, result 2), asserting the call result after each transformation.
    if not tf2.enabled():
      self.skipTest("TF2 test")

    @def_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Traced sample function adding its two inputs, with the add node named x_plus_y.
      return math_ops.add(x, y, name="x_plus_y")

    one = constant_op.constant(1)
    cf = f.get_concrete_function(one, one)
    ctx = runtime_client.GlobalPythonEagerContext()
    rt = runtime_client.Runtime(ctx)
    fname = cf.function_def.signature.name

    # 1 + 1 = 2 -> 1 * 1 = 1
    rt.TransformFunction(fname, "test-pass")

    self.assertAllEqual(f(one, one), 1)

    # 1 * 1 = 1 -> 1 + 1 = 2
    rt.TransformFunction(
        fname, "test-pass-tf-dialect", runtime_client.Runtime.Dialect.TF
    )

    self.assertAllEqual(f(one, one), 2)


class RuntimeClientMultiWorkersTest(test.TestCase):
  # [CLASS DOC]
  # Tests the runtime client against a local two-worker cluster. Fixtures: remote connection to
  # both workers and two device name attributes (device0, device1) for worker tasks 0 and 1.

  @test_util.run_v2_only
  def setUp(self):
    # [FUNCTION DOC]
    # TF2-only. Calls the parent setUp, starts a local cluster with two workers, connects the
    # eager context to both and records the CPU device names of the two worker tasks.
    super().setUp()

    workers, _ = test_util.create_local_cluster(2, 0)

    remote.connect_to_remote_host(
        [workers[0].target, workers[1].target])

    self.device0 = "/job:worker/replica:0/task:0/device:CPU:0"
    self.device1 = "/job:worker/replica:0/task:1/device:CPU:0"

  @test_util.run_v2_only
  def tearDown(self):
    # [FUNCTION DOC]
    # TF2-only. Calls the parent tearDown, then enters a null device scope (never exits it) and
    # resets the eager context so later tests are not polluted by the cluster setup.
    super().tearDown()
    # Clear the current device scope to avoid polluting other test cases.
    ops.device(None).__enter__()
    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  @test_util.run_v2_only
  def test_transform_function_in_remote_contexts(self):
    # [FUNCTION DOC]
    # TF2-only. Runs an addition function on each worker device (result 3.0 and device checked),
    # then transforms it with the test pass (addition becomes multiplication), re-creates the
    # fetched definition and reruns on both workers, asserting the result is 2.0 and the
    # result device is still the requested worker.
    """Tests if function_defs in remote contexts could be transformed."""

    @def_function.function
    def add(x, y):
      # [FUNCTION DOC]
      # Traced sample function adding its two inputs, with the add node named x_plus_y.
      return math_ops.add(x, y, name="x_plus_y")

    inputs = [1.0, 2.0]

    with ops.device(self.device0):
      result = add(*inputs)

    self.assertAllEqual(result, 3.0)
    self.assertEqual(result.device, self.device0)

    with ops.device(self.device1):
      result = add(*inputs)

    self.assertAllEqual(result, 3.0)
    self.assertEqual(result.device, self.device1)

    cf = add.get_concrete_function(*inputs)
    function_name = cf.function_def.signature.name

    ctx = runtime_client.GlobalPythonEagerContext()
    rt = runtime_client.Runtime(ctx)
    # "test-pass" converts add to multiply.
    rt.TransformFunction(function_name, "test-pass")
    fndef = rt.GetFunctionProto(function_name)
    rt.CreateFunction(fndef)

    with ops.device(self.device0):
      result = add(*inputs)

    self.assertAllEqual(result, 2.0)
    self.assertEqual(result.device, self.device0)

    with ops.device(self.device1):
      result = add(*inputs)

    self.assertAllEqual(result, 2.0)
    self.assertEqual(result.device, self.device1)


if __name__ == "__main__":
  context.set_soft_device_placement(False)
  test_pass.RegisterTestPass()
  test.main()
