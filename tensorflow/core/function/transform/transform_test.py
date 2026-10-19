# [MODULE DOC]
# Unit tests for the high level function transformation API (transform.transform_function).
# Covers Python callback transforms, MLIR pass transforms (using a registered test pass), lists of
# each, keyword inputs, function spec preservation, custom gradients, nested library functions,
# transforms applied inside a traced function, and saving/loading a transformed module.
# Defines shared helpers: an add-to-multiply rewriter, a small module and a transform wrapper.
# When run as main, registers the test MLIR pass before starting the test runner.
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
"""Tests for high-level function transformation API."""

import collections

from absl.testing import parameterized

from tensorflow.core.function.testing import test_pass
from tensorflow.core.function.transform import transform
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.module import module as module_lib
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_lib
from tensorflow.python.saved_model import save as save_lib


def add_to_multiply(fndef):
  # [FUNCTION DOC]
  # HELPER: shared test transformation callback used by most tests in this file.
  # Mutates the given function definition in place: every node whose name ends with the marker
  # name "x_plus_y" is renamed to "x_times_y" and its operation type becomes multiplication, and
  # every node input ending with the marker output reference is renamed accordingly. Returns nothing.
  for node in fndef.node_def:
    if node.name.endswith("x_plus_y"):
      node.name = node.name.replace("x_plus_y", "x_times_y")
      node.op = "Mul"
    for idx, inp in enumerate(node.input):
      if inp.endswith("x_plus_y:z:0"):
        node.input[idx] = inp.replace("x_plus_y", "x_times_y")


class Model(module_lib.Module):
  # [CLASS DOC]
  # HELPER: small module fixture for the save/load test. Holds one tf.function method that adds
  # its two inputs (named "x_plus_y") and optionally adds two more. No instance state.

  @def_function.function
  def f(self, x, y, add_2):
    # [FUNCTION DOC]
    # HELPER: traced method of the fixture module. Inputs x and y are tensors; add_2 is a boolean
    # that is evaluated by Python/autograph. Returns x plus y, plus 2 when add_2 is true.
    r = math_ops.add(x, y, name="x_plus_y")
    if add_2:
      return r + 2
    else:
      return r


def apply_transform(f, transform_fn):
  # [FUNCTION DOC]
  # HELPER: test utility that wraps a tf.function so that each time it is traced, it is transformed.
  # Inputs: f is a tf.function; transform_fn is a callback editing function definitions.
  # Returns a new tf.function which, when traced with some arguments, transforms f for those
  # arguments and calls the result.
  """Wrapper to apply a transformation on every traced tf.function."""

  @def_function.function
  def wrapped(*args):
    # [FUNCTION DOC]
    # HELPER: traced wrapper. Takes any positional arguments, transforms f using them as the inputs,
    # calls the transformed function with the same arguments and returns its result.
    updated_cf = transform.transform_function(
        f, inputs=args, transform_fn=transform_fn)
    return updated_cf(*args)

  return wrapped


class TransformTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (with parameterized support) for transform.transform_function. Each test builds small
  # tf.functions that add tensors, transforms the addition into multiplication with callbacks and/or
  # the test MLIR pass, and checks results, gradients and structure. No shared fixtures except the
  # module-level helpers. Most tests run only in v2 behavior.

  @parameterized.named_parameters(
      dict(
          testcase_name="transform",
          transform_fn=add_to_multiply,
          mlir_pipeline=None),
      dict(
          testcase_name="mlir_pipeline",
          transform_fn=None,
          mlir_pipeline="test-pass"),
      dict(
          testcase_name="transform_and_mlir_pipeline",
          transform_fn=add_to_multiply,
          mlir_pipeline="test-pass"),
      dict(
          testcase_name="transform_list",
          transform_fn=[add_to_multiply],
          mlir_pipeline=None),
      dict(
          testcase_name="mlir_pipeline_list",
          transform_fn=None,
          mlir_pipeline=["test-pass"]),
      dict(
          testcase_name="transform_list_and_mlir_pipeline_list",
          transform_fn=[add_to_multiply],
          mlir_pipeline=["test-pass"]),
  )
  @test_util.run_v2_only
  def test_concrete_function_with(self, transform_fn, mlir_pipeline):
    # [FUNCTION DOC]
    # Parameterized over callback only, pass only, both, and list forms of each. Scenario: a
    # tf.function with a float scalar input signature adds two inputs. It is transformed so the add
    # becomes a multiply. Asserts that calling with 1 and 1 gives 1, that the result can be called
    # from another traced function (1 times 1 plus 10 equals 11), and that gradients through it
    # follow multiplication (inputs 2 and 3 give gradient sum 3 plus 2 equals 5).

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec((), dtype=dtypes.float32),
        tensor_spec.TensorSpec((), dtype=dtypes.float32)
    ])
    def f(x, y):
      # [FUNCTION DOC]
      # Traced function under test: returns the sum of x and y, named with the marker name.
      return math_ops.add(x, y, name="x_plus_y")

    # transfrom f(x, y): x + y -> f(x, y): x * y
    f = transform.transform_function(
        f, transform_fn=transform_fn, mlir_pipeline=mlir_pipeline)
    one = constant_op.constant(1.0)
    self.assertEqual(f(one, one), 1.0)

    @def_function.function
    def f2(x, y):
      # [FUNCTION DOC]
      # Traced caller: calls the transformed function and adds 10 to its result.
      z = f(x, y)
      return math_ops.add(z, 10.0)

    self.assertEqual(f2(one, one), 11.0)

    @def_function.function
    def f_g(x, y):
      # [FUNCTION DOC]
      # Traced function computing the gradients of the transformed function's output with respect
      # to both inputs and returning their sum.
      z = f(x, y)
      dz_dx, dz_dy = gradients_impl.gradients(z, [x, y])
      return math_ops.add(dz_dx, dz_dy)

    self.assertEqual(
        f_g(constant_op.constant(2.0), constant_op.constant(3.0)), 5.0)

  @test_util.run_v2_only
  def test_transform_with_keywords(self):
    # [FUNCTION DOC]
    # Scenario: a tf.function of two arguments is transformed (callback plus test pass) with one
    # positional input and one keyword input. Asserts the transformed function returns 1 when called
    # with positional, mixed positional/keyword, and all-keyword arguments.

    @def_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Traced function under test: returns the marker-named sum of x and y.
      return math_ops.add(x, y, name="x_plus_y")

    one = constant_op.constant(1.0)

    # transfrom f(x, y): x + y -> f(x, y): x * y
    g = transform.transform_function(
        f,
        inputs=[one],
        kw_inputs={"y": one},
        transform_fn=add_to_multiply,
        mlir_pipeline="test-pass")

    self.assertEqual(g(one, one), 1.0)
    self.assertEqual(g(one, y=one), 1.0)
    self.assertEqual(g(x=one, y=one), 1.0)

  @test_util.run_v2_only
  def test_transform_with_keywords_only(self):
    # [FUNCTION DOC]
    # Scenario: like the keyword test but all inputs are passed as keyword inputs and no positional
    # inputs. Asserts the transformed function returns 1 for positional, mixed, and keyword calls.

    @def_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Traced function under test: returns the marker-named sum of x and y.
      return math_ops.add(x, y, name="x_plus_y")

    one = constant_op.constant(1.0)

    # transfrom f(x, y): x + y -> f(x, y): x * y
    g = transform.transform_function(
        f,
        inputs=None,
        kw_inputs={"x": one, "y": one},
        transform_fn=add_to_multiply,
        mlir_pipeline="test-pass")

    self.assertEqual(g(one, one), 1.0)
    self.assertEqual(g(one, y=one), 1.0)
    self.assertEqual(g(x=one, y=one), 1.0)

  @test_util.run_v2_only
  def test_function_spec(self):
    # [FUNCTION DOC]
    # Scenario: transform a tf.function called with plain Python integers. Asserts the original
    # returns 2, the transformed returns 1, and that the pretty printed signatures (ignoring the
    # first line, which holds the name) of original and transformed functions are the same.

    @def_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Traced function under test: returns the marker-named sum of x and y.
      return math_ops.add(x, y, name="x_plus_y")

    args = [1, 1]
    self.assertEqual(f(*args), 2)

    updated_f = transform.transform_function(
        f, inputs=args, transform_fn=add_to_multiply)
    self.assertEqual(updated_f(*args), 1)

    self.assertSequenceAlmostEqual(
        f.get_concrete_function(
            *args).pretty_printed_signature().split("\n")[1:],
        updated_f.pretty_printed_signature().split("\n")[1:])

  @test_util.run_v2_only
  def test_transform_with_custom_gradients(self):
    # [FUNCTION DOC]
    # Scenario: a function uses a custom-gradient add whose gradient is that of a product. After
    # transforming add into multiply, asserts the value is 1 for inputs 1 and 1, and that the
    # gradient sum for inputs 2 and 3 is 5, showing the custom gradient survived replication.

    @custom_gradient.custom_gradient
    def add(x, y):
      # [FUNCTION DOC]
      # Custom-gradient function: returns the marker-named sum and a gradient callable.
      e = math_ops.add(x, y, name="x_plus_y")

      # custom gradient that returns gradients of x * y instead of x + y
      def grad(upstream):
        # [FUNCTION DOC]
        # Gradient callable: takes the upstream gradient and returns gradients as if for a product
        # (upstream times y for x, upstream times x for y).
        dz_dx = y
        dz_dy = x
        return upstream * dz_dx, upstream * dz_dy

      return e, grad

    @def_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Traced function under test: calls the custom-gradient add.
      return add(x, y)

    one = constant_op.constant(1.0)
    f = transform.transform_function(
        f, inputs=[one, one], transform_fn=add_to_multiply)
    self.assertEqual(f(one, one), 1.0)

    @def_function.function
    def f_g(x, y):
      # [FUNCTION DOC]
      # Traced function returning the sum of gradients of the transformed function's output with
      # respect to both inputs.
      z = f(x, y)
      dz_dx, dz_dy = gradients_impl.gradients(z, [x, y])
      return math_ops.add(dz_dx, dz_dy)

    self.assertEqual(
        f_g(constant_op.constant(2.0), constant_op.constant(3.0)), 5.0)

  @test_util.run_v2_only
  def test_transform_with_nested_function(self):
    # [FUNCTION DOC]
    # Scenario: a function contains a nested function with a while loop that repeatedly adds a
    # captured value starting from 1 until the value reaches 3, then adds x. Transforming only the
    # outer function gives 3 for inputs 1 and 2. Then every library function is also targeted with
    # the same callback; asserts the result becomes 4 since the nested additions are multiplied too.

    @def_function.function
    def f(x, z):
      # [FUNCTION DOC]
      # Traced outer function: calls the nested function and returns x plus its result (marker named).

      @def_function.function
      def add():
        # [FUNCTION DOC]
        # Nested traced function: runs a while loop that adds the captured z (marker named) to a
        # counter starting at 1.0 while the counter is below 3.0, and returns the final counter.
        i = constant_op.constant(1.0)
        c = lambda i: math_ops.less(i, 3.0)
        b = lambda i: (math_ops.add(i, z, name="x_plus_y"))
        i = while_loop.while_loop_v2(c, b, [i])
        return i

      y = add()
      return math_ops.add(x, y, name="x_plus_y")

    one = constant_op.constant(1.0)
    two = constant_op.constant(2.0)
    inputs = [one, two]

    # By default only `f` is transformed.
    updated_f = transform.transform_function(
        f, inputs=inputs, transform_fn=add_to_multiply)
    self.assertEqual(updated_f(*inputs), 3.0)  # 1 x (1 + 2) = 3

    # Extract all the functions in `f`'s library that we want to transform.
    nested_transforms = {}
    gdef = f.get_concrete_function(*inputs).graph.as_graph_def()
    for fdef in gdef.library.function:
      nested_transforms[fdef.signature.name] = add_to_multiply

    # Transform `f` and all of its library functions.
    updated_f = transform.transform_function(
        f,
        inputs=inputs,
        transform_fn=add_to_multiply,
        nested_fn_transforms=nested_transforms)
    self.assertEqual(updated_f(*inputs), 4.0)  # 1 x (1 x 2 x 2) = 4

  @parameterized.named_parameters(
      dict(
          testcase_name="fn",
          transform_fn=add_to_multiply),
      dict(
          testcase_name="mlir",
          mlir_pipeline="test-pass"),
      dict(
          testcase_name="fn_and_mlir",
          transform_fn=add_to_multiply,
          mlir_pipeline="test-pass"),
  )
  @test_util.run_v2_only
  def test_nested_python_function_transform_with(self,
                                                 transform_fn=None,
                                                 mlir_pipeline=None):
    # [FUNCTION DOC]
    # Parameterized over callback only, pass only, and both (both default to none). Scenario: a
    # tf.function defines and calls a plain Python inner function that adds its captured inputs.
    # Asserts the original gives 3 for inputs 1 and 2, and the transformed function gives 2,
    # showing the inlined inner addition was transformed to multiplication.

    @def_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Traced function under test: calls a Python inner function and returns its result.

      def inner_add():
        # [FUNCTION DOC]
        # Plain inner function: returns the marker-named sum of the captured x and y.
        return math_ops.add(x, y, name="x_plus_y")

      return inner_add()

    inputs = [1.0, 2.0]
    self.assertEqual(f(*inputs), 3.0)  # 1 + 2 = 3

    # Transform `f`.
    updated_f = transform.transform_function(
        f,
        inputs=inputs,
        transform_fn=transform_fn,
        mlir_pipeline=mlir_pipeline)
    # Nested Python functions should be transformed.
    self.assertEqual(updated_f(*inputs), 2.0)  # 1 x 2 = 2

  @parameterized.named_parameters(
      dict(
          testcase_name="fn_and_nested_fn",
          transform_fn=add_to_multiply,
          nested_fn=True),
      dict(
          testcase_name="fn_and_nested_mlir",
          transform_fn=add_to_multiply,
          nested_mlir=True),
      dict(
          testcase_name="mlir_and_nested_fn",
          mlir_pipeline="test-pass",
          nested_fn=True),
      dict(
          testcase_name="mlir_and_nested_mlir",
          mlir_pipeline="test-pass",
          nested_mlir=True),
      dict(
          testcase_name="mlir_fn_and_nested_mlir_fn",
          transform_fn=add_to_multiply,
          mlir_pipeline="test-pass",
          nested_fn=True,
          nested_mlir=True),
  )
  @test_util.run_v2_only
  def test_nested_transform_with(self,
                                 transform_fn=None,
                                 mlir_pipeline=None,
                                 nested_fn=False,
                                 nested_mlir=False):
    # [FUNCTION DOC]
    # Parameterized over combinations of top-level callback/pass and nested callback/pass
    # transforms (flags default to off; callback and pass default to none). Scenario: an outer
    # tf.function adds x to the result of a nested tf.function adding y and z, with inputs 1, 2, 4
    # chosen so each combination gives a distinct value. The original must give 7. The library
    # functions of the outer function are collected, mapped to the callback and/or pass according to
    # the flags, and the outer function is transformed with them. Asserts the result is 8, meaning
    # both additions became multiplications.

    @def_function.function
    def f(x, y, z):
      # [FUNCTION DOC]
      # Traced outer function: returns x plus the nested function's result (marker named).

      @def_function.function
      def inner_add():
        # [FUNCTION DOC]
        # Nested traced function: returns the marker-named sum of the captured y and z.
        return math_ops.add(y, z, name="x_plus_y")

      return math_ops.add(x, inner_add(), name="x_plus_y")

    # 1, 2, 4 are picked so the following combinations create different results.
    # 1 + (2 + 4) = 7
    # 1 + (2 * 4) = 9
    # 1 * (2 + 4) = 6
    # 1 * (2 * 4) = 8
    inputs = [1.0, 2.0, 4.0]
    self.assertEqual(f(*inputs), 7.0)  # 1 + (2 + 4) = 7

    # Extract all the functions in `f`'s library that we want to transform.
    nested_fn_transforms = {}
    nested_mlir_transforms = {}

    cf = f.get_concrete_function(*inputs)
    gdef = cf.graph.as_graph_def()
    for fdef in gdef.library.function:
      fdef_name = fdef.signature.name
      if nested_fn:
        nested_fn_transforms[fdef_name] = add_to_multiply
      if nested_mlir:
        nested_mlir_transforms[fdef_name] = "test-pass"

    # Transform `f` and all of its library functions.
    updated_f = transform.transform_function(
        f,
        inputs=inputs,
        transform_fn=transform_fn,
        mlir_pipeline=mlir_pipeline,
        nested_fn_transforms=nested_fn_transforms,
        nested_mlir_transforms=nested_mlir_transforms)
    self.assertEqual(updated_f(*inputs), 8.0)  # 1 x (2 x 4) = 8

  @parameterized.named_parameters(
      dict(
          testcase_name="fn_and_nested_fn",
          transform_fn=add_to_multiply,
          nested_fn=True),
      dict(
          testcase_name="fn_and_nested_mlir",
          transform_fn=add_to_multiply,
          nested_mlir=True),
      dict(
          testcase_name="mlir_and_nested_fn",
          mlir_pipeline="test-pass",
          nested_fn=True),
      dict(
          testcase_name="mlir_and_nested_mlir",
          mlir_pipeline="test-pass",
          nested_mlir=True),
      dict(
          testcase_name="mlir_fn_and_nested_mlir_fn",
          transform_fn=add_to_multiply,
          mlir_pipeline="test-pass",
          nested_fn=True,
          nested_mlir=True),
  )
  @test_util.run_v2_only
  def test_nested_transform_in_tf_function_with(self,
                                                transform_fn=None,
                                                mlir_pipeline=None,
                                                nested_fn=False,
                                                nested_mlir=False):
    # [FUNCTION DOC]
    # Parameterized like the nested transform test. Scenario: the transformation itself is performed
    # while tracing an outer function g, which defines f (with a nested inner function), collects f's
    # library functions, transforms f using callbacks and/or passes per the flags, and calls the
    # result. After tracing g, counts operation types across the library functions of its graph and
    # asserts no addition operations remain and exactly two multiplication operations exist.

    @def_function.function
    def g(x, y, z):
      # [FUNCTION DOC]
      # Traced outer function: defines f, transforms it at trace time using the test inputs
      # defined later in the test, and returns the transformed function applied to x, y, z.

      @def_function.function
      def f(x, y, z):
        # [FUNCTION DOC]
        # Function to be transformed: returns x plus the inner function's result (marker named).

        @def_function.function
        def inner_add():
          # [FUNCTION DOC]
          # Nested traced function: returns the marker-named sum of captured y and z.
          return math_ops.add(y, z, name="x_plus_y")

        return math_ops.add(x, inner_add(), name="x_plus_y")

      nested_fn_transforms = {}
      nested_mlir_transforms = {}

      cf = f.get_concrete_function(*inputs)
      gdef = cf.graph.as_graph_def()
      for fdef in gdef.library.function:
        fdef_name = fdef.signature.name
        if nested_fn:
          nested_fn_transforms[fdef_name] = add_to_multiply
        if nested_mlir:
          nested_mlir_transforms[fdef_name] = "test-pass"

      updated_f = transform.transform_function(
          f,
          inputs=inputs,
          transform_fn=transform_fn,
          mlir_pipeline=mlir_pipeline,
          nested_fn_transforms=nested_fn_transforms,
          nested_mlir_transforms=nested_mlir_transforms)

      return updated_f(x, y, z)

    inputs = [1.0, 2.0, 4.0]
    graph_def = g.get_concrete_function(*inputs).graph.as_graph_def()
    # Confirm all "AddV2" nodes in the library functions of graph_def are
    # transformed to "Mul".
    ops = collections.Counter()
    for fdef in graph_def.library.function:
      for node in fdef.node_def:
        ops[node.op] += 1

    self.assertNotIn("AddV2", ops)
    self.assertEqual(ops["Mul"], 2)

  @test_util.run_v2_only
  def test_save_transform_for_all_signatures(self):
    # [FUNCTION DOC]
    # Scenario: a module method adds two ints and optionally 2. Asserts original results 3 and 5.
    # The method is replaced by a wrapper that transforms it for every traced signature. Asserts
    # multiplication results (2, 4, and 2.0 for floats). Saves the module under a fixed temporary
    # directory, loads it back, and asserts the loaded method gives 2 for ints and about 4.2 for
    # float inputs 1.1, 2.0 with the flag true. Side effect: writes a saved model to disk.
    m = Model()
    # Originally f does addition
    self.assertEqual(
        m.f(
            constant_op.constant(1, dtypes.int32),
            constant_op.constant(2, dtypes.int32),
            constant_op.constant(False, dtypes.bool)), 3)

    self.assertEqual(
        m.f(
            constant_op.constant(1, dtypes.int32),
            constant_op.constant(2, dtypes.int32),
            constant_op.constant(True, dtypes.bool)), 5)

    # Transform every input signature of f to a multiply
    m.f = apply_transform(m.f, add_to_multiply)

    # Validate arbitrary signatures.
    self.assertEqual(
        m.f(
            constant_op.constant(1, dtypes.int32),
            constant_op.constant(2, dtypes.int32),
            constant_op.constant(False, dtypes.bool)), 2)
    self.assertEqual(
        m.f(
            constant_op.constant(1, dtypes.int32),
            constant_op.constant(2, dtypes.int32),
            constant_op.constant(True, dtypes.bool)), 4)
    self.assertEqual(
        m.f(
            constant_op.constant(1.0, dtypes.float32),
            constant_op.constant(2.0, dtypes.float32),
            constant_op.constant(False, dtypes.bool)), (2.0))

    # Save and restore the model.
    save_lib.save(m, "/tmp/testing_model")
    m_loaded = load_lib.load("/tmp/testing_model")

    # Validate the restored model.
    self.assertEqual(
        m_loaded.f(
            constant_op.constant(1, dtypes.int32),
            constant_op.constant(2, dtypes.int32),
            constant_op.constant(False, dtypes.bool)), 2)
    self.assertEqual(
        m_loaded.f(
            constant_op.constant(1.1, dtypes.float32),
            constant_op.constant(2.0, dtypes.float32),
            constant_op.constant(True, dtypes.bool)), (4.2))


if __name__ == "__main__":
  test_pass.RegisterTestPass()
  test.main()
