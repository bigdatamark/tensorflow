# [MODULE DOC]
# Unit tests for the polymorphic function (tf.function) implementation in
# polymorphic_function.py. The module defines a few small shared helpers (a cache lister, example
# indexed-slices builders, a type-spec extractor, a decorator imitation and a decorated-method
# holder), a benchmark class, two dataclass-based composite value types, and the very large
# FunctionTest case plus several smaller test classes further down.
# The tests cover tracing and retracing, input signatures, shape relaxation (reduce_retracing),
# variable capture and creation, gradients, autograph conversion, control flow, device placement,
# saving, cancellation, error messages and concrete-function behaviour.
# It is a test module only: importing it registers no flags or globals beyond what the imported
# TensorFlow modules do, and running it as a main program starts the TensorFlow test runner.
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

import collections
import dataclasses
import functools
import itertools
import multiprocessing.pool
import pickle
import platform
import re
import sys
import time
import weakref

from absl.testing import parameterized
import numpy

from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.lang import directives
from tensorflow.python.checkpoint.checkpoint import Checkpoint
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import function as tf_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_sendrecv_ops
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator


def total_function_cache(defined):
  # [FUNCTION DOC]
  # HELPER: small shared test utility used by tests that count how many concrete functions were traced.
  # Returns the list of every concrete function currently held in the cache of the given polymorphic
  # function object ("defined"), by calling its private listing method. No side effects.
  return defined._list_all_concrete_functions()  # pylint: disable=protected-access


def _example_indexed_slices_with_dense_shape():
  # [FUNCTION DOC]
  # HELPER: builds sample input for the tests that pass sparse gradient-like values to functions.
  # Takes no arguments and returns an indexed-slices object with values 1 and 2, indices 0 and 1, and
  # an explicit dense shape of two. Creates constant tensors in the current context.
  return indexed_slices.IndexedSlices(
      constant_op.constant([1, 2]), constant_op.constant([0, 1]),
      constant_op.constant([2]))


def _example_indexed_slices_without_dense_shape():
  # [FUNCTION DOC]
  # HELPER: builds sample input for the tests that pass sparse gradient-like values to functions.
  # Takes no arguments and returns an indexed-slices object with values 1 and 2 and indices 0 and 1
  # but no dense shape (the dense shape is left unset). Creates constant tensors in the current context.
  return indexed_slices.IndexedSlices(
      constant_op.constant([1, 2]), constant_op.constant([0, 1]))


def _spec_for_value(value):
  # [FUNCTION DOC]
  # HELPER: converts concrete call arguments into the type specifications used to request concrete
  # functions in tests.
  # Takes an arbitrary value. If it is a nested structure, returns the same structure with every leaf
  # converted recursively; if it is a tensor or composite tensor, returns its type specification;
  # any other value is returned unchanged. No side effects, no exceptions of its own.
  """Returns the (nested) TypeSpec for a value."""
  if nest.is_nested(value):
    return nest.map_structure(_spec_for_value, value)
  elif isinstance(value, (tensor_lib.Tensor, composite_tensor.CompositeTensor)):
    return type_spec.type_spec_from_value(value)
  else:
    return value


# This dummy decorator imitates ordinary decorators utilizing tf_decorator.
def dummy_tf_decorator(method):
  # [FUNCTION DOC]
  # HELPER: test decorator that imitates ordinary user decorators built on the TensorFlow decorator
  # utility, used to check that function wrapping and signature handling work through such decorators.
  # Takes a callable (method) and returns a wrapper made via the make-decorator utility, which
  # forwards all positional and keyword arguments to the original callable and returns its result.

  def wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: the inner forwarding function created by the dummy decorator.
    # Accepts any positional and keyword arguments, calls the wrapped method with them and returns
    # whatever it returns. No other side effects.
    return method(*args, **kwargs)

  return tf_decorator.make_decorator(method, wrapper)


def undecorated_function(x):
  # [FUNCTION DOC]
  # HELPER: plain module-level function used as a decorator-free target in tests.
  # Takes a number and returns it multiplied by three (floating point). No side effects.
  return x * 3.


class _HasDecoratedMethod(object):
  # [CLASS DOC]
  # HELPER: tiny fixture class for tests of tf.function applied to methods.
  # Holds no state; defines a single method wrapped by the polymorphic function decorator, used to
  # check that decorated methods are bound properly to instances.

  @polymorphic_function.function
  def f(self, x):
    # [FUNCTION DOC]
    # Method decorated as a polymorphic function. Takes an instance and a numeric value x and returns x
    # multiplied by three (floating point). No side effects.
    return x * 3.


class FunctionBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class for the tf.function implementation. Inherits the TensorFlow benchmark base class
  # and holds no state; each benchmark method times one operation and reports the result.
  """Benchmark the tf.function implementation."""

  def benchmark_repeat_captures_property_access(self):
    # [FUNCTION DOC]
    # Benchmark: creates one hundred variables, builds a function that sums them, obtains its concrete
    # function graph, then reads the graph's captures property one million times and reports the
    # iteration count and mean wall time per access. Measures the cost of repeated captures property
    # access. Returns nothing; side effect is the benchmark report.
    n_iters = 1000000
    n_captures = 100
    vs = []
    for _ in range(n_captures):
      vs.append(variables.Variable(1.0))

    def f():
      # [FUNCTION DOC]
      # Nested function that adds up all of the captured variables (by index) and returns the sum. Takes no
      # arguments; it is the function under benchmark.
      result = 0
      for idx in range(n_captures):
        result += vs[idx]
      return result

    pf = polymorphic_function.function(f)
    g = pf.get_concrete_function().graph

    start_time = time.time()
    for _ in range(n_iters):
      temp = g.captures  # pylint: disable=unused-variable
    duration = time.time() - start_time

    self.report_benchmark(iters=n_iters, wall_time=duration / float(n_iters))


@dataclasses.dataclass
class MaskedTensor:
  # [CLASS DOC]
  # Test dataclass pairing a boolean mask with a tensor value. It implements the flatten and unflatten
  # protocol so that tf.function can treat it as a composite value: the mask is metadata and the
  # tensor is the single component. Used as an argument type in tracing tests.
  mask: bool
  value: tensor_lib.Tensor

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # Returns a pair: a one-element metadata tuple holding the mask, and a one-element components tuple
    # holding the tensor value. No side effects.
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, leaves):
    # [FUNCTION DOC]
    # Class method that rebuilds an instance from the flattened form. Takes the metadata (mask first)
    # and the list of leaves (tensor value first) and returns a new MaskedTensor. No side effects.
    mask = metadata[0]
    value = leaves[0]
    return MaskedTensor(mask=mask, value=value)


@dataclasses.dataclass
class MaskedTensorPair:
  # [CLASS DOC]
  # Test dataclass holding a list of boolean masks and two MaskedTensor values. Implements the
  # flatten and unflatten protocol: the masks list is the metadata and the two values are the
  # components. Used to test nested composite arguments in tracing.
  masks: list[bool]
  value1: MaskedTensor
  value2: MaskedTensor

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # Returns a pair: metadata tuple containing the masks list, and components tuple containing the two
    # masked values in order. No side effects.
    metadata = (self.masks,)
    components = (self.value1, self.value2)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, leaves):
    # [FUNCTION DOC]
    # Class method rebuilding an instance. Takes the metadata (masks first) and the leaves, which must
    # contain exactly two values; unpacking raises an error otherwise. Returns a new MaskedTensorPair.
    masks = metadata[0]
    value1, value2 = leaves
    return MaskedTensorPair(masks=masks, value1=value1, value2=value2)


# TODO(mdan): Organize these tests.
class FunctionTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for the polymorphic function (tf.function). It combines the TensorFlow test base
  # class with parameterized testing. The fixture splits the first physical CPU into four logical
  # devices. Tests cover tracing, retracing, shape relaxation, variables, implements attributes,
  # gradients, device placement, concurrency and many other behaviours; most are independent.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup. Calls the parent setup, then configures the first physical CPU device as four
    # logical CPU devices so tests can place work on cpu:0 to cpu:3. Side effect: changes global device
    # configuration, which must happen before the runtime is initialized.
    super().setUp()
    cpus = config.list_physical_devices('CPU')
    # Set 4 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])

  def testBasic(self):
    # [FUNCTION DOC]
    # Wraps matmul as a function and applies it twice (transposing the first argument each time) on a
    # two-by-two matrix. Asserts the flattened results equal the expected values for both products.
    matmul = polymorphic_function.function(math_ops.matmul)
    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    sq = matmul(t, t, transpose_a=True)
    sq2 = matmul(sq, t, transpose_a=True)
    self.assertAllEqual(sq.numpy().reshape(-1), [10, 14, 14, 20])
    self.assertAllEqual(sq2.numpy().reshape(-1), [52, 76, 74, 108])

  def testPythonFunctionNotCallable(self):
    # [FUNCTION DOC]
    # Passes a non-callable value (the number one) to the function decorator and asserts that a type
    # error mentioning "is not a callable object" is raised.
    with self.assertRaisesRegex(TypeError, 'is not a callable object'):
      polymorphic_function.function(1)

  def testOnExitCallback(self):
    # [FUNCTION DOC]
    # Registers exit callbacks (one per function) on the default function graph while tracing an outer
    # function that calls an inner function. Asserts the callbacks fire once per traced function when
    # tracing finishes, in inner-then-outer order, and again after a forced retrace with a new input
    # shape. Also asserts the callback has not yet run while tracing is still in progress.
    values = []

    def append_1():
      # [FUNCTION DOC]
      # Callback that appends the value one to the shared values list. Used as the exit callback of the
      # inner function.
      values.append(1)

    def append_2():
      # [FUNCTION DOC]
      # Callback that appends the value two to the shared values list. Used as the exit callback of the
      # outer function.
      values.append(2)

    def g(x):
      # [FUNCTION DOC]
      # Inner traced function: records a copy of the list, registers the first callback as an exit
      # callback on the current function graph, asserts the list has not changed yet, and returns its
      # input plus one.
      old_values = list(values)
      ops.add_exit_callback_to_default_func_graph(append_1)
      self.assertEqual(old_values, values)
      return x + 1

    tf_g = polymorphic_function.function(g)

    def f(x):
      # [FUNCTION DOC]
      # Outer traced function: copies the values list, registers the second callback as an exit callback,
      # asserts the list has not changed yet, and returns the result of calling the wrapped inner function.
      old_values = list(values)
      ops.add_exit_callback_to_default_func_graph(append_2)
      self.assertEqual(old_values, values)
      return tf_g(x)

    tf_f = polymorphic_function.function(f)
    self.assertEmpty(values)
    tf_f(constant_op.constant(1.0))
    self.assertEqual(values, [1, 2])  # Once for g, once for f.
    tf_f(constant_op.constant([1.0]))  # force a retrace
    self.assertEqual(values, [1, 2, 1, 2])  # And again.

  def testCannotAddExitCallbackWhenNotInFunctionScope(self):
    # [FUNCTION DOC]
    # Calls the add-exit-callback utility outside any function graph and asserts it raises a runtime
    # error saying it cannot be used when not building a function.
    with self.assertRaisesRegex(RuntimeError, 'when not building a function.'):
      ops.add_exit_callback_to_default_func_graph(lambda: None)

  def testVariable(self):
    # [FUNCTION DOC]
    # Wraps a function that adds an argument, a variable created outside the function, and a second
    # variable passed as an argument. Calls it and asserts the evaluated result is three.
    v1 = variables.Variable(1.0)
    add = polymorphic_function.function(lambda x, v: x + v1 + v)
    v2 = variables.Variable(1.0)
    x = constant_op.constant(1.0)
    r = add(x, v2)
    self.assertEqual(3.0, self.evaluate(r))

  def testVariableOnly(self):
    # [FUNCTION DOC]
    # Wraps a function that increments its argument in place and returns it. Passes a variable and
    # asserts the value becomes two; then passes a plain tensor and asserts an attribute error
    # (missing assign method) is raised.
    v = variables.Variable(1.0)
    add = polymorphic_function.function(lambda x: x.assign_add(1.0))
    r1 = add(v)
    self.assertEqual(2.0, self.evaluate(r1))
    c = constant_op.constant(1.0)
    with self.assertRaisesRegex(AttributeError, 'no attribute'):
      add(c)

  def testVariableMultiFunction(self):
    # [FUNCTION DOC]
    # Tests passing the same variable twice into nested functions. An inner function sums two variable
    # arguments and a constant, an outer function passes one variable for both. Asserts both functions
    # return five when given a variable of value two and a constant of one.

    @polymorphic_function.function
    def second(dup_var, dup_var_2, some_const):
      # [FUNCTION DOC]
      # Inner function: returns the sum of two variable arguments and a constant argument.
      return dup_var + dup_var_2 + some_const

    @polymorphic_function.function
    def first(dup_var, some_const):
      # [FUNCTION DOC]
      # Outer function: calls the second function passing its first argument twice together with the
      # constant, and returns the result (exercises duplicate variable arguments).
      return second(dup_var, dup_var, some_const)

    my_const = constant_op.constant(1)
    my_var = variables.Variable(2, dtype=dtypes.int32)
    self.assertEqual(second(my_var, my_var, my_const).numpy(), 5)
    self.assertEqual(first(my_var, my_const).numpy(), 5)

  @test_util.disable_tfrt('Packed tensor is not supported in tfrt yet.')
  def testPackedVariable(self):
    # [FUNCTION DOC]
    # Creates resource variables on cpu:0, cpu:1 and cpu:2, packs their handles into two packed eager
    # tensors (each spanning two devices), and defines a function that increments each packed handle and
    # reads the variables on specific devices. Asserts the function's argument attributes carry the
    # composite-device attribute naming each packed handle's device, and that the returned values are
    # the original values plus the increments. Skipped under the TFRT runtime (decorator).
    with ops.device('/cpu:0'):
      v0_0 = resource_variable_ops.ResourceVariable(1.0)
    with ops.device('/cpu:1'):
      v0_1 = resource_variable_ops.ResourceVariable(2.0)
      v1_0 = resource_variable_ops.ResourceVariable(3.0)
    with ops.device('/cpu:2'):
      v1_1 = resource_variable_ops.ResourceVariable(4.0)

    packed_var_0 = ops.pack_eager_tensors([v0_0.handle, v0_1.handle])
    packed_var_1 = ops.pack_eager_tensors([v1_0.handle, v1_1.handle])

    # TODO(b/145922293): use ResourceVariable.assign_add and
    # ResourceVariable.read_value directly once we support packing multiple
    # ResourceVariable into one ResourceVariable.
    @polymorphic_function.function
    def read_var():
      # [FUNCTION DOC]
      # Traced function that adds five to the first packed variable handle and six to the second,
      # then reads the first on cpu:0 and cpu:1 and the second on cpu:1 and cpu:2, returning the four
      # read values as a tuple. Captures the packed handles from the enclosing test.
      resource_variable_ops.assign_add_variable_op(packed_var_0,
                                                   constant_op.constant(5.0))
      resource_variable_ops.assign_add_variable_op(packed_var_1,
                                                   constant_op.constant(6.0))
      with ops.device('/cpu:0'):
        read0 = resource_variable_ops.read_variable_op(
            packed_var_0, dtype=dtypes.float32)
      with ops.device('/cpu:1'):
        read1 = resource_variable_ops.read_variable_op(
            packed_var_0, dtype=dtypes.float32)
        read2 = resource_variable_ops.read_variable_op(
            packed_var_1, dtype=dtypes.float32)
      with ops.device('/cpu:2'):
        read3 = resource_variable_ops.read_variable_op(
            packed_var_1, dtype=dtypes.float32)

      return read0, read1, read2, read3

    arg_attrs = read_var.get_concrete_function().function_def.arg_attr
    self.assertLen(arg_attrs, 2)
    self.assertEqual(arg_attrs[0].attr['_composite_device'].s,
                     compat.as_bytes(packed_var_0.device))
    self.assertEqual(arg_attrs[1].attr['_composite_device'].s,
                     compat.as_bytes(packed_var_1.device))

    self.assertAllEqual(read_var(), (1 + 5, 2 + 5, 3 + 6, 4 + 6))

  def testImplementsAttributeBasic(self):
    # [FUNCTION DOC]
    # In graph mode, wraps an addition function with the experimental implements option set to a name,
    # calls it and takes its gradient, then inspects the graph's function library. Asserts three
    # function definitions exist: the forward and backward ones lack the implements attribute while the
    # main one carries the attribute with the given name.
    v = polymorphic_function.function(
        experimental_implements='func')(lambda x, y: x + y)
    with context.graph_mode(), self.cached_session():
      a = array_ops.placeholder(dtypes.float32, ())
      b = array_ops.placeholder(dtypes.float32, ())
      v(a, b)
      gradients_impl.gradients(v(a, b), [a, b])
      fdefs = ops.get_default_graph().as_graph_def().library.function
      self.assertLen(fdefs, 3)
      not_present = 0
      present = 0
      for f in fdefs:
        name = f.signature.name
        if 'forward' in name or 'backward' in name:
          not_present += 1
          self.assertNotIn(attributes_lib.IMPLEMENTS,
                           f.attr, f)
        else:
          present += 1
          self.assertEqual(
              f.attr[attributes_lib.IMPLEMENTS].s,
              'func'.encode('ascii'), f)
      self.assertEqual(not_present, 2, fdefs)
      self.assertEqual(present, 1, fdefs)

  def testImplementsAttributeAssertsOnSideInput(self):
    # [FUNCTION DOC]
    # In graph mode, wraps with the implements option a function that captures a tensor created
    # outside it. Asserts that calling it raises an assertion error saying variables are always
    # captured, and that no function ends up registered in the graph library.
    with context.graph_mode(), self.cached_session():
      z = array_ops.zeros(0)
      v = polymorphic_function.function(
          experimental_implements='func')(lambda x, y: x + y + z)
      a = array_ops.ones((1,))
      b = array_ops.ones((1,))
      with self.assertRaisesRegex(AssertionError,
                                  'variables are always captured'):
        v(a, b)
      functions = ops.get_default_graph().as_graph_def().library.function
      self.assertEmpty(functions)

  def testImplementsAttributeWorksWithGradientTape(self):
    # [FUNCTION DOC]
    # Wraps a function computing x plus y squared with an implements option and differentiates it with
    # a gradient tape over two variables (three and two). Asserts the gradients are four with respect
    # to y and one with respect to x.
    add = lambda x, y: x + y**2
    add = polymorphic_function.function(experimental_implements='MyFunc')(add)
    x = variables.Variable(3.0)
    y = variables.Variable(2.0)

    with backprop.GradientTape() as tape:
      g = add(x, y)

    dg_dy, dg_dx = tape.gradient(g, [y, x])
    self.assertEqual(dg_dy.numpy(), 4.0)
    self.assertEqual(dg_dx.numpy(), 1.0)

  def testImplementsAttributeWorksOnVariables(self):
    # [FUNCTION DOC]
    # In graph mode, calls an implements-annotated function with two variables and again with the same
    # variable twice. Asserts only one function was created, that evaluating reads current variable
    # values (two, then three after an increment of the first variable).
    with context.graph_mode(), self.cached_session():
      v = polymorphic_function.function(
          experimental_implements='func')(lambda x, y: x + y)
      a = variables.Variable((1.0,))
      b = variables.Variable((1.0,))
      r1 = v(a, b)
      _ = v(a, a)
      functions = ops.get_default_graph().as_graph_def().library.function
      # Verify that we created only one function
      self.assertLen(functions, 1)
      # Verify that self.evaluate() reads the current values.
      a.initializer.run()
      b.initializer.run()
      self.assertEqual(self.evaluate(r1), 2)

      self.evaluate(a.assign_add([1]))
      self.assertEqual(self.evaluate(r1), 3)

  def testImplementsAttributeWorksOnConstants(self):
    # [FUNCTION DOC]
    # In graph mode, calls an implements-annotated addition with a variable and a Python constant in
    # both argument orders. Asserts a single function is created with two input arguments (constants
    # are not inlined) and that both results evaluate to three after variable initialization.
    with context.graph_mode(), self.cached_session():
      v = polymorphic_function.function(
          experimental_implements='func')(lambda x, y: x + y)
      a = variables.Variable(1.0)
      r1 = v(a, 2.)
      r2 = v(2., a)
      functions = ops.get_default_graph().as_graph_def().library.function
      self.assertLen(functions, 1)
      self.assertLen(functions[0].signature.input_arg, 2)
      # Verify that self.evaluate() reads the current values.
      a.initializer.run()
      self.assertEqual(self.evaluate(r1), 3)
      self.assertEqual(self.evaluate(r2), 3)

  def testImplementsAttributeSpecializes(self):
    # [FUNCTION DOC]
    # In graph mode, calls an implements-annotated addition with a variable plus a list, and a list of
    # two elements plus the variable. Asserts two functions are created, each still with two input
    # arguments (nothing inlined), and the evaluated results are the expected arrays.
    with context.graph_mode(), self.cached_session():
      v = polymorphic_function.function(
          experimental_implements='func')(lambda x, y: x + y)
      a = variables.Variable(1.0)
      r1 = v(a, [2.])
      r2 = v([2., 2], a)
      functions = ops.get_default_graph().as_graph_def().library.function
      self.assertLen(functions, 2)
      # Ensure that all parameters are still there and haven't been inlined!

      self.assertLen(functions[0].signature.input_arg, 2)
      self.assertLen(functions[1].signature.input_arg, 2)
      # Verify that self.evaluate() reads the current values.
      a.initializer.run()
      numpy.testing.assert_equal(self.evaluate(r1), [3.])
      numpy.testing.assert_equal(self.evaluate(r2), [3., 3.])

  def testImplementsWorksWithTensorSpec(self):
    # [FUNCTION DOC]
    # Gets a concrete function for an implements-annotated addition using two unknown-shape float
    # specs, calls it with plain floats one and two and asserts the result is three.
    v = polymorphic_function.function(
        experimental_implements='func')(lambda x, y: x + y)
    v = v.get_concrete_function(
        tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32),
        tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32))
    x = v(1., 2.)
    self.assertEqual(x.numpy(), 3.)

  def testImplementsAttributeAsNameAttrList(self):
    # [FUNCTION DOC]
    # Passes a textual name-and-attributes description as the implements option, in graph mode, and
    # takes the gradient. Asserts three functions are produced; forward and backward lack the attribute
    # while the main function has a function-valued attribute named embedding_matmul with two
    # attributes.
    implements_attr = (
        'name: "embedding_matmul" attr {   key: "key1"   value {     i: 2   } '
        '} attr {   key: "key2"   value {     b: false   } }')
    v = polymorphic_function.function(
        experimental_implements=implements_attr)(lambda x, y: x + y)
    with context.graph_mode(), self.cached_session():
      a = array_ops.placeholder(dtypes.float32, ())
      b = array_ops.placeholder(dtypes.float32, ())
      v(a, b)
      gradients_impl.gradients(v(a, b), [a, b])
      fdefs = ops.get_default_graph().as_graph_def().library.function
      self.assertLen(fdefs, 3)
      not_present = 0
      present = 0
      for f in fdefs:
        name = f.signature.name
        if 'forward' in name or 'backward' in name:
          not_present += 1
          self.assertNotIn(attributes_lib.IMPLEMENTS,
                           f.attr, f)
        else:
          present += 1
          attr_value = f.attr[attributes_lib.IMPLEMENTS]
          self.assertIsNotNone(attr_value.func, f)
          self.assertEqual(attr_value.func.name, 'embedding_matmul')
          name_attrs = attr_value.func.attr
          self.assertLen(name_attrs, 2)
      self.assertEqual(not_present, 2, fdefs)
      self.assertEqual(present, 1, fdefs)

  def testDisableACDAttribute(self):
    # [FUNCTION DOC]
    # Wraps the same function that reads and increments a captured variable twice: once normally and
    # once with the experimental attribute that disables automatic control dependencies. Asserts the
    # normal concrete graph has two control outputs and the one with the attribute disabled has none.
    v = resource_variable_ops.ResourceVariable(1.0)

    def foo(x, y):
      # [FUNCTION DOC]
      # Function that reads a captured (nonlocal) variable, adds the sum of its two arguments to the
      # variable, and returns the value read before the update.
      nonlocal v
      t = v.read_value()
      v.assign_add(x + y)
      return t

    with_acd = polymorphic_function.function(foo)
    without_acd = polymorphic_function.function(
        foo, experimental_attributes={'_disable_acd': True}
    )

    with_acd_control_outputs = with_acd.get_concrete_function(
        tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32),
        tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32),
    ).graph.control_outputs
    without_acd_control_outputs = without_acd.get_concrete_function(
        tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32),
        tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32),
    ).graph.control_outputs

    self.assertLen(with_acd_control_outputs, 2)
    self.assertEmpty(without_acd_control_outputs)

  def testReduceTracingWithNestedTFFunction(self):
    # [FUNCTION DOC]
    # Nests two functions that use the reduce-retracing option, each incrementing a variable. Runs the
    # outer function twice under a gradient tape and asserts the outputs (four and five, then seven and
    # eight) and that the gradient with respect to the variable is ones both times.
    v = resource_variable_ops.ResourceVariable([1., 2.])

    @polymorphic_function.function(reduce_retracing=True)
    def inner_test_fn(x):
      # [FUNCTION DOC]
      # Inner function: adds two to each element of the variable argument in place and returns it.
      x.assign_add([2., 2.])
      return x

    @polymorphic_function.function(reduce_retracing=True)
    def test_fn(x):
      # [FUNCTION DOC]
      # Outer function: adds one to each element of the variable argument in place, then returns the
      # result of the inner function applied to it.
      x.assign_add([1., 1.])
      return inner_test_fn(x)

    with backprop.GradientTape() as tape:
      y = test_fn(v)

    grad = tape.gradient(y, v)
    self.assertAllEqual(y, [4., 5.])
    self.assertAllEqual(grad, [1., 1.])

    with backprop.GradientTape() as tape:
      y = test_fn(v)

    grad = tape.gradient(y, v)
    self.assertAllEqual(y, [7., 8.])
    self.assertAllEqual(grad, [1., 1.])

  def testInputShapeRelaxationOnInstanceMethod(self):
    # [FUNCTION DOC]
    # Defines a class whose method uses the reduce-retracing option and records whether it was traced
    # with an unknown first dimension. Calls the bound method with empty, then one-element, then
    # two-element inputs and asserts the unknown dimension is only seen after the second distinct shape,
    # showing the option is preserved when methods are bound to instances.
    # Test that reduce_retracing is passed during
    # instance method bounding.
    unknown_dim = [False]

    class Foo:
      # [CLASS DOC]
      # Local helper class holding one method decorated with the reduce-retracing option, used to test
      # instance-method binding.

      @polymorphic_function.function(reduce_retracing=True)
      def func(self, a):
        # [FUNCTION DOC]
        # Method: sets an outer flag when the first dimension of its tensor argument is unknown at trace
        # time, and returns the argument plus one.
        if a._shape_tuple()[0] is None:
          unknown_dim[0] = True
        return a + 1

    foo = Foo()
    foo.func(constant_op.constant([]))
    self.assertFalse(unknown_dim[0])

    foo.func(constant_op.constant([1.0]))
    self.assertTrue(unknown_dim[0])

    foo.func(constant_op.constant([1.0, 2.0]))
    self.assertTrue(unknown_dim[0])

  def testInputShapeFunctionRelaxationWithRaggedTensors(self):
    # [FUNCTION DOC]
    # Calls a reduce-retracing function with ragged tensors and checks, through the recorded type
    # spec, when a retrace happens: first call traces; same spec does not retrace; different component
    # shapes do not retrace; changing the row count relaxes to unknown rows and retraces, later row
    # count changes do not retrace; changing ragged rank retraces, and changing it again relaxes.
    traced_type_spec = [None]

    @polymorphic_function.function(reduce_retracing=True)
    def func(x):
      # [FUNCTION DOC]
      # Traced function that stores the type spec of its argument in an outer list and returns the
      # argument; the recorded spec is how the test detects a retrace.
      traced_type_spec[0] = x._type_spec
      return x

    def check_trace(x, expected_trace):
      # [FUNCTION DOC]
      # Helper: clears the recorded type spec, calls the function with the given input, and asserts the
      # recorded spec equals the expected one (null/none meaning no retrace occurred).
      traced_type_spec[0] = None
      func(x)
      self.assertEqual(traced_type_spec[0], expected_trace)

    check_trace(  # Initial call gets traced.
        ragged_factory_ops.constant([[1], [2, 3, 4]]),
        ragged_tensor.RaggedTensorSpec([2, None], dtypes.int32))
    check_trace(  # Input TypeSpec is the same -> no retrace.
        ragged_factory_ops.constant([[1, 2], [3, 4]]), None)
    check_trace(  # Even if component tensor shapes change -> no retrace.
        ragged_factory_ops.constant([[1, 2], [3, 4, 5, 6]]), None)
    check_trace(  # Different TypeSpec shape (nrows): relax & retrace
        ragged_factory_ops.constant([[1], [2], [3]]),
        ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32))
    check_trace(  # Different nrows again: relax & retrace
        ragged_factory_ops.constant([[1], [2], [3], [4]]), None)
    check_trace(  # Different nrows yet again: not retrace
        ragged_factory_ops.constant([[1]]), None)
    check_trace(  # Different ragged_rank: retrace
        ragged_factory_ops.constant([[[1]]]),
        ragged_tensor.RaggedTensorSpec([1, None, None], dtypes.int32))
    check_trace(  # Different ragged_rank again: retrace & relax
        ragged_factory_ops.constant([[[1]], [[2]]]),
        ragged_tensor.RaggedTensorSpec([None, None, None], dtypes.int32))

  def testInputShapeFunctionRelaxationWithStructuredTensors(self):
    # [FUNCTION DOC]
    # Same retracing checks as the ragged test but with structured tensors. Different field names each
    # cause a retrace; calling again with the same field name but a different shape relaxes the shape
    # and retraces; later calls with further shapes reuse the relaxed graph (no retrace).
    traced_type_spec = [None]

    @polymorphic_function.function(reduce_retracing=True)
    def func(x):
      # [FUNCTION DOC]
      # Traced function that stores the type spec of its argument in an outer list and returns the
      # argument; used to detect retracing.
      traced_type_spec[0] = x._type_spec
      return x

    def check_trace(x, expected_trace):
      # [FUNCTION DOC]
      # Helper: resets the recorded spec to nothing, calls the function with the input, and asserts the
      # recorded spec equals the expected value (nothing means no retrace).
      traced_type_spec[0] = None
      func(x)
      self.assertEqual(traced_type_spec[0], expected_trace)

    # If we have TypeSpecs that differ in ways other than just their shape,
    # then retrace each time.
    check_trace(
        structured_tensor.StructuredTensor.from_pyval({'a': [1]}),
        structured_tensor.StructuredTensor.Spec._from_fields_and_rank(
            fields={'a': tensor_lib.TensorSpec((1,), dtypes.int32)}, rank=0))
    check_trace(
        structured_tensor.StructuredTensor.from_pyval({'b': [1]}),
        structured_tensor.StructuredTensor.Spec._from_fields_and_rank(
            fields={'b': tensor_lib.TensorSpec((1,), dtypes.int32)}, rank=0))
    check_trace(
        structured_tensor.StructuredTensor.from_pyval({'c': [1]}),
        structured_tensor.StructuredTensor.Spec._from_fields_and_rank(
            fields={'c': tensor_lib.TensorSpec((1,), dtypes.int32)}, rank=0))

    # But if we call again with only shape different, then do relax:
    check_trace(  # relax & retrace
        structured_tensor.StructuredTensor.from_pyval({'a': [1, 2]}),
        structured_tensor.StructuredTensor.Spec._from_fields_and_rank(
            fields={'a': tensor_lib.TensorSpec((None,), dtypes.int32)},
            rank=0))
    check_trace(  # use relaxed graph
        structured_tensor.StructuredTensor.from_pyval({'a': [1, 2, 3]}), None)
    check_trace(  # use relaxed graph
        structured_tensor.StructuredTensor.from_pyval({'a': [1, 2, 3, 4]}),
        None)

  def testInputShapeFunctionRelaxationWithDatasetIterators(self):
    # [FUNCTION DOC]
    # Verifies relaxation with dataset iterators, whose type specs carry information not derivable
    # from components. Iterators over datasets of shape one-by-two trace once and reuse; two-by-two
    # relaxes the first dimension to unknown and retraces; three and four rows reuse it; a
    # two-by-one dataset relaxes both dimensions to unknown and retraces.
    # For dataset iterators, the TypeSpec includes type information that's
    # not derivable from the component tensors.  Make sure that the TypeSpec
    # shapes get relaxed as appropriate.

    traced_type_spec = [None]

    @polymorphic_function.function(reduce_retracing=True)
    def func(x):
      # [FUNCTION DOC]
      # Traced function that stores the type spec of its argument in an outer list and returns the
      # argument; used to detect retracing.
      traced_type_spec[0] = x._type_spec
      return x

    def check_trace(x, expected_trace):
      # [FUNCTION DOC]
      # Helper: resets the recorded spec, calls the function with the input, and asserts the recorded
      # spec equals the expected one (nothing meaning no new trace happened).
      traced_type_spec[0] = None
      func(x)
      self.assertEqual(traced_type_spec[0], expected_trace)

    ds_1_2 = dataset_ops.DatasetV2.from_tensors(array_ops.zeros([1, 2]))
    ds_2_2 = dataset_ops.DatasetV2.from_tensors(array_ops.zeros([2, 2]))
    ds_3_2 = dataset_ops.DatasetV2.from_tensors(array_ops.zeros([3, 2]))
    ds_4_2 = dataset_ops.DatasetV2.from_tensors(array_ops.zeros([4, 2]))
    ds_2_1 = dataset_ops.DatasetV2.from_tensors(array_ops.zeros([2, 1]))
    check_trace(  # shape=[1, 2]: retrace
        dataset_ops.make_one_shot_iterator(ds_1_2),
        iterator_ops.IteratorSpec(
            tensor_lib.TensorSpec([1, 2], dtypes.float32)))
    check_trace(  # shape=[1, 2]: no retrace (use the [1, 2] graph)
        dataset_ops.make_one_shot_iterator(ds_1_2), None)
    check_trace(  # shape=[2, 2]: relax to [None, 2] and retrace
        dataset_ops.make_one_shot_iterator(ds_2_2),
        iterator_ops.IteratorSpec(
            tensor_lib.TensorSpec([None, 2], dtypes.float32)))
    check_trace(  # shape=[3, 2]: no retrace (use the [None, 2] graph)
        dataset_ops.make_one_shot_iterator(ds_3_2), None)
    check_trace(  # shape=[4, 2]: no retrace (use the [None, 2] graph)
        dataset_ops.make_one_shot_iterator(ds_4_2), None)
    check_trace(  # shape=[2, 1]: relax to [None, None] and retrace
        dataset_ops.make_one_shot_iterator(ds_2_1),
        iterator_ops.IteratorSpec(
            tensor_lib.TensorSpec([None, None], dtypes.float32)))

  def testCapturesVariables(self):
    # [FUNCTION DOC]
    # Creates a trainable and a non-trainable variable outside the function and a third variable lazily
    # inside it on first trace. Gets the concrete function and asserts its variables set holds all
    # three, its trainable variables hold only the trainable two, and both match the graph's own
    # variable lists.
    a = variables.Variable(1.0, trainable=False)
    b = variables.Variable(1.0)
    cc = [None]

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function that creates a new variable on first call (storing it in an outer holder) and
      # returns the sum of the three variables plus one.
      c = cc[0]
      if c is None:
        c = cc[0] = variables.Variable(1.)
      return a + b + c + 1

    cf = f.get_concrete_function()
    c = cc[0]

    captured_variables = {v.ref() for v in (a, b, c)}
    trainable_variables = {v.ref() for v in (b, c)}
    self.assertEqual({v.ref() for v in cf.variables}, captured_variables)
    self.assertEqual({v.ref() for v in cf.trainable_variables},
                     trainable_variables)
    self.assertEqual(cf.variables, cf.graph.variables)
    self.assertEqual(cf.trainable_variables, cf.graph.trainable_variables)

  def testNestedShapeFunctionRelaxation(self):
    # [FUNCTION DOC]
    # Inner reduce-retracing function receives shape vectors of growing length (one, two, three...)
    # and records the traced shape; outer function receives tensors of growing rank. Asserts the first
    # traced shape is one and, for ranks two to five, the inner function was relaxed to unknown shape.
    traced_shape = None
    # The inner function will go through shape relaxation because the shapes it
    # receives will be [1], [2], [3], ...
    @polymorphic_function.function(reduce_retracing=True)
    def bar(x_shape):
      # [FUNCTION DOC]
      # Inner function: records the traced static shape tuple of its argument in an outer variable and
      # returns the argument.
      nonlocal traced_shape
      traced_shape = x_shape._shape_tuple()
      return x_shape

    # The outer function will not go through shape relaxation because the shapes
    # it receives will be [1], [[1]], [[[1]]], ...
    @polymorphic_function.function(reduce_retracing=True)
    def foo(ones):
      # [FUNCTION DOC]
      # Outer function: returns the result of calling the inner function with the dynamic shape of its
      # argument.
      return bar(array_ops.shape(ones))

    self.assertAllEqual(self.evaluate(foo(array_ops.ones([1]))), [1])
    self.assertEqual(traced_shape, (1,))

    for rank in range(2, 6):
      x_shape = self.evaluate(foo(array_ops.ones([1] * rank)))
      self.assertAllEqual(x_shape, [1] * rank)
      self.assertEqual(traced_shape, (None,))

  def testNoHash(self):
    # [FUNCTION DOC]
    # Calls a function with a set (unhashable, unsupported type) and asserts a type error saying a
    # generic trace type could not be generated.

    @polymorphic_function.function()
    def f(_):
      # [FUNCTION DOC]
      # Traced function that ignores its argument and returns one (float).
      return 1.0

    with self.assertRaisesRegex(
        TypeError, r'Could not generate a generic TraceType'):
      f(set([]))

  def testBasicGraphMode(self):
    # [FUNCTION DOC]
    # Defines a function calling a wrapped matmul of its argument with itself, calls it on a two-by-two
    # matrix and asserts the result equals the direct matrix product.
    matmul = polymorphic_function.function(math_ops.matmul)

    @polymorphic_function.function
    def sq(a):
      # [FUNCTION DOC]
      # Function that returns the wrapped matmul of its argument with itself.
      return matmul(a, a)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    out = sq(t)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testNestedInputsGraphMode(self):
    # [FUNCTION DOC]
    # Passes a named tuple of two dictionaries of tensors to a function that multiplies the two
    # contained matrices. Asserts the result equals the direct matrix product, showing nested structure
    # inputs are flattened properly.
    matmul = polymorphic_function.function(math_ops.matmul)

    pair = collections.namedtuple('pair', ['a', 'b'])

    @polymorphic_function.function
    def a_times_b(inputs):
      # [FUNCTION DOC]
      # Takes a named tuple whose fields are dictionaries and returns the matrix product of the tensor
      # under key a in the first and key b in the second.
      return matmul(inputs.a['a'], inputs.b['b'])

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    out = a_times_b(pair({'a': t}, {'b': t}))
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testNestedOutputsGraphMode(self):
    # [FUNCTION DOC]
    # Calls a function returning a named tuple of two matrix products and asserts the result is
    # close to the expected named tuple, showing nested structured outputs are preserved.
    matmul = polymorphic_function.function(math_ops.matmul)

    pair = collections.namedtuple('pair', ['a', 'b'])

    @polymorphic_function.function()
    def pairs_mul(pair_a, pair_b):
      # [FUNCTION DOC]
      # Takes two named tuples and returns a named tuple whose first field is the product of the
      # first fields and whose second is the product of the second fields.
      return pair(matmul(pair_a.a, pair_b.a), matmul(pair_a.b, pair_b.b))

    a = constant_op.constant([[1.0, 2.0], [1.0, 2.0]])
    b = constant_op.constant([[3.0, 4.0], [3.0, 4.0]])

    out = pairs_mul(pair(a, b), pair(b, a))
    expected = pair(
        math_ops.matmul(a, b).numpy(),
        math_ops.matmul(b, a).numpy())
    self.assertAllClose(out, expected)

  def testNestedFunctionGraphNotOutOfDate(self):
    # [FUNCTION DOC]
    # Defines an outer method that stores a concrete function of an inner function on its instance.
    # After calling it, checks the inner concrete graph's outer graph is the outer function's graph,
    # then deletes the instance and asserts the outer function and its graph can be garbage collected
    # (weak references become empty) while the inner graph's outer graph falls back to the default graph.

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Inner function that returns the constant one (float).
      return constant_op.constant(1.)

    class _Model(object):
      # [CLASS DOC]
      # Local helper class whose one method, decorated as a function, stores a concrete function on the
      # instance. Used for the garbage-collection check.

      @polymorphic_function.function
      def g(self):
        # [FUNCTION DOC]
        # Method that stores the concrete function of the outer-scope inner function as an attribute of the
        # instance. Returns nothing.
        self.f = f.get_concrete_function()

    model = _Model()
    model.g()
    concrete = model.f
    weak_g_graph = weakref.ref(model.g.get_concrete_function().graph)
    self.assertIs(weak_g_graph(), concrete.graph.outer_graph)
    weak_g = weakref.ref(model.g)
    del model
    self.assertIsNone(weak_g())
    self.assertIsNone(weak_g_graph())
    self.assertIsNotNone(concrete.graph.outer_graph)
    self.assertIs(ops.get_default_graph(), concrete.graph.outer_graph)

  def testBasicGraphFunction(self):
    # [FUNCTION DOC]
    # Gets a concrete function for the matmul-squaring function with a sample matrix, asserts its output
    # shape is two-by-two, calls it and asserts the result equals the direct product.
    matmul = polymorphic_function.function(math_ops.matmul)

    @polymorphic_function.function
    def sq(a):
      # [FUNCTION DOC]
      # Function that returns the wrapped matmul of its argument with itself.
      return matmul(a, a)

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    sq_op = sq.get_concrete_function(t)
    self.assertEqual(sq_op.output_shapes, tensor_shape.TensorShape([2, 2]))
    out = sq_op(t)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testGetConcreteFunctionThreadSafety(self):
    # [FUNCTION DOC]
    # Calls the get-concrete-function method from one hundred threads in a pool and asserts all calls
    # returned the same single concrete function object (the set size is one).

    @polymorphic_function.function
    def sq():
      # [FUNCTION DOC]
      # Function with no arguments that builds a constant matrix and returns its product with itself.
      t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
      return math_ops.matmul(t, t)

    concrete_functions = []

    def thread_func(_):
      # [FUNCTION DOC]
      # Thread target: ignores its argument, gets the concrete function and appends it to a shared list.
      cf = sq.get_concrete_function()
      concrete_functions.append(cf)

    num_threads = 100
    pool = multiprocessing.pool.ThreadPool(num_threads)
    _ = pool.map(thread_func, list(range(num_threads)))

    self.assertLen(set(concrete_functions), 1)

  def testGetConcreteFunctionThreadSafetyWithArgs(self):
    # [FUNCTION DOC]
    # Gets concrete functions for a variadic add-all function from two threads with one hundred tensor
    # arguments, then calls the first one with those arguments. Checks that concurrent tracing with
    # many arguments does not crash or deadlock; makes no explicit equality assertion.

    @polymorphic_function.function
    def add_100(*args):
      # [FUNCTION DOC]
      # Variadic function that returns the elementwise sum of all positional arguments.
      return math_ops.add_n(args)

    p = multiprocessing.pool.ThreadPool(2)
    args = (constant_op.constant(1.),) * 100
    f1, f2 = p.map(add_100.get_concrete_function, [args] * 2)
    # I see about len(args) + max(0, len(args) - 3) arguments expected.
    f1(*args)
    del f2

  def testInputSpecGraphFunction(self):
    # [FUNCTION DOC]
    # Gets a concrete function using an unknown-shape float spec and asserts its output shape is
    # unknown in both dimensions; then calls it on two different-valued two-by-two matrices and asserts
    # each result equals the direct matrix product.
    matmul = polymorphic_function.function(math_ops.matmul)

    @polymorphic_function.function
    def sq(a):
      # [FUNCTION DOC]
      # Function that returns the wrapped matmul of its argument with itself.
      return matmul(a, a)

    sq_op = sq.get_concrete_function(
        tensor_lib.TensorSpec((None, None), dtypes.float32))
    self.assertEqual([None, None], sq_op.output_shapes.as_list())

    t1 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    out1 = sq_op(t1)
    self.assertAllEqual(out1, math_ops.matmul(t1, t1).numpy())

    t2 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    out2 = sq_op(t2)
    self.assertAllEqual(out2, math_ops.matmul(t2, t2).numpy())

  def testNestedInputSpecGraphFunction(self):
    # [FUNCTION DOC]
    # Gets concrete functions from nested specs (a list holding a tuple of two specs), once with
    # unnamed and once with named specs. Asserts unknown output shapes, then calls the named one by
    # keyword with the names given and the auto-named one positionally, and asserts both equal the
    # direct matrix product.
    matmul = polymorphic_function.function(math_ops.matmul)

    @polymorphic_function.function
    def sq(mats):
      # [FUNCTION DOC]
      # Takes a list containing one pair of matrices and returns their matrix product.
      ((a, b),) = mats
      return matmul(a, b)

    sq_op_autonamed = sq.get_concrete_function([(
        tensor_lib.TensorSpec((None, None), dtypes.float32),
        tensor_lib.TensorSpec((None, None), dtypes.float32),
    )])
    self.assertEqual([None, None], sq_op_autonamed.output_shapes.as_list())

    sq_op = sq.get_concrete_function([(
        tensor_lib.TensorSpec((None, None), dtypes.float32, name='first_mat'),
        tensor_lib.TensorSpec((None, None), dtypes.float32, name='second_mat'),
    )])
    self.assertEqual([None, None], sq_op.output_shapes.as_list())

    t1 = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    t2 = constant_op.constant([[1.4, 2.4], [3.4, 4.4]])
    out = sq_op(first_mat=t1, second_mat=t2)
    self.assertAllEqual(out, math_ops.matmul(t1, t2).numpy())
    self.assertAllEqual(
        sq_op_autonamed(t1, t2),
        math_ops.matmul(t1, t2).numpy())

  def testExecutingStatelessDefunConcurrently(self):
    # [FUNCTION DOC]
    # Runs a stateless doubling function on one hundred inputs from a thread pool and asserts the
    # outputs equal twice each input in order.

    @polymorphic_function.function
    def stateless(x):
      # [FUNCTION DOC]
      # Returns its argument multiplied by two.
      return math_ops.multiply(2.0, x)

    pool = multiprocessing.pool.ThreadPool()
    inputs = [constant_op.constant(1.0 * x) for x in range(100)]
    outputs = [float(out) for out in pool.map(stateless, inputs)]
    expected = [float(2.0 * x) for x in inputs]
    self.assertSequenceEqual(outputs, expected)

  def testExecutingManyStatelessDefunsConcurrently(self):
    # [FUNCTION DOC]
    # Calls a stateless function from a thread pool with one hundred distinct plain object arguments
    # (each causing its own trace) and asserts every output is four.

    @polymorphic_function.function
    def stateless(x):
      # [FUNCTION DOC]
      # Deletes its argument and returns the product of two and two (always four).
      del x
      return math_ops.multiply(2.0, 2.0)

    pool = multiprocessing.pool.ThreadPool()
    # `pool.map` below instantiates 100 functions, one for each object.
    objects = [object() for _ in range(100)]
    outputs = [float(out) for out in pool.map(stateless, objects)]
    expected = [4.0] * 100
    self.assertSequenceEqual(outputs, expected)

  @test_util.disable_tfrt('b/169431085: This test is flaky on tfrt')
  def testExecutingStatefulDefunConcurrently(self):
    # [FUNCTION DOC]
    # Runs a stateful function that assigns its argument to a shared variable on one hundred inputs
    # from a thread pool, then asserts the variable holds zero. Skipped under the TFRT runtime because
    # it is flaky there (decorator).

    v = resource_variable_ops.ResourceVariable(1.0)

    @polymorphic_function.function
    def stateful(x):
      # [FUNCTION DOC]
      # Assigns its argument to the captured variable and returns nothing.
      v.assign(x)

    pool = multiprocessing.pool.ThreadPool()
    inputs = [constant_op.constant(0.0)] * 100
    pool.map(stateful, inputs)
    self.assertEqual(float(v.read_value()), 0.0)

  def testExecutingManyStatefulDefunsConcurrently(self):
    # [FUNCTION DOC]
    # Runs a stateful function that assigns zero to a shared variable from a thread pool with one
    # hundred distinct objects (each traced separately), then asserts the variable is zero.

    v = resource_variable_ops.ResourceVariable(1.0)

    @polymorphic_function.function
    def stateful(x):
      # [FUNCTION DOC]
      # Deletes its argument, assigns zero to the captured variable and returns the assign result.
      del x
      return v.assign(0.0)

    pool = multiprocessing.pool.ThreadPool()
    # `pool.map` below instantiates 100 functions, one for each object.
    pool.map(stateful, [object() for _ in range(100)])
    self.assertEqual(float(v.read_value()), 0.0)

  def testShareRendezvous(self):
    # [FUNCTION DOC]
    # Disables the meta optimizer so functions are not inlined, defines send and receive functions
    # marked as sharing the rendezvous, wraps them in body functions with an integer signature and a
    # loop condition, and runs them through a functional while loop in graph mode with a count of two.
    # Passes if the send and receive match up across the loop boundary without hanging or erroring.
    # KNOWN ISSUE: the optimizer option is set on the global context and is never restored, so it
    # leaks into later tests.

    # Disable grappler from inlining the functions. Note we run the send & recv
    # in graph mode since with eager mode the function should automatically be
    # inlined.
    context.context().set_optimizer_experimental_options(
        {'disable_meta_optimizer': True})

    cpu = '/device:CPU:0'

    signature = [tensor_lib.TensorSpec([], dtypes.int32)]

    @polymorphic_function.function
    def send():
      # [FUNCTION DOC]
      # Creates constant one and sends it with the send op named x between the CPU device and itself;
      # returns the constant.
      x = constant_op.constant(1)
      gen_sendrecv_ops.send(x, 'x', cpu, 0, cpu)
      return x

    send._shared_rendezvous = True  # pylint: disable=protected-access

    @polymorphic_function.function(input_signature=signature)
    def send_body(n):
      # [FUNCTION DOC]
      # Calls the send function and returns its loop argument minus one.
      send()
      return n - 1

    @polymorphic_function.function
    def recv():
      # [FUNCTION DOC]
      # Receives an int32 tensor named x on the CPU device and returns it.
      return gen_sendrecv_ops.recv(dtypes.int32, 'x', cpu, 0, cpu)

    recv._shared_rendezvous = True  # pylint: disable=protected-access

    @polymorphic_function.function(input_signature=signature)
    def recv_body(n):
      # [FUNCTION DOC]
      # Calls the receive function and returns its loop argument minus one.
      recv()
      return n - 1

    @polymorphic_function.function(input_signature=signature)
    def cond_fn(n):
      # [FUNCTION DOC]
      # Loop condition: returns whether the argument is greater than zero.
      return n > 0

    # Instead of calling the send & recv functions directly we want to call them
    # through a functional while to ensure the rendezvous is shared across the
    # while boundary.
    @polymorphic_function.function
    def fn(n):
      # [FUNCTION DOC]
      # Runs a functional while loop calling the send body, then returns the result of a functional while
      # loop running the receive body, both using the loop condition and the argument as the loop state.
      functional_ops.While([n], cond_fn.get_concrete_function(),
                           send_body.get_concrete_function())
      return functional_ops.While([n], cond_fn.get_concrete_function(),
                                  recv_body.get_concrete_function())

    # Use a graph context since functions will not be automatically inlined
    with context.graph_mode(), self.cached_session():
      self.evaluate(fn(2))

  def disabled_testRandomSeed(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name starts with "disabled_" and not "test", so the test runner never
    # collects it; this test is dead code.
    # Intended scenario: with seed one, two calls of a random-normal function should differ, and
    # resetting the seed to one should reproduce the first value.

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Returns a scalar random normal sample.
      return random_ops.random_normal(())

    random_seed.set_random_seed(1)
    x = f()
    self.assertNotEqual(x, f())
    random_seed.set_random_seed(1)
    self.assertAllEqual(f(), x)

  def testNestedInputsGraphFunction(self):
    # [FUNCTION DOC]
    # Gets a concrete function for a function taking a named tuple of dictionaries, using specs named
    # a and b of shape two-by-two. Asserts the output shape is two-by-two and calling it with keyword
    # arguments a and b gives the direct matrix product.
    matmul = polymorphic_function.function(math_ops.matmul)

    pair = collections.namedtuple('pair', ['a', 'b'])

    @polymorphic_function.function
    def a_times_b(inputs):
      # [FUNCTION DOC]
      # Takes a named tuple whose fields are dictionaries and returns the matrix product of the entry
      # under key a in the first and key b in the second.
      return matmul(inputs.a['a'], inputs.b['b'])

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    sq_op = a_times_b.get_concrete_function(
        pair(
            dict(a=tensor_lib.TensorSpec([2, 2], dtypes.float32, 'a')),
            dict(b=tensor_lib.TensorSpec([2, 2], dtypes.float32, 'b'))))
    self.assertEqual(sq_op.output_shapes, tensor_shape.TensorShape([2, 2]))
    out = sq_op(a=t, b=t)
    self.assertAllEqual(out, math_ops.matmul(t, t).numpy())

  def testNestedOutputGraphFunction(self):
    # [FUNCTION DOC]
    # Gets a concrete function whose output is a tuple of a matrix product and a dictionary with a
    # scalar constant. Asserts the output shapes and dtypes keep the nested structure and that calling
    # it returns the product and one.
    matmul = polymorphic_function.function(math_ops.matmul)

    @polymorphic_function.function
    def sq(a):
      # [FUNCTION DOC]
      # Returns a tuple: the matrix product of the argument with itself, and a dictionary with key b
      # mapping to the constant one.
      return (matmul(a, a), {'b': constant_op.constant(1.0)})

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])

    sq_op = sq.get_concrete_function(t)
    self.assertEqual(sq_op.output_shapes, (tensor_shape.TensorShape([2, 2]), {
        'b': tensor_shape.TensorShape([])
    }))
    self.assertEqual(sq_op.output_dtypes, (dtypes.float32, {
        'b': dtypes.float32
    }))
    (a, b) = sq_op(t)
    self.assertAllEqual(a, math_ops.matmul(t, t).numpy())
    self.assertAllEqual(b['b'].numpy(), 1.0)

  def testZipStrictBuiltin(self):
    # [FUNCTION DOC]
    # Skips unless the interpreter is Python 3.10 or newer. Traces a function using the builtin strict
    # zip on lists and asserts the first zipped item equals two, checking autograph handles that
    # keyword.
    major, minor, _ = platform.python_version_tuple()
    if not (major == '3' and int(minor) >= 10):
      self.skipTest('strict zip is only supported in Python 3.10+')

    @polymorphic_function.function
    def foo(x):
      # [FUNCTION DOC]
      # Returns a list of tuples built by strictly zipping the argument with itself twice.
      return list(zip([x], [x], strict=True))

    self.assertEqual(foo(2)[0][0].numpy(), 2)

  def testGraphFunctionNoneOutput(self):
    # [FUNCTION DOC]
    # Gets a concrete function whose function returns nothing. Asserts its output dtypes and shapes are
    # null and that calling it returns null.
    @polymorphic_function.function
    def fn(unused_a, unused_b):
      # [FUNCTION DOC]
      # Ignores both arguments and returns nothing.
      return None

    x = constant_op.constant(1)
    fn_op = fn.get_concrete_function(x, x)
    self.assertEqual(fn_op.output_dtypes, None)
    self.assertEqual(fn_op.output_shapes, None)
    self.assertAllEqual(fn_op(x, x), None)

  def testDefunCapturedInt32(self):
    # [FUNCTION DOC]
    # Captures an int32 constant in a function that adds it to itself and asserts the integer result
    # is two (int32 values are captured and returned correctly).
    x = constant_op.constant(1, dtype=dtypes.int32)

    @polymorphic_function.function
    def add_int32s():
      # [FUNCTION DOC]
      # Returns the captured int32 constant added to itself.
      return x + x

    self.assertEqual(2, int(add_int32s()))

  def testDefunReadVariable(self):
    # [FUNCTION DOC]
    # Reads a captured resource variable inside a function and asserts the float result is one.
    v = resource_variable_ops.ResourceVariable(1.0)

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Returns the current value of the captured variable.
      return v.read_value()

    self.assertEqual(1.0, float(f()))

  def testDefunAssignAddVariable(self):
    # [FUNCTION DOC]
    # Adds a captured constant to a captured variable inside a function and returns the read value;
    # asserts the result is three.
    v = resource_variable_ops.ResourceVariable(1.0)
    x = constant_op.constant(2.0)

    @polymorphic_function.function
    def test_assign_add():
      # [FUNCTION DOC]
      # Adds the captured tensor to the captured variable and returns the read value.
      v.assign_add(x)
      return v.read_value()

    self.assertEqual(3.0, float(test_assign_add()))

  @test_util.run_in_graph_and_eager_modes
  def testTensorInitializationInFunctionRaisesError(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Inside a function, creating a resource variable from a
    # tensor computed in the function is asserted to raise a value error saying it could not be lifted
    # out of the function.

    @polymorphic_function.function
    def tensor_init():
      # [FUNCTION DOC]
      # Asserts that creating a resource variable from a constant tensor raises a value error mentioning
      # "could not be lifted out". Returns nothing.
      with self.assertRaisesRegex(ValueError, 'could not be lifted out'):
        resource_variable_ops.ResourceVariable(constant_op.constant(2.0))

    tensor_init()

  @test_util.run_in_graph_and_eager_modes
  def testCallableTensorInitializationInFunction(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Creates a resource variable from an initializer callable inside a
    # function, returns its read value, initializes variables if in graph mode and asserts the value
    # is two.

    @polymorphic_function.function
    def tensor_init():
      # [FUNCTION DOC]
      # Creates a resource variable (stored on the test instance) from a callable returning two, and
      # returns its read value.
      self.v = resource_variable_ops.ResourceVariable(
          lambda: constant_op.constant(2.0))
      return self.v.read_value()

    value = tensor_init()
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
    self.assertEqual(self.evaluate(value), 2.0)

  @test_util.also_run_as_tf_function
  def testInitScopeTensorInitializationInFunction(self):
    # [FUNCTION DOC]
    # Also run as a tf.function. Creates a constant inside an init scope within the function and uses
    # it to build a resource variable, then asserts the returned value is two.

    @polymorphic_function.function
    def tensor_init():
      # [FUNCTION DOC]
      # Creates a constant inside an init scope, builds a resource variable from it (kept on the test
      # instance) and returns the variable's read value.
      with ops.init_scope():
        const = constant_op.constant(2.0)
      # Note: this variable bypasses tf.function's variable creation
      # requirements by bypassing variable_creator_scope by using
      # ResourceVariable instead of Variable.
      self.v = resource_variable_ops.ResourceVariable(const)
      return self.v.read_value()

    value = tensor_init()
    self.assertAllEqual(value, 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testGetConcreteFunctionCreatesVariables(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Gets a concrete function from a function that creates a variable
    # on first call, initializes variables, and asserts both the concrete function and the
    # function itself return five.

    v_holder = []

    @polymorphic_function.function
    def tensor_init():
      # [FUNCTION DOC]
      # Creates a variable with value five only the first time (stored in a holder) and returns its read
      # value.
      if not v_holder:
        v_holder.append(variables.Variable(5.))
      return v_holder[0].read_value()

    concrete = tensor_init.get_concrete_function()
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(5., self.evaluate(concrete()))
    self.assertAllEqual(5., self.evaluate(tensor_init()))

  def testDefunShapeInferenceWithCapturedResourceVariable(self):
    # [FUNCTION DOC]
    # Wraps a function that multiplies a captured variable by a constant (asserting the static output
    # shape is two-by-two) and returns the variable's handle. Asserts the handle has resource dtype and
    # scalar shape, and that reading it gives two-by-two shape.
    v = resource_variable_ops.ResourceVariable([[1, 2], [3, 4]])

    def f():
      # [FUNCTION DOC]
      # Multiplies a captured variable by a constant matrix, asserts the static shape is two-by-two, and
      # returns the variable's resource handle.
      x = constant_op.constant([[1, 2], [3, 4]])
      out = math_ops.matmul(v, x)
      self.assertEqual(out.shape, tensor_shape.TensorShape([2, 2]))
      # We do not return v directly since the tensor conversion function of
      # ResourceVariable returns the read value and not the resource itself.
      return v._handle

    compiled = polymorphic_function.function(f)
    var_handle = compiled()
    self.assertEqual(var_handle.dtype, dtypes.resource)
    self.assertEqual(var_handle.shape, tensor_shape.TensorShape([]))
    var_t = resource_variable_ops.read_variable_op(var_handle, dtype=v.dtype)
    self.assertEqual(var_t.shape, tensor_shape.TensorShape([2, 2]))

  def testShapeInferenceForMoreSpecificInput(self):
    # [FUNCTION DOC]
    # Builds a function with an unknown-shape input signature that reshapes to rows of three. Inside
    # another function, calls it with a ten-by-ten-by-three tensor and asserts the result shape equals
    # that of the unwrapped reshape.

    def f(a):
      # [FUNCTION DOC]
      # Reshapes its argument into rows of length three.
      return array_ops.reshape(a, [-1, 3])

    signature = [tensor_lib.TensorSpec(None, dtypes.float32)]
    compiled = polymorphic_function.function(f, input_signature=signature)

    @polymorphic_function.function
    def use_f():
      # [FUNCTION DOC]
      # Creates a zeros tensor of shape ten by ten by three and asserts the shapes from the plain and
      # wrapped function agree.
      inputs = array_ops.zeros([10, 10, 3])
      self.assertAllEqual(f(inputs).shape, compiled(inputs).shape)

    use_f()

  def testDefunShapeInferenceWithCapturedResourceVariableInGraphMode(self):
    # [FUNCTION DOC]
    # Same as the eager variant but in graph mode: the inner function multiplies a captured resource
    # variable and returns its handle; asserts the handle's dtype and shape and the shape of the value
    # read from it.
    with context.graph_mode():
      v = resource_variable_ops.ResourceVariable([[1, 2], [3, 4]])

      def f():
        # [FUNCTION DOC]
        # Multiplies the captured variable by a constant, asserts the shape is two-by-two and returns the
        # variable's resource handle.
        x = constant_op.constant([[1, 2], [3, 4]])
        out = math_ops.matmul(v, x)
        self.assertEqual(out.shape, tensor_shape.TensorShape([2, 2]))
        # We do not return v directly since the tensor conversion function of
        # ResourceVariable returns the read value and not the resource itself.
        return v._handle

      compiled = polymorphic_function.function(f)
      var_handle = compiled()
      self.assertEqual(var_handle.dtype, dtypes.resource)
      self.assertEqual(var_handle.shape, tensor_shape.TensorShape([]))
      var_t = resource_variable_ops.read_variable_op(var_handle, dtype=v.dtype)
      self.assertEqual(var_t.shape, tensor_shape.TensorShape([2, 2]))

  def testDefunShapeInferenceWithCapturedVariableInGraphMode(self):
    # [FUNCTION DOC]
    # In graph mode, wraps a function that multiplies a captured variable by a constant and asserts the
    # static shape is two-by-two; calling it checks shape inference works while creating the function.
    with context.graph_mode():
      v = variables.Variable([[1, 2], [3, 4]])

      def f():
        # [FUNCTION DOC]
        # Multiplies the captured variable by a constant and asserts the static shape is two-by-two.
        # Returns nothing.
        x = constant_op.constant([[1, 2], [3, 4]])
        out = math_ops.matmul(v, x)
        self.assertEqual(out.shape, tensor_shape.TensorShape([2, 2]))

      # Check that shape inference works while creating the defun
      compiled = polymorphic_function.function(f)
      compiled()

  def testDefunShapeInferenceWithCapturedTensorListInGraphMode(self):
    # [FUNCTION DOC]
    # In graph mode, builds a tensor list with two pushed elements captured by a function that pops
    # one and returns the remaining list. Asserts the popped value has scalar shape in the function and
    # again outside after popping from the output list.
    with context.graph_mode():
      tensor_list = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=ops.convert_to_tensor([], dtype=dtypes.int32))
      tensor_list = list_ops.tensor_list_push_back(tensor_list,
                                                   constant_op.constant(1.0))
      tensor_list = list_ops.tensor_list_push_back(tensor_list,
                                                   constant_op.constant(2.0))

      def f():
        # [FUNCTION DOC]
        # Pops the last element from the captured list, asserts its shape is scalar and returns the list.
        tl, value = list_ops.tensor_list_pop_back(
            tensor_list, element_dtype=dtypes.float32)
        self.assertEqual(value.shape, tensor_shape.TensorShape([]))
        return tl

      compiled = polymorphic_function.function(f)
      output_tensor_list = compiled()
      _, value = list_ops.tensor_list_pop_back(
          output_tensor_list, element_dtype=dtypes.float32)
      self.assertEqual(value.shape, tensor_shape.TensorShape([]))

  def testRunMetadata(self):
    # [FUNCTION DOC]
    # Enables run metadata on the context, calls a squaring function on cpu:0, exports and disables
    # metadata and asserts exactly one partition graph was recorded.

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument squared.
      return x * x

    with ops.device('cpu:0'):
      context.enable_run_metadata()
      f(constant_op.constant(1.0))
    run_metadata = context.export_run_metadata()
    context.disable_run_metadata()
    self.assertLen(run_metadata.partition_graphs, 1)

  def testGraphModeCaptureVariable(self):
    # [FUNCTION DOC]
    # In a graph-mode session, defines a class holding a resource variable and a method that doubles
    # it, wraps the bound method as a function, initializes variables and asserts the evaluated result
    # is two.
    with context.graph_mode(), self.cached_session():

      class HasAVar:
        # [CLASS DOC]
        # Local helper class holding one resource variable (value one) and a method returning it doubled.

        def __init__(self):
          # [FUNCTION DOC]
          # Creates a resource variable initialized to one and stores it on the instance.
          self.v = resource_variable_ops.ResourceVariable(1.0)

        def call(self):
          # [FUNCTION DOC]
          # Returns the stored variable multiplied by two.
          return self.v * 2

      o = HasAVar()
      self.evaluate(variables.global_variables_initializer())
      call = polymorphic_function.function(o.call)
      op = call()
      self.assertAllEqual(self.evaluate(op), 2.0)

  def testGraphModeManyFunctions(self):
    # [FUNCTION DOC]
    # In graph mode, defines a square function and a second function calling it plus one; asserts
    # calling the second on two gives five.
    with ops.Graph().as_default(), self.cached_session():

      @polymorphic_function.function
      def f(x):
        # [FUNCTION DOC]
        # Returns its argument squared.
        return x * x

      @polymorphic_function.function
      def g(x):
        # [FUNCTION DOC]
        # Returns the squaring function applied to its argument plus one.
        return f(x) + 1

      self.assertAllEqual(g(constant_op.constant(2.0)), 5.0)

  def testDict(self):
    # [FUNCTION DOC]
    # Calls a function returning a dictionary and asserts the value under the key name is the input
    # plus one (two).

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns a dictionary mapping the key name to the argument plus one.
      return {'name': x + 1}

    self.assertAllEqual(f(constant_op.constant(1.0))['name'], 2.0)

  def testWeakrefInputsRejected(self):
    # [FUNCTION DOC]
    # Passes a weak reference to an instance of a dummy class to a function and asserts a type error
    # mentioning "weakref" is raised (weak-reference inputs are not supported as arguments).

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    class Dummy:
      # [CLASS DOC]
      # Empty local class whose instance is the target of the weak reference passed to the function.
      pass

    o = Dummy()
    wr = weakref.ref(o)

    with self.assertRaisesRegex(TypeError, 'weakref'):
      f(wr)

  def testTensorConversionWithDefun(self):
    # [FUNCTION DOC]
    # Calls a function that adds a constant three to its tensor argument with a tensor of two and
    # asserts the result is five.

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns the argument added to the constant three.
      return math_ops.add(x, constant_op.constant(3))

    self.assertAllEqual(5, f(constant_op.constant(2)))

  def testTensorConversionCall(self):
    # [FUNCTION DOC]
    # Defines an add-three function and a second function that applies it twice; asserts calling the
    # second with two gives eight.

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns the argument added to the constant three.
      return math_ops.add(x, constant_op.constant(3))

    @polymorphic_function.function
    def g(x):
      # [FUNCTION DOC]
      # Applies the add-three function twice to its argument and returns the result.
      return f(f(x))

    self.assertAllEqual(8, g(constant_op.constant(2)))

  def testCallShape(self):
    # [FUNCTION DOC]
    # Calls a function from within another function and asserts inside it that the result of the
    # inner call keeps a scalar static shape. The outer function returns nothing.

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument plus one.
      return x + 1

    @polymorphic_function.function
    def g(x):
      # [FUNCTION DOC]
      # Calls the inner function, asserts the result has an empty (scalar) static shape and returns
      # nothing.
      x = f(x)
      self.assertEqual(x.shape.as_list(), [])
      return None

    g(constant_op.constant(1.0))

  def testNestedDefunWithNoOutputAndTapedInput(self):
    # [FUNCTION DOC]
    # Creates a resource variable and calls an outer function that passes a derived value to an inner
    # function which returns nothing and takes a variable-derived input. Passes if no error is raised
    # when a function with no outputs is nested.
    three = resource_variable_ops.ResourceVariable(3.0, name='v')

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Adds its argument to a captured variable but returns nothing (the result is discarded).
      # This function intentionally takes a taped variable as input,
      # but does not return any values
      math_ops.add(x, three)

    @polymorphic_function.function
    def g(x):
      # [FUNCTION DOC]
      # Adds its argument to the captured variable and passes the sum to the inner function; returns
      # nothing.
      y = math_ops.add(x, three)
      f(y)

    g(three)

  def testGatherResourceWithDefun(self):
    # [FUNCTION DOC]
    # Creates a resource variable on cpu:0 and a function that sums two gathered elements of it.
    # Asserts the plain call and the wrapped call give the same result.
    with ops.device('cpu:0'):
      v = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

    def sum_gather():
      # [FUNCTION DOC]
      # Returns the sum of the elements of the captured variable at indices one and two.
      return math_ops.reduce_sum(array_ops.gather(v, [1, 2]))

    defined = polymorphic_function.function(sum_gather)
    self.assertAllEqual(sum_gather(), defined())

  @parameterized.named_parameters([
      ('IndexedSlicesWithDenseShape',
       _example_indexed_slices_with_dense_shape,),
      ('IndexedSlicesWithoutDenseShape',
       _example_indexed_slices_without_dense_shape,),
      ('RaggedTensorRaggedRank1', ragged_tensor.RaggedTensor.from_row_lengths,
       {'values': [1, 2, 3], 'row_lengths': [2, 0, 1]}),
      ('RaggedTensorRaggedRank2',
       ragged_tensor.RaggedTensor.from_nested_row_lengths,
       {'flat_values': [1, 2, 3], 'nested_row_lengths': [[1, 2], [2, 0, 1]]}),
      ('SparseTensor', sparse_tensor.SparseTensor,
       {'values': [1, 2, 3], 'indices': [[0], [8], [10]], 'dense_shape': [20]}),
  ])  # pyformat: disable
  def testReturnCompositeTensorWithDefun(self,
                                         factory_fn,
                                         factory_kwargs={},
                                         input_signature=None):
    # [FUNCTION DOC]
    # Parameterized over composite tensor kinds (indexed slices with and without dense shape, ragged
    # tensors of rank one and two, a sparse tensor). Takes a factory callable, a mapping of keyword
    # arguments for it and an optional input signature. Builds a composite value, calls a function
    # with that signature that returns the captured composite, and asserts the output has the same
    # type, the same nested structure, and equal flattened components as the input.
    input_ct = factory_fn(**factory_kwargs)

    @polymorphic_function.function(input_signature=input_signature)
    def f():
      # [FUNCTION DOC]
      # Takes no arguments and returns the captured composite tensor.
      return input_ct

    output_ct = f()
    self.assertIsInstance(output_ct, type(input_ct))
    nest.assert_same_structure(input_ct, output_ct, expand_composites=True)

    input_flat = nest.flatten(input_ct, expand_composites=True)
    output_flat = nest.flatten(output_ct, expand_composites=True)
    for (input_component, output_component) in zip(input_flat, output_flat):
      self.assertAllEqual(input_component, output_component)

  @parameterized.named_parameters([
      ('IndexedSlicesWithDenseShape',
       _example_indexed_slices_with_dense_shape,),
      ('IndexedSlicesWithoutDenseShape',
       _example_indexed_slices_without_dense_shape,),
      ('RaggedTensorRaggedRank1',
       ragged_tensor.RaggedTensor.from_row_lengths,
       {'values': [1, 2, 3], 'row_lengths': [2, 0, 1]}),
      ('RaggedTensorRaggedRank2',
       ragged_tensor.RaggedTensor.from_nested_row_lengths,
       {'flat_values': [1, 2, 3], 'nested_row_lengths': [[1, 2], [2, 0, 1]]}),
      ('SparseTensor',
       sparse_tensor.SparseTensor,
       {'values': [1, 2, 3], 'indices': [[0], [8], [10]], 'dense_shape': [20]}),
      ('RaggedTensorRaggedRank1WithSignature',
       ragged_tensor.RaggedTensor.from_row_lengths,
       {'values': [1, 2, 3], 'row_lengths': [2, 0, 1]},
       [ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)]),
      ('RaggedTensorRaggedRank2WithSignature',
       ragged_tensor.RaggedTensor.from_nested_row_lengths,
       {'flat_values': [1, 2, 3], 'nested_row_lengths': [[1, 2], [2, 0, 1]]},
       [ragged_tensor.RaggedTensorSpec([None, None, None], dtypes.int32)]),
      ('SparseTensorWithSignature',
       sparse_tensor.SparseTensor,
       {'values': [1, 2, 3], 'indices': [[0], [8], [10]], 'dense_shape': [20]},
       [sparse_tensor.SparseTensorSpec([None], dtypes.int32)]),
  ])  # pyformat: disable
  def testCompositeAsArgumentTensorWithDefun(self,
                                             factory_fn,
                                             factory_kwargs={},
                                             input_signature=None):
    # [FUNCTION DOC]
    # Parameterized over the same composite kinds, with and without an explicit input signature. Passes
    # a composite value as an argument to an identity function and asserts the output has the same type
    # and structure as the input and equal flattened components.
    input_ct = factory_fn(**factory_kwargs)

    @polymorphic_function.function(input_signature=input_signature)
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    output_ct = f(input_ct)
    self.assertIsInstance(output_ct, type(input_ct))
    nest.assert_same_structure(input_ct, output_ct, expand_composites=True)

    input_flat = nest.flatten(input_ct, expand_composites=True)
    output_flat = nest.flatten(output_ct, expand_composites=True)
    for (input_component, output_component) in zip(input_flat, output_flat):
      self.assertAllEqual(input_component, output_component)

  def testTracedCompositeDiscardsShapeInfo(self):
    # [FUNCTION DOC]
    # Calls a function with a ragged tensor and a sparse tensor and asserts, while tracing, that the
    # placeholders for the sparse tensor's indices and values have an unknown first dimension, because
    # the type specs deliberately drop element-count information. The function returns its inputs.
    # SparseTensorSpec intentionally excludes info about the number of elements
    # that are in a sparse tensor (which is recorded as st.indices.shape[0] and
    # st.values.shape[0]).  Similarly, RaggedTensorSpec intentionally excludes
    # info about the total number of values in a RaggedTensor (stored as
    # rt.values.shape[0]).  This test checks that the placeholders created by
    # tf.function() properly mask this shape info.
    @polymorphic_function.function
    def f(rt, st):
      # [FUNCTION DOC]
      # Asserts the sparse tensor's indices have unknown leading dimension and its values have unknown
      # shape, then returns both arguments as a pair.
      self.assertEqual(st.indices.shape.as_list()[:1], [None])
      self.assertEqual(st.values.shape.as_list(), [None])
      return (rt, st)

    rt = ragged_factory_ops.constant([[1, 2], [3]])
    st = sparse_tensor.SparseTensor([[0]], [0], [10])
    f(rt, st)

  @test_util.run_gpu_only
  def testFunctionOnDevice(self):
    # [FUNCTION DOC]
    # GPU-only test. Moves a tensor to the GPU, calls a wrapped add of it with itself, brings the result
    # back to the CPU and asserts it equals two.
    x = constant_op.constant([1.]).gpu()
    f = polymorphic_function.function(math_ops.add)
    y = f(x, x).cpu()
    self.assertAllEqual(y, [2.])

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testOpInFunctionWithConflictingResourceInputs(self):
    # [FUNCTION DOC]
    # GPU-only; runs in graph and eager modes. Creates two variables on the CPU and one on the GPU and
    # runs an Adam resource update using all three handles within a function. Asserts an invalid
    # argument error about incompatible colocation groups/assigned devices is raised on execution.
    with ops.device('/cpu:0'):
      v_cpu = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0],
                                                     name='cpu')
      v_also_cpu = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0],
                                                          name='also_cpu')

    with ops.device('/gpu:0'):
      v_gpu = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0],
                                                     name='gpu')

    @polymorphic_function.function
    def resource_apply_adam():
      # [FUNCTION DOC]
      # Applies the Adam update op to the three captured variable handles with fixed scalar settings and
      # a gradient of ones, then returns the integer one.
      gen_training_ops.resource_apply_adam(
          v_cpu.handle,
          v_gpu.handle,
          v_also_cpu.handle,
          1.0,  # beta1_power
          1.0,  # beta2_power
          1.0,  # learning_rate
          1.0,  # beta1
          1.0,  # beta2
          1.0,  # epsilon,
          [1.0, 1.0, 1.0],  # grad
          False)  # use_locking
      return 1

    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Cannot place the graph because a reference or resource edge connects '
        'colocation groups with incompatible assigned devices'):
      if not context.executing_eagerly():
        self.evaluate(variables.global_variables_initializer())
      self.evaluate(resource_apply_adam())

  @test_util.run_gpu_only
  def testFunctionHandlesInputsOnDifferentDevices(self):
    # [FUNCTION DOC]
    # GPU-only. Calls a wrapped reshape with a GPU value and a CPU shape tensor (the op needs the shape
    # in host memory) and asserts the result equals a column of one and two.
    # The Reshape op requires the shape tensor to be placed in host memory.
    reshape = polymorphic_function.function(array_ops.reshape)
    value = constant_op.constant([1., 2.]).gpu()
    shape = constant_op.constant([2, 1])
    reshaped = reshape(value, shape).cpu()
    self.assertAllEqual(reshaped, [[1], [2]])

  @test_util.run_gpu_only
  def testFunctionHandlesInputsPlacedOnTheWrongDeviceGracefully(self):
    # [FUNCTION DOC]
    # GPU-only. Calls a wrapped reshape with a CPU value and a shape tensor placed on the GPU; passes if
    # no error is raised.
    # The Reshape op requires the shape tensor to be placed in host memory.
    reshape = polymorphic_function.function(array_ops.reshape)
    value = constant_op.constant([1., 2.])
    shape = constant_op.constant([2, 1]).gpu()
    reshape(value, shape)  # No error is raised

  def testNoneOutput(self):
    # [FUNCTION DOC]
    # Calls a function that ignores its input and returns nothing; asserts the call returns nothing.

    @polymorphic_function.function
    def my_function(_):
      # [FUNCTION DOC]
      # Ignores its argument and returns nothing.
      return None

    self.assertAllEqual(my_function(1), None)

  def testNestedFunctions(self):
    # [FUNCTION DOC]
    # Defines an older-style graph function (add of two int32 values) and a polymorphic function that
    # calls it with one extra constant; asserts calling with two returns three.
    # TensorFlow function (which is what would be used in TensorFlow graph
    # construction).
    @tf_function.Defun(dtypes.int32, dtypes.int32)
    def add(a, b):
      # [FUNCTION DOC]
      # Graph function taking two int32 tensors and returning their sum.
      return math_ops.add(a, b)

    @polymorphic_function.function
    def add_one(x):
      # [FUNCTION DOC]
      # Returns the result of calling the old-style add function with its argument and the constant one.
      return add(x, 1)

    self.assertAllEqual(3, add_one(constant_op.constant(2)))

  def testVariableCaptureInNestedFunctions(self):
    # [FUNCTION DOC]
    # Reads a captured int32 variable in an inner function called by an outer one; asserts the result
    # is one.
    v = resource_variable_ops.ResourceVariable(1, dtype=dtypes.int32)

    @polymorphic_function.function
    def inner_read():
      # [FUNCTION DOC]
      # Returns the read value of the captured variable.
      return v.read_value()

    @polymorphic_function.function
    def outer():
      # [FUNCTION DOC]
      # Returns the result of calling the inner function.
      return inner_read()

    self.assertEqual(1, int(outer()))

  def testReturnCapturedEagerTensor(self):
    # [FUNCTION DOC]
    # Creates an eager constant and a function that returns it as a captured value; asserts the result
    # is one.
    t = constant_op.constant(1)

    @polymorphic_function.function
    def read():
      # [FUNCTION DOC]
      # Returns the captured eager tensor.
      return t

    self.assertEqual(1, int(read()))

  def testReturnCapturedGraphTensor(self):
    # [FUNCTION DOC]
    # In graph mode with a session, creates a constant, wraps a function returning it, and asserts the
    # evaluated result is one.
    with context.graph_mode(), self.cached_session():
      t = constant_op.constant(1)

      @polymorphic_function.function
      def read():
        # [FUNCTION DOC]
        # Returns the captured graph tensor.
        return t

      self.assertEqual(1, int(self.evaluate(read())))

  def testConcreteFunctionType(self):
    # [FUNCTION DOC]
    # Gets a concrete function (scalar int32 spec) from a function returning a dictionary of its
    # input and a captured tensor. Asserts calling it returns the right values, the function type has
    # exactly one parameter named x with the scalar spec, exactly one capture keyed by the captured
    # tensor's identity with the scalar spec, and the output type equals the trace type of the result.
    y = constant_op.constant(1)

    @polymorphic_function.function
    def foo(x):
      # [FUNCTION DOC]
      # Returns a dictionary mapping the key input to its argument and the key capture to the captured
      # tensor.
      return {'input': x, 'capture': y}

    cf = foo.get_concrete_function(tensor_lib.TensorSpec([], dtypes.int32))
    x = constant_op.constant(2)
    output = cf(x)
    self.assertEqual(set(output.keys()), {'input', 'capture'})
    self.assertEqual(output['input'].numpy(), 2)
    self.assertEqual(output['capture'].numpy(), 1)

    parameters = list(cf.function_type.parameters.values())
    self.assertLen(parameters, 1)
    self.assertEqual(parameters[0].name, 'x')
    self.assertEqual(
        parameters[0].type_constraint,
        tensor_lib.TensorSpec([], dtypes.int32),
    )

    captures = cf.function_type.captures
    self.assertLen(captures, 1)
    self.assertEqual(captures[id(y)], tensor_lib.TensorSpec([], dtypes.int32))

    output = cf.function_type.output
    self.assertEqual(output, trace_type.from_value({'input': x, 'capture': y}))

  def testSequenceInputs(self):
    # [FUNCTION DOC]
    # Wraps the clip-by-global-norm op, calls it with a list of two tensors and a clip norm, and asserts
    # the returned list elements and the global norm are all tensors.
    clip_by_global_norm = polymorphic_function.function(
        clip_ops.clip_by_global_norm)
    t_list = [constant_op.constant(1.0), constant_op.constant(2.0)]
    clipped_list, global_norm = clip_by_global_norm(t_list,
                                                    constant_op.constant(.2))
    for t in clipped_list:
      self.assertIsInstance(t, tensor_lib.Tensor)
    self.assertIsInstance(global_norm, tensor_lib.Tensor)

  def testNestedSequenceInputs(self):
    # [FUNCTION DOC]
    # Calls a function whose input is a nested structure of lists, tuples and tensors. Asserts the
    # returned nested outputs have the right structure (including a tuple inside a list), values (such
    # as two, eight, four, six, ten) and that the second result is the sum fifteen.

    def my_op(inputs):
      # [FUNCTION DOC]
      # Unpacks a nested input into pieces and returns a pair: a nested list structure of doubled values
      # (with one inner tuple), and the sum of several of the inputs.
      a, b, c = inputs
      e, f = b
      g, h = e
      return [a + a, [tuple([f + f, g + g]), h + h], c + c], a + f + g + h + c

    my_eager_op = polymorphic_function.function(my_op)
    ret = my_eager_op([
        constant_op.constant(1),
        [(constant_op.constant(2), constant_op.constant(3)),
         constant_op.constant(4)],
        constant_op.constant(5)
    ])
    self.assertLen(ret, 2)
    self.assertAllEqual(ret[0][0], 2)
    self.assertAllEqual(ret[0][1][0][0], 8)
    self.assertAllEqual(ret[0][1][0][1], 4)
    self.assertIsInstance(ret[0][1][0], tuple)
    self.assertAllEqual(ret[0][1][1], 6)
    self.assertAllEqual(ret[0][2], 10)
    self.assertAllEqual(ret[1], 15)

  def testVariableNamesRespectNameScopesWithDefun(self):
    # [FUNCTION DOC]
    # Inside a function creates a variable in a name scope (not skipped under eager) and asserts the
    # variable's name has the scope prefix. Calls the function.

    @polymorphic_function.function
    def create_variable():
      # [FUNCTION DOC]
      # Opens a name scope, creates a resource variable named bar in it and asserts its name is the
      # scoped name followed by the zero output index.
      with ops.name_scope('foo', skip_on_eager=False):
        v = resource_variable_ops.ResourceVariable(0.0, name='bar')
      self.assertEqual(v.name, 'foo/bar:0')

    create_variable()

  def testVariableNamesRespectNameScopesWithDefunInGraph(self):
    # [FUNCTION DOC]
    # Same as the previous test but in graph mode with an explicit default graph context; asserts the
    # variable name keeps the name scope prefix.
    with context.graph_mode():

      @polymorphic_function.function
      def create_variable():
        # [FUNCTION DOC]
        # Opens a name scope, creates a resource variable named bar with two values and asserts its name is
        # the scoped name.
        with ops.name_scope('foo', skip_on_eager=False):
          v = resource_variable_ops.ResourceVariable([1.0, 2.0], name='bar')
        self.assertEqual(v.name, 'foo/bar:0')

      with ops.get_default_graph().as_default():
        create_variable()

  @test_util.run_in_graph_and_eager_modes
  def testVariablesPlacedOnOutsideDevice(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Creates, inside a method called under a CPU device scope, a variable
    # lazily on an object, and asserts the variable's device mentions the CPU.

    class _Obj(object):
      # [CLASS DOC]
      # Local helper class with a lazily created variable and a function-decorated method that creates it.

      def __init__(self):
        # [FUNCTION DOC]
        # Initializes the variable attribute to nothing.
        self.v = None

      @polymorphic_function.function
      def f(self):
        # [FUNCTION DOC]
        # Method that creates a variable on first call (stored on the instance) and returns the variable
        # plus one.
        if self.v is None:
          self.v = variables.Variable(1.)
        return self.v + 1.

    has_device = _Obj()
    with ops.device('cpu:0'):
      has_device.f()
    self.assertIn('CPU', has_device.v.device)

  @test_util.run_in_graph_and_eager_modes
  def testCallingGraphFunctionOnDifferentDevice(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Obtains a concrete function under a CPU device scope and calls it
    # under the same device, another logical CPU, and no device, asserting the value is always zero.
    # Then does the same for a concrete function obtained without a device scope.

    def func():
      # [FUNCTION DOC]
      # Returns the constant zero.
      return constant_op.constant(0)

    defined = polymorphic_function.function(func)
    with ops.device('cpu:0'):
      cpu_graph_function = defined.get_concrete_function()

    with ops.device('cpu:0'):
      self.assertEqual(
          self.evaluate(cpu_graph_function()), self.evaluate(func()))

    with ops.device('cpu:1'):
      self.assertEqual(0., self.evaluate(cpu_graph_function()))

    with ops.device(None):
      self.assertEqual(0., self.evaluate(cpu_graph_function()))

    default_graph_function = defined.get_concrete_function()
    self.assertEqual(
        self.evaluate(default_graph_function()), self.evaluate(func()))

    with ops.device('cpu:1'):
      self.assertEqual(0., self.evaluate(default_graph_function()))

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testColocateWithRespected(self):
    # [FUNCTION DOC]
    # GPU-only; graph and eager modes. Creates tensors on cpu:0 and gpu:0 and calls a function that
    # returns the placement op's device string under colocation with each; asserts the CPU placement is
    # reported under the first and the GPU placement under the second.
    # TODO(b/113291792): Use multiple CPUs instead of a GPU.
    with ops.device('cpu:0'):
      x = array_ops.identity(1.0)

    with ops.device('gpu:0'):
      y = array_ops.identity(1.0)

    @polymorphic_function.function
    def foo():
      # [FUNCTION DOC]
      # Returns the result of the test op that reports the device it ran on.
      return test_ops.device_placement_op()

    with ops.colocate_with(x):
      self.assertIn(compat.as_bytes('CPU:0'), self.evaluate(foo()))

    with ops.colocate_with(y):
      self.assertIn(compat.as_bytes('GPU:0'), self.evaluate(foo()))

  @parameterized.parameters([(True), (False)])
  def testVariablesAreTracked(self, reduce_retracing):
    # [FUNCTION DOC]
    # Parameterized over the reduce-retracing option. Calls a function multiplying a captured variable
    # by a vector, then changes the variable to two and calls with a longer vector. Asserts the function
    # sees the updated value (one, then two and four).
    v = resource_variable_ops.ResourceVariable(1.0)

    def foo(x):
      # [FUNCTION DOC]
      # Returns the captured variable multiplied by its argument.
      return v * x

    defined = polymorphic_function.function(
        foo, reduce_retracing=reduce_retracing)

    x = constant_op.constant([1.0])
    self.assertEqual(1., self.evaluate(defined(x)))
    v.assign(2.)

    x = constant_op.constant([1.0, 2.0])
    self.assertAllEqual([2., 4.], self.evaluate(defined(x)))

  def testInputSignatureMustBeSequenceOfTensorSpecs(self):
    # [FUNCTION DOC]
    # Passes a signature containing a plain tuple of numbers to the function decorator and asserts a
    # type error about the signature needing a nested sequence of tensor specs.

    def foo(a, b):
      # [FUNCTION DOC]
      # Deletes both arguments and returns nothing.
      del a
      del b

    # Signatures must consist exclusively of `TensorSpec` objects.
    signature = [(2, 3), tensor_lib.TensorSpec([2, 3], dtypes.float32)]
    with self.assertRaisesRegex(TypeError, 'input_signature.*nested sequence'):
      polymorphic_function.function(foo, input_signature=signature)

  @test_util.run_in_graph_and_eager_modes
  def testInputsIncompatibleWithSignatureRaisesError(self):
    # [FUNCTION DOC]
    # Runs in both modes. Wraps with a fixed signature of one float vector of length two. A valid call
    # succeeds; inputs with the wrong dtype or shape raise type errors saying they cannot be cast; too
    # many arguments or none raise type errors about positional arguments; requesting a concrete
    # function with a mismatching spec raises a type error about casting.

    def foo(a):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return a

    signature = [tensor_lib.TensorSpec(shape=(2,), dtype=dtypes.float32)]
    defined = polymorphic_function.function(foo, input_signature=signature)

    # Valid call
    defined(array_ops.ones([2]))

    # Invalid shapes.
    with self.assertRaisesRegex(
        TypeError, r'Can not cast .*dtype=tf.int32.* to .*dtype=tf.float32.*'
    ):
      defined(array_ops.ones([3], dtype=dtypes.int32))

    # Invalid shapes.
    with self.assertRaisesRegex(TypeError, 'Can not cast.*'):
      defined(array_ops.ones([3]))

    with self.assertRaisesRegex(TypeError, 'Can not cast.*'):
      defined(array_ops.ones([2, 1]))

    # Wrong number of arguments.
    with self.assertRaisesRegex(TypeError, 'too many positional arguments'):
      defined(array_ops.ones([2]), array_ops.ones([2]))
    with self.assertRaisesRegex(TypeError, 'missing a required argument'):
      defined()

    with self.assertRaisesRegex(
        TypeError, r'Can not cast .*shape=\(3,\).* to .*shape=\(2,\).*'
    ):
      defined.get_concrete_function(
          tensor_lib.TensorSpec(shape=(3,), dtype=dtypes.float32))

  def testMismatchedConcreteSignatureRaisesError(self):
    # [FUNCTION DOC]
    # Inside a function, gets a concrete function traced for the Python integer one and calls it with a
    # tensor, asserting a type error about binding inputs failing; calling a tensor-traced concrete
    # function with the integer one succeeds; calling the integer-traced one with the integer two
    # raises the binding error again.

    @polymorphic_function.function
    def run_test():
      # [FUNCTION DOC]
      # Outer function (called immediately) holding the checks described above.

      @polymorphic_function.function
      def f(x):
        # [FUNCTION DOC]
        # Returns its argument unchanged.
        return x

      with self.assertRaisesRegex(
          TypeError, 'Binding inputs to tf.function failed .*'):
        f.get_concrete_function(1)(constant_op.constant(1))

      f.get_concrete_function(constant_op.constant(1))(1)

      with self.assertRaisesRegex(
          TypeError, 'Binding inputs to tf.function failed .*'):
        f.get_concrete_function(1)(2)

    run_test()

  def testInputSignatureConversionWithDefaultArg(self):
    # [FUNCTION DOC]
    # Wraps a function with a defaulted flag argument under a signature of a float scalar and a bool
    # scalar. Asserts calling positionally, with the flag true, and with the flag false returns the
    # input, the input and its negation respectively.

    def foo(a, training=True):
      # [FUNCTION DOC]
      # Returns the input if training is true, otherwise returns the input multiplied by minus one.
      if training:
        return a
      else:
        return -1.0 * a

    signature = [
        tensor_lib.TensorSpec([], dtypes.float32),
        tensor_lib.TensorSpec([], dtypes.bool),
    ]
    defined = polymorphic_function.function(foo, input_signature=signature)
    a = constant_op.constant(1.0)
    self.assertAllEqual(a.numpy(), defined(a))
    self.assertAllEqual(a.numpy(), defined(a, training=True))
    self.assertAllEqual(-a.numpy(), defined(a, training=False))

  def testVariableSpecWithInputSignature(self):
    # [FUNCTION DOC]
    # Passes a signature containing a variable spec to the function decorator and asserts a type error
    # saying the signature does not support variable specs.

    def f(v):
      # [FUNCTION DOC]
      # Increments the variable argument by one; never actually traced because the decorator raises.
      v.assign_add(1)

    signature = [
        resource_variable_ops.VariableSpec(shape=[], dtype=dtypes.int32)
    ]
    with self.assertRaisesRegex(TypeError,
                                "input_signature doesn't support VariableSpec"):
      polymorphic_function.function(f, input_signature=signature)

  def testDefuningInstanceMethod(self):
    # [FUNCTION DOC]
    # Defines a class with a plain method and a function-decorated method with a default argument that
    # is a captured tensor. Calling the decorated method on an instance returns the input and the
    # default; asserts values one and two.

    integer = constant_op.constant(2, dtypes.int64)

    class Foo:
      # [CLASS DOC]
      # Local helper class with an ordinary method and a function-decorated method.

      def one(self, tensor):
        # [FUNCTION DOC]
        # Returns its argument unchanged.
        return tensor

      @polymorphic_function.function
      def two(self, tensor, other=integer):
        # [FUNCTION DOC]
        # Calls the first method with the tensor and returns a pair of that result and the other argument
        # (default: the captured int64 tensor).
        return self.one(tensor), other

    foo = Foo()
    t = constant_op.constant(1.0)
    one, two = foo.two(t)
    self.assertEqual(one.numpy(), 1.0)
    self.assertEqual(two.numpy(), 2)

  def testDefuningInstanceMethodWithDefaultArgument(self):
    # [FUNCTION DOC]
    # Decorated method with a defaulted tensor argument called without arguments; asserts it returns
    # the default value.

    integer = constant_op.constant(2, dtypes.int64)

    class Foo:
      # [CLASS DOC]
      # Local helper class with a function-decorated method with a default argument.

      @polymorphic_function.function
      def func(self, other=integer):
        # [FUNCTION DOC]
        # Returns the other argument (default: the captured int64 tensor).
        return other

    foo = Foo()
    self.assertEqual(foo.func().numpy(), int(integer))

  def testPythonCallWithSideEffects(self):
    # [FUNCTION DOC]
    # Asserts that the Python side effect (appending to a list) happens only during tracing: the first
    # call appends once, the second call (reusing the trace) does not, and calling the underlying
    # Python function directly appends again.
    state = []

    @polymorphic_function.function
    def side_effecting_function():
      # [FUNCTION DOC]
      # Appends zero to the outer list. Returns nothing.
      state.append(0)

    side_effecting_function()
    self.assertAllEqual(state, [0])

    # The second invocation should call the graph function, which shouldn't
    # trigger the list append.
    side_effecting_function()
    self.assertAllEqual(state, [0])

    # Whereas calling the python function directly should create a side-effect.
    side_effecting_function.python_function()
    self.assertAllEqual(state, [0, 0])

  def testFunctionWithNestedFunctionCallAndSideEffects(self):
    # [FUNCTION DOC]
    # Two variables are each incremented by an inner function called twice inside an outer function
    # returning their sum. Asserts the result is four, so all side effects ran even though the
    # optimizer may inline the inner calls.
    v1 = variables.Variable(1.0)
    v2 = variables.Variable(1.0)

    @polymorphic_function.function
    def add_one(a):
      # [FUNCTION DOC]
      # Increments the variable argument by one.
      a.assign_add(1.0)

    # Grappler will inline calls to `add_one` into the function body, we check
    # that all side-effects were executed.
    @polymorphic_function.function
    def side_effecting_function(a, b):
      # [FUNCTION DOC]
      # Calls the increment function on each argument and returns their sum.
      add_one(a)
      add_one(b)
      return a + b

    result = side_effecting_function(v1, v2)
    self.assertEqual(result.numpy(), 4.0)

  def testRegisterConcreteFunction(self):
    # [FUNCTION DOC]
    # Builds an add function and a composite function that calls the add concrete function. In a
    # graph, adds the composite to the graph with its gradient functions and asserts exactly six
    # functions are registered (inference, forward and backward for each) with matching name patterns;
    # executes both and asserts values, and that the count of functions remains six afterwards.

    @polymorphic_function.function
    def py_add(x, y):
      # [FUNCTION DOC]
      # Returns the sum of its two arguments.
      return math_ops.add(x, y)

    py_add(array_ops.ones([]), array_ops.ones([]))
    add = py_add.get_concrete_function(
        tensor_lib.TensorSpec(None, dtypes.float32),
        tensor_lib.TensorSpec(None, dtypes.float32))

    @polymorphic_function.function
    def py_composite(x, y):
      # [FUNCTION DOC]
      # Returns a pair: the first argument and the result of the registered add concrete function.
      return x, add(x, y)

    py_composite(array_ops.ones([]), array_ops.ones([]))
    composite = py_composite.get_concrete_function(
        tensor_lib.TensorSpec(None, dtypes.float32),
        tensor_lib.TensorSpec(None, dtypes.float32))

    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        composite.add_to_graph()
        composite.add_gradient_functions_to_graph()

        graph = ops.get_default_graph()
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 6)
        # two sets of functions, each of them are (inference, forward, backward)
        functions = list(graph._functions.values())
        captured_function_names = [
            f.cached_definition.signature.name for f in functions
        ]
        expected_func_name_regex = [
            '.*inference.*py_composite.*',
            '.*inference.*py_add.*',
            '.*forward.*py_composite.*',
            '.*forward.*py_add.*',
            '.*inference.*backward.*py_composite.*',
            '.*inference.*backward.*py_add.*',
        ]
        for expected, found in zip(expected_func_name_regex,
                                   captured_function_names):
          self.assertRegex(found, expected)

        composite_t, composite_double = composite(t, t)
        double = add(t, t)
        self.assertAllEqual([[2, 4], [6, 8]], self.evaluate(double))
        self.assertAllEqual([[2, 4], [6, 8]], self.evaluate(composite_double))
        self.assertAllEqual([[1, 2], [3, 4]], self.evaluate(composite_t))
        # Make sure the pre registered function is used, and no other function
        # is added.
        self.assertLen(graph._functions, 6)

  def testEagerCaptures(self):
    # [FUNCTION DOC]
    # Checks the eager capture rule: a large (256 element) tensor and a variable are captured as
    # placeholders while a small (4 element) tensor is captured as a constant, relative to the eager
    # constant size threshold. For each, traces a function and asserts it has one internal capture with
    # the expected op type.
    with context.eager_mode():
      large_tensor = array_ops.ones(shape=(256,))
      self.assertGreater(256, capture_container._EAGER_CONST_THRESHOLD)

      small_tensor = array_ops.ones(shape=(4,))
      self.assertLessEqual(4, capture_container._EAGER_CONST_THRESHOLD)

      v = resource_variable_ops.ResourceVariable(0.0)

    for captured, op_type in [(large_tensor, 'Placeholder'),
                              (small_tensor, 'Const'), (v, 'Placeholder')]:

      @polymorphic_function.function
      def test_fn():
        # [FUNCTION DOC]
        # Returns the captured loop value plus one.
        return captured + 1  # pylint: disable=cell-var-from-loop

      g = test_fn.get_concrete_function().graph
      internal_captures = g.internal_captures
      self.assertLen(internal_captures, 1)
      self.assertEqual(internal_captures[0].op.type, op_type)

  @parameterized.parameters([(True), (False)])
  def testVariableAliasIdInStructuredInputSignature(self, reduce_retracing):
    # [FUNCTION DOC]
    # Parameterized over reduce-retracing. Asserts that calling with the same variable twice yields
    # variable specs with alias id zero for both inputs, and with two different variables yields alias
    # ids zero and one.

    @polymorphic_function.function(reduce_retracing=reduce_retracing)
    def foo(v1, v2):
      # [FUNCTION DOC]
      # Returns the sum of its two variable arguments.
      return v1 + v2

    v1 = resource_variable_ops.ResourceVariable(1.0)
    v2 = resource_variable_ops.ResourceVariable(2.0)
    graph_function = foo.get_concrete_function(v1, v1)
    args_sig, _ = graph_function.graph.structured_input_signature
    expected_spec = resource_variable_ops.VariableSpec([], alias_id=0)
    self.assertLen(args_sig, 2)
    self.assertEqual(args_sig[0], expected_spec)
    self.assertEqual(args_sig[1], expected_spec)

    graph_function = foo.get_concrete_function(v1, v2)
    args_sig, _ = graph_function.graph.structured_input_signature
    expected_spec1 = resource_variable_ops.VariableSpec([], alias_id=0)
    expected_spec2 = resource_variable_ops.VariableSpec([], alias_id=1)
    self.assertLen(args_sig, 2)
    self.assertEqual(args_sig[0], expected_spec1)
    self.assertEqual(args_sig[1], expected_spec2)

  def testStructuredSignatureAndMultipleVariables(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test skips itself immediately (pending an upstream bug about variables becoming
    # composite tensors), so the remaining body never runs.
    # Intended scenario: a concrete function traced with the same variable twice should work for that
    # case and raise a type error when called with two different variables.
    self.skipTest('b/209081027: Enable this test after Variable becomes a '
                  'CompositeTensor and Variable gets expand to handle tensor.')

    @polymorphic_function.function
    def foo(v1, v2):
      # [FUNCTION DOC]
      # Returns the sum of its two variable arguments (never executed because the test skips).
      return v1 + v2

    v1 = resource_variable_ops.ResourceVariable(1.0)
    v2 = resource_variable_ops.ResourceVariable(2.0)
    graph_function = foo.get_concrete_function(v1, v1)
    self.assertAllEqual(graph_function(v1, v1), 2.0)
    with self.assertRaises(TypeError):
      graph_function(v1, v2)

  def _total_function_cache_def_func(self, defined):
    # [FUNCTION DOC]
    # HELPER: lists concrete functions for the variable-retracing tests.
    # Takes a polymorphic function object and returns all its cached concrete functions through the
    # private listing method. Duplicates the module-level helper of a similar name.
    return defined._list_all_concrete_functions()  # pylint: disable=protected-access

  @parameterized.parameters([(True), (False)])
  def testVariableRetracingOnDtypeChanges(self, reduce_retracing):
    # [FUNCTION DOC]
    # Parameterized over reduce-retracing. Calls with two float variables (one trace), then with two
    # integer variables, and asserts the cache grows to two concrete functions.

    @polymorphic_function.function(reduce_retracing=reduce_retracing)
    def defined(a, b):
      # [FUNCTION DOC]
      # Returns the sum of its two arguments.
      return a + b

    x1 = resource_variable_ops.ResourceVariable(0.0)
    x2 = resource_variable_ops.ResourceVariable(0.0)

    defined(x1, x2)
    self.assertLen(self._total_function_cache_def_func(defined), 1)

    # Should expect retracing for new dtypes
    y1 = resource_variable_ops.ResourceVariable(0)
    y2 = resource_variable_ops.ResourceVariable(1)
    defined(y1, y2)
    self.assertLen(self._total_function_cache_def_func(defined), 2)

  def testVariableRetracingDtypeShape(self):
    # [FUNCTION DOC]
    # Calls with two scalar float variables (one trace), two vector variables (two traces) and two
    # matrix variables (three traces) and asserts the cache size increases each time.

    @polymorphic_function.function
    def defined(a, b):
      # [FUNCTION DOC]
      # Returns the sum of its two arguments.
      return a + b

    x1 = resource_variable_ops.ResourceVariable(0.0)
    x2 = resource_variable_ops.ResourceVariable(0.0)

    defined(x1, x2)
    self.assertLen(self._total_function_cache_def_func(defined), 1)

    y1 = resource_variable_ops.ResourceVariable([0.0, 1.0])
    y2 = resource_variable_ops.ResourceVariable([0.0, 1.0])

    defined(y1, y2)
    self.assertLen(self._total_function_cache_def_func(defined), 2)

    z1 = resource_variable_ops.ResourceVariable([[0.0, 1.0]])
    z2 = resource_variable_ops.ResourceVariable([[0.0, 1.0]])
    defined(z1, z2)
    self.assertLen(self._total_function_cache_def_func(defined), 3)

  def testFunctionModifiesInputList(self):
    # [FUNCTION DOC]
    # Asserts that functions which modify a list argument in place (append, extend, insert, pop,
    # reverse, remove, clear, or append via a keyword argument) each raise a value error saying the
    # function should not modify its argument. Each function is defined and called inside an
    # assertion block.
    # Tests on `list` methods that do in place modification, except `list.sort`
    # since it cannot even be "defunned" in the first place

    def get_list():
      # [FUNCTION DOC]
      # Returns a fresh list of two constant tensors.
      return [constant_op.constant(0.), constant_op.constant(1.)]

    expected_msg = '.*() should not modify'

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def append(l):
        # [FUNCTION DOC]
        # Appends a constant to the list argument (a forbidden mutation).
        l.append(constant_op.constant(0.))

      append(get_list())

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def extend(l):
        # [FUNCTION DOC]
        # Extends the list argument with a one-element list.
        l.extend([constant_op.constant(0.)])

      extend(get_list())

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def insert(l):
        # [FUNCTION DOC]
        # Inserts a constant at the front of the list argument.
        l.insert(0, constant_op.constant(0.))

      insert(get_list())

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def pop(l):
        # [FUNCTION DOC]
        # Removes the last item of the list argument.
        l.pop()

      pop(get_list())

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def reverse(l):
        # [FUNCTION DOC]
        # Reverses the list argument in place.
        l.reverse()

      reverse(get_list())

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def remove(l):
        # [FUNCTION DOC]
        # Removes the first element of the list argument from it.
        l.remove(l[0])

      remove(get_list())

    # `list.clear` is a method that is in Py3 but not Py2
    if sys.version.startswith('3'):

      with self.assertRaisesRegex(ValueError, expected_msg):

        @polymorphic_function.function
        def clear(l):
          # [FUNCTION DOC]
          # Clears the list argument.
          l.clear()

        clear(get_list())

    # One last test for keyword arguments
    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def kwdappend(**kwargs):
        # [FUNCTION DOC]
        # Takes keyword arguments, fetches the list under key l and appends a constant to it.
        l = kwargs['l']
        l.append(constant_op.constant(0.))

      kwdappend(l=get_list())

  def testFunctionModifiesInputDict(self):
    # [FUNCTION DOC]
    # Asserts that functions mutating a dictionary argument (clear, pop, popitem, update, ...) raise a
    # value error saying the function should not modify its argument.

    def get_dict():
      # [FUNCTION DOC]
      # Returns a fresh dictionary mapping t1 and t2 to constant tensors.
      return {'t1': constant_op.constant(0.), 't2': constant_op.constant(1.)}

    expected_msg = '.* should not modify'

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def clear(m):
        # [FUNCTION DOC]
        # Clears the dictionary argument.
        m.clear()

      clear(get_dict())

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def pop(m):
        # [FUNCTION DOC]
        # Removes the entry for key t1 from the dictionary argument.
        m.pop('t1')

      pop(get_dict())

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def popitem(m):
        # [FUNCTION DOC]
        # Removes an arbitrary entry from the dictionary argument.
        m.popitem()

      popitem(get_dict())

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def update(m):
        # [FUNCTION DOC]
        # Adds or replaces the entry t1 in the dictionary argument.
        m.update({'t1': constant_op.constant(3.)})

      update(get_dict())

    with self.assertRaisesRegex(ValueError, expected_msg):

      @polymorphic_function.function
      def setdefault(m):
        # [FUNCTION DOC]
        # Sets a default entry t3 on the dictionary argument (a forbidden mutation when the key is absent).
        m.setdefault('t3', constant_op.constant(3.))

      setdefault(get_dict())

  def testFunctionModifiesInputNest(self):
    # [FUNCTION DOC]
    # Asserts that in-place modification of a nested structure argument is detected and raises a value
    # error naming the function: first appending to a list inside a dictionary inside a list, then a
    # case where the flat list of tensors stays the same but the structure changes (moving an element
    # between inner lists).
    with self.assertRaisesRegex(ValueError, 'modify.* should not modify'):

      @polymorphic_function.function
      def modify(n):
        # [FUNCTION DOC]
        # Appends a constant to the list stored under key t1 of the first element of the nested argument.
        n[0]['t1'].append(constant_op.constant(1.))

      nested_input = [{
          't1': [constant_op.constant(0.),
                 constant_op.constant(1.)],
      },
                      constant_op.constant(2.)]

      modify(nested_input)

    with self.assertRaisesRegex(ValueError,
                                'modify_same_flat.* should not modify'):

      # The flat list doesn't change whereas the true structure changes
      @polymorphic_function.function
      def modify_same_flat(n):
        # [FUNCTION DOC]
        # Moves the first element of the second inner list to the end of the first inner list, so the flat
        # tensor sequence is unchanged but the structure is not.
        n[0].append(n[1].pop(0))

      nested_input = [[constant_op.constant(0.)],
                      [constant_op.constant(1.),
                       constant_op.constant(2.)]]

      modify_same_flat(nested_input)

  def testFunctionStackInErrorMessage(self):
    # [FUNCTION DOC]
    # Chains three functions (the outermost calls the middle which asserts on the result of the
    # innermost) and asserts that the invalid-argument error raised when the assert fails mentions the
    # call chain between the outer two functions and the failing node's definition location, and does
    # not mention the innermost function.
    # KNOWN ISSUE: the test skips itself when executing eagerly (pending an upstream bug) and has no
    # decorator forcing graph mode, so under the default eager execution its body never runs.
    if context.executing_eagerly():
      # TODO(b/122736651): Remove this skipTest once fixed.
      self.skipTest('Error interpolation is not working when function is '
                    'invoked without PartitionedCallOp.')

    @polymorphic_function.function()
    def fn3(x):
      # [FUNCTION DOC]
      # Returns its argument plus two.
      return x + 2

    @polymorphic_function.function()
    def fn2(x):
      # [FUNCTION DOC]
      # Asserts that the innermost function's result equals three (raising at run time if not) and
      # returns the integer two.
      check_ops.assert_equal(fn3(x), 3)
      return 2

    @polymorphic_function.function()
    def fn(x):
      # [FUNCTION DOC]
      # Returns the result of the middle function applied to its argument.
      return fn2(x)

    with self.assertRaises(errors.InvalidArgumentError) as cm:
      fn(2)
    e = cm.exception
    self.assertIn('fn -> fn2', e.message)
    self.assertIn('node assert_equal/Assert/Assert (defined at', e.message)
    self.assertNotIn('fn3', e.message)

  @test_util.run_gpu_only
  def testFunctionIsNotPinned(self):
    # [FUNCTION DOC]
    # GPU-only. Verifies that the eager runtime does not pin a function to the CPU: a function that
    # generates a random normal tensor under a GPU device scope, called under a GPU scope, returns a
    # tensor whose device name mentions the GPU.
    """Tests that functions aren't pinned to the CPU by the eager runtime."""
    seed1, seed2 = 79, 25
    shape = constant_op.constant([4, 7])
    dtype = dtypes.float32

    @polymorphic_function.function
    def func():
      # [FUNCTION DOC]
      # Generates a random standard normal tensor of the captured shape and dtype with fixed seeds under
      # a GPU device scope and returns it.
      with ops.device('GPU:0'):
        return gen_random_ops.random_standard_normal(
            shape, dtype=dtype, seed=seed1, seed2=seed2)

    with ops.device('GPU:0'):
      x = func()
      self.assertRegex(x.device, 'GPU')

  def testLimitedRetracingWithCompositeTensors(self):
    # [FUNCTION DOC]
    # Calls a function ten times with three different kinds of ragged tensors (different ragged ranks
    # and structure, but element values that vary). Asserts the function is traced exactly three times
    # in total, showing that varying values or lengths does not cause extra traces.
    trace_count = [0]

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Increments the outer trace counter (so it counts traces) and returns its argument.
      trace_count[0] += 1
      return x

    for i in range(10):
      f(ragged_factory_ops.constant([[1, 2], [i]]))
      f(ragged_factory_ops.constant([[1, 2], [], [3, 4, 5]]))
      f(ragged_factory_ops.constant([[[1, 2], [3]], [[4, 5, 6]]]))
      self.assertEqual(trace_count[0], 3)

  def testCompositeTensorsWithReducedRetracing(self):
    # [FUNCTION DOC]
    # Calls a reduce-retracing identity function with a ragged tensor and asserts the output equals the
    # input elementwise.
    inp = ragged_factory_ops.constant([[1, 2], [3]])

    @polymorphic_function.function(reduce_retracing=True)
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    output = f(inp)
    self.assertTrue(math_ops.reduce_all(math_ops.equal(inp, output)))

  def testMultipleInputsWithReducedRetracing(self):
    # [FUNCTION DOC]
    # Calls a reduce-retracing function with two ragged tensors and two variables returning them as a
    # list; asserts each output equals the corresponding input.
    tensor1 = ragged_factory_ops.constant([[1, 2], [3]])
    tensor2 = ragged_factory_ops.constant([[[1, 2], [3]], [[4, 5, 6]]])
    variable1 = variables.Variable(1.0)
    variable2 = variables.Variable(2.0)

    @polymorphic_function.function(reduce_retracing=True)
    def f(a, b, c, d):
      # [FUNCTION DOC]
      # Returns a list of its four arguments.
      return [a, b, c, d]

    output = f(tensor1, tensor2, variable1, variable2)
    self.assertTrue(math_ops.reduce_all(math_ops.equal(tensor1, output[0])))
    self.assertTrue(math_ops.reduce_all(math_ops.equal(tensor2, output[1])))
    self.assertTrue(math_ops.reduce_all(math_ops.equal(variable1, output[2])))
    self.assertTrue(math_ops.reduce_all(math_ops.equal(variable2, output[3])))

  def test_concrete_function_shape_mismatch(self):
    # [FUNCTION DOC]
    # Gets a concrete function for a length-one vector. Calling it directly from eager mode with a
    # length-two vector works (no extra shape checking). Calling it from inside another function with a
    # length-two vector raises a type error saying the spec for shape two cannot be cast to shape one.

    @polymorphic_function.function
    def f(argument_name):
      # [FUNCTION DOC]
      # Returns its argument plus one.
      return argument_name + 1.

    f_concrete = f.get_concrete_function(constant_op.constant([1.]))

    # Calling a function from eager doesn't do any shape checking above what
    # kernels do while executing.
    self.assertAllEqual([2., 3.],
                        f_concrete(constant_op.constant([1., 2.])).numpy())

    @polymorphic_function.function
    def g():
      # [FUNCTION DOC]
      # Calls the captured concrete function with a length-two vector. Returns nothing.
      f_concrete(constant_op.constant([1., 2.]))

    with self.assertRaisesRegex(
        TypeError,
        r'Can not cast TensorSpec\(shape=\(2,\).* to TensorSpec\(shape=\(1,\)',
    ):
      g()

  @test_util.run_in_graph_and_eager_modes
  def test_shape_inference_with_symbolic_shapes(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Gets a concrete function from a function using dynamic shapes
    # (identity with a name that collides, transpose, shape extraction, scaling and a reshape using
    # computed batch sizes), then calls it twice within another function with differently shaped
    # inputs. Asserts the static shapes of both outputs inside the function and the evaluated shapes
    # outside.

    @polymorphic_function.function
    def _uses_symbolic_shapes(w, x, y):
      # [FUNCTION DOC]
      # Takes three tensors w, x, y. Passes x through an identity with a deliberately colliding name,
      # transposes its first two dimensions, computes batch sizes from the shapes of x and y, scales y by
      # w and reshapes it into integer-divided batch count, x batch size and the rest.
      x = array_ops.identity(x, name='name_collision')
      x = array_ops.transpose(x, [1, 0, 2])
      x_batch = array_ops.shape(x)[0]
      y_batch = array_ops.shape(y)[0]
      y *= w
      n = y_batch // x_batch
      return array_ops.reshape(y, [n, x_batch, -1])

    conc = _uses_symbolic_shapes.get_concrete_function(
        tensor_lib.TensorSpec(None, dtypes.float32),
        tensor_lib.TensorSpec(None, dtypes.float32),
        tensor_lib.TensorSpec(None, dtypes.float32))

    @polymorphic_function.function
    def _call_concrete():
      # [FUNCTION DOC]
      # Creates a constant and an identity with the same name as in the concrete function (to test name
      # collisions), calls the concrete function twice with different shapes, asserts the static output
      # shapes and returns both outputs.
      c = constant_op.constant(1.)
      array_ops.identity(c, name='name_collision')
      output1 = conc(
          array_ops.ones([2]), array_ops.ones([5, 4, 2]),
          array_ops.ones([20, 2]))
      self.assertEqual([5, 4, 2], output1.shape)
      output2 = conc(
          array_ops.ones([3]), array_ops.ones([5, 4, 3]),
          array_ops.ones([40, 3]))
      self.assertEqual([10, 4, 3], output2.shape)
      return output1, output2

    output1, output2 = _call_concrete()
    self.assertEqual((5, 4, 2), self.evaluate(output1).shape)
    self.assertEqual((10, 4, 3), self.evaluate(output2).shape)

  def testAutoGraphContext(self):
    # [FUNCTION DOC]
    # Asserts that, inside a function, the autograph control status is enabled, and that the status
    # outside the function afterwards is the same as before the call.

    @polymorphic_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Asserts the current autograph control status is enabled. Returns nothing.
      self.assertEqual(ag_ctx.control_status_ctx().status,
                       ag_ctx.Status.ENABLED)

    prev_status = ag_ctx.control_status_ctx().status
    test_fn()
    self.assertEqual(ag_ctx.control_status_ctx().status, prev_status)

  @test_util.disable_tfrt('b/170435618')
  def testCancelBeforeFunctionExecution(self):
    # [FUNCTION DOC]
    # Eager only (skipped otherwise; skipped under TFRT). Creates a queue and a function dequeuing from
    # it, wraps its concrete function as cancelable with a cancellation manager, cancels first and
    # then calls it; asserts a cancelled error is raised.
    if not context.executing_eagerly():
      self.skipTest('eager only')

    q = data_flow_ops.FIFOQueue(1, dtypes.int32)

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Dequeues and returns an element of the captured queue (blocking if empty).
      return q.dequeue()

    c_mgr = cancellation.CancellationManager()
    cancelable_func = c_mgr.get_cancelable_function(f.get_concrete_function())

    c_mgr.start_cancel()
    with self.assertRaises(errors.CancelledError):
      cancelable_func()

  @test_util.disable_tfrt('b/170435618')
  def testCancelBlockedFunctionExecution(self):
    # [FUNCTION DOC]
    # Eager only. The function blocks dequeuing from an empty queue; a checked background thread
    # cancels via the cancellation manager after half a second. Asserts the blocked call raises a
    # cancelled error and joins the thread.
    if not context.executing_eagerly():
      self.skipTest('eager only')

    q = data_flow_ops.FIFOQueue(1, dtypes.int32)

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Dequeues and returns an element of the captured queue (blocking if empty).
      return q.dequeue()

    c_mgr = cancellation.CancellationManager()
    cancelable_func = c_mgr.get_cancelable_function(f.get_concrete_function())

    def cancel_thread():
      # [FUNCTION DOC]
      # Thread body: sleeps half a second, then starts cancellation on the cancellation manager.
      time.sleep(0.5)
      c_mgr.start_cancel()

    t = self.checkedThread(cancel_thread)
    t.start()
    with self.assertRaises(errors.CancelledError):
      cancelable_func()
    t.join()

  @test_util.disable_tfrt('b/170435618')
  def testCancelAfterFunctionExecution(self):
    # [FUNCTION DOC]
    # Eager only. Enqueues the value thirty-seven first, calls the cancelable function and asserts it
    # returns thirty-seven; then starts cancellation, which is a no-op after completion.
    if not context.executing_eagerly():
      self.skipTest('eager only')

    q = data_flow_ops.FIFOQueue(1, dtypes.int32)
    q.enqueue(37)

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Dequeues and returns an element of the captured queue.
      return q.dequeue()

    c_mgr = cancellation.CancellationManager()
    cancelable_func = c_mgr.get_cancelable_function(f.get_concrete_function())

    self.assertAllEqual(37, cancelable_func().numpy())

    # Cancellation after the function executes is a no-op.
    c_mgr.start_cancel()

  @test_util.run_in_graph_and_eager_modes
  def testConcreteFunctionWithNestedTensorInputs(self):
    # [FUNCTION DOC]
    # Runs in both modes. Builds concrete functions from the same two-argument function (a dictionary
    # of tensors and a tuple of a tensor and an integer) using several argument conventions (positional,
    # keyword, specs). For each, calls it with structured positional, keyword, mixed and flat tensor
    # arguments and asserts the output is a two-tuple with values 1200 and 34.

    @polymorphic_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Returns a tuple: the sum of entries a and b of the first argument and the sum of the two items of
      # the second.
      return (x['a'] + x['b'], y[0] + y[1])

    a = constant_op.constant(1000)
    b = constant_op.constant(200)
    c = constant_op.constant(30)
    d = {'a': a, 'b': b}
    e = (c, 4)

    # Test different argument signatures when constructing the concrete func.
    for cf in [
        f.get_concrete_function(d, e),
        f.get_concrete_function(d, y=e),
        f.get_concrete_function(y=e, x=d),
        f.get_concrete_function(_spec_for_value(d), _spec_for_value(e)),
        f.get_concrete_function(_spec_for_value(d), y=_spec_for_value(e)),
        f.get_concrete_function(y=_spec_for_value(e), x=_spec_for_value(d))
    ]:
      # Test different calling conventions when calling the concrete func.
      for output in [
          cf(d, e),  # structured signature
          cf(d, y=e),  # structured signature w/ kwarg
          cf(y=e, x=d),  # structured signature w/ 2 kwargs
          cf(a, b, c),  # flat signature
      ]:
        self.assertIsInstance(output, tuple)
        self.assertLen(output, 2)
        self.assertAllEqual(output[0], 1200)
        self.assertAllEqual(output[1], 34)

  @test_util.run_in_graph_and_eager_modes
  def testConcreteFunctionWithNestedNonTensorInputs(self):
    # [FUNCTION DOC]
    # Runs in both modes. Binds a non-tensor tuple for the second argument when getting the concrete
    # function in three ways and calls with positional and keyword arguments; asserts the output values
    # add up to 1253.

    @polymorphic_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Returns a tuple: sum of entries a and b of the first argument and sum of the two items of the
      # second.
      return (x['a'] + x['b'], y[0] + y[1])

    a = {'a': constant_op.constant(1000), 'b': constant_op.constant(200)}
    b = (50, 3)

    for cf in [  # argument y is bound to non-Tensor value (50, 3).
        f.get_concrete_function(a, b),
        f.get_concrete_function(a, y=b),
        f.get_concrete_function(x=a, y=b)
    ]:
      for output in [cf(a, b), cf(x=a, y=b)]:
        self.assertAllEqual(output[0] + output[1], 1253)

  @test_util.run_in_graph_and_eager_modes
  def testConcreteFunctionWithNonTensorStringInputs(self):
    # [FUNCTION DOC]
    # Runs in both modes. Gets a concrete function with a string tensor and a Python string and calls it
    # with the tensor only, by keyword, and with both; asserts the joined output equals the string ab.

    @polymorphic_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Returns the string join of its two arguments.
      return string_ops.string_join([x, y])

    a = constant_op.constant('a')
    b = 'b'

    cf = f.get_concrete_function(a, b)
    for output in [cf(a), cf(x=a), cf(a, b), cf(x=a, y=b)]:
      self.assertAllEqual(output, b'ab')

  @test_util.run_in_graph_and_eager_modes
  def testConcreteFunctionWithBoundNestedNonTensorInputs(self):
    # [FUNCTION DOC]
    # Runs in both modes. Binds a dictionary of Python numbers as the first argument (with an extra
    # key that is ignored) and a tuple with a tensor as the second; asserts that every calling
    # convention gives outputs adding up to 3234.

    @polymorphic_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Returns a tuple: sum of entries a and b of the first argument and sum of the two items of the
      # second.
      return (x['a'] + x['b'], y[0] + y[1])

    a = {'a': 3000, 'b': 200, 'c': 9000}
    b = (constant_op.constant(30), 4)

    for cf in [  # argument x is bound to non-tensor value `a`
        f.get_concrete_function(a, b),
        f.get_concrete_function(a, y=b),
        f.get_concrete_function(x=a, y=b)
    ]:
      for output in [cf(a, b), cf(a, y=b), cf(x=a, y=b)]:
        self.assertAllEqual(output[0] + output[1], 3234)

  @test_util.run_in_graph_and_eager_modes
  def testConcreteFunctionWithAllBoundNestedNonTensorInputs(self):
    # [FUNCTION DOC]
    # Runs in both modes. Binds all arguments to Python values; the concrete function can then be
    # called with no arguments, positionally or by keyword; asserts the outputs add up to 5555.

    @polymorphic_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Returns a tuple: sum of entries a and b of the first argument and sum of the two items of the
      # second.
      return (x['a'] + x['b'], y[0] + y[1])

    a = {'a': 5000, 'b': 500}
    b = (50, 5)

    cf = f.get_concrete_function(a, b)
    for output in [cf(), cf(a, b), cf(x=a, y=b)]:
      self.assertAllEqual(output[0] + output[1], 5555)

  @test_util.run_in_graph_and_eager_modes
  def testConcreteFunctionMethodWithVarargs(self):
    # [FUNCTION DOC]
    # Runs in both modes. Defines a module subclass whose method has a variadic argument and a signature
    # of two float scalars, gets its concrete function and calls it with two numbers; passes if no error
    # is raised.
    float32_scalar = tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32)

    class MyModel(module.Module):
      # [CLASS DOC]
      # Local module subclass holding one function-decorated method with a variadic argument and an
      # explicit input signature.

      @polymorphic_function.function(
          input_signature=[float32_scalar, float32_scalar])
      def add(self, *arg):
        # [FUNCTION DOC]
        # Returns the sum of the variadic arguments using the add op.
        return math_ops.add(*arg)

    m = MyModel()
    cf = m.add.get_concrete_function()
    cf(-12.0, 3.0)

  @test_util.run_in_graph_and_eager_modes
  def testConcreteFunctionStructuredSignatureKeywordOrder(self):
    # [FUNCTION DOC]
    # Runs in both modes. A function taking only keyword arguments sorts them and joins them into a
    # string. A concrete function is built with seven keywords; calls with the keywords in different
    # orders must all return the same string of sorted name=value pairs, showing keyword arguments are
    # fed in the correct order.
    # Check that keyword-only arguments are sorted appropriately, so that they
    # feed the right tensor into each input.
    @polymorphic_function.function
    def g(**kwargs):
      # [FUNCTION DOC]
      # Takes arbitrary keyword arguments, sorts them by name, joins each pair with an equals sign and
      # joins the pairs with a comma and space; returns the string tensor.
      return string_ops.reduce_join(
          string_ops.reduce_join(
              ops.convert_to_tensor(sorted(kwargs.items())),
              axis=1,
              separator='='),
          axis=0,
          separator=', ')

    s = constant_op.constant('s')
    g.get_concrete_function(q=s, a=s, p=s, r=s, v=s, m=s, l=s)
    self.assertAllEqual(
        g(m='a', r='b', v='c', q='d', l='e', a='f', p='g'),
        b'a=f, l=e, m=a, p=g, q=d, r=b, v=c')
    self.assertAllEqual(
        g(q='d', a='f', p='g', r='b', v='c', m='a', l='e'),
        b'a=f, l=e, m=a, p=g, q=d, r=b, v=c')
    self.assertAllEqual(
        g(a='f', l='e', m='a', p='g', q='d', r='b', v='c'),
        b'a=f, l=e, m=a, p=g, q=d, r=b, v=c')

  def testSameConcreteFunctionDifferentKwargOrder(self):
    # [FUNCTION DOC]
    # Calls a keyword-only function with the same keywords in two different orders and asserts only one
    # concrete function was cached.
    @polymorphic_function.function
    def foo(**kwargs):
      # [FUNCTION DOC]
      # Returns keyword a plus keyword b cast to float.
      return kwargs['a'] + math_ops.cast(kwargs['b'], dtypes.float32)

    foo(a=constant_op.constant(1.0), b=constant_op.constant(1))
    foo(b=constant_op.constant(1), a=constant_op.constant(1.0))

    self.assertLen(total_function_cache(foo), 1)

  def testEmptyInputSignatures(self):
    # [FUNCTION DOC]
    # Defines methods with an empty input signature and zero, one or two defaulted arguments;
    # asserts each decorated method reports an empty input signature.

    class Foo:
      # [CLASS DOC]
      # Local helper class with three function-decorated methods, each with an empty input signature.

      @polymorphic_function.function(input_signature=[])
      def bar_none(self):
        # [FUNCTION DOC]
        # Returns the integer one.
        return 1

      @polymorphic_function.function(input_signature=[])
      def bar_one(self, x=0):
        # [FUNCTION DOC]
        # Returns its single defaulted argument.
        return x

      @polymorphic_function.function(input_signature=[])
      def bar_two(self, x=0, y=1):
        # [FUNCTION DOC]
        # Returns the sum of its two defaulted arguments.
        return x + y

    foo = Foo()
    self.assertEqual(foo.bar_none.input_signature, ())
    self.assertEqual(foo.bar_one.input_signature, ())
    self.assertEqual(foo.bar_two.input_signature, ())

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters([
      dict(
          testcase_name='MissingArg',
          conc_args=lambda: (1, constant_op.constant(2)),
          call_args=lambda: (1,),
          error=r'missing a required argument: \'y\'',
      ),
      dict(
          testcase_name='MissingVararg',
          conc_args=lambda: (1, 2, constant_op.constant(1.0)),
          call_args=lambda: (1, 2),
          error=r'missing a required argument: \'varargs_0\'',
      ),
      dict(
          testcase_name='ExtraPositionalArg',
          conc_args=lambda: (1, 2),
          call_args=lambda: (1, 2, 3),
          error=r'too many positional arguments',
      ),
      dict(
          testcase_name='MissingKeywordOnlyArg',
          conc_args=lambda: (1, 2),
          conc_kwargs=lambda: {'c': constant_op.constant(1.0)},
          call_args=lambda: (1, 2),
          error=r'missing a required( keyword-only)? argument: \'c\'',
      ),
      dict(
          testcase_name='ExtraKeywordArg',
          conc_args=lambda: (1, 2),
          call_args=lambda: (1, 2),
          call_kwargs=lambda: {'c': constant_op.constant(1.0)},
          error=r'got an unexpected keyword argument',
      ),
      dict(
          testcase_name='ExpectedRaggedGotNest',
          conc_args=lambda: (ragged_factory_ops.constant([[1, 2], [3]]),),
          call_args=lambda: ({'a': constant_op.constant([1, 2, 3])}, 5),
          error=(
              r'Binding inputs .* failed .* don\'t have the same nested'
              r' structure'
          ),
      ),
      dict(
          testcase_name='WrongRaggedRank',
          conc_args=lambda: (ragged_factory_ops.constant([[1, 2], [3]]),),
          call_args=lambda: (ragged_factory_ops.constant([[[1]]]), 5),
          error=(
              r'Binding inputs .* failed .* don\'t have the same nested'
              r' structure'
          ),
      ),
      dict(
          testcase_name='WrongRaggedDType',
          conc_args=lambda: (ragged_factory_ops.constant([[1]]),),
          call_args=lambda: (ragged_factory_ops.constant([[1.0]]), 5),
          error=(
              r'Binding inputs .* failed.*dtype=tf.float32.* to'
              r' .*dtype=tf.int32.*'
          ),
      ),
      dict(
          testcase_name='ExpectedDictGotTensor',
          conc_args=lambda: (
              {'a': constant_op.constant(1), 'b': constant_op.constant(1)},
          ),
          call_args=lambda: (constant_op.constant(1), 5),
          error=r'Binding inputs .* failed .*Can not cast .*Tensor.* to a Dict',
      ),
      dict(
          testcase_name='ExpectedTupleGotTensor',
          conc_args=lambda: (
              (constant_op.constant(1), constant_op.constant(2)),
          ),
          call_args=lambda: (constant_op.constant(1), 5),
          error=r'Binding inputs .* failed .*Can not cast .*Tensor.* to tuple',
      ),
      dict(
          testcase_name='WrongDType',
          conc_args=lambda: (constant_op.constant(1),),
          call_args=lambda: (constant_op.constant(1.0), 5),
          exception=(
              TypeError,
              errors.InvalidArgumentError,
              # on xla_gpu, we get InternalError instead.
              errors.InternalError,
          ),
      ),
      dict(
          testcase_name='ExpectedIntGotDifferentInt',
          conc_args=lambda: (5,),
          call_args=lambda: (8, 5),
          error=r'Binding inputs .* failed .*Can not cast 8 to .*5',
      ),
      dict(
          testcase_name='ExpectedIntGotTensor',
          conc_args=lambda: (5,),
          call_args=lambda: (constant_op.constant(6), 5),
          error=r'Binding inputs .* failed .*Can not cast .*Tensor.* to .*5',
      ),
      dict(
          testcase_name='TwoValuesForArgument',
          conc_args=lambda: (1, 2),
          call_args=lambda: (1, 2),
          call_kwargs=lambda: {'x': 3},
          error=r'got an unexpected keyword argument \'x\'',
      ),
  ])
  # pylint: enable=g-long-lambda
  @test_util.run_in_graph_and_eager_modes
  def testConcreteFunctionStructuredSignatureError(self,
                                                   conc_args=(),
                                                   conc_kwargs=None,
                                                   call_args=(),
                                                   call_kwargs=None,
                                                   error='.*',
                                                   exception=TypeError):
    # [FUNCTION DOC]
    # Parameterized by a table of scenarios (missing argument, extra positional or keyword argument,
    # wrong ragged structure, wrong dtype, int versus tensor and so on). Each case provides callables or
    # values for the arguments used to get the concrete function and to call it, an expected error
    # message pattern and an expected exception type (default type error). Asserts that calling the
    # concrete function with the structured signature raises that error. Runs in both modes.
    # Inputs: conc_args and conc_kwargs (arguments for concrete function creation), call_args and
    # call_kwargs (arguments for the call), error (message pattern) and exception (type).
    """Tests for errors in the structured signature.

    Args:
      conc_args: Positional arguments used for get_concrete_function.
      conc_kwargs: Keyword arguments used for get_concrete_function.
      call_args: Positional arguments used to call the function.
      call_kwargs: Keyword arguments used to call the function.
      error: Expected exception message.
      exception: Expected exception type.
    """
    conc_args = conc_args() if callable(conc_args) else conc_args
    conc_kwargs = conc_kwargs() if callable(conc_kwargs) else conc_kwargs or {}
    call_args = call_args() if callable(call_args) else call_args
    call_kwargs = call_kwargs() if callable(call_kwargs) else call_kwargs or {}
    self.assertIsInstance(conc_args, tuple)
    self.assertIsInstance(call_args, tuple)
    self.assertIsInstance(conc_kwargs, dict)
    self.assertIsInstance(call_kwargs, dict)

    @polymorphic_function.function
    def func(x, y=5, *varargs, **kwargs):  # pylint: disable=keyword-arg-before-vararg
      # [FUNCTION DOC]
      # Takes a required argument, a defaulted argument, variadic positional arguments and keywords;
      # discards all but the first and returns it.
      del y, varargs, kwargs
      return x

    conc = func.get_concrete_function(*conc_args, **conc_kwargs)
    with self.assertRaisesRegex(exception, error):
      self.evaluate(conc(*call_args, **call_kwargs))

  # pylint: disable=g-long-lambda
  @parameterized.named_parameters([
      dict(
          testcase_name='MissingArg',
          conc_args=lambda: (constant_op.constant(1), constant_op.constant(2)),
          call_args=lambda: (constant_op.constant(1),),
          error=r'func\(x, y\) missing required arguments: y'),
      dict(
          testcase_name='TwoValuesForArg',
          conc_args=lambda: (constant_op.constant(1), constant_op.constant(2)),
          call_args=lambda: (constant_op.constant(1),),
          call_kwargs=lambda: {
              'x': constant_op.constant(1),
              'y': constant_op.constant(1)
          },
          error=r"func\(x, y\) got two values for 'x'"),
      dict(
          testcase_name='ExtraPositionalArg',
          conc_args=lambda: (constant_op.constant(1), constant_op.constant(2)),
          call_args=lambda: (constant_op.constant(1), constant_op.constant(2),
                             constant_op.constant(3)),
          error=r'func\(x, y\) takes 2 .* got 3'),
      dict(
          testcase_name='UnexpectedKeywordArg',
          conc_args=lambda: (constant_op.constant(1),),
          call_args=lambda: (constant_op.constant(1),),
          call_kwargs=lambda: {'c': constant_op.constant(1)},
          error=r'func\(x\) got unexpected keyword arguments: c'),
      dict(
          testcase_name='MissingVararg',
          conc_args=lambda: (constant_op.constant(1), constant_op.constant(2),
                             constant_op.constant(3)),
          call_args=lambda: (constant_op.constant(1), constant_op.constant(2)),
          error=r'func\(x, y, varargs_0\) missing required '
          r'arguments: varargs_0'),
      dict(
          testcase_name='MissingKeywordArg',
          conc_args=lambda: (constant_op.constant(1), constant_op.constant(2)),
          conc_kwargs=lambda: {'c': constant_op.constant(1)},
          call_args=lambda: (constant_op.constant(1), constant_op.constant(2)),
          error=r'func\(x, y, c\) missing required arguments: c'),
      dict(
          testcase_name='ExpectedTensorGotInt',
          conc_args=lambda: (constant_op.constant(1), constant_op.constant(2)),
          call_args=lambda: (5, constant_op.constant(2)),
          error=r'func\(x, y\): expected argument #0\(zero-based\) to be '
          r'a Tensor; got int \(5\)'),
      dict(
          testcase_name='WrongDType',
          conc_args=lambda: (constant_op.constant(1),),
          call_args=lambda: (constant_op.constant(1.0),),
          exception=(
              ValueError,
              errors.InvalidArgumentError,
              # on xla_gpu, we get InternalError instead.
              errors.InternalError)),
      dict(
          testcase_name='MissingKeywordArgNestPiece',
          conc_args=lambda: (constant_op.constant(1), constant_op.constant(2)),
          conc_kwargs=lambda: {'c': ragged_factory_ops.constant([[1]])},
          call_args=lambda: (constant_op.constant(1), constant_op.constant(2)),
          call_kwargs=lambda: {'c': constant_op.constant(1)},
          error=r'func\(x, y, c, c_1\) missing required arguments: c_1'),
  ])
  # pylint: enable=g-long-lambda
  @test_util.run_in_graph_and_eager_modes
  def testConcreteFunctionFlatSignatureError(self,
                                             conc_args=(),
                                             conc_kwargs=None,
                                             call_args=(),
                                             call_kwargs=None,
                                             error='.*',
                                             exception=TypeError):
    # [FUNCTION DOC]
    # Same scenario table idea as the structured error test but calls the concrete function through its
    # private flat-signature entry. Expects errors reporting the flat signature (missing or duplicate
    # arguments, wrong counts, unexpected keyword, wrong type or dtype). Runs in both modes.
    # Inputs: conc_args and conc_kwargs (concrete function creation), call_args and call_kwargs (the
    # call), error (message pattern) and exception (type).
    """Tests for errors in the flat signature.

    Args:
      conc_args: Positional arguments used for get_concrete_function.
      conc_kwargs: Keyword arguments used for get_concrete_function.
      call_args: Positional arguments used to call the function.
      call_kwargs: Keyword arguments used to call the function.
      error: Expected exception message.
      exception: Expected exception type.
    """
    conc_args = conc_args() if callable(conc_args) else conc_args
    conc_kwargs = conc_kwargs() if callable(conc_kwargs) else conc_kwargs or {}
    call_args = call_args() if callable(call_args) else call_args
    call_kwargs = call_kwargs() if callable(call_kwargs) else call_kwargs or {}
    self.assertIsInstance(conc_args, tuple)
    self.assertIsInstance(call_args, tuple)
    self.assertIsInstance(conc_kwargs, dict)
    self.assertIsInstance(call_kwargs, dict)

    @polymorphic_function.function
    def func(x, y=5, *varargs, **kwargs):  # pylint: disable=keyword-arg-before-vararg
      # [FUNCTION DOC]
      # Takes a required argument, a defaulted argument, variadic positional arguments and keywords;
      # discards all but the first and returns it.
      del y, varargs, kwargs
      return x

    conc = func.get_concrete_function(*conc_args, **conc_kwargs)
    with self.assertRaisesRegex(exception, error):
      self.evaluate(conc._call_with_flat_signature(call_args, call_kwargs))  # pylint: disable=protected-access

  @test_util.run_in_graph_and_eager_modes
  def testConcreteFunctionAmbiguousSignature(self):
    # [FUNCTION DOC]
    # Runs in both modes. Builds a concrete function from specs whose names are swapped relative to the
    # argument names. Calls it by keyword; asserts the structured signature takes precedence over the
    # flat one, giving fifty-six.
    # When both the flat & structured signatures are applicable, but they
    # give different results, we use the structured signature.  Note: we expect
    # this to be extremely rare.
    @polymorphic_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Returns the first argument times ten plus the second.
      return x * 10 + y

    conc = f.get_concrete_function(
        x=tensor_lib.TensorSpec(None, dtypes.int32, name='y'),
        y=tensor_lib.TensorSpec(None, dtypes.int32, name='x'))

    result = conc(x=constant_op.constant(5), y=constant_op.constant(6))
    self.assertAllEqual(result, 56)

  def testPrettyPrintedSignature(self):
    # [FUNCTION DOC]
    # Gets a concrete function from a scalar and a vector for a function with a defaulted argument and
    # an unused argument. Asserts the brief and verbose pretty-printed signatures equal expected
    # strings, the repr matches a pattern and the string conversion equals the prefix plus the details.

    @polymorphic_function.function
    def func(x, kangaroo=None, octopus=7):
      # [FUNCTION DOC]
      # Deletes two arguments and returns the first (x).
      del octopus, kangaroo
      return x

    scalar = constant_op.constant(5)
    vector = constant_op.constant([10, 10, 20])

    concrete_fn = func.get_concrete_function(scalar, vector)
    summary = (
        '(x: TensorSpec(shape=(), dtype=tf.int32, name=None), kangaroo:'
        ' TensorSpec(shape=(3,), dtype=tf.int32, name=None), octopus:'
        ' Literal[7]) -> TensorSpec(shape=(), dtype=tf.int32, name=None)'
    )
    details = (
        'Input Parameters:\n'
        + '  x (POSITIONAL_OR_KEYWORD): TensorSpec(shape=(),'
        ' dtype=tf.int32, name=None)\n'
        + '  kangaroo (POSITIONAL_OR_KEYWORD):'
        ' TensorSpec(shape=(3,), dtype=tf.int32, name=None)\n'
        + '  octopus (POSITIONAL_OR_KEYWORD): Literal[7]\n'
        + 'Output Type:\n'
        + '  TensorSpec(shape=(), dtype=tf.int32, name=None)\n'
        + 'Captures:\n'
        + '  None'
    )
    self.assertEqual(
        concrete_fn.pretty_printed_signature(verbose=False), summary
    )
    self.assertEqual(
        concrete_fn.pretty_printed_signature(verbose=True), details
    )
    self.assertRegex(repr(concrete_fn), r'<ConcreteFunction .* at .*')
    self.assertEqual(str(concrete_fn), 'ConcreteFunction {}'.format(details))

  def testPrettyPrintedExplicitSignatureWithKeywordArg(self):
    # [FUNCTION DOC]
    # Builds a function with an explicit unknown-shape signature and a defaulted second argument and
    # asserts the brief and verbose pretty-printed signatures equal expected strings.

    @polymorphic_function.function(
        input_signature=[tensor_lib.TensorSpec(None)])
    def fn(a, b=1):
      # [FUNCTION DOC]
      # Returns the first argument plus the second (default one).
      return a + b

    concrete_fn = fn.get_concrete_function()
    self.assertEqual(
        concrete_fn.pretty_printed_signature(False),
        '(a: TensorSpec(shape=<unknown>, dtype=tf.float32, name=None), b:'
        ' Literal[1]) -> TensorSpec(shape=<unknown>, dtype=tf.float32,'
        ' name=None)',
    )
    self.assertEqual(
        concrete_fn.pretty_printed_signature(True),
        'Input Parameters:\n'
        + '  a (POSITIONAL_OR_KEYWORD):'
        ' TensorSpec(shape=<unknown>, dtype=tf.float32, name=None)\n'
        + '  b (POSITIONAL_OR_KEYWORD): Literal[1]\n'
        + 'Output Type:\n'
        + '  TensorSpec(shape=<unknown>, dtype=tf.float32, name=None)\n'
        + 'Captures:\n'
        + '  None',
    )

  def testPrettyPrintedSignatureLoadedNamedTuple(self):
    # [FUNCTION DOC]
    # Saves a module with a concrete function that took two named tuples as signature to a fixed
    # temporary path, loads it back and asserts the pretty-printed serving signature lists the
    # flattened keyword-only inputs (a, a_1, b, b_1) and the dictionary output.
    # KNOWN ISSUE: it writes to the fixed shared path in the temporary directory and never cleans it
    # up, so parallel or repeated runs can collide.
    Point = collections.namedtuple('Point', ['x', 'y'])

    @polymorphic_function.function
    def fn(b, a):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Ignores both arguments and returns the float one.
      return 1.

    b = Point(
        x=constant_op.constant(1., dtype=dtypes.float32),
        y=constant_op.constant(1., dtype=dtypes.float32))
    a = Point(
        x=constant_op.constant(1, dtype=dtypes.int32),
        y=constant_op.constant(1, dtype=dtypes.int32))

    mod = module.Module()
    f = fn.get_concrete_function(b, a)
    save(mod, '/tmp/f', signatures=f)
    loaded = load('/tmp/f')

    printed = loaded.signatures['serving_default'].pretty_printed_signature()
    self.assertEqual(
        printed,
        'Input Parameters:\n'
        + "  a (KEYWORD_ONLY): TensorSpec(shape=(), dtype=tf.int32, name='a')\n"
        + '  a_1 (KEYWORD_ONLY): TensorSpec(shape=(),'
        " dtype=tf.int32, name='a_1')\n"
        + '  b (KEYWORD_ONLY): TensorSpec(shape=(),'
        " dtype=tf.float32, name='b')\n"
        + '  b_1 (KEYWORD_ONLY):'
        " TensorSpec(shape=(), dtype=tf.float32, name='b_1')\n"
        + 'Output Type:\n'
        + "  Dict[['output_0', TensorSpec(shape=(), dtype=tf.float32,"
        " name='output_0')]]\n"
        + 'Captures:\n'
        + '  None',
    )

  @test_util.run_in_graph_and_eager_modes
  def testIndexedSlicesAsGradientsForConcreteFunctions(self):
    # [FUNCTION DOC]
    # Runs in both modes. A function computes the sum over an axis; another uses a gradient tape to
    # watch its input, gathers part of the result, averages it and returns the gradient. Calls it and
    # passes if no error is raised (gradients may be sparse indexed slices).

    @polymorphic_function.function
    def summing_rnn(inputs):
      # [FUNCTION DOC]
      # Returns the sum of the input along axis one.
      return math_ops.reduce_sum(inputs, axis=1)

    @polymorphic_function.function
    def gradients(inputs):
      # [FUNCTION DOC]
      # Watches the input on a gradient tape, computes a gathered, averaged loss through the other
      # function and returns the gradient of the loss with respect to the input.
      with backprop.GradientTape() as tape:
        tape.watch(inputs)
        hidden = summing_rnn(inputs)
        hidden = array_ops.gather(hidden, constant_op.constant([0]))
        loss = math_ops.reduce_mean(hidden)
      return tape.gradient(loss, inputs)

    gradients(constant_op.constant([[[1.0], [2.0]]]))  # No error is raised

  def testWithExtraWrapper(self):
    # [FUNCTION DOC]
    # Applies the function decorator on top of an extra decorator on a module method with a defaulted
    # argument. Calling with two and three must return six (the default one is applied).

    class Foo(module.Module):
      # [CLASS DOC]
      # Local module subclass with a variable attribute and a doubly decorated method.

      def __init__(self):
        # [FUNCTION DOC]
        # Calls the parent constructor and sets the variable attribute to nothing.
        super().__init__()
        self.var = None

      @polymorphic_function.function
      @dummy_tf_decorator
      def add(self, x, y, z=1):
        # [FUNCTION DOC]
        # Returns the sum of the three arguments when the variable attribute is nothing (otherwise returns
        # nothing).
        if self.var is None:
          return x + y + z

    foo = Foo()
    self.assertEqual(foo.add(2, 3).numpy(), 6)

  @parameterized.parameters([
      (polymorphic_function.function, dummy_tf_decorator),
      (dummy_tf_decorator, polymorphic_function.function),
      (polymorphic_function.function, polymorphic_function.function)
  ])
  def testWithExtraWrapperRedundantArgs(self, decorator1, decorator2):
    # [FUNCTION DOC]
    # Parameterized over three orderings of two decorators (function and extra wrapper). Calls the
    # decorated method with a positional argument and the same parameter by keyword and asserts a type
    # error mentioning multiple values for the argument.

    class Foo(module.Module):
      # [CLASS DOC]
      # Local module subclass with a variable attribute and a method decorated by the two parameterized
      # decorators.

      def __init__(self):
        # [FUNCTION DOC]
        # Calls the parent constructor and sets the variable attribute to nothing.
        super().__init__()
        self.var = None

      @decorator1
      @decorator2
      def add1(self, x, y):
        # [FUNCTION DOC]
        # Returns the sum of the two arguments when the variable attribute is nothing.
        if self.var is None:
          return x + y

    foo = Foo()
    with self.assertRaisesRegex(TypeError, 'multiple values for argument'):
      foo.add1(2, x=3)  # pylint: disable=redundant-keyword-arg,no-value-for-parameter

  def testWithExtraWrapperMissingArgs(self):
    # [FUNCTION DOC]
    # Asserts that, for three methods decorated with combinations of the function decorator and the
    # extra wrapper, omitting a required argument raises a type error naming the missing argument (y
    # when only x is given, x when only y is given).

    class Foo(module.Module):
      # [CLASS DOC]
      # Local module subclass with a variable attribute and three decorated methods.

      def __init__(self):
        # [FUNCTION DOC]
        # Calls the parent constructor and sets the variable attribute to nothing.
        super().__init__()
        self.var = None

      @polymorphic_function.function
      @dummy_tf_decorator
      def add1(self, x, y):
        # [FUNCTION DOC]
        # Method with function decorator over extra wrapper; returns the sum of its arguments.
        if self.var is None:
          return x + y

      @polymorphic_function.function
      @dummy_tf_decorator
      def add2(self, x, y):
        # [FUNCTION DOC]
        # Same as the first method: function decorator over extra wrapper; returns the sum of its
        # arguments.
        if self.var is None:
          return x + y

      @polymorphic_function.function
      @polymorphic_function.function
      def add3(self, x, y):
        # [FUNCTION DOC]
        # Method with the function decorator applied twice; returns the sum of its arguments.
        if self.var is None:
          return x + y

    foo = Foo()
    with self.assertRaisesRegex(TypeError,
                                'missing a required argument: \'y\''):
      foo.add1(2)  # pylint: disable=no-value-for-parameter

    with self.assertRaisesRegex(TypeError,
                                'missing a required argument: \'x\''):
      foo.add1(y=2)  # pylint: disable=no-value-for-parameter

    with self.assertRaisesRegex(TypeError,
                                'missing a required argument: \'y\''):
      foo.add2(2)  # pylint: disable=no-value-for-parameter

    with self.assertRaisesRegex(TypeError,
                                'missing a required argument: \'x\''):
      foo.add2(y=2)  # pylint: disable=no-value-for-parameter

    with self.assertRaisesRegex(TypeError,
                                'missing a required argument: \'y\''):
      foo.add3(2)  # pylint: disable=no-value-for-parameter

    with self.assertRaisesRegex(TypeError,
                                'missing a required argument: \'x\''):
      foo.add3(y=2)  # pylint: disable=no-value-for-parameter

  def testMissingArgsTfFunctionedMethod(self):
    # [FUNCTION DOC]
    # Wraps a bound ordinary method and a bound already-decorated method with the function decorator.
    # Calling with a missing required argument must raise a type error saying a required argument is
    # missing; calling with all arguments by keyword works.

    class A:
      # [CLASS DOC]
      # Local helper class with a plain method and a decorated method, both taking two positional
      # arguments and returning them as a pair.

      def func(self, position_arg1, position_arg2):
        # [FUNCTION DOC]
        # Returns the two arguments as a pair.
        return position_arg1, position_arg2

      @polymorphic_function.function
      def decorated_method(self, position_arg1, position_arg2):
        # [FUNCTION DOC]
        # Function-decorated method returning the two arguments as a pair.
        return position_arg1, position_arg2

    a_instance = A()
    tf_method_pos = polymorphic_function.function(a_instance.func)
    with self.assertRaisesRegex(TypeError, 'missing a required argument'):
      tf_method_pos(position_arg2='foo')

    # tf.function-decorated instance methods need to be tested because of
    # the __get__ method implementation.
    tf_func_decorated_method = polymorphic_function.function(
        a_instance.decorated_method)
    tf_func_decorated_method(position_arg1='foo', position_arg2='bar')
    with self.assertRaisesRegex(TypeError, 'missing a required argument'):
      tf_func_decorated_method(position_arg2='bar')

  def testMissingArgsTfFunctionedObject(self):
    # [FUNCTION DOC]
    # Wraps a callable object with the function decorator; calling with both keywords works and calling
    # with one missing raises a type error about a missing required argument.

    class A:
      # [CLASS DOC]
      # Local helper class whose call operator takes two arguments and returns them as a pair.

      def __call__(self, position_arg1, position_arg2):
        # [FUNCTION DOC]
        # Returns the two positional arguments as a pair.
        return position_arg1, position_arg2

    a_instance = A()

    # A tf.function-decorated callable object needs to be tested because of
    # the special inspect results.
    tf_func_obj = polymorphic_function.function(a_instance)
    tf_func_obj(position_arg1=1, position_arg2=2)
    with self.assertRaisesRegex(TypeError, 'missing a required argument'):
      tf_func_obj(position_arg2='bar')

  def testMissingArgsTfFunctionedFunctions(self):
    # [FUNCTION DOC]
    # Checks missing-argument errors for plain functions wrapped by the decorator (two
    # positional arguments, positional plus defaulted, three positional). In each case calling with a required argument omitted raises a type error about a missing required argument, while valid calls work.

    def func_pos(position_arg1, position_arg2):
      # [FUNCTION DOC]
      # Returns its two arguments as a pair.
      return position_arg1, position_arg2

    def func_with_default(position_arg, named_arg=None):
      # [FUNCTION DOC]
      # Returns the positional argument and the named defaulted argument as a pair.
      return position_arg, named_arg

    def func_pos_3args(position_arg1, position_arg2, position_arg3):
      # [FUNCTION DOC]
      # Returns its three positional arguments as a triple.
      return position_arg1, position_arg2, position_arg3

    tf_func_pos = polymorphic_function.function(func_pos)
    with self.assertRaisesRegex(
        TypeError, 'missing a required argument'):
      tf_func_pos(position_arg2='foo')

    tf_func_with_default = polymorphic_function.function(func_with_default)
    tf_func_with_default(position_arg='bar')
    with self.assertRaisesRegex(TypeError, 'missing a required argument'):
      tf_func_with_default(named_arg='foo')

    tf_func_pos_3args = polymorphic_function.function(func_pos_3args)
    with self.assertRaisesRegex(TypeError, 'missing a required argument'):
      tf_func_pos_3args(position_arg2='foo')

  def testShapeInferencePropagateConstNestedStack(self):
    # [FUNCTION DOC]
    # Calls an outer function with a fixed signature of a three-by-six integer matrix, which calls an
    # inner function (unknown-shape matrix and a scalar given as keyword five) that builds a shape from
    # the dynamic first dimension and the scalar via a stack op and returns ones of that shape. Asserts
    # the static shape is three-by-five inside the outer function and that the result equals ones.

    @polymorphic_function.function(input_signature=[
        tensor_lib.TensorSpec((None, None), dtype=dtypes.int32),
        tensor_lib.TensorSpec((), dtype=dtypes.int32),
    ])
    def f(x, s):
      # [FUNCTION DOC]
      # Takes a matrix and a scalar: stacks the matrix's first dimension and the scalar into a shape
      # vector and returns an int32 tensor of ones with that shape.
      old_shape = array_ops.shape(x)
      new_shape = array_ops_stack.stack([old_shape[0], s], axis=0)
      y = array_ops.ones(shape=new_shape, dtype=dtypes.int32)
      return y

    @polymorphic_function.function(input_signature=[
        tensor_lib.TensorSpec(shape=(3, 6), dtype=dtypes.int32)
    ])
    def g(x):
      # [FUNCTION DOC]
      # Calls the inner function with the matrix and scalar five, asserts the static result shape is
      # three by five and returns the result.
      y = f(x, s=5)
      assert y.shape.as_list() == [3, 5], y.shape.as_list()
      return y

    self.assertAllEqual(
        g(array_ops.zeros([3, 6], dtype=dtypes.int32)), array_ops.ones([3, 5]))

  def testShapeInferencePropagateConstNestedUnstackStack(self):
    # [FUNCTION DOC]
    # Like the previous test but the inner function derives the first dimension by unstacking the shape
    # vector before stacking it with the scalar. Asserts the static shape inside the outer function is
    # three by five and the result is ones.

    @polymorphic_function.function(input_signature=[
        tensor_lib.TensorSpec((None, None), dtype=dtypes.int32),
        tensor_lib.TensorSpec((), dtype=dtypes.int32),
    ])
    def f(x, s):
      # [FUNCTION DOC]
      # Unstacks the shape of the matrix, stacks its first element with the scalar and returns ones of
      # that shape.
      s0, _ = array_ops_stack.unstack(array_ops.shape(x), axis=0)
      new_shape = array_ops_stack.stack([s0, s], axis=0)
      y = array_ops.ones(shape=new_shape, dtype=dtypes.int32)
      return y

    @polymorphic_function.function(input_signature=[
        tensor_lib.TensorSpec(shape=(3, 6), dtype=dtypes.int32)
    ])
    def g(x):
      # [FUNCTION DOC]
      # Calls the inner function with scalar five, asserts the static shape is three by five and returns
      # the result.
      y = f(x, s=5)
      assert y.shape.as_list() == [3, 5], y.shape.as_list()
      return y

    self.assertAllEqual(
        g(array_ops.zeros([3, 6], dtype=dtypes.int32)), array_ops.ones([3, 5]))

  def testShapeInferencePropagateConstNestedConcat(self):
    # [FUNCTION DOC]
    # An inner function with three scalar integer signature entries builds a shape via concatenation
    # and returns ones of that shape; the outer function calls it with constants one, two, three and
    # asserts the static shape is one by two by three. Asserts the final result equals ones of that shape.

    @polymorphic_function.function(input_signature=[
        tensor_lib.TensorSpec((), dtype=dtypes.int32),
        tensor_lib.TensorSpec((), dtype=dtypes.int32),
        tensor_lib.TensorSpec((), dtype=dtypes.int32),
    ])
    def f(d1, d2, d3):
      # [FUNCTION DOC]
      # Concatenates the three scalar inputs into a shape vector and returns an int32 tensor of ones with
      # that shape.
      new_shape = array_ops.concat([[d1], [d2], [d3]], axis=-1)
      y = array_ops.ones(shape=new_shape, dtype=dtypes.int32)
      return y

    @polymorphic_function.function()
    def g():
      # [FUNCTION DOC]
      # Calls the inner function with one, two and three, asserts the static result shape is [1, 2, 3]
      # and returns the result.
      y = f(1, 2, 3)
      assert y.shape.as_list() == [1, 2, 3], y.shape.as_list()
      return y

    self.assertAllEqual(g(), array_ops.ones([1, 2, 3]))

  def testShapeInferencePropagateConstDoubleNested(self):
    # [FUNCTION DOC]
    # Same as the concat test but the outer function wraps the inner function with the function
    # decorator again before calling it, giving a doubly nested trace. Asserts static shape one by two by
    # three and a result of ones.

    @polymorphic_function.function(input_signature=[
        tensor_lib.TensorSpec((), dtype=dtypes.int32),
        tensor_lib.TensorSpec((), dtype=dtypes.int32),
        tensor_lib.TensorSpec((), dtype=dtypes.int32),
    ])
    def f(d1, d2, d3):
      # [FUNCTION DOC]
      # Concatenates the three scalar inputs into a shape vector and returns an int32 tensor of ones of
      # that shape.
      new_shape = array_ops.concat([[d1], [d2], [d3]], axis=-1)
      y = array_ops.ones(shape=new_shape, dtype=dtypes.int32)
      return y

    @polymorphic_function.function()
    def g():
      # [FUNCTION DOC]
      # Wraps the inner function again with the function decorator, calls it with one, two and three,
      # asserts the static shape is [1, 2, 3] and returns the result.
      y = polymorphic_function.function(f)(1, 2, 3)
      assert y.shape.as_list() == [1, 2, 3], y.shape.as_list()
      return y

    self.assertAllEqual(g(), array_ops.ones([1, 2, 3]))

  @test_util.run_v2_only
  def testControlDependencyAfterInline(self):
    # [FUNCTION DOC]
    # Runs only under TF2 behavior. One function assigns a variable to one and another adds one to it;
    # an outer function asserts the first returns one and the second returns two. The outer function is
    # run thirty times to gain confidence that control dependencies are kept after inlining (since the
    # inlined graph cannot be inspected).
    v = variables.Variable(0.)

    @polymorphic_function.function
    def assign():
      # [FUNCTION DOC]
      # Assigns one to the captured variable and returns the assign result.
      return v.assign(1.)

    @polymorphic_function.function
    def assign_add():
      # [FUNCTION DOC]
      # Adds one to the captured variable and returns the result.
      return v.assign_add(1.)

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Asserts the first function returns one and the second returns two; returns nothing.
      check_ops.assert_equal_v2(assign(), 1.)
      check_ops.assert_equal_v2(assign_add(), 2.)

    # We don't have a way to inspect the inlined graph in Python, so we run it
    # multiple times to have more confidence the dependency is correct.
    for _ in range(30):
      f()

  @test_util.run_v2_only
  def testReadInFuncWriteOutside(self):
    # [FUNCTION DOC]
    # Runs only under TF2 behavior. Thirty times: creates a variable of one, a function that reads it
    # and adds one, and an outer function that calls it and then increments the variable by two,
    # returning the earlier read. Asserts the result is two each time, testing for a potential race
    # between a read in a nested function and a later write.
    # Run many times since we are testing for a potential race condition.
    for _ in range(30):
      # pylint: disable=cell-var-from-loop
      v = variables.Variable(1.)

      @polymorphic_function.function
      def add_one():
        # [FUNCTION DOC]
        # Returns the captured variable plus one.
        return v + 1.

      @polymorphic_function.function
      def get_v_plus_one():
        # [FUNCTION DOC]
        # Calls the add-one function, then adds two to the captured variable, and returns the value
        # computed before the update.
        v_plus_one = add_one()
        v.assign_add(2.0)
        return v_plus_one

      self.assertAllEqual(get_v_plus_one(), 2.0)

  def testOpExpandErrorMessage(self):
    # [FUNCTION DOC]
    # Calls a function with a conditional that, at run time, goes to the branch that calls a Python
    # function returning a float where an integer is declared. Asserts an invalid-argument error whose
    # message mentions graph execution error and the function's name.

    @polymorphic_function.function
    def test_fn():
      # [FUNCTION DOC]
      # Returns constant one if a constant false is truthy, otherwise returns an eager Python function op
      # whose result has the wrong type for the declared output.
      if array_ops.constant(False):
        return array_ops.constant(1)
      else:
        return script_ops.eager_py_func(
            func=lambda: array_ops.constant([2.]), inp=(), Tout=dtypes.int32)

    error_pattern = re.compile(r'Graph execution error.*test_fn', re.DOTALL)
    with self.assertRaisesRegex(errors.InvalidArgumentError, error_pattern):
      test_fn()

  def testNoVariables(self):
    # [FUNCTION DOC]
    # Calls a doubling function with four and asserts eight.

    @polymorphic_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Returns its argument doubled.
      return 2 * x

    self.assertAllEqual(fn(constant_op.constant(4.0)), 8.0)

  def testFailIfVariablesAreCreatedMoreThanOnce(self):
    # [FUNCTION DOC]
    # Calls a function that creates a new variable every time it is traced and asserts a value error is
    # raised (variables may only be created on the first call).

    @polymorphic_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Creates a new variable and returns it plus the argument.
      return variables.Variable(1.0) + x

    with self.assertRaises(ValueError):
      fn(1.0)

  def testFailIfVariablesAreCreatedMoreThanOnceNoWeakRef(self):
    # [FUNCTION DOC]
    # Same as the previous test, but the variable created is held in an outer list so it is not
    # garbage collected; asserts a value error is raised.
    state = []

    @polymorphic_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Creates a variable, appends it to the outer list and returns it plus the argument.
      state.append(variables.Variable(1.0))
      return state[-1] + x

    with self.assertRaises(ValueError):
      fn(1.0)

  def testRange(self):
    # [FUNCTION DOC]
    # Calls a function with a Python range object as an argument and asserts the result is one.

    @polymorphic_function.function
    def f(unused_x):
      # [FUNCTION DOC]
      # Ignores its argument and returns one (float).
      return 1.0

    self.assertAllEqual(f(range(5)), 1.0)

  def testCorrectVariableCreation(self):
    # [FUNCTION DOC]
    # Calls a function which creates a variable the first time only; asserts results two and six for
    # inputs one and three.

    state = []

    @polymorphic_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Creates a variable of value two on first call (stored in outer list) and returns it times the
      # argument.
      if not state:
        state.append(variables.Variable(2.0))
      return state[0] * x

    self.assertAllEqual(fn(constant_op.constant(1.0)), 2.0)
    self.assertAllEqual(fn(constant_op.constant(3.0)), 6.0)

  def testFunctionInitializer(self):
    # [FUNCTION DOC]
    # Same as above with a variable created from an initializer callable; asserts result two for input
    # one.

    state = []

    @polymorphic_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Creates, on first call, a variable whose initial value comes from a callable returning two, then
      # returns the variable times the argument.
      if not state:
        state.append(variables.Variable(lambda: 2.0))
      return state[0] * x

    self.assertAllEqual(fn(constant_op.constant(1.0)), 2.0)

  def testFunctionMultipleVariableInitializer(self):
    # [FUNCTION DOC]
    # Creates two variables from callables (two and five) on the first call; asserts the function
    # returns two and five for input one.

    state = []

    @polymorphic_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Creates two variables with callable initializers on first call and returns each multiplied by the
      # argument.
      if not state:
        state.append(variables.Variable(lambda: 2.0))
        state.append(variables.Variable(lambda: 5.0))
      return state[0] * x, state[1] * x

    self.assertAllEqual(fn(constant_op.constant(1.0)), [2.0, 5.0])

  def testFunctionInitializationFunction(self):
    # [FUNCTION DOC]
    # Gets an initialization function for a function that creates a variable. Asserts that one variable
    # exists, that it is not yet initialized, and that after calling the initialization function its
    # value is two.

    state = []

    @polymorphic_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Creates a variable of value two on first call and returns it times the argument.
      if not state:
        state.append(variables.Variable(2.0))
      return state[0] * x

    init_fn = fn.get_initialization_function(constant_op.constant(1.0))
    self.assertLen(state, 1)
    self.assertFalse(
        resource_variable_ops.var_is_initialized_op(state[0].handle))
    init_fn()
    self.assertEqual(state[0].numpy(), 2.0)

  def testVariableInitializerNotConstant(self):
    # [FUNCTION DOC]
    # Calls a function whose variable's initial value depends on the (first) input value; asserts two
    # and six for inputs one and three, showing the initial value is computed once.

    state = []

    @polymorphic_function.function
    def fn(x):
      # [FUNCTION DOC]
      # Creates a variable initialized with two times the argument on first call, and returns it times
      # the argument.
      if not state:
        state.append(variables.Variable(2.0 * x))
      return state[0] * x

    self.assertAllEqual(fn(constant_op.constant(1.0)), 2.0)
    self.assertAllEqual(fn(constant_op.constant(3.0)), 6.0)

  def testLegacyGraphModeVariables(self):
    # [FUNCTION DOC]
    # In a legacy graph and session, calls a function that creates a variable; after initializing
    # variables asserts the variable is two and the function result for three is six.
    with ops.Graph().as_default(), self.test_session() as sess:
      state = []

      @polymorphic_function.function
      def fn(x):
        # [FUNCTION DOC]
        # Creates a variable of value two on first call and returns it times the argument.
        if not state:
          state.append(variables.Variable(2.0))
        return state[0] * x

      result = fn(3.0)

      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(sess.run(state[0]), 2.0)
      self.assertAllEqual(self.evaluate(result), 6.0)

  def testLegacyGraphModeVariablesNonTrivialInitializer(self):
    # [FUNCTION DOC]
    # Legacy graph mode: the variable's initial value is computed through several ops (square then
    # square root plus the square). After initialization asserts the variable is six and the result for
    # three is eighteen.
    with ops.Graph().as_default(), self.test_session() as sess:
      state = []

      @polymorphic_function.function
      def fn(x):
        # [FUNCTION DOC]
        # On first call builds a variable whose initial value is computed from constants (two, four, and
        # the square root of four), then returns it times the argument.
        if not state:
          two = constant_op.constant(2.0)
          four = two * two
          two_again = math_ops.sqrt(four)
          state.append(variables.Variable(two_again + four))
        return state[0] * x

      result = fn(3.0)

      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(sess.run(state[0]), 6.0)
      self.assertAllEqual(self.evaluate(result), 18.0)

  def testLegacyGraphModeInputDependentInitializerFails(self):
    # [FUNCTION DOC]
    # In a legacy graph, a function creates a variable initialized from its input. Calling it raises an
    # unliftable-error whose message mentions the transitive dependency on the input.
    with ops.Graph().as_default():
      state = []

      @polymorphic_function.function
      def fn(x):
        # [FUNCTION DOC]
        # Creates a variable initialized with two times the argument on first call and returns it times the
        # argument.
        if not state:
          state.append(variables.Variable(2.0 * x))
        return state[0] * x

      with self.assertRaisesRegex(lift_to_graph.UnliftableError,
                                  r'transitively.* mul .* x'):
        fn(constant_op.constant(3.0))

  def testMethod(self):
    # [FUNCTION DOC]
    # Defines a class with a function-decorated method that lazily creates a variable. Asserts the
    # method returns six for three, that after assigning three to the variable it returns nine (no
    # re-creation), and that a second instance gets its own variable (six).

    class MyModel:
      # [CLASS DOC]
      # Local helper class holding a variable attribute (initially nothing) and a decorated method that
      # creates it lazily.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets the variable attribute to nothing.
        self.var = None

      @polymorphic_function.function
      def apply(self, x):
        # [FUNCTION DOC]
        # Creates the variable (value two) if missing and returns it times the argument.
        if self.var is None:
          self.var = variables.Variable(2.0)
        return self.var * x

    m0 = MyModel()
    self.assertAllEqual(m0.apply(3.0), 6.0)
    # Calling twice to exercise that we do not recreate variables.
    m0.var.assign(3.0)
    self.assertAllEqual(m0.apply(3.0), 9.0)

    m1 = MyModel()
    self.assertAllEqual(m1.apply(3.0), 6.0)

  def testMethodExtensionType(self):
    # [FUNCTION DOC]
    # Defines an extension type with two decorated methods and checks them on an instance built from a
    # list of values and a mask. Asserts the default-filling method returns one, minus one, three and
    # the summing method (with a loop and condition converted by autograph) returns four.

    class MaskedTensorExtensionType(extension_type.ExtensionType):
      # [CLASS DOC]
      # Local extension type holding a values tensor and a mask tensor, with two function-decorated
      # methods.
      values: tensor_lib.Tensor
      mask: tensor_lib.Tensor

      @polymorphic_function.function
      def with_default(self, default_value):
        # [FUNCTION DOC]
        # Returns the values where the mask is true and the given default value elsewhere.
        return array_ops.where_v2(self.mask, self.values, default_value)

      @polymorphic_function.function
      def sum(self):
        # [FUNCTION DOC]
        # Loops over the values, adding those whose mask is true, and returns the total (exercises
        # autograph conversion of a loop and conditional).
        # Use a loop & conditional to test that autograph works correctly.
        result = 0
        for i in range(array_ops.size(self.values)):
          if self.mask[i]:
            result += self.values[i]
        return result

    mt = MaskedTensorExtensionType([1, 2, 3], [True, False, True])
    self.assertAllEqual(mt.with_default(-1), [1, -1, 3])
    self.assertAllEqual(mt.sum(), 4)

  def test_functools_partial(self):
    # [FUNCTION DOC]
    # Wraps a partial of an addition lambda with the first argument bound to one and calls it with two;
    # asserts the result is three.
    self.assertAllClose(
        3.,
        polymorphic_function.function(
            functools.partial(lambda x, y: x + y,
                              1.))(constant_op.constant(2.)))

  def test_functools_partial_new_default(self):
    # [FUNCTION DOC]
    # Wraps a partial of a function with two defaulted arguments where the second default is changed;
    # asserts calling with no arguments gives nine and overriding the second argument with eight gives
    # eleven.
    def f(x=3, y=7):
      # [FUNCTION DOC]
      # Returns the sum of its two arguments (defaults three and seven).
      return x + y

    func = polymorphic_function.function(functools.partial(f, y=6))
    self.assertEqual(func().numpy(), 9)
    self.assertEqual(func(y=8).numpy(), 11)

  def test_functools_partial_keywords(self):
    # [FUNCTION DOC]
    # Wraps a partial that binds both arguments by keyword to zero vectors; asserts calling with no
    # arguments yields a zero vector.
    def f(x, y):
      # [FUNCTION DOC]
      # Returns the sum of its two arguments.
      return x + y

    func = polymorphic_function.function(
        functools.partial(f, x=array_ops.zeros([1]), y=array_ops.zeros([1])))
    self.assertAllEqual(func(), [0.0])

  def test_functools_partial_single_positional(self):
    # [FUNCTION DOC]
    # Wraps a partial binding the first argument to a constant one; asserts calling with five returns
    # six.
    def f(x, y):
      # [FUNCTION DOC]
      # Returns the sum of its two arguments.
      return x + y

    func = polymorphic_function.function(
        functools.partial(f, constant_op.constant(1)))
    self.assertAllEqual(func(5), 6)

  def test_complicated_partial_with_defaults(self):
    # [FUNCTION DOC]
    # Wraps a partial of a function with several defaulted arguments (binding a first positional
    # function and overriding one default by keyword) and calls it with two tensors; the inner function
    # asserts the defaults and overridden value are as expected.

    def identity(*args):
      # [FUNCTION DOC]
      # Returns its positional arguments as a tuple.
      return args

    def dynamic_unroll(core_fn,
                       input_sequence,
                       initial_state,
                       sequence_length=None,
                       parallel_iterations=1,
                       swap_memory=False):
      # [FUNCTION DOC]
      # Takes a core function, an input sequence, an initial state and three defaulted settings; asserts
      # the sequence length default is nothing, parallel iterations is one and swap memory was overridden
      # to true; returns the sequence and state.
      del core_fn
      self.assertIs(None, sequence_length)
      self.assertEqual(1, parallel_iterations)
      self.assertTrue(swap_memory)
      return input_sequence, initial_state

    input_sequence = random_ops.random_uniform([1, 1, 1])
    initial_state = random_ops.random_uniform([1, 1])

    func = polymorphic_function.function(
        functools.partial(dynamic_unroll, identity, swap_memory=True))
    func(input_sequence, initial_state)

  def test_unspecified_default_argument(self):
    # [FUNCTION DOC]
    # Wraps a lambda with a defaulted second argument using a signature of one int32 scalar; asserts
    # calling with one gives three.
    wrapped = polymorphic_function.function(
        lambda x, y=2: x + y,
        input_signature=[tensor_lib.TensorSpec((), dtypes.int32)])
    self.assertEqual(3, wrapped(constant_op.constant(1)).numpy())

  def test_concrete_function_from_signature(self):
    # [FUNCTION DOC]
    # Gets concrete functions from a function with an input signature, once with no arguments and once
    # with the matching spec. Asserts the doubled results (one and four) and that the structured input
    # signature has the spec named x.

    @polymorphic_function.function(
        input_signature=[tensor_lib.TensorSpec(None, dtypes.float32)])
    def compute(x):
      # [FUNCTION DOC]
      # Returns its argument times two.
      return 2. * x

    concrete = compute.get_concrete_function()
    self.assertAllClose(1., concrete(constant_op.constant(0.5)))
    concrete = compute.get_concrete_function(
        tensor_lib.TensorSpec(None, dtypes.float32))
    self.assertAllClose(4., concrete(constant_op.constant(2.)))
    signature_args, _ = concrete.structured_input_signature
    self.assertEqual(signature_args,
                     (tensor_lib.TensorSpec(
                         None, dtypes.float32, name='x'),))

  def testInputSignatureMissingTensorSpecsMethod(self):
    # [FUNCTION DOC]
    # Defines a module with six methods (extra positional arguments, keywords, defaults, variadic,
    # ...) and wraps each with a signature containing one spec. For methods with more arguments than
    # specs and no defaults (the first five), calling raises a type error about the input signature
    # missing a type constraint; the sixth, whose remaining arguments are defaulted, works and returns
    # five.

    class MyModule(module.Module):
      # [CLASS DOC]
      # Local module subclass holding six methods with different parameter shapes (required, keyword,
      # default, variadic combinations) used as targets.

      def f1(self, arg1, arg2, arg3):
        # [FUNCTION DOC]
        # Takes three required arguments and does nothing.
        pass

      def f2(self, arg1, arg2, arg3, **kwargs):
        # [FUNCTION DOC]
        # Takes three required arguments plus arbitrary keywords and does nothing.
        pass

      def f3(self, arg1, arg2, arg3, arg4=4, **kwargs):
        # [FUNCTION DOC]
        # Takes three required arguments, a defaulted fourth and arbitrary keywords; does nothing.
        pass

      def f4(self, arg1, arg2, arg3, *args):
        # [FUNCTION DOC]
        # Takes three required arguments plus variadic positionals; does nothing.
        pass

      def f5(self, arg1, arg2, arg3, *args, **kwargs):
        # [FUNCTION DOC]
        # Takes three required arguments, variadic positionals and keywords; does nothing.
        pass

      def f6(self, arg1, arg4=4, **kwargs):
        # [FUNCTION DOC]
        # Takes one required argument, a defaulted argument and keywords; returns the sum of the first two.
        return arg1 + arg4

    m = MyModule()
    tf_func_dec = polymorphic_function.function(
        input_signature=(tensor_lib.TensorSpec([], dtypes.int32),))
    error_message = 'input_signature missing type constraint'
    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(m.f1)(1, 2, 3)

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(m.f2)(1, 2, 3)

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(m.f3)(1, 2, 3)

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(m.f4)(1, 2, 3)

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(m.f5)(1, 2, 3)

    self.assertEqual(tf_func_dec(m.f6)(1).numpy(), 5)

  def testInputSignatureMissingTensorSpecsFunction(self):
    # [FUNCTION DOC]
    # Same as the method variant but with plain local functions: the five functions with unspecified
    # required arguments raise a type error about a missing type constraint when called through the
    # decorator with a one-spec signature; the last (remaining arguments defaulted) returns five.
    # KNOWN ISSUE: the closing comment is misspelled so the intended re-enable of the unused-argument
    # lint check never takes effect.
    tf_func_dec = polymorphic_function.function(
        input_signature=(tensor_lib.TensorSpec([], dtypes.int32),))
    error_message = 'input_signature missing type constraint'
    # pylint: disable=unused-argument
    def f1(arg1, arg2, arg3):
      # [FUNCTION DOC]
      # Takes three required arguments and does nothing.
      pass

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(f1)(1, 2, 3)

    def f2(arg1, arg2, arg3, **kwargs):
      # [FUNCTION DOC]
      # Takes three required arguments plus arbitrary keywords and does nothing.
      pass

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(f2)(1, 2, 3)

    def f3(arg1, arg2, arg3, arg4=4, **kwargs):
      # [FUNCTION DOC]
      # Takes three required arguments, a defaulted fourth and keywords; does nothing.
      pass

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(f3)(1, 2, 3)

    def f4(arg1, arg2, arg3, *args):
      # [FUNCTION DOC]
      # Takes three required arguments plus variadic positionals; does nothing.
      pass

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(f4)(1, 2, 3)

    def f5(arg1, arg2, arg3, *args, **kwargs):
      # [FUNCTION DOC]
      # Takes three required arguments, variadic positionals and keywords; does nothing.
      pass

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(f5)(1, 2, 3)
    # pyline: enable=unused-argument

    def f6(arg1, arg4=4, **kwargs):
      # [FUNCTION DOC]
      # Takes one required argument, a defaulted argument and keywords; returns the sum of the first two.
      return arg1 + arg4
    self.assertEqual(tf_func_dec(f6)(1).numpy(), 5)

  def testInputSignatureMissingTensorSpecsLambdaFunction(self):
    # [FUNCTION DOC]
    # Same scenario using lambdas instead of defined functions: five lambdas whose arguments are not
    # all covered by the one-spec signature raise a type error about a missing type constraint; a
    # lambda with a single required argument and a default works and returns five.
    tf_func_dec = polymorphic_function.function(
        input_signature=(tensor_lib.TensorSpec([], dtypes.int32),))
    error_message = 'input_signature missing type constraint'
    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(lambda ar1, arg2, arg3: None)(1, 2, 3)

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(lambda arg1, arg2, arg3, **kwargs: None)(1, 2, 3)

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(lambda arg1, arg2, arg3, arg4=4, **kwargs: None)(1, 2, 3)

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(lambda arg1, arg2, arg3, *args: None)(1, 2, 3)

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(lambda arg1, arg2, arg3, *args, **kwargs: None)(1, 2, 3)

    self.assertEqual(
        tf_func_dec(lambda arg1, arg4=4, **kwargs: arg1 + arg4)(1).numpy(), 5)

  @parameterized.named_parameters(('_method', 'method'),
                                  ('_function', 'function'),
                                  ('_lambda_function', 'lambda_function'))
  def testInputSignaturePartialFuncMissingTensorSpecs(self, func_type):
    # [FUNCTION DOC]
    # Parameterized over a method, a function and a lambda with four arguments (the last defaulted).
    # Takes the kind of callable to test. Wraps partials of it (binding different numbers of
    # arguments) with a one-spec signature; asserts a type error about the missing type constraint when
    # not all non-defaulted arguments are covered, and that binding three arguments by position and the
    # last by keyword then calling with one value returns eleven.
    if func_type == 'method':
      class MyModule(module.Module):
        # [CLASS DOC]
        # Local module subclass with a four-argument method (the last defaulted) that returns the sum.

        def f(self, arg1, arg2, arg3, arg4=4):
          # [FUNCTION DOC]
          # Method returning the sum of its four arguments.
          return arg1 + arg2 + arg3 + arg4
      f = MyModule().f
    elif func_type == 'function':
      def f(arg1, arg2, arg3, arg4=4):
        # [FUNCTION DOC]
        # Function returning the sum of its four arguments.
        return arg1 + arg2 + arg3 + arg4
    else:  # lambda_function
      f = lambda arg1, arg2, arg3, arg4=4: arg1 + arg2 + arg3 + arg4

    error_message = 'input_signature missing type constraint'
    tf_func_dec = polymorphic_function.function(
        input_signature=(tensor_lib.TensorSpec([], dtypes.int32),)
    )
    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(functools.partial(f, 1))(2, 3)

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(functools.partial(f, arg4=5))(1, 2, 3)

    with self.assertRaisesRegex(TypeError, error_message):
      tf_func_dec(functools.partial(f, 1, arg4=5))(2, 3)

    self.assertAllEqual(
        tf_func_dec(functools.partial(f, 1, 2, arg4=5))(3),
        array_ops.constant(11),
    )

  @test_util.run_in_graph_and_eager_modes
  def test_variable_naming(self):
    # [FUNCTION DOC]
    # Runs in both modes. A module creates variables in three ways: in a function-decorated method, an
    # ordinary method, and an ordinary method inside a name scope. Asserts the first variable is named
    # v:0 and the third z_scope/z:0.
    class HasVars(module.Module):
      # [CLASS DOC]
      # Local module subclass holding three lazily created variable attributes, each created by a
      # different method.

      def __init__(self):
        # [FUNCTION DOC]
        # Sets all three variable attributes to nothing (without calling the parent constructor).
        self.x = None
        self.y = None
        self.z = None

      @polymorphic_function.function
      def make_x(self):
        # [FUNCTION DOC]
        # Function-decorated method that creates a variable named v if not yet created.
        if self.x is None:
          self.x = variables.Variable(1., name='v')

      def make_y(self):
        # [FUNCTION DOC]
        # Ordinary method that creates a variable named v if not yet created.
        if self.y is None:
          self.y = variables.Variable(1., name='v')

      def make_z(self):
        # [FUNCTION DOC]
        # Ordinary method that, inside a name scope z_scope (not skipped in eager mode), creates a variable
        # named z if not yet created.
        if self.z is None:
          with ops.name_scope('z_scope', skip_on_eager=False):
            self.z = variables.Variable(1., name='z')

    root = HasVars()
    root.make_x()
    root.make_y()
    root.make_z()
    self.assertEqual('v:0', root.x.name)
    self.assertEqual('z_scope/z:0', root.z.name)

  def test_concrete_function_keyword_arguments(self):
    # [FUNCTION DOC]
    # Checks which names are used for concrete-function keyword calls: a spec named y makes the
    # function callable as y and the structured signature reports that name; asking again with an
    # unnamed spec still reports y; a new name z updates it; for a nested list input the name of the
    # inner spec is used.
    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    conc = f.get_concrete_function(
        tensor_lib.TensorSpec(None, dtypes.float32, 'y'))
    conc(y=constant_op.constant(3.0))
    signature_args, _ = conc.structured_input_signature
    self.assertEqual('y', signature_args[0].name)

    # If name is not specified, the previously named one will be returned.
    conc = f.get_concrete_function(tensor_lib.TensorSpec(None, dtypes.float32))
    conc(x=constant_op.constant(3.0))
    signature_args, _ = conc.structured_input_signature
    self.assertEqual('y', signature_args[0].name)

    # New name will return updated signature.
    conc = f.get_concrete_function(
        tensor_lib.TensorSpec(None, dtypes.float32, 'z')
    )
    conc(x=constant_op.constant(3.0))
    signature_args, _ = conc.structured_input_signature
    self.assertEqual('z', signature_args[0].name)

    @polymorphic_function.function
    def g(x):
      # [FUNCTION DOC]
      # Returns the first element of its argument.
      return x[0]

    conc = g.get_concrete_function(
        [tensor_lib.TensorSpec(None, dtypes.float32, 'z'), 2])
    conc(z=constant_op.constant(3.0))
    signature_args, _ = conc.structured_input_signature
    self.assertEqual('z', signature_args[0][0].name)

  def testRuntimeErrorNotSticky(self):
    # [FUNCTION DOC]
    # Calls a function with an assert on zero: passes with zero, raises an invalid-argument error with
    # one, and then passes again with zero (a runtime error does not leave the function in a bad state).

    @polymorphic_function.function
    def fail(i):
      # [FUNCTION DOC]
      # Asserts the argument equals zero with the message ick. Returns nothing.
      control_flow_assert.Assert(math_ops.equal(i, 0), ['ick'])

    fail(constant_op.constant(0))  # OK
    with self.assertRaises(errors.InvalidArgumentError):
      fail(constant_op.constant(1))  # InvalidArgument: "ick"
    fail(constant_op.constant(0))  # OK

  def testUnderscoreName(self):
    # [FUNCTION DOC]
    # Calls a function whose only parameter is named with a single underscore; asserts the result is
    # two for input one.

    @polymorphic_function.function
    def f(_):
      # [FUNCTION DOC]
      # Returns the argument added to itself.
      return _ + _

    self.assertAllEqual(2.0, f(constant_op.constant(1.0)))

  def test_serialization_signature_cache(self):
    # [FUNCTION DOC]
    # Calls a function with two differently shaped and typed argument pairs, then asserts the
    # serialization listing of concrete functions has structured signatures matching both (named specs
    # x and y), with no keyword arguments.

    @polymorphic_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Returns both arguments as a pair.
      return x, y

    f(constant_op.constant([[3., 4.]]), constant_op.constant([2.]))
    f(constant_op.constant([[3, 4, 5]]), constant_op.constant([2]))

    signatures_args = set()
    concrete_functions = f._list_all_concrete_functions_for_serialization()
    for concrete_function in concrete_functions:
      args, kwargs = concrete_function.structured_input_signature
      signatures_args.add(args)
      self.assertEqual(dict(), kwargs)

    self.assertEqual(
        signatures_args,
        set(((tensor_lib.TensorSpec([1, 2], dtypes.float32, name='x'),
              tensor_lib.TensorSpec([1], dtypes.float32, name='y')),
             (tensor_lib.TensorSpec([1, 3], dtypes.int32, name='x'),
              tensor_lib.TensorSpec([1], dtypes.int32, name='y')))))

  @test_util.assert_no_garbage_created
  def testFunctionReferenceCycles(self):
    # [FUNCTION DOC]
    # Checked for no garbage creation. Calls a function, deletes it and asserts its weak reference has
    # died, showing the function is not part of a reference cycle.
    fn = polymorphic_function.function(lambda x: 2. * x)
    fn(constant_op.constant(4.0))
    weak_fn = weakref.ref(fn)
    del fn
    # Tests that the weak reference we made to the function is now dead, which
    # means the object has been deleted. This should be true as long as the
    # function itself is not involved in a reference cycle.
    self.assertIs(None, weak_fn())

  @test_util.assert_no_garbage_created
  def testMethodReferenceCycles(self):
    # [FUNCTION DOC]
    # Checked for no garbage creation. Calls a decorated method on an instance, deletes the instance
    # and asserts the weak reference to the bound function is dead.
    has_decorated_method = _HasDecoratedMethod()
    has_decorated_method.f(constant_op.constant(5.))
    weak_fn = weakref.ref(has_decorated_method.f)
    del has_decorated_method
    # Tests that the weak reference we made to the function is now dead, which
    # means the object has been deleted. This should be true as long as the
    # function itself is not involved in a reference cycle.
    self.assertIs(None, weak_fn())

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testErrorMessageWhenGraphTensorIsPassedToEager(self):
    # [FUNCTION DOC]
    # Checked for no new Python objects in eager mode. A function creates a tensor and uses it in an
    # init scope (escaping the function graph); asserts a type error whose message mentions this test
    # module and being out of scope.

    @polymorphic_function.function
    def failing_function():
      # [FUNCTION DOC]
      # Creates a constant, then adds it to itself inside an init scope (which runs eagerly where the
      # graph tensor is invalid). Returns nothing.
      a = constant_op.constant(1.)

      with ops.init_scope():
        _ = a + a

    with self.assertRaisesRegex(
        TypeError,
        re.compile('polymorphic_function_test.*out of scope', re.DOTALL)):
      failing_function()

  def testSymbolicTensorIllegalCaptureCallTimeError(self):
    # [FUNCTION DOC]
    # One function stores its argument (a graph tensor) in an outer variable; a second function adds
    # the stored tensor to its input. Calling the first then the second raises a type error mentioning
    # this module and being out of scope.
    x = None

    @polymorphic_function.function
    def f1(a):
      # [FUNCTION DOC]
      # Stores its argument in an outer variable and returns it.
      nonlocal x
      x = a
      return a

    @polymorphic_function.function
    def f2(b):
      # [FUNCTION DOC]
      # Returns its argument plus the tensor stored by the other function (illegal capture).
      return b + x

    f1(constant_op.constant(1))
    with self.assertRaisesRegex(
        TypeError,
        re.compile('polymorphic_function_test.*out of scope', re.DOTALL)):
      f2(constant_op.constant(2))

  def testSymbolicTensorIllegalCaptureTraceTimeError(self):
    # [FUNCTION DOC]
    # Calls a function that iterates over a symbolic range using Python-level indexing of an input
    # whose shape is partly unknown; asserts an inaccessible-tensor error about being out of scope.

    @polymorphic_function.function
    def f(inputs):
      # [FUNCTION DOC]
      # Takes the leading two dimensions of the input shape, loops over a symbolic range of the first and
      # collects each indexed slice into a list returned.
      num_steps, _ = inputs.shape[:2]
      outputs = []
      for t in math_ops.range(num_steps):
        outputs.append(inputs[t])
      return outputs

    with self.assertRaisesRegex(errors.InaccessibleTensorError, 'out of scope'):
      f(array_ops.zeros(shape=(8, 42, 3)))

  def testNonUniqueNamesGetConcreteFunction(self):
    # [FUNCTION DOC]
    # Gets a concrete function with a three-tensor tuple named x and a keyword d; asserts the flattened
    # inputs are named x, x_1, x_2 and d and that calling by those names or positionally gives ten.
    @polymorphic_function.function
    def non_unique_arg_names(x, **kwargs):
      # [FUNCTION DOC]
      # Takes a tuple of three tensors and keywords; returns the three plus the keyword d.
      a, b, c = x
      d = kwargs['d']
      return a + b + c + d

    concrete = non_unique_arg_names.get_concrete_function(
        (tensor_lib.TensorSpec(None, dtypes.float32),
         tensor_lib.TensorSpec(None, dtypes.float32),
         tensor_lib.TensorSpec(None, dtypes.float32)),
        d=tensor_lib.TensorSpec(None, dtypes.float32))
    self.assertAllClose(
        10.,
        concrete(x=constant_op.constant(1.),
                 x_1=constant_op.constant(2.),
                 x_2=constant_op.constant(3.),
                 d=constant_op.constant(4.)))
    self.assertAllClose(
        10.,
        concrete(constant_op.constant(1.),
                 constant_op.constant(2.),
                 constant_op.constant(3.),
                 constant_op.constant(4.)))

  def testDuplicatedSanitizedNames(self):
    # [FUNCTION DOC]
    # Calls with two keyword arguments whose names collide once sanitized and asserts a value error
    # about a name collision after sanitization.
    @polymorphic_function.function
    def foo(**kwargs):
      # [FUNCTION DOC]
      # Returns the sum of the keyword arguments a_b and a/b.
      return kwargs['a_b'] + kwargs['a/b']

    error_message = 'Name collision after sanitization.'
    with self.assertRaisesRegex(ValueError, error_message):
      foo(**{'a_b': 1, 'a/b': 2})

  def testVariableCreatorScope(self):
    # [FUNCTION DOC]
    # Within a variable-creator scope, calls a function that creates a variable and asserts the
    # creator saw exactly the variables the function created.
    created_variables = []
    captured_variables = []

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Creates a variable on first call (stored in outer list) and returns it plus one.
      if not created_variables:
        created_variables.append(variables.Variable(1.))
      return created_variables[0] + 1.

    def capture_creator(next_creator, **kwargs):
      # [FUNCTION DOC]
      # Variable creator hook: delegates to the next creator with the same keywords, records the created
      # variable and returns it.
      created = next_creator(**kwargs)
      captured_variables.append(created)
      return created

    with variable_scope.variable_creator_scope(capture_creator):
      f()
    self.assertEqual(created_variables, captured_variables)

  def testVarAlreadyInitializedNoClobbering(self):
    # [FUNCTION DOC]
    # A function creates a variable and a second variable that is assigned ten in an init scope (so it
    # is initialized outside the function). After obtaining the concrete function and calling it with
    # two, asserts the results are thirteen and fourteen, showing the pre-initialized value is not
    # overwritten by its initializer.
    v_holder = []

    @polymorphic_function.function
    def add_var(x):
      # [FUNCTION DOC]
      # Creates the two variables on first call (the second assigned ten in an init scope) and returns
      # their sum plus the argument.
      if not v_holder:
        v = variables.Variable([1., 2.])
        v_holder.append(v)
        already_initialized = variables.Variable(3.)
        with ops.init_scope():
          already_initialized.assign(10.)
        v_holder.append(already_initialized)
      return v_holder[0] + v_holder[1] + x

    add_var.get_concrete_function(constant_op.constant(2.))
    self.assertAllClose([13., 14.], add_var(constant_op.constant(2.)))

  def testSameVariableTwice(self):
    # [FUNCTION DOC]
    # Calls an addition function with the same variable for both arguments; asserts the result is two.
    v = variables.Variable(1.0)

    @polymorphic_function.function
    def add(a, b):
      # [FUNCTION DOC]
      # Returns the sum of its two arguments.
      return a + b

    self.assertAllEqual(add(v, v), 2.0)

  def testSameVariableTwiceWithReducedRetracing(self):
    # [FUNCTION DOC]
    # Same as above with reduce retracing enabled and a variable of two; asserts four.
    v = variables.Variable(2.0)

    @polymorphic_function.function(reduce_retracing=True)
    def add(a, b):
      # [FUNCTION DOC]
      # Returns the sum of its two arguments.
      return a + b

    self.assertAllEqual(add(v, v), 4.0)

  def testVariableUpdate(self):
    # [FUNCTION DOC]
    # Counts traces of a function doubling a variable in place. Calling with a float variable traces
    # once; another float variable of the same type and shape does not retrace; an integer variable
    # forces a retrace. Asserts the variable values are doubled each time (two, four, eight).
    v1 = variables.Variable(1.0)
    v2 = variables.Variable(2.0)
    v3 = variables.Variable(4, dtype=dtypes.int32)

    trace_count = [0]

    @polymorphic_function.function
    def double_variable(x):
      # [FUNCTION DOC]
      # Increments the trace counter and adds the variable's own value to itself.
      trace_count[0] += 1
      x.assign_add(x.read_value())

    self.assertEqual(trace_count[0], 0)
    double_variable(v1)
    self.assertEqual(trace_count[0], 1)
    self.assertEqual(self.evaluate(v1), 2.0)
    double_variable(v2)
    # No retracing because v2's data type and shape are the same as v1
    self.assertEqual(trace_count[0], 1)
    self.assertEqual(self.evaluate(v2), 4.0)
    double_variable(v3)
    # Retracing because of data type change
    self.assertEqual(trace_count[0], 2)
    self.assertEqual(self.evaluate(v3), 8)

  def testShapeCache(self):
    # [FUNCTION DOC]
    # Gets concrete functions twice with the same unknown-length integer vector spec and asserts they
    # are the same object.
    @polymorphic_function.function
    def func(x):
      # [FUNCTION DOC]
      # Returns its argument doubled.
      return 2 * x

    func_a = func.get_concrete_function(
        tensor_lib.TensorSpec([None], dtypes.int32))
    func_b = func.get_concrete_function(
        tensor_lib.TensorSpec([None], dtypes.int32))

    self.assertIs(func_a, func_b)

  def testCacheWithinSaveContext(self):
    # [FUNCTION DOC]
    # Asserts concrete functions for the same input are identical normally, but within save contexts
    # with different variable policies (expand distributed versus none) a different concrete function
    # is returned for each.

    @polymorphic_function.function
    def func(x):
      # [FUNCTION DOC]
      # Returns its argument doubled.
      return 2 * x

    func_a = func.get_concrete_function(constant_op.constant(2.))
    func_b = func.get_concrete_function(constant_op.constant(2.))

    self.assertIs(func_a, func_b)

    with save_context.save_context(
        save_options.SaveOptions(experimental_variable_policy=save_options
                                 .VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES)):
      func_c = func.get_concrete_function(constant_op.constant(2.))

    with save_context.save_context(
        save_options.SaveOptions(
            experimental_variable_policy=save_options.VariablePolicy.NONE)):
      func_d = func.get_concrete_function(constant_op.constant(2.))

    self.assertIsNot(func_a, func_c)
    self.assertIsNot(func_a, func_d)

  def testInitializationInNestedCall(self):
    # [FUNCTION DOC]
    # A function creates a pre-initialized variable (assigned ten in an init scope) and another;
    # a wrapper function calls it. Asserts results thirteen and fourteen, then after setting the second
    # variable to eleven fourteen and fifteen.
    v_holder = []

    @polymorphic_function.function
    def add_var(x):
      # [FUNCTION DOC]
      # Creates the two variables on first call (the second assigned ten in an init scope) and returns
      # their sum plus the argument.
      if not v_holder:
        v = variables.Variable([1., 2.])
        v_holder.append(v)
        already_initialized = variables.Variable(3.)
        with ops.init_scope():
          already_initialized.assign(10.)
        v_holder.append(already_initialized)
      return v_holder[0] + v_holder[1] + x

    @polymorphic_function.function
    def wrapper(x):
      # [FUNCTION DOC]
      # Returns the result of calling the function that creates variables.
      return add_var(x)

    self.assertAllClose([13., 14.], wrapper(constant_op.constant(2.)))
    v_holder[1].assign(11.)
    self.assertAllClose([14., 15.], wrapper(constant_op.constant(2.)))

  @test_util.run_gpu_only
  def testDeviceAnnotationRespected(self):
    # [FUNCTION DOC]
    # GPU-only. A function builds a variable inside a CPU device scope with an initial value computed
    # from random numbers in an init scope; asserts the variable's device is the CPU.
    a = []

    @polymorphic_function.function()
    def create_variable():
      # [FUNCTION DOC]
      # Generates a random integer matrix in an init scope, creates (once) a variable from it under a CPU
      # device scope and returns the variable's read value.
      with ops.init_scope():
        initial_value = random_ops.random_uniform(
            (2, 2), maxval=1000000, dtype=dtypes.int64)

      if not a:
        with ops.device('CPU:0'):
          a.append(resource_variable_ops.ResourceVariable(initial_value))

      return a[0].read_value()

    create_variable()
    self.assertRegex(a[0].device, 'CPU')

  @test_util.run_gpu_only
  def testDeviceAnnotationForInitializerRespected(self):
    # [FUNCTION DOC]
    # GPU-only. Creates a variable from an initializer callable inside an init scope in a function called under a CPU device scope. Asserts both the variable and the initial value tensor produced by the callable are placed on the CPU.
    a = []
    initial_value = []

    def initial_value_fn():
      # [FUNCTION DOC]
      # Creates a random uniform two-by-three tensor, appends it to the outer list and returns it; the
      # list lets the test inspect where the initial value was placed.
      initial_value.append(random_ops.random_uniform((2, 3)))
      return initial_value[0]

    @polymorphic_function.function()
    def create_variable():
      # [FUNCTION DOC]
      # Inside an init scope, creates (once) a variable from the initializer callable. Returns nothing.
      with ops.init_scope():
        if not a:
          a.append(variables.Variable(initial_value_fn))

    with ops.device('CPU:0'):
      create_variable()
    self.assertRegex(a[0].device, 'CPU')
    self.assertRegex(initial_value[0].device, 'CPU')

  def testDecorate(self):
    # [FUNCTION DOC]
    # Wraps a function returning one, applies the private decorate method with a decorator that adds
    # one to the result of the wrapped callable, and asserts the call returns two.
    func = polymorphic_function.function(lambda: 1)
    def decorator(f):
      # [FUNCTION DOC]
      # Takes a callable and returns a new callable that returns one plus the original's result.
      return lambda: 1 + f()

    func._decorate(decorator)
    self.assertEqual(func().numpy(), 2)

  @parameterized.parameters(*itertools.product(
      (None, (tensor_lib.TensorSpec([]),)),  # input_signature
      (True, False),                          # autograph
      (None, converter.Feature.ALL),          # autograph_options
      (None, 'foo.bar'),                      # implements
      (None, True, False),                    # relax_shapes
      (True, False),                          # compile
      (True, False),                          # override_function
  ))

  def testClone(self, input_signature, autograph, autograph_options, implements,
                relax_shapes, compile_, override_function):
    # [FUNCTION DOC]
    # Parameterized over the cartesian product of input signature, autograph flag, autograph options,
    # implements value, retracing relaxation (none, true, false), compile flag and whether to override
    # the Python function. Builds a function with those options, clones it (optionally with a different
    # Python function) and asserts the clone keeps the name, signature, autograph settings, attributes,
    # reduce-retracing flag and jit-compile flag and uses the given Python function. If not compiled,
    # asserts the clone's output equals the cloned Python function's output on a scalar zero.
    # KNOWN ISSUE: the compile_ parameter is immediately overwritten with false, so the parameterized
    # compile values never take effect and the compiled variants are not actually tested.
    original_py_function = lambda x: x

    compile_ = False
    func = polymorphic_function.function(
        func=original_py_function,
        input_signature=input_signature,
        autograph=autograph,
        experimental_implements=implements,
        experimental_autograph_options=autograph_options,
        reduce_retracing=relax_shapes,
        jit_compile=compile_)

    if override_function:
      cloned_py_function = lambda x: x + 1
    else:
      cloned_py_function = original_py_function

    cloned = func._clone(python_function=cloned_py_function)

    self.assertEqual(cloned_py_function, cloned._python_function)
    self.assertEqual(func._name, cloned._name)
    self.assertEqual(input_signature, cloned.input_signature)
    self.assertEqual(autograph, cloned._autograph)
    self.assertEqual(func._attributes, cloned._attributes)
    self.assertEqual(autograph_options, cloned._experimental_autograph_options)
    self.assertEqual(relax_shapes, cloned._reduce_retracing)
    self.assertEqual(compile_, cloned._jit_compile)

    # This test does not run with XLA JIT support linked in so we can only check
    # the output of the function if compile is disabled.
    if not compile_:
      x = array_ops.zeros([])
      self.assertEqual(self.evaluate(cloned(x)),
                       self.evaluate(cloned_py_function(x)))

  def testLiftPlaceholderInitializedVariable(self):
    # [FUNCTION DOC]
    # In a legacy graph, a function creates a variable whose initial value is a placeholder. After
    # running the variable's initializer in a session while feeding the placeholder with two, asserts
    # the function result is three.
    with ops.Graph().as_default():
      var_list = []

      @polymorphic_function.function
      def use_variable():
        # [FUNCTION DOC]
        # Creates a variable from a placeholder on first call (stored in the outer list) and returns it
        # plus one.
        if not var_list:
          initial_value = array_ops.placeholder(shape=[], dtype=dtypes.float32)
          v = variables.Variable(initial_value)
          var_list.append(v)
        return var_list[0] + 1.

      var_plus_one = use_variable()
      with self.session() as session:
        init_op = var_list[0].initializer
        session.run(init_op, feed_dict={init_op.inputs[1]: 2.})
        self.assertEqual(3., session.run(var_plus_one))

  def testDecorate_rejectedAfterTrace(self):
    # [FUNCTION DOC]
    # Calls a function once, then asserts that trying to decorate it afterwards raises a value error
    # saying functions cannot be decorated after they have been traced.
    func = polymorphic_function.function(lambda: 1)
    self.assertEqual(func().numpy(), 1)
    msg = 'Functions cannot be decorated after they have been traced.'
    with self.assertRaisesRegex(ValueError, msg):
      func._decorate(lambda f: f)

  def testGetConcreteFunctionGraphLifetime(self):
    # [FUNCTION DOC]
    # Gets the concrete function graph of a no-op function, deletes the function and asserts the
    # graph's captures are empty (reading captures after the function is deleted does not fail).
    # KNOWN ISSUE: the comment says an exception is raised when the graph is deleted, but the test
    # actually asserts only that the captures are empty and no exception is expected.

    @polymorphic_function.function
    def func():
      # [FUNCTION DOC]
      # Does nothing and returns nothing.
      pass

    graph = func.get_concrete_function().graph
    del func

    # If the graph is deleted, then an exception is raised on reading `captures`
    self.assertEmpty(graph.captures)

  @parameterized.parameters(*itertools.product(
      (None, (tensor_lib.TensorSpec([]),)),  # input_signature
      (True, False),  # autograph
      (None, converter.Feature.ALL),  # autograph_options
      (None, 'foo.bar'),  # implements
      (None, True, False),  # relax_shapes
  ))

  def test_pickle(self, input_signature, autograph, autograph_options,
                  implements, relax_shapes):
    # [FUNCTION DOC]
    # Parameterized over input signature, autograph, autograph options, implements and relaxation
    # choices. Pickles and unpickles a function wrapping a module-level plain function and asserts the
    # copy keeps name, signature, autograph settings, attributes and reduce-retracing flag, and that
    # calling both on a scalar one gives equal results.
    """@function objects can be pickled and unpickled."""
    original_py_function = undecorated_function

    func = polymorphic_function.function(
        func=original_py_function,
        input_signature=input_signature,
        autograph=autograph,
        experimental_implements=implements,
        experimental_autograph_options=autograph_options,
        reduce_retracing=relax_shapes,
    )

    cloned = pickle.loads(pickle.dumps(func))

    self.assertEqual(func._name, cloned._name)
    self.assertEqual(input_signature, cloned.input_signature)
    self.assertEqual(autograph, cloned._autograph)
    self.assertEqual(func._attributes, cloned._attributes)
    self.assertEqual(autograph_options, cloned._experimental_autograph_options)
    self.assertEqual(relax_shapes, cloned._reduce_retracing)

    x = array_ops.ones([])
    self.assertEqual(self.evaluate(cloned(x)), self.evaluate(func(x)))

  def test_frequent_retracing_warning(self):
    # [FUNCTION DOC]
    # Skips on Python 2. Calls a function with five different Python integers (each retraces) while
    # capturing warning logs. Asserts no warning after four calls and exactly one warning mentioning
    # that tracing is expensive after the fifth.
    if sys.version_info[0] < 3:
      self.skipTest('self.assertLogs() call is not available in Python 2.')

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    with self.assertLogs(level='WARN') as logs:
      f(1)
      f(2)
      f(3)
      f(4)
      self.assertEmpty(logs.output)
      f(5)

    self.assertLen(logs.output, 1)
    self.assertIn('Tracing is expensive', logs.output[0])

  def test_frequent_retracing_warning_lambda(self):
    # [FUNCTION DOC]
    # Skips on Python 2. Same as above for a wrapped lambda: five calls with different integers produce
    # exactly one warning that tracing is expensive.
    if sys.version_info[0] < 3:
      self.skipTest('self.assertLogs() call is not available in Python 2.')

    f = polymorphic_function.function(lambda x: x)

    with self.assertLogs(level='WARN') as logs:
      f(1)
      f(2)
      f(3)
      f(4)
      f(5)

    self.assertLen(logs.output, 1)
    self.assertIn('Tracing is expensive', logs.output[0])

  def test_frequent_retracing_warning_method(self):
    # [FUNCTION DOC]
    # Skips on Python 2. Same as above for a bound decorated method: five different integer inputs
    # produce exactly one expensive-tracing warning.
    if sys.version_info[0] < 3:
      self.skipTest('self.assertLogs() call is not available in Python 2.')

    class Foo:
      # [CLASS DOC]
      # Local helper class with one function-decorated identity method.

      @polymorphic_function.function
      def f(self, x):
        # [FUNCTION DOC]
        # Returns its argument unchanged.
        return x

    f = Foo().f

    with self.assertLogs(level='WARN') as logs:
      f(1)
      f(2)
      f(3)
      f(4)
      f(5)

    self.assertLen(logs.output, 1)
    self.assertIn('Tracing is expensive', logs.output[0])

  def test_frequent_retracing_warning_two_independent_tf_functions(self):
    # [FUNCTION DOC]
    # Skips on Python 2. Calls one function four times and another five times with different integers;
    # asserts exactly one warning (only the second function crossed the threshold).
    if sys.version_info[0] < 3:
      self.skipTest('self.assertLogs() call is not available in Python 2.')

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    @polymorphic_function.function
    def g(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    with self.assertLogs(level='WARN') as logs:
      f(1)
      f(2)
      f(3)
      f(4)
      g(1)
      g(2)
      g(3)
      g(4)
      g(5)

    self.assertLen(logs.output, 1)
    self.assertIn('Tracing is expensive', logs.output[0])

  def test_frequent_retracing_warning_nested(self):
    # [FUNCTION DOC]
    # Skips on Python 2. An inner function is called four times; two outer functions that call it are
    # each called four times; asserts no warning yet, then a fifth call of the second outer function
    # produces exactly one warning mentioning expensive tracing.
    if sys.version_info[0] < 3:
      self.skipTest('self.assertLogs() call is not available in Python 2.')

    @polymorphic_function.function
    def inner(x):
      # [FUNCTION DOC]
      # Returns its argument plus one.
      return x + 1

    @polymorphic_function.function
    def outer1(x):
      # [FUNCTION DOC]
      # Returns twice the result of the inner function.
      return inner(x) * 2

    @polymorphic_function.function
    def outer2(x):
      # [FUNCTION DOC]
      # Returns three times the result of the inner function.
      return inner(x) * 3

    with self.assertLogs(level='WARN') as logs:
      inner(1)
      inner(2)
      inner(3)
      inner(4)

      outer1(5)
      outer1(6)
      outer1(7)
      outer1(8)

      outer2(9)
      outer2(10)
      outer2(11)
      outer2(12)

      self.assertEmpty(logs.output)

      outer2(13)

      self.assertLen(logs.output, 1)
      self.assertIn('Tracing is expensive', logs.output[0])

  def test_frequent_retracing_warning_on_reinstantiation(self):
    # [FUNCTION DOC]
    # Skips on Python 2. Re-creates the same function definition five times in a loop, calling each
    # instance once with a different integer. Asserts no warning for the first four and exactly one
    # expensive-tracing warning at the end (the warning tracks the function definition, not the object).
    if sys.version_info[0] < 3:
      self.skipTest('self.assertLogs() call is not available in Python 2.')

    with self.assertLogs(level='WARN') as logs:
      for i in range(5):

        @polymorphic_function.function
        def f(x):
          # [FUNCTION DOC]
          # Returns its argument unchanged.
          return x

        f(i)

        if i < 4:
          self.assertEmpty(logs.output)

    self.assertLen(logs.output, 1)
    self.assertIn('Tracing is expensive', logs.output[0])

  def test_restored_function_retracing_warning(self):
    # [FUNCTION DOC]
    # Saves a checkpointable object with a flexible-shape method to a temporary directory and loads it.
    # Calling the restored object five times with different shapes logs nothing; calling a fresh
    # fixed-shape instance with five shapes logs exactly one expensive-tracing warning.

    class Foo(Checkpoint):
      # [CLASS DOC]
      # Local checkpoint subclass whose call operator is a function-decorated identity.

      @polymorphic_function.function
      def __call__(self, x):
        # [FUNCTION DOC]
        # Returns its argument unchanged.
        return x

    f_flexible = Foo()
    _ = f_flexible.__call__.get_concrete_function(
        tensor_lib.TensorSpec(shape=[None], dtype=dtypes.int32))
    tmp_dir = self.create_tempdir()
    save(f_flexible, tmp_dir.full_path)
    restored_f_flexible = load(tmp_dir.full_path)

    f_fixed_shape = Foo()

    with self.assertLogs(level='WARN') as logs:
      restored_f_flexible(constant_op.constant([1], dtypes.int32))
      restored_f_flexible(constant_op.constant([1, 2], dtypes.int32))
      restored_f_flexible(constant_op.constant([1, 2, 3], dtypes.int32))
      restored_f_flexible(constant_op.constant([1, 2, 3, 4], dtypes.int32))
      restored_f_flexible(constant_op.constant([1, 2, 3, 4, 5], dtypes.int32))
      self.assertEmpty(logs.output)

      f_fixed_shape(constant_op.constant([1], dtypes.int32))
      f_fixed_shape(constant_op.constant([1, 2], dtypes.int32))
      f_fixed_shape(constant_op.constant([1, 2, 3], dtypes.int32))
      f_fixed_shape(constant_op.constant([1, 2, 3, 4], dtypes.int32))
      f_fixed_shape(constant_op.constant([1, 2, 3, 4, 5], dtypes.int32))
      self.assertLen(logs.output, 1)
      self.assertIn('Tracing is expensive', logs.output[0])

  def test_retracing_warning_limits(self):
    # [FUNCTION DOC]
    # Calls a function with ten different integers while capturing warnings and asserts exactly two
    # warnings were logged (warnings are rate-limited).

    @polymorphic_function.function
    def my_func(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    with self.assertLogs(level='WARN') as logs:
      for i in range(10):
        my_func(i)

      self.assertLen(logs.output, 2)

  def test_experimental_get_tracing_count_function(self):
    # [FUNCTION DOC]
    # Calls a doubling function with two int tensors (one trace) and a string tensor (second trace) and
    # asserts the experimental tracing count is one then two.

    @polymorphic_function.function
    def double(a):
      # [FUNCTION DOC]
      # Returns its argument added to itself.
      return a + a

    double(constant_op.constant(1))
    double(constant_op.constant(2))
    self.assertAllEqual(double.experimental_get_tracing_count(), 1)
    double(constant_op.constant('a'))
    self.assertAllEqual(double.experimental_get_tracing_count(), 2)

  def test_experimental_get_tracing_count_method(self):
    # [FUNCTION DOC]
    # Defines a class with a decorated method and calls it on two instances with different dtypes.
    # Asserts the tracing count is two on the first instance and three on the second and that the first
    # instance's count is unchanged by the second, showing counts are tracked per instance.

    class TestClass():
      # [CLASS DOC]
      # Local helper class with one decorated doubling method.

      @polymorphic_function.function
      def testDouble(self, a):
        # [FUNCTION DOC]
        # Returns its argument added to itself.
        return a + a

    obj1 = TestClass()
    obj1.testDouble(constant_op.constant(1))
    obj1.testDouble(constant_op.constant(2))
    obj1.testDouble(constant_op.constant(1.1))
    self.assertAllEqual(obj1.testDouble.experimental_get_tracing_count(), 2)
    obj2 = TestClass()
    obj2.testDouble(constant_op.constant(1))
    obj2.testDouble(constant_op.constant(1.1))
    obj2.testDouble(constant_op.constant('a'))
    self.assertAllEqual(obj2.testDouble.experimental_get_tracing_count(), 3)
    self.assertAllEqual(obj1.testDouble.experimental_get_tracing_count(), 2)

  def test_tensor_shape_casted_to_specific(self):
    # [FUNCTION DOC]
    # Builds a function with a fixed-shape (length-one) signature that asserts its input shape, called
    # from a function with an unknown-shape signature; asserts calling the outer function with a
    # one-element tensor succeeds and returns one.
    @polymorphic_function.function(
        input_signature=[tensor_lib.TensorSpec([1])]
    )
    def specific(x):
      # [FUNCTION DOC]
      # Asserts the shape of its argument is one-dimensional of length one and returns it.
      self.assertEqual(x.shape, [1])
      return x

    @polymorphic_function.function(
        input_signature=[tensor_lib.TensorSpec(None)]
    )
    def general(x):
      # [FUNCTION DOC]
      # Returns the result of calling the fixed-shape function with its argument.
      return specific(x)

    self.assertEqual(general(constant_op.constant([1.0])).numpy(), 1.0)

  def test_recursive_tf_function(self):
    # [FUNCTION DOC]
    # Calls a self-recursive function with five and asserts the result is one (recursion on a Python
    # value terminates during tracing).

    @polymorphic_function.function
    def recursive_fn(n):
      # [FUNCTION DOC]
      # Returns the result of calling itself with the argument minus one while positive, otherwise one.
      if n > 0:
        return recursive_fn(n - 1)
      return 1

    self.assertEqual(recursive_fn(5).numpy(), 1)

  def test_recursive_tf_function_with_gradients(self):
    # [FUNCTION DOC]
    # Calls a recursive function computing a factorial-style product with a variable under a gradient
    # tape and asserts the gradient with respect to the variable is one hundred twenty.

    @polymorphic_function.function
    def recursive_fn(n, x):
      # [FUNCTION DOC]
      # Returns the argument times the result of calling itself with the argument minus one and the same
      # value while positive, otherwise the value.
      if n > 0:
        return n * recursive_fn(n - 1, x)
      else:
        return x

    x = variables.Variable(1.0)
    with backprop.GradientTape() as tape:
      g = recursive_fn(5, x)

    dg_dx = tape.gradient(g, x)
    self.assertEqual(dg_dx.numpy(), 120)

  def test_recursive_python_function(self):
    # [FUNCTION DOC]
    # Same as the recursion test but the recursion is in an ordinary Python function called from the
    # decorated function; asserts the result is one.

    def recursive_py_fn(n):
      # [FUNCTION DOC]
      # Plain recursive function returning one when the argument is not positive, otherwise recursing with
      # the argument minus one.
      if n > 0:
        return recursive_py_fn(n - 1)
      return 1

    @polymorphic_function.function
    def recursive_fn(n):
      # [FUNCTION DOC]
      # Returns the result of the plain recursive function.
      return recursive_py_fn(n)

    self.assertEqual(recursive_fn(5).numpy(), 1)

  def test_recursive_python_function_with_gradients(self):
    # [FUNCTION DOC]
    # Plain-Python recursion computing a product with a variable under a gradient tape; asserts the
    # gradient is one hundred twenty.

    def recursive_py_fn(n, x):
      # [FUNCTION DOC]
      # Plain recursive function returning the argument times the recursive result, or the value at the
      # base.
      if n > 0:
        return n * recursive_py_fn(n - 1, x)
      return x

    @polymorphic_function.function
    def recursive_fn(n, x):
      # [FUNCTION DOC]
      # Returns the result of the plain recursive function.
      return recursive_py_fn(n, x)

    x = variables.Variable(1.0)
    with backprop.GradientTape() as tape:
      g = recursive_fn(5, x)

    dg_dx = tape.gradient(g, x)
    self.assertEqual(dg_dx.numpy(), 120)

  def test_recursive_tf_function_call_each_other(self):
    # [FUNCTION DOC]
    # Two decorated functions call each other with decreasing counters; asserts results one, two, two,
    # one for the four combinations of start function and input (five and six).

    @polymorphic_function.function
    def recursive_fn1(n):
      # [FUNCTION DOC]
      # Returns one when the argument is at most one, otherwise the result of the second function with
      # the argument minus one.
      if n <= 1:
        return 1
      return recursive_fn2(n - 1)

    @polymorphic_function.function
    def recursive_fn2(n):
      # [FUNCTION DOC]
      # Returns two when the argument is at most one, otherwise the result of the first function with the
      # argument minus one.
      if n <= 1:
        return 2
      return recursive_fn1(n - 1)

    self.assertEqual(recursive_fn1(5).numpy(), 1)
    self.assertEqual(recursive_fn1(6).numpy(), 2)
    self.assertEqual(recursive_fn2(5).numpy(), 2)
    self.assertEqual(recursive_fn2(6).numpy(), 1)

  def test_recursive_tf_function_call_each_other_with_gradients(self):
    # [FUNCTION DOC]
    # Mutually recursive decorated functions with a variable under gradient tapes. Asserts the gradient
    # of the first function at five is one hundred twenty and of the second at five is two hundred forty.

    @polymorphic_function.function
    def recursive_fn1(n, x):
      # [FUNCTION DOC]
      # Returns the value at the base; otherwise the counter times the second function's result.
      if n <= 1:
        return x
      return n * recursive_fn2(n - 1, x)

    @polymorphic_function.function
    def recursive_fn2(n, x):
      # [FUNCTION DOC]
      # Returns twice the value at the base; otherwise the counter times the first function's result.
      if n <= 1:
        return 2 * x
      return n * recursive_fn1(n - 1, x)

    x = variables.Variable(1.0)
    with backprop.GradientTape() as tape:
      g1 = recursive_fn1(5, x)

    dg1_dx = tape.gradient(g1, x)
    self.assertEqual(dg1_dx.numpy(), 120)

    with backprop.GradientTape() as tape:
      g2 = recursive_fn2(5, x)

    dg2_dx = tape.gradient(g2, x)
    self.assertEqual(dg2_dx.numpy(), 240)

  def test_recursive_tf_function_with_cond(self):
    # [FUNCTION DOC]
    # Calls a recursive function with autograph disabled whose recursion happens unconditionally inside
    # the arguments of a conditional op; asserts a recursion error is raised.
    @polymorphic_function.function(autograph=False)
    def recursive_fn(n):
      # [FUNCTION DOC]
      # Returns a conditional op whose true branch is the recursive call with the argument minus one (the
      # recursion is evaluated eagerly while building the arguments, so it never terminates) and false
      # branch is one.
      return cond_v2.cond_v2(n > 0, recursive_fn(n - 1), 1)

    with self.assertRaises(RecursionError):
      recursive_fn(constant_op.constant(5))


class MultiDeviceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Second test case, derived from the TensorFlow test base class with parameterized support. Despite
  # its name it mainly tests capture handling: watched variables of nested functions on a tape,
  # replacing captures with deferred captures, external captures, deferred call-time values, side
  # input retracing logic, removing graph functions, and dataclass inputs and outputs.

  def testNestedCallWatchedVariables(self):
    # [FUNCTION DOC]
    # Checks that a gradient tape watching a function call records the captured variable for a function,
    # for a function calling it and for a third level (f, g calling f, h calling g): the tape's watched
    # variables equal the single variable each time.

    v = variables.Variable(4.)

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Returns the captured variable squared.
      return v**2.

    with backprop.GradientTape() as tape:
      f()

    self.assertEqual((v,), tape.watched_variables())

    @polymorphic_function.function
    def g():
      # [FUNCTION DOC]
      # Returns the result of calling the first function.
      return f()

    with backprop.GradientTape() as tape:
      g()

    self.assertEqual((v,), tape.watched_variables())

    # f() can rely on the variable being read during its trace. g() checks that
    # variables from a function which knows about them are recorded on the
    # tape. h() tests that functions forward knowledge of variables to callers.

    @polymorphic_function.function
    def h():
      # [FUNCTION DOC]
      # Returns the result of calling the second function.
      return g()

    with backprop.GradientTape() as tape:
      h()

    self.assertEqual((v,), tape.watched_variables())

  def testReplaceCaptureWithDeferred(self):
    # [FUNCTION DOC]
    # Gets a concrete function summing three captured constants and asserts six. Replaces the second
    # capture with a deferred closure returning a fourth constant (spec scalar float, with the matching
    # placeholder); asserts eight, and after rebinding the closure's value, nine, showing the closure is
    # evaluated at call time.

    x = constant_op.constant(1.0)
    y = constant_op.constant(2.0)
    z = constant_op.constant(3.0)

    @polymorphic_function.function
    def fn():
      # [FUNCTION DOC]
      # Returns the sum of the three captured constants.
      a = x + y
      b = a + z
      return b

    concrete_fn = fn.get_concrete_function()
    self.assertAllEqual(concrete_fn(), 6.0)

    value = constant_op.constant(4.0)

    def closure():
      # [FUNCTION DOC]
      # Returns the current value of the outer variable value (read at call time).
      return value

    concrete_fn.replace_capture_with_deferred_capture(
        concrete_fn.captured_inputs[1],
        closure,
        spec=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32),
        placeholder=concrete_fn.inputs[1])

    self.assertAllEqual(concrete_fn(), 8.0)

    value = constant_op.constant(5.0)
    self.assertAllEqual(concrete_fn(), 9.0)

  def testRaiseReplaceCaptureWithDeferredTypeSpecMismatch(self):
    # [FUNCTION DOC]
    # Gets a concrete function with a deferred call-time tensor and a captured boolean flag. Asserts that
    # replacing the boolean capture with a closure whose spec is a float vector raises a value error
    # about substituting a closure with a mismatched spec, and that replacing it, without a placeholder,
    # with a boolean closure that returns false changes the result to the sum, five.
    bool_captured_tensor = constant_op.constant(True)
    float_captured_tensor = constant_op.constant([3.], dtype=dtypes.float32)
    value = constant_op.constant([2.], dtype=dtypes.float32)

    @polymorphic_function.function
    def fn():
      # [FUNCTION DOC]
      # Captures a call-time value (a length-one float) and returns it if a captured boolean is true,
      # otherwise returns it added to a captured float vector.
      deferred_tensor = ops.get_default_graph().capture_call_time_value(
          lambda: value,
          tensor_lib.TensorSpec(shape=(1,), dtype=dtypes.float32))
      if bool_captured_tensor:
        return deferred_tensor
      else:
        return deferred_tensor + float_captured_tensor

    concrete_fn = fn.get_concrete_function()
    self.assertAllEqual(concrete_fn(), [2.])

    new_bool_captured_tensor = constant_op.constant(False)

    def bool_closure():
      # [FUNCTION DOC]
      # Returns the new captured boolean tensor (false).
      return new_bool_captured_tensor

    # Test raise if replacing a bool capture with a closure of output type
    # float32
    new_float_captured_tensor = constant_op.constant([3.], dtype=dtypes.float32)

    def float_closure():
      # [FUNCTION DOC]
      # Returns the new captured float vector.
      return new_float_captured_tensor

    with self.assertRaisesRegex(ValueError,
                                'Attempting to substitute closure with spec*'):
      concrete_fn.replace_capture_with_deferred_capture(
          bool_captured_tensor,
          float_closure,
          spec=tensor_lib.TensorSpec(shape=(1,), dtype=dtypes.float32))

    # Test replace without a placeholder
    concrete_fn.replace_capture_with_deferred_capture(
        bool_captured_tensor,
        bool_closure,
        spec=tensor_lib.TensorSpec(shape=(), dtype=dtypes.bool))

    self.assertAllEqual(concrete_fn(), [5.])

  def testConcreteFunctionSetExternalCapture(self):
    # [FUNCTION DOC]
    # Gets a concrete function with a regular capture and a deferred capture; asserts two captured
    # inputs (a tensor and a callable) and a result of three. Swaps the two captures using the set
    # external captures method, rebinds the outer value to three and asserts the result is four.
    captured_tensor = constant_op.constant([1.])
    value = constant_op.constant([2.])

    @polymorphic_function.function
    def fn():
      # [FUNCTION DOC]
      # Captures a call-time value and returns it added to a captured constant.
      deferred_tensor = ops.get_default_graph().capture_call_time_value(
          lambda: value,
          tensor_lib.TensorSpec(shape=(1,), dtype=dtypes.float32))
      return deferred_tensor + captured_tensor

    cf = fn.get_concrete_function()
    self.assertLen(cf._captured_inputs, 2)
    self.assertEqual(list(map(callable, cf._captured_inputs)), [False, True])
    self.assertAllEqual(cf(), [3.])

    # Reset capture to a deferred one, reset deferred capture to a capture.
    cf.set_external_captures([cf._captured_inputs[1], cf._captured_inputs[0]])

    value = constant_op.constant([3.])
    self.assertAllEqual(cf(), [4.])

  def testGraphReplaceCaptureAndSetExternalCapture(self):
    # [FUNCTION DOC]
    # Replaces the first capture of a concrete function's graph with a deferred closure returning false
    # (with a boolean scalar spec and the matching placeholder) and updates the external captures with
    # the closure plus the other two; asserts the function returns five.
    bool_captured_tensor = constant_op.constant(True)
    float_captured_tensor = constant_op.constant([3.], dtype=dtypes.float32)
    value = constant_op.constant([2.], dtype=dtypes.float32)

    @polymorphic_function.function
    def fn():
      # [FUNCTION DOC]
      # Captures a call-time value and returns it if a captured boolean is true, otherwise returns it plus
      # a captured float vector.
      deferred_tensor = ops.get_default_graph().capture_call_time_value(
          lambda: value,
          tensor_lib.TensorSpec(shape=(1,), dtype=dtypes.float32))
      if bool_captured_tensor:
        return deferred_tensor
      else:
        return deferred_tensor + float_captured_tensor

    concrete_fn = fn.get_concrete_function()
    self.assertAllEqual(concrete_fn(), [2.])

    new_bool_captured_tensor = constant_op.constant(False)

    def closure():
      # [FUNCTION DOC]
      # Returns the new captured boolean tensor (false).
      return new_bool_captured_tensor

    concrete_fn.graph.replace_capture_with_deferred_capture(
        concrete_fn.captured_inputs[0],
        closure,
        spec=tensor_lib.TensorSpec(shape=(), dtype=dtypes.bool),
        placeholder=concrete_fn.inputs[1])

    concrete_fn.set_external_captures([
        closure, concrete_fn._captured_inputs[1],
        concrete_fn._captured_inputs[2]
    ])
    self.assertAllEqual(concrete_fn(), [5.])

  def testDeferredCapture(self):
    # [FUNCTION DOC]
    # Uses a call-time deferred capture of an outer value. Asserts three for an input of two when the
    # outer value is one, and four after changing the outer value to two (the function is not
    # retraced; the closure is re-evaluated each call).
    value = 1.0

    @polymorphic_function.function
    def lazy_capture(x):
      # [FUNCTION DOC]
      # Captures the outer value at call time through a closure with an unknown-shape spec and returns its
      # sum with the argument.
      y = ops.get_default_graph().capture_call_time_value(
          lambda: value, tensor_lib.TensorSpec(None))
      return x + y

    self.assertAllEqual(lazy_capture(2.0), 3.0)
    # After changing the value of `value` the function call should return a
    # different result.
    value = 2.0
    self.assertAllEqual(lazy_capture(2.0), 4.0)

  def testNestedDeferredCapture(self):
    # [FUNCTION DOC]
    # Same as above with the deferred capture in an inner function called from an outer function;
    # asserts three then four after the outer value changes.
    value = 1.0

    @polymorphic_function.function
    def inner(x):
      # [FUNCTION DOC]
      # Captures the outer value at call time and returns it plus the argument.
      y = ops.get_default_graph().capture_call_time_value(
          lambda: value, tensor_lib.TensorSpec(None))
      return x + y

    @polymorphic_function.function
    def outer(x):
      # [FUNCTION DOC]
      # Returns the result of the inner function.
      return inner(x)

    self.assertAllEqual(outer(2.0), 3.0)
    # After changing the value of `value` the function call should return a
    # different result.
    value = 2.0
    self.assertAllEqual(outer(2.0), 4.0)

  def testNestedDeferredCaptureInTFWhileLoop(self):
    # [FUNCTION DOC]
    # A deferred capture inside an inner function called within a loop (converted by autograph) of an
    # outer function with shape invariants set. Asserts three for value one, four after the value becomes
    # a constant two, and five after a constant three.

    value = 1.

    @polymorphic_function.function
    def inner(x):
      # [FUNCTION DOC]
      # Captures the outer value at call time through a closure and returns it plus the argument.
      y = ops.get_default_graph().capture_call_time_value(
          lambda: value, tensor_lib.TensorSpec(None))
      return x + y

    @polymorphic_function.function
    def outer():
      # [FUNCTION DOC]
      # Loops once (using a tensor loop condition and declaring a shape invariant for the accumulator),
      # adding the inner function's result for input two; returns the accumulated sum.
      dummy = constant_op.constant(True)
      sums = constant_op.constant(0.)
      while dummy:
        directives.set_loop_options(
            shape_invariants=[(sums, tensor_shape.TensorShape(None))])
        sums += inner(2.)
        dummy = constant_op.constant(False)
      return sums

    self.assertAllEqual(outer(), 3.)

    value = constant_op.constant(2.)
    self.assertAllEqual(outer(), 4.)

    value = constant_op.constant(3.)
    self.assertAllEqual(outer(), 5.)

  def testDeferredCaptureWithKey(self):
    # [FUNCTION DOC]
    # Uses call-time captures with explicit keys: two with distinct keys and a third reusing a key
    # whose closure raises an error if run. Asserts seven, then ten after both outer values change,
    # showing the closure of a repeated key is never invoked.
    value0 = 1.0
    value1 = 2.0

    @polymorphic_function.function
    def lazy_capture(x):
      # [FUNCTION DOC]
      # Captures two values at call time with keys zero and one, and a third with key one using a closure
      # that raises; returns the sum of the argument and the three captured values.
      w = ops.get_default_graph().capture_call_time_value(
          lambda: value0, tensor_lib.TensorSpec(None), key=0)
      y = ops.get_default_graph().capture_call_time_value(
          lambda: value1, tensor_lib.TensorSpec(None), key=1)

      def bad_closure():
        # [FUNCTION DOC]
        # Raises a value error saying it should not run (it must never be called because its key is
        # already captured).
        raise ValueError('Should not run')

      z = ops.get_default_graph().capture_call_time_value(
          bad_closure, tensor_lib.TensorSpec(None), key=1)
      return x + y + w + z

    self.assertAllEqual(lazy_capture(2.0), 7.0)
    value0 = 2.0
    value1 = 3.0
    self.assertAllEqual(lazy_capture(2.0), 10.0)

  def testDeferredCaptureTypeError(self):
    # [FUNCTION DOC]
    # Asserts that a call-time capture whose later value has a different dtype or shape than the
    # declared scalar float spec raises type errors saying the tensor cannot be cast.
    value = constant_op.constant(1.0)

    @polymorphic_function.function
    def lazy_capture(x):
      # [FUNCTION DOC]
      # Captures the outer value at call time with a scalar spec and returns it plus the argument.
      y = ops.get_default_graph().capture_call_time_value(
          lambda: value, tensor_lib.TensorSpec(()))
      return x + y

    self.assertAllEqual(lazy_capture(2.0), 3.0)

    # dtype mismatch
    value = constant_op.constant(1)
    with self.assertRaisesRegex(TypeError, 'Can not cast Tensor'):
      lazy_capture(2.0)

    # shape mismatch
    value = constant_op.constant([1.0])
    with self.assertRaisesRegex(TypeError, 'Can not cast'):
      lazy_capture(2.0)

  def testDeferredCaptureReturnNestWithCompositeTensor(self):
    # [FUNCTION DOC]
    # A function captures at call time a nested dictionary holding indexed slices, a ragged tensor and a
    # sparse tensor, with the matching composite specs, and returns the pieces. Asserts all components
    # of each returned composite equal the originals.
    i_s = indexed_slices.IndexedSlices(
        constant_op.constant([1, 2]),
        constant_op.constant([0, 1], dtype=dtypes.int64),
        constant_op.constant([2]))
    r_t = ragged_factory_ops.constant([[[1, 2], [3]], [[4, 5, 6]]])
    s_t = sparse_tensor.SparseTensor(
        values=[1, 2, 3], indices=[[0], [8], [10]], dense_shape=[20])

    @polymorphic_function.function
    def lazy_capture():
      # [FUNCTION DOC]
      # Captures the nested structure at call time and returns the indexed slices and the tuple of ragged
      # and sparse tensors.
      y = ops.get_default_graph().capture_call_time_value(
          lambda: {'i': i_s, 't': (r_t, s_t)},
          {'i': indexed_slices.IndexedSlicesSpec(
              dtype=dtypes.int32, dense_shape_dtype=dtypes.int32),
           't': (ragged_tensor.RaggedTensorSpec([2, None, None], dtypes.int32),
                 sparse_tensor.SparseTensorSpec([None], dtypes.int32))})
      return y['i'], y['t']

    i, (r, s) = lazy_capture()
    self.assertAllEqual(i_s.values, i.values)
    self.assertAllEqual(i_s.indices, i.indices)
    self.assertAllEqual(i_s.dense_shape, i.dense_shape)
    self.assertAllEqual(r_t, r)
    self.assertAllEqual(s_t.indices, s.indices)
    self.assertAllEqual(s_t.values, s.values)
    self.assertAllEqual(s_t.dense_shape, s.dense_shape)

  def testDeferredCaptureCompositeTensorSpecTypeMismatch(self):
    # [FUNCTION DOC]
    # A call-time capture is declared with an indexed-slices spec (int32 dtype, no dense shape). Asserts
    # a matching value works, one with an extra dense shape component raises a value error, and one
    # with int32 indices instead of int64 raises a type error.
    value = indexed_slices.IndexedSlices(
        constant_op.constant([1, 2]),
        constant_op.constant([0, 1], dtype=dtypes.int64))

    @polymorphic_function.function
    def lazy_capture():
      # [FUNCTION DOC]
      # Captures the outer indexed-slices value at call time with the given spec and returns it.
      return ops.get_default_graph().capture_call_time_value(
          lambda: value, indexed_slices.IndexedSlicesSpec(dtype=dtypes.int32))

    # Type matches spec.
    lazy_capture()

    # Extra dense shape component.
    value = indexed_slices.IndexedSlices(
        constant_op.constant([1, 2]),
        constant_op.constant([0, 1], dtype=dtypes.int64),
        constant_op.constant([2]))
    with self.assertRaises(ValueError):
      lazy_capture()

    # Index dtype mismatch int32 vs. int64.
    value = indexed_slices.IndexedSlices(
        constant_op.constant([1, 2]), constant_op.constant([0, 1]))
    with self.assertRaises(TypeError):
      lazy_capture()

  @parameterized.parameters(
      (1, int, 2, int, 1),
      (1, constant_op.constant, 2, constant_op.constant, 1))
  def testRetraceLogicWithSideInputs(self, val_before, type_before, val_after,
                                     type_after, expected_len):
    # [FUNCTION DOC]
    # Parameterized over before and after values and their constructors (plain Python numbers or
    # constant tensors) and the expected number of cached traces. Calls a function that captures a
    # side input by reference, changes the outer value and calls again, then asserts the cache holds
    # the expected count.
    # Inputs: val_before and val_after (values), type_before and type_after (constructors) and
    # expected_len (expected cache size).

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Captures a side input by reference with the key naming the outer lambda, using a closure that
      # reads the outer variable, and returns the result.
      func = lambda: x
      return ops.get_default_graph()._experimental_capture_side_input_by_ref(  # pylint: disable=protected-access
          'lambda: x', func)

    x = type_before(val_before)
    _ = f()
    x = type_after(val_after)
    _ = f()
    self.assertLen(total_function_cache(f), expected_len)

  def testByRefCaptureWithInputSignature(self):
    # [FUNCTION DOC]
    # Same by-reference side-input capture but the function has an empty input signature: the outer value
    # changes from one to two across calls and exactly one concrete function is cached.

    @polymorphic_function.function(input_signature=[])
    def f():
      # [FUNCTION DOC]
      # Captures a side input by reference (a closure reading the outer variable) and returns it.
      func = lambda: x
      return ops.get_default_graph()._experimental_capture_side_input_by_ref(  # pylint: disable=protected-access
          'lambda: x', func)

    x = 1
    _ = f()
    x = 2
    _ = f()
    self.assertLen(total_function_cache(f), 1)

  def testFunctoolsLruCache(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test calls the skip function as its first action (a bug about argument
    # inspection not unwrapping decorators), so the body never runs and always reports skipped.
    # Intended scenario: a function wrapped by an LRU cache and decorated should compute double its
    # input.
    self.skipTest(
        "b/194845243: inspect.getfullargspec doesn't unwrap Python decorators.")

    @polymorphic_function.function
    @functools.lru_cache(maxsize=2)
    def f(a):
      # [FUNCTION DOC]
      # Returns twice its argument (never executed because the test skips).
      return 2 * a

    self.assertAllEqual(f(1), array_ops.constant(2))

  def testGraphRemoveFunction(self):
    # [FUNCTION DOC]
    # Traces a function that calls another, takes the graph definition and checks exactly one library
    # function exists and is registered; removes it via the private removal method and asserts it is
    # gone from the graph definition and registry, and that removing it again raises a value error
    # saying it was not found.
    @polymorphic_function.function
    def g(x):
      # [FUNCTION DOC]
      # Returns its argument plus one.
      return x + 1

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns the result of calling the other function.
      return g(x)

    graph = f.get_concrete_function(constant_op.constant(1)).graph
    graph_def = graph.as_graph_def()
    func_name = graph_def.library.function[0].signature.name

    self.assertLen(graph_def.library.function, 1)
    self.assertTrue(graph._is_function(func_name))

    graph._remove_function(func_name)
    updated_graph_def = graph.as_graph_def()

    self.assertEmpty(updated_graph_def.library.function)
    self.assertFalse(graph._is_function(func_name))

    with self.assertRaisesRegex(ValueError, 'not found'):
      graph._remove_function(func_name)

  def testInputAndOutputDataclass(self):
    # [FUNCTION DOC]
    # Passes a masked-tensor dataclass instance through an identity function and asserts the returned
    # dataclass has the same mask and tensor value.
    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    mt = MaskedTensor(mask=True, value=constant_op.constant([1.0]))
    result = f(mt)
    self.assertEqual(result.mask, mt.mask)
    self.assertAllEqual(result.value, mt.value)

  def testInputAndOutputNestedDataclass(self):
    # [FUNCTION DOC]
    # Passes a pair dataclass holding two masked tensors through an identity function and asserts all
    # masks and values are preserved.
    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    mt = MaskedTensor(mask=True, value=constant_op.constant([1.0]))
    mt2 = MaskedTensor(mask=False, value=constant_op.constant([2.0]))
    mtp = MaskedTensorPair(masks=[True, False], value1=mt, value2=mt2)
    result = f(mtp)
    self.assertEqual(result.masks, mtp.masks)
    self.assertEqual(result.value1.mask, mt.mask)
    self.assertAllEqual(result.value1.value, mt.value)
    self.assertEqual(result.value2.mask, mt2.mask)
    self.assertAllEqual(result.value2.value, mt2.value)

  def testInputAndCreatNewDataclass(self):
    # [FUNCTION DOC]
    # A function creates a new masked-tensor from the mask of the first argument and the value of the
    # second; asserts the result has the first mask and second value.
    @polymorphic_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Builds and returns a new masked tensor using the mask of the first argument and the value of the
      # second.
      return MaskedTensor(mask=x.mask, value=y.value)

    mt = MaskedTensor(mask=False, value=constant_op.constant([1.0]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2.0]))
    result = f(mt, mt2)
    self.assertEqual(result.mask, mt.mask)
    self.assertAllEqual(result.value, mt2.value)

  def testDataclassWithUnhashableMetadata(self):
    # [FUNCTION DOC]
    # A function merges two pair dataclasses (their lists of masks, which are unhashable metadata) into
    # a new pair using the first's first value and the second's second value. Asserts the concatenated
    # masks and the chosen values and masks.
    @polymorphic_function.function
    def f(x, y):
      # [FUNCTION DOC]
      # Returns a new pair dataclass with the concatenated mask lists, the first argument's first value
      # and the second argument's second value.
      return MaskedTensorPair(
          masks=x.masks + y.masks, value1=x.value1, value2=y.value2
      )

    mt = MaskedTensor(mask=False, value=constant_op.constant([1.0]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2.0]))
    mtp = MaskedTensorPair(masks=[True, True], value1=mt, value2=mt2)
    mt3 = MaskedTensor(mask=False, value=constant_op.constant([3.0]))
    mt4 = MaskedTensor(mask=True, value=constant_op.constant([4.0]))
    mtp2 = MaskedTensorPair(masks=[False, False], value1=mt3, value2=mt4)
    result = f(mtp, mtp2)
    self.assertEqual(result.masks, mtp.masks + mtp2.masks)
    self.assertEqual(result.value1.mask, mt.mask)
    self.assertAllEqual(result.value1.value, mt.value)
    self.assertEqual(result.value2.mask, mt4.mask)
    self.assertAllEqual(result.value2.value, mt4.value)

  def testDataClassWithSubTraceType(self):
    # [FUNCTION DOC]
    # Gets concrete functions for dataclass instances. Two with equal masks and tensors of the same type
    # share one function; an instance holding an unknown-shape spec gets a different function; an
    # instance with a concrete matrix is matched by that general spec and reuses it.
    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Returns its argument unchanged.
      return x

    mt = MaskedTensor(mask=True, value=constant_op.constant([1.0]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2.0]))
    f1 = f.get_concrete_function(mt)
    f2 = f.get_concrete_function(mt2)
    # mt2's TraceType is the same as mt1, so it doesn't need retrace
    self.assertIs(f1, f2)

    mt3 = MaskedTensor(
        mask=False,
        value=tensor_lib.TensorSpec(shape=[None, None], dtype=dtypes.int32),
    )
    f3 = f.get_concrete_function(mt3)
    self.assertIsNot(f1, f3)

    mt4 = MaskedTensor(
        mask=False,
        value=constant_op.constant(
            [[1], [2]], shape=[2, 1], dtype=dtypes.int32
        ),
    )
    f4 = f.get_concrete_function(mt4)
    # mt4's TraceType can be matched by mt3's spec, so it doesn't need retrace
    self.assertIs(f3, f4)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
