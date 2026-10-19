# [MODULE DOC]
# Unit tests for the tracing compilation layer of the polymorphic function machinery. The tests
# drive the tracing entry points (call and trace functions with tracing options) through a small
# decorator defined in this file that mimics a function-compilation decorator. They cover:
# caching and retracing behavior keyed on argument types, shape relaxation, numpy and variable
# handling, input signatures (nested, dict, composite, keyword-only), function registration in
# graphs, grappler implementation selection, capture handling, multi-device placement, and
# argument naming of exported concrete functions.
# Import-time side effects: optionally imports the attrs library; the module-level setup hook
# enables eager execution and configures four logical CPU devices before the tests run.
# Several tests need a GPU and are skipped otherwise.
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

import copy
import functools
import itertools
import weakref

from absl.testing import parameterized
import numpy

from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache as function_cache_lib
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.util import compat
from tensorflow.python.util import nest


try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


def compiled_fn(fn=None, **tracing_options):
  # [FUNCTION DOC]
  # HELPER: shared test decorator used by nearly every test in this file to build a traceable callable.
  # Decorator factory that wraps a Python callable so each call is traced and executed through the
  # tracing compilation layer. If fn is omitted (decorator used with options) it returns a partial
  # application awaiting the function. Keyword options become tracing options; an optional input
  # signature entry is removed from the options and used to build the function type and default
  # values, which are stored into the options. Returns a wrapper callable that binds arguments
  # against the function type (applying defaults) and calls the compiled function; the wrapper
  # also carries a method for obtaining a concrete (traced) function without calling it.
  # Exceptions from signature handling (for example invalid signatures) propagate to the caller.
  """Decorator that compiles/calls wrapped function."""
  if fn is None:
    return functools.partial(compiled_fn, **tracing_options)

  signature = tracing_options.pop('input_signature', None)
  function_type, default_values = function_type_utils.make_function_type(
      fn, signature
  )
  tracing_options['polymorphic_type'] = function_type
  tracing_options['default_values'] = default_values

  def wrapped(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: inner wrapper produced by the compiled-function test decorator.
    # Binds the given positional and keyword arguments to the function type, filling in default
    # values, then calls the traced function with fresh tracing options built from the original
    # function and the captured options. Returns whatever the compiled function returns; binding
    # errors (wrong argument counts or names) propagate.
    bound_args = function_type.bind_with_defaults(args, kwargs, default_values)
    return tracing_compilation.call_function(
        bound_args.args,
        bound_args.kwargs,
        tracing_compilation.TracingOptions(fn, **tracing_options),
    )

  def trace(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: inner helper exposed as the concrete-function getter of the test decorator.
    # Traces the wrapped function for the given positional and keyword arguments (which may be
    # example values or type specifications) without executing it, using fresh tracing options.
    # Returns the resulting concrete function; no default binding is applied here.
    return tracing_compilation.trace_function(
        args,
        kwargs,
        tracing_compilation.TracingOptions(fn, **tracing_options),
    )

  wrapped.get_concrete_function = trace

  return wrapped


class TracingCompilationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test case covering the tracing compilation entry points: cache hits and retracing
  # rules, shape relaxation, input signatures, variable handling, function registration in graphs,
  # implementation swapping and executor selection. Uses the file-level compiled-function
  # decorator; no shared fixtures beyond the base test case. Runs in eager (v2) mode by default,
  # some tests also run in graph mode.

  @test_util.run_in_graph_and_eager_modes
  def testBackwardNoneGradient(self):
    # [FUNCTION DOC]
    # Test: a compiled function returns a float loss and an integer counter variable. Computing
    # the gradient of the loss with respect to a model variable (via a gradient tape in eager mode
    # or graph gradients otherwise) must work even though the integer output has no gradient.
    # Asserts the gradient value equals negative six for an input of seven.
    model = variables.Variable(1.0, name='model')
    count = variables.Variable(0)

    @compiled_fn
    def forward_pass(value):
      # [FUNCTION DOC]
      # Nested compiled function: increments the counter variable, computes half the mean squared
      # residual between the input and the model variable, and returns the loss and the counter.
      count.assign_add(1)
      residuals = value - model
      loss = 0.5 * math_ops.reduce_mean(math_ops.pow(residuals, 2))
      # Note: count is an integer, so its doutput will be None
      return loss, count

    def reduce_fn(x):
      # [FUNCTION DOC]
      # Nested helper: calls the compiled forward pass on the input and returns the gradient of
      # the loss with respect to the model variable together with the counter. Uses a gradient tape
      # when executing eagerly and graph-mode gradient computation otherwise.
      if context.executing_eagerly():
        with backprop.GradientTape() as t:
          loss, count = forward_pass(x)
        return t.gradient(loss, model), count
      loss, count = forward_pass(x)
      grad_only = gradients_impl.gradients(loss, model)
      return grad_only, count

    g, _ = reduce_fn(constant_op.constant([7.0]))

    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(nest.flatten(self.evaluate(g)), [-6.0])

  def testExternalControlDependency(self):
    # [FUNCTION DOC]
    # Test: in graph mode, a compiled function body opens a control-dependency scope on an
    # assign-add operation created outside the function. Calling the function must run that
    # external operation, so the variable ends at two. Asserts the variable value is 2.0.
    with ops.Graph().as_default(), self.test_session():
      v = variables.Variable(1.0)
      v.initializer.run()

      op = v.assign_add(1.0)

      @compiled_fn
      def f():
        # [FUNCTION DOC]
        # Nested compiled function with no inputs: returns the constant one inside a control
        # dependency on the externally created assign-add operation.
        with ops.control_dependencies([op]):
          return 1.0

      self.evaluate(f())
      self.assertAllEqual(self.evaluate(v), 2.0)

  def testInputShapeFunctionRelaxation(self):
    # [FUNCTION DOC]
    # Test: with retracing reduction enabled, a function is called with tensors of shape zero,
    # then one, then two elements. Asserts the first trace sees a fully known shape, the second
    # call causes a second trace with the shape relaxed to unknown, and the third call reuses the
    # relaxed trace (cache holds one, then two entries).
    unknown_dim = [False]

    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(reduce_retracing=True, function_cache=function_cache)
    def func(a):
      # [FUNCTION DOC]
      # Nested compiled function: records in the enclosing flag list whether the traced input has
      # an unknown first dimension, and returns the input plus one.
      if a._shape_tuple()[0] is None:
        unknown_dim[0] = True
      return a + 1

    func(constant_op.constant([]))
    self.assertFalse(unknown_dim[0])
    self.assertLen(function_cache, 1)

    func(constant_op.constant([1.0]))
    self.assertTrue(unknown_dim[0])
    self.assertLen(function_cache, 2)

    func(constant_op.constant([1.0, 2.0]))
    self.assertTrue(unknown_dim[0])

  def testNestedInputShapeFunctionRelaxation(self):
    # [FUNCTION DOC]
    # Test: shape relaxation with a nested list argument holding a scalar and a vector tensor,
    # plus a string first argument. Varying the vector length relaxes the shape (cache grows to
    # two, then stays); changing the string argument forces a new cache key, and afterwards the
    # relaxed type is traced again. Asserts the unknown-dimension flag and cache sizes (1, 2, 2,
    # 3, 4) at each step.
    unknown_dim = [False]

    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(reduce_retracing=True, function_cache=function_cache)
    def func(a_, b_=None):
      # [FUNCTION DOC]
      # Nested compiled function: the first argument is only used to select the cache entry. It
      # asserts the first element of the nested list is a scalar, records whether the second
      # element has an unknown first dimension, and returns the scalar plus one.
      del a_  # Only used to check which cache is used.
      self.assertEqual(b_[0]._shape_tuple(), ())
      if b_[1]._shape_tuple()[0] is None:
        unknown_dim[0] = True
      return b_[0] + 1

    a = 'hi'
    b0 = constant_op.constant(1.0)
    func(a, b_=[b0, constant_op.constant([])])
    self.assertFalse(unknown_dim[0])
    self.assertLen(function_cache, 1)

    func(a, b_=[b0, constant_op.constant([1.0])])
    self.assertTrue(unknown_dim[0])
    self.assertLen(function_cache, 2)

    func(a, b_=[b0, constant_op.constant([1.0, 1.0])])
    self.assertTrue(unknown_dim[0])
    self.assertLen(function_cache, 2)

    unknown_dim[0] = False

    # Now do the same except with a new a which is not a tensor; this should
    # change the cache key.
    a = 'bye'
    func(a, b_=[b0, constant_op.constant([])])
    self.assertFalse(unknown_dim[0])
    self.assertLen(function_cache, 3)

    # We relax the type traced previously.
    func(a, b_=[b0, constant_op.constant([1.0])])
    self.assertTrue(unknown_dim[0])
    self.assertLen(function_cache, 4)

  @test_util.run_v2_only
  def testGraphEagerIsolation(self):
    # [FUNCTION DOC]
    # Test (v2 only): a function that creates a variable and reads it is called eagerly through
    # the tracing call entry point, then again inside a fresh default graph. Asserts the eager
    # result equals one and that the call inside the graph yields a scalar-shaped tensor, showing
    # eager and graph contexts do not interfere.
    def f_py():
      # [FUNCTION DOC]
      # Nested Python function under test: creates a variable with value one, stores it on the
      # test instance, and returns its read value.
      self.v = variables.Variable(1.0)
      return self.v.read_value()

    f = lambda: tracing_compilation.call_function(  # pylint: disable=g-long-lambda
        tracing_options=tracing_compilation.TracingOptions(f_py, 'f')
    )
    self.assertAllEqual(f(), 1.0)

    with ops.Graph().as_default():
      self.assertEqual(f().shape, ())

  @test_util.run_v2_only
  def testCompilationNumpyArraysConvertedToTensors(self):
    # [FUNCTION DOC]
    # Test (v2 only): numpy array arguments, including mutable arrays and ndarray subclasses, are
    # converted to tensors when tracing. Calling with same-shaped arrays of different values must
    # not retrace (cache stays at one); scalar numpy arrays vs. tensors yield a second trace
    # shared across numpy and tensor inputs. Asserts the traced function sees tensors, values
    # follow the latest array contents, and the cache size is never above two.
    def f(x):
      # [FUNCTION DOC]
      # Nested function under test: asserts its argument is a tensor (not a numpy array) during
      # tracing and returns it unchanged.
      self.assertIsInstance(x, tensor_lib.Tensor)
      return x

    x = random_ops.random_uniform([2, 2]).numpy()
    function_cache = function_cache_lib.FunctionCache()
    defined = compiled_fn(f, function_cache=function_cache)
    defined(x)
    self.assertLen(function_cache, 1)

    x = random_ops.random_uniform([2, 2]).numpy()
    defined(x)
    # A NumPy array with different values but the same shape and dtype
    # shouldn't trigger another function definition.
    self.assertLen(function_cache, 1)

    np_ones = numpy.ones([], numpy.float32)
    np_zeros = numpy.zeros([], numpy.float32)
    tf_ones = array_ops.ones([])
    tf_zeros = array_ops.zeros([])

    # Test that the numpy array is properly an argument to the graph function.
    self.assertEqual(1.0, defined(np_ones).numpy())
    self.assertLen(function_cache, 2)
    self.assertEqual(0.0, defined(np_zeros).numpy())
    self.assertEqual(1.0, defined(tf_ones).numpy())
    self.assertEqual(0.0, defined(tf_zeros).numpy())
    self.assertLen(function_cache, 2)

    # Test that mutable inputs are supported.
    mutable = numpy.ones([], numpy.float32)
    self.assertEqual(1.0, defined(mutable).numpy())
    mutable.fill(0)
    self.assertEqual(0.0, defined(mutable).numpy())

    class MyNdarray(numpy.ndarray):
      # [CLASS DOC]
      # Trivial subclass of the numpy array type used to check that array subclasses are also
      # converted to tensors. Adds no state or behavior.
      pass

    # Test that the subclasses of ndarray are converted too.
    self.assertEqual(1.0, defined(np_ones.view(MyNdarray)).numpy())
    self.assertEqual(0.0, defined(np_zeros.view(MyNdarray)).numpy())

    # We should not have triggered any re-tracing of the python function.
    self.assertLen(function_cache, 2)

  @test_util.run_v2_only
  def testNumpyDtypeInputSupported(self):
    # [FUNCTION DOC]
    # Test (v2 only): a numpy scalar type object can be passed as a Python argument to a compiled
    # function and is treated as a static value. Calls with float32 and int32 types and different
    # numbers and asserts each result equals the number cast to that type.
    @compiled_fn
    def f(x, dtype):
      # [FUNCTION DOC]
      # Nested compiled function: converts the number to the given numpy type and returns it as
      # a constant tensor.
      return constant_op.constant(dtype(x))

    self.assertEqual(f(1, numpy.float32).numpy(), numpy.float32(1))
    self.assertEqual(f(2, numpy.float32).numpy(), numpy.float32(2))
    self.assertEqual(f(1, numpy.int32).numpy(), numpy.int32(1))
    self.assertEqual(f(2, numpy.int32).numpy(), numpy.int32(2))

  @test_util.run_v2_only
  def testCompilationNumpyArraysConvertedToTensorsInKwargs(self):
    # [FUNCTION DOC]
    # Test (v2 only): same as the positional numpy conversion test but the array is passed as a
    # keyword argument. Same-shaped arrays with different values do not retrace (cache stays at
    # one), and numpy or tensor scalar inputs yield the expected values (one and zero).
    def f(**kwargs):
      # [FUNCTION DOC]
      # Nested function under test: removes the entry named x from the keyword arguments, asserts
      # it is a tensor, and returns it.
      x = kwargs.pop('x')
      self.assertIsInstance(x, tensor_lib.Tensor)
      return x

    x = random_ops.random_uniform([2, 2]).numpy()
    function_cache = function_cache_lib.FunctionCache()
    defined = compiled_fn(f, function_cache=function_cache)
    defined(x=x)
    self.assertLen(function_cache, 1)

    x = random_ops.random_uniform([2, 2]).numpy()
    defined(x=x)
    # A NumPy array with different values but the same shape and dtype
    # shouldn't trigger another function definition.
    self.assertLen(function_cache, 1)

    # Test that the numpy array is properly an argument to the graph function.
    self.assertEqual(1.0, defined(x=numpy.ones([])).numpy())
    self.assertEqual(0.0, defined(x=numpy.zeros([])).numpy())
    self.assertEqual(1.0, defined(x=array_ops.ones([])).numpy())
    self.assertEqual(0.0, defined(x=array_ops.zeros([])).numpy())

  @test_util.run_v2_only
  def testFuncListAttr(self):
    # [FUNCTION DOC]
    # Test (v2 only): a compiled function builds a case operation whose branches are concrete
    # functions made from several callables (plain functions, a lambda, a function with a default
    # argument, and partial applications). Calls with branch indices zero to four and an
    # out-of-range index; asserts each returns the ones vector scaled by one to five and that the
    # out-of-range index runs the last (default) branch.
    @compiled_fn
    def test_function(val):
      # [FUNCTION DOC]
      # Nested compiled function: given a branch index tensor, defines five branch callables and
      # returns the output list of a case operation over their concrete functions.
      def fn1():
        # [FUNCTION DOC]
        # Branch returning a vector of ten ones.
        return array_ops.ones([10])

      fn2 = lambda: array_ops.ones([10]) * 2

      def fn3(x=3):
        # [FUNCTION DOC]
        # Branch returning a vector of ten ones scaled by its argument, which defaults to three.
        return array_ops.ones([10]) * x

      fn4 = functools.partial(fn3, x=4)
      fn5 = functools.partial(fn3, 5)

      return gen_functional_ops.case(
          val,
          [],
          [dtypes.float32],
          [
              compiled_fn(f).get_concrete_function()
              for f in (fn1, fn2, fn3, fn4, fn5)
          ],
      )

    ones = array_ops.ones([10])
    self.assertAllEqual([ones], test_function(0))
    self.assertAllEqual([ones * 2], test_function(1))
    self.assertAllEqual([ones * 3], test_function(2))
    self.assertAllEqual([ones * 4], test_function(3))
    self.assertAllEqual([ones * 5], test_function(4))
    self.assertAllEqual([ones * 5], test_function(22))  # default branch

  @test_util.enable_control_flow_v2
  def testVariableInLoopInFunction(self):
    # [FUNCTION DOC]
    # Test (control flow v2 enabled): a traced function runs a while loop whose body requests a
    # scoped variable. The condition is false so the body is only traced. Asserts the traced
    # function can be called and returns a scalar-shaped result.
    def test_function_py():
      # [FUNCTION DOC]
      # Nested function under test: runs a while loop with initial value zero, with a loop
      # condition that is always false and a body that fetches a scalar variable by name.
      def loop_test(_):
        # [FUNCTION DOC]
        # Loop condition that ignores its argument and always returns false.
        return False

      def loop_body(_):
        # [FUNCTION DOC]
        # Loop body that ignores its argument and returns the scalar variable named a, creating
        # it if needed.
        return variable_scope.get_variable('a', shape=())

      return while_loop.while_loop(loop_test, loop_body, [0.0])

    test_function = tracing_compilation.trace_function(
        tracing_options=tracing_compilation.TracingOptions(
            test_function_py, 'test_function'
        )
    )

    self.assertEqual(test_function().shape, [])

  @test_util.run_in_graph_and_eager_modes
  def testCompilationForcesResourceVariables(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): a function that creates a plain variable inside a traced
    # function is called once; asserts the created variable is a resource variable.
    def variable_creator():
      # [FUNCTION DOC]
      # Nested function under test: creates a float variable, keeps it on the test instance, and
      # returns its read value.
      self.v = variables.Variable(0.0)
      return self.v.read_value()

    defined = tracing_compilation.trace_function(
        tracing_options=tracing_compilation.TracingOptions(
            variable_creator, 'variable_creator'
        )
    )
    defined()  # Create the variable.
    self.assertIsInstance(self.v, resource_variable_ops.ResourceVariable)

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testFunctionWithResourcesOnDifferentDevices(self):
    # [FUNCTION DOC]
    # Test (GPU only, graph and eager modes): two resource variables with identical values live on
    # CPU and GPU. A function sums gathered elements from each. Asserts the compiled function's
    # result equals running the Python function directly.
    with ops.device('/cpu:0'):
      v_cpu = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

    with ops.device('/gpu:0'):
      v_gpu = resource_variable_ops.ResourceVariable([0.0, 1.0, 2.0])

    def sum_gather():
      # [FUNCTION DOC]
      # Nested function under test: returns the sums of elements at indices one and two of the
      # CPU variable and of the GPU variable, as a pair.
      cpu_result = math_ops.reduce_sum(array_ops.gather(v_cpu, [1, 2]))
      gpu_result = math_ops.reduce_sum(array_ops.gather(v_gpu, [1, 2]))
      return cpu_result, gpu_result

    defined = compiled_fn(sum_gather)
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
    expected = self.evaluate(sum_gather())
    self.assertAllEqual(expected, self.evaluate(defined()))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testCallOptionsMemory(self):
    # [FUNCTION DOC]
    # Test: under a decorator that fails if new Python objects leak in eager mode, resets the
    # context's function call options to nothing and calls a compiled function. Verifies no
    # objects are leaked. Side effect: modifies the global eager context's call options.
    @compiled_fn
    def model(x):
      # [FUNCTION DOC]
      # Nested compiled function: returns its input plus a constant one.
      return x + constant_op.constant(1.0)

    # This happens with a lot of option toggles, e.g. soft device placement
    context.context().function_call_options = None
    model(constant_op.constant(2.0))

  @test_util.run_in_graph_and_eager_modes
  def testVariablesPlacedOnOutsideDevice(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): a method decorated as compiled lazily creates a variable on
    # first call. The call is made inside a CPU device scope; asserts the variable's device string
    # mentions CPU.
    class _Obj(object):
      # [CLASS DOC]
      # HELPER: tiny test object holding an optional variable that is created lazily by a compiled
      # method.

      def __init__(self):
        # [FUNCTION DOC]
        # Initializes the object with its variable slot set to nothing.
        self.v = None

      @compiled_fn
      def f(self):
        # [FUNCTION DOC]
        # Compiled method: creates the variable (value one) if absent and returns it plus one.
        if self.v is None:
          self.v = variables.Variable(1.0)
        return self.v + 1.0

    has_device = _Obj()
    with ops.device('cpu:0'):
      has_device.f()
    self.assertIn('CPU', has_device.v.device)

  def testCacheObjectHashCollisions(self):
    # [FUNCTION DOC]
    # Test: two distinct objects whose hashes always collide are passed to a compiled function.
    # Asserts each distinct object produces its own cache entry (cache size one, then two), i.e.
    # the cache relies on more than the hash.
    class Foo:
      # [CLASS DOC]
      # Test class whose instances all have the same hash value, used to force hash collisions.

      def __hash__(self):
        # [FUNCTION DOC]
        # Returns a constant hash value (forty-two) for every instance.
        return 42

    def func(foo):
      # [FUNCTION DOC]
      # Nested function under test: returns a constant tensor holding the identity number of the
      # object, so the result differs per object.
      return constant_op.constant([id(foo)])

    function_cache = function_cache_lib.FunctionCache()
    defined = compiled_fn(func, function_cache=function_cache)
    foo_1 = Foo()
    defined(foo_1)
    self.assertLen(function_cache, 1)

    foo_2 = Foo()
    defined(foo_2)
    self.assertLen(function_cache, 2)

  def testCacheTensorDtypeCollision(self):
    # [FUNCTION DOC]
    # Test: a compiled function is called with a one-element matrix of complex64 and then of
    # complex128. Asserts the differing dtypes produce separate cache entries (one, then two).
    def func(t):
      # [FUNCTION DOC]
      # Nested function under test: returns the input added to itself.
      return t + t

    function_cache = function_cache_lib.FunctionCache()
    defined = compiled_fn(func, function_cache=function_cache)
    t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(function_cache, 1)

    t = constant_op.constant([[1.0]], dtype=dtypes.complex128)
    defined(t)
    self.assertLen(function_cache, 2)

  def testCacheTensorShapeCollision(self):
    # [FUNCTION DOC]
    # Test: a compiled function is called with a complex64 tensor of shape one-by-one and then of
    # shape one. Asserts the differing shapes produce separate cache entries (one, then two).
    def func(t):
      # [FUNCTION DOC]
      # Nested function under test: returns the input added to itself.
      return t + t

    function_cache = function_cache_lib.FunctionCache()
    defined = compiled_fn(func, function_cache=function_cache)
    t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(function_cache, 1)

    t = constant_op.constant([1.0], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(function_cache, 2)

  def testCacheTensorShapeDtypeCollision(self):
    # [FUNCTION DOC]
    # Test: a compiled function is called with a complex64 matrix and then with a complex128
    # vector (both shape and dtype differ). Asserts two separate cache entries result.
    def func(t):
      # [FUNCTION DOC]
      # Nested function under test: returns the input added to itself.
      return t + t

    function_cache = function_cache_lib.FunctionCache()
    defined = compiled_fn(func, function_cache=function_cache)
    t = constant_op.constant([[1.0]], dtype=dtypes.complex64)
    defined(t)
    self.assertLen(function_cache, 1)

    t = constant_op.constant([1.0], dtype=dtypes.complex128)
    defined(t)
    self.assertLen(function_cache, 2)

  def testCacheTensorUnknownShapesCollisionRelaxedShapes(self):
    # [FUNCTION DOC]
    # Test (graph mode): with retracing reduction on, placeholders of shape scalar, then one
    # element, then two elements, then a constant of three elements are passed. Asserts the cache
    # has one, two, two, two entries: the third call relaxes the shape to unknown length and the
    # constant matches the relaxed shape.
    def func(t):
      # [FUNCTION DOC]
      # Nested function under test: returns the input added to itself.
      return t + t

    with context.graph_mode(), self.cached_session():
      function_cache = function_cache_lib.FunctionCache()
      defined = compiled_fn(
          func, reduce_retracing=True, function_cache=function_cache
      )

      p = array_ops.placeholder(dtype=dtypes.float32, shape=[])
      defined(p)
      self.assertLen(function_cache, 1)

      p = array_ops.placeholder(dtype=dtypes.float32, shape=[1])
      defined(p)
      self.assertLen(function_cache, 2)

      p = array_ops.placeholder(dtype=dtypes.float32, shape=[2])
      defined(p)
      # Gradual shape relaxation is performed; and the common shape between
      # [1] and [2] is one containing unknown dimensions.
      self.assertLen(function_cache, 2)

      t = constant_op.constant([1.0, 1.0, 1.0], dtype=dtypes.float32)
      defined(t)
      # Shape (3,) matches the relaxed shape TensorShape([None])
      self.assertLen(function_cache, 2)

  def testPythonFunctionWithDefaultArgs(self):
    # [FUNCTION DOC]
    # Test: a function with two defaulted parameters is called with different mixes of positional
    # and keyword arguments. Asserts calls that bind to the same effective arguments (including
    # defaults and reordered keywords) share a cache entry, and distinct effective values add one;
    # cache sizes are 1, 2, 2, 3, 3, 3.
    def func(foo, bar=1, baz=2):
      # [FUNCTION DOC]
      # Nested function under test: discards all three arguments and returns nothing.
      del foo
      del bar
      del baz
      return

    function_cache = function_cache_lib.FunctionCache()
    defined = compiled_fn(func, function_cache=function_cache)
    defined(0, baz=20)
    self.assertLen(function_cache, 1)

    defined(1)  # bar=1, baz=2
    self.assertLen(function_cache, 2)

    # This matches the previous call.
    defined(foo=1)
    self.assertLen(function_cache, 2)

    defined(1, 2, 3)
    self.assertLen(function_cache, 3)

    # This matches the previous call.
    defined(1, bar=2, baz=3)
    self.assertLen(function_cache, 3)

    # This matches the previous call.
    defined(1, baz=3, bar=2)
    self.assertLen(function_cache, 3)

  @test_util.run_v2_only
  def testFunctoolsPartialUnwrappedCorrectly(self):
    # [FUNCTION DOC]
    # Test (v2 only): a partial application of a three-parameter function (first argument bound
    # positionally, third bound by keyword) is compiled and called with one argument. Asserts the
    # compiled results match the plain Python call.
    def full_function(a, b, c=3):
      # [FUNCTION DOC]
      # Nested function: returns its three arguments as a tuple.
      return a, b, c

    partial = functools.partial(full_function, 1, c=4)
    a, b, c = partial(2)

    defined = compiled_fn(partial)
    func_a, func_b, func_c = defined(2)
    self.assertEqual(func_a.numpy(), a)
    self.assertEqual(func_b.numpy(), b)
    self.assertEqual(func_c.numpy(), c)

  def testInputSignatureWithMatchingInputs(self):
    # [FUNCTION DOC]
    # Test: a function compiled with an input signature of a length-two float vector is called
    # with matching input and concrete functions are requested with and without arguments. Asserts
    # a single cache entry. A second function with a partially unknown shape signature accepts
    # inputs with different second dimensions without creating a new function.
    def foo(a):
      # [FUNCTION DOC]
      # Nested function under test: asserts its input has shape of two elements and returns it.
      self.assertEqual(a.shape, (2,))
      return a

    function_cache = function_cache_lib.FunctionCache()
    signature = [tensor_lib.TensorSpec(shape=(2,), dtype=dtypes.float32)]
    defined = compiled_fn(
        foo, input_signature=signature, function_cache=function_cache
    )
    a = array_ops.ones([2])
    self.assertAllEqual(a, defined(a))
    self.assertLen(function_cache, 1)
    self.assertAllEqual(a, defined.get_concrete_function()(a))
    self.assertAllEqual(a, defined.get_concrete_function(a)(a))
    self.assertAllEqual(
        a,
        defined.get_concrete_function(
            tensor_lib.TensorSpec((2,), dtype=dtypes.float32)
        )(a),
    )
    self.assertLen(function_cache, 1)

    def bar(a):
      # [FUNCTION DOC]
      # Nested function under test: asserts its input has a known first dimension of two and an
      # unknown second dimension, and returns it.
      self.assertEqual(a._shape_tuple(), (2, None))
      return a

    signature = [tensor_lib.TensorSpec((2, None), dtypes.float32)]
    defined = compiled_fn(bar, input_signature=signature)
    a = array_ops.ones([2, 1])
    out = defined(a)
    self.assertLen(function_cache, 1)
    self.assertAllEqual(out, a)

    # Changing the second dimension shouldn't create a new function.
    b = array_ops.ones([2, 3])
    out = defined(b)
    self.assertLen(function_cache, 1)
    self.assertAllEqual(out, b)

  def testInputSignatureWithDictInPositionalArgs(self):
    # [FUNCTION DOC]
    # Test: a function taking variadic positional and keyword arguments is called with a keyword
    # argument twice (one cache entry) and then with a dictionary as the second positional
    # argument. Asserts the dictionary form is treated as a different call signature (two entries).
    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(function_cache=function_cache)
    def f(*_args, **_kwargs):
      # [FUNCTION DOC]
      # Nested compiled function accepting any arguments and returning nothing.
      return None

    f(1, x=2)
    self.assertLen(function_cache, 1)
    f(1, x=2)
    self.assertLen(function_cache, 1)
    f(1, {'x': 2})
    self.assertLen(function_cache, 2)

  def testInputSignatureWithCompatibleInputs(self):
    # [FUNCTION DOC]
    # Test: with a rank-two float signature of unknown dimensions, nested lists and numpy arrays
    # convertible to that type are accepted and give the right shapes. Asserts a rank-one list
    # and a list of strings each raise a type error.
    rank2_spec = tensor_lib.TensorSpec(
        shape=(None, None), dtype=dtypes.float32
    )

    @compiled_fn(input_signature=[rank2_spec])
    def func(a):
      # [FUNCTION DOC]
      # Nested compiled function: asserts the traced input has fully unknown dimensions and
      # returns its dynamic shape.
      self.assertEqual([None, None], a.shape.as_list())
      return array_ops.shape(a)

    self.assertAllEqual([3, 1], func([[0], [1.0], [1]]))
    self.assertAllEqual([2, 2], func(numpy.array([[1, 1], [2, 2]])))

    with self.assertRaises(TypeError):
      func([0.0, 1.0, 2.0])  # Wrong shape.

    with self.assertRaises(TypeError):
      func([['wrong dtype']])

  @test_util.run_v2_only
  def testNestedInputSignatures(self):
    # [FUNCTION DOC]
    # Test (v2 only): a function compiled with a nested signature (a list of two partially
    # unknown matrices, then a vector) is called with matching tensors, with differently sized
    # unspecified dimensions, and with plain nested lists. Asserts output structure and values
    # match the inputs and that only one cache entry is ever created.
    def expected_foo(a, b):
      # [FUNCTION DOC]
      # Nested reference function: returns its two arguments as a list, used to compute the
      # expected structure.
      return [a, b]

    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(
        input_signature=[
            [tensor_lib.TensorSpec((2, None), dtypes.float32)] * 2,
            tensor_lib.TensorSpec((1,), dtypes.float32),
        ],
        function_cache=function_cache,
    )
    def foo(a, b):
      # [FUNCTION DOC]
      # Nested compiled function: asserts the first argument is a list of two tensors with shape
      # of two by unknown and the second is a length-one vector, then returns both as a list.
      self.assertEqual(a[0]._shape_tuple(), (2, None))
      self.assertEqual(a[1]._shape_tuple(), (2, None))
      self.assertEqual(b._shape_tuple(), (1,))
      return [a, b]

    a = array_ops.ones([2, 1])
    b = array_ops.ones([1])
    expected = expected_foo([a, a], b)
    out = foo([a, a], b)
    self.assertLen(function_cache, 1)
    nest.assert_same_structure(out, expected)
    self.assertAllEqual(out[0][0], a)
    self.assertAllEqual(out[0][1], a)
    self.assertAllEqual(out[1], b)

    # Changing the unspecified dimensions shouldn't create a new function.
    a = array_ops.ones([2, 3])
    b = array_ops.ones([2, 5])
    c = array_ops.ones([1])
    expected = expected_foo([a, b], c)
    out = foo([a, b], c)
    self.assertLen(function_cache, 1)
    nest.assert_same_structure(out, expected)
    self.assertAllEqual(out[0][0], a)
    self.assertAllEqual(out[0][1], b)
    self.assertAllEqual(out[1], c)

    # Passing compatible inputs should work.
    a = a.numpy().tolist()
    b = b.numpy().tolist()
    c = c.numpy().tolist()
    out = foo([a, b], c)
    self.assertLen(function_cache, 1)
    nest.assert_same_structure(out, expected)
    self.assertAllEqual(out[0][0], a)
    self.assertAllEqual(out[0][1], b)
    self.assertAllEqual(out[1], c)

  @test_util.run_v2_only
  def testNestedInputSignaturesWithDict(self):
    # [FUNCTION DOC]
    # Test (v2 only): a function compiled with a signature consisting of a mapping of three named
    # tensor specs is called with matching tensors and then with nested Python lists. Asserts the
    # output mapping has the same structure and values as the input.
    def expected_bar(a):
      # [FUNCTION DOC]
      # Nested reference function: returns its argument unchanged, used for the expected value.
      return a

    @compiled_fn(
        input_signature=[{
            'a': tensor_lib.TensorSpec((2, None), dtypes.float32),
            'b': tensor_lib.TensorSpec((2, None), dtypes.float32),
            'c': tensor_lib.TensorSpec((1,), dtypes.float32),
        }]
    )
    def bar(a):
      # [FUNCTION DOC]
      # Nested compiled function: asserts the mapping's entries a and b have shape two by unknown
      # and c has shape one, then returns the mapping unchanged.
      self.assertEqual(a['a']._shape_tuple(), (2, None))
      self.assertEqual(a['b']._shape_tuple(), (2, None))
      self.assertEqual(a['c']._shape_tuple(), (1,))
      return a

    a = array_ops.ones([2, 3])
    b = array_ops.ones([1])
    inputs = {'a': a, 'b': a, 'c': b}
    expected = expected_bar(inputs)
    out = bar(inputs)
    nest.assert_same_structure(out, expected)
    self.assertAllEqual(out['a'], expected['a'])
    self.assertAllEqual(out['b'], expected['b'])
    self.assertAllEqual(out['c'], expected['c'])

    # Passing compatible inputs should work.
    a = a.numpy().tolist()
    b = b.numpy().tolist()
    inputs = {'a': a, 'b': a, 'c': b}
    out = bar(inputs)
    nest.assert_same_structure(out, expected)
    self.assertAllEqual(out['a'], expected['a'])
    self.assertAllEqual(out['b'], expected['b'])
    self.assertAllEqual(out['c'], expected['c'])

  def testInputSignatureMustBeSequenceOfTensorSpecs(self):
    # [FUNCTION DOC]
    # Test: supplying a mapping instead of a list or tuple as the input signature must raise a
    # type error whose message says the signature must be a tuple or a list.
    def foo(a, b):
      # [FUNCTION DOC]
      # Nested function whose arguments are discarded; returns nothing.
      del a
      del b

    # Signatures must be either lists or tuples on their outermost levels.
    signature = {'t1': tensor_lib.TensorSpec([], dtypes.float32)}
    with self.assertRaisesRegex(
        TypeError, 'input_signature must be either a tuple or a list.*'
    ):
      compiled_fn(foo, input_signature=signature)

  def testInputsIncompatibleWithNestedSignatureRaisesError(self):
    # [FUNCTION DOC]
    # Test: with a signature of two lists of two length-one vectors, calls with a first list of
    # three elements, or a second list of three elements, must raise type errors; the correctly
    # sized call succeeds.
    def foo(a, b):
      # [FUNCTION DOC]
      # Nested function: returns its two arguments as a list.
      return [a, b]

    signature = [
        [tensor_lib.TensorSpec((1,), dtypes.float32)] * 2,
        [tensor_lib.TensorSpec((1,), dtypes.float32)] * 2,
    ]
    defined = compiled_fn(foo, input_signature=signature)
    a = array_ops.ones([1])

    with self.assertRaises(TypeError):
      defined([a, a, a], [a])

    with self.assertRaises(TypeError):
      defined([a], [a, a, a])
    defined([a, a], [a, a])

  @test_util.run_v2_only
  def testUnderspecifiedInputSignature(self):
    # [FUNCTION DOC]
    # Test (v2 only): a signature covers only the first parameter of a function that has a
    # defaulted flag. Calling with the default works and returns the input; passing the flag
    # explicitly raises a value error because the signature does not cover it.
    @compiled_fn(
        input_signature=[
            tensor_lib.TensorSpec([], dtypes.float32),
        ]
    )
    def foo(a, training=True):
      # [FUNCTION DOC]
      # Nested compiled function: returns the input if the flag is true, otherwise returns the
      # input negated.
      if training:
        return a
      else:
        return -1.0 * a

    x = constant_op.constant(1.0)
    with self.assertRaises(ValueError):
      foo(x, training=False)

    self.assertAllEqual(x.numpy(), foo(x).numpy())

  @test_util.run_v2_only
  def testInputSignatureWithPartialFunction(self):
    # [FUNCTION DOC]
    # Test (v2 only): a partial application with bound first and third arguments is compiled with
    # a signature of one scalar float. Asserts the result of calling with a scalar tensor equals
    # the plain Python result.
    def full_function(a, b, c=3.0):
      # [FUNCTION DOC]
      # Nested function: returns its three arguments as a tuple.
      return a, b, c

    partial = functools.partial(full_function, 1, c=4)
    a, b, c = partial(2.0)
    signature = [tensor_lib.TensorSpec([], dtypes.float32)]
    defined = compiled_fn(partial, input_signature=signature)
    x = constant_op.constant(2.0)
    func_a, func_b, func_c = defined(x)
    self.assertEqual(func_a.numpy(), a)
    self.assertEqual(func_b.numpy(), b)
    self.assertEqual(func_c.numpy(), c)

  @test_util.run_v2_only
  def testInputSignatureWithKeywordPositionalArgs(self):
    # [FUNCTION DOC]
    # Test (v2 only): a function with a float and an int64 signature is called positionally,
    # entirely by keyword in either order, and mixed. Asserts all calls give the same outputs (one
    # and two) and reuse one cache entry.
    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(
        input_signature=[
            tensor_lib.TensorSpec([], dtypes.float32),
            tensor_lib.TensorSpec([], dtypes.int64),
        ],
        function_cache=function_cache,
    )
    def foo(flt, integer):
      # [FUNCTION DOC]
      # Nested compiled function: returns its float and integer arguments as a pair.
      return flt, integer

    flt = constant_op.constant(1.0)
    integer = constant_op.constant(2, dtypes.int64)

    out1, out2 = foo(flt, integer)
    self.assertLen(function_cache, 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

    out1, out2 = foo(flt=flt, integer=integer)
    self.assertLen(function_cache, 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

    out1, out2 = foo(integer=integer, flt=flt)
    self.assertLen(function_cache, 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

    out1, out2 = foo(flt, integer=integer)
    self.assertLen(function_cache, 1)
    self.assertEqual(out1.numpy(), 1.0)
    self.assertEqual(out2.numpy(), 2)

  @test_util.run_v2_only
  def testInputSignatureWithKeywordArgs(self):
    # [FUNCTION DOC]
    # Test (v2 only): a function that also accepts arbitrary keyword arguments is compiled with
    # a two-element signature; its concrete function is called with a float and an int scalar.
    # Asserts the result equals five and five.
    def foo(a, b, **kwargs):
      # [FUNCTION DOC]
      # Nested function: discards extra keyword arguments and returns the first two as a pair.
      del kwargs
      return a, b

    x = compiled_fn(
        foo,
        input_signature=[
            tensor_lib.TensorSpec([], dtypes.float32),
            tensor_lib.TensorSpec([], dtypes.int32),
        ],
    ).get_concrete_function()
    result = x(constant_op.constant(5.0), constant_op.constant(5))
    self.assertAllEqual(result, [5.0, 5])

  def testInputSignatureWithCompositeTensors(self):
    # [FUNCTION DOC]
    # Test: a function compiled with a ragged tensor signature (three rows, unknown row lengths)
    # is called with ragged inputs. Matching row count with different lengths reuses one cache
    # entry and returns equal values and row splits; a different row count or dtype raises a type
    # error; a different rank raises a value error.
    def f(rt):
      # [FUNCTION DOC]
      # Nested function under test: asserts the ragged input has unknown values length and four
      # row split entries, and returns it.
      self.assertEqual(rt.values.shape.as_list(), [None])
      self.assertEqual(rt.row_splits.shape.as_list(), [4])
      return rt

    signature = [
        ragged_tensor.RaggedTensorSpec(shape=[3, None], dtype=dtypes.int32)
    ]
    function_cache = function_cache_lib.FunctionCache()
    defined = compiled_fn(
        f, input_signature=signature, function_cache=function_cache
    )
    rt1 = ragged_factory_ops.constant([[1], [], [2, 3, 4]])
    out1 = defined(rt1)
    self.assertLen(function_cache, 1)
    self.assertAllEqual(out1.values, rt1.values)
    self.assertAllEqual(out1.row_splits, rt1.row_splits)

    # Changing the row lengths shouldn't create a new function.
    rt2 = ragged_factory_ops.constant([[1, 2], [3, 4], [5]])
    out2 = defined(rt2)
    self.assertLen(function_cache, 1)
    self.assertAllEqual(out2.values, rt2.values)
    self.assertAllEqual(out2.row_splits, rt2.row_splits)

    # Different number of rows
    rt3 = ragged_factory_ops.constant([[1, 2], [3, 4], [5], [6]])
    with self.assertRaises(TypeError):
      defined(rt3)

    # Different dtype
    rt4 = ragged_factory_ops.constant([[1.0, 2.0], [], [3.0]])
    with self.assertRaises(TypeError):
      defined(rt4)

    # Different rank
    rt5 = ragged_factory_ops.constant([[[1]], [[2]], [[3]]])
    with self.assertRaises(ValueError):
      defined(rt5)

  @test_util.run_v2_only
  def testInputSignatureWithKeywordOnlyArgs(self):
    # [FUNCTION DOC]
    # Test (v2 only): a function with a defaulted positional and a defaulted keyword-only
    # parameter is compiled with a two-tensor signature, also via partial applications binding the
    # extras (including a tensor). Asserts sums of ten or eleven. Then saves a module with the
    # concrete function to a fixed temporary path, reloads it and calls the serving signature
    # with named tensors, asserting eleven. Side effect: writes a saved model under a temp path.
    def f(a, b, c=3, *, d=4):
      # [FUNCTION DOC]
      # Nested function under test: asserts a and b are tensors, c is an int, d is an int or
      # tensor, and returns the sum of all four.
      self.assertIsInstance(a, tensor_lib.Tensor)
      self.assertIsInstance(b, tensor_lib.Tensor)
      self.assertIsInstance(c, int)
      self.assertIsInstance(d, (int, tensor_lib.Tensor))
      return a + b + c + d

    signature = [
        tensor_lib.TensorSpec(shape=[], dtype=dtypes.int32),
        tensor_lib.TensorSpec(shape=[], dtype=dtypes.int32),
    ]
    defined = compiled_fn(f, input_signature=signature)
    self.assertEqual(defined(1, 2).numpy(), 10)

    defined = compiled_fn(functools.partial(f, c=4), input_signature=signature)
    self.assertEqual(defined(1, 2).numpy(), 11)

    defined = compiled_fn(functools.partial(f, d=5), input_signature=signature)
    self.assertEqual(defined(1, 2).numpy(), 11)

    defined = compiled_fn(
        functools.partial(f, d=array_ops.constant(5)), input_signature=signature
    )
    self.assertEqual(defined(1, 2).numpy(), 11)

    mod = module.Module()
    save(mod, '/tmp/kwonlyf', defined.get_concrete_function(*signature))
    loaded = load('/tmp/kwonlyf')
    result = loaded.signatures['serving_default'](
        a=array_ops.constant(1),
        b=array_ops.constant(2),
        d=array_ops.constant(5),
    )
    self.assertEqual(result['output_0'].numpy(), 11)

  def testInputSignatureWithKeywordOnlyArgsNoDefaults(self):
    # [FUNCTION DOC]
    # Test: compiling a function (or lambda) that has a keyword-only parameter without a default
    # while an input signature is given must raise a type error saying the parameter must have a
    # default value.
    signature = [
        tensor_lib.TensorSpec(shape=[], dtype=dtypes.int32),
        tensor_lib.TensorSpec(shape=[], dtype=dtypes.int32),
    ]

    def test_func(a, *, b):
      # [FUNCTION DOC]
      # Nested function with a required keyword-only parameter: returns the sum of its arguments.
      return a + b

    with self.assertRaisesRegex(
        TypeError,
        (
            'Since input_signature is defined, keyword-only parameter `b` must'
            ' have a default value'
        ),
    ):
      compiled_fn(test_func, input_signature=signature)

    test_func_lambda = lambda a, *, b: a + b
    with self.assertRaisesRegex(
        TypeError,
        (
            'Since input_signature is defined, keyword-only parameter `b` must'
            ' have a default value'
        ),
    ):
      compiled_fn(test_func_lambda, input_signature=signature)

  def testTensorKeywordArguments(self):
    # [FUNCTION DOC]
    # Test: tensor arguments passed positionally, by keyword, reordered, or mixed share one cache
    # entry; swapping which tensor goes to which parameter (different shapes) creates a second
    # entry. Asserts cache sizes and that outputs are the second argument (vector or scalar).
    def foo(a, b):
      # [FUNCTION DOC]
      # Nested function under test: ignores the first argument and returns the second.
      del a
      return b

    function_cache = function_cache_lib.FunctionCache()
    defined = compiled_fn(foo, function_cache=function_cache)
    a = constant_op.constant(2.0)
    b = constant_op.constant([1.0, 2.0])
    one = defined(a, b)
    self.assertLen(function_cache, 1)

    two = defined(a=a, b=b)
    self.assertLen(function_cache, 1)

    three = defined(b=b, a=a)
    self.assertLen(function_cache, 1)

    four = defined(a, b=b)
    self.assertLen(function_cache, 1)

    # The next call corresponds to a new input signature, hence
    # we expect another function to be defined.
    five = defined(b, a)
    self.assertLen(function_cache, 2)

    six = defined(a=b, b=a)
    self.assertLen(function_cache, 2)

    seven = defined(b=a, a=b)
    self.assertLen(function_cache, 2)

    self.assertAllEqual(one, [1.0, 2.0])
    self.assertAllEqual(two, [1.0, 2.0])
    self.assertAllEqual(three, [1.0, 2.0])
    self.assertAllEqual(four, [1.0, 2.0])
    self.assertAllEqual(five, 2.0)
    self.assertAllEqual(six, 2.0)
    self.assertAllEqual(seven, 2.0)

  def testFunctionWithInvalidAttribute(self):
    # [FUNCTION DOC]
    # Test: tracing with an attribute whose name is not supported (an experimental-prefixed name)
    # must raise a value error stating tracing compilation does not support that attribute.
    def add(x, y):
      # [FUNCTION DOC]
      # Nested function: returns the elementwise sum of its two inputs.
      return math_ops.add(x, y)

    with self.assertRaisesRegex(
        ValueError,
        'Tracing compilation does not support `experimental_1` as an'
        ' attribute.',
    ):
      tracing_compilation.trace_function(
          (1, 2),
          tracing_options=tracing_compilation.TracingOptions(
              add, 'add', attributes={'experimental_1': 'value1'}
          ),
      )

  def testRegisterFunction(self):
    # [FUNCTION DOC]
    # Test (graph mode): concrete functions for a named matmul and add are registered in the
    # default graph together with their gradient functions. Asserts six functions are present
    # (inference, forward, backward for each) with names matching the expected patterns and
    # cross-linked forward and backward attributes. Calling the functions then evaluates to the
    # expected matrix product and doubled matrix and must not add further functions.
    @compiled_fn(name='add', function_cache=function_cache_lib.FunctionCache())
    def add(x, y):
      # [FUNCTION DOC]
      # Nested compiled function: elementwise sum of the inputs.
      return math_ops.add(x, y)

    def matmul(x, y):
      # [FUNCTION DOC]
      # Nested function: matrix product of the inputs.
      return math_ops.matmul(x, y)

    defun_matmul = compiled_fn(
        matmul, name='matmul', function_cache=function_cache_lib.FunctionCache()
    )

    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        concrete_func_matmul = defun_matmul.get_concrete_function(t, t)
        concrete_func_matmul.add_to_graph()
        concrete_func_matmul.add_gradient_functions_to_graph()

        concrete_func_add = add.get_concrete_function(t, t)
        concrete_func_add.add_to_graph()
        concrete_func_add.add_gradient_functions_to_graph()

        graph = ops.get_default_graph()
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 6)
        # two sets of functions, each of them are (inference, forward, backward)
        functions = list(graph._functions.values())
        captured_function_names = [
            f.cached_definition.signature.name for f in functions
        ]
        expected_func_name_regex = [
            '.*inference.*matmul.*',
            '.*forward.*matmul.*',
            '.*inference.*backward.*matmul.*',
            '.*inference.*add.*',
            '.*forward.*add.*',
            '.*inference.*backward.*add.*',
        ]
        for i in range(len(functions)):
          self.assertRegex(
              captured_function_names[i], expected_func_name_regex[i]
          )

        # Check the forward and backward function has the correct attributes.
        self.assertEqual(
            functions[1].cached_definition.attr['backward_function_name'].s,
            functions[2].name,
        )
        self.assertEqual(
            functions[2].cached_definition.attr['forward_function_name'].s,
            functions[1].name,
        )

        self.assertEqual(
            functions[4].cached_definition.attr['backward_function_name'].s,
            functions[5].name,
        )
        self.assertEqual(
            functions[5].cached_definition.attr['forward_function_name'].s,
            functions[4].name,
        )

        sq = defun_matmul(t, t)
        double = add(t, t)
        self.assertAllEqual(sq.eval().reshape(-1), [7, 10, 15, 22])
        self.assertAllEqual(double.eval().reshape(-1), [2, 4, 6, 8])
        # Make sure the pre registered function is used, and no other function
        # is added.
        self.assertLen(graph._functions, 6)
        functions = list(graph._functions.values())
        for i in range(len(functions)):
          self.assertEqual(
              captured_function_names[i],
              functions[i].cached_definition.signature.name,
          )

  @test_util.run_v2_only
  def testRegisterConcreteFunction(self):
    # [FUNCTION DOC]
    # Test (v2 only): a compiled add function is traced eagerly and its concrete function is
    # called by a second compiled function (composite). In graph mode, registering the composite
    # and its gradient functions adds six functions in the expected name order; calling the
    # composite and add functions gives the expected values and adds no more functions.
    @compiled_fn(
        name='py_add', function_cache=function_cache_lib.FunctionCache()
    )
    def py_add(x, y):
      # [FUNCTION DOC]
      # Nested compiled function: elementwise sum of the inputs.
      return math_ops.add(x, y)

    py_add(array_ops.ones([]), array_ops.ones([]))
    add = py_add.get_concrete_function(
        tensor_lib.TensorSpec(None, dtypes.float32),
        tensor_lib.TensorSpec(None, dtypes.float32),
    )

    @compiled_fn(
        name='py_composite', function_cache=function_cache_lib.FunctionCache()
    )
    def py_composite(x, y):
      # [FUNCTION DOC]
      # Nested compiled function: returns the first input and the result of calling the
      # previously traced add concrete function on both inputs.
      return x, add(x, y)

    py_composite(array_ops.ones([]), array_ops.ones([]))
    composite = py_composite.get_concrete_function(
        tensor_lib.TensorSpec(None, dtypes.float32),
        tensor_lib.TensorSpec(None, dtypes.float32),
    )

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
        for expected, found in zip(
            expected_func_name_regex, captured_function_names
        ):
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
    # Test: eager tensors captured by a compiled function become constants if small (at or below
    # the capture size threshold) and placeholders if large; eager variables are captured as
    # placeholders. For each captured value, traces a function and asserts exactly one internal
    # capture exists with the expected operation type. Reads a private threshold constant.
    with context.eager_mode():
      large_tensor = array_ops.ones(shape=(256,))
      self.assertGreater(256, capture_container._EAGER_CONST_THRESHOLD)

      small_tensor = array_ops.ones(shape=(4,))
      self.assertLessEqual(4, capture_container._EAGER_CONST_THRESHOLD)

      v = resource_variable_ops.ResourceVariable(0.0)

    for captured, op_type in [
        (large_tensor, 'Placeholder'),
        (small_tensor, 'Const'),
        (v, 'Placeholder'),
    ]:

      @compiled_fn
      def test_fn():
        # [FUNCTION DOC]
        # Nested compiled function with no inputs: returns the loop-captured value plus one.
        return captured + 1  # pylint: disable=cell-var-from-loop

      g = test_fn.get_concrete_function().graph
      internal_captures = g.internal_captures
      self.assertLen(internal_captures, 1)
      self.assertEqual(internal_captures[0].op.type, op_type)

  def testRegisterFunctionWithInputSignature(self):
    # [FUNCTION DOC]
    # Test (graph mode): a matmul compiled with a fixed two-by-two signature is registered with
    # its gradient functions (three functions). Requesting the concrete function again with no
    # arguments and registering it again must not add functions.
    def matmul(x, y):
      # [FUNCTION DOC]
      # Nested function: matrix product of the inputs.
      return math_ops.matmul(x, y)

    defun_matmul = compiled_fn(
        matmul,
        input_signature=[
            tensor_lib.TensorSpec(shape=(2, 2), dtype=dtypes.float32),
            tensor_lib.TensorSpec(shape=(2, 2), dtype=dtypes.float32),
        ],
        function_cache=function_cache_lib.FunctionCache(),
    )
    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        concrete_func = defun_matmul.get_concrete_function(t, t)
        concrete_func.add_to_graph()
        concrete_func.add_gradient_functions_to_graph()

        graph = ops.get_default_graph()
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 3)

        # Test register function with cache, note inputs are ignored.
        concrete_func = defun_matmul.get_concrete_function()
        concrete_func.add_to_graph()
        concrete_func.add_gradient_functions_to_graph()
        graph = ops.get_default_graph()
        self.assertLen(graph._functions, 3)

  def testRegisterFunctionWithCache(self):
    # [FUNCTION DOC]
    # Test (graph mode): the same compiled matmul is traced for two different tensors of the same
    # type and both concrete functions are registered with gradients. Asserts only three functions
    # exist in the graph since both share one trace.
    def matmul(x, y):
      # [FUNCTION DOC]
      # Nested function: matrix product of the inputs.
      return math_ops.matmul(x, y)

    defun_matmul = compiled_fn(
        matmul, function_cache=function_cache_lib.FunctionCache()
    )

    with context.graph_mode(), self.cached_session():
      with ops.get_default_graph().as_default():
        t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
        t2 = constant_op.constant([[2.0, 3.0], [4.0, 5.0]])
        concrete_func_t = defun_matmul.get_concrete_function(t, t)
        concrete_func_t.add_to_graph()
        concrete_func_t.add_gradient_functions_to_graph()

        concrete_func_t2 = defun_matmul.get_concrete_function(t2, t2)
        concrete_func_t2.add_to_graph()
        concrete_func_t2.add_gradient_functions_to_graph()

        graph = ops.get_default_graph()
        # Only one function is registered since the input param are in same type
        # pylint: disable=protected-access
        self.assertLen(graph._functions, 3)

  @test_util.run_v2_only
  def testCallingFunctionWithDifferentVariables(self):
    # [FUNCTION DOC]
    # Test (v2 only): a concrete function traced for one variable takes it as an input (one input,
    # no captures); calling it twice increments to one then two. A second function that ignores
    # its variable argument gives one when called with the traced variable or a different one.
    @compiled_fn
    def foo(v):
      # [FUNCTION DOC]
      # Nested compiled function: adds one to the variable and returns its new value.
      v.assign_add(1.0)
      return v.read_value()

    v = resource_variable_ops.ResourceVariable(0.0)
    graph_function = foo.get_concrete_function(v)
    self.assertLen(graph_function.inputs, 1)
    self.assertEmpty(graph_function.captured_inputs)

    self.assertEqual(float(graph_function(v)), 1.0)
    self.assertEqual(float(graph_function(v)), 2.0)

    w = resource_variable_ops.ResourceVariable(0.0)

    @compiled_fn
    def bar(v):
      # [FUNCTION DOC]
      # Nested compiled function: ignores its argument and returns the constant one.
      del v
      return constant_op.constant(1.0)

    graph_function = bar.get_concrete_function(v)
    self.assertEqual(float(graph_function(v)), 1.0)
    self.assertEqual(float(graph_function(w)), 1.0)

  def testCallingFunctionWithNonTensorsFails(self):
    # [FUNCTION DOC]
    # Test: a concrete function traced for a float tensor is called with a string. Asserts a type
    # error or value error is raised.
    @compiled_fn
    def foo(x):
      # [FUNCTION DOC]
      # Nested compiled function: returns its input unchanged.
      return x

    graph_function = foo.get_concrete_function(constant_op.constant(1.0))
    with self.assertRaises((TypeError, ValueError)):
      graph_function('Not a Tensor.')

  @parameterized.parameters([
      (
          compiled_fn(
              attributes={
                  'api_implements': 'random_boost',
                  'api_preferred_device': 'CPU',
              }
          ),
          compiled_fn(
              attributes={
                  'api_implements': 'random_boost',
                  'api_preferred_device': 'GPU',
              }
          ),
      ),
      (
          compiled_fn(
              attributes={
                  'api_implements': 'random_boost',
                  'api_preferred_device': 'CPU',
              }
          ),
          compiled_fn(
              attributes={
                  'api_implements': 'random_boost',
                  'api_preferred_device': 'GPU',
              }
          ),
      ),
  ])
  @test_util.run_v2_only
  def testSwapImplementationWithGrapplerPlugin(
      self, cpu_decorator, gpu_decorator
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the two parameter tuples in the decorator list are identical, so the case is
    # run twice with the same inputs.
    # Parameterized test (v2 only): two compiled functions that declare the same implementation
    # name but prefer CPU and GPU devices (decorators are supplied as parameters). In graph mode
    # with the implementation selector enabled and a tiny minimum graph size, registers the CPU
    # concrete function and calls the GPU one. Asserts the result is five if a GPU is available,
    # otherwise three (grappler falls back to the CPU implementation).
    # Set the min_graph_nodes to -1 since the graph in this test is too small,
    # and will be ignored by grappler if don't set this.
    rewrites = rewriter_config_pb2.RewriterConfig()
    rewrites.implementation_selector = rewriter_config_pb2.RewriterConfig.ON
    rewrites.min_graph_nodes = -1
    graph_options = config_pb2.GraphOptions(
        rewrite_options=rewrites, build_cost_model=1
    )
    config_proto = config_pb2.ConfigProto(graph_options=graph_options)

    with context.graph_mode(), self.cached_session(
        config=config_proto, graph=ops.Graph(), use_gpu=True
    ):

      @cpu_decorator
      def cpu_boost(x):
        # [FUNCTION DOC]
        # Nested function (CPU-preferred implementation): adds two to its input.
        return math_ops.add(x, 2.0)

      @gpu_decorator
      def gpu_boost(x):
        # [FUNCTION DOC]
        # Nested function (GPU-preferred implementation): adds four to its input.
        return math_ops.add(x, 4.0)

      x = constant_op.constant(1.0)

      concrete_func = cpu_boost.get_concrete_function(x)
      concrete_func.add_to_graph()
      concrete_func.add_gradient_functions_to_graph()
      y = gpu_boost(x)
      y_value = self.evaluate(y)

      if test.is_gpu_available():
        self.assertEqual(y_value, 5.0)
      else:
        # Grappler fallback to use the CPU impl even called with GPU function.
        self.assertEqual(y_value, 3.0)

  @test_util.disable_tfrt(
      "b/174712583: TFRT doesn't support behavior "
      'equivalent to implementation_selector for function'
  )
  def testSwapImplementationInEager(self):
    # [FUNCTION DOC]
    # Test (skipped when not eager; disabled for TFRT): sets optimizer options on the global
    # context (tiny minimum graph size, implementation selector on, meta optimizer enabled), then
    # defines CPU- and GPU-preferred implementations of one named API. A wrapper registers the CPU
    # one and calls the GPU one under a CPU device scope; asserts the CPU implementation runs
    # (result three). Side effect: modifies global optimizer options.
    if not context.executing_eagerly():
      self.skipTest('eager only')

    # testSharedRendezvous sets the disable_meta_optimizer flag to True
    # if that subtest runs before this one, then having that set to True
    # will cause this subtest to fail. To avoid that scenario, explicitly
    # set the disable_meta_optimizer flag to false here
    context.context().set_optimizer_experimental_options({
        'min_graph_nodes': -1,
        'implementation_selector': True,
        'disable_meta_optimizer': False,
    })

    @compiled_fn(
        attributes={
            'api_implements': 'foo',
            'api_preferred_device': 'CPU',
        }
    )
    def on_cpu(x):
      # [FUNCTION DOC]
      # Nested CPU-preferred implementation: adds two to its input.
      return x + 2

    @compiled_fn(
        attributes={
            'api_implements': 'foo',
            'api_preferred_device': 'GPU',
        }
    )
    def on_gpu(x):
      # [FUNCTION DOC]
      # Nested GPU-preferred implementation: adds four to its input.
      return x + 4

    @compiled_fn
    def run_on_cpu(t):
      # [FUNCTION DOC]
      # Nested compiled function: registers the CPU implementation's concrete function and its
      # gradient functions in the current graph, then calls the GPU implementation under a CPU
      # device scope and returns its result.
      concrete_func = on_cpu.get_concrete_function(t)
      concrete_func.add_to_graph()
      concrete_func.add_gradient_functions_to_graph()
      with ops.device('CPU:0'):
        return on_gpu(t)

    # Expect to run the on_cpu branch, regardless whether gpu is available.
    self.assertEqual(run_on_cpu(constant_op.constant(1)).numpy(), 3)

  def testCompilationFunctionSeparateGraphs(self):
    # [FUNCTION DOC]
    # Test (graph mode): an outer compiled function conditionally calls an inner compiled one,
    # each with its own cache. Calls in one graph then a new graph; asserts the inner function is
    # retraced in the new graph (cache sizes: outer 1, 2, 3 and inner 1, 1, 2).
    with context.graph_mode():
      add_cache = function_cache_lib.FunctionCache()

      @compiled_fn(function_cache=add_cache)
      def add(x):
        # [FUNCTION DOC]
        # Nested compiled function: returns its input plus five.
        return x + 5

      maybe_add_cache = function_cache_lib.FunctionCache()

      @compiled_fn(function_cache=maybe_add_cache)
      def maybe_add(x, should_add):
        # [FUNCTION DOC]
        # Nested compiled function: if the flag is true returns the result of calling the inner
        # add function, otherwise returns the input unchanged.
        if should_add:
          return add(x)
        else:
          return x

      with ops.Graph().as_default():
        x = constant_op.constant(11)
        maybe_add(x, True)
        self.assertLen(maybe_add_cache, 1)
        self.assertLen(add_cache, 1)

        maybe_add(x, False)
        self.assertLen(maybe_add_cache, 2)
        self.assertLen(add_cache, 1)

      with ops.Graph().as_default():
        x = constant_op.constant(11)
        maybe_add(x, True)
        self.assertLen(maybe_add_cache, 3)
        self.assertLen(add_cache, 2)

  def testCacheKeyOverlappingShapes(self):
    # [FUNCTION DOC]
    # Test: tensors of shapes twelve-by-one and one-by-twenty-one (whose digits overlap when
    # concatenated) must have distinct cache keys, both when calling a function directly and when
    # tracing a wrapper that calls it. Asserts cache sizes one then two in each case.
    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(function_cache=function_cache)
    def defined(t):
      # [FUNCTION DOC]
      # Nested compiled function: returns its input unchanged.
      return t

    defined(array_ops.zeros([12, 1]))
    self.assertLen(function_cache, 1)
    defined(array_ops.zeros([1, 21]))
    self.assertLen(function_cache, 2)

    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(function_cache=function_cache)
    def defined_again(t):
      # [FUNCTION DOC]
      # Nested compiled function: returns the result of calling the previously compiled function.
      return defined(t)

    defined_again.get_concrete_function(array_ops.zeros([12, 1]))
    self.assertLen(function_cache, 1)
    defined_again.get_concrete_function(array_ops.zeros([1, 21]))
    self.assertLen(function_cache, 2)

  def testCacheTensorSpecIdenticalToTensor(self):
    # [FUNCTION DOC]
    # Test: requesting a concrete function with a tensor spec derived from a tensor and with the
    # tensor itself must return the very same concrete function object.
    @compiled_fn(function_cache=function_cache_lib.FunctionCache())
    def defined(t):
      # [FUNCTION DOC]
      # Nested compiled function: returns its input unchanged.
      return t

    z = array_ops.zeros([2, 2])
    z_spec = tensor_lib.TensorSpec.from_tensor(z)
    self.assertIs(
        defined.get_concrete_function(z_spec), defined.get_concrete_function(z)
    )

  def testCacheKeyNestedLists(self):
    # [FUNCTION DOC]
    # Test: three scalar tensors arranged as a list with a nested list of one versus a nested list
    # of two give different structures and therefore different cache entries (one then two).
    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(function_cache=function_cache)
    def defined(l):
      # [FUNCTION DOC]
      # Nested compiled function: returns its input unchanged.
      return l

    a = constant_op.constant(1.0)
    b = constant_op.constant(2.0)
    c = constant_op.constant(3.0)
    defined([[a], b, c])
    self.assertLen(function_cache, 1)

    defined([[a, b], c])
    self.assertLen(function_cache, 2)

  def testCacheKeyAttrsClass(self):
    # [FUNCTION DOC]
    # Test (skipped if the attrs library is missing): instances of an attrs class holding tensors
    # are passed to a compiled function. Two instances with the same nested structure share a
    # cache entry; one with a different arrangement of fields adds a second entry.
    if attr is None:
      self.skipTest('attr module is unavailable.')

    @attr.s
    class TestClass:
      # [CLASS DOC]
      # Attrs-decorated test class with two attribute fields, a and b, used as a structured
      # argument.
      a = attr.ib()
      b = attr.ib()

    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(function_cache=function_cache)
    def defined(l):
      # [FUNCTION DOC]
      # Nested compiled function: returns its input unchanged.
      return l

    defined(
        TestClass(
            constant_op.constant(1.0),
            [constant_op.constant(2.0), constant_op.constant(3.0)],
        )
    )
    self.assertLen(function_cache, 1)
    defined(
        TestClass(
            constant_op.constant(1.0),
            [constant_op.constant(2.0), constant_op.constant(3.0)],
        )
    )
    self.assertLen(function_cache, 1)

    defined(
        TestClass(
            [constant_op.constant(1.0), constant_op.constant(2.0)],
            constant_op.constant(3.0),
        )
    )
    self.assertLen(function_cache, 2)

  def testDistinctVariablesNoRetracing(self):
    # [FUNCTION DOC]
    # Test: a function taking three distinct resource variables is called, then called with the
    # same variables rearranged. Asserts one cache entry since the aliasing pattern is the same.
    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(function_cache=function_cache)
    def defined(a, b, c):
      # [FUNCTION DOC]
      # Nested compiled function: returns the sum of its three inputs.
      return a + b + c

    x = resource_variable_ops.ResourceVariable(0.0)
    y = resource_variable_ops.ResourceVariable(0.0)
    z = resource_variable_ops.ResourceVariable(0.0)

    # We generate cache keys based on unique combinations of resource ids.
    defined(x, y, z)
    self.assertLen(function_cache, 1)

    # Re-arranging arguments should not cause cache miss
    # because the three inputs are still distinct
    defined(z, y, x)
    self.assertLen(function_cache, 1)

  def testRetracingOnDifferentVaribleCombinationPatterns(self):
    # [FUNCTION DOC]
    # Test: calls with variable aliasing patterns (all distinct; first two same; a renamed copy
    # of that pattern; last two same, repeated). Asserts a new trace occurs only when the aliasing
    # pattern is new (cache sizes 1, 2, 2, 3, 3).
    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(function_cache=function_cache)
    def defined(a, b, c):
      # [FUNCTION DOC]
      # Nested compiled function: returns the sum of its three inputs.
      return a + b + c

    x = resource_variable_ops.ResourceVariable(0.0)
    y = resource_variable_ops.ResourceVariable(0.0)
    z = resource_variable_ops.ResourceVariable(0.0)

    defined(x, y, z)
    self.assertLen(function_cache, 1)

    # Retracing because the first two arguments are the same
    defined(x, x, z)
    self.assertLen(function_cache, 2)

    # Replacing x with y does not cause cache miss
    # because the combination stays the same as (x, x, z)
    defined(y, y, z)
    self.assertLen(function_cache, 2)

    # A different combination pattern causes cache miss
    defined(z, y, y)
    self.assertLen(function_cache, 3)
    defined(z, y, y)
    self.assertLen(function_cache, 3)

  @test_util.run_v2_only
  def testDeepcopyVariableNoRetracing(self):
    # [FUNCTION DOC]
    # Test (v2 only): calls a three-input function with distinct variables, then with a deep copy
    # of the first variable substituted. Asserts no retracing (cache size stays one).
    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(function_cache=function_cache)
    def defined(a, b, c):
      # [FUNCTION DOC]
      # Nested compiled function: returns the sum of its three inputs.
      return a + b + c

    x = resource_variable_ops.ResourceVariable(0.0)
    y = resource_variable_ops.ResourceVariable(0.0)
    z = resource_variable_ops.ResourceVariable(0.0)
    defined(x, y, z)
    self.assertLen(function_cache, 1)

    x_copy = copy.deepcopy(x)
    defined(x_copy, y, z)
    self.assertLen(function_cache, 1)

  @test_util.disable_tfrt('b/173429686')
  @test_util.run_v2_only
  def testExecutorType(self):
    # [FUNCTION DOC]
    # Test (v2 only; disabled for TFRT): runs a function adding five and asserts result five.
    # Selecting a non-existent function executor type must raise a not-found error naming it;
    # the empty string, the default name and nothing are accepted and give five.
    @compiled_fn
    def add_five(x):
      # [FUNCTION DOC]
      # Nested compiled function: returns its input plus five.
      return x + 5

    self.assertEqual(
        5, add_five(constant_op.constant(0, dtype=dtypes.int32)).numpy()
    )

    with self.assertRaisesRegex(errors.NotFoundError, 'NON_EXISTENT_EXECUTOR'):
      with context.function_executor_type('NON_EXISTENT_EXECUTOR'):
        add_five(constant_op.constant(0, dtype=dtypes.int32))

    for executor_type in ('', 'DEFAULT', None):
      with context.function_executor_type(executor_type):
        self.assertAllEqual(
            5, add_five(constant_op.constant(0, dtype=dtypes.int32)).numpy()
        )

  @test_util.assert_no_garbage_created
  def testReferenceCycles(self):
    # [FUNCTION DOC]
    # Test: under a decorator asserting no garbage is created, a compiled lambda is called once,
    # a weak reference is taken and the function deleted. Asserts the weak reference is dead,
    # meaning the compiled function is not part of a reference cycle.
    fn = compiled_fn(lambda x: 2.0 * x)

    fn(constant_op.constant(4.0))
    weak_fn = weakref.ref(fn)
    del fn
    # Tests that the weak reference we made to the function is now dead, which
    # means the object has been deleted. This should be true as long as the
    # function itself is not involved in a reference cycle.
    self.assertIs(None, weak_fn())

  @test_util.run_in_graph_and_eager_modes
  def testShapeCaching(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): a function returning the shape of its input is called with
    # different shaped tensors directly and through a wrapper with an unknown-shape signature.
    # Asserts each call returns the actual shape of its input rather than a cached stale shape.
    @compiled_fn
    def func(x):
      # [FUNCTION DOC]
      # Nested compiled function: returns the dynamic shape of its input.
      return array_ops.shape(x)

    @compiled_fn(
        input_signature=[tensor_lib.TensorSpec([None, None], dtypes.float32)]
    )
    def calls_func(x):
      # [FUNCTION DOC]
      # Nested compiled function with a two-dimensional unknown-size signature: returns the result
      # of calling the shape function on its input.
      return func(x)

    self.assertAllEqual([1, 1], self.evaluate(func(array_ops.zeros([1, 1]))))
    self.assertAllEqual([2, 2], self.evaluate(func(array_ops.zeros([2, 2]))))
    self.assertAllEqual(
        [3, 3], self.evaluate(calls_func(array_ops.zeros([3, 3])))
    )

  def testLimitedRetracing(self):
    # [FUNCTION DOC]
    # Test: a function counting its traces is called fifty times with six differently shaped
    # inputs. Asserts the total trace count stays below thirteen, i.e. at most about twice per
    # distinct input type.
    trace_count = [0]

    function_cache = function_cache_lib.FunctionCache()

    @compiled_fn(function_cache=function_cache)
    def func(x):
      # [FUNCTION DOC]
      # Nested compiled function: increments the shared trace counter (runs only while tracing)
      # and returns its input.
      trace_count[0] += 1
      return x

    for _ in range(50):
      func(constant_op.constant(3.0))
      func(constant_op.constant(4.0))
      func(constant_op.constant([[1.0, 2.0]]))
      func(constant_op.constant([[]]))
      func(constant_op.constant([[3.0, 4.0], [5.0, 6.0]]))
      func(constant_op.constant([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))
    # Tracing more than twice per input doesn't make sense.
    self.assertLess(trace_count[0], 13)


class CompilationCollectionTest(test.TestCase):
  # [CLASS DOC]
  # Tests that compiled functions can read graph collections of the enclosing graph, and that
  # collection writes made inside the function body do not leak out. Uses graph mode sessions.

  def testCollectionValueAccess(self):
    # [FUNCTION DOC]
    # Test (graph mode): collections x and y hold two and five in the outer graph. A compiled
    # function reads them, adds them, and also adds an item to collection z. Asserts the result is
    # seven, the outer collections are unchanged, and collection z stays empty in the outer graph.
    """Read values from graph collections inside of defun."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = 2
        y = 5
        ops.add_to_collection('x', x)
        ops.add_to_collection('y', y)

        @compiled_fn
        def fn():
          # [FUNCTION DOC]
          # Nested compiled function with no inputs: reads the first items of collections x and y,
          # adds them as constants, adds seven to collection z, and returns the sum.
          x_const = constant_op.constant(ops.get_collection('x')[0])
          y_const = constant_op.constant(ops.get_collection('y')[0])
          z = math_ops.add(x_const, y_const)
          ops.add_to_collection('z', 7)
          return z

        self.assertEqual(7, int(self.evaluate(fn())))
        self.assertEqual(ops.get_collection('x'), [2])
        self.assertEqual(ops.get_collection('y'), [5])
        self.assertEqual(ops.get_collection('z'), [])

  def testCollectionVariableValueAccess(self):
    # [FUNCTION DOC]
    # Test (graph mode): a resource variable defined outside is read inside a compiled function.
    # After initializing variables, asserts the function returns one and the global variables
    # collection of the outer graph holds exactly one entry.
    """Read variable value from graph collections inside of defun."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        v = resource_variable_ops.ResourceVariable(1.0)

        @compiled_fn
        def f():
          # [FUNCTION DOC]
          # Nested compiled function with no inputs: returns the captured variable's value.
          return v.read_value()

        self.evaluate(variables.global_variables_initializer())
        self.assertEqual(1.0, float(self.evaluate(f())))
        self.assertLen(ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES), 1)


class MultiDeviceCompilationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # GPU-only tests of compiled functions that span CPU and GPU devices: outputs on several
  # devices, int32 handling, colocation, resource inputs and outputs, nested calls, input and
  # output device mapping, and unused argument pruning. Each test is skipped without a GPU.

  @test_util.run_gpu_only
  def testMultiDeviceOutput(self):
    # [FUNCTION DOC]
    # Test (GPU only): a function computes the same matrix product on CPU and on GPU and returns
    # both. Asserts values are correct and each output's backing device matches where it was
    # computed.
    """Tests that functions can produce outputs on multiple devices."""

    @compiled_fn
    def func(a, b, transpose_a):
      # [FUNCTION DOC]
      # Nested compiled function: computes a matrix product (optionally transposing the first
      # input) once under a CPU device scope and once under a GPU scope, returning both results.
      with ops.device('/device:CPU:0'):
        m1 = math_ops.matmul(a, b, transpose_a=transpose_a)
      with ops.device('/device:GPU:0'):
        m2 = math_ops.matmul(a, b, transpose_a=transpose_a)
      return m1, m2

    t = constant_op.constant([[1.0, 2.0], [3.0, 4.0]])
    m1, m2 = func(t, t, transpose_a=True)
    self.assertAllEqual(m1.numpy(), [[10, 14], [14, 20]])
    self.assertRegex(m1.backing_device, 'CPU')
    self.assertAllEqual(m2.numpy(), [[10, 14], [14, 20]])
    self.assertRegex(m2.backing_device, 'GPU')

  @test_util.run_gpu_only
  def testEmptyBody(self):
    # [FUNCTION DOC]
    # Test (GPU only): a function that only swaps its two inputs (one on CPU, one on GPU). Asserts
    # the outputs have the right values and keep the opposite devices (first output on GPU, second
    # on CPU).
    @compiled_fn
    def func(a, b):
      # [FUNCTION DOC]
      # Nested compiled function that performs no computation: returns its inputs swapped.
      return b, a

    with ops.device('/device:CPU:0'):
      a = array_ops.identity(3.0)
    with ops.device('/device:GPU:0'):
      b = array_ops.identity(5.0)

    m1, m2 = func(a, b)
    self.assertAllEqual(m1.numpy(), 5.0)
    self.assertRegex(m1.backing_device, 'GPU')
    self.assertAllEqual(m2.numpy(), 3.0)
    self.assertRegex(m2.backing_device, 'CPU')

  @test_util.run_gpu_only
  def testMultiDeviceInt32(self):
    # [FUNCTION DOC]
    # Test (GPU only): int32 scalars on CPU and GPU plus an int32 resource variable are passed to a
    # function that computes on each device. Asserts values are right and that int32 outputs are
    # placed on CPU regardless of computing device; repeats with the CPU and GPU inputs flipped
    # to check the function is re-instantiated correctly.
    """Tests that multi-device functions can take and output INT32s.

    When an INT32 device tensor is fed into a function, it is copied to CPU
    by the eager runtime. The function sees all INT32 inputs on CPU.

    We set allocator attribute 'on_host' for INT32 outputs. They can be
    partitioned into the GPU component function, but will be allocated on
    CPU nevertheless.

    There is experimental support for `ints_on_device` in
    FunctionLibraryRuntime now. We can try that.
    """
    with ops.device('/device:CPU:0'):
      int_cpu = constant_op.constant(3, dtype=dtypes.int32)
      resource = resource_variable_ops.ResourceVariable(5, dtype=dtypes.int32)
    with ops.device('/device:GPU:0'):
      int_gpu = constant_op.constant(7, dtype=dtypes.int32)

    @compiled_fn
    def func(int_cpu, resource, int_gpu):
      # [FUNCTION DOC]
      # Nested compiled function: computes the first input times the resource plus the third on
      # CPU, and the third times the resource plus the first plus one on GPU; returns both.
      with ops.device('/device:CPU:0'):
        m1 = int_cpu * resource + int_gpu
      with ops.device('/device:GPU:0'):
        # This computation will happen on GPU but m2 will be copied to CPU.
        m2 = int_gpu * resource + int_cpu + 1
      return m1, m2

    m1, m2 = func(int_cpu, resource, int_gpu)
    self.assertAllEqual(m1.numpy(), 22)
    self.assertRegex(m1.backing_device, 'CPU')
    self.assertAllEqual(m2.numpy(), 39)
    self.assertRegex(m2.backing_device, 'CPU')

    # flip arguments
    m1, m2 = func(int_gpu, resource, int_cpu)
    self.assertAllEqual(m1.numpy(), 38)
    self.assertRegex(m1.backing_device, 'CPU')
    self.assertAllEqual(m2.numpy(), 23)
    self.assertRegex(m2.backing_device, 'CPU')

  @test_util.run_gpu_only
  def testMultiDeviceColocateWith(self):
    # [FUNCTION DOC]
    # Test (GPU only): for every pairing of CPU and GPU placement for two inputs, a function
    # multiplies each input while colocating with it. Asserts the results (two and thirty) and that
    # each output lives on the same device as its input.
    """Tests that function's outputs respect colocation constraints."""

    @compiled_fn
    def func(a, b):
      # [FUNCTION DOC]
      # Nested compiled function: doubles the first input colocated with it and triples the
      # second colocated with it, returning both.
      with ops.colocate_with(a):
        ra = 2 * a
      with ops.colocate_with(b):
        rb = 3 * b
      return ra, rb

    devices = ['/device:CPU:0', '/device:GPU:0']
    for dev1, dev2 in itertools.product(devices, devices):
      with ops.device(dev1):
        a = array_ops.identity(1.0)
      with ops.device(dev2):
        b = array_ops.identity(10.0)

      ra, rb = func(a, b)
      self.assertEqual(ra.numpy(), 2.0)
      self.assertRegex(ra.backing_device, dev1)
      self.assertEqual(rb.numpy(), 30.0)
      self.assertRegex(rb.backing_device, dev2)

  @test_util.run_gpu_only
  def testMultiDeviceResources(self):
    # [FUNCTION DOC]
    # Test (GPU only): resource variables live on CPU and GPU, some passed as arguments and some
    # captured. Asserts products and output devices are right, and after calling with the
    # arguments' devices flipped that the function is re-instantiated and results follow the new
    # devices.
    with ops.device('/device:CPU:0'):
      c1 = resource_variable_ops.ResourceVariable(2.0)
      c2 = resource_variable_ops.ResourceVariable(7.0)
    with ops.device('/device:GPU:0'):
      g1 = resource_variable_ops.ResourceVariable(3.0)
      g2 = resource_variable_ops.ResourceVariable(5.0)

    @compiled_fn
    def func(resource1, resource2):
      # [FUNCTION DOC]
      # Nested compiled function: multiplies the first resource by the captured GPU variable on
      # CPU, and the second resource by the captured CPU variable on GPU; returns both products.
      with ops.device('/device:CPU:0'):
        result1 = resource1 * g2
      with ops.device('/device:GPU:0'):
        result2 = resource2 * c2
      return result1, result2

    r1, r2 = func(c1, g1)
    self.assertEqual(r1.numpy(), 10.0)
    self.assertRegex(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 21.0)
    self.assertRegex(r2.backing_device, 'GPU')

    # Call with flipped inputs. Check that we look at resource's
    # device and reinstantiates the function when inputs' devices change.
    r1, r2 = func(g1, c1)
    self.assertEqual(r1.numpy(), 15.0)
    self.assertRegex(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 14.0)
    self.assertRegex(r2.backing_device, 'GPU')

  @test_util.run_gpu_only
  def testOutputResources(self):
    # [FUNCTION DOC]
    # Test (GPU only): a function returns resource handles of a CPU and a GPU variable along with
    # computed products. Asserts product values and devices, that returned handles live on CPU
    # and can be read to give the variables' values, and that this still holds after flipping
    # the inputs.
    with ops.device('/device:CPU:0'):
      c1 = resource_variable_ops.ResourceVariable(2.0)
    with ops.device('/device:GPU:0'):
      g1 = resource_variable_ops.ResourceVariable(3.0)

    @compiled_fn
    def func(resource1, resource2):
      # [FUNCTION DOC]
      # Nested compiled function: multiplies the first resource by five on CPU and the second by
      # seven on GPU; returns the first product, the first handle, the second product and the
      # second handle.
      with ops.device('/device:CPU:0'):
        result1 = resource1 * 5
      with ops.device('/device:GPU:0'):
        result2 = resource2 * 7
      return result1, resource1.handle, result2, resource2.handle

    r1, res1, r2, res2 = func(c1, g1)
    self.assertEqual(r1.numpy(), 10.0)
    self.assertRegex(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 21.0)
    self.assertRegex(r2.backing_device, 'GPU')

    def check_handle(handle, expected_value):
      # [FUNCTION DOC]
      # Nested helper: asserts the resource handle tensor is on CPU, reads the variable through
      # the handle and asserts the value equals the expected number.
      self.assertRegex(handle.backing_device, 'CPU')
      tensor = gen_resource_variable_ops.read_variable_op(
          handle, dtypes.float32
      )
      self.assertEqual(tensor.numpy(), expected_value)

    # Check that handles returned from functions are on CPU and an op using
    # the resource handle is correctly placed on the device backing the
    # resource.
    check_handle(res1, 2.0)
    check_handle(res2, 3.0)

    # Call with flipped inputs to make sure the same the function is
    # reinstantiated and eager runtime does not mess up the device assignment
    # for ops consuming handles returned from defuns.
    r1, res1, r2, res2 = func(g1, c1)
    self.assertEqual(r1.numpy(), 15.0)
    self.assertRegex(r1.backing_device, 'CPU')
    self.assertEqual(r2.numpy(), 14.0)
    self.assertRegex(r2.backing_device, 'GPU')
    check_handle(res1, 3.0)
    check_handle(res2, 2.0)

  @test_util.run_gpu_only
  def testPassResourceThroughNestedFunctionCall(self):
    # [FUNCTION DOC]
    # Test (GPU only): a GPU resource variable is passed to an outer function that calls an inner
    # non-inlined, non-specialized function under a CPU scope. Asserts the doubled value is six and
    # the result is on CPU.
    """Test passing GPU resource to noinline function call placed on CPU.

    PartitionedCallOp must not enforce any particular device assignment for the
    resource output. Inner function marked as `_nospecialize`, so Grappler would
    not prune unused function output.
    """

    with ops.device('/device:GPU:0'):
      g1 = resource_variable_ops.ResourceVariable(3.0)

    @compiled_fn(attributes={'_noinline': True, '_nospecialize': True})
    def inner(resource1):
      # [FUNCTION DOC]
      # Nested compiled function (not inlined, not specialized): returns the resource doubled and
      # the resource handle.
      return resource1 * 2, resource1.handle

    @compiled_fn
    def outer(resource1):
      # [FUNCTION DOC]
      # Nested compiled function: calls the inner function under a CPU device scope, ignores the
      # handle output and returns the doubled value.
      with ops.device('/device:CPU:0'):
        r1, _ = inner(resource1)
      return r1

    r1 = outer(g1)

    self.assertEqual(r1.numpy(), 6.0)
    self.assertRegex(r1.backing_device, 'CPU')

  @test_util.run_gpu_only
  def testReturnResourceFromNestedFunctionCall(self):
    # [FUNCTION DOC]
    # Test (GPU only): an inner non-inlined function increments a GPU variable and returns a
    # product and the resource handle; an outer function calls it under a CPU scope and returns
    # both. Asserts the product is ten on CPU and the returned handle (on CPU) reads as five.
    """Test returning GPU resource from noinline function call placed on CPU.

    When inferring output devices for the return value, do not set a device for
    returns of DT_RESOURCE data type based on the device assignment of the node
    that produced that resource. As an example function call placed on CPU can
    return resources on GPU.
    """

    with ops.device('/device:GPU:0'):
      g1 = resource_variable_ops.ResourceVariable(3.0)

    @compiled_fn(attributes={'_noinline': True})
    def inner(resource1):
      # [FUNCTION DOC]
      # Nested compiled function (not inlined): adds two to the resource variable, then returns
      # the variable doubled and its handle.
      resource1.assign_add(2.0)
      return resource1 * 2, resource1.handle

    @compiled_fn
    def outer(resource1):
      # [FUNCTION DOC]
      # Nested compiled function: calls the inner function under a CPU scope and returns both
      # its outputs.
      with ops.device('/device:CPU:0'):
        r1, res1 = inner(resource1)
      return r1, res1

    r1, res1 = outer(g1)

    self.assertEqual(r1.numpy(), 10.0)
    self.assertRegex(r1.backing_device, 'CPU')

    def check_handle(handle, expected_value):
      # [FUNCTION DOC]
      # Nested helper: asserts the handle is on CPU, reads the variable through it and asserts the
      # value equals the expected number.
      self.assertRegex(handle.backing_device, 'CPU')
      tensor = gen_resource_variable_ops.read_variable_op(
          handle, dtypes.float32
      )
      self.assertEqual(tensor.numpy(), expected_value)

    # Check that handles returned from functions are on CPU and an op using
    # the resource handle is correctly placed on the device backing the
    # resource.
    check_handle(res1, 5.0)

  @test_util.run_gpu_only
  def testComplexInputOutputDevicePattern(self):
    # [FUNCTION DOC]
    # Test (GPU only): four CPU values and four GPU values (resources and tensors) are passed in a
    # scrambled order to a function whose computations are split across CPU and GPU. Asserts each
    # output sits on the expected device and carries the expected numeric value.
    """Tests input/output mapping logic in partitioning."""
    with ops.device('/device:CPU:0'):
      rc0 = resource_variable_ops.ResourceVariable(2.0)
      rc1 = resource_variable_ops.ResourceVariable(3.0)
      cc0 = array_ops.identity(5.0)
      cc1 = array_ops.identity(7.0)
    with ops.device('/device:GPU:0'):
      rg0 = resource_variable_ops.ResourceVariable(11.0)
      rg1 = resource_variable_ops.ResourceVariable(13.0)
      cg0 = array_ops.identity(17.0)
      cg1 = array_ops.identity(19.0)

    # Make sure tensors are on expected devices.
    for tensor in [cc0, cc1]:
      self.assertRegex(tensor.backing_device, 'CPU:0')
    for tensor in [cg0, cg1]:
      self.assertRegex(tensor.backing_device, 'GPU:0')

    @compiled_fn
    def func(rc0, cc0, cg0, rc1, cg1, rg0, rg1, cc1):
      # [FUNCTION DOC]
      # Nested compiled function: computes intermediate products on CPU and GPU, then combines
      # them with further products on the opposite devices; returns two combined values and two
      # intermediates.
      with ops.device('/device:CPU:0'):
        m1 = rc0 * cg0
      with ops.device('/device:GPU:0'):
        m2 = rg0 * cc0

      with ops.device('/device:CPU:0'):
        r1 = 1000.0 * m2 + rc1 * cg1
      with ops.device('/device:GPU:0'):
        r2 = 1000.0 * m1 + rg1 * cc1

      return r1, r2, m2, m1

    r1, r2, m2, m1 = func(rc0, cc0, cg0, rc1, cg1, rg0, rg1, cc1)
    self.assertRegex(m1.backing_device, 'CPU')
    self.assertRegex(r1.backing_device, 'CPU')
    self.assertRegex(m2.backing_device, 'GPU')
    self.assertRegex(r2.backing_device, 'GPU')
    self.assertEqual(m1.numpy(), 34.0)
    self.assertEqual(r1.numpy(), 55000.0 + 3.0 * 19.0)
    self.assertEqual(m2.numpy(), 55.0)
    self.assertEqual(r2.numpy(), 34000.0 + 13.0 * 7.0)

  @test_util.run_gpu_only
  def testArgumentPruning(self):
    # [FUNCTION DOC]
    # Test (GPU only): a function receives five tensors on mixed devices but uses only three.
    # Asserts the result equals the product of the used values.
    """Tests functions taking unnecessary arguments."""
    with ops.device('/device:CPU:0'):
      c1 = constant_op.constant(5.0)
      c2 = constant_op.constant(7.0)

    with ops.device('/device:GPU:0'):
      g1 = constant_op.constant(11.0)
      g2 = constant_op.constant(13.0)
      g3 = constant_op.constant(17.0)

    @compiled_fn
    def func(g1, g2, c1, g3, c2):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # Nested compiled function: ignores the first two inputs and returns the product of the
      # third, fourth and fifth.
      # arguments g1 and g2 are unused and can be pruned by grappler.
      return c1 * g3 * c2

    result = func(g1, g2, c1, g3, c2)
    self.assertEqual(result.numpy(), 5.0 * 7.0 * 17.0)


class CompilationArgumentNamingTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that concrete functions expose input operations named after the Python parameters (or
  # tensor spec names), including for variadic, keyword, dict-returning, method and
  # input-signature cases, and that the user-specified-name attribute matches.
  """Tests for recognizable export signatures from concrete functions."""

  @test_util.run_v2_only
  def testBasic(self):
    # [FUNCTION DOC]
    # Test (v2 only): a two-input function is traced with a vector spec and a scalar spec.
    # Asserts input operation names and user-specified name attributes are a and b, two structured
    # outputs exist, and calling by position or by keyword gives three and two.
    @compiled_fn
    def fn(a, b):
      # [FUNCTION DOC]
      # Nested compiled function: returns the sum and the product of its inputs.
      return a + b, a * b

    # Call the function to make def_function happy
    fn(array_ops.ones([]), array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        tensor_lib.TensorSpec(shape=(None,), dtype=dtypes.float32),
        tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32),
    )
    self.assertEqual(['a', 'b'], [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'a', b'b'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs],
    )
    self.assertLen(fn_op.graph.structured_outputs, 2)
    self.assertAllClose(
        [3.0, 2.0], fn_op(constant_op.constant(1.0), constant_op.constant(2.0))
    )
    self.assertAllClose(
        [3.0, 2.0],
        fn_op(a=constant_op.constant(1.0), b=constant_op.constant(2.0)),
    )

  def testVariable(self):
    # [FUNCTION DOC]
    # Test: like the basic naming test but the second argument is a variable rather than a spec.
    # Asserts the input names and user-specified names are a and b and there are two outputs.
    @compiled_fn
    def fn(a, b):
      # [FUNCTION DOC]
      # Nested compiled function: returns the sum and the product of its inputs.
      return a + b, a * b

    # Call the function to make def_function happy
    fn(array_ops.ones([]), array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        tensor_lib.TensorSpec(shape=(None,), dtype=dtypes.float32),
        variables.Variable(1.0),
    )
    self.assertEqual(['a', 'b'], [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'a', b'b'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs],
    )
    self.assertLen(fn_op.graph.structured_outputs, 2)

  def testDictReturned(self):
    # [FUNCTION DOC]
    # Test: a function with a defaulted pair and defaulted scalar returns a mapping. Traces it
    # with specs for x and y only, then with custom spec names for the pair and y, then with a
    # positional named spec. Asserts input operation names and user-specified names follow the
    # spec names (or parameter names when unnamed) in order, and the output keys are alpha and beta.
    @compiled_fn
    def fn(x, z=(1.0, 2.0), y=3.0):
      # [FUNCTION DOC]
      # Nested compiled function: unpacks the pair and returns a mapping of alpha (x plus y plus
      # first pair element) and beta (x times y plus second pair element).
      z1, z2 = z
      return {'alpha': x + y + z1, 'beta': x * y + z2}

    # Call the function to make def_function happy
    fn(array_ops.ones([]))

    fn_op = fn.get_concrete_function(
        x=tensor_lib.TensorSpec(shape=(None,), dtype=dtypes.float32),
        y=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32),
    )
    self.assertEqual(['x', 'y'], [inp.op.name for inp in fn_op.inputs])
    self.assertEqual(
        [b'x', b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op.inputs],
    )
    self.assertEqual(
        {'alpha', 'beta'}, set(fn_op.graph.structured_outputs.keys())
    )

    fn_op2 = fn.get_concrete_function(
        z=(
            tensor_lib.TensorSpec(
                shape=(None,), dtype=dtypes.float32, name='z_first'
            ),
            tensor_lib.TensorSpec(
                shape=(), dtype=dtypes.float32, name='z_second'
            ),
        ),
        y=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'),
        x=4.0,
    )
    self.assertEqual(
        ['z_first', 'z_second', 'custom'],
        [inp.op.name for inp in fn_op2.inputs],
    )
    self.assertEqual(
        [b'z_first', b'z_second', b'custom'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op2.inputs],
    )

    fn_op3 = fn.get_concrete_function(
        tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='custom'),
        z=(
            tensor_lib.TensorSpec(
                shape=(None,), dtype=dtypes.float32, name='z1'
            ),
            tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='z2'),
        ),
        y=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32),
    )
    self.assertEqual(
        ['custom', 'z1', 'z2', 'y'], [inp.op.name for inp in fn_op3.inputs]
    )
    self.assertEqual(
        [b'custom', b'z1', b'z2', b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in fn_op3.inputs],
    )

  def testMethod(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test unconditionally skips itself partway through (marked not working), so
    # the final block that requests a concrete function with a renamed spec is dead code.
    # Test: a bound method is compiled and traced with a scalar spec twice. Asserts the input
    # operation name and user-specified name attribute are x each time.
    class HasMethod(object):
      # [CLASS DOC]
      # HELPER: tiny class exposing one identity method, used to test compiling a bound method.

      def method(self, x):
        # [FUNCTION DOC]
        # Returns its argument unchanged.
        return x

    has_method = HasMethod()
    compiled_method = compiled_fn(has_method.method)
    class_op = compiled_method.get_concrete_function(
        tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32)
    )
    self.assertEqual(['x'], [inp.op.name for inp in class_op.inputs])
    self.assertEqual(
        [b'x'],
        [inp.op.get_attr('_user_specified_name') for inp in class_op.inputs],
    )

    method_op = compiled_method.get_concrete_function(
        tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32)
    )
    self.assertEqual(['x'], [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'x'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs],
    )
    # TODO(allenl): It should be possible to override names when exporting. Do
    # TensorSpec names need to go in cache keys? Or maybe get_concrete_function
    # should always retrace?
    self.skipTest('Not working')
    method_op = has_method.method.get_concrete_function(
        tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='y')
    )
    self.assertEqual(['y'], [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs],
    )

  def testMethodSignature(self):
    # [FUNCTION DOC]
    # Test: a bound method is compiled with an input signature holding a named float64 spec of
    # unknown shape. Requests the concrete function twice and asserts the input operation name and
    # user-specified name attribute are y both times.
    class HasMethod(object):
      # [CLASS DOC]
      # HELPER: tiny class with one method, used to test compiling a bound method with a signature.

      def method(self, x):
        # [FUNCTION DOC]
        # Hashes the instance (to ensure the receiver is not a weak proxy, which cannot be hashed
        # here) and returns the argument unchanged.
        hash(self)  # No weak proxies passed as `self`
        return x

    has_method = HasMethod()
    compiled_method = compiled_fn(
        has_method.method,
        input_signature=(
            tensor_lib.TensorSpec(shape=None, dtype=dtypes.float64, name='y'),
        ),
    )

    method_op = compiled_method.get_concrete_function()
    self.assertEqual(['y'], [inp.op.name for inp in method_op.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op.inputs],
    )
    method_op2 = compiled_method.get_concrete_function()
    self.assertEqual(['y'], [inp.op.name for inp in method_op2.inputs])
    self.assertEqual(
        [b'y'],
        [inp.op.get_attr('_user_specified_name') for inp in method_op2.inputs],
    )

  def testVariadic(self):
    # [FUNCTION DOC]
    # Test: a function taking one regular, extra positional and extra keyword arguments is traced
    # with several specs, some named. Asserts input operation names and user-specified names are
    # x, y, args_1, second_variadic, z and cust, i.e. named specs keep their names and keyword
    # arguments use the keyword name unless the spec has its own name.
    @compiled_fn
    def variadic_fn(x, *args, **kwargs):
      # [FUNCTION DOC]
      # Nested compiled function: returns the first input plus the sum of all other positional and
      # keyword inputs.
      return x + math_ops.add_n(list(args) + list(kwargs.values()))

    # Call the function to make def_function happy
    variadic_fn(array_ops.ones([]), array_ops.ones([]))
    variadic_op = variadic_fn.get_concrete_function(
        tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32),
        tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32, name='y'),
        tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32),
        tensor_lib.TensorSpec(
            shape=(), dtype=dtypes.float32, name='second_variadic'
        ),
        z=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32),
        zz=tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='cust'),
    )
    self.assertEqual(
        ['x', 'y', 'args_1', 'second_variadic', 'z', 'cust'],
        [inp.op.name for inp in variadic_op.inputs],
    )
    self.assertEqual(
        [b'x', b'y', b'args_1', b'second_variadic', b'z', b'cust'],
        [inp.op.get_attr('_user_specified_name') for inp in variadic_op.inputs],
    )

  def testVariadicInputSignature(self):
    # [FUNCTION DOC]
    # Test: a variadic function compiled with a four-spec signature (two named) and an explicit
    # name. After calling once, asserts the concrete function name contains the given name and the
    # input operation names and user-specified names are x, y, args_1 and z.
    @compiled_fn(
        input_signature=(
            tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32),
            tensor_lib.TensorSpec(shape=None, dtype=dtypes.float32, name='y'),
            tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32),
            tensor_lib.TensorSpec(shape=(), dtype=dtypes.float32, name='z'),
        ),
        name='variadic_fn',
    )
    def variadic_fn(x, *args):
      # [FUNCTION DOC]
      # Nested compiled function: returns the first input plus the sum of the remaining inputs.
      return x + math_ops.add_n(list(args))

    # Call the function to make def_function happy
    variadic_fn(
        array_ops.ones([]),
        array_ops.ones([]),
        array_ops.ones([]),
        array_ops.ones([]),
    )
    variadic_op = variadic_fn.get_concrete_function()
    self.assertIn(b'variadic_fn', variadic_op.name)
    self.assertEqual(
        ['x', 'y', 'args_1', 'z'], [inp.op.name for inp in variadic_op.inputs]
    )
    self.assertEqual(
        [b'x', b'y', b'args_1', b'z'],
        [inp.op.get_attr('_user_specified_name') for inp in variadic_op.inputs],
    )


class DevicePlacementTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that device annotations inside compiled functions are respected and that function
  # definitions are independent of the device scope active at call time. Relies on the four
  # logical CPU devices configured by the module setup hook.

  @test_util.run_in_graph_and_eager_modes
  def testMultipleDeviceCheck(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): a function placing a device-reporting op under a CPU scope is
    # compiled and called under a CPU device scope. Asserts the reported device string contains
    # the first CPU.
    def f():
      # [FUNCTION DOC]
      # Nested function under test: under a CPU device scope returns the output of the test
      # operation that reports its placement.
      with ops.device('cpu'):
        return test_ops.device_placement_op()

    func = compiled_fn(f)
    with ops.device('cpu:0'):
      output = self.evaluate(func())
      self.assertIn(compat.as_bytes('CPU:0'), output)

  @test_util.run_in_graph_and_eager_modes
  def testDeviceAnnotationsRespected(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): a function places placement-reporting ops on CPUs zero, one,
    # two and an unannotated one. Called with no scope, under CPU three, and under CPU zero.
    # Asserts one cache entry throughout (definitions are independent of call-site device),
    # annotated ops stay on their devices, and the unannotated op follows the call-site scope.
    def multi_device_fn():
      # [FUNCTION DOC]
      # Nested function under test: creates four placement-reporting ops, three with explicit CPU
      # annotations and one without, and returns all four outputs.
      with ops.device('/cpu:0'):
        s0 = test_ops.device_placement_op()
      with ops.device('/cpu:1'):
        s1 = test_ops.device_placement_op()
      with ops.device('/cpu:2'):
        s2 = test_ops.device_placement_op()
      s3 = test_ops.device_placement_op()
      return s0, s1, s2, s3

    function_cache = function_cache_lib.FunctionCache()
    defined = compiled_fn(multi_device_fn, function_cache=function_cache)
    outputs = self.evaluate(defined())
    self.assertLen(function_cache, 1)
    self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
    self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
    self.assertIn(compat.as_bytes('CPU:2'), outputs[2])

    with ops.device('/cpu:3'):
      outputs = self.evaluate(defined())
    # All function definitions are agnostic to call site devices.
    self.assertLen(function_cache, 1)
    self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
    self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
    self.assertIn(compat.as_bytes('CPU:2'), outputs[2])
    self.assertIn(compat.as_bytes('CPU:3'), outputs[3])

    with ops.device('/cpu:0'):
      outputs = self.evaluate(defined())
    self.assertLen(function_cache, 1)
    self.assertIn(compat.as_bytes('CPU:0'), outputs[0])
    self.assertIn(compat.as_bytes('CPU:1'), outputs[1])
    self.assertIn(compat.as_bytes('CPU:2'), outputs[2])
    self.assertIn(compat.as_bytes('CPU:0'), outputs[3])


def setUpModule():
  # [FUNCTION DOC]
  # Module-level test fixture run once before the tests: enables eager execution globally and
  # splits the first physical CPU into four logical CPU devices. Takes no inputs and returns
  # nothing; side effects are on global runtime configuration (it may raise a runtime error if
  # the runtime was already initialized).
  ops.enable_eager_execution()
  cpus = config.list_physical_devices('CPU')
  # Set 4 virtual CPUs
  config.set_logical_device_configuration(
      cpus[0],
      [
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration(),
      ],
  )


if __name__ == '__main__':
  test.main()
