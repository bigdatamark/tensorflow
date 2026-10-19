# [MODULE DOC]
# Tests for forward-mode automatic differentiation (the forward accumulator, its jvp dispatch
# function, and the pack/push utilities). Contains module-level helpers that compute jvps,
# forward-mode Jacobians, gradients, Hessian-vector products and full Hessians, plus a numeric
# gradient checker comparing forward, backward and finite-difference Jacobians. Test classes
# cover core accumulator behavior, control flow inside functions, Hessians, and batched tangents.
# Defines four module-level traced functions used by the control flow tests. The main block
# enables eager execution (graph mode is not tested) before running the tests.
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

import functools
import gc
import weakref

from absl.testing import parameterized
import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import forwardprop
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import test
from tensorflow.python.util import nest

_X11_35_DERIVATIVES = [
    1.1**3.5, 3.5 * 1.1**2.5, 3.5 * 2.5 * 1.1**1.5, 3.5 * 2.5 * 1.5 * 1.1**0.5
]


# TODO(allenl): Move this somewhere useful once forward gradients are stable.
def _jvp(f, primals, tangents):
  # [FUNCTION DOC]
  # HELPER: small shared test utility computing a jvp with a forward accumulator.
  # Inputs: a function taking the primals as positional arguments, a list of primals and
  # matching tangents. Runs the function under an accumulator and returns a pair: the function
  # output and its jvp (zeros for unconnected outputs).
  """Compute the jacobian of `f` at `primals` multiplied by `tangents`."""
  with forwardprop.ForwardAccumulator(primals, tangents) as acc:
    primals_out = f(*primals)
  return primals_out, acc.jvp(
      primals_out, unconnected_gradients=UnconnectedGradients.ZERO)


def _jacfwd(f, primals):
  # [FUNCTION DOC]
  # HELPER: computes the Jacobian of a function at the primals using forward mode.
  # For each element of each flattened primal, builds a one-hot tangent (zeros elsewhere), takes
  # the jvp, flattens it into a column, and stacks the columns per primal. Returns a structure
  # matching the primals with one Jacobian matrix per primal.
  """Compute the jacobian of `f` at `primals` using forward-mode autodiff."""
  jac_flat = []
  flat_primals = nest.flatten(primals)
  tangent_mask = [
      array_ops.zeros_like(primal, dtype=primal.dtype)
      for primal in flat_primals
  ]
  for primal_index, primal in enumerate(flat_primals):
    primal_vector = array_ops.reshape(primal, [-1])
    primal_vector_length = array_ops.size(primal_vector)
    jac_columns = []
    for element_index in math_ops.range(primal_vector_length):
      mask = array_ops.one_hot(
          element_index, primal_vector_length, dtype=primal.dtype)
      tangent_mask[primal_index] = array_ops.reshape(mask,
                                                     array_ops.shape(primal))
      jac_columns.append(
          nest.map_structure(
              functools.partial(array_ops.reshape, shape=[-1]),
              _jvp(f, primals, nest.pack_sequence_as(primals,
                                                     tangent_mask))[1]))
    jac_flat.append(array_ops_stack.stack(jac_columns, axis=1))
    tangent_mask[primal_index] = array_ops.zeros_like(primal)
  return nest.pack_sequence_as(primals, jac_flat)


def _jvp_batch(f, primal, tangents):
  # [FUNCTION DOC]
  # HELPER: computes jvps for a batch of tangents by vectorizing the single-tangent jvp helper
  # over the batch, with the function wrapped as a traced function. Returns the vectorized
  # result (pair of outputs and jvps, each batched).
  tf_function = def_function.function(f)

  return control_flow_ops.vectorized_map(
      functools.partial(_jvp, tf_function, primal), tangents)


def _jvp_batch_matmul(f, primals, tangent_batch):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring is copied from the single-tangent jvp helper; this actually
  # computes the full forward Jacobian once and multiplies it with each tangent in a batch.
  # HELPER: reference implementation for checking batched jvps.
  # Computes the forward-mode Jacobian, then vectorizes a per-tangent matrix-vector product over
  # the batch of tangents. Returns the batch of jvps.
  """Compute the jacobian of `f` at `primals` multiplied by `tangents`."""
  jac_fwd = _jacfwd(f, primals)

  def jac_mul(tangent):
    # [FUNCTION DOC]
    # HELPER: flattens one tangent, multiplies the precomputed Jacobian by it as a column vector
    # and reshapes the product back to the tangent's shape.
    flat_tangent = array_ops.reshape(tangent, shape=[-1])
    tangent_vector = array_ops.expand_dims(flat_tangent, 1)
    jvp_vector = math_ops.matmul(jac_fwd, tangent_vector)
    return array_ops.reshape(jvp_vector, tangent.shape)

  return control_flow_ops.vectorized_map(jac_mul, tangent_batch)


def _grad(f, argnums=0):
  # [FUNCTION DOC]
  # HELPER: returns a function computing the reverse-mode gradient of f with respect to the
  # positional argument at index argnums (default 0), using a gradient tape, with zeros for
  # unconnected gradients.
  """Return a function which computes the gradient of `f`."""

  def _f(*params):
    # [FUNCTION DOC]
    # HELPER: the returned gradient function. Watches all parameters on a tape, evaluates the
    # wrapped function and returns the gradient of the output with respect to the selected
    # parameter, using zeros where unconnected.
    with backprop.GradientTape() as tape:
      tape.watch(params)
      primals_out = f(*params)
    return tape.gradient(
        primals_out,
        params[argnums],
        unconnected_gradients=UnconnectedGradients.ZERO)

  return _f


def _gradfwd(f, argnums=0, f_out_dtypes=dtypes.float32):
  # [FUNCTION DOC]
  # HELPER: returns a function computing the gradient of f in forward mode with respect to the
  # positional argument at index argnums (default 0), by looping over one-hot tangents without
  # pfor. f_out_dtypes (default 32-bit float) gives the output dtype structure for the loop.
  """Return a function which computes the gradient of `f` in forward mode."""

  def _f(*params):
    # [FUNCTION DOC]
    # HELPER: the returned forward-mode gradient function; loops over one-hot masks of the
    # selected parameter and collects the jvp for each.

    def _single_jvp(param_mask):
      # [FUNCTION DOC]
      # HELPER: runs the wrapped function under an accumulator watching the selected parameter
      # with the given mask tangent and returns the jvp of the output.
      with forwardprop.ForwardAccumulator(
          primals=[params[argnums]], tangents=param_mask) as acc:
        primals_out = f(*params)
      return acc.jvp(primals_out)

    # Building up a function to run with pfor takes a bit too long since we're
    # only running it a handful of times.
    return _vectorize_parameters(
        _single_jvp, [params[argnums]], use_pfor=False, dtype=f_out_dtypes)

  return _f


def _hvp(f, primals, tangents):
  # [FUNCTION DOC]
  # HELPER: computes a forward-over-backward Hessian-vector product. Under an accumulator, a
  # tape watches the primals, evaluates the scalar-valued function (asserting its output is a
  # scalar shape) and takes the gradient; returns the accumulator's jvp of that gradient.
  """Compute a forward-over-back Hessian-vector product."""
  with forwardprop.ForwardAccumulator(primals, tangents) as acc:
    with backprop.GradientTape() as tape:
      tape.watch(primals)
      f_out = f(*primals)
      f_out.shape.assert_is_compatible_with([])
    return acc.jvp(tape.gradient(f_out, primals))


def _vectorize_parameters(f, params, use_pfor, dtype):
  # [FUNCTION DOC]
  # HELPER: calls f once per scalar element of the concatenated parameters, passing a list of
  # one-hot tangent tensors shaped like each parameter. Uses vectorized mapping if use_pfor is
  # true, otherwise a sequential map with the given output dtype structure. Returns the stacked
  # results.
  """Loop over `params`, providing a one-hot mask to `f` for each."""
  parameter_sizes = [array_ops.size(param) for param in params]
  total_size = math_ops.add_n(parameter_sizes)

  def _wrapper(index):
    # [FUNCTION DOC]
    # HELPER: builds a one-hot vector over all parameter elements for the given index, splits it
    # per parameter, reshapes each piece to the parameter's shape and calls the wrapped function.
    full_onehot = array_ops.one_hot(index, total_size)
    split_onehot = array_ops.split(full_onehot, parameter_sizes)
    tangents = [
        array_ops.reshape(v, array_ops.shape(param))
        for param, v in zip(params, split_onehot)
    ]
    return f(tangents)

  if use_pfor:
    return control_flow_ops.vectorized_map(_wrapper, math_ops.range(total_size))

  return map_fn.map_fn(_wrapper, math_ops.range(total_size), dtype)


def _forward_over_back_hessian(f, params, use_pfor, dtype=None):
  # [FUNCTION DOC]
  # HELPER: computes the full Hessian of a scalar function using forward-over-backward hessian
  # vector products over one-hot vectors. use_pfor selects vectorized mapping versus looping;
  # dtype is required when looping. Returns a structure of Hessian slices per parameter.
  """Computes the full Hessian matrix for the scalar-valued f(*params).

  Args:
    f: A function taking `params` and returning a scalar.
    params: A possibly nested structure of tensors.
    use_pfor: If true, uses `tf.vectorized_map` calls instead of looping.
    dtype: Required if `use_pfor=False`. A possibly nested structure of dtypes
      (e.g. `tf.float32`) matching the structure of `f`'s returns.

  Returns:
    A possibly nested structure of matrix slices corresponding to `params`. Each
    slice has shape [P, p_s] where `p_s` is the number of parameters (`tf.size`)
    in the corresponding element of `params` and `P` is the total number of
    parameters (`sum_s(p_s)`). The full matrix can be obtained by concatenating
    along the second axis.
  """
  return _vectorize_parameters(
      functools.partial(_hvp, f, params),
      params,
      use_pfor=use_pfor,
      dtype=dtype)


def _test_gradients(testcase,
                    f,
                    primals,
                    order,
                    delta=1e-3,
                    rtol=1e-2,
                    atol=1e-6,
                    srtol=1e-6,
                    satol=1e-6):
  # [FUNCTION DOC]
  # HELPER: shared numeric gradient checker. Inputs: the test case, a function, a list of
  # primals, the gradient order (positive integer), and tolerances: a finite difference delta,
  # relative and absolute tolerances for numeric comparison, and tighter relative and absolute
  # tolerances (srtol, satol) for comparing forward and backward symbolic Jacobians. Raises a
  # value error if order is below one. For order above one it first recursively checks the
  # reverse gradient function at the lower order. Then it asserts the numeric Jacobian matches the
  # symbolic backward Jacobian and the forward Jacobian, and that the two symbolic ones match
  # closely.
  """Tests forward/backward jacobians of `f`'s [0, `order`)-order gradients."""
  if order < 1:
    raise ValueError(
        "`order` should be a positive integer, got '{}'.".format(order))
  if order > 1:
    _test_gradients(
        testcase=testcase,
        f=_grad(f),
        primals=primals,
        order=order - 1,
        delta=delta,
        rtol=rtol,
        atol=atol,
        srtol=srtol,
        satol=satol)
  sym_jac_back, num_jac = gradient_checker_v2.compute_gradient(
      f, primals, delta=delta)
  testcase.assertAllClose(num_jac, sym_jac_back, rtol=rtol, atol=atol)
  sym_jac_fwd = _jacfwd(f, primals)
  testcase.assertAllClose(num_jac, sym_jac_fwd, rtol=rtol, atol=atol)
  # And the symbolic computations should be much closer.
  testcase.assertAllClose(sym_jac_back, sym_jac_fwd, rtol=srtol, atol=satol)


@test_util.with_eager_op_as_function
class ForwardpropTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for the forward accumulator and its jvp dispatch function; parameterized,
  # run with eager ops executed as functions. No shared fixtures.

  def testJVPFunction(self):
    # [FUNCTION DOC]
    # Calls the jvp dispatch function directly for an add op and a multiply op with given
    # tangents, and asserts the results are the sum of tangents and the product rule values.
    add_outputs = (constant_op.constant(4.),)
    vp, = forwardprop._jvp_dispatch(
        op_name="Add",
        attr_tuple=(),
        inputs=(constant_op.constant(1.), constant_op.constant(3.)),
        outputs=add_outputs,
        tangents=(
            constant_op.constant(1.),
            constant_op.constant(5.),
        ))
    self.assertAllClose(1. + 5., self.evaluate(vp))

    mul_outputs = (constant_op.constant([20.]),)
    vp, = forwardprop._jvp_dispatch(
        op_name="Mul",
        attr_tuple=(),
        inputs=(constant_op.constant([4.]), constant_op.constant([5.])),
        outputs=mul_outputs,
        tangents=(
            constant_op.constant([2.]),
            constant_op.constant([3.]),
        ))
    self.assertAllClose([2. * 5. + 3. * 4.], self.evaluate(vp))

  def testJVPFunctionWithBatchOfTangents(self):
    # [FUNCTION DOC]
    # Calls the jvp dispatch function in batch mode with batches of tangents for add and
    # multiply, and asserts the per-batch-element results.
    add_outputs = (constant_op.constant(4.),)
    jvp_flat = forwardprop._jvp_dispatch(
        op_name="Add",
        attr_tuple=(),
        inputs=(constant_op.constant(1.), constant_op.constant(3.)),
        outputs=add_outputs,
        tangents=(
            constant_op.constant([1., 2., 3.]),
            constant_op.constant([4., 5., 6.]),
        ),
        use_batch=True)

    # Using evaluate and asserting with just a list works too
    # but the output is more explicit this way
    self.assertAllClose([constant_op.constant([1. + 4., 2. + 5., 3. + 6.])],
                        jvp_flat)

    mul_outputs = (constant_op.constant([20.]),)
    jvp_flat = forwardprop._jvp_dispatch(
        op_name="Mul",
        attr_tuple=(),
        inputs=(constant_op.constant([4.]), constant_op.constant([5.])),
        outputs=mul_outputs,
        tangents=(
            constant_op.constant([[1.], [0.], [1.]]),
            constant_op.constant([[0.], [1.], [1.]]),
        ),
        use_batch=True)
    self.assertAllClose([constant_op.constant([[5.], [4.], [5. + 4.]])],
                        jvp_flat)

  def testJVPFunctionRaisesError(self):
    # [FUNCTION DOC]
    # Calls the batch jvp dispatch with tangents whose shapes do not match the primals and
    # asserts a value error saying the tangent was expected to be of another shape.
    sum_outputs = (constant_op.constant(6.),)

    with self.assertRaisesRegex(ValueError, r".*was expected to be of shape*"):
      forwardprop._jvp_dispatch(
          op_name="Add",
          attr_tuple=(),
          inputs=(constant_op.constant(2.), constant_op.constant(4.)),
          outputs=sum_outputs,
          tangents=(constant_op.constant([1., 2.]),
                    constant_op.constant([[1.], [2.]])),
          use_batch=True)

  def testNonDifferentiableOpWithInputTangent(self):
    # [FUNCTION DOC]
    # Nested accumulators watch the same tensor; a zeros-like op (non-differentiable) is run.
    # Asserts both accumulators return no jvp for the result.
    x = constant_op.constant(1.)
    with forwardprop.ForwardAccumulator(x, 2.) as acc1:
      with forwardprop.ForwardAccumulator(x, 2.) as acc2:
        y = array_ops.zeros_like(x)
      self.assertIsNone(acc1.jvp(y))
    self.assertIsNone(acc2.jvp(y))

  def testRunFunctionsEagerly(self):
    # [FUNCTION DOC]
    # Enables the global run-functions-eagerly setting (restored afterward even on failure),
    # multiplies a watched tensor by three under an accumulator, and asserts the jvp is six,
    # i.e. forward mode still works when traced functions run eagerly.
    try:
      original_setting = def_function.functions_run_eagerly()
      def_function.run_functions_eagerly(True)
      x = constant_op.constant(1.)
      with forwardprop.ForwardAccumulator(x, 2.) as acc:
        y = x * 3.
      self.assertAllClose(6., acc.jvp(y))
    finally:
      def_function.run_functions_eagerly(original_setting)

  def testJVPFunctionUsedByAccumulatorForOps(self):
    # [FUNCTION DOC]
    # Within an accumulator, runs an add, then replaces the registered native jvp function with
    # one returning a constant, and runs another add. Asserts the first jvp is the real value
    # and the second is the constant, proving the accumulator calls the registered function. The
    # original function is re-registered afterward.
    previous_fn = forwardprop._jvp_dispatch
    try:
      x = constant_op.constant(1.)
      with forwardprop.ForwardAccumulator(x, 2.) as acc:
        y = x + x
        pywrap_tfe.TFE_Py_RegisterJVPFunction(
            lambda *args, **kwargs: [constant_op.constant(-15.)])
        z = x + x
      self.assertAllClose(4., acc.jvp(y))
      self.assertAllClose(-15., acc.jvp(z))
    finally:
      pywrap_tfe.TFE_Py_RegisterJVPFunction(previous_fn)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testFunctionCacheLimited(self):
    # [FUNCTION DOC]
    # Runs adds on tensors of growing size, twice the trace count limit times, so both exact and
    # relaxed shape tracing are used, and asserts the jvp is always twice the ones tangent.
    # Also checks no new Python objects leak.
    # Every time this loop is executed, it will create a slightly larger Tensor
    # and push it through Add's gradient.
    # We run TRACE_COUNT_LIMIT x 2 so that it is tested with both
    # experimental_relax_shapes on and off.
    for execution_count in range(forwardprop._TRACE_COUNT_LIMIT*2):
      x = array_ops.zeros([execution_count])
      with forwardprop.ForwardAccumulator(x, array_ops.ones_like(x)) as acc:
        y = x + x
      self.assertAllClose(2. * array_ops.ones_like(x), acc.jvp(y))

  def testVariableUnwatchedZero(self):
    # [FUNCTION DOC]
    # Asks an accumulator for the jvp of a variable it does not watch. Asserts the default is no
    # value and that requesting zero for unconnected gradients gives zeros of the variable's shape.
    v = variables.Variable([[1.]])
    x = constant_op.constant(1.)
    xt = constant_op.constant(2.)
    with forwardprop.ForwardAccumulator(x, xt) as acc:
      pass
    self.assertIsNone(acc.jvp(v))
    self.assertAllClose([[0.]], acc.jvp(v, unconnected_gradients="zero"))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testFunctionReturnsResource(self):
    # [FUNCTION DOC]
    # Calls a traced function that returns an input tensor and a variable handle under an
    # accumulator, and asserts the jvp of the tensor output is still the tangent (resource
    # outputs must not break forward propagation).
    v = variables.Variable([[1.]])
    x = constant_op.constant(1.)
    xt = constant_op.constant(2.)

    @def_function.function
    def f(a):
      # [FUNCTION DOC]
      # HELPER: returns its input together with the handle of the captured variable.
      return a, v.handle

    with forwardprop.ForwardAccumulator(x, xt) as acc:
      y, _ = f(x)
    self.assertAllClose(2., acc.jvp(y))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testMultipleWatchesAdd(self):
    # [FUNCTION DOC]
    # Asserts that listing a tensor twice as a primal raises a value error mentioning multiple
    # times, then that watching the same tensor repeatedly through the private watch method sums
    # the tangents (3, then 13, then 24), and that downstream results use the summed tangent.
    x = constant_op.constant(-2.)
    with self.assertRaisesRegex(ValueError, "multiple times"):
      with forwardprop.ForwardAccumulator([x, x], [1., 2.]):
        pass
    with forwardprop.ForwardAccumulator([x], [3.]) as acc:
      self.assertAllClose(3., acc.jvp(x))
      acc._watch(x, constant_op.constant(10.))
      self.assertAllClose(13., acc.jvp(x))
      acc._watch(x, constant_op.constant(11.))
      self.assertAllClose(24., acc.jvp(x))
      y = constant_op.constant(3.) * x
    self.assertAllClose(24., acc.jvp(x))
    self.assertAllClose(24. * 3., acc.jvp(y))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testReenter(self):
    # [FUNCTION DOC]
    # Asserts that entering an already active accumulator raises a value error (already
    # recording), that operations after exit are not tracked, and that re-entering after exit
    # works and propagates a jvp through a product.
    x = constant_op.constant(-2.)
    with forwardprop.ForwardAccumulator(x, 1.5) as acc:
      self.assertAllClose(1.5, acc.jvp(x))
      y = 4. * x
      self.assertAllClose(6., acc.jvp(y))
      with self.assertRaisesRegex(ValueError, "already recording"):
        with acc:
          pass
    z = 4. * x
    self.assertIsNone(acc.jvp(z))
    with acc:
      yy = y * y
    self.assertAllClose(6. * -8. * 2., acc.jvp(yy))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDeadTensorsJVPCleared(self):
    # [FUNCTION DOC]
    # Deletes the watched primal and its tangent tensor, then asserts via weak references that
    # they are freed (the accumulator holds no strong reference), and that derived tensors and
    # their jvps are also freed once deleted.
    x = array_ops.ones([100])
    x_weak = weakref.ref(x)
    grad_tensor = constant_op.constant(array_ops.zeros([100]))
    grad_tensor_weak = weakref.ref(grad_tensor)
    with forwardprop.ForwardAccumulator(x, grad_tensor) as acc:
      derived_tensor = constant_op.constant(2.) * x
      del grad_tensor
      self.assertAllClose(array_ops.zeros([100]), acc.jvp(x))
      del x
      self.assertIsNone(x_weak())
      self.assertIsNone(grad_tensor_weak())
      derived_tensor_weak = weakref.ref(derived_tensor)
      derived_tensor_grad = acc.jvp(derived_tensor)
      derived_tensor_grad_weak = weakref.ref(derived_tensor_grad)
      del derived_tensor
      del derived_tensor_grad
      self.assertIsNone(derived_tensor_weak())
      self.assertIsNone(derived_tensor_grad_weak())

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testJVPManual(self):
    # [FUNCTION DOC]
    # Uses the jvp helper on a sine function with given primal and tangent; asserts the primal
    # output equals the sine and the tangent equals cosine times the input tangent.
    primal, tangent = _jvp(math_ops.sin, (constant_op.constant(0.1),),
                           (constant_op.constant(0.2),))
    self.assertAllClose(math_ops.sin(0.1), primal)
    self.assertAllClose(math_ops.cos(0.1) * 0.2, tangent)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNumericHigherOrder(self):
    # [FUNCTION DOC]
    # Runs the numeric gradient checker up to third order on a function mixing sine, tangent and
    # sums over a 2 by 2 float tensor, with a looser absolute tolerance for symbolic agreement.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: the function under test; multiplies sine by tangent elementwise, adds the total
      # sum, and takes the product over axis one.
      pointwise = math_ops.sin(x) * math_ops.tan(x)
      return math_ops.reduce_prod(
          pointwise + math_ops.reduce_sum(pointwise), axis=1)

    _test_gradients(
        self,
        f,
        [constant_op.constant([[2.0, 3.0], [1.0, 4.0]])],
        order=3,
        srtol=1e-6,
        satol=1e-3,
    )

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNumericHigherOrderFloat64(self):
    # [FUNCTION DOC]
    # Same as the previous numeric higher-order test but with 64-bit float inputs and default
    # tolerances.

    def f(x):
      # [FUNCTION DOC]
      # HELPER: the function under test; sine times tangent elementwise, plus total sum, then
      # product over axis one.
      pointwise = math_ops.sin(x) * math_ops.tan(x)
      return math_ops.reduce_prod(
          pointwise + math_ops.reduce_sum(pointwise), axis=1)

    _test_gradients(
        self,
        f,
        [constant_op.constant([[2.0, 3.0], [1.0, 4.0]], dtype=dtypes.float64)],
        order=3)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testCustomGradient(self):
    # [FUNCTION DOC]
    # Defines a function with a custom gradient (sine computed in numpy, gradient using cosine)
    # and runs the numeric gradient checker to third order.

    @custom_gradient.custom_gradient
    def f(x):
      # [FUNCTION DOC]
      # HELPER: the function with a custom gradient; returns numpy sine of the input and the
      # gradient function.

      def grad(dy):
        # [FUNCTION DOC]
        # HELPER: custom gradient; multiplies the upstream gradient by the cosine of the input.
        return dy * math_ops.cos(x)

      return np.sin(x.numpy()), grad

    _test_gradients(self, f, [constant_op.constant([1., 2.])], order=3)

  # TODO(allenl): investigate why assert_no_new_pyobjects_executing_eagerly()
  # fails around this test?
  def testExceptionCustomGradientRecomputeGradForward(self):
    # [FUNCTION DOC]
    # Computes a forward Jacobian of a function wrapped with the recompute-gradient decorator and
    # asserts a not-implemented error saying recompute_grad tried to transpose.

    @custom_gradient.recompute_grad
    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; product of squared hyperbolic tangents.
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    with self.assertRaisesRegex(NotImplementedError,
                                "recompute_grad tried to transpose"):
      primals = [constant_op.constant([1.])]
      sym_jac_fwd = _jacfwd(f, primals)

  def testExceptionInCustomGradientNotSwallowed(self):
    # [FUNCTION DOC]
    # Under an accumulator, calls a custom-gradient function whose gradient raises a value error
    # with a known message; asserts that exact error propagates to the caller.

    @custom_gradient.custom_gradient
    def f(unused_x):
      # [FUNCTION DOC]
      # HELPER: returns a constant and a gradient function that always raises.

      def grad(unused_dy):
        # [FUNCTION DOC]
        # HELPER: always raises a value error with the test error string.
        raise ValueError("test_error_string")

      return 1., grad

    c = constant_op.constant(1.)
    d = constant_op.constant(2.)
    with forwardprop.ForwardAccumulator(c, d):
      with self.assertRaisesRegex(ValueError, "test_error_string"):
        f(c)

  @parameterized.named_parameters([("EluM5", -0.5, nn_ops.elu),
                                   ("EluP5", [0.5], nn_ops.elu),
                                   ("SwishP5", 0.5, nn_impl.swish),
                                   ("SwishM5", [-0.5], nn_impl.swish)])
  def testElementwiseNNOps(self, value, op_fn):
    # [FUNCTION DOC]
    # Parameterized over elu and swish with scalar and list inputs; runs the numeric gradient
    # checker to third order for the given op function on the given value.
    _test_gradients(self, op_fn, [constant_op.constant(value)], order=3)

  def testFusedBatchNormGradsInference(self):
    # [FUNCTION DOC]
    # Builds a small 4D input with fixed scale, offset, mean and variance and checks gradients to
    # second order (with a looser absolute tolerance) of fused batch norm in inference mode with
    # respect to input, scale and offset.

    x_shape = [4, 10, 10, 2]
    increment = 3. / math_ops.reduce_prod(
        constant_op.constant(x_shape, dtype=dtypes.float32))
    x = array_ops.reshape(math_ops.range(-2., 1., increment), x_shape)
    scale = constant_op.constant([1., 1.1])
    offset = constant_op.constant([-0.5, -0.6])
    mean = constant_op.constant([-1.3, 1.4])
    variance = constant_op.constant([0.7, 0.9])
    epsilon = 0.001

    def _bn_fused(x_arg, scale_arg, offset_arg):
      # [FUNCTION DOC]
      # HELPER: applies fused batch norm in inference mode with the fixed mean, variance and
      # epsilon from the enclosing test and returns only the normalized output.
      return nn_impl.fused_batch_norm(
          x_arg,
          scale_arg,
          offset_arg,
          mean,
          variance,
          epsilon=epsilon,
          is_training=False)[0]

    _test_gradients(self, _bn_fused, [x, scale, offset], order=2, atol=1e-2)

  def testPushPopAccumulatorState(self):
    # [FUNCTION DOC]
    # Defines a custom-gradient function whose gradient pushes forwardprop state, watches a copy
    # of the input with the incoming gradient as tangent, and reads its jvp. Asserts the outer
    # accumulator's jvp of the function output equals tangent times cosine of the primal.
    # Note that this example is somewhat contrived. push_forwardprop_state is
    # probably only useful in practice for building functions that compute jvps
    # alongside their usual outputs.
    c = constant_op.constant(1.)
    d = constant_op.constant(2.)
    with forwardprop.ForwardAccumulator(c, d) as acc:

      @custom_gradient.custom_gradient
      def f(x):
        # [FUNCTION DOC]
        # HELPER: custom-gradient sine; returns sine of the input and a gradient function.
        y = math_ops.sin(x.numpy())

        def grad(dy):
          # [FUNCTION DOC]
          # HELPER: within pushed forwardprop state, watches a copy of the input with the given
          # tangent, computes its sine and returns the upstream value times its jvp.
          with forwardprop_util.push_forwardprop_state():
            x_copy = constant_op.constant(x.numpy())
            acc._watch(x_copy, dy)
            y_copy = math_ops.sin(x_copy)
          return dy * acc.jvp(y_copy)

        return y, grad

      output = f(c)
      self.assertAllClose(d * math_ops.cos(c), acc.jvp(output))

  @parameterized.named_parameters([
      ("Order{}".format(order), order, expected)
      for order, expected in enumerate(_X11_35_DERIVATIVES)
  ])
  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testHigherOrderPureForward(self, order, expected):
    # [FUNCTION DOC]
    # Parameterized by derivative order and expected value. Repeatedly wraps a power function in a
    # forward-gradient transformation order times (nested accumulators) and asserts the result at
    # a point equals the expected analytic derivative.

    def _forwardgrad(f):
      # [FUNCTION DOC]
      # HELPER: returns a function computing the forward-mode derivative of f at a scalar point.

      def _compute_forwardgrad(primal):
        # [FUNCTION DOC]
        # HELPER: watches the primal with a unit tangent, evaluates f and returns the jvp.
        tangent = constant_op.constant(1.)
        with forwardprop.ForwardAccumulator(primal, tangent) as acc:
          primal_out = f(primal)
        return acc.jvp(primal_out)

      return _compute_forwardgrad

    def _forward(x):
      # [FUNCTION DOC]
      # HELPER: the base function, raising the input to the power three and a half.
      return x**3.5

    f = _forward
    primal = constant_op.constant(1.1)
    for _ in range(order):
      f = _forwardgrad(f)
    self.assertAllClose(expected, f(primal))

  @parameterized.named_parameters([("Function", def_function.function),
                                   ("NoFunction", lambda f: f)])
  def testGradPureForward(self, decorator):
    # [FUNCTION DOC]
    # Parameterized by whether the function is traced or plain. With two nested accumulators on
    # the same primal, asserts the output, inner jvp (first derivative), outer jvp of the inner
    # (second derivative), and that the reversed-order lookup gives no value.

    @decorator
    def f(x):
      # [FUNCTION DOC]
      # HELPER: raises the input to the power three and a half.
      return x**3.5

    primal = constant_op.constant(1.1)
    with forwardprop.ForwardAccumulator(primal,
                                        constant_op.constant(1.)) as outer_acc:
      with forwardprop.ForwardAccumulator(primal,
                                          constant_op.constant(1.)) as acc:
        primal_out = f(primal)
    inner_jvp = acc.jvp(primal_out)
    outer_jvp = outer_acc.jvp(inner_jvp)
    self.assertAllClose(1.1**3.5, primal_out)
    self.assertAllClose(3.5 * 1.1**2.5, inner_jvp)
    self.assertAllClose(3.5 * 2.5 * 1.1**1.5, outer_jvp)
    self.assertIsNone(acc.jvp(outer_acc.jvp(primal_out)))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testJVPPacking(self):
    # [FUNCTION DOC]
    # Nests two accumulators (the outer also watching the inner tangent) and calls the tangent
    # packing utility on the input and then on the output of a multiplication. Asserts the packed
    # tangents and the index structure (which accumulator watches which tensor) and the jvp values.
    two = constant_op.constant(2.)
    primal_in = constant_op.constant(1.)
    inner_jvp = constant_op.constant(3.)
    with forwardprop.ForwardAccumulator(
        [primal_in, inner_jvp],
        [constant_op.constant(2.),
         constant_op.constant(4.)]) as outer_acc:
      with forwardprop.ForwardAccumulator(primal_in, inner_jvp) as inner_acc:
        packed_input_indices, packed_input_tangents = (
            forwardprop_util.pack_tangents([primal_in]))
        self.assertAllClose([3., 2., 4.], packed_input_tangents)
        expected_indices = (
            # inner_acc watches primal_in
            (
                (0, 1),),
            # outer_acc watches primal_in and inner_jvp
            ((0, 2), (1, 3)))
        self.assertAllEqual(expected_indices, packed_input_indices)
        primal_out = primal_in * two
        self.assertAllClose(6., inner_acc.jvp(primal_out))
        self.assertAllClose(4., outer_acc.jvp(primal_out))
        self.assertAllClose(8., outer_acc.jvp(inner_acc.jvp(primal_out)))
        packed_output_indices, packed_output_tangents = (
            forwardprop_util.pack_tangents([primal_out]))
        self.assertAllClose([6., 4., 8.], packed_output_tangents)
        self.assertAllEqual(expected_indices, packed_output_indices)

  def testFunctionGradInFunctionPureForward(self):
    # [FUNCTION DOC]
    # Like the nested pure-forward test, but the accumulators are created inside a traced function
    # that calls another traced function. Asserts the output, first and second derivatives.

    @def_function.function
    def take_gradients():
      # [FUNCTION DOC]
      # HELPER: inside a traced function, runs the nested accumulators and returns the output,
      # inner jvp and outer jvp; asserts the reversed lookup gives no value.

      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: raises the input to the power three and a half.
        return x**3.5

      primal = constant_op.constant(1.1)
      with forwardprop.ForwardAccumulator(
          primal, constant_op.constant(1.)) as outer_acc:
        with forwardprop.ForwardAccumulator(primal,
                                            constant_op.constant(1.)) as acc:
          primal_out = f(primal)
      inner_jvp = acc.jvp(primal_out)
      outer_jvp = outer_acc.jvp(inner_jvp)
      self.assertIsNone(acc.jvp(outer_acc.jvp(primal_out)))
      return primal_out, inner_jvp, outer_jvp

    primal_out, inner_jvp, outer_jvp = take_gradients()
    self.assertAllClose(1.1**3.5, primal_out)
    self.assertAllClose(3.5 * 1.1**2.5, inner_jvp)
    self.assertAllClose(3.5 * 2.5 * 1.1**1.5, outer_jvp)

  def testFunctionGrad(self):
    # [FUNCTION DOC]
    # Runs the numeric gradient checker to third order on a traced function (product of squared
    # hyperbolic tangents).

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # HELPER: function under test; product of squared hyperbolic tangents.
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    _test_gradients(self, f, [constant_op.constant([1., 2.])], order=3)

  def testReusingJVP(self):
    # [FUNCTION DOC]
    # Calls a traced matrix-multiply on two identical large matrices with different tangents
    # under one accumulator (testing that cached jvp functions are not wrongly shared). Asserts
    # the two outputs match and each jvp equals the product-rule value for its own tangent.

    # Inner helper below computes the expected jvp.
    m1 = random_ops.random_uniform((256, 2096))
    m2 = array_ops.identity(m1)
    tangent1 = random_ops.random_uniform((256, 2096))
    tangent2 = random_ops.random_uniform((256, 2096))
    matmul = def_function.function(math_ops.matmul)

    with forwardprop.ForwardAccumulator(
        primals=[m1, m2], tangents=[tangent1, tangent2]) as acc:
      result1 = matmul(m1, m1, transpose_b=True)
      result2 = matmul(m2, m2, transpose_b=True)

    def _expected(mat, tangent):
      # [FUNCTION DOC]
      # HELPER: expected jvp of multiplying a matrix by its own transpose: tangent times matrix
      # transposed plus matrix times tangent transposed.
      return (math_ops.matmul(tangent, mat, transpose_b=True) +
              math_ops.matmul(mat, tangent, transpose_b=True))

    self.assertAllClose(result1, result2)
    self.assertAllClose(_expected(m1, tangent1), acc.jvp(result1))
    self.assertAllClose(_expected(m2, tangent2), acc.jvp(result2))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testHVPMemory(self):
    # [FUNCTION DOC]
    # Computes a forward-over-backward Hessian-vector product once, only checking that no new
    # Python objects are leaked (decorator-based check); no value is asserted.

    def fun(x):
      # [FUNCTION DOC]
      # HELPER: product of squared hyperbolic tangents, a scalar-valued function.
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    primals = constant_op.constant([1., 2., 3.])
    tangents = constant_op.constant([3., 4., 5.])
    _hvp(fun, (primals,), (tangents,))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testHVPCorrectness(self):
    # [FUNCTION DOC]
    # Computes the Hessian-vector product by forward-over-backward both eagerly and in a traced
    # function, and via a full Hessian from nested tapes (backward-over-backward) multiplied by the
    # vector; asserts all agree.

    def fun(x):
      # [FUNCTION DOC]
      # HELPER: product of squared hyperbolic tangents, a scalar-valued function.
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    primals = constant_op.constant([1., 2., 3.])
    tangents = constant_op.constant([3., 4., 5.])
    forwardback_hvp_eager, = _hvp(fun, (primals,), (tangents,))
    forwardback_hvp_function, = def_function.function(_hvp)(fun, (primals,),
                                                            (tangents,))

    with backprop.GradientTape(persistent=True) as g:
      g.watch(primals)
      with backprop.GradientTape() as gg:
        gg.watch(primals)
        out = fun(primals)
      grad = array_ops_stack.unstack(gg.gradient(out, primals))
    hessian = []
    for i in range(3):
      hessian.append(g.gradient(grad[i], primals))
    hessian = array_ops_stack.stack(hessian, axis=0)
    backback_hvp = math_ops.tensordot(hessian, tangents, axes=1)

    self.assertAllClose(backback_hvp, forwardback_hvp_eager)
    self.assertAllClose(backback_hvp, forwardback_hvp_function)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testShouldRecordAndStopRecord(self):
    # [FUNCTION DOC]
    # Inside an accumulator and a tape, checks the possible-gradient-types code and the
    # should-record query before and after the tape watches a tensor, and inside a
    # stop-recording scope (nothing recorded). Asserts a tensor created while recording was
    # stopped has no jvp and no gradient.
    c = constant_op.constant(1.)
    c_tangent = constant_op.constant(2.)
    with forwardprop.ForwardAccumulator(c, c_tangent) as acc:
      with backprop.GradientTape() as tape:
        self.assertFalse(record.should_record_backprop([c]))
        self.assertEqual(1, pywrap_tfe.TFE_Py_TapeSetPossibleGradientTypes([c]))
        tape.watch(c)
        self.assertEqual(2, pywrap_tfe.TFE_Py_TapeSetPossibleGradientTypes([c]))
        self.assertTrue(record.should_record_backprop([c]))
        with record.stop_recording():
          self.assertEqual(0,
                           pywrap_tfe.TFE_Py_TapeSetPossibleGradientTypes([c]))
          self.assertFalse(record.should_record_backprop([c]))
          d = c * 2.
        self.assertEqual(2, pywrap_tfe.TFE_Py_TapeSetPossibleGradientTypes([c]))
        self.assertTrue(record.should_record_backprop([c]))
        self.assertFalse(record.should_record_backprop([d]))
        self.assertIsNone(acc.jvp(d))
      self.assertIsNone(tape.gradient(d, c))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testRecordingSelectively(self):
    # [FUNCTION DOC]
    # Creates tensors with recording stopped, then manually records one multiplication for
    # forward-mode only and another for backward-mode only. Asserts the forward-only one has a jvp
    # but no gradient and the backward-only one has a gradient but no jvp.
    c = constant_op.constant(1.)
    c_tangent = constant_op.constant(2.)
    with forwardprop.ForwardAccumulator(c, c_tangent) as acc:
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(c)
        with record.stop_recording():
          two = constant_op.constant(2.)
          d = c * two
          three = constant_op.constant(3.)
          e = c * three
        self.assertIsNone(acc.jvp(d))
        self.assertIsNone(acc.jvp(e))
        self.assertIsNone(tape.gradient(d, c))
        self.assertIsNone(tape.gradient(e, c))
        record.record_operation_forwardprop_only(
            "CustomForwardMul", [d], [c, two], lambda dd: (two * dd, c * dd),
            None)
        record.record_operation_backprop_only("CustomBackwardMul", [e],
                                              [c, three], lambda de:
                                              (three * de, c * de))
        self.assertAllClose(4., acc.jvp(d))
        self.assertIsNone(acc.jvp(e))
        self.assertIsNone(tape.gradient(d, c))
        self.assertAllClose(3., tape.gradient(e, c))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testOpWithNoTrainableOutputs(self):
    # [FUNCTION DOC]
    # Under an accumulator watching a variable, runs an in-place subtraction (an op without
    # trainable outputs) and asserts the variable's new value, i.e. no crash occurs.
    v = variables.Variable(1.)
    with forwardprop.ForwardAccumulator(v, 11.):
      v.assign_sub(0.5)
      self.assertAllClose(0.5, self.evaluate(v))

  # TODO(b/141025187): Add a no_new_pyobjects decorator.
  def testVariableReadInFunction(self):
    # [FUNCTION DOC]
    # Under an accumulator watching a variable, calls a traced function that reads the variable
    # and a doubled read; asserts the values and that the jvps are the tangent and twice it.
    v = variables.Variable(1.)
    with forwardprop.ForwardAccumulator(v, 11.) as acc:

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: returns the variable value and twice the variable value.
        return v.read_value(), 2. * v.read_value()

      result = f()
      self.assertAllClose((1.0, 2.), result)
      self.assertAllClose((11., 22.), acc.jvp(result))

  @parameterized.named_parameters([("ForwardPropFirst", True),
                                   ("TapeFirst", False)])
  def testForwardOverBackwardMemoryEfficiency(self, forward_prop_first):
    # [FUNCTION DOC]
    # Parameterized by whether the accumulator or tape is created first (nesting, not creation
    # order, decides watching). With the garbage collector disabled, asserts that a tape inside
    # an accumulator does not record the jvp computation and does not keep intermediate tensors
    # alive, and that the tape yields no gradient of the jvp.
    # Watching depends on nesting, not creation order
    c = constant_op.constant(1.)
    if forward_prop_first:
      forward_accumulator = forwardprop.ForwardAccumulator(c, .1)
      gradient_tape = backprop.GradientTape()
    else:
      gradient_tape = backprop.GradientTape()
      forward_accumulator = forwardprop.ForwardAccumulator(c, .1)
    try:
      gc.disable()
      with gradient_tape as tape:
        # Adding and removing the tape multiple times in different nesting
        # patterns does not affect watch ordering.
        pass
      with forward_accumulator as acc:
        with gradient_tape as tape:
          tape.watch(c)
          d = math_ops.cos(c)
          self.assertFalse(record.should_record_backprop((acc.jvp(d),)))
          e = math_ops.cos(acc.jvp(d))
          math_ops.cos(e)
          weak_e = weakref.ref(e)
          del e
          self.assertIsNone(weak_e())
        self.assertIsNone(tape.gradient(acc.jvp(d), c))
    finally:
      gc.enable()

  @parameterized.named_parameters([("ForwardPropFirst", True),
                                   ("TapeFirst", False)])
  def testBackwardOverForward(self, forward_prop_first):
    # [FUNCTION DOC]
    # Parameterized by creation order. With the tape outside the accumulator, asserts the tape
    # records the jvp computation and its gradient equals the expected second derivative of cosine
    # scaled by the tangent.
    c = constant_op.constant(1.)
    # Watching depends on nesting, not creation order
    if forward_prop_first:
      forward_accumulator = forwardprop.ForwardAccumulator(c, .1)
      gradient_tape = backprop.GradientTape()
    else:
      gradient_tape = backprop.GradientTape()
      forward_accumulator = forwardprop.ForwardAccumulator(c, .1)
    with gradient_tape as tape:
      with forward_accumulator as acc:
        tape.watch(c)
        d = math_ops.cos(c)
        self.assertTrue(record.should_record_backprop((acc.jvp(d),)))
      self.assertAllClose(-.1 * math_ops.cos(1.), tape.gradient(acc.jvp(d), c))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testRecordingWithJVPIndices(self):
    # [FUNCTION DOC]
    # Packs the tangent of a watched tensor and manually records a forward-only operation that
    # carries its own inline jvp indices; asserts the accumulator reads the output's jvp from
    # the supplied tangent.
    c = constant_op.constant(1.)
    with forwardprop.ForwardAccumulator(c, 10.) as acc:
      packed_input_tangents = forwardprop_util.pack_tangents([c]).tangents
      self.assertAllClose([10.], packed_input_tangents)
      d = constant_op.constant(2.)
      d_tangent = constant_op.constant(3.)
      record.record_operation_forwardprop_only("FunctionWithInlineJVPs",
                                               [d] + [d_tangent],
                                               [c] + packed_input_tangents,
                                               None, (((0, 1),),))
      self.assertAllClose(3., acc.jvp(d))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testSpecialForwardFunctionUsed(self):
    # [FUNCTION DOC]
    # Records operations with custom forward functions (one scaling the tangent, one with no
    # inputs, one passing it through) and asserts the accumulator uses them, giving minus twenty
    # for the chained results.
    c = constant_op.constant(1.)
    d = constant_op.constant(2.)
    e = constant_op.constant(3.)
    with forwardprop.ForwardAccumulator(c, 10.) as acc:
      record.record_operation("ForwardIsSpecial", [d], [c], None,
                              lambda jvp: [-2. * jvp])
      self.assertAllClose(-20., acc.jvp(d))
      record.record_operation("ForwardIsSpecial2", [], [], None, lambda: [])
      record.record_operation("ForwardIsSpecial3", [e], [d], None,
                              lambda x: [x])
      self.assertAllClose(-20., acc.jvp(e))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testVariableWatched(self):
    # [FUNCTION DOC]
    # Watches a variable with a tangent vector and asserts the jvp of the variable itself and of
    # expressions scaling it or adding a constant.
    v = variables.Variable([1., 2., 3.])
    with forwardprop.ForwardAccumulator(v, constant_op.constant([.1, -.2,
                                                                 .3])) as acc:
      self.assertAllClose([.1, -.2, .3], acc.jvp(v))
      x = v * 2.
      self.assertAllClose([.2, -.4, .6], acc.jvp(x))
      x2 = v + .1
      self.assertAllClose([.1, -.2, .3], acc.jvp(x2))

  def testUnconnectedGradients(self):
    # [FUNCTION DOC]
    # Asserts the unconnected-gradients option for jvp lookups: watched tensor gives its tangent
    # in both modes; an unwatched tensor gives zero in zero mode and no value in none mode.
    x = constant_op.constant(-1.)
    with forwardprop.ForwardAccumulator(x, 0.1) as acc:
      self.assertAllClose(0.1, acc.jvp(x, unconnected_gradients="zero"))
      self.assertAllClose(0.1, acc.jvp(x, unconnected_gradients="none"))
      y = constant_op.constant(-2.)
      self.assertAllClose(0.0, acc.jvp(y, unconnected_gradients="zero"))
      self.assertIsNone(acc.jvp(y, unconnected_gradients="none"))

  # TODO(kkb): One weakref instance is created with warmup_iters=2,
  # investigate.
  @test_util.assert_no_new_pyobjects_executing_eagerly(warmup_iters=3)
  def testVariableWatchedFunction(self):
    # [FUNCTION DOC]
    # Defines a module whose traced method creates a variable lazily and computes jvps inside the
    # function; calls it and asserts the jvps of the variable and derived values.

    class _Model(module.Module):
      # [CLASS DOC]
      # HELPER: small module holding a lazily created variable and a traced method that computes
      # jvps for it.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: starts with no variable.
        self._v = None

      @def_function.function
      def compute_jvps(self):
        # [FUNCTION DOC]
        # HELPER: creates the variable on first trace, watches it with a fixed tangent, computes
        # a doubled and an offset value, and returns the jvps of the variable and both values.
        if self._v is None:
          self._v = variables.Variable([1., 2., 3.])
        with forwardprop.ForwardAccumulator(self._v,
                                            constant_op.constant([.1, -.2,
                                                                  .3])) as acc:
          x = self._v * 2.
          x2 = self._v + .1
        return acc.jvp((self._v, x, x2))

    model = _Model()
    v_jvp, x_jvp, x2_jvp = model.compute_jvps()
    self.assertAllClose([.1, -.2, .3], v_jvp)
    self.assertAllClose([.2, -.4, .6], x_jvp)
    self.assertAllClose([.1, -.2, .3], x2_jvp)

  def testIndexSlicesGrad(self):
    # [FUNCTION DOC]
    # Gathers an element of a watched vector eagerly (gradient uses sparse index slices) and
    # asserts the jvp equals the corresponding tangent element.
    x = constant_op.constant([1.])

    with forwardprop.ForwardAccumulator(x, constant_op.constant([3.])) as acc:
      y = array_ops.gather(x, 0)
    self.assertAllClose(3., acc.jvp(y))

  def testIndexSlicesGradInFunction(self):
    # [FUNCTION DOC]
    # Same as the eager index slices test but the gather happens in a traced function.

    @def_function.function
    def f(a):
      # [FUNCTION DOC]
      # HELPER: gathers the first element of the input.
      return array_ops.gather(a, 0)

    x = constant_op.constant([1.])

    with forwardprop.ForwardAccumulator(x, constant_op.constant([3.])) as acc:
      y = f(x)
    self.assertAllClose(3., acc.jvp(y))

  # NOTE: assert_no_new_pyobjects_executing_eagerly fails flakily on this
  # test... could be something wrong with the test decorator, or some sort of
  # nondeterministic caching.
  def testMirroredVariableWatched(self):
    # [FUNCTION DOC]
    # Creates a variable under a mirrored distribution strategy and runs a per-replica function
    # that watches it and asserts jvps of the variable and derived values.

    def _replicated(input_tangent):
      # [FUNCTION DOC]
      # HELPER: per-replica function; watches the mirrored variable with the given tangent and
      # asserts the jvps of the variable, its double and its offset.
      with forwardprop.ForwardAccumulator(v, input_tangent) as acc:
        self.assertAllClose([.1, -.2, .3], acc.jvp(v))
        x = v * 2.
        self.assertAllClose([.2, -.4, .6], acc.jvp(x))
        x2 = v + .1
        self.assertAllClose([.1, -.2, .3], acc.jvp(x2))

    strategy = mirrored_strategy.MirroredStrategy()
    with strategy.scope():
      v = variables.Variable([1., 2., 3.])
      strategy.run(_replicated, args=(constant_op.constant([.1, -.2, .3]),))

  # TODO(b/141025187): Add a no_new_pyobjects decorator.
  def testArgumentUnused(self):
    # [FUNCTION DOC]
    # Calls a traced function that ignores its argument under an accumulator and asserts the
    # output value and that its jvp is absent.
    v = constant_op.constant(1.)
    with forwardprop.ForwardAccumulator(v, 11.) as acc:

      @def_function.function
      def _f(x):
        # [FUNCTION DOC]
        # HELPER: deletes its argument and returns the constant one.
        del x
        return constant_op.constant(1.)

      result = _f(v)
      self.assertAllClose(1.0, result)
      self.assertIsNone(acc.jvp(result))


@def_function.function
def _has_loop(iters, y):
  # [FUNCTION DOC]
  # HELPER: traced function with a loop; sums y times each loop index (cast to float) for the
  # given number of iterations. Returns the sum. Used by the control flow tests.
  ret = 0.
  for i in math_ops.range(iters):
    ret += y * math_ops.cast(i, dtypes.float32)
  return ret


@def_function.function
def _has_cond(k, y):
  # [FUNCTION DOC]
  # HELPER: traced function with a conditional; returns three times y if k is greater than one,
  # otherwise zero. Used by the control flow tests.
  if k > 1:
    ret = 3. * y
  else:
    ret = 0.
  return ret


@def_function.function
def _fprop_while(iters, y):
  # [FUNCTION DOC]
  # HELPER: traced function creating an accumulator inside it, running a loop summing y times
  # the loop index, and returning the jvp of the sum (tangent one).
  with forwardprop.ForwardAccumulator(y, 1.) as acc:
    ret = 0.
    for i in math_ops.range(iters):
      ret += y * math_ops.cast(i, dtypes.float32)
  return acc.jvp(ret)


@def_function.function
def _fprop_cond(k, y):
  # [FUNCTION DOC]
  # HELPER: traced function creating an accumulator inside it, evaluating a conditional (three
  # times y if k is above one, else zero) and returning the jvp (tangent one).
  with forwardprop.ForwardAccumulator(y, 1.) as acc:
    if k > 1:
      ret = 3. * y
    else:
      ret = 0.
  return acc.jvp(ret)


class ControlFlowTests(test.TestCase):
  # [CLASS DOC]
  # Tests forward accumulators with traced functions containing loops and conditionals, using the
  # module-level traced helper functions.

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testOfFunctionWhile(self):
    # [FUNCTION DOC]
    # Accumulator outside a traced loop function; asserts the jvp equals ten for five iterations.
    y = constant_op.constant(1.)
    with forwardprop.ForwardAccumulator(y, 1.) as acc:
      self.assertAllClose(10., acc.jvp(_has_loop(constant_op.constant(5), y)))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testOfFunctionCond(self):
    # [FUNCTION DOC]
    # Accumulator outside a traced conditional function; asserts jvp three for a true branch
    # and zero for the false branch.
    y = constant_op.constant(1.)
    with forwardprop.ForwardAccumulator(y, 1.) as acc:
      self.assertAllClose(3., acc.jvp(_has_cond(constant_op.constant(5), y)))
      self.assertAllClose(0., acc.jvp(_has_cond(constant_op.constant(0), y)))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testInFunctionWhile(self):
    # [FUNCTION DOC]
    # Calls the traced function that builds an accumulator around a loop; asserts result ten.
    self.assertAllClose(
        10., _fprop_while(constant_op.constant(5), constant_op.constant(1.)))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testInFunctionCond(self):
    # [FUNCTION DOC]
    # Calls the traced function that builds an accumulator around a conditional; asserts three
    # for the true branch and zero for the false branch.
    self.assertAllClose(
        3., _fprop_cond(constant_op.constant(5), constant_op.constant(1.)))
    self.assertAllClose(
        0., _fprop_cond(constant_op.constant(0), constant_op.constant(1.)))


class HessianTests(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests computing full Hessians via forward-over-backward products.

  def testHessian1D(self):
    # [FUNCTION DOC]
    # Builds a quadratic form with a random matrix and vector. Computes the Hessian eagerly
    # with looping, in a traced function with looping, and in a traced function with vectorized
    # mapping; asserts each equals the matrix plus its transpose.
    # Note: stolen from ops/gradients_test.py
    m = 4
    rng = np.random.RandomState([1, 2, 3])
    mat_value = rng.randn(m, m).astype("float32")
    x_value = rng.randn(m).astype("float32")
    hess_value = mat_value + mat_value.T
    mat = variables.Variable(mat_value)

    def _f(x):
      # [FUNCTION DOC]
      # HELPER: the quadratic form, sum of the outer product of x with itself weighted by the matrix.
      return math_ops.reduce_sum(x[:, None] * mat * x[None, :])

    hessian_eager, = _forward_over_back_hessian(
        _f, [constant_op.constant(x_value)],
        use_pfor=False,
        dtype=[dtypes.float32])
    self.assertAllClose(hess_value, hessian_eager)
    hessian_function, = def_function.function(_forward_over_back_hessian)(
        _f, [constant_op.constant(x_value)],
        use_pfor=False,
        dtype=[dtypes.float32])
    self.assertAllClose(hess_value, hessian_function)
    hessian_pfor, = def_function.function(_forward_over_back_hessian)(
        _f, [constant_op.constant(x_value)],
        use_pfor=True,
        dtype=[dtypes.float32])
    self.assertAllClose(hess_value, hessian_pfor)


class BatchTests(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests batched forward accumulators (many tangents at once).

  @parameterized.parameters([(math_ops.sin, (2, 3), 5),
                             (math_ops.sin, (2, 3, 4), 10)])
  def testJVPBatchCorrectness(self, f, primal_shape, batch_size):
    # [FUNCTION DOC]
    # Parameterized by function, primal shape and batch size. Compares the vectorized batch jvp
    # with the Jacobian-matrix-multiply reference for random primals and tangent batches.
    primals = [random_ops.random_uniform(primal_shape)]
    tangent_batch = [random_ops.random_uniform([batch_size, *primal_shape])]
    self.assertAllClose(
        _jvp_batch(f, primals, tangent_batch)[1],
        _jvp_batch_matmul(f, primals, *tangent_batch))

  def testBatchCorrectness(self):
    # [FUNCTION DOC]
    # Uses the private batch accumulator with a batch of three tangents for two scalars, multiplies
    # them, and asserts the batch of jvps (by the product rule).
    x = constant_op.constant(2.0)
    y = constant_op.constant(5.0)
    tangents = (
        constant_op.constant([1., 0., 1.]),
        constant_op.constant([0., 1., 1.]),
    )
    with forwardprop.ForwardAccumulator._batch_accumulator((x, y),
                                                           tangents) as acc:
      z = x * y
    self.assertAllClose(acc.jvp(z), constant_op.constant([5.0, 2.0, 7.0]))

  @parameterized.named_parameters([("ForwardPropFirst", True),
                                   ("TapeFirst", False)])
  def testBatchBackwardOverForward(self, forward_prop_first):
    # [FUNCTION DOC]
    # Parameterized by creation order. With a persistent tape outside a batch accumulator,
    # asserts the tape records the batched jvps and the per-element gradients equal the expected
    # negative tangent times cosine.
    x = constant_op.constant(1.)
    tangents = random_ops.random_normal(shape=[10], seed=1)
    expected = [-t * math_ops.cos(1.) for t in tangents]
    if forward_prop_first:
      batch_acc = forwardprop.ForwardAccumulator._batch_accumulator(x, tangents)
      gradient_tape = backprop.GradientTape(persistent=True)
    else:
      gradient_tape = backprop.GradientTape(persistent=True)
      batch_acc = forwardprop.ForwardAccumulator._batch_accumulator(x, tangents)
    with gradient_tape as tape:
      with batch_acc as acc:
        tape.watch(x)
        y = math_ops.cos(x)
        self.assertTrue(record.should_record_backprop((acc.jvp(y),)))
        jvps = acc.jvp(y)
      d2y_dx2 = [tape.gradient(dy_dx, x) for dy_dx in jvps]
    self.assertAllClose(expected, d2y_dx2)


if __name__ == "__main__":
  # TODO(allenl): Also test with 1.x-style graph mode.
  ops.enable_eager_execution()
  test.main()
