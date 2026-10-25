# [MODULE DOC]
# Tests for interoperability between TF NumPy (the experimental numpy API), core TensorFlow ops,
# and the real NumPy library. Covers gradient tapes, cond and while loops, tf.function, distribution
# strategies, py_function, datasets, Keras layers and models, vectorized_map, jacobians, forward
# accumulation and map_fn, always checking that TF NumPy arrays flow through and come back as the
# array type. Defines three test classes (the second is subclassed by the others). Import-time
# side effect: binds the module-level names np (TF NumPy) and onp (real NumPy). Run as a script it sets
# the legacy dtype conversion mode, enables NumPy methods on tensors and eager execution, and starts
# the test runner.
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
"""Tests for interop between TF ops, numpy_ops, and numpy methods."""

import numpy as onp
import tensorflow.compat.v2 as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops.numpy_ops import np_math_ops

# Used instead of "import tensorflow(dot)experimental.numpy as np" due to
# copybara issues.
np = tf.experimental.numpy


# Tests for code snippet put in README.md
class ReadmeTest(tf.test.TestCase):
  # [CLASS DOC]
  # Tests mirroring the code snippets from the TF NumPy README: broadcasting add, type promotion,
  # mixing with TF tensors, mixing with real NumPy, device placement and use inside tf.function.
  # No shared fixtures.

  def testBroadcastAdd(self):
    # [FUNCTION DOC]
    # Adds a [2,1] array of ones to a [1,2] array of ones with TF NumPy and compares the broadcast result
    # to the same computation in real NumPy using close-equality.
    x_np = np.ones([2, 1]) + np.ones([1, 2])
    x_onp = onp.ones([2, 1]) + onp.ones([1, 2])
    self.assertAllClose(x_onp, x_np)

  def testTypePromotion(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the reference value x_onp is computed with the TF NumPy module (np), not real NumPy
    # KNOWN ISSUE: (onp), so the test compares TF NumPy against itself and does not validate NumPy parity.
    # Adds an int16 and a uint8 array of ones with different shapes, then asserts the two results have the
    # same dtype and values.
    x_np = np.ones([1, 2], dtype=np.int16) + np.ones([2, 1], dtype=np.uint8)
    x_onp = np.ones([1, 2], dtype=np.int16) + np.ones([2, 1], dtype=np.uint8)
    self.assertEqual(x_onp.dtype, x_np.dtype)
    self.assertAllClose(x_onp, x_np)

  def testTFInterop(self):
    # [FUNCTION DOC]
    # Sums the result of adding a TF NumPy ones array and a TF ones tensor, and compares with the same
    # sum computed purely in real NumPy.
    x_np = np.sum(np.ones([1, 2]) + tf.ones([2, 1]))
    x_onp = onp.sum(onp.ones([1, 2]) + onp.ones([2, 1]))
    self.assertAllClose(x_onp, x_np)

  def testOnpInterop(self):
    # [FUNCTION DOC]
    # Adds a TF NumPy ones array to a real NumPy ones array, sums with real NumPy, and compares with an
    # all-real-NumPy computation.
    x_np = onp.sum(np.ones([1, 2]) + onp.ones([2, 1]))
    x_onp = onp.sum(onp.ones([1, 2]) + onp.ones([2, 1]))
    self.assertAllClose(x_onp, x_np)

  def testDevice(self):
    # [FUNCTION DOC]
    # Checks device placement. If a GPU is available, creates an array under the first GPU scope and
    # asserts its tensor device name contains GPU. Always creates an array under the first CPU scope and
    # asserts the device name contains CPU.
    if tf.test.is_gpu_available():
      with tf.device('GPU:0'):
        x = np.ones([1, 2])
      self.assertIn('GPU', tf.convert_to_tensor(x).device)
    with tf.device('CPU:0'):
      x = np.ones([1, 2])
    self.assertIn('CPU', tf.convert_to_tensor(x).device)

  def testFunction(self):
    # [FUNCTION DOC]
    # Checks TF NumPy inside tf.function. Defines a traced function summing the sum of two inputs, calls it
    # with a TF NumPy array and a TF tensor, and compares with the real NumPy result.

    @tf.function
    def f(x, y):
      # [FUNCTION DOC]
      # HELPER: traced function used by testFunction.
      # Inputs x and y are array-like values; returns the sum of the elements of their element-wise sum.
      return np.sum(x + y)

    x_np = f(np.ones([1, 2]), tf.ones([2, 1]))
    x_onp = onp.sum(onp.ones([1, 2]) + onp.ones([2, 1]))
    self.assertAllClose(x_onp, x_np)


class InteropTest(tf.test.TestCase):
  # [CLASS DOC]
  # Base test case for interop of TF NumPy arrays with TF machinery (gradients, control flow, strategies,
  # datasets, Keras, pfor, jacobians). Its setUp makes sure three logical CPU devices exist. It is also
  # the base class of FunctionTest and VariableTest, whose inherited tests run again in those classes.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Calls the base setUp, then looks up the first physical CPU and, if no logical device
    # configuration exists for it, configures three logical CPU devices (needed by the distribution
    # strategy test). Side effect: changes global device configuration; may fail if devices were already
    # initialized.
    super(InteropTest, self).setUp()
    physical_devices = tf.config.list_physical_devices('CPU')
    configs = tf.config.get_logical_device_configuration(physical_devices[0])
    if configs is None:
      logical_devices = [
          tf.config.LogicalDeviceConfiguration() for _ in range(3)
      ]
      tf.config.set_logical_device_configuration(physical_devices[0],
                                                 logical_devices)

  def testGradientTapeInterop(self):
    # [FUNCTION DOC]
    # Watches two TF NumPy scalars on a gradient tape, computes scaled values, and gets gradients.
    # Asserts both gradients are TF NumPy arrays with values 2.0 and 3.0.
    with tf.GradientTape() as t:
      x = np.asarray(3.0)
      y = np.asarray(2.0)

      t.watch([x, y])

      xx = 2 * x
      yy = 3 * y

    dx, dy = t.gradient([xx, yy], [x, y])

    self.assertIsInstance(dx, np.ndarray)
    self.assertIsInstance(dy, np.ndarray)
    self.assertAllClose(dx, 2.0)
    self.assertAllClose(dy, 3.0)

  def testGradientTapeNoneGradients(self):
    # [FUNCTION DOC]
    # Computes a value from a watched variable on a tape and requests the gradient with respect to an
    # unrelated array; asserts the gradient is nothing/null.
    y = np.asarray(2.0)

    with tf.GradientTape() as t:
      x = np.asarray(3.0)
      t.watch([x])
      z = 2 * x

    dz = t.gradient(z, y)

    self.assertIsNone(dz)

  def testCondInterop(self):
    # [FUNCTION DOC]
    # Runs conditional control flow on a TF NumPy scalar, both eagerly and wrapped in tf.function, with two
    # conditionals (positive check and negative check), and asserts results equal the input plus 1 and
    # plus 2 in both modes.
    x = np.asarray(3.0)

    def fn(x):
      # [FUNCTION DOC]
      # HELPER: function under test for testCondInterop.
      # Input x is a scalar array. Returns a pair of results from two conditionals, adding 1 or 2 depending
      # on the sign of x.
      x_plus_1 = tf.cond(x > 0, lambda: x + 1, lambda: x + 2)
      x_plus_2 = tf.cond(x < 0, lambda: x + 1, lambda: x + 2)

      return x_plus_1, x_plus_2

    raw_x_plus_1, raw_x_plus_2 = fn(x)
    fn_x_plus_1, fn_x_plus_2 = tf.function(fn)(x)

    self.assertAllClose(raw_x_plus_1, x + 1)
    self.assertAllClose(raw_x_plus_2, x + 2)

    self.assertAllClose(fn_x_plus_1, x + 1)
    self.assertAllClose(fn_x_plus_2, x + 2)

  def testWhileInterop(self):
    # [FUNCTION DOC]
    # Runs a while loop on a TF NumPy scalar counter up to 10000 with 20 parallel iterations, both eagerly
    # and inside tf.function, asserting the final count is 10000.

    def fn():
      # [FUNCTION DOC]
      # HELPER: function under test for testWhileInterop.
      # Takes no inputs; creates a zero counter array and returns the while-loop result list that increments
      # it until it reaches 10000.
      x = np.asarray(0)
      c = lambda x: x < 10000
      b = lambda x: [x + 1]
      return tf.while_loop(c, b, [x], parallel_iterations=20)

    self.assertEqual(10000, fn()[0])
    self.assertEqual(10000, tf.function(fn)()[0])

  def testTensorTFNPArrayInterop(self):
    # [FUNCTION DOC]
    # Adds a TF NumPy scalar and a TF tensor in both operand orders and asserts both results are plain TF
    # tensors with value 10.
    arr = np.asarray(0.)
    t = tf.constant(10.)

    arr_plus_t = arr + t
    t_plus_arr = t + arr

    self.assertIsInstance(arr_plus_t, tf.Tensor)
    self.assertIsInstance(t_plus_arr, tf.Tensor)
    self.assertEqual(10., arr_plus_t.numpy())
    self.assertEqual(10., t_plus_arr.numpy())

  def testTensorTFNPOp(self):
    # [FUNCTION DOC]
    # Applies the TF NumPy square function to a TF tensor and asserts the result is an array type with
    # value 100.
    t = tf.constant(10.)

    sq = np.square(t)
    self.assertIsInstance(sq, np.ndarray)
    self.assertEqual(100., sq)

  def testTFNPArrayTFOpInterop(self):
    # [FUNCTION DOC]
    # Applies the TF square op to a TF NumPy scalar and asserts the result is a tensor with value 100.
    arr = np.asarray(10.)

    # TODO(nareshmodi): Test more ops.
    sq = tf.square(arr)
    self.assertIsInstance(sq, tf.Tensor)
    self.assertEqual(100., sq.numpy())

  def testTFNPArrayNPOpInterop(self):
    # [FUNCTION DOC]
    # Applies real NumPy square to a TF NumPy one-element array and asserts the result is a real NumPy
    # array with value 100.
    arr = np.asarray([10.])

    # TODO(nareshmodi): Test more ops.
    sq = onp.square(arr)
    self.assertIsInstance(sq, onp.ndarray)
    self.assertEqual(100., sq[0])

# TODO(b/171313773): why doesn't tensor have __array_module__
  def testArrayModule(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test skips itself immediately, so the remaining lines (array-module protocol checks
    # KNOWN ISSUE: and the nested Dummy class) never run; they are dead code until tensors support it.
    # Intended to check that the array-module protocol returns the TF NumPy module, and not-implemented
    # for unknown types. Currently only calls a skip with the reason that tensors lack that method.
    self.skipTest("Tensor doesn't have __array_module__")
    arr = np.asarray([10])

    module = arr.__array_module__((tf.Tensor,))
    self.assertIs(module, tf.experimental.numpy)

    class Dummy:
      # [CLASS DOC]
      # HELPER: placeholder class used by the skipped testArrayModule as an unknown array type.
      # Empty class with no state.
      pass
    module = arr.__array_module__((tf.Tensor, Dummy))
    self.assertIs(module, NotImplemented)

# TODO(nareshmodi): Fails since the autopacking code doesn't use
# nest.flatten.
#   def testAutopacking(self):
#     arr1 = np.asarray(1.)
#     arr2 = np.asarray(2.)
#     arr3 = np.asarray(3.)
#     t = ops.convert_to_tensor_v2([arr1, arr2, arr3])
#     self.assertEqual(t.numpy(), [1., 2., 3.])

  def testDistStratInterop(self):
    # [FUNCTION DOC]
    # Uses a mirrored strategy over three CPU devices to run a function creating a per-replica array from
    # the replica id and multiplying by a constant array. Asserts the three local results are arrays with
    # values 0, 5 and 10, and the sum-reduced value is 15. Relies on the three logical CPUs from setUp.
    strategy = tf.distribute.MirroredStrategy(
        devices=['CPU:0', 'CPU:1', 'CPU:2'])

    multiplier = np.asarray(5.)

    @tf.function
    def run():
      # [FUNCTION DOC]
      # HELPER: replica function run by testDistStratInterop.
      # Takes no inputs; reads the replica context, wraps the replica id as an array and returns it
      # multiplied by the enclosing multiplier array.
      ctx = tf.distribute.get_replica_context()
      val = np.asarray(ctx.replica_id_in_sync_group)
      return val * multiplier

    distributed_values = strategy.run(run)
    reduced = strategy.reduce(
        tf.distribute.ReduceOp.SUM, distributed_values, axis=None)

    values = strategy.experimental_local_results(distributed_values)

    # Note that this should match the number of virtual CPUs.
    self.assertLen(values, 3)
    self.assertIsInstance(values[0], np.ndarray)
    self.assertIsInstance(values[1], np.ndarray)
    self.assertIsInstance(values[2], np.ndarray)
    self.assertAllClose(values[0], 0)
    self.assertAllClose(values[1], 5)
    self.assertAllClose(values[2], 10)

    # "strategy.reduce" doesn't rewrap in ndarray.
    # self.assertIsInstance(reduced, np.ndarray)
    self.assertAllClose(reduced, 15)

  @test_util.disable_tfrt('b/180469928')
  def testPyFuncInterop(self):
    # [FUNCTION DOC]
    # Round-trips TF NumPy arrays through py_function inside tf.function and asserts the result is an
    # array with value 3. Disabled under the TFRT runtime by a decorator.
    def py_func_fn(a, b):
      # [FUNCTION DOC]
      # HELPER: Python function wrapped by py_function in testPyFuncInterop.
      # Inputs a and b are values; returns their sum.
      return a + b

    @tf.function
    def fn(a, b):
      # [FUNCTION DOC]
      # HELPER: traced function under test for testPyFuncInterop.
      # Inputs a and b are arrays; calls the Python function through py_function using a's dtype and
      # returns the output as a TF NumPy array.
      result = tf.py_function(py_func_fn, [a, b], a.dtype)
      return np.asarray(result)

    a = np.asarray(1.)
    b = np.asarray(2.)

    result = fn(a, b)
    self.assertIsInstance(result, np.ndarray)
    self.assertAllClose(result, 3.)

  def testDatasetInterop(self):
    # [FUNCTION DOC]
    # Tests datasets of TF NumPy arrays: from a whole array, from slices, after a map using NumPy add, and
    # batched in twos. In each case asserts yielded elements are arrays with the expected values.
    values = [1, 2, 3, 4, 5, 6]
    values_as_array = np.asarray(values)

    # Tensor dataset
    dataset = tf.data.Dataset.from_tensors(values_as_array)

    for value, value_from_dataset in zip([values_as_array], dataset):
      self.assertIsInstance(value_from_dataset, np.ndarray)
      self.assertAllEqual(value_from_dataset, value)

    # Tensor slice dataset
    dataset = tf.data.Dataset.from_tensor_slices(values_as_array)

    for value, value_from_dataset in zip(values, dataset):
      self.assertIsInstance(value_from_dataset, np.ndarray)
      self.assertAllEqual(value_from_dataset, value)

    # # TODO(nareshmodi): as_numpy_iterator() doesn't work.
    # items = list(dataset.as_numpy_iterator())

    # Map over a dataset.
    dataset = dataset.map(lambda x: np.add(x, 1))

    for value, value_from_dataset in zip(values, dataset):
      self.assertIsInstance(value_from_dataset, np.ndarray)
      self.assertAllEqual(value_from_dataset, value + 1)

    # Batch a dataset.
    dataset = tf.data.Dataset.from_tensor_slices(values_as_array).batch(2)

    for value, value_from_dataset in zip([[1, 2], [3, 4], [5, 6]], dataset):
      self.assertIsInstance(value_from_dataset, np.ndarray)
      self.assertAllEqual(value_from_dataset, value)

  def testKerasInterop(self):
    # [FUNCTION DOC]
    # Builds a Keras model whose layer applies the TF NumPy square function, feeds a real NumPy array and
    # a TF NumPy array, and asserts outputs are arrays equal to the squared input.
    # Return an ndarray from the model.
    inputs = tf.keras.layers.Input(shape=(10,))
    output_layer = tf.keras.layers.Lambda(np.square)(inputs)
    model = tf.keras.Model([inputs], output_layer)

    values = onp.arange(10, dtype=onp.float32).reshape((1, 10))
    values_as_array = np.asarray(values)

    result = model(values)
    self.assertIsInstance(result, np.ndarray)
    self.assertAllClose(result, onp.square(values))

    result = model(values_as_array)
    self.assertIsInstance(result, np.ndarray)
    self.assertAllClose(result, onp.square(values))

  def testKerasInteropSequential(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the final dense-layer call result is never asserted, so that part only checks it does not raise.
    # Defines a custom Keras layer projecting inputs with a random matrix variable via TF NumPy matmul,
    # builds a sequential model of a dense layer and the projection, calls it on random float32 input and
    # asserts the output is an array. Finally calls a plain dense layer on array input.
    class ProjectionLayer(tf.keras.layers.Layer):
      # [CLASS DOC]
      # HELPER: Keras layer defined inside testKerasInteropSequential.
      # Linear projection to a given number of units using TF NumPy math; holds a units count and a
      # trainable weight variable created lazily on build.
      """Linear projection layer using TF NumPy."""

      def __init__(self, units):
        # [FUNCTION DOC]
        # Constructor. Input units is the output dimension. Calls the base constructor and stores units.
        super(ProjectionLayer, self).__init__()
        self._units = units

      def build(self, input_shape):
        # [FUNCTION DOC]
        # Creates the weight on first call. Input input_shape is the input shape; makes a random normal
        # matrix of shape (input width, units) in float32 scaled by the square root of units, and stores it as
        # a trainable variable in the layer. Returns nothing.
        stddev = np.sqrt(self._units).astype(np.float32)
        initial_value = np.random.randn(input_shape[1], self._units).astype(
            np.float32) / stddev
        # Note that TF NumPy can interoperate with tf.Variable.
        self.w = tf.Variable(initial_value, trainable=True)

      def call(self, inputs):
        # [FUNCTION DOC]
        # Forward pass. Input inputs is an array; returns its matrix product with the weight variable.
        return np.matmul(inputs, self.w)

    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(100), ProjectionLayer(2)])
    output = model.call(np.random.randn(10, 100).astype(np.float32))

    self.assertIsInstance(output, np.ndarray)

    dense_layer = tf.keras.layers.Dense(100)
    output = dense_layer(np.random.randn(10, 100).astype(np.float32))

  def testPForInterop(self):
    # [FUNCTION DOC]
    # Tests vectorized_map with TF NumPy arrays: maps an outer product over a batch of 100 and a transpose
    # lambda; asserts results are arrays with the expected shapes.
    def outer_product(a):
      # [FUNCTION DOC]
      # HELPER: per-example function mapped by testPForInterop.
      # Input a is one array; returns its tensor dot with itself over zero axes (outer product).
      return np.tensordot(a, a, 0)

    batch_size = 100
    a = np.ones((batch_size, 32, 32))
    c = tf.vectorized_map(outer_product, a)

    self.assertIsInstance(c, np.ndarray)
    self.assertEqual(c.shape, (batch_size, 32, 32, 32, 32))

    c = tf.vectorized_map(lambda x: x.T, a)

    self.assertIsInstance(c, np.ndarray)
    self.assertEqual(c.shape, (batch_size, 32, 32))

  def testJacobian(self):
    # [FUNCTION DOC]
    # Computes the jacobian of x*x*y for two watched arrays on a tape. Asserts both jacobians are arrays and
    # equal the diagonal matrices of 2*x*y and x*x.
    with tf.GradientTape() as g:
      x = np.asarray([1., 2.])
      y = np.asarray([3., 4.])
      g.watch(x)
      g.watch(y)
      z = x * x * y

    jacobian = g.jacobian(z, [x, y])
    answer = [tf.linalg.diag(2 * x * y), tf.linalg.diag(x * x)]

    self.assertIsInstance(jacobian[0], np.ndarray)
    self.assertIsInstance(jacobian[1], np.ndarray)
    self.assertAllClose(jacobian, answer)

  def testBatchJacobian(self):
    # [FUNCTION DOC]
    # Computes the batch jacobian of x*x*y with respect to x for a two-example batch; asserts the result is
    # an array equal to the stacked diagonals of 2*x*y per example.
    with tf.GradientTape() as g:
      x = np.asarray([[1., 2.], [3., 4.]])
      y = np.asarray([[3., 4.], [5., 6.]])
      g.watch(x)
      g.watch(y)
      z = x * x * y

    batch_jacobian = g.batch_jacobian(z, x)
    answer = tf.stack(
        [tf.linalg.diag(2 * x[0] * y[0]),
         tf.linalg.diag(2 * x[1] * y[1])])

    self.assertIsInstance(batch_jacobian, np.ndarray)
    self.assertAllClose(batch_jacobian, answer)

  def testForwardprop(self):
    # [FUNCTION DOC]
    # Uses a forward accumulator with a tangent for an array; asserts the jvp of doubling is an array with
    # values 6 and 8, and that the jvp of an unrelated array is nothing/null.
    x = np.asarray([1., 2.])
    xt = np.asarray([3., 4.])
    with tf.autodiff.ForwardAccumulator(x, xt) as acc:
      y = x * 2.
    yt = acc.jvp(y)
    self.assertIsInstance(yt, np.ndarray)
    self.assertAllClose([6., 8.], yt)
    z = np.asarray([1.])
    self.assertIsNone(acc.jvp(z))

  def testMapFn(self):
    # [FUNCTION DOC]
    # Applies map_fn to a pair of the same array with a function adding 1 to elements; asserts both
    # outputs are arrays with values 2 and 3.
    x = np.asarray([1., 2.])
    mapped_x = tf.map_fn(lambda x: (x[0]+1, x[1]+1), (x, x))

    self.assertIsInstance(mapped_x[0], np.ndarray)
    self.assertIsInstance(mapped_x[1], np.ndarray)
    self.assertAllClose(mapped_x[0], [2., 3.])
    self.assertAllClose(mapped_x[1], [2., 3.])


class FunctionTest(InteropTest):
  # [CLASS DOC]
  # Re-runs every InteropTest test (by inheritance) and adds tests that TF NumPy arrays work in
  # tf.function: plain function call, and unsupported operations that must raise a type error when traced.

  def testFunctionInterop(self):
    # [FUNCTION DOC]
    # Wraps an addition lambda in tf.function and asserts the raw and traced results are both arrays and
    # are close.
    x = np.asarray(3.0)
    y = np.asarray(2.0)

    add = lambda x, y: x + y
    add_fn = tf.function(add)

    raw_result = add(x, y)
    fn_result = add_fn(x, y)

    self.assertIsInstance(raw_result, np.ndarray)
    self.assertIsInstance(fn_result, np.ndarray)
    self.assertAllClose(raw_result, fn_result)

  def testLen(self):
    # [FUNCTION DOC]
    # Traces without autograph a function taking the length of data-dependent where results; asserts calling
    # it with a boolean array raises a type error. The comment says the test could be removed.

    # len can be fixed by autograph.
    # TODO(wangpeng): this test can just be removed
    @tf.function(autograph=False)
    def f(x):
      # [FUNCTION DOC]
      # HELPER: traced function used by testLen.
      # Input x is an array; returns the length of the first output of the where operation.
      # Note that shape of input to len is data dependent.
      return len(np.where(x)[0])

    t = np.asarray([True, False, True])
    with self.assertRaises(TypeError):
      f(t)

  def testIter(self):
    # [FUNCTION DOC]
    # Traces a function that unpacks its array argument into two values; asserts calling it raises a type
    # error because iteration over a symbolic array is unsupported.

    @tf.function
    def f(x):
      # [FUNCTION DOC]
      # HELPER: traced function used by testIter.
      # Input x is an array; unpacks it into two values and returns them as a pair.
      y, z = x
      return y, z

    with self.assertRaises(TypeError):
      f(np.asarray([3, 4]))

  def testIndex(self):
    # [FUNCTION DOC]
    # Traces a function using an array to index a Python list; asserts calling it raises a type error.

    @tf.function
    def f(x):
      # [FUNCTION DOC]
      # HELPER: traced function used by testIndex.
      # Input x is an array; returns the list element at that index.
      return [0, 1][x]

    with self.assertRaises(TypeError):
      f(np.asarray([1]))


class VariableTest(InteropTest):
  # [CLASS DOC]
  # Re-runs every InteropTest test (by inheritance) and adds a test of TF NumPy with variables.

  def test(self):
    # [FUNCTION DOC]
    # Applies TF NumPy square to a TF variable; asserts an array with value 4. Then under a control
    # dependency on adding it to the variable, reads the variable and asserts it is 6.
    tf_var = tf.Variable(2.0)
    value = np.square(tf_var)
    self.assertIsInstance(value, np.ndarray)
    self.assertAllClose(4.0, value)
    with tf.control_dependencies([tf_var.assign_add(value)]):
      tf_var_value = tf_var.read_value()
    self.assertAllClose(6.0, tf_var_value)


if __name__ == '__main__':
  ops.set_dtype_conversion_mode('legacy')
  np_math_ops.enable_numpy_methods_on_tensor()
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
