# [MODULE DOC]
# Tests for the DistributedVariable wrapper in values_v2. Contains an abstract interface-conformance
# test base class (removed from the module namespace after the concrete subclass is defined so it
# is not collected), a concrete subclass that wraps two CPU-device variables, a parameterized test
# class exercising reads, assignments and strategy run across TPU / mirrored strategies with and
# without packed handles and tf.function, and a helper that builds indexed slices.
# Running the file as a program starts the distribute test main.

# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the distributed values library."""

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import values_v2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib


class _VariableInterfaceTestBase(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # HELPER: abstract base of tests checking that a distributed variable satisfies the Variable and
  # resource-variable interfaces (methods and properties exist and behave the same on all
  # devices). Subclasses must supply the variable factory. Behavior that differs per replica or
  # device is deliberately not covered here. Deleted from the module namespace after use so the
  # base class is not run itself.
  # This test verifies that DistributedVariable/AutoSyncVariable conforms to
  # Variable and ResourceVariable interface, i.e. the methods and properties are
  # all defined. It verifies methods and properties that have the same code path
  # under different replicas/devices as well. It is not intended to verify
  # methods and properties that behave differently under different
  # replicas/devices; those should be covered separate tests.

  def create_variable(self, initial_value=1., **kwargs):
    # [FUNCTION DOC]
    # HELPER: abstract factory that subclasses override. Parameters: initial_value (default one)
    # and extra keyword arguments passed to the variable. Always raises a not-implemented error here.
    raise NotImplementedError

  @property
  def devices(self):
    # [FUNCTION DOC]
    # HELPER: property returning the list of device names the variables are created on: the first
    # two CPU devices.
    return ["CPU:0", "CPU:1"]

  # ==== Begin Variable interface ===
  # Please follow the same order as methods and properties defined in
  # tf.Variable.

  def testStringify(self):
    # [FUNCTION DOC]
    # Creates a variable and checks that both its string form and debug representation are strings.
    v = self.create_variable()
    self.assertIsInstance(v.__str__(), str)
    self.assertIsInstance(v.__repr__(), str)

  def testDenseRead(self):
    # [FUNCTION DOC]
    # Creates a scalar variable with value one and asserts that both the value method and the read
    # value method return one.
    v = self.create_variable(1.)
    self.assertEqual(v.value(), 1.)
    self.assertEqual(v.read_value(), 1.)

  def testShape(self):
    # [FUNCTION DOC]
    # Creates a one-element vector variable; asserts the shape and the get-shape result are a
    # single dimension of size one, that setting the same shape works, and that setting an
    # incompatible two-dimensional shape raises a value error mentioning incompatibility.
    v = self.create_variable([1.])
    self.assertEqual(v.shape, (1,))
    self.assertEqual(v.get_shape(), (1,))
    v.set_shape((1,))
    with self.assertRaisesRegex(ValueError, "not compatible"):
      v.set_shape((1, 1))

  @combinations.generate(combinations.combine(trainable=[True, False]))
  def testTrainable(self, trainable):
    # [FUNCTION DOC]
    # Parameterized over trainable true and false: creates a variable with that flag and asserts the
    # variable reports the same trainable value.
    v = self.create_variable(trainable=trainable)
    self.assertEqual(v.trainable, trainable)

  @combinations.generate(
      combinations.combine(synchronization=[
          variables_lib.VariableSynchronization.ON_READ,
          variables_lib.VariableSynchronization.ON_WRITE,
          variables_lib.VariableSynchronization.AUTO,
          variables_lib.VariableSynchronization.NONE,
      ]))
  def testSynchronization(self, synchronization):
    # [FUNCTION DOC]
    # Parameterized over the four synchronization modes (on read, on write, auto, none): creates a
    # variable with the mode and asserts the variable reports that same mode.
    v = self.create_variable(synchronization=synchronization)
    self.assertEqual(v.synchronization, synchronization)

  @combinations.generate(
      combinations.combine(aggregation=[
          variables_lib.VariableAggregation.MEAN,
          variables_lib.VariableAggregation.SUM,
          variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
          variables_lib.VariableAggregation.NONE,
      ]))
  def testAggregation(self, aggregation):
    # [FUNCTION DOC]
    # Parameterized over the four aggregation modes (mean, sum, only first replica, none): creates a
    # variable with the mode and asserts the variable reports that same aggregation.
    v = self.create_variable(aggregation=aggregation)
    self.assertEqual(v.aggregation, aggregation)

  @combinations.generate(combinations.combine(mode="graph"))
  def testEval(self):
    # [FUNCTION DOC]
    # Graph mode only. Creates a variable, runs the global variable initializer in a cached session
    # and asserts that evaluating the variable gives one.
    v = self.create_variable(1.)
    with self.cached_session():
      self.evaluate(variables_lib.global_variables_initializer())
      self.assertEqual(v.eval(), 1.)

  def testInitialValueEager(self):
    # [FUNCTION DOC]
    # In eager mode, asserts that reading the initial value property of a variable raises a runtime
    # error.
    v = self.create_variable(1.)
    with self.assertRaises(RuntimeError):
      v.initial_value  # pylint: disable=pointless-statement

  @combinations.generate(combinations.combine(mode="graph"))
  def testInitialValueGraph(self):
    # [FUNCTION DOC]
    # Graph mode only. Asserts that evaluating the initial value tensor of a variable created with
    # one gives one.
    v = self.create_variable(1.)
    self.assertEqual(self.evaluate(v.initial_value), 1.)

  def testConstraint(self):
    # [FUNCTION DOC]
    # Creates a variable with a constraint callback that adds one; asserts that calling the
    # variable's constraint on one gives two (the same callback is exposed).
    v = self.create_variable(constraint=lambda x: x + 1.)
    self.assertEqual(v.constraint(1.), 2.)

  def testDenseUpdate(self):
    # [FUNCTION DOC]
    # Exercises assign, assign-add and assign-sub on a scalar variable starting at one, with
    # locking and names and read value true (returns the new value: two, four, four after
    # subtracting) and read value false (returns nothing in eager mode), checking the variable
    # value after each step (three, five, three). Then, inside a compiled function, asserts that
    # each update with read value false returns an operation object.
    v = self.create_variable(1.)
    self.assertEqual(
        v.assign(2., use_locking=True, name="assign", read_value=True), 2.)
    self.assertIsNone(v.assign(3., read_value=False))
    self.assertEqual(v, 3.)
    self.assertEqual(
        v.assign_add(1., use_locking=True, name="assign_add", read_value=True),
        4.)
    self.assertIsNone(v.assign_add(1., read_value=False))
    self.assertEqual(v, 5.)
    self.assertEqual(
        v.assign_sub(1., use_locking=True, name="assign_sub", read_value=True),
        4.)
    self.assertIsNone(v.assign_sub(1., read_value=False))
    self.assertEqual(v, 3.)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: traced body for the test; asserts that assign, assign-add and assign-sub with read
      # value false each return an operation object inside a function. Returns nothing.
      self.assertIsInstance(v.assign(1., read_value=False), ops.Operation)
      self.assertIsInstance(v.assign_add(1., read_value=False), ops.Operation)
      self.assertIsInstance(v.assign_sub(1., read_value=False), ops.Operation)

    f()

  def testSparseUpdate(self):
    # [FUNCTION DOC]
    # Starts with a three-element zero vector and applies, in order, scatter add, divide, max, min,
    # multiply, subtract, update and batch update with small indexed slices (with locking and a
    # name). After each call asserts the returned tensor equals the expected vector computed by
    # hand (for example after the first add: one, zero, two).
    v = self.create_variable([0., 0., 0.])
    self.assertAllEqual(
        v.scatter_add(
            _make_index_slices(values=[1., 2.], indices=[0, 2]),
            use_locking=True,
            name="add"), [1., 0., 2.])
    self.assertAllEqual(
        v.scatter_div(
            _make_index_slices(values=[4., 2.], indices=[0, 2]),
            use_locking=True,
            name="div"), [0.25, 0., 1.])
    self.assertAllEqual(
        v.scatter_max(
            _make_index_slices(values=[1., 0.5], indices=[1, 2]),
            use_locking=True,
            name="max"), [0.25, 1., 1.])
    self.assertAllEqual(
        v.scatter_min(
            _make_index_slices(values=[1., 0.5], indices=[0, 1]),
            use_locking=True,
            name="min"), [0.25, 0.5, 1.])
    self.assertAllEqual(
        v.scatter_mul(
            _make_index_slices(values=[2., 0.5], indices=[0, 1]),
            use_locking=True,
            name="mul"), [0.5, 0.25, 1.])
    self.assertAllEqual(
        v.scatter_sub(
            _make_index_slices(values=[2., 0.5], indices=[0, 1]),
            use_locking=True,
            name="sub"), [-1.5, -0.25, 1.])
    self.assertAllEqual(
        v.scatter_update(
            _make_index_slices(values=[2., 0.5], indices=[0, 1]),
            use_locking=True,
            name="update"), [2., 0.5, 1.])
    self.assertAllEqual(
        v.batch_scatter_update(
            _make_index_slices(values=[1., 1.5], indices=[0, 1]),
            use_locking=True,
            name="update"), [1., 1.5, 1.])

  def testSparseNdUpdate(self):
    # [FUNCTION DOC]
    # Starts with a four-element zero vector; applies n-dimensional scatter subtract, add and update
    # with index lists and values, asserting after each the full expected vector.
    v = self.create_variable([0., 0., 0., 0.])
    self.assertAllEqual(
        v.scatter_nd_sub([[3], [1]], [1., 2.], name="sub"), [0., -2., 0., -1.])
    self.assertAllEqual(
        v.scatter_nd_add([[2], [0]], [1., 2.], name="add"), [2., -2., 1., -1.])
    self.assertAllEqual(
        v.scatter_nd_update([[1], [3]], [3., 3.], name="update"),
        [2., 3., 1., 3.])

  def testSparseRead(self):
    # [FUNCTION DOC]
    # On a two-by-two matrix variable, asserts sparse read at rows one then zero returns those rows
    # in that order, and gather-nd at the given coordinates returns the two matching elements.
    v = self.create_variable([[1., 2.], [3., 4.]])
    self.assertAllEqual(
        v.sparse_read([1, 0], name="read"), [[3., 4.], [1., 2.]])
    self.assertAllEqual(
        v.gather_nd([[1, 0], [0, 1]], name="gather_nd"), [3., 2.])

  def testTensorConversion(self):
    # [FUNCTION DOC]
    # Asserts that converting a one-element variable to a tensor yields its value.
    v = self.create_variable([1.])
    self.assertEqual(ops.convert_to_tensor(v), [1.])

  def testHash(self):
    # [FUNCTION DOC]
    # In eager mode, asserts a variable cannot be used directly as a mapping key (type error), but
    # its reference object can: the key lookup works for the same variable and a different
    # variable's reference is not present.
    v = self.create_variable()
    w = self.create_variable()
    d = {}
    with self.assertRaises(TypeError):
      d[v] = 1
    d[v.ref()] = 1
    self.assertEqual(d[v.ref()], 1)
    self.assertNotIn(w.ref(), d)

  @combinations.generate(combinations.combine(mode="graph"))
  def testHashGraph(self):
    # [FUNCTION DOC]
    # Graph mode only: a variable can be used directly as a mapping key; lookup of the same variable
    # works and a different variable is not found.
    v = self.create_variable()
    w = self.create_variable()
    d = {v: 1}
    self.assertEqual(d[v], 1)
    self.assertNotIn(w, d)

  def testEquality(self):
    # [FUNCTION DOC]
    # In eager mode, variables compare by value: two variables holding one are equal, and one
    # holding two differs from them.
    v = self.create_variable(1.)
    w = self.create_variable(2.)
    x = self.create_variable(1.)
    self.assertEqual(v, x)
    self.assertNotEqual(v, w)

  @combinations.generate(combinations.combine(mode="graph"))
  def testEqualityGraph(self):
    # [FUNCTION DOC]
    # Graph mode only: equality is object identity, so two distinct variables with equal values are
    # not equal while a variable equals itself.
    # In legacy graph mode, tensor equality is object equality
    v = self.create_variable(1.)
    w = self.create_variable(1.)
    self.assertNotEqual(v, w)
    self.assertEqual(v, v)

  def testIteration(self):
    # [FUNCTION DOC]
    # Asserts that iterating over a one-element vector variable yields a list equal to its value.
    v = self.create_variable([1.])
    self.assertEqual([1.], list(iter(v)))

  def testProperties(self):
    # [FUNCTION DOC]
    # In eager mode, asserts the name, internal shared name and device are strings, the initializer
    # is nothing/null, dtype is 32-bit float, and accessing the op or graph property raises an
    # attribute error.
    v = self.create_variable()
    self.assertIsInstance(v.name, str)
    # _shared_name is also part of the interface. E.g. it's used in optimizer to
    # determine slot variable key.
    self.assertIsInstance(v._shared_name, str)
    self.assertIsNone(v.initializer)
    self.assertIsInstance(v.device, str)
    self.assertEqual(v.dtype, dtypes.float32)
    with self.assertRaises(AttributeError):
      v.op  # pylint: disable=pointless-statement
    with self.assertRaises(AttributeError):
      v.graph  # pylint: disable=pointless-statement

  @combinations.generate(combinations.combine(mode="graph"))
  def testPropertiesGraph(self):
    # [FUNCTION DOC]
    # Graph mode only: asserts the initializer and op properties are operations and the graph
    # property is a graph.
    v = self.create_variable()
    self.assertIsInstance(v.initializer, ops.Operation)
    self.assertIsInstance(v.op, ops.Operation)
    self.assertIsInstance(v.graph, ops.Graph)

  def testProtoConversion(self):
    # [FUNCTION DOC]
    # Asserts that converting a variable to a proto, and creating one from a proto, each raise a
    # type error because both are unsupported.
    # to_proto and from_proto are not supported.
    v = self.create_variable([1, 2])
    with self.assertRaises(TypeError):
      v.to_proto()
    with self.assertRaises(TypeError):
      v.from_proto(variable_def=None)

  def testSaveSliceInfo(self):
    # [FUNCTION DOC]
    # Sets a save-slice-info object on the variable and asserts both the getter and the direct
    # attribute return that identical object.
    v = self.create_variable()
    slice_info = variables_lib.Variable.SaveSliceInfo()
    v._set_save_slice_info(slice_info)
    self.assertIs(v._get_save_slice_info(), slice_info)
    # Some code accesses _save_slice_info directly without using the getter.
    self.assertIs(v._save_slice_info, slice_info)

  def testOperatorOverride(self):
    # [FUNCTION DOC]
    # Uses a variable holding seven and asserts arithmetic operators (add, subtract, multiply,
    # true and floor division, modulo, power, negate, invert, absolute value), comparison
    # operators, and bitwise operators (and, or, xor) work with a plain number on either side and
    # with the variable on both sides where applicable, giving the expected results.
    v = self.create_variable(7)
    self.assertEqual(v + 1, 8)
    self.assertEqual(3 + v, 10)
    self.assertEqual(v + v, 14)
    self.assertEqual(v - 2, 5)
    self.assertEqual(13 - v, 6)
    self.assertEqual(v - v, 0)
    self.assertEqual(v * 2, 14)
    self.assertEqual(3 * v, 21)
    self.assertEqual(v * v, 49)
    self.assertEqual(v / 2, 3.5)
    self.assertEqual(14 / v, 2.)
    self.assertEqual(v // 2, 3)
    self.assertEqual(15 // v, 2)
    self.assertEqual(v % 2, 1)
    self.assertEqual(16 % v, 2)
    # pylint: disable=g-generic-assert
    self.assertTrue(v < 12)
    self.assertTrue(v <= 12)
    self.assertFalse(v > 12)
    self.assertFalse(v >= 12)
    self.assertFalse(12 < v)
    self.assertFalse(12 <= v)
    self.assertTrue(12 > v)
    self.assertTrue(12 >= v)
    # pylint: enable=g-generic-assert
    self.assertEqual(v & 3, 3)
    self.assertEqual(11 & v, 3)
    self.assertEqual(v | 8, 15)
    self.assertEqual(16 | v, 23)
    self.assertEqual(v ^ 3, 4)
    self.assertEqual(11 ^ v, 12)
    self.assertEqual(pow(v, 3), 343)
    # TODO(b/178748613): pow(v, 3, 10) fails.
    self.assertEqual(pow(2, v), 128)
    self.assertEqual(-v, -7)
    self.assertEqual(~v, ~7)
    self.assertEqual(abs(v), 7)

  def testSlice(self):
    # [FUNCTION DOC]
    # On a three-element variable, asserts indexing returns the element, assigning through a slice
    # of the last element updates it to four, and the whole variable then equals one, two, four.
    v = self.create_variable([1., 2., 3.])
    self.assertEqual(v[1], 2.)
    v[2].assign(4.)
    self.assertAllEqual(v, [1., 2., 4.])

  # ==== End Variable interface ===

  # ==== Begin ResourceVariable interface ===
  def testHandle(self):
    # [FUNCTION DOC]
    # Asserts the variable's handle is a tensor whose data type is the resource type.
    v = self.create_variable()
    self.assertIsInstance(v.handle, tensor.Tensor)
    self.assertEqual(v.handle.dtype, dtypes.resource)

  def testInGraphMode(self):
    # [FUNCTION DOC]
    # Asserts the internal in-graph-mode flag (used widely internally) is false for a variable
    # created in eager mode.
    # This is protected but used in a lot of places internally.
    v = self.create_variable()
    self.assertFalse(v._in_graph_mode)

  def testUniqueId(self):
    # [FUNCTION DOC]
    # Asserts two separately created variables have different internal unique ids (used by the
    # optimizer to build slot variable keys).
    # This is used in optimizer as part of slot variable key.
    v = self.create_variable()
    w = self.create_variable()
    self.assertNotEqual(v._unique_id, w._unique_id)

  def testIsResourceVariable(self):
    # [FUNCTION DOC]
    # Asserts the resource-variable check reports true for the variable.
    v = self.create_variable()
    self.assertTrue(resource_variable_ops.is_resource_variable(v))
  # ==== End ResourceVariable interface ===

  @combinations.generate(combinations.combine(mode="graph"))
  def testAsGraphElement(self):
    # [FUNCTION DOC]
    # Graph mode only. In a new graph, creates a variable, finalizes the graph (no new operations
    # allowed), runs the initializer, and asserts that evaluating the variable's graph element
    # gives one, showing the graph-element hook does not add operations.
    g = ops.Graph()
    with g.as_default():
      v = self.create_variable(1.)
      g.finalize()
      self.evaluate(v.initializer)
      # _as_graph_element shouldn't create new operations.
      self.assertEqual(self.evaluate(v._as_graph_element()), 1.)


class DistributedVariableInterfaceTest(_VariableInterfaceTestBase):
  # [CLASS DOC]
  # Runs the whole interface-conformance suite against the distributed variable wrapper built from
  # one ordinary variable per device in the base class device list.

  def create_variable(self, initial_value=1., **kwargs):
    # [FUNCTION DOC]
    # HELPER: factory used by inherited tests. Creates an ordinary variable with the initial value
    # and extra keyword arguments on each device, then returns a distributed variable wrapping them.
    variables = []
    for device in self.devices:
      with ops.device(device):
        variables.append(
            variables_lib.Variable(initial_value, **kwargs))
    return values_v2.DistributedVariable(variables)


# Prevent the base class from running.
del _VariableInterfaceTestBase


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.tpu_strategy,
            strategy_combinations.mirrored_strategy_with_two_cpus,
            strategy_combinations.mirrored_strategy_with_two_gpus,
        ],
        enable_packed_handle=[True, False],
        tf_function=[combinations.tf_function, combinations.no_tf_function]))
class DistributedVariableTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests (via the decorator) over TPU, two-CPU mirrored and two-GPU mirrored
  # strategies, packed handle on or off, and compiled-function or eager execution. Each test creates
  # a distributed variable with one component per parameter device of the strategy.

  def create_variable(self, strategy, initial_value, enable_packed_handle,
                      **kwargs):
    # [FUNCTION DOC]
    # HELPER: builds a distributed variable. Creates one ordinary variable per parameter device of
    # the strategy (under that device) from the initial value and extra keyword arguments, and wraps
    # them with the given packed-handle flag. Returns the distributed variable.
    variables = []
    for device in strategy.extended.parameter_devices:
      with ops.device(device):
        variables.append(variables_lib.Variable(initial_value, **kwargs))
    return values_v2.DistributedVariable(
        variables, enable_packed_handle=enable_packed_handle)

  def assertReplica(self, distributed_var, values):
    # [FUNCTION DOC]
    # HELPER: assertion utility. Pairs each component variable of the distributed variable with the
    # corresponding expected value in the list and asserts they are equal.
    for var, value in zip(distributed_var._variables, values):
      self.assertAllEqual(var, value)

  def testRead(self, strategy, enable_packed_handle, tf_function):
    # [FUNCTION DOC]
    # Creates a distributed variable at zero, assigns one on the first parameter device and two on
    # the second. Three functions then read the value and the value-method result under the first
    # device, the second device, and an unrelated CPU device. Asserts the reads give one, two, and
    # one respectively (unrelated device falls back to the primary component).
    v = self.create_variable(strategy, 0., enable_packed_handle)

    with ops.device(strategy.extended.parameter_devices[0]):
      v.assign(1.)
    with ops.device(strategy.extended.parameter_devices[1]):
      v.assign(2.)

    @tf_function
    def read_device0():
      # [FUNCTION DOC]
      # HELPER: under the first parameter device, returns the read value and the value result.
      with ops.device(strategy.extended.parameter_devices[0]):
        return v.read_value(), v.value()

    @tf_function
    def read_device1():
      # [FUNCTION DOC]
      # HELPER: under the second parameter device, returns the read value and the value result.
      with ops.device(strategy.extended.parameter_devices[1]):
        return v.read_value(), v.value()

    @tf_function
    def read_other_device():
      # [FUNCTION DOC]
      # HELPER: under the first CPU device (not a component device in general), returns the read
      # value and the value result.
      with ops.device("CPU:0"):
        return v.read_value(), v.value()

    self.assertAllEqual(read_device0(), [1., 1.])
    self.assertAllEqual(read_device1(), [2., 2.])
    self.assertAllEqual(read_other_device(), [1., 1.])

  def testAssign(self, strategy, enable_packed_handle, tf_function):
    # [FUNCTION DOC]
    # Creates a distributed variable at zero. Assigns one under the first parameter device and two
    # under the second (through functions); asserts components are one and two. Then, under the
    # first CPU device, assigns three and asserts only the primary component changed (three, two).
    v = self.create_variable(strategy, 0., enable_packed_handle)

    @tf_function
    def update_device0():
      # [FUNCTION DOC]
      # HELPER: under the first parameter device, assigns one to the variable.
      with ops.device(strategy.extended.parameter_devices[0]):
        v.assign(1.)

    @tf_function
    def update_device1():
      # [FUNCTION DOC]
      # HELPER: under the second parameter device, assigns two to the variable.
      with ops.device(strategy.extended.parameter_devices[1]):
        v.assign(2.)

    update_device0()
    update_device1()
    self.assertReplica(v, [1., 2.])

    with ops.device("CPU:0"):
      # Update the primary replica.
      v.assign(3.)
      self.assertReplica(v, [3., 2.])

  def testStrategyRun(self, strategy, enable_packed_handle, tf_function):
    # [FUNCTION DOC]
    # Skips on TPU when running eagerly (unsupported). Otherwise creates a distributed variable at
    # zero, runs on every replica an update that assigns a per-replica value (one for the first
    # replica, two for the second), asserts the components are one and two, then runs a read on every
    # replica, gathers the results and asserts they are one and two.
    if (test_util.is_tpu_strategy(strategy) and
        tf_function is combinations.no_tf_function):
      self.skipTest("tpu doesn't support eager")
    v = self.create_variable(strategy, 0., enable_packed_handle)

    @tf_function
    def update(per_replica):
      # [FUNCTION DOC]
      # HELPER: replica function that assigns the given per-replica value to the variable.
      v.assign(per_replica)

    @tf_function
    def read():
      # [FUNCTION DOC]
      # HELPER: replica function returning the variable's read value on the current replica.
      return v.read_value()

    strategy.run(
        update, args=(test_util.create_per_replica(strategy, [1., 2.]),))
    self.assertReplica(v, [1., 2.])
    self.assertAllEqual(
        test_util.gather(strategy, strategy.run(read)), [1., 2.])


def _make_index_slices(values, indices, dense_shape=None):
  # [FUNCTION DOC]
  # HELPER: small shared test utility building sparse updates. Parameters: values and indices are
  # lists (or tensors) of update rows and their positions; dense_shape is optional (default none).
  # Each input is passed through an identity op (the shape only if given) and combined into an
  # indexed-slices object, which is returned.
  if dense_shape:
    dense_shape = array_ops.identity(dense_shape)
  return indexed_slices.IndexedSlices(
      array_ops.identity(values), array_ops.identity(indices), dense_shape)


if __name__ == "__main__":
  test_util.main()
