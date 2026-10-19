# [MODULE DOC]
# Unit tests for the wrap_function module, which turns v1-style graph-building Python functions
# into callable eager concrete functions. Covers wrapping with variables, pruning to a subset of
# inputs and outputs (including ragged inputs, operations and captured variables), gradients
# through pruned functions, collection isolation between wrapped functions, building functions
# from a graph definition, variable-name uniquification, and the WrappedGraph helper that can
# share variables across wrapped functions. Contains two test cases and a main entry point that
# enables eager execution and runs the tests. No import-time side effects beyond imports.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import os


from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer as graph_def_importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


class WrapFunctionTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the module-level wrap_function API: wrapping graph-style functions,
  # pruning wrapped functions, variable lifting, collection isolation and building functions
  # from graph definitions. Has no shared fixtures beyond the base test case.

  def testDocString(self):
    # [FUNCTION DOC]
    # Test of the documented usage example: wraps a function taking a tensor and a Python
    # boolean flag, with the flag fixed to add; calling twice shows the lifted variable keeps
    # state (6 then 7). Wrapping again with the flag set to subtract gives a fresh trace and
    # fresh variable, so calls give 4 then 3.

    def f(x, do_add):
      # [FUNCTION DOC]
      # Graph-style function under test: creates a variable with initial value five and
      # assigns-adds or assigns-subtracts the input depending on the do_add flag, returning the
      # variable's value read after the update (control-dependent on the update).
      v = variables.Variable(5.0)
      if do_add:
        op = v.assign_add(x)
      else:
        op = v.assign_sub(x)
      with ops.control_dependencies([op]):
        return v.read_value()

    f_add = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32), True])

    self.assertAllEqual(f_add(1.0), 6.0)
    self.assertAllEqual(f_add(1.0), 7.0)

    # Can call tf.compat.v1.wrap_function again to get a new trace, a new set
    # of variables, and possibly different non-template arguments.
    f_sub = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32), False])

    self.assertAllEqual(f_sub(1.0), 4.0)
    self.assertAllEqual(f_sub(1.0), 3.0)

  def testPrune(self):
    # [FUNCTION DOC]
    # Test that a wrapped function of two float inputs can be pruned to one input and one
    # output (the square of the first input); the pruned function called with 2.0 returns a
    # list holding 4.0. The tensors to feed and fetch are recorded by the traced function.

    x_in = []
    x_out = []

    def f(x, y):
      # [FUNCTION DOC]
      # Traced function: records its first input and the squared first input into the outer
      # lists, and returns the square of x plus twice the square of y.
      x_in.append(x)
      xx = x * x
      x_out.append(xx)
      return xx, 2 * y*y

    f_wrapped = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtypes.float32)] * 2)

    f_pruned = f_wrapped.prune(x_in[0], [x_out[0]])
    self.assertAllEqual(f_pruned(ops.convert_to_tensor(2.0)), [4.0])

  def testPruneRagged(self):
    # [FUNCTION DOC]
    # Test pruning with a ragged-tensor input: wraps a function whose first argument has a
    # ragged spec, prunes it to that input and its squared output, then calls the pruned
    # function with the ragged tensor's flat values and row splits (the convention for
    # structured inputs) and expects the ragged result of squaring each element.

    x_in = []
    x_out = []

    def f(x, y):
      # [FUNCTION DOC]
      # Traced function: records the ragged input and its square into outer lists and returns
      # the square of x together with the square of y.
      x_in.append(x)
      xx = x * x
      x_out.append(xx)
      return xx, y * y

    x_spec = ragged_tensor.RaggedTensorSpec([None, None], dtypes.float32)
    y_spec = tensor_spec.TensorSpec((), dtypes.float32)

    f_wrapped = wrap_function.wrap_function(f, [x_spec, y_spec])

    f_pruned = f_wrapped.prune(x_in[0], x_out[0])
    rt = ragged_factory_ops.constant([[1.0, 2.0], [3.0]])
    expected = ragged_factory_ops.constant_value([[1.0, 4.0], [9.0]])

    # Note: when we call f_pruned, we must pass the RaggedTensor in using
    # its components, since that's the current convention for how concrete
    # functions handle structured inputs.
    self.assertAllEqual(f_pruned(rt.values, rt.row_splits), expected)

  def _assert_single_captured_variable_argument(self, graph_def):
    # [FUNCTION DOC]
    # HELPER: assertion helper for testVariableLifting.
    # Asserts that the given graph definition's library holds exactly one function and that the
    # function has exactly one input argument whose type is the resource type (a captured
    # variable). Fails the test otherwise; returns nothing.
    # The single FunctionDef should have one argument, a captured variable
    function_def, = graph_def.library.function
    self.assertLen(function_def.signature.input_arg, 1)
    function_arg, = function_def.signature.input_arg
    self.assertEqual(dtypes.resource, dtypes.as_dtype(function_arg.type))

  def testVariableLifting(self):
    # [FUNCTION DOC]
    # Test that variables restored from an imported meta graph are lifted out of a wrapped
    # function. Builds and saves a graph with one variable in a temp directory, wraps a function
    # that imports the meta graph, and checks one lifted variable, that a pruned initializer
    # shares the variable and captures, that running the initializer inside an eager-compiled
    # function sets the variable, that both resulting function graphs take a single resource
    # argument, and that the wrapped and pruned functions return the variable plus one.
    save_prefix = os.path.join(self.get_temp_dir(), 'meta_graph_test')

    export_graph = ops.Graph()
    with export_graph.as_default():
      v = variables.Variable(1.)
      array_ops.identity(v + 1., name='output')
      saver = saver_lib.Saver([v])
      with self.test_session() as session:
        session.run(v.initializer)
        saver.save(session, save_prefix)

    def importer():
      # [FUNCTION DOC]
      # Traced function: imports the saved meta graph into the default graph and returns the
      # graph element named output.
      saver_lib.import_meta_graph(save_prefix + '.meta')
      return ops.get_default_graph().as_graph_element('output:0')

    wrapped = wrap_function.wrap_function(importer, [])
    lifted_variables = list(wrapped.graph.variables)
    self.assertLen(lifted_variables, 1)
    initializer = wrapped.prune(
        [], wrapped.graph.as_graph_element(v.initializer.name))
    self.assertEqual(lifted_variables, list(initializer.graph.variables))
    self.assertEqual(initializer.graph.external_captures,
                     wrapped.graph.external_captures)

    @def_function.function
    def wraps_initializer():
      # [FUNCTION DOC]
      # Compiled function that simply invokes the pruned initializer; returns nothing.
      initializer()

    wraps_initializer()
    self.assertEqual(1., lifted_variables[0].numpy())
    wrapped_initializer_graphdef = (
        wraps_initializer.get_concrete_function().graph.as_graph_def())
    self._assert_single_captured_variable_argument(wrapped_initializer_graphdef)

    @def_function.function
    def wraps_wrapped():
      # [FUNCTION DOC]
      # Compiled function that calls the wrapped function and returns its result.
      return wrapped()

    # Verify that the original graph also has the correct signature.
    wrapped_wrapped_graphdef = (
        wraps_wrapped.get_concrete_function().graph.as_graph_def())
    self._assert_single_captured_variable_argument(wrapped_wrapped_graphdef)
    # Now check that the graph runs wrapped, from eager, and when pruned.
    self.assertAllEqual(wraps_wrapped().numpy(),
                        lifted_variables[0].numpy() + 1.)
    self.assertAllEqual(wrapped().numpy(), lifted_variables[0].numpy() + 1.)
    pruned = wrapped.prune([], wrapped.graph.as_graph_element('output:0'))
    self.assertAllEqual(wrapped().numpy(), pruned().numpy())

  def testNoArguments(self):
    # [FUNCTION DOC]
    # Test that a function with no inputs can be wrapped with an empty signature and that
    # calling it returns the constant one.

    def f():
      # [FUNCTION DOC]
      # Traced function with no inputs; returns the float constant one.
      return constant_op.constant(1.)

    f_wrapped = wrap_function.wrap_function(f, [])
    self.assertAllEqual(1.0, f_wrapped())

  def testPruneCaptures(self):
    # [FUNCTION DOC]
    # Test pruning a wrapped function that uses both an outer eager variable (captured) and a
    # variable created inside the trace. The unpruned call gives 6.0; pruning with the
    # function's own inputs as feeds, and with an empty feed list, must each still return 6.0.

    v1 = variables.Variable(2.)

    def f():
      # [FUNCTION DOC]
      # Traced function: creates a variable of value three and returns the product of the
      # captured outer variable, the new variable and one, with the output named fetch.
      v2 = variables.Variable(3.)
      return array_ops.identity(v1 * v2 * constant_op.constant(1.), 'fetch')

    f_wrapped = wrap_function.wrap_function(f, [])
    self.assertAllEqual(6.0, f_wrapped())

    # Test pruning directly on the inputs
    pruned = f_wrapped.prune(
        feeds=f_wrapped.inputs,
        fetches=f_wrapped.graph.get_tensor_by_name('fetch:0'))
    self.assertAllEqual(6.0, pruned())

    # Test pruning with no inputs
    pruned = f_wrapped.prune(
        feeds=(),
        fetches=f_wrapped.graph.get_tensor_by_name('fetch:0'))
    self.assertAllEqual(6.0, pruned())

  def testCollectionsIsolation(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the final assertion indexes the losses key string (taking its first
    # character) rather than the collection result, so it compares an empty list with a
    # tensor and does not really test what the preceding comment claims.
    # Test that graph collections of two different wrapped functions are isolated. Each traced
    # function creates a variable and adds a loss to the losses collection; each wrapped graph
    # must then hold exactly one loss and one trainable variable (the one it created), and the
    # two graphs must not share collection entries. Expected outputs are 6.0 and 8.0.

    v1 = variables.Variable(2.)
    v2_holder = []
    def f():
      # [FUNCTION DOC]
      # Traced function: creates a variable of value three, records it in the outer list, adds
      # a loss tensor to the losses collection and returns the product of the outer variable,
      # the new variable and one, named fetch.
      v2 = variables.Variable(3.)
      v2_holder.append(v2)
      ops.add_to_collection(ops.GraphKeys.LOSSES, v2 * constant_op.constant(3.))
      return array_ops.identity(v1 * v2 * constant_op.constant(1.), 'fetch')

    f_wrapped = wrap_function.wrap_function(f, [])
    self.assertAllEqual(6.0, f_wrapped())
    self.assertEqual(
        len(f_wrapped.graph.get_collection(ops.GraphKeys.LOSSES)), 1)
    f_var_collection = f_wrapped.graph.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertEqual(len(f_var_collection), 1)
    self.assertIs(f_var_collection[0], v2_holder[0])

    v3_holder = []
    def g():
      # [FUNCTION DOC]
      # Traced function analogous to the first one but creating a variable of value four;
      # records it in a second outer list, adds a loss to the losses collection and returns the
      # product of the outer variable, the new variable and one, named fetch.
      v3 = variables.Variable(4.)
      v3_holder.append(v3)
      ops.add_to_collection(ops.GraphKeys.LOSSES, v3 * constant_op.constant(3.))
      return array_ops.identity(v1 * v3 * constant_op.constant(1.), 'fetch')

    g_wrapped = wrap_function.wrap_function(g, [])
    self.assertAllEqual(8.0, g_wrapped())
    self.assertEqual(
        len(g_wrapped.graph.get_collection(ops.GraphKeys.LOSSES)), 1)
    g_var_collection = g_wrapped.graph.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES)
    self.assertEqual(len(g_var_collection), 1)
    self.assertIs(g_var_collection[0], v3_holder[0])

    # Both have only one value, and their values aren't equal. So no sharing.
    self.assertIsNot(g_wrapped.graph.get_collection(ops.GraphKeys.LOSSES[0]),
                     f_wrapped.graph.get_collection(ops.GraphKeys.LOSSES)[0])

  def testGradientsOfPrune(self):
    # [FUNCTION DOC]
    # Test that gradients flow through both a wrapped function and its pruned version. For the
    # product of two variables and an input, the output is 6.0 and the gradients with respect
    # to the input and the two variables are 6.0, 3.0 and 2.0 in each case.

    v1 = variables.Variable(2.)
    v2_holder = []

    def f(z):
      # [FUNCTION DOC]
      # Traced function: creates a variable of value three, records it in the outer list and
      # returns the product of the outer variable, the new variable and the input z, named
      # fetch.
      v2 = variables.Variable(3.)
      v2_holder.append(v2)
      return array_ops.identity(v1 * v2 * z, 'fetch')

    f_wrapped = wrap_function.wrap_function(
        f, [tensor_spec.TensorSpec((), dtype=dtypes.float32)])

    x = constant_op.constant(1.)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      out = f_wrapped(x)
    grads = tape.gradient(out, [x, v1, v2_holder[0]])

    self.assertAllEqual(6.0, out)
    self.assertAllEqual([6.0, 3.0, 2.0], grads)

    pruned = f_wrapped.prune(
        feeds=f_wrapped.inputs,
        fetches=f_wrapped.graph.get_tensor_by_name('fetch:0'))

    x = constant_op.constant(1.)
    with backprop.GradientTape() as tape:
      tape.watch(x)
      out = pruned(x)
    grads = tape.gradient(out, [x, v1, v2_holder[0]])

    self.assertAllEqual(6.0, out)
    self.assertAllEqual([6.0, 3.0, 2.0], grads)

  def testPruneOperations(self):
    # [FUNCTION DOC]
    # Test pruning with operations as fetches. First, fetching only the increment operation
    # yields a function returning a tuple holding nothing and increments the variable once.
    # Second, a function with a placeholder step input is pruned to fetch the increment
    # operation and another tensor; the pruned function takes the step and the captured variable
    # resource as inputs, returns nothing for the operation and one for the tensor, and the
    # variable becomes 3. A function pruned to only the tensor must not increment the variable.

    v = variables.Variable(0)

    def f():
      # [FUNCTION DOC]
      # Traced function with no return value: increments the outer variable by one without
      # reading it, naming the operation increment.
      v.assign_add(1, name='increment', read_value=False)

    f_wrapped = wrap_function.wrap_function(f, [])
    pruned = f_wrapped.prune(
        feeds=(),
        fetches=(f_wrapped.graph.get_operation_by_name('increment'),))
    self.assertEqual((None,), pruned())
    self.assertEqual(1, self.evaluate(v))

    del f, f_wrapped

    def f1():
      # [FUNCTION DOC]
      # Traced function: increments the outer variable by a scalar integer placeholder named
      # step (operation named increment) and returns a constant one named other.
      v.assign_add(
          array_ops.placeholder(shape=[], dtype=dtypes.int32, name='step'),
          name='increment', read_value=False)
      return constant_op.constant(1, name='other')

    f_wrapped = wrap_function.wrap_function(f1, [])
    increments = f_wrapped.prune(
        feeds=(f_wrapped.graph.get_tensor_by_name('step:0')),
        fetches=(f_wrapped.graph.get_operation_by_name('increment'),
                 f_wrapped.graph.get_tensor_by_name('other:0')))
    first_output, second_output = increments(constant_op.constant(2))
    self.assertEqual(['step:0', 'increment/resource:0'],
                     [t.name for t in increments.inputs])
    self.assertIs(None, first_output)
    self.assertEqual(1, second_output.numpy())
    self.assertEqual(3, v.numpy())
    does_not_increment = f_wrapped.prune(
        feeds=(f_wrapped.graph.get_tensor_by_name('step:0')),
        fetches=f_wrapped.graph.get_tensor_by_name('other:0'))
    self.assertEqual(1, does_not_increment(constant_op.constant(3)).numpy())
    self.assertEqual(3, v.numpy())

  def testPruneStatefulOpsFromWrappedFunc(self):
    # [FUNCTION DOC]
    # Test that a wrapped function follows graph pruning rules: stateful assign operations not
    # in the fan-in of the returned value are dropped. Wrapping a function that increments two
    # variables but returns its input, with and without an explicit name, must return the input
    # while leaving both variables at zero.

    v0 = variables.Variable(0)
    v1 = variables.Variable(0)

    # When we wrap a function, we expect it to be executed with 'tf.Graph`
    # rules: it's allowed to prune all ops that are not in transitive fanin of
    # the fetches.
    def f(x):
      # [FUNCTION DOC]
      # Traced function: increments two outer variables (stateful operations) and returns its
      # input unchanged.
      v0.assign_add(1, name='increment_v0')
      v1.assign_add(1, name='increment_v1')
      return x

    f_wrapped = wrap_function.wrap_function(f, [1])

    self.assertEqual(1, f_wrapped().numpy())
    self.assertEqual(0, v0.numpy())
    self.assertEqual(0, v1.numpy())

    f_wrapped_with_name = wrap_function.wrap_function(f, [2], name='func')

    self.assertEqual(2, f_wrapped_with_name().numpy())
    self.assertEqual(0, v0.numpy())
    self.assertEqual(0, v1.numpy())

  def test_operation_returned(self):
    # [FUNCTION DOC]
    # Test pruning where the fetch is an operation, given as an operation object, a name string
    # or a tensor-info message. The pruned function lists that operation as its control output;
    # the variable stays zero until the pruned function is called, after which it equals one.

    v = variables.Variable(0)

    def f():
      # [FUNCTION DOC]
      # Traced function with no return value: assigns one to the outer variable without
      # reading it, naming the operation assign_to_v.
      v.assign(1, read_value=False, name='assign_to_v')

    f_wrapped = wrap_function.wrap_function(f, [])
    operation_to_fetch = f_wrapped.graph.get_operation_by_name('assign_to_v')
    f_pruned = f_wrapped.prune(
        [], operation_to_fetch)
    self.assertEqual(
        ['assign_to_v'],
        [operation.name for operation in f_pruned.graph.control_outputs])
    self.assertEqual(0, v.numpy())
    f_pruned()
    self.assertEqual(1, v.numpy())
    f_wrapped.prune([], 'assign_to_v')()
    f_wrapped.prune([], meta_graph_pb2.TensorInfo(name='assign_to_v'))()

  def test_function_from_graph_def(self):
    # [FUNCTION DOC]
    # Test that a function can be rebuilt from a graph definition. Traces an add-one function
    # with an unknown leading dimension, exports its graph definition, rebuilds a callable from
    # it using the input and output tensor names, and expects calling it on one to give two.
    @def_function.function
    def make_graph_def(x):
      # [FUNCTION DOC]
      # Compiled function under trace; returns its input plus one.
      return x + 1.

    original_func_graph = make_graph_def.get_concrete_function(
        tensor_spec.TensorSpec([None, 2], dtypes.float32)).graph
    graph_def = original_func_graph.as_graph_def()
    revived_function = wrap_function.function_from_graph_def(
        graph_def, inputs=original_func_graph.inputs[0].name,
        outputs=original_func_graph.outputs[0].name)
    self.assertEqual(2., revived_function(constant_op.constant(1.)).numpy())

  def test_create_variables_with_same_name(self):
    # [FUNCTION DOC]
    # Test that variables created with the same name in one wrapped function get unique names
    # (v and v_1) and that the uniquification restarts in a separate wrapping call, checked
    # through the holder's variable table for two different traced functions.
    def f():
      # [FUNCTION DOC]
      # Traced function: creates two variables both named v (values zero and one) and returns
      # them as a pair.
      v1 = variables.Variable(0, name='v')
      v2 = variables.Variable(1, name='v')
      return v1, v2

    f_wrapped = wrap_function.wrap_function(f, [])
    self.assertDictEqual(
        {'v:0': 0, 'v_1:0': 1},  # assert that variable names are uniquified
        {v.name: v.numpy()
         for v in f_wrapped._variable_holder.variables.values()})

    # Uniquification should reset in separate calls to wrap_function.
    def f2():
      # [FUNCTION DOC]
      # Traced function like the previous one but with initial values three and four; returns
      # the two same-named variables as a pair.
      v1 = variables.Variable(3, name='v')
      v2 = variables.Variable(4, name='v')
      return v1, v2

    f_wrapped_2 = wrap_function.wrap_function(f2, [])
    self.assertDictEqual(
        {'v:0': 3, 'v_1:0': 4},
        {v.name: v.numpy()
         for v in f_wrapped_2._variable_holder.variables.values()})


class WrappedGraphTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the WrappedGraph helper, which wraps several functions into one shared graph
  # and optionally shares variables through a variable holder. Covers variable creation,
  # collections, variable sharing in the same and in different graphs, imported dataset
  # functions and wrapped functions returning operations.

  def testAddFunction(self):
    # [FUNCTION DOC]
    # Test that a function mixing a plain variable and a get-variable variable, both named v,
    # gives the same result when wrapped in a WrappedGraph as when evaluated in a regular
    # session graph. Compares the session result against the wrapped call with input five.

    def fn(x):
      # [FUNCTION DOC]
      # Function under test: creates two integer variables named v (initial three and four)
      # and returns their sum plus the input x.
      v = variables.Variable(3, name='v')
      v2 = variable_scope.get_variable(
          'v', initializer=init_ops.Constant(4), shape=[], dtype=dtypes.int32)
      return v + v2 + x

    with self.cached_session() as sess:
      result = fn(constant_op.constant(5))
      sess.run(variables.global_variables_initializer())
      expected = sess.run(result)

    g = wrap_function.WrappedGraph()
    signature = [tensor_spec.TensorSpec([], dtypes.int32)]
    wrapped_fn = g.wrap_function(fn, signature)
    self.assertEqual(expected, wrapped_fn(constant_op.constant(5)).numpy())

  def testCollections(self):
    # [FUNCTION DOC]
    # Test that collections populated by variable creation are visible on the WrappedGraph's
    # graph and also inside a later wrapped function: one trainable variable, two entries in
    # collection a and one in collection b.

    def fn(x):
      # [FUNCTION DOC]
      # Function under test: creates a non-trainable v1 variable in collection a and a
      # get-variable variable in collections a and b, and returns their sum plus x.
      v = variable_v1.VariableV1(
          3, name='v', trainable=False, collections=['a'])
      v2 = variable_scope.get_variable(
          'v', initializer=init_ops.Constant(4), shape=[], dtype=dtypes.int32,
          collections=['a', 'b'])
      return v + v2 + x

    def assert_collections(graph):
      # [FUNCTION DOC]
      # HELPER: assertion helper for testCollections.
      # Asserts the given graph has one trainable variable, two items in collection a and one
      # item in collection b. Returns nothing.
      self.assertLen(graph.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES), 1)
      self.assertLen(graph.get_collection('a'), 2)
      self.assertLen(graph.get_collection('b'), 1)

    g = wrap_function.WrappedGraph()
    g.wrap_function(fn, [tensor_spec.TensorSpec([], dtypes.int32)])
    assert_collections(g.graph)

    def assert_fn():
      # [FUNCTION DOC]
      # Function wrapped to check collections from inside a trace: asserts the collection
      # counts on the default graph and returns one (a return value is required).
      assert_collections(ops.get_default_graph())
      return 1  # Return is required

    # Assert that collections are accessible within a wrapped function.
    g.wrap_function(assert_fn, [])

  def testShareVariablesSameGraph(self):
    # [FUNCTION DOC]
    # Test variable sharing among functions wrapped by one WrappedGraph. Functions using the
    # same scope and variable name share one variable whose initial value comes from the first
    # wrapped function (three); one function uses a different scope and stays independent.
    # Checks computed results before and after an increment, and that the graph's variable table
    # has exactly the two keys reuse/v and no_reuse/v.

    def add_v1(x):
      # [FUNCTION DOC]
      # Function under test: gets or reuses variable v in scope reuse (initial three) and
      # returns it plus x.
      with variable_scope.variable_scope(
          'reuse', reuse=variable_scope.AUTO_REUSE):
        v = variable_scope.get_variable(
            'v', initializer=init_ops.Constant(3), shape=[], dtype=dtypes.int32)
      return v + x

    def subtract_v1(x):
      # [FUNCTION DOC]
      # Function under test: gets or reuses variable v in scope reuse (initial four, ignored if
      # it already exists) and returns it minus x.
      with variable_scope.variable_scope(
          'reuse', reuse=variable_scope.AUTO_REUSE):
        v = variable_scope.get_variable(
            'v', initializer=init_ops.Constant(4), shape=[], dtype=dtypes.int32)
      return v - x

    def different_variable_fn_v1(x):
      # [FUNCTION DOC]
      # Function under test: gets variable v in a separate scope no_reuse (initial five) and
      # returns it multiplied by x.
      with variable_scope.variable_scope(
          'no_reuse', reuse=variable_scope.AUTO_REUSE):
        v = variable_scope.get_variable(
            'v', initializer=init_ops.Constant(5), shape=[], dtype=dtypes.int32)
      return v * x

    def increment_variable_v1(x):
      # [FUNCTION DOC]
      # Function under test: gets or reuses variable v in scope reuse (initial six, ignored if
      # it already exists), adds x to it in place and returns the updated value.
      with variable_scope.variable_scope(
          'reuse', reuse=variable_scope.AUTO_REUSE):
        v = variable_scope.get_variable(
            'v', initializer=init_ops.Constant(6), shape=[], dtype=dtypes.int32)
      return v.assign_add(x)

    g = wrap_function.WrappedGraph()
    signature = [tensor_spec.TensorSpec([], dtypes.int32)]
    add = g.wrap_function(add_v1, signature)
    subtract = g.wrap_function(subtract_v1, signature)
    different_variable_fn = g.wrap_function(different_variable_fn_v1, signature)
    increment_variable = g.wrap_function(increment_variable_v1, signature)

    self.assertEqual(10, add(constant_op.constant(7)).numpy())
    self.assertEqual(35, different_variable_fn(constant_op.constant(7)).numpy())

    # The shared variable has a starting value of 3 because add_v1 was wrapped
    # first.
    self.assertEqual(-4, subtract(constant_op.constant(7)).numpy())
    self.assertEqual(10, increment_variable(constant_op.constant(7)).numpy())

    # Check that variable updates
    self.assertEqual(17, add(constant_op.constant(7)).numpy())
    self.assertEqual(3, subtract(constant_op.constant(7)).numpy())

    # Sanity check - result from this function shouldn't change.
    self.assertEqual(35, different_variable_fn(constant_op.constant(7)).numpy())

    self.assertAllEqual({'reuse/v', 'no_reuse/v'}, set(g.variables.keys()))

  def testShareVariablesDifferentGraphs(self):
    # [FUNCTION DOC]
    # Test variable sharing across separate WrappedGraph objects that use one shared variable
    # holder with sharing enabled. Variables with the same name share state even in different
    # graphs (first creator wins, initial value three); a variable in a different name scope
    # stays separate. Checks results, updates, and that the holder's table has exactly the keys
    # v and different_scope/v.

    def add_v1(x):
      # [FUNCTION DOC]
      # Function under test: creates variable v with initial three (shared via the holder) and
      # returns it plus x.
      v = variables.Variable(3, name='v')
      return v + x

    def subtract_v1(x):
      # [FUNCTION DOC]
      # Function under test: creates variable v with initial four (reuses the shared one if it
      # exists) and returns it minus x.
      v = variables.Variable(4, name='v')
      return v - x

    def different_variable_fn_v1(x):
      # [FUNCTION DOC]
      # Function under test: creates variable v with initial five inside the name scope
      # different_scope (a separate shared entry) and returns it multiplied by x.
      with ops.name_scope('different_scope'):
        v = variables.Variable(5, name='v')
      return v * x

    def increment_variable_v1(x):
      # [FUNCTION DOC]
      # Function under test: creates variable v with initial six (reuses the shared one if it
      # exists), adds x to it in place and returns the updated value.
      v = variables.Variable(6, name='v')
      return v.assign_add(x)

    signature = [tensor_spec.TensorSpec([], dtypes.int32)]
    vh = wrap_function.VariableHolder(share_variables=True)
    new_graph = lambda: wrap_function.WrappedGraph(variable_holder=vh)

    add = new_graph().wrap_function(add_v1, signature)
    subtract = new_graph().wrap_function(subtract_v1, signature)
    different_variable_fn = new_graph().wrap_function(
        different_variable_fn_v1, signature)
    increment_variable = new_graph().wrap_function(
        increment_variable_v1, signature)

    self.assertEqual(10, add(constant_op.constant(7)).numpy())
    self.assertEqual(35, different_variable_fn(constant_op.constant(7)).numpy())

    # Because the variable in add_v1 was created first, its starting value is 3
    # instead of the values defined in subtract_v1 or increment_variable_v1.
    self.assertEqual(-4, subtract(constant_op.constant(7)).numpy())
    self.assertEqual(10, increment_variable(constant_op.constant(7)).numpy())

    # Check that variable updates
    self.assertEqual(17, add(constant_op.constant(7)).numpy())
    self.assertEqual(3, subtract(constant_op.constant(7)).numpy())

    # Sanity check - result from this function shouldn't change.
    self.assertEqual(35, different_variable_fn(constant_op.constant(7)).numpy())

    self.assertAllEqual({'v', 'different_scope/v'}, set(vh.variables.keys()))

  @test_util.run_in_graph_and_eager_modes
  def testImportedFunctionsRegistered(self):
    # [FUNCTION DOC]
    # Test, in both graph and eager modes (skipped on GPU), that functions from an imported
    # graph definition (a dataset reduce taking a variant dataset) are registered so that a
    # wrapped function can run them. Builds the graph with a variant placeholder, wraps a
    # function importing it, feeds a dataset converted to a variant and evaluates the result.
    if test_util.is_gpu_available():
      self.skipTest('not a GPU test')
    with ops.Graph().as_default() as graph:
      x = array_ops.placeholder(dtypes.variant, shape=[], name='foo')
      ds = dataset_ops.from_variant(x, structure=(
          tensor_spec.TensorSpec([], dtypes.int32)))
      y = ds.reduce(array_ops.zeros([], dtype=dtypes.int32), lambda p, q: p + q)

    graph_def = graph.as_graph_def()

    def fn_to_wrap(a):
      # [FUNCTION DOC]
      # Function to wrap: imports the captured graph definition mapping the variant
      # placeholder to the argument a and returns the first requested element (the reduce
      # result).
      returned_elements = graph_def_importer.import_graph_def(
          graph_def, input_map={x.name: a}, return_elements=[y.name])
      return returned_elements[0]

    wrapped_fn = wrap_function.wrap_function(
        fn_to_wrap, [tensor_spec.TensorSpec((), dtypes.variant)])
    ds = dataset_ops.Dataset.from_tensor_slices([10, 20])
    v = dataset_ops.to_variant(ds)
    self.evaluate(wrapped_fn(v))

  def testReturnOp(self):
    # [FUNCTION DOC]
    # Test that a function returning an operation (not a tensor) can be wrapped: the variable
    # created in the WrappedGraph starts at three and becomes twelve after calling the wrapped
    # function with twelve.

    def update_var_v1(x):
      # [FUNCTION DOC]
      # Function under test: creates variable v with initial three, assigns x to it and
      # returns the assign operation itself.
      v = variables.Variable(3, name='v')
      update_op = state_ops.assign(v, x).op
      return update_op

    g = wrap_function.WrappedGraph()
    signature = [tensor_spec.TensorSpec([], dtypes.int32)]
    update_var = g.wrap_function(update_var_v1, signature)

    self.assertEqual(g.variables['v'].numpy(), 3)
    update_var(constant_op.constant(12))
    self.assertEqual(g.variables['v'].numpy(), 12)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
