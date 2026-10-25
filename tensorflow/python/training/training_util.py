# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Global step utilities for training: public (v1-exported) functions to read, find, create,
# get-or-create and validate the global step, plus private helpers that cache a read of the step
# and build an increment op, and the write_graph alias to the graph writer. Defines the cache
# collection key constant. No import-time side effects beyond export registration.
"""Utility functions for training."""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import cond
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

# Picked a long key value to minimize the chance of collision with user defined
# collection keys.
GLOBAL_STEP_READ_KEY = 'global_step_read_op_cache'

# TODO(drpng): remove this after legacy uses are resolved.
write_graph = graph_io.write_graph


@tf_export(v1=['train.global_step'])
def global_step(sess, global_step_tensor):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the second argument may be the name of an operation, but in
  # eager mode the code calls a numpy conversion on it, which fails for a plain name string.
  # Reads the current global step as a plain integer. Inputs: sess, a session (ignored in eager
  # mode); global_step_tensor, a variable or tensor holding the step. In eager mode it converts
  # the tensor to a number directly; otherwise it runs the tensor in the session. Returns an
  # integer. Errors from the session run propagate. No other side effects.
  """Small helper to get the global step.

  ```python
  # Create a variable to hold the global_step.
  global_step_tensor = tf.Variable(10, trainable=False, name='global_step')
  # Create a session.
  sess = tf.compat.v1.Session()
  # Initialize the variable
  sess.run(global_step_tensor.initializer)
  # Get the variable value.
  print('global_step: %s' % tf.compat.v1.train.global_step(sess,
  global_step_tensor))

  global_step: 10
  ```

  Args:
    sess: A TensorFlow `Session` object.
    global_step_tensor:  `Tensor` or the `name` of the operation that contains
      the global step.

  Returns:
    The global step value.
  """
  if context.executing_eagerly():
    return int(global_step_tensor.numpy())
  return int(sess.run(global_step_tensor))


@tf_export(v1=['train.get_global_step'])
def get_global_step(graph=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring does not mention that when the global step collection holds
  # more than one tensor the function only logs an error and returns nothing instead of raising.
  # Finds the global step tensor of a graph. Input: graph, optional (defaults to the default
  # graph). If the global step collection has exactly one item it is used; if empty, it looks up
  # a tensor named global_step with output index zero and returns nothing if absent. A found
  # tensor is validated with the assert helper. Returns the tensor or nothing. Raises a type
  # error (from validation) for a wrong type, non-integer dtype or non-scalar shape. Logs an
  # error in the ambiguous case.
  """Get the global step tensor.

  The global step tensor must be an integer variable. We first try to find it
  in the collection `GLOBAL_STEP`, or by name `global_step:0`.

  Args:
    graph: The graph to find the global step in. If missing, use default graph.

  Returns:
    The global step variable, or `None` if none was found.

  Raises:
    TypeError: If the global step tensor has a non-integer type, or if it is not
      a `Variable`.

  @compatibility(TF2)
  With the deprecation of global graphs, TF no longer tracks variables in
  collections. In other words, there are no global variables in TF2. Thus, the
  global step functions have been removed  (`get_or_create_global_step`,
  `create_global_step`, `get_global_step`) . You have two options for migrating:

  1. Create a Keras optimizer, which generates an `iterations` variable. This
     variable is automatically incremented when calling `apply_gradients`.
  2. Manually create and increment a `tf.Variable`.

  Below is an example of migrating away from using a global step to using a
  Keras optimizer:

  Define a dummy model and loss:

  >>> def compute_loss(x):
  ...   v = tf.Variable(3.0)
  ...   y = x * v
  ...   loss = x * 5 - x * v
  ...   return loss, [v]

  Before migrating:

  >>> g = tf.Graph()
  >>> with g.as_default():
  ...   x = tf.compat.v1.placeholder(tf.float32, [])
  ...   loss, var_list = compute_loss(x)
  ...   global_step = tf.compat.v1.train.get_or_create_global_step()
  ...   global_init = tf.compat.v1.global_variables_initializer()
  ...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
  ...   train_op = optimizer.minimize(loss, global_step, var_list)
  >>> sess = tf.compat.v1.Session(graph=g)
  >>> sess.run(global_init)
  >>> print("before training:", sess.run(global_step))
  before training: 0
  >>> sess.run(train_op, feed_dict={x: 3})
  >>> print("after training:", sess.run(global_step))
  after training: 1

  Using `get_global_step`:

  >>> with g.as_default():
  ...   print(sess.run(tf.compat.v1.train.get_global_step()))
  1

  Migrating to a Keras optimizer:

  >>> optimizer = tf.keras.optimizers.SGD(.01)
  >>> print("before training:", optimizer.iterations.numpy())
  before training: 0
  >>> with tf.GradientTape() as tape:
  ...   loss, var_list = compute_loss(3)
  ...   grads = tape.gradient(loss, var_list)
  ...   optimizer.apply_gradients(zip(grads, var_list))
  >>> print("after training:", optimizer.iterations.numpy())
  after training: 1

  @end_compatibility
  """
  graph = graph or ops.get_default_graph()
  global_step_tensor = None
  global_step_tensors = graph.get_collection(ops.GraphKeys.GLOBAL_STEP)
  if len(global_step_tensors) == 1:
    global_step_tensor = global_step_tensors[0]
  elif not global_step_tensors:
    try:
      global_step_tensor = graph.get_tensor_by_name('global_step:0')
    except KeyError:
      return None
  else:
    logging.error('Multiple tensors in global_step collection.')
    return None

  assert_global_step(global_step_tensor)
  return global_step_tensor


@tf_export(v1=['train.create_global_step'])
def create_global_step(graph=None):
  # [FUNCTION DOC]
  # Creates the global step variable. Input: graph, optional (default graph). Raises a value
  # error if a global step already exists in that graph. Creates a non-trainable, scalar 64-bit
  # integer variable named global_step, initialized to zero, using the first-replica
  # aggregation, registered in the global variables and global step collections. In eager mode
  # it is created on the CPU device; in graph mode inside the given graph under the root name
  # scope. Returns the new variable.
  """Create global step tensor in graph.

  Args:
    graph: The graph in which to create the global step tensor. If missing, use
      default graph.

  Returns:
    Global step tensor.

  Raises:
    ValueError: if global step tensor is already defined.

  @compatibility(TF2)
  With the deprecation of global graphs, TF no longer tracks variables in
  collections. In other words, there are no global variables in TF2. Thus, the
  global step functions have been removed  (`get_or_create_global_step`,
  `create_global_step`, `get_global_step`) . You have two options for migrating:

  1. Create a Keras optimizer, which generates an `iterations` variable. This
     variable is automatically incremented when calling `apply_gradients`.
  2. Manually create and increment a `tf.Variable`.

  Below is an example of migrating away from using a global step to using a
  Keras optimizer:

  Define a dummy model and loss:

  >>> def compute_loss(x):
  ...   v = tf.Variable(3.0)
  ...   y = x * v
  ...   loss = x * 5 - x * v
  ...   return loss, [v]

  Before migrating:

  >>> g = tf.Graph()
  >>> with g.as_default():
  ...   x = tf.compat.v1.placeholder(tf.float32, [])
  ...   loss, var_list = compute_loss(x)
  ...   global_step = tf.compat.v1.train.create_global_step()
  ...   global_init = tf.compat.v1.global_variables_initializer()
  ...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
  ...   train_op = optimizer.minimize(loss, global_step, var_list)
  >>> sess = tf.compat.v1.Session(graph=g)
  >>> sess.run(global_init)
  >>> print("before training:", sess.run(global_step))
  before training: 0
  >>> sess.run(train_op, feed_dict={x: 3})
  >>> print("after training:", sess.run(global_step))
  after training: 1

  Migrating to a Keras optimizer:

  >>> optimizer = tf.keras.optimizers.SGD(.01)
  >>> print("before training:", optimizer.iterations.numpy())
  before training: 0
  >>> with tf.GradientTape() as tape:
  ...   loss, var_list = compute_loss(3)
  ...   grads = tape.gradient(loss, var_list)
  ...   optimizer.apply_gradients(zip(grads, var_list))
  >>> print("after training:", optimizer.iterations.numpy())
  after training: 1

  @end_compatibility
  """
  graph = graph or ops.get_default_graph()
  if get_global_step(graph) is not None:
    raise ValueError('"global_step" already exists.')
  if context.executing_eagerly():
    with ops.device('cpu:0'):
      return variable_scope.get_variable(
          ops.GraphKeys.GLOBAL_STEP,
          shape=[],
          dtype=dtypes.int64,
          initializer=init_ops.zeros_initializer(),
          trainable=False,
          aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA,
          collections=[
              ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP
          ])
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    return variable_scope.get_variable(
        ops.GraphKeys.GLOBAL_STEP,
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP])


@tf_export(v1=['train.get_or_create_global_step'])
def get_or_create_global_step(graph=None):
  # [FUNCTION DOC]
  # Returns the existing global step of the graph, or creates one if none exists. Input:
  # graph, optional (default graph). Side effect: may add a new variable to the graph. Errors
  # from lookup or creation propagate (type errors for invalid existing steps).
  """Returns and create (if necessary) the global step tensor.

  Args:
    graph: The graph in which to create the global step tensor. If missing, use
      default graph.

  Returns:
    The global step tensor.

  @compatibility(TF2)
  With the deprecation of global graphs, TF no longer tracks variables in
  collections. In other words, there are no global variables in TF2. Thus, the
  global step functions have been removed  (`get_or_create_global_step`,
  `create_global_step`, `get_global_step`) . You have two options for migrating:

  1. Create a Keras optimizer, which generates an `iterations` variable. This
     variable is automatically incremented when calling `apply_gradients`.
  2. Manually create and increment a `tf.Variable`.

  Below is an example of migrating away from using a global step to using a
  Keras optimizer:

  Define a dummy model and loss:

  >>> def compute_loss(x):
  ...   v = tf.Variable(3.0)
  ...   y = x * v
  ...   loss = x * 5 - x * v
  ...   return loss, [v]

  Before migrating:

  >>> g = tf.Graph()
  >>> with g.as_default():
  ...   x = tf.compat.v1.placeholder(tf.float32, [])
  ...   loss, var_list = compute_loss(x)
  ...   global_step = tf.compat.v1.train.get_or_create_global_step()
  ...   global_init = tf.compat.v1.global_variables_initializer()
  ...   optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.1)
  ...   train_op = optimizer.minimize(loss, global_step, var_list)
  >>> sess = tf.compat.v1.Session(graph=g)
  >>> sess.run(global_init)
  >>> print("before training:", sess.run(global_step))
  before training: 0
  >>> sess.run(train_op, feed_dict={x: 3})
  >>> print("after training:", sess.run(global_step))
  after training: 1

  Migrating to a Keras optimizer:

  >>> optimizer = tf.keras.optimizers.SGD(.01)
  >>> print("before training:", optimizer.iterations.numpy())
  before training: 0
  >>> with tf.GradientTape() as tape:
  ...   loss, var_list = compute_loss(3)
  ...   grads = tape.gradient(loss, var_list)
  ...   optimizer.apply_gradients(zip(grads, var_list))
  >>> print("after training:", optimizer.iterations.numpy())
  after training: 1

  @end_compatibility
  """
  graph = graph or ops.get_default_graph()
  global_step_tensor = get_global_step(graph)
  if global_step_tensor is None:
    global_step_tensor = create_global_step(graph)
  return global_step_tensor


@tf_export(v1=['train.assert_global_step'])
def assert_global_step(global_step_tensor):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it asserts a scalar, but the shape check only raises when
  # the shape is fully defined and not scalar; a partially unknown shape is accepted.
  # Validates a candidate global step. Input: a variable, resource variable or tensor. Raises a
  # type error if it is not one of those, if its base dtype is not an integer type, or if its
  # fully-defined shape is not a scalar. Returns nothing.
  """Asserts `global_step_tensor` is a scalar int `Variable` or `Tensor`.

  Args:
    global_step_tensor: `Tensor` to test.
  """
  if not (isinstance(global_step_tensor, variables.Variable) or
          isinstance(global_step_tensor, tensor.Tensor) or
          resource_variable_ops.is_resource_variable(global_step_tensor)):
    raise TypeError('Existing "global_step" must be a Variable or Tensor: %s.' %
                    global_step_tensor)

  if not global_step_tensor.dtype.base_dtype.is_integer:
    raise TypeError('Existing "global_step" does not have integer type: %s' %
                    global_step_tensor.dtype)

  if (global_step_tensor.get_shape().ndims != 0 and
      global_step_tensor.get_shape().is_fully_defined()):
    raise TypeError('Existing "global_step" is not scalar: %s' %
                    global_step_tensor.get_shape())


def _get_global_step_read(graph=None):
  # [FUNCTION DOC]
  # HELPER: lookup used by the global step increment logic.
  # Looks in the graph's cache collection for the global step read tensor. Input: graph,
  # optional (default graph). Returns the single cached tensor or nothing if there is none.
  # Raises a runtime error if the collection holds more than one item.
  """Gets global step read tensor in graph.

  Args:
    graph: The graph in which to create the global step read tensor. If missing,
      use default graph.

  Returns:
    Global step read tensor.

  Raises:
    RuntimeError: if multiple items found in collection GLOBAL_STEP_READ_KEY.
  """
  graph = graph or ops.get_default_graph()
  global_step_read_tensors = graph.get_collection(GLOBAL_STEP_READ_KEY)
  if len(global_step_read_tensors) > 1:
    raise RuntimeError('There are multiple items in collection {}. '
                       'There should be only one.'.format(GLOBAL_STEP_READ_KEY))

  if len(global_step_read_tensors) == 1:
    return global_step_read_tensors[0]
  return None


def _get_or_create_global_step_read(graph=None):
  # [FUNCTION DOC]
  # HELPER: builds the cached read of the global step for the increment logic.
  # Input: graph, optional (default graph). Returns an existing cached read tensor if present;
  # returns nothing if the graph has no global step. Otherwise, for a variable it builds a
  # conditional that reads the value if initialized and falls back to the initial value
  # (so reading before initialization is safe); for non-variables it uses the tensor itself.
  # Adds zero to make a copy as a plain tensor and stores it in the cache collection under the
  # global step name scope. Returns the cached tensor.
  """Gets or creates global step read tensor in graph.

  Args:
    graph: The graph in which to create the global step read tensor. If missing,
      use default graph.

  Returns:
    Global step read tensor if there is global_step_tensor else return None.
  """
  graph = graph or ops.get_default_graph()
  global_step_read_tensor = _get_global_step_read(graph)
  if global_step_read_tensor is not None:
    return global_step_read_tensor
  global_step_tensor = get_global_step(graph)
  if global_step_tensor is None:
    return None
  # add 'zero' so that it will create a copy of variable as Tensor.
  with graph.as_default() as g, g.name_scope(None):
    with g.name_scope(global_step_tensor.op.name + '/'):
      # must ensure that global_step is initialized before this run.
      if isinstance(global_step_tensor, variables.Variable):
        global_step_value = cond.cond(
            variable_v1.is_variable_initialized(global_step_tensor),
            global_step_tensor.read_value,
            lambda: global_step_tensor.initial_value)
      else:
        global_step_value = global_step_tensor

      global_step_read_tensor = global_step_value + 0
      ops.add_to_collection(GLOBAL_STEP_READ_KEY, global_step_read_tensor)
  return _get_global_step_read(graph)


def _increment_global_step(increment, graph=None):
  # [FUNCTION DOC]
  # HELPER: used by training hooks and loops to bump the global step.
  # Inputs: increment, amount to add; graph, optional (default graph). Raises a value error if
  # the graph has no global step. Builds an add-assign op on the global step that depends on the
  # cached read tensor (created if needed), under the root name scope. Returns that op/tensor.
  graph = graph or ops.get_default_graph()
  global_step_tensor = get_global_step(graph)
  if global_step_tensor is None:
    raise ValueError(
        'Global step tensor should be created by '
        'tf.train.get_or_create_global_step before calling increment.')
  global_step_read_tensor = _get_or_create_global_step_read(graph)
  with graph.as_default() as g, g.name_scope(None):
    with g.name_scope(global_step_tensor.op.name + '/'):
      with ops.control_dependencies([global_step_read_tensor]):
        return state_ops.assign_add(global_step_tensor, increment)
