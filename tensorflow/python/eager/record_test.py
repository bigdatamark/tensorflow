# [MODULE DOC]
# Basic tests for gradient recording: gradient computation through the functional gradient APIs
# on multi-output ops, custom-gradient ops, and the variable watcher scope (trainable vs
# non-trainable variables, nested scopes, variables created inside the scope). Defines two
# custom-gradient helper functions at import time. Runs the test main entry point when executed
# directly.
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
"""Basic tests for gradients."""


from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
# Importing nn_grad for the registration functions.
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables


@custom_gradient.custom_gradient
def two_outputs(a, b):
  # [FUNCTION DOC]
  # HELPER: custom-gradient op used by testTwoOutputs.
  # Computes the matrix product of a and b and the sum of all its entries, returning both as a
  # two-element list together with a hand-written gradient function.
  mm = math_ops.matmul(a, b)
  r = math_ops.reduce_sum(mm)

  def grad(dmm, dr):
    # [FUNCTION DOC]
    # HELPER: gradient callback for the two-output op. Given upstream gradients for the product
    # (dmm) and for the sum (dr), returns a list of gradients for the two inputs: each combines
    # the product gradient with a term from the sum gradient (ones multiplied by dr) through
    # matmul with a transposed operand.
    return [
        math_ops.matmul(dmm, b, transpose_b=True) +
        math_ops.matmul(array_ops.ones_like(b * dr), b, transpose_b=True),
        math_ops.matmul(a, dmm, transpose_b=True) +
        math_ops.matmul(a, array_ops.ones_like(a) * dr, transpose_b=True)
    ]

  return [mm, r], grad


@custom_gradient.custom_gradient
def gradient_is_constant(x):
  # [FUNCTION DOC]
  # HELPER: custom-gradient test op.
  # KNOWN ISSUE: this function is never used in this file (dead code), and its name says the
  # gradient is constant although the forward value is x squared.
  # Returns the square of x with a gradient callback that passes the upstream gradient through
  # unchanged (ignoring x).
  result = x * x

  def grad(dr):
    # [FUNCTION DOC]
    # HELPER: gradient callback returning a one-element list containing the upstream gradient.
    return [dr]

  return result, grad


class TapeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for gradients through the functional gradient APIs, including multi-output and
  # custom-gradient ops. No shared fixtures.

  def testMultiOutput(self):
    # [FUNCTION DOC]
    # Computes gradients in eager mode of a function that adds two matrices, splits the sum into
    # two rows and adds them, then builds the same computation in graph mode and asserts both
    # gradients (with respect to each input) are equal.

    def fn(x, y):
      # [FUNCTION DOC]
      # Function under differentiation: adds inputs, splits into two parts along the first axis
      # and returns their sum.
      c = x + y
      # Multiple outputs from split.
      d, f = array_ops.split(c, 2)
      return d + f

    a = constant_op.constant([[1., 0.], [0., 1.]])
    b = constant_op.constant([[1., 2.], [3., 4.]])
    da, db = backprop.gradients_function(fn, [0, 1])(a, b)
    with context.graph_mode(), self.cached_session():
      tf_a = constant_op.constant([[1, 0], [0, 1]], dtype=dtypes.float32)
      tf_b = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float32)
      tf_c = tf_a + tf_b
      tf_d, tf_f = array_ops.split(tf_c, 2, axis=1)
      tf_e = tf_d + tf_f
      tf_da, tf_db = gradients_impl.gradients(tf_e, [tf_a, tf_b])

      self.assertAllEqual(da, self.evaluate(tf_da))
      self.assertAllEqual(db, self.evaluate(tf_db))

  def testBasicFunctional(self):
    # [FUNCTION DOC]
    # Takes the gradient of the summed matrix product with respect to the argument named a, and
    # asserts it equals ones multiplied by the transpose of b.

    def forward(a, b):
      # [FUNCTION DOC]
      # Function under differentiation: returns the sum of all entries of the product of a and b.
      mm = math_ops.matmul(a, b)
      return math_ops.reduce_sum(mm)

    aa = constant_op.constant([[1., 0.], [0., 1.]])
    bb = constant_op.constant([[1., 2.], [3., 4.]])
    da, = backprop.gradients_function(forward, ['a'])(aa, bb)
    self.assertAllEqual(da,
                        math_ops.matmul(
                            array_ops.ones_like(aa),
                            array_ops.transpose(bb)).numpy())

  def testBasicFunctionalPositionalArg(self):
    # [FUNCTION DOC]
    # Same as the basic functional test but selects the argument by position index 0 instead of by
    # name; asserts the same expected gradient.

    def forward(a, b):
      # [FUNCTION DOC]
      # Function under differentiation: returns the sum of all entries of the product of a and b.
      mm = math_ops.matmul(a, b)
      return math_ops.reduce_sum(mm)

    aa = constant_op.constant([[1., 0.], [0., 1.]])
    bb = constant_op.constant([[1., 2.], [3., 4.]])
    da, = backprop.gradients_function(forward, [0])(aa, bb)
    self.assertAllEqual(da,
                        math_ops.matmul(
                            array_ops.ones_like(aa),
                            array_ops.transpose(bb)).numpy())

  def testBasicFunctionalWithValue(self):
    # [FUNCTION DOC]
    # Uses the value-and-gradient API on the summed matrix product. Asserts the gradient with
    # respect to a equals ones times the transpose of b, and the returned value equals a direct
    # evaluation of the function.

    def forward(a, b):
      # [FUNCTION DOC]
      # Function under differentiation: returns the sum of all entries of the product of a and b.
      mm = math_ops.matmul(a, b)
      return math_ops.reduce_sum(mm)

    aa = constant_op.constant([[1., 0.], [0., 1.]])
    bb = constant_op.constant([[1., 2.], [3., 4.]])
    val, (da,) = backprop.val_and_grad_function(forward, ['a'])(aa, bb)
    self.assertAllEqual(da,
                        math_ops.matmul(
                            array_ops.ones_like(aa),
                            array_ops.transpose(bb)))
    self.assertAllEqual(val, forward(aa, bb))

  def testTwoOutputs(self):
    # [FUNCTION DOC]
    # Differentiates a function using the two-output custom-gradient op in eager mode, and
    # compares with the graph-mode gradient of twice the sum of the product (the expected
    # equivalent); asserts both input gradients are equal.

    def fn(x, y):
      # [FUNCTION DOC]
      # Function under differentiation: calls the custom op and returns the sum output plus the
      # sum of the product output.
      mm, r = two_outputs(x, y)
      return r + math_ops.reduce_sum(mm)

    a = constant_op.constant([[1., 0.], [0., 1.]])
    b = constant_op.constant([[1., 2.], [3., 4.]])
    da, db = backprop.gradients_function(fn, [0, 1])(a, b)
    with context.graph_mode(), self.cached_session():
      tf_a = constant_op.constant([[1, 0], [0, 1]], dtype=dtypes.float32)
      tf_b = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float32)
      tf_mm = math_ops.matmul(tf_a, tf_b)
      tf_rr = 2 * math_ops.reduce_sum(tf_mm)
      tf_da, tf_db = gradients_impl.gradients(tf_rr, [tf_a, tf_b])

      self.assertAllEqual(da, self.evaluate(tf_da))
      self.assertAllEqual(db, self.evaluate(tf_db))

  def testGcTwoOutputs(self):
    # [FUNCTION DOC]
    # Differentiates the first output of sparse softmax cross-entropy (an op with two outputs)
    # for a single logit and label, and asserts the gradient with respect to the logits is zero.

    def fn(x, y):
      # [FUNCTION DOC]
      # Function under differentiation: returns the first output (loss) of sparse softmax
      # cross-entropy for logits x and labels y.
      return nn_ops.sparse_softmax_cross_entropy_with_logits(logits=x,
                                                             labels=y)[0]

    labels = constant_op.constant([0])
    logits = constant_op.constant([[0.0]])
    grad, = backprop.gradients_function(fn, [0])(logits, labels)
    self.assertAllEqual(grad, [[0.0]])

  def testTfTensor(self):
    # [FUNCTION DOC]
    # Differentiates the identity function at a scalar tensor and asserts the gradient is 1.0.

    def fn(x):
      # [FUNCTION DOC]
      # Identity function: returns its input.
      return x

    t = constant_op.constant(1.0)
    g, = backprop.gradients_function(fn, [0])(t)
    self.assertAllEqual(g, 1.0)


class VariableWatcherTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the variable watcher scope. No shared fixtures.

  def testBasic(self):
    # [FUNCTION DOC]
    # Updates two trainable variables inside a watcher scope and asserts both are reported, in
    # order.
    var1 = variables.Variable(0.0)
    var2 = variables.Variable(1.0)
    with record.VariableWatcher() as variable_watcher:
      var1.assign_add(1.0)
      var2.assign_add(2.0)

    self.assertAllEqual(variable_watcher.watched_variables(), (var1, var2))

  def testNonTrainableVariables(self):
    # [FUNCTION DOC]
    # Updates one trainable and one non-trainable variable inside a watcher scope and asserts only
    # the trainable one is reported.
    var1 = variables.Variable(0.0)
    var2 = variables.Variable(1.0, trainable=False)
    with record.VariableWatcher() as variable_watcher:
      var1.assign_add(1.0)
      var2.assign_add(2.0)

    self.assertAllEqual(variable_watcher.watched_variables(), (var1,))

  def testMultipleScopes(self):
    # [FUNCTION DOC]
    # Nests two watcher scopes; updates one variable in the outer scope only and another inside the
    # inner scope. Asserts the outer watcher reports both variables and the inner reports only
    # the second.
    var1 = variables.Variable(0.0)
    var2 = variables.Variable(1.0)
    with record.VariableWatcher() as variable_watcher1:
      var1.assign_add(1.0)
      with record.VariableWatcher() as variable_watcher2:
        var2.assign_add(2.0)

    # variable_watcher1 should see both vars and variable_watcher2 only sees
    # var2
    self.assertAllEqual(variable_watcher1.watched_variables(), (var1, var2))
    self.assertAllEqual(variable_watcher2.watched_variables(), (var2,))

  def testCreateVariables(self):
    # [FUNCTION DOC]
    # Creates and updates two variables inside a watcher scope and asserts both are reported, in
    # order.
    with record.VariableWatcher() as variable_watcher:
      var1 = variables.Variable(0.0)
      var2 = variables.Variable(1.0)
      var1.assign_add(1.0)
      var2.assign_add(2.0)

    self.assertAllEqual(variable_watcher.watched_variables(), (var1, var2))


if __name__ == '__main__':
  test.main()
