# [MODULE DOC]
# Unit tests for the evaluation module of the training package (private evaluate-once routine,
# stop-after-N-evals hook and eval-step helper). A tiny logistic classifier is trained on a
# memorizable synthetic dataset to produce checkpoints, which are then restored and evaluated
# in graph mode. Defines small helper functions and one test case class; runs under the test
# main entry point. No import-time registrations.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for tf.training.evaluation."""

import os

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import metrics as metrics_module
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import evaluation
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import training

_USE_GLOBAL_STEP = 0


def logistic_classifier(inputs):
  # [FUNCTION DOC]
  # HELPER: shared model builder for the tests in this file.
  # Builds a single dense layer with one output unit and sigmoid activation on the given input
  # tensor and returns the resulting prediction tensor. Adds layer variables to the graph.
  return layers.dense(inputs, 1, activation=math_ops.sigmoid)


def local_variable(init_value, name):
  # [FUNCTION DOC]
  # HELPER: shared fixture builder for the tests in this file.
  # Creates a non-trainable 32-bit float variable with the given name and initial value
  # (init_value) and registers it only in the local-variables collection. Returns the
  # variable. Adds it to the current graph.
  return variable_scope.get_variable(
      name,
      dtype=dtypes.float32,
      initializer=init_value,
      trainable=False,
      collections=[ops.GraphKeys.LOCAL_VARIABLES])


class EvaluateOnceTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the private evaluate-once routine together with its stop-after-N hook
  # and eval-step helper. Shared fixture: a 16 by 4 one-hot-like input set and random binary
  # labels (seeded) created in setUp, plus a helper that trains and checkpoints a classifier.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup run before each test. Seeds the random generator, then builds a 16 by 4
    # input array and 16 random binary labels so that each row has a single active feature
    # position determined by its label plus a random bit, making the data linearly separable
    # and memorizable. Stores them on the test instance. Returns nothing.
    super(EvaluateOnceTest, self).setUp()

    # Create an easy training set:
    np.random.seed(0)

    self._inputs = np.zeros((16, 4))
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

    for i in range(16):
      j = int(2 * self._labels[i] + np.random.randint(0, 2))
      self._inputs[i, j] = 1

  def _train_model(self, checkpoint_dir, num_steps):
    # [FUNCTION DOC]
    # HELPER: produces checkpoints for the tests.
    # Inputs: checkpoint_dir (directory to write checkpoints into) and num_steps (number of
    # training steps). In a fresh graph with a fixed seed, trains the logistic classifier with
    # gradient descent (learning rate 1.0) on log loss in a monitored training session stopped
    # after num_steps. If num_steps is at least 300 it asserts the final loss is below 0.015.
    # Side effects: writes checkpoint files to the directory. Returns nothing.
    """Trains a simple classification model.

    Note that the data has been configured such that after around 300 steps,
    the model has memorized the dataset (e.g. we can expect %100 accuracy).

    Args:
      checkpoint_dir: The directory where the checkpoint is written to.
      num_steps: The number of steps to train for.
    """
    with ops.Graph().as_default():
      random_seed.set_random_seed(0)
      tf_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
      tf_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

      tf_predictions = logistic_classifier(tf_inputs)
      loss_op = losses.log_loss(labels=tf_labels, predictions=tf_predictions)

      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1.0)
      train_op = optimizer.minimize(loss_op,
                                    training.get_or_create_global_step())

      with monitored_session.MonitoredTrainingSession(
          checkpoint_dir=checkpoint_dir,
          hooks=[basic_session_run_hooks.StopAtStepHook(num_steps)]) as session:
        loss = None
        while not session.should_stop():
          _, loss = session.run([train_op, loss_op])

        if num_steps >= 300:
          assert loss < .015

  def testEvaluatePerfectModel(self):
    # [FUNCTION DOC]
    # Scenario: train to completion (300 steps), then evaluate accuracy from the latest
    # checkpoint with a stop-after-one-evaluation hook, an accuracy update as the eval op and the
    # accuracy as a final op. Asserts the final accuracy exceeds 0.99.
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluate_perfect_model_once')

    # Train a Model to completion:
    self._train_model(checkpoint_dir, num_steps=300)

    # Run
    inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    labels = constant_op.constant(self._labels, dtype=dtypes.float32)
    logits = logistic_classifier(inputs)
    predictions = math_ops.round(logits)

    accuracy, update_op = metrics_module.accuracy(labels, predictions)

    checkpoint_path = saver.latest_checkpoint(checkpoint_dir)

    final_ops_values = evaluation._evaluate_once(
        checkpoint_path=checkpoint_path,
        eval_ops=update_op,
        final_ops={'accuracy': (accuracy, update_op)},
        hooks=[
            evaluation._StopAfterNEvalsHook(1),
        ])
    self.assertGreater(final_ops_values['accuracy'], .99)

  def testEvaluateWithFiniteInputs(self):
    # [FUNCTION DOC]
    # Scenario: train to completion, then evaluate with inputs from a one-epoch input producer
    # batched by 6 (smaller final batch allowed) and a stop-after hook with an unlimited count,
    # so evaluation ends when the input runs out. Final ops are accuracy and the eval step
    # counter. Asserts accuracy exceeds 0.99 and the eval step equals 4 (two full batches, one
    # partial batch, and the final run that hits the end of input).
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  'evaluate_with_finite_inputs')

    # Train a Model to completion:
    self._train_model(checkpoint_dir, num_steps=300)

    # Run evaluation. Inputs are fed through input producer for one epoch.
    all_inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    all_labels = constant_op.constant(self._labels, dtype=dtypes.float32)

    single_input, single_label = training.slice_input_producer(
        [all_inputs, all_labels], num_epochs=1)
    inputs, labels = training.batch([single_input, single_label], batch_size=6,
                                    allow_smaller_final_batch=True)

    logits = logistic_classifier(inputs)
    predictions = math_ops.round(logits)

    accuracy, update_op = metrics_module.accuracy(labels, predictions)

    checkpoint_path = saver.latest_checkpoint(checkpoint_dir)

    final_ops_values = evaluation._evaluate_once(
        checkpoint_path=checkpoint_path,
        eval_ops=update_op,
        final_ops={
            'accuracy': (accuracy, update_op),
            'eval_steps': evaluation._get_or_create_eval_step()
        },
        hooks=[
            evaluation._StopAfterNEvalsHook(None),
        ])
    self.assertTrue(final_ops_values['accuracy'] > .99)
    # Runs evaluation for 4 iterations. First 2 evaluate full batch of 6 inputs
    # each; the 3rd iter evaluates the remaining 4 inputs, and the last one
    # triggers an error which stops evaluation.
    self.assertEqual(final_ops_values['eval_steps'], 4)

  def testEvalOpAndFinalOp(self):
    # [FUNCTION DOC]
    # Scenario: after a one-step checkpoint, a local variable is incremented by the eval op for
    # five evaluations (stop-after hook) and a final op reads it plus a constant 9. Asserts the
    # final value equals the evaluation count plus 9, and that the caller's hook list was not
    # modified by the evaluation routine.
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'eval_ops_and_final_ops')

    # Train a model for a single step to get a checkpoint.
    self._train_model(checkpoint_dir, num_steps=1)
    checkpoint_path = saver.latest_checkpoint(checkpoint_dir)

    # Create the model so we have something to restore.
    inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    logistic_classifier(inputs)

    num_evals = 5
    final_increment = 9.0

    my_var = local_variable(0.0, name='MyVar')
    eval_ops = state_ops.assign_add(my_var, 1.0)
    final_ops = array_ops.identity(my_var) + final_increment

    final_hooks = [evaluation._StopAfterNEvalsHook(num_evals),]
    initial_hooks = list(final_hooks)
    final_ops_values = evaluation._evaluate_once(
        checkpoint_path=checkpoint_path,
        eval_ops=eval_ops,
        final_ops={'value': final_ops},
        hooks=final_hooks)
    self.assertEqual(final_ops_values['value'], num_evals + final_increment)
    self.assertEqual(initial_hooks, final_hooks)

  def testMultiEvalStepIncrements(self):
    # [FUNCTION DOC]
    # Scenario: the eval ops list both increments a local variable and increments the eval step
    # itself, so each run advances the eval step by two (once by the user op, once by the
    # routine). With a stop-after hook of 6 evaluations, asserts the variable was incremented
    # only 3 times (half of the requested count).
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'eval_ops_and_final_ops')

    # Train a model for a single step to get a checkpoint.
    self._train_model(checkpoint_dir, num_steps=1)
    checkpoint_path = saver.latest_checkpoint(checkpoint_dir)

    # Create the model so we have something to restore.
    inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    logistic_classifier(inputs)

    num_evals = 6

    my_var = local_variable(0.0, name='MyVar')
    # In eval ops, we also increase the eval step one more time.
    eval_ops = [state_ops.assign_add(my_var, 1.0),
                state_ops.assign_add(
                    evaluation._get_or_create_eval_step(), 1, use_locking=True)]
    expect_eval_update_counts = num_evals // 2

    final_ops = array_ops.identity(my_var)

    final_ops_values = evaluation._evaluate_once(
        checkpoint_path=checkpoint_path,
        eval_ops=eval_ops,
        final_ops={'value': final_ops},
        hooks=[evaluation._StopAfterNEvalsHook(num_evals),])
    self.assertEqual(final_ops_values['value'], expect_eval_update_counts)

  def testOnlyFinalOp(self):
    # [FUNCTION DOC]
    # Scenario: evaluate from a one-step checkpoint with no eval ops, only a final op that
    # reads a local variable (initial 0) plus 9. Asserts the final value equals 9.
    checkpoint_dir = os.path.join(self.get_temp_dir(), 'only_final_ops')

    # Train a model for a single step to get a checkpoint.
    self._train_model(checkpoint_dir, num_steps=1)
    checkpoint_path = saver.latest_checkpoint(checkpoint_dir)

    # Create the model so we have something to restore.
    inputs = constant_op.constant(self._inputs, dtype=dtypes.float32)
    logistic_classifier(inputs)

    final_increment = 9.0

    my_var = local_variable(0.0, name='MyVar')
    final_ops = array_ops.identity(my_var) + final_increment

    final_ops_values = evaluation._evaluate_once(
        checkpoint_path=checkpoint_path, final_ops={'value': final_ops})
    self.assertEqual(final_ops_values['value'], final_increment)


if __name__ == '__main__':
  test.main()
