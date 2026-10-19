# [MODULE DOC]
# Unit tests for the TFLite authoring compatibility decorator. Each test decorates a small tf
# function, calls it, and then inspects the recorded compatibility messages (Select TF ops
# warnings, custom op errors, empty logs, raised exceptions) or decorator behavior (method
# decoration, name preservation, single check). Requires the full TensorFlow package and the
# TFLite converter. Runs the test main when executed as a script.
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
"""Unit tests for authoring package."""
# pylint: disable=g-direct-tensorflow-import

import tensorflow as tf

from tensorflow.lite.python.authoring import authoring


class TFLiteAuthoringTest(tf.test.TestCase):
  # [CLASS DOC]
  # Test case covering the authoring compatibility decorator: warnings for Select TF ops, errors
  # for custom ops, converter option pass-through, exception raising, method decoration and
  # single-run verification. No shared fixtures; each test builds its own decorated function.

  def test_simple_cosh(self):
    # [FUNCTION DOC]
    # Scenario: a decorated function computing hyperbolic cosine of a float vector. Calls it with
    # a zero vector. Expects the result to equal one and the log to contain the Select TF ops
    # warning for the cosh op, and the last log entry to mention this test file name.
    @authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def f(x):
      # [FUNCTION DOC]
      # Test helper: returns hyperbolic cosine of the input tensor.
      return tf.cosh(x)

    result = f(tf.constant([0.0]))
    log_messages = f.get_compatibility_log()
    self.assertEqual(result, tf.constant([1.0]))
    self.assertIn(
        "COMPATIBILITY WARNING: op 'tf.Cosh' require(s) \"Select TF Ops\" for "
        "model conversion for TensorFlow Lite. "
        "https://www.tensorflow.org/lite/guide/ops_select", log_messages)

    # Check the op location ends with filename of the this test.
    self.assertIn("authoring_test.py", log_messages[-1])

  def test_simple_cosh_raises_CompatibilityError(self):
    # [FUNCTION DOC]
    # Scenario: the decorator is created with raising enabled and wraps a cosh function. Calling
    # it must raise the compatibility error; afterwards the log still contains the Select TF ops
    # warning for cosh.
    @authoring.compatible(raise_exception=True)
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def f(x):
      # [FUNCTION DOC]
      # Test helper: returns hyperbolic cosine of the input tensor.
      return tf.cosh(x)

    # Check if the CompatibilityError exception is raised.
    with self.assertRaises(authoring.CompatibilityError):
      result = f(tf.constant([0.0]))
      del result
    log_messages = f.get_compatibility_log()
    self.assertIn(
        "COMPATIBILITY WARNING: op 'tf.Cosh' require(s) \"Select TF Ops\" for "
        "model conversion for TensorFlow Lite. "
        "https://www.tensorflow.org/lite/guide/ops_select", log_messages)

  def test_flex_compatibility(self):
    # [FUNCTION DOC]
    # Scenario: a function applying tanh, a 3-D convolution with all-ones filter, erf and tanh to a
    # five-dimensional input. Calls it with ones. Expects the log to contain the Select TF ops
    # warning naming the erf op.
    @authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[3, 3, 3, 3, 3], dtype=tf.float32)
    ])
    def f(inp):
      # [FUNCTION DOC]
      # Test helper: chain of tanh, 3-D convolution with ones filter (same padding), erf, tanh;
      # returns the final tensor.
      tanh = tf.math.tanh(inp)
      conv3d = tf.nn.conv3d(
          tanh,
          tf.ones([3, 3, 3, 3, 3]),
          strides=[1, 1, 1, 1, 1],
          padding="SAME")
      erf = tf.math.erf(conv3d)
      output = tf.math.tanh(erf)
      return output

    f(tf.ones(shape=(3, 3, 3, 3, 3), dtype=tf.float32))
    log_messages = f.get_compatibility_log()
    self.assertIn(
        "COMPATIBILITY WARNING: op 'tf.Erf' require(s) \"Select TF Ops\" for "
        "model conversion for TensorFlow Lite. "
        "https://www.tensorflow.org/lite/guide/ops_select", log_messages)

  def test_compatibility_error_generic(self):
    # [FUNCTION DOC]
    # Scenario: a function building a shuffled range dataset, decorated with default settings.
    # Calls it. Expects an error message saying the dataset ops (seed generator, range, shuffle)
    # are not natively supported and need a custom operator.
    @authoring.compatible
    @tf.function
    def f():
      # [FUNCTION DOC]
      # Test helper: builds and returns a range dataset of three items, shuffled with reshuffling.
      dataset = tf.data.Dataset.range(3)
      dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
      return dataset

    f()
    log_messages = f.get_compatibility_log()
    self.assertIn(
        "COMPATIBILITY ERROR: op 'tf.DummySeedGenerator, tf.RangeDataset,"
        " tf.ShuffleDatasetV3' is(are) not natively supported by TensorFlow"
        " Lite. You need to provide a custom operator."
        " https://www.tensorflow.org/lite/guide/ops_custom",
        log_messages,
    )

  def test_compatibility_error_custom(self):
    # [FUNCTION DOC]
    # Scenario: same shuffled dataset function, but with a target spec allowing both builtin and
    # Select TF ops. Expects the custom operator error message listing the dataset ops anyway.
    target_spec = tf.lite.TargetSpec()
    target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    @authoring.compatible(converter_target_spec=target_spec)
    @tf.function
    def f():
      # [FUNCTION DOC]
      # Test helper: builds and returns a range dataset of three items, shuffled with reshuffling.
      dataset = tf.data.Dataset.range(3)
      dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
      return dataset

    f()
    log_messages = f.get_compatibility_log()
    self.assertIn(
        "COMPATIBILITY ERROR: op 'tf.DummySeedGenerator, tf.RangeDataset, "
        "tf.ShuffleDatasetV3' is(are) not natively supported by "
        "TensorFlow Lite. You need to provide a custom operator. "
        "https://www.tensorflow.org/lite/guide/ops_custom", log_messages)

  def test_simple_variable(self):
    # [FUNCTION DOC]
    # Scenario: a decorated function multiplying its input by an external variable. Calls it with
    # a value of two. Expects the result to be two and the compatibility log to be empty.
    external_var = tf.Variable(1.0)
    @authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def f(x):
      # [FUNCTION DOC]
      # Test helper: multiplies the input by the captured external variable.
      return x * external_var

    result = f(tf.constant(2.0, shape=(1)))
    log_messages = f.get_compatibility_log()

    self.assertEqual(result, tf.constant([2.0]))
    self.assertEmpty(log_messages)

  def test_class_method(self):
    # [FUNCTION DOC]
    # Scenario: the decorator applied to a method of a module class, computing cosh. Calls the
    # method on an instance. Expects the result to equal one and the method's log to contain the
    # Select TF ops warning for cosh.
    class Model(tf.Module):
      # [CLASS DOC]
      # Test helper module with one decorated method.

      @authoring.compatible
      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.float32)
      ])
      def eval(self, x):
        # [FUNCTION DOC]
        # Test helper method: returns hyperbolic cosine of the input.
        return tf.cosh(x)

    m = Model()
    result = m.eval(tf.constant([0.0]))
    log_messages = m.eval.get_compatibility_log()

    self.assertEqual(result, tf.constant([1.0]))
    self.assertIn(
        "COMPATIBILITY WARNING: op 'tf.Cosh' require(s) \"Select TF Ops\" for "
        "model conversion for TensorFlow Lite. "
        "https://www.tensorflow.org/lite/guide/ops_select", log_messages)

  def test_decorated_function_type(self):
    # [FUNCTION DOC]
    # Scenario: decorated cosine function. Expects the correct result, the original function name
    # to be preserved, and the decorated object's concrete function to be usable to create a
    # TFLite converter and convert successfully.
    @authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def func(x):
      # [FUNCTION DOC]
      # Test helper: returns cosine of the input tensor.
      return tf.cos(x)

    result = func(tf.constant([0.0]))
    self.assertEqual(result, tf.constant([1.0]))

    # Check if the decorator keeps __name__ attribute.
    self.assertEqual(func.__name__, "func")

    # Check if the decorator works with get_concrete_function method.
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [func.get_concrete_function()], func)
    converter.convert()

  def test_decorated_class_method_type(self):
    # [FUNCTION DOC]
    # Scenario: decorated cosine method of a module class. Expects the correct result, the method
    # name preserved, and conversion through a TFLite converter built from the method's concrete
    # function and the module to succeed.
    class Model(tf.Module):
      # [CLASS DOC]
      # Test helper module with one decorated method.

      @authoring.compatible
      @tf.function(input_signature=[
          tf.TensorSpec(shape=[None], dtype=tf.float32)
      ])
      def eval(self, x):
        # [FUNCTION DOC]
        # Test helper method: returns cosine of the input.
        return tf.cos(x)

    m = Model()
    result = m.eval(tf.constant([0.0]))
    self.assertEqual(result, tf.constant([1.0]))

    # Check if the decorator keeps __name__ attribute.
    self.assertEqual(m.eval.__name__, "eval")

    # Check if the decorator works with get_concrete_function method.
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [m.eval.get_concrete_function()], m)
    converter.convert()

  def test_simple_cosh_multiple(self):
    # [FUNCTION DOC]
    # Scenario: a decorated cosh function called three times. Expects exactly two log messages
    # (summary plus op location detail), showing the check ran only once.
    @authoring.compatible
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float32)
    ])
    def f(x):
      # [FUNCTION DOC]
      # Test helper: returns hyperbolic cosine of the input tensor.
      return tf.cosh(x)

    f(tf.constant([1.0]))
    f(tf.constant([2.0]))
    f(tf.constant([3.0]))
    warning_messages = f.get_compatibility_log()

    # Test if compatiblility checks happens only once.
    # The number of warning_messages will be 2 by op location detail.
    self.assertEqual(2, len(warning_messages))

  def test_user_tf_ops_all_filtered(self):
    # [FUNCTION DOC]
    # Scenario: target spec allowing Select TF ops and declaring all three dataset ops as user TF
    # ops. Calls the shuffled dataset function. Expects an empty log.
    target_spec = tf.lite.TargetSpec()
    target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    target_spec.experimental_select_user_tf_ops = [
        "RangeDataset", "DummySeedGenerator", "ShuffleDatasetV3"
    ]
    @authoring.compatible(converter_target_spec=target_spec)
    @tf.function
    def f():
      # [FUNCTION DOC]
      # Test helper: builds and returns a range dataset of three items, shuffled with reshuffling.
      dataset = tf.data.Dataset.range(3)
      dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
      return dataset

    f()
    log_messages = f.get_compatibility_log()
    self.assertEmpty(log_messages)

  def test_user_tf_ops_partial_filtered(self):
    # [FUNCTION DOC]
    # Scenario: only the seed generator op is declared as a user TF op. Calls the shuffled dataset
    # function. Expects a custom operator error listing only the range and shuffle dataset ops.
    target_spec = tf.lite.TargetSpec()
    target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    target_spec.experimental_select_user_tf_ops = [
        "DummySeedGenerator"
    ]
    @authoring.compatible(converter_target_spec=target_spec)
    @tf.function
    def f():
      # [FUNCTION DOC]
      # Test helper: builds and returns a range dataset of three items, shuffled with reshuffling.
      dataset = tf.data.Dataset.range(3)
      dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
      return dataset

    f()
    log_messages = f.get_compatibility_log()
    self.assertIn(
        "COMPATIBILITY ERROR: op 'tf.RangeDataset, tf.ShuffleDatasetV3' is(are) "
        "not natively supported by TensorFlow Lite. You need to provide a "
        "custom operator. https://www.tensorflow.org/lite/guide/ops_custom",
        log_messages)

  def test_allow_custom_ops(self):
    # [FUNCTION DOC]
    # Scenario: decorator configured to allow custom ops plus builtin and Select TF ops. Calls the
    # shuffled dataset function. Expects an empty log.
    target_spec = tf.lite.TargetSpec()
    target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    @authoring.compatible(
        converter_allow_custom_ops=True, converter_target_spec=target_spec)
    @tf.function
    def f():
      # [FUNCTION DOC]
      # Test helper: builds and returns a range dataset of three items, shuffled with reshuffling.
      dataset = tf.data.Dataset.range(3)
      dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
      return dataset

    f()
    log_messages = f.get_compatibility_log()
    self.assertEmpty(log_messages)

  def test_gpu_compatible(self):
    # [FUNCTION DOC]
    # Scenario: builtin-only target spec with the GPU backend requested, wrapping a cosine
    # function over a four by four input. Calls it with ones. Expects an empty log, meaning no GPU
    # incompatibility is reported.
    target_spec = tf.lite.TargetSpec()
    target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    target_spec.experimental_supported_backends = ["GPU"]

    @authoring.compatible(converter_target_spec=target_spec)
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[4, 4], dtype=tf.float32)])
    def func(x):
      # [FUNCTION DOC]
      # Test helper: returns cosine of the input tensor.
      return tf.cos(x)

    func(tf.ones(shape=(4, 4), dtype=tf.float32))
    log_messages = func.get_compatibility_log()
    self.assertEmpty(log_messages)


if __name__ == "__main__":
  tf.test.main()
