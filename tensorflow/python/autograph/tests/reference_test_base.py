# [MODULE DOC]
# HELPER: base-class module shared by the AutoGraph reference tests in this directory.
# Provides TestCase, a TensorFlow test case with utilities to run a function natively (eager
# Python) and compiled (tf.function or to_graph conversion), capturing printed output, returned
# results and raised exceptions, then compare them. Subclass tests call the assert helpers.
# Import-time side effects: none; setUp sets the AutoGraph strict-conversion environment
# variable for the process. Runs the TF test main when executed directly.
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
"""Reference tests check that a function is compiled correctly."""

import io
import numbers
import os
import sys
import traceback

import numpy as np
import tensorflow as tf


class TestCase(tf.test.TestCase):
  # [CLASS DOC]
  # HELPER: base class for the reference tests. Instance attributes set in setUp: autograph_opts
  # (optional-feature settings passed to compilation, default nothing), all_inputs_tensors (when
  # true, inputs are converted to tensors before compiled runs, default false) and
  # allow_exceptions (when false, an error from the compiled run fails the test, default false).
  """Base class for the reference tests."""

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setUp, sets the process environment variable that makes AutoGraph conversion
    # strict, and resets the three configuration attributes to their defaults.
    # Side effect: modifies the process environment.
    super(TestCase, self).setUp()
    os.environ['AUTOGRAPH_STRICT_CONVERSION'] = '1'
    self.autograph_opts = None
    self.all_inputs_tensors = False
    self.allow_exceptions = False

  # TODO(mdan): Consider rewriting as a context manager.
  def _run_with_output_capture(self, func):
    # [FUNCTION DOC]
    # HELPER: runs the callable func (no arguments) with standard output redirected to an in-memory
    # buffer. Returns a triple of the result, the captured output text and the captured exception;
    # on success the exception is nothing, on failure the result and output are nothing and the
    # traceback is printed to the real standard output. Always restores standard output.
    """Executes `func`, capturing stdout."""
    out_capturer = io.StringIO()
    results = None
    captured_out = None
    captured_err = None
    try:
      sys.stdout = out_capturer
      results = func()
      captured_out = out_capturer.getvalue()
    except Exception as e:  # pylint:disable=broad-except
      sys.stdout = sys.__stdout__
      captured_err = e
      print('*** Capturing exception:\n{}\n'.format(traceback.format_exc()))
    finally:
      sys.stdout = sys.__stdout__
      out_capturer.close()
    return results, captured_out, captured_err

  def _as_tensors(self, args):
    # [FUNCTION DOC]
    # HELPER: converts a sequence of arguments to a list in which numbers, lists and numpy arrays
    # become constant tensors, dictionaries are converted value-by-value recursively, and anything
    # else is passed through unchanged. Returns the new list.
    """Converts args to tensors."""
    tensor_args = []
    for a in args:
      if isinstance(a, (numbers.Number, list, np.ndarray)):
        tensor_arg = tf.constant(a)
      elif isinstance(a, dict):
        keys = tuple(a.keys())
        tensor_arg = dict(zip(keys, self._as_tensors([a[k] for k in keys])))
      else:
        tensor_arg = a
      tensor_args.append(tensor_arg)
    return tensor_args

  def run_native(self, f, *args):
    # [FUNCTION DOC]
    # Calls f with args under output capture. Despite the name it is also used for compiled
    # functions. Returns the triple of result, captured output and captured exception.
    return self._run_with_output_capture(lambda: f(*args))

  def _deep_equal(self, left, right):
    # [FUNCTION DOC]
    # HELPER: structural equality check. Tensors are compared via their numpy values; sparse
    # tensors compare indices, values and shape; arrays use array equality; lists and tuples are
    # compared element-wise (KNOWN ISSUE: zip truncates, so sequences of different lengths that
    # share a prefix compare equal); everything else uses ordinary equality. Returns a boolean.
    """Compares two possibly-nested structures."""
    if isinstance(left, tf.Tensor):
      return self._deep_equal(left.numpy(), right)
    if isinstance(right, tf.Tensor):
      return self._deep_equal(left, right.numpy())
    if isinstance(left, tf.SparseTensor) and isinstance(right, tf.SparseTensor):
      return (self._deep_equal(left.indices, right.indices)
              and self._deep_equal(left.values, right.values)
              and self._deep_equal(left.shape, right.shape))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
      return np.array_equal(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
      return all(self._deep_equal(l, r) for l, r in zip(left, right))
    return left == right

  def assertResultsMatch(self,
                         f,
                         args,
                         native_data,
                         compiled_data):
    # [FUNCTION DOC]
    # Asserts that two result triples (result, printed output, exception) are equivalent. Both
    # outputs and results must be deeply equal, and the exception types must be identical (their
    # messages are not compared). On mismatch fails the test with a message showing both sides
    # and the call. Parameters: f (the function, used for its name), args (its arguments),
    # native_data and compiled_data (triples). Returns nothing.
    """Asserts that native_data matches compiled_data."""
    native_results, native_out, native_err = native_data
    compiled_results, compiled_out, compiled_err = compiled_data
    str_args = '(%s)' % ', '.join(str(a) for a in args)
    # Using a manual verification to avoid a second compilation on success.
    # For exceptions, we don't enforce that they are the same, only that
    # both paths raised.
    # TODO(mdan): Add an API that returns both object and source code instead.
    outputs_equal = (
        self._deep_equal(native_results, compiled_results) and
        native_out == compiled_out)
    errors_equivalent = type(native_err) == type(compiled_err)  # pylint:disable=unidiomatic-typecheck
    if (not outputs_equal or not errors_equivalent):
      self.fail('Native and compiled functions are not equivalent.\n\n'
                'Native results: %s\n'
                'Compiled results: %s\n'
                'Native out: %s\n'
                'Compiled out: %s\n'
                'Native error: %s: %s\n'
                'Compiled error: %s: %s\n'
                'Native call: %s%s\n'
                'Check the logs for the generated code.'
                '' % (
                    native_results,
                    compiled_results,
                    native_out,
                    compiled_out,
                    type(native_err).__name__,
                    native_err,
                    type(compiled_err).__name__,
                    compiled_err,
                    f.__name__,
                    str_args,
                ))

  def function(self, f, xla=False):
    # [FUNCTION DOC]
    # Wraps f with tf.function using the test's autograph options; xla (default false) turns on
    # experimental compilation. Returns the wrapped callable.
    return tf.function(
        f,
        experimental_autograph_options=self.autograph_opts,
        experimental_compile=xla)

  def convert(self, f):
    # [FUNCTION DOC]
    # Converts f with AutoGraph's to_graph conversion using the test's autograph options.
    # Returns the converted function.
    return tf.autograph.to_graph(
        f, experimental_optional_features=self.autograph_opts)

  def assertFunctionMatchesEagerStatefulInput(self, f, args):
    # [FUNCTION DOC]
    # Like the stateless variant but args is a callable that creates fresh inputs; it is called
    # separately for the compiled run, the native run and the comparison. Runs compiled and
    # native versions and asserts results match.
    """Like assertFunctionMatchesEager but creates new inputs each time."""
    compiled_data = self.run_native(self.function(f), *args())
    native_data = self.run_native(f, *args())
    self.assertResultsMatch(f, args(), native_data, compiled_data)

  def assertFunctionMatchesEager(self, f, *args, xla=False):
    # [FUNCTION DOC]
    # Runs f compiled with tf.function (xla optional, default false) and natively with the same
    # positional args, converting args to tensors first if all_inputs_tensors is set. Unless
    # allow_exceptions is set, a compiled-run error fails the test with its message. Then asserts
    # the results, printed output and exception types match.
    if self.all_inputs_tensors:
      args = self._as_tensors(args)
    compiled_data = self.run_native(self.function(f, xla=xla), *args)
    if not self.allow_exceptions:
      _, _, compiled_err = compiled_data
      if compiled_err is not None:
        self.fail(str(compiled_err))
    native_data = self.run_native(f, *args)
    self.assertResultsMatch(f, args, native_data, compiled_data)

  def assertConvertedMatchesNative(self, f, *args):
    # [FUNCTION DOC]
    # Runs f after AutoGraph conversion (to_graph) and natively with the same args and asserts the
    # results, output and exception types match.
    compiled_data = self.run_native(self.convert(f), *args)
    native_data = self.run_native(f, *args)
    self.assertResultsMatch(f, args, native_data, compiled_data)


if __name__ == '__main__':
  tf.test.main()
