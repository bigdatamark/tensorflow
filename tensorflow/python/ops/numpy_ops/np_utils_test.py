# [MODULE DOC]
# Unit tests for the np_utils helper module of the tf-numpy ops. Covers the np_doc decorator:
# docstring generation in inlined, linked (dev, stable, version) and invalid-form modes, name
# only usage, signature mismatch handling (error and non-error modes), and dtype extraction from
# tensor-like objects. Runs under the TensorFlow test runner when executed as a script.
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
"""Tests for utils.py."""

from absl.testing import parameterized

from tensorflow.python.framework import dtypes
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.platform import test


class UtilsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for np_utils. Shared fixture: setUp remembers the global numpy doc
  # form and signature-mismatch strictness, and tearDown restores them so tests that change the
  # globals do not leak into each other.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the base setUp, then saves the current numpy doc form and the current
    # signature-mismatch-is-error flag on the instance for restoration in tearDown.
    super(UtilsTest, self).setUp()
    self._old_np_doc_form = np_utils.get_np_doc_form()
    self._old_is_sig_mismatch_an_error = np_utils.is_sig_mismatch_an_error()

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture: restores the saved doc form and signature-mismatch flag in np_utils, then calls
    # the base tearDown.
    np_utils.set_np_doc_form(self._old_np_doc_form)
    np_utils.set_is_sig_mismatch_an_error(self._old_is_sig_mismatch_an_error)
    super(UtilsTest, self).tearDown()

  # pylint: disable=unused-argument
  def testNpDocInlined(self):
    # [FUNCTION DOC]
    # Scenario: doc form is set to inlined and a function taking x and z is decorated with np_doc
    # against a numpy-like function taking x, y, z, with x declared unsupported. Asserts that the
    # docstring is the header, the unsupported list naming x and y (y added automatically because
    # it is missing from the function), the function's own docstring, and the inlined numpy
    # docstring.
    def np_fun(x, y, z):
      # [FUNCTION DOC]
      # Stand-in numpy function with three parameters; does nothing and returns nothing.
      """np_fun docstring."""
      return
    np_utils.set_np_doc_form('inlined')
    @np_utils.np_doc(None, np_fun=np_fun, unsupported_params=['x'])
    def f(x, z):
      # [FUNCTION DOC]
      # Function under decoration; does nothing and returns nothing. Only its docstring matters.
      """f docstring."""
      return
    expected = """TensorFlow variant of NumPy's `np_fun`.

Unsupported arguments: `x`, `y`.

f docstring.

Documentation for `numpy.np_fun`:

np_fun docstring."""
    self.assertEqual(expected, f.__doc__)

  @parameterized.named_parameters([
      (version, version, link) for version, link in  # pylint: disable=g-complex-comprehension
      [('dev',
        'https://numpy.org/devdocs/reference/generated/numpy.np_fun.html'),
       ('stable',
        'https://numpy.org/doc/stable/reference/generated/numpy.np_fun.html'),
       ('1.16',
        'https://numpy.org/doc/1.16/reference/generated/numpy.np_fun.html')
      ]])
  def testNpDocLink(self, version, link):
    # [FUNCTION DOC]
    # Parameterized scenario over doc forms dev, stable and 1.16 with the expected numpy page
    # URL for each. Decorates a two-parameter function against a three-parameter stand-in with x
    # unsupported. Asserts the docstring has the header, the unsupported list naming x and y, the
    # function docstring and a sentence linking to the expected URL.
    def np_fun(x, y, z):
      # [FUNCTION DOC]
      # Stand-in numpy function with three parameters; does nothing and returns nothing.
      """np_fun docstring."""
      return
    np_utils.set_np_doc_form(version)
    @np_utils.np_doc(None, np_fun=np_fun, unsupported_params=['x'])
    def f(x, z):
      # [FUNCTION DOC]
      # Function under decoration; does nothing and returns nothing. Only its docstring matters.
      """f docstring."""
      return
    expected = """TensorFlow variant of NumPy's `np_fun`.

Unsupported arguments: `x`, `y`.

f docstring.

See the NumPy documentation for [`numpy.np_fun`](%s)."""
    expected = expected % (link)
    self.assertEqual(expected, f.__doc__)

  @parameterized.parameters([None, 1, 'a', '1a', '1.1a', '1.1.1a'])
  def testNpDocInvalid(self, invalid_flag):
    # [FUNCTION DOC]
    # Parameterized scenario with invalid doc forms (nothing, an integer, and malformed version
    # strings). Decorates a function as in the other tests. Asserts that no link or numpy text is
    # appended: the docstring ends after the function's own docstring and a trailing blank line.
    def np_fun(x, y, z):
      # [FUNCTION DOC]
      # Stand-in numpy function with three parameters; does nothing and returns nothing.
      """np_fun docstring."""
      return
    np_utils.set_np_doc_form(invalid_flag)
    @np_utils.np_doc(None, np_fun=np_fun, unsupported_params=['x'])
    def f(x, z):
      # [FUNCTION DOC]
      # Function under decoration; does nothing and returns nothing. Only its docstring matters.
      """f docstring."""
      return
    expected = """TensorFlow variant of NumPy's `np_fun`.

Unsupported arguments: `x`, `y`.

f docstring.

"""
    self.assertEqual(expected, f.__doc__)

  def testNpDocName(self):
    # [FUNCTION DOC]
    # Scenario: inlined doc form and np_doc given only a name (foo) that does not exist in numpy.
    # Asserts that the docstring uses that name in the header, contains the function's own
    # docstring, and has no numpy text since no numpy function was found.
    np_utils.set_np_doc_form('inlined')
    @np_utils.np_doc('foo')
    def f():
      # [FUNCTION DOC]
      # Function under decoration; does nothing and returns nothing. Only its docstring matters.
      """f docstring."""
      return
    expected = """TensorFlow variant of NumPy's `foo`.

f docstring.

"""
    self.assertEqual(expected, f.__doc__)

  def testDtypeOfTensorLikeClass(self):
    # [FUNCTION DOC]
    # Scenario: a minimal object that claims to be tensor-like and exposes a float32 dtype is
    # passed to the dtype extraction helper. Asserts the result equals the float32 dtype.

    class TensorLike:
      # [CLASS DOC]
      # Minimal fake tensor: stores a dtype, reports itself as tensor-like and exposes the dtype.

      def __init__(self, dtype):
        # [FUNCTION DOC]
        # Stores the given dtype on the instance. Returns nothing.
        self._dtype = dtype

      @property
      def is_tensor_like(self):
        # [FUNCTION DOC]
        # Marker property recognized by TensorFlow type checks; always returns true.
        return True

      @property
      def dtype(self):
        # [FUNCTION DOC]
        # Returns the dtype given at construction.
        return self._dtype

    t = TensorLike(dtypes.float32)
    self.assertEqual(np_utils._maybe_get_dtype(t), dtypes.float32)

  # pylint: disable=unused-variable
  def testSigMismatchIsError(self):
    # [FUNCTION DOC]
    # Scenario: strict mode is enabled (skipped if signature support is missing). Three functions
    # are decorated against a numpy-like function taking x, y with default, and extra keywords.
    # Asserts a type error mentioning an unfound parameter (extra parameter a), one mentioning a
    # differing kind (a plain parameter named like the keywords catch-all), and one saying
    # parameter y should have a default value (missing default).
    """Tests that signature mismatch is an error (when configured so)."""
    if not np_utils._supports_signature():
      self.skipTest('inspect.signature not supported')

    np_utils.set_is_sig_mismatch_an_error(True)

    def np_fun(x, y=1, **kwargs):
      # [FUNCTION DOC]
      # Stand-in numpy function used as the reference signature; does nothing, returns nothing.
      return

    with self.assertRaisesRegex(TypeError, 'Cannot find parameter'):
      @np_utils.np_doc(None, np_fun=np_fun)
      def f1(a):
        # [FUNCTION DOC]
        # Function with a parameter absent from the reference signature; never called.
        return

    with self.assertRaisesRegex(TypeError, 'is of kind'):
      @np_utils.np_doc(None, np_fun=np_fun)
      def f2(x, kwargs):
        # [FUNCTION DOC]
        # Function whose second parameter has a different kind than the reference; never called.
        return

    with self.assertRaisesRegex(
        TypeError, 'Parameter y should have a default value'):
      @np_utils.np_doc(None, np_fun=np_fun)
      def f3(x, y):
        # [FUNCTION DOC]
        # Function whose y lacks the default the reference has; never called.
        return

  def testSigMismatchIsNotError(self):
    # [FUNCTION DOC]
    # Scenario: strict mode is disabled. Decorates functions whose signatures differ from the
    # reference (extra parameter, missing default). Asserts nothing explicitly; the test passes
    # if no error is raised during decoration.
    """Tests that signature mismatch is not an error (when configured so)."""
    np_utils.set_is_sig_mismatch_an_error(False)

    def np_fun(x, y=1, **kwargs):
      # [FUNCTION DOC]
      # Stand-in numpy function used as the reference signature; does nothing, returns nothing.
      return

    # The following functions all have signature mismatches, but they shouldn't
    # throw errors when is_sig_mismatch_an_error() is False.

    @np_utils.np_doc(None, np_fun=np_fun)
    def f1(a):
      # [FUNCTION DOC]
      # Function with a parameter absent from the reference signature; never called.
      return

    def f2(x, kwargs):
      # [FUNCTION DOC]
      # KNOWN ISSUE: unlike f1 and f3 this function is not decorated, so it exercises nothing.
      # Plain function with mismatching kind; never called.
      return

    @np_utils.np_doc(None, np_fun=np_fun)
    def f3(x, y):
      # [FUNCTION DOC]
      # Function whose y lacks the default the reference has; never called.
      return

  # pylint: enable=unused-variable


if __name__ == '__main__':
  test.main()
