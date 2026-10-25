# [MODULE DOC]
# Library behind the TensorFlow doctest runners. Provides a doctest output checker that compares
# the expected and actual output of a docstring example leniently: memory addresses and tensor
# wrappers are normalized, and floating point numbers are pulled out of the text and compared
# numerically with a tolerance instead of as text. Also contains a private helper class that splits
# a string into its non-number text parts and the array of numbers. Used by the doctest runner and
# by the fenced cell doctest library. No import-time side effects beyond compiling regular
# expressions.
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
"""Run doctests for tensorflow."""

import doctest
import re
import textwrap

import numpy as np


class _FloatExtractor(object):
  # [CLASS DOC]
  # HELPER: float extraction helper for the doctest output checker. A callable object holding one
  # compiled verbose regular expression that recognizes decimal numbers (with optional sign,
  # fraction, exponent or imaginary suffix) that stand alone, i.e. are not part of a word or
  # preceded by a period. Calling it on a string splits the text around the numbers.
  # KNOWN ISSUE: the docstring of the call method says the result array has 32-bit float type, but
  # the code builds the array from python floats, giving the default 64-bit type.
  """Class for extracting floats from a string.

  For example:

  >>> text_parts, floats = _FloatExtractor()("Text 1.0 Text")
  >>> text_parts
  ["Text ", " Text"]
  >>> floats
  np.array([1.0])
  """

  # Note: non-capturing groups "(?" are not returned in matched groups, or by
  # re.split.
  _FLOAT_RE = re.compile(
      r"""
      (                          # Captures the float value.
        (?:
           [-+]|                 # Start with a sign is okay anywhere.
           (?:                   # Otherwise:
               ^|                # Start after the start of string
               (?<=[^\w.])       # Not after a word char, or a .
           )
        )
        (?:                      # Digits and exponent - something like:
          {digits_dot_maybe_digits}{exponent}?|   # "1.0" "1." "1.0e3", "1.e3"
          {dot_digits}{exponent}?|                # ".1" ".1e3"
          {digits}{exponent}|                     # "1e3"
          {digits}(?=j)                           # "300j"
        )
      )
      j?                         # Optional j for cplx numbers, not captured.
      (?=                        # Only accept the match if
        $|                       # * At the end of the string, or
        [^\w.]                   # * Next char is not a word char or "."
      )
      """.format(
          # Digits, a "." and optional more digits: "1.1".
          digits_dot_maybe_digits=r'(?:[0-9]+\.(?:[0-9]*))',
          # A "." with trailing digits ".23"
          dot_digits=r'(?:\.[0-9]+)',
          # digits: "12"
          digits=r'(?:[0-9]+)',
          # The exponent: An "e" or "E", optional sign, and at least one digit.
          # "e-123", "E+12", "e12"
          exponent=r'(?:[eE][-+]?[0-9]+)'),
      re.VERBOSE)

  def __call__(self, string):
    # [FUNCTION DOC]
    # Splits a string into text and numbers. Input: string, text to scan. Splits it with the
    # number pattern (the pattern has one capturing group, so numbers occupy the odd positions),
    # collects the even-position pieces as text parts and converts the odd-position pieces to
    # floats. Returns a pair: the list of text parts (one more than the count of numbers) and a
    # numpy array of the numbers (empty if none). Raises a value error only if a matched piece
    # cannot be converted, e.g. an imaginary-looking number is not stripped of its suffix by the
    # pattern, which excludes it from the capture.
    """Extracts floats from a string.

    >>> text_parts, floats = _FloatExtractor()("Text 1.0 Text")
    >>> text_parts
    ["Text ", " Text"]
    >>> floats
    np.array([1.0])

    Args:
      string: the string to extract floats from.

    Returns:
      A (string, array) pair, where `string` has each float replaced by "..."
      and `array` is a `float32` `numpy.array` containing the extracted floats.
    """
    texts = []
    floats = []
    for i, part in enumerate(self._FLOAT_RE.split(string)):
      if i % 2 == 0:
        texts.append(part)
      else:
        floats.append(float(part))

    return texts, np.array(floats)


class TfDoctestOutputChecker(doctest.OutputChecker, object):
  # [CLASS DOC]
  # Doctest output checker with tolerant comparison, extending the standard doctest checker.
  # State: a float extractor, the outcome flags of the latest check (text comparison good, float
  # count good) and the floats extracted from the last expected and actual output. Class level
  # holds the address pattern, tensor-string pattern and a failure message advising where to read
  # about writing testable docstrings. The check method is called by doctest per example; the
  # difference method appends hints to failure reports.
  # KNOWN ISSUE: the outcome flags and extracted floats are not reset between checks and the float
  # attributes are not initialized in the constructor, so a failure report can use stale values
  # from an earlier example when the check returned early.
  """Customizes how `want` and `got` are compared, see `check_output`."""

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Passes all arguments to the standard checker, creates the float extractor and
    # initializes the text-good and float-size-good flags to null. Returns nothing.
    super(TfDoctestOutputChecker, self).__init__(*args, **kwargs)
    self.extract_floats = _FloatExtractor()
    self.text_good = None
    self.float_size_good = None

  _ADDRESS_RE = re.compile(r'\bat 0x[0-9a-f]*?>')
  # TODO(yashkatariya): Add other tensor's string substitutions too.
  # tf.RaggedTensor doesn't need one.
  _NUMPY_OUTPUT_RE = re.compile(r'<tf.Tensor.*?numpy=(.*?)>', re.DOTALL)

  def _allclose(self, want, got, rtol=1e-3, atol=1e-3):
    # [FUNCTION DOC]
    # HELPER: numeric comparison for the output check. Inputs: want and got, arrays of floats;
    # rtol and atol, relative and absolute tolerances (both default to one thousandth). Returns
    # true if all elements are close within the tolerances (raises a value error if shapes cannot
    # be broadcast).
    return np.allclose(want, got, rtol=rtol, atol=atol)

  def _tf_tensor_numpy_output(self, string):
    # [FUNCTION DOC]
    # HELPER: normalizer for the output check. Input: string. Replaces every tensor repr (text
    # starting with the tensor marker and containing a numpy field) by just the numpy field text,
    # matching across line breaks. Returns a pair: the modified string and a boolean saying whether
    # anything changed.
    modified_string = self._NUMPY_OUTPUT_RE.sub(r'\1', string)
    return modified_string, modified_string != string

  MESSAGE = textwrap.dedent("""\n
        #############################################################
        Check the documentation (https://www.tensorflow.org/community/contribute/docs_ref) on how to
        write testable docstrings.
        #############################################################""")

  def check_output(self, want, got, optionflags):
    # [FUNCTION DOC]
    # Decides whether the actual output of a doctest example matches the documented output.
    # Inputs: want, expected output text (may be null); got, actual output text; optionflags,
    # doctest option flags passed to the base comparison.
    # Order of decisions: true when output was produced but none was documented; true when the texts
    # are identical (null expected is treated as empty). Otherwise addresses in the expected text
    # become ellipses, tensor reprs are reduced to their numpy field (applied to the actual text
    # only if the expected text contained one), floats are extracted from both and the expected text
    # is rewritten with ellipses in place of floats. The base checker must accept the text
    # (otherwise false). If the expected text had no floats, true (floats in actual output are
    # ignored). If the number of floats differs, false. Otherwise returns whether the floats are
    # close within tolerance.
    # Side effects: records the extracted floats and the intermediate flags on the instance.
    """Compares the docstring output to the output gotten by running the code.

    Python addresses in the output are replaced with wildcards.

    Float values in the output compared as using `np.allclose`:

      * Float values are extracted from the text and replaced with wildcards.
      * The wildcard text is compared to the actual output.
      * The float values are compared using `np.allclose`.

    The method returns `True` if both the text comparison and the numeric
    comparison are successful.

    The numeric comparison will fail if either:

      * The wrong number of floats are found.
      * The float values are not within tolerence.

    Args:
      want: The output in the docstring.
      got: The output generated after running the snippet.
      optionflags: Flags passed to the doctest.

    Returns:
      A bool, indicating if the check was successful or not.
    """

    # If the docstring's output is empty and there is some output generated
    # after running the snippet, return True. This is because if the user
    # doesn't want to display output, respect that over what the doctest wants.
    if got and not want:
      return True

    if want is None:
      want = ''

    if want == got:
      return True

    # Replace python's addresses with ellipsis (`...`) since it can change on
    # each execution.
    want = self._ADDRESS_RE.sub('at ...>', want)

    # Replace tf.Tensor strings with only their numpy field values.
    want, want_changed = self._tf_tensor_numpy_output(want)
    if want_changed:
      got, _ = self._tf_tensor_numpy_output(got)

    # Separate out the floats, and replace `want` with the wild-card version
    # "result=7.0" => "result=..."
    want_text_parts, self.want_floats = self.extract_floats(want)
    # numpy sometimes pads floats in arrays with spaces
    # got: [1.2345, 2.3456, 3.0   ]  want:  [1.2345, 2.3456, 3.0001]
    # And "normalize whitespace" only works when there's at least one space,
    # so strip them and let the wildcard handle it.
    want_text_parts = [part.strip(' ') for part in want_text_parts]
    want_text_wild = '...'.join(want_text_parts)
    if '....' in want_text_wild:
      # If a float comes just after a period you'll end up four dots and the
      # first three count as the ellipsis. Replace it with three dots.
      want_text_wild = re.sub(r'\.\.\.\.+', '...', want_text_wild)

    # Find the floats in the string returned by the test
    _, self.got_floats = self.extract_floats(got)

    self.text_good = super(TfDoctestOutputChecker, self).check_output(
        want=want_text_wild, got=got, optionflags=optionflags)
    if not self.text_good:
      return False

    if self.want_floats.size == 0:
      # If there are no floats in the "want" string, ignore all the floats in
      # the result. "np.array([ ... ])" matches "np.array([ 1.0, 2.0 ])"
      return True

    self.float_size_good = (self.want_floats.size == self.got_floats.size)

    if self.float_size_good:
      return self._allclose(self.want_floats, self.got_floats)
    else:
      return False

  def output_difference(self, example, got, optionflags):
    # [FUNCTION DOC]
    # Builds the failure report text. Inputs: example, the doctest example; got, the actual output;
    # optionflags, doctest flags. If the text matched but the number of floats differed, appends a
    # caution that hiding only some float output with an ellipsis is unsupported. Always appends the
    # standard message about writing testable docstrings, joins the parts with line breaks and
    # delegates to the base class to produce the difference text. Returns that string.
    got = [got]

    # If the some of the float output is hidden with `...`, `float_size_good`
    # will be False. This is because the floats extracted from the string is
    # converted into a 1-D numpy array. Hence hidding floats is not allowed
    # anymore.
    if self.text_good:
      if not self.float_size_good:
        got.append("\n\nCAUTION: tf_doctest doesn't work if *some* of the "
                   "*float output* is hidden with a \"...\".")

    got.append(self.MESSAGE)
    got = '\n'.join(got)
    return (super(TfDoctestOutputChecker,
                  self).output_difference(example, got, optionflags))
