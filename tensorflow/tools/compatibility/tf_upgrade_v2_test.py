# [MODULE DOC]
# Unit tests for the TensorFlow 2.0 upgrade script (tf_upgrade_v2). The main test class feeds small
#   pieces of old-style source text through the code upgrader built from the 2.0 change
#   specification and checks the rewritten text, the report and the errors for renames, keyword and
#   positional argument changes, contrib migrations, summary and import handling. Several tests also
#   walk the whole public v1 and v2 API to check that conversion targets and argument names exist. A
#   second test class checks in-place file processing with temporary files. Helper functions at the
#   top resolve dotted API names, list argument names and parse converted call text. Importing the
#   module has no side effects; running it as a script starts the standard test runner.
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
"""Tests for tf 2.0 upgrader."""

import inspect
import io
import os
import tempfile

from absl.testing import parameterized
import tensorflow.compat.v1 as tf
# OSS TF V2 import placeholder.

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect
from tensorflow.tools.common import public_api
from tensorflow.tools.common import traverse
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import tf_upgrade_v2


def get_symbol_for_name(root, name):
  # [FUNCTION DOC]
  # HELPER: lets tests resolve dotted API names to live objects.
  # Resolves a dotted public API name such as the root module name followed by sub-names into the
  #   actual object.
  # Inputs: root is the object to start from (normally the tensorflow module); name is a dotted
  #   string whose first part is skipped because it names the root itself.
  # Returns the object found by walking the remaining parts one attribute at a time. Raises an
  #   attribute error if any part is missing. No side effects.
  name_parts = name.split(".")
  symbol = root
  # Iterate starting with second item since 1st item is "tf.".
  for part in name_parts[1:]:
    symbol = getattr(symbol, part)
  return symbol


def get_args(symbol):
  # [FUNCTION DOC]
  # HELPER: lets tests compare argument names of API functions.
  # KNOWN ISSUE: the fallback branch using the legacy argument-spec helper is dead code on any
  #   Python 3 interpreter, because the signature inspector always exists there.
  # Returns the list of names of positional-or-keyword parameters of a callable, in declaration
  #   order, ignoring variadic positional and keyword parameters.
  # Input: symbol, any callable. May raise a value or type error if the callable has no inspectable
  #   signature. No side effects.
  if hasattr(inspect, "signature"):
    signature = inspect.signature(symbol)
    # Ignore *args and **kwargs for now.
    return [param.name for param in signature.parameters.values()
            if param.kind == param.POSITIONAL_OR_KEYWORD]
  return tf_inspect.getargspec(symbol)[0]


def get_func_and_args_from_str(call_str):
  # [FUNCTION DOC]
  # HELPER: lets tests inspect the output of the upgrader for a converted call.
  # KNOWN ISSUE: the split is naive; it breaks on commas inside nested calls and, if the text has no
  #   parenthesis, the function name loses its last character. Test inputs avoid these cases.
  # Parses a call written as text (a name followed by parenthesized keyword arguments) and splits it
  #   into the function name and the list of argument names.
  # Input: call_str, the call text. Returns a pair: the function name (everything before the first
  #   opening parenthesis) and a list of argument names (text before each equals sign, trimmed,
  #   empty entries dropped).
  # No side effects.
  """Parse call string to get function and argument names.

  Args:
    call_str: Call string must be in the form:
              `tf.foo(arg1=val1, arg2=val2, ...)`.

  Returns:
    (function_name, list of arg names) tuple.
  """
  open_paren_index = call_str.find("(")
  close_paren_index = call_str.rfind(")")

  function_name = call_str[:call_str.find("(")]
  args = call_str[open_paren_index + 1:close_paren_index].split(",")
  args = [arg.split("=")[0].strip() for arg in args]
  args = [arg for arg in args if arg]  # filter out empty strings
  return function_name, args


class TestUpgrade(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Main test case for the version 2.0 upgrade script. Each test method feeds a small piece of
  #   old-style source text through the upgrader built from the 2.0 change specification and checks
  #   the rewritten text, the generated report and the list of errors or warnings.
  # Class-level fixtures: two mappings from public API names to objects, one for the v2 API and one
  #   for the v1 API, filled once in the class setup. Instances are created by the test runner; the
  #   parameterized mixin supplies data-driven tests.
  # Helpers: an upgrade method for one text, a multi-file variant that reuses one upgrader, and a
  #   rename-correctness checker.
  """Test various APIs that have been changed in 2.0.

  We also test whether a converted file is executable. test_file_v1_10.py
  aims to exhaustively test that API changes are convertible and actually
  work when run with current TensorFlow.
  """

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # HELPER: builds the shared symbol tables used by the API-wide tests.
    # Class-level fixture run once before the tests. Calls the parent setup, then creates two empty
    #   mappings on the class: v2_symbols and v1_symbols.
    # If the compat v2 namespace exists, walks it with the public API visitor (skipping the compat
    #   sub-namespaces) and records every exported v2 name, prefixed with the root module name,
    #   mapped to its undecorated object. Does the same for compat v1 into v1_symbols.
    # No inputs besides the class; returns nothing. Side effect: sets class attributes.
    super(TestUpgrade, cls).setUpClass()
    cls.v2_symbols = {}
    cls.v1_symbols = {}
    if hasattr(tf.compat, "v2"):

      def symbol_collector(unused_path, unused_parent, children):
        # [FUNCTION DOC]
        # HELPER: visitor callback used by the class setup.
        # Called by the API traversal for each module with a list of children pairs. For each child
        #   it unwraps decorators, looks up the v2 export names of the object and stores each
        #   prefixed name in the class v2 table. Parent and path arguments are ignored. Returns
        #   nothing.
        for child in children:
          _, attr = tf_decorator.unwrap(child[1])
          api_names_v2 = tf_export.get_v2_names(attr)
          for name in api_names_v2:
            cls.v2_symbols["tf." + name] = attr

      visitor = public_api.PublicAPIVisitor(symbol_collector)
      visitor.private_map["tf.compat"] = ["v1", "v2"]
      traverse.traverse(tf.compat.v2, visitor)

    if hasattr(tf.compat, "v1"):

      def symbol_collector_v1(unused_path, unused_parent, children):
        # [FUNCTION DOC]
        # HELPER: visitor callback used by the class setup.
        # Same as the v2 collector but uses the v1 export names and stores them in the class v1
        #   table. Parent and path arguments are ignored. Returns nothing.
        for child in children:
          _, attr = tf_decorator.unwrap(child[1])
          api_names_v1 = tf_export.get_v1_names(attr)
          for name in api_names_v1:
            cls.v1_symbols["tf." + name] = attr

      visitor = public_api.PublicAPIVisitor(symbol_collector_v1)
      visitor.private_map["tf.compat"] = ["v1", "v2"]
      traverse.traverse(tf.compat.v1, visitor)

  def _upgrade(self,
               old_file_text,
               import_rename=False,
               upgrade_compat_v1_import=False):
    # [FUNCTION DOC]
    # HELPER: runs the upgrader on one piece of text for all tests in this class.
    # Inputs: old_file_text, the source text to convert; import_rename, optional flag (default
    #   false) telling the change spec to also rename the tensorflow import to the compat v2 form;
    #   upgrade_compat_v1_import, optional flag (default false) allowing the compat v1 import to be
    #   upgraded.
    # Wraps the text in an in-memory input stream, builds a code upgrader around the 2.0 change
    #   specification and processes it into an in-memory output stream under fake file names.
    # Returns a four-item tuple: the count of changes, the report text, the list of errors, and the
    #   converted text. No files are touched.
    in_file = io.StringIO(old_file_text)
    out_file = io.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(
        tf_upgrade_v2.TFAPIChangeSpec(
            import_rename, upgrade_compat_v1_import=upgrade_compat_v1_import))
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return count, report, errors, out_file.getvalue()

  def _upgrade_multiple(self, upgrade_compat_v1_import, old_file_texts):
    # [FUNCTION DOC]
    # HELPER: runs one upgrader on several texts in sequence.
    # Inputs: upgrade_compat_v1_import, flag passed to the change specification (import renaming is
    #   always on); old_file_texts, a list of source texts.
    # Creates a single upgrader and processes each text in order, so any state carried between files
    #   by the upgrader persists.
    # Returns a list with one entry per text; each entry is a list of count, report, errors and
    #   converted text. No files are touched.
    upgrader = ast_edits.ASTCodeUpgrader(
        tf_upgrade_v2.TFAPIChangeSpec(True, upgrade_compat_v1_import))
    results = []
    for old_file_text in old_file_texts:
      in_file = io.StringIO(old_file_text)
      out_file = io.StringIO()
      count, report, errors = (
          upgrader.process_opened_file("test.py", in_file,
                                       "test_out.py", out_file))
      results.append([count, report, errors, out_file.getvalue()])
    return results

  def testParseError(self):
    # [FUNCTION DOC]
    # Scenario: the input is not valid source (a dangling operator).
    # Action: run the upgrader.
    # Asserts: the report contains the message saying the file failed to parse. Takes no inputs,
    #   returns nothing.
    _, report, unused_errors, unused_new_text = self._upgrade(
        "import tensorflow as tf\na + \n")
    self.assertNotEqual(report.find("Failed to parse"), -1)

  def testReport(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the assertion is weak; it checks the truthiness of a text search result, which is
    #   true for not-found (minus one) and false only when the message sits at position zero, so it
    #   almost never fails.
    # Scenario: a call to the old angle function.
    # Action: upgrade it and search the report for the rename message.
    # Takes no inputs, returns nothing.
    text = "tf.angle(a)\n"
    _, report, unused_errors, unused_new_text = self._upgrade(text)
    # This is not a complete test, but it is a sanity test that a report
    # is generating information.
    self.assertTrue(
        report.find("Renamed function `tf.angle` to "
                    "`tf.math.angle`"))

  def testRename(self):
    # [FUNCTION DOC]
    # Scenario: simple renames of old math functions (conjugate, reciprocal square root,
    #   log-sigmoid), including nested calls.
    # Asserts the converted text moves them into the math namespace.
    text = "tf.conj(a)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.math.conj(a)\n")
    text = "tf.rsqrt(tf.log_sigmoid(3.8))\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.math.rsqrt(tf.math.log_sigmoid(3.8))\n")

  def testAllAPI(self):
    # [FUNCTION DOC]
    # Walks every symbol of the compat v1 API (skipping contrib), upgrades a bare reference to each,
    #   and fails if the converted name is not a compat v1 or v2 reference, is not in the v2 symbol
    #   table, and does not involve internal namespaces.
    # Returns immediately if the compat v2 namespace is unavailable. Uses the class v2 table; fails
    #   with an assertion message naming the offending symbol.
    if not hasattr(tf.compat, "v2"):
      return

    # Converts all symbols in the v1 namespace to the v2 namespace, raising
    # an error if the target of the conversion is not in the v2 namespace.
    # Please regenerate the renames file or edit any manual renames if this
    # test fails.
    def conversion_visitor(unused_path, unused_parent, children):
      # [FUNCTION DOC]
      # HELPER: traversal callback for the all-API test.
      # For each child of the visited module, unwraps decorators, finds its v1 export names,
      #   upgrades each name as text and asserts the output is acceptable (compat prefix, known v2
      #   symbol, or internal). Fails the test through an assertion otherwise. Ignores the path and
      #   parent arguments; returns nothing.
      for child in children:
        _, attr = tf_decorator.unwrap(child[1])
        api_names = tf_export.get_v1_names(attr)
        for name in api_names:
          _, _, _, text = self._upgrade("tf." + name)
          if (text and
              not text.startswith("tf.compat.v1") and
              not text.startswith("tf.compat.v2") and
              text not in self.v2_symbols and
              # Ignore any symbol that contains __internal__
              "__internal__" not in text):
            self.assertFalse(
                True, "Symbol %s generated from %s not in v2 API" % (
                    text, name))

    visitor = public_api.PublicAPIVisitor(conversion_visitor)
    visitor.do_not_descend_map["tf"].append("contrib")
    visitor.private_map["tf.compat"] = ["v1", "v2"]
    traverse.traverse(tf.compat.v1, visitor)

  def testAllAPIV1(self):
    # [FUNCTION DOC]
    # Two-pass traversal of the compat v1 API. First pass (collect mode) gathers all v1 names into a
    #   set; second pass upgrades each name and fails if the result is not a compat reference nor a
    #   known v1 name.
    # Local state: a collect flag and the symbol set, shared with the callback through closure; the
    #   flag is flipped between the two traversals. Takes no inputs, returns nothing.
    collect = True
    v1_symbols = set([])

    # Converts all symbols in the v1 namespace to the v2 namespace, raising
    # an error if the target of the conversion is not in the v1 namespace.
    def conversion_visitor(unused_path, unused_parent, children):
      # [FUNCTION DOC]
      # HELPER: traversal callback for the v1 all-API test.
      # In collect mode adds each prefixed v1 export name to the set. Otherwise upgrades the name
      #   and asserts the output starts with a compat prefix or is in the set, failing with a
      #   message otherwise. Reads the flag from the enclosing scope. Returns nothing.
      for child in children:
        _, attr = tf_decorator.unwrap(child[1])
        api_names = tf_export.get_v1_names(attr)
        for name in api_names:
          if collect:
            v1_symbols.add("tf." + name)
          else:
            _, _, _, text = self._upgrade("tf." + name)
            if (text and
                not text.startswith("tf.compat.v1") and
                not text.startswith("tf.compat.v2") and
                text not in v1_symbols):
              self.assertFalse(
                  True, "Symbol %s generated from %s not in v1 API" % (
                      text, name))

    visitor = public_api.PublicAPIVisitor(conversion_visitor)
    visitor.do_not_descend_map["tf"].append("contrib")
    visitor.private_map["tf.compat"] = ["v1", "v2"]
    traverse.traverse(tf.compat.v1, visitor)
    collect = False
    traverse.traverse(tf.compat.v1, visitor)

  def testV1KeywordArgNames(self):
    # [FUNCTION DOC]
    # Checks that every keyword rename registered in the change specification names an argument that
    #   really exists in the v1 function.
    # For each exported v1 API that has keyword renames, reads the function's argument names and
    #   asserts the type of the rename table is a mapping and each old name is present.
    # Traverses compat v1 skipping contrib. Fails through assertions.
    all_keyword_renames = (
        tf_upgrade_v2.TFAPIChangeSpec().function_keyword_renames)

    # Visitor that verifies V1 argument names.
    def arg_test_visitor(unused_path, unused_parent, children):
      # [FUNCTION DOC]
      # HELPER: traversal callback for the v1 keyword-name test.
      # For each child, gets its v1 export names; for names with keyword renames asserts each old
      #   argument name appears in the function's argument list, with a descriptive failure message.
      #   Returns nothing.
      for child in children:
        _, attr = tf_decorator.unwrap(child[1])
        names_v1 = tf_export.get_v1_names(attr)

        for name in names_v1:
          name = "tf.%s" % name
          if name not in all_keyword_renames:
            continue
          arg_names_v1 = tf_inspect.getargspec(attr)[0]
          keyword_renames = all_keyword_renames[name]
          self.assertEqual(type(keyword_renames), dict)

          # Assert that v1 function has valid v1 argument names.
          for from_name, _ in keyword_renames.items():
            self.assertIn(
                from_name, arg_names_v1,
                "%s not found in %s arguments: %s" %
                (from_name, name, str(arg_names_v1)))

    visitor = public_api.PublicAPIVisitor(arg_test_visitor)
    visitor.do_not_descend_map["tf"].append("contrib")
    visitor.private_map["tf.compat"] = ["v1", "v2"]
    traverse.traverse(tf.compat.v1, visitor)

  def testV2KeywordArgNames(self):
    # [FUNCTION DOC]
    # Checks that the converted keyword arguments are valid v2 and v1 arguments.
    # For every exported function, builds a call text assigning each v1 argument its index, upgrades
    #   it, parses the output, and verifies: functions moved to compat v1 are not ones with keyword
    #   renames, compat v2 targets exist, each output argument exists in the v2 function (plus a few
    #   allowed exceptions) and also in the v1 function (with two functions exempt).
    # Skips symbols needing manual change, the print function, and internal namespaces. Returns
    #   immediately if compat v2 is missing.
    # This test converts a call of the form:
    # tf.foo(arg1=0, arg2=1, ...)
    # to 2.0. Then, checks that converted function has valid argument names.
    if not hasattr(tf.compat, "v2"):
      return
    v2_arg_exceptions = {
        "verify_shape_is_now_always_true",
        # These arguments should not be used, they just specify
        # that a function takes named arguments.
        "keyword_required",
        "_sentinel",
    }
    v1_name_exceptions = {
        "tf.print",  # requires print_function import
    }
    function_warnings = (
        tf_upgrade_v2.TFAPIChangeSpec().function_warnings)
    function_transformers = (
        tf_upgrade_v2.TFAPIChangeSpec().function_transformers)
    keyword_renames = (
        tf_upgrade_v2.TFAPIChangeSpec().function_keyword_renames)

    # Visitor that converts to V2 and checks V2 argument names.
    def conversion_visitor(unused_path, unused_parent, children):
      # [FUNCTION DOC]
      # HELPER: traversal callback holding the main loop of the v2 keyword-name test.
      # Described in the parent test: for each function child, for each v1 name, creates and
      #   upgrades a keyword-style call, then asserts the converted names and arguments are valid in
      #   v2 and v1. Uses enclosing-scope tables. Fails through assertions or the fail call. Returns
      #   nothing.
      for child in children:
        _, attr = tf_decorator.unwrap(child[1])
        if not tf_inspect.isfunction(attr):
          continue
        names_v1 = tf_export.get_v1_names(attr)
        arg_names_v1 = get_args(attr)

        for name in names_v1:
          tf_name = "tf.%s" % name
          if tf_name in function_warnings or tf_name in function_transformers:
            continue  # These require manual change
          if tf_name in v1_name_exceptions:
            continue
          # Assert that arg names after converting to v2 are present in
          # v2 function.
          # 1. First, create an input of the form:
          #    tf.foo(arg1=val1, arg2=val2, ...)
          args = ",".join(
              ["%s=%d" % (from_name, from_index)
               for from_index, from_name in enumerate(arg_names_v1)])
          text_input = "%s(%s)" % (tf_name, args)
          # 2. Convert the input to V2.
          _, _, _, text = self._upgrade(text_input)
          new_function_name, new_args = get_func_and_args_from_str(text)
          if "__internal__" in new_function_name:
            # Skip the tf.__internal__ and tf.keras.__internal__ API.
            continue
          if new_function_name == "tf.compat.v1.%s" % name:
            if tf_name in keyword_renames:
              # If we rename arguments, new function must be available in 2.0.
              # We should not be using compat.v1 in this case.
              self.fail(
                  "Function '%s' is not in 2.0 when converting\n%s\nto\n%s" %
                  (new_function_name, text_input, text))
            continue
          if new_function_name.startswith("tf.compat.v2"):
            self.assertIn(new_function_name.replace("tf.compat.v2.", "tf."),
                          self.v2_symbols)
            continue
          # 3. Verify V2 function and arguments.
          args_v2 = get_args(self.v2_symbols[new_function_name])
          args_v2.extend(v2_arg_exceptions)
          for new_arg in new_args:
            self.assertIn(
                new_arg, args_v2,
                "Invalid argument '%s' in 2.0 when converting\n%s\nto\n%s.\n"
                "Supported arguments: %s" % (
                    new_arg, text_input, text, str(args_v2)))
          # 4. Verify that the argument exists in v1 as well.
          if new_function_name in set(["tf.nn.ctc_loss",
                                       "tf.saved_model.save"]):
            continue
          args_v1 = get_args(self.v1_symbols[new_function_name])
          args_v1.extend(v2_arg_exceptions)
          for new_arg in new_args:
            self.assertIn(
                new_arg, args_v1,
                "Invalid argument '%s' in 1.0 when converting\n%s\nto\n%s.\n"
                "Supported arguments: %s" % (
                    new_arg, text_input, text, str(args_v1)))

    visitor = public_api.PublicAPIVisitor(conversion_visitor)
    visitor.do_not_descend_map["tf"].append("contrib")
    visitor.private_map["tf.compat"] = ["v1", "v2"]
    traverse.traverse(tf.compat.v1, visitor)

  def testPositionsMatchArgGiven(self):
    # [FUNCTION DOC]
    # Checks that the argument-position warnings registered in the change spec point to the right
    #   positions.
    # For each registered method (skipping contrib ones), resolves the real function
    #   (optimizer-style names starting with a wildcard are mapped to the train optimizer class and
    #   positions shifted by one for self), reads its argument spec and asserts the argument at the
    #   stated position has the stated name. Fails on mismatch.
    full_dict = tf_upgrade_v2.TFAPIChangeSpec().function_arg_warnings
    method_names = list(full_dict.keys())
    for method_name in method_names:
      args = list(full_dict[method_name].keys())
      if "contrib" in method_name:
        # Skip descending and fetching contrib methods during test. These are
        # not available in the repo anymore.
        continue
      elif method_name.startswith("*."):
        # special case for optimizer methods
        method = method_name.replace("*", "tf.train.Optimizer")
      else:
        method = method_name

      method = get_symbol_for_name(tf, method)
      arg_spec = tf_inspect.getfullargspec(method)
      for (arg, pos) in args:
        # to deal with the self argument on methods on objects
        if method_name.startswith("*."):
          pos += 1
        self.assertEqual(arg_spec[0][pos], arg)

  def testReorderFileNeedsUpdate(self):
    # [FUNCTION DOC]
    # Consistency test for the generated reorder table.
    # Asserts that every name in the reordered-function-names set is a key of the function-reorders
    #   table, and that every non-manual key has at least one v1 alias present in the reordered
    #   names set. Failure messages tell the developer to rerun the generator for the reorders map.
    reordered_function_names = (
        tf_upgrade_v2.TFAPIChangeSpec().reordered_function_names)
    function_reorders = (
        tf_upgrade_v2.TFAPIChangeSpec().function_reorders)
    manual_function_reorders = (
        tf_upgrade_v2.TFAPIChangeSpec().manual_function_reorders)

    added_names_message = """Some function names in
self.reordered_function_names are not in reorders_v2.py.
Please run the following commands to update reorders_v2.py:
bazel run tensorflow/tools/compatibility/update:generate_v2_reorders_map
"""
    removed_names_message = """%s in self.reorders_v2 does not match
any name in self.reordered_function_names.
Please run the following commands to update reorders_v2.py:
bazel run tensorflow/tools/compatibility/update:generate_v2_reorders_map
"""
    self.assertTrue(
        reordered_function_names.issubset(function_reorders),
        added_names_message)
    # function_reorders should contain reordered_function_names
    # and their TensorFlow V1 aliases.
    for name in function_reorders:
      if name in manual_function_reorders:
        continue
      # get other names for this function
      attr = get_symbol_for_name(tf.compat.v1, name)
      _, attr = tf_decorator.unwrap(attr)
      v1_names = tf_export.get_v1_names(attr)
      self.assertTrue(v1_names)
      v1_names = ["tf.%s" % n for n in v1_names]
      # check if any other name is in
      self.assertTrue(
          any(n in reordered_function_names for n in v1_names),
          removed_names_message % name)

  def testRenameConstant(self):
    # [FUNCTION DOC]
    # Scenario: reference to a constant, alone and as a call argument.
    # Asserts it moves to the sysconfig namespace.
    text = "tf.MONOLITHIC_BUILD\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.sysconfig.MONOLITHIC_BUILD\n")
    text = "some_call(tf.MONOLITHIC_BUILD)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "some_call(tf.sysconfig.MONOLITHIC_BUILD)\n")

  def testRenameArgs(self):
    # [FUNCTION DOC]
    # Scenario: pooling call with eight positional arguments.
    # Asserts the first four stay positional and the remaining ones get keywords, including the
    #   renamed dilation rate and the renamed padding-related ones.
    text = ("tf.nn.pool(input_a, window_shape_a, pooling_type_a, padding_a, "
            "dilation_rate_a, strides_a, name_a, data_format_a)\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text,
                     ("tf.nn.pool(input_a, window_shape_a, pooling_type_a, "
                      "padding=padding_a, dilations=dilation_rate_a, "
                      "strides=strides_a, name=name_a, "
                      "data_format=data_format_a)\n"))

  def testReorder(self):
    # [FUNCTION DOC]
    # Scenario: boolean mask with positional arguments.
    # Asserts the last two arguments become keyword arguments in the new order (name and axis).
    text = "tf.boolean_mask(a, b, c, d)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, "tf.boolean_mask(a, b, name=c, axis=d)\n")

  def testLearningRateDecay(self):
    # [FUNCTION DOC]
    # Scenario: each old learning-rate decay function in the train namespace called with two
    #   arguments.
    # Asserts the report suggests switching to the Keras schedules.
    for decay in ["tf.train.exponential_decay",
                  "tf.train.polynomial_decay", "tf.train.natural_exp_decay",
                  "tf.train.inverse_time_decay", "tf.train.cosine_decay",
                  "tf.train.cosine_decay_restarts",
                  "tf.train.linear_cosine_decay",
                  "tf.train.noisy_linear_cosine_decay",
                  "tf.train.piecewise_constant_decay",
                 ]:

      text = "%s(a, b)\n" % decay
      _, report, unused_errors, _ = self._upgrade(text)
      self.assertIn("switch to the schedules in "
                    "`tf.keras.optimizers.schedules`", report)

  def verify_compat_v1_rename_correctness(self, values, ns_prefix=""):
    # [FUNCTION DOC]
    # HELPER: shared check for the initializer test.
    # Inputs: values, list of function names; ns_prefix, optional namespace (default empty) inserted
    #   after the root module name with a trailing dot added if non-empty.
    # For each name, upgrades a two-argument call and asserts it is rewritten with the compat v1
    #   prefix at the same place. Fails through assertions; returns nothing. Mutates only its local
    #   prefix.
    if ns_prefix:
      ns_prefix += "."
    for v in values:
      text = "tf." + ns_prefix + v + "(a, b)"
      _, _, _, new_text = self._upgrade(text)
      self.assertEqual("tf.compat.v1." + ns_prefix + v + "(a, b)", new_text)

  def testInitializers(self):
    # [FUNCTION DOC]
    # Scenario: three lists of initializer names (functions in the initializers namespace, top-level
    #   initializer classes, and Keras initializers).
    # Asserts every one is redirected to compat v1 under the right namespace, using the helper.
    initializers = [
        "zeros",
        "ones",
        "constant",
        "random_uniform",
        "random_normal",
        "truncated_normal",
        "variance_scaling",
        "orthogonal",
        "glorot_uniform",
        "glorot_normal",
        "identity",
        "lecun_normal",
        "lecun_uniform",
        "he_normal",
        "he_uniform",
    ]
    self.verify_compat_v1_rename_correctness(
        initializers, ns_prefix="initializers")

    initializers = [
        "zeros_initializer",
        "ones_initializer",
        "constant_initializer",
        "random_uniform_initializer",
        "random_normal_initializer",
        "truncated_normal_initializer",
        "variance_scaling_initializer",
        "orthogonal_initializer",
        "glorot_uniform_initializer",
        "glorot_normal_initializer",
    ]
    self.verify_compat_v1_rename_correctness(initializers)

    initializers = [
        "zeros",
        "ones",
        "Ones",
        "Zeros",
        "constant",
        "Constant",
        "VarianceScaling",
        "Orthogonal",
        "orthogonal",
        "Identity",
        "identity",
        "glorot_uniform",
        "glorot_normal",
        "lecun_normal",
        "lecun_uniform",
        "he_normal",
        "he_uniform",
        "TruncatedNormal",
        "truncated_normal",
        "RandomUniform",
        "uniform",
        "random_uniform",
        "RandomNormal",
        "normal",
        "random_normal",
    ]
    self.verify_compat_v1_rename_correctness(
        initializers, ns_prefix="keras.initializers")

  def testContribXavierInitializer(self):
    # [FUNCTION DOC]
    # Scenario: Xavier initializers from contrib and slim aliases with different combinations of
    #   uniform flag, seed and data type, positional and keyword.
    # Asserts each is rewritten to a variance scaling initializer with fan-average mode, and the
    #   uniform flag becomes a conditional distribution expression when not a literal. Loops over
    #   two contrib aliases.
    for contrib_alias in ["tf.contrib.", "contrib_"]:
      text = contrib_alias + "layers.xavier_initializer()\n"
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=\"uniform\")\n",
      )

      text = "slim.xavier_initializer(True or False)\n"
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=(\"uniform\" if True or False else "
          "\"truncated_normal\"))\n",
      )

      text = "slim.xavier_initializer(uniform=(True or False))\n"
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=(\"uniform\" if True or False else "
          "\"truncated_normal\"))\n",
      )

      text = contrib_alias + "layers.xavier_initializer_conv2d(False, 12)\n"
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=(\"uniform\" if False else \"truncated_normal\"), "
          "seed=12)\n",
      )

      text = (contrib_alias + "layers.xavier_initializer_conv2d("
              "False, 12, tf.float32)\n")
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=(\"uniform\" if False else \"truncated_normal\"), "
          "seed=12, "
          "dtype=tf.float32)\n",
      )

      text = (contrib_alias + "layers.xavier_initializer("
              "False, 12, dtypes=tf.float32)\n")
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(
          new_text,
          "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, "
          "mode=\"fan_avg\", "
          "distribution=(\"uniform\" if False else \"truncated_normal\"), "
          "seed=12, "
          "dtypes=tf.float32)\n",
      )

  def testVarianceScalingInitializer(self):
    # [FUNCTION DOC]
    # Scenario: variance scaling initializers from contrib/slim with factor, mode, uniform flag, and
    #   data type given in various ways.
    # Asserts conversion to the Keras variance scaling initializer with scale mapped from factor,
    #   mode lowered, and distribution chosen from the uniform flag.
    text = ("tf.contrib.layers.variance_scaling_initializer("
            "mode=(\"FAN\" + \"_AVG\"))\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0, "
        "mode=(\"FAN\" + \"_AVG\").lower())\n",
    )

    text = ("slim.variance_scaling_initializer("
            "uniform=(True or False), mode=(\"FAN\" + \"_AVG\"))\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.compat.v1.keras.initializers.VarianceScaling(scale=2.0, "
        "distribution=(\"uniform\" if True or False else \"truncated_normal\"),"
        " mode=(\"FAN\" + \"_AVG\").lower())\n",
    )

    text = "tf.contrib.layers.variance_scaling_initializer(factor=1.0)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0)\n",
    )

    text = ("tf.contrib.layers.variance_scaling_initializer("
            "12.0, \"FAN_AVG\", True, dtypes=tf.float32)\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.compat.v1.keras.initializers.VarianceScaling(12.0, "
        "(\"FAN_AVG\").lower(), "
        "(\"uniform\" if True else \"truncated_normal\"), "
        "dtypes=tf.float32)\n",
    )

  def testMetrics(self):
    # [FUNCTION DOC]
    # Scenario: each metric name in the old metrics namespace.
    # Asserts redirect to compat v1 metrics and that the report says they were replaced by
    #   object-oriented versions.
    metrics = [
        "accuracy",
        "auc",
        "average_precision_at_k",
        "false_negatives",
        "false_negatives_at_thresholds",
        "false_positives",
        "false_positives_at_thresholds",
        "mean",
        "mean_absolute_error",
        "mean_cosine_distance",
        "mean_iou",
        "mean_per_class_accuracy",
        "mean_relative_error",
        "mean_squared_error",
        "mean_tensor",
        "percentage_below",
        "precision",
        "precision_at_k",
        "precision_at_thresholds",
        "precision_at_top_k",
        "recall",
        "recall_at_k",
        "recall_at_thresholds",
        "recall_at_top_k",
        "root_mean_squared_error",
        "sensitivity_at_specificity",
        "sparse_average_precision_at_k",
        "sparse_precision_at_k",
        "specificity_at_sensitivity",
        "true_negatives",
        "true_negatives_at_thresholds",
        "true_positives",
        "true_positives_at_thresholds",
    ]
    for m in metrics:
      text = "tf.metrics." + m + "(a, b)"
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual("tf.compat.v1.metrics." + m + "(a, b)", new_text)
      self.assertIn(
          "tf.metrics have been replaced with object oriented versions", report)

  def testLosses(self):
    # [FUNCTION DOC]
    # Scenario: each loss name in the old losses namespace.
    # Asserts redirect to compat v1 losses and that the report says they were replaced by
    #   object-oriented versions.
    losses = [
        "absolute_difference",
        "add_loss",
        "compute_weighted_loss",
        "cosine_distance",
        "get_losses",
        "get_regularization_loss",
        "get_regularization_losses",
        "get_total_loss",
        "hinge_loss",
        "huber_loss",
        "log_loss",
        "mean_pairwise_squared_error",
        "mean_squared_error",
        "sigmoid_cross_entropy",
        "softmax_cross_entropy",
        "sparse_softmax_cross_entropy",
    ]
    for l in losses:
      text = "tf.losses." + l + "(a, b)"
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual("tf.compat.v1.losses." + l + "(a, b)", new_text)
      self.assertIn(
          "tf.losses have been replaced with object oriented versions", report)

  def testExtractGlimpse(self):
    # [FUNCTION DOC]
    # Scenario: glimpse extraction calls with the uniform-noise flag positional, keyword and
    #   multiline with comments, and one without arguments.
    # Asserts the flag becomes a noise choice expression, comments and layout survive, and the
    #   argument-less call is unchanged with no errors.
    text = ("tf.image.extract_glimpse(x, size, off, False, "
            "False, False, name=\"foo\")\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.image.extract_glimpse(x, size, off, False, "
        "False, 'uniform' if (False) else 'gaussian', name=\"foo\")\n",
    )

    text = ("tf.image.extract_glimpse(x, size, off, centered=False, "
            "normalized=False, uniform_noise=True if uniform_noise else "
            "False, name=\"foo\")\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.image.extract_glimpse(x, size, off, centered=False, "
        "normalized=False, noise='uniform' if (True if uniform_noise else "
        "False) else 'gaussian', name=\"foo\")\n",
    )

    text = ("tf.image.extract_glimpse(x,\n"
            "                         size,\n"
            "                         off,\n"
            "                         centered=True,\n"
            "                         normalized=True, # Stuff before\n"
            "                         uniform_noise=False,\n"
            "                         name=\"foo\")# Stuff after\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text, "tf.image.extract_glimpse(x,\n"
        "                         size,\n"
        "                         off,\n"
        "                         centered=True,\n"
        "                         normalized=True, # Stuff before\n"
        "                         noise='uniform' if (False) else 'gaussian',\n"
        "                         name=\"foo\")# Stuff after\n")

    text = "tf.image.extract_glimpse(x)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, text)
    self.assertEqual(errors, [])

  def testDropout(self):
    # [FUNCTION DOC]
    # Scenario: dropout with keep probability positional, keyword, in a multiline call with
    #   comments, and with no arguments.
    # Asserts conversion to the rate form as one minus the value, comment preservation, and for the
    #   bare call no change plus an error saying it was called without arguments.
    text = "tf.nn.dropout(x, keep_prob, name=\"foo\")\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.nn.dropout(x, rate=1 - (keep_prob), name=\"foo\")\n",
    )

    text = "tf.nn.dropout(x, keep_prob=.4, name=\"foo\")\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.nn.dropout(x, rate=1 - (.4), name=\"foo\")\n",
    )

    text = (
        "tf.nn.dropout(x,  # Stuff before\n"
        "              keep_prob=.4,  # Stuff after\n"
        "              name=\"foo\")\n"
    )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.nn.dropout(x,  # Stuff before\n"
        "              rate=1 - (.4),  # Stuff after\n"
        "              name=\"foo\")\n",
    )

    text = "tf.nn.dropout(x)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, text)
    self.assertIn("tf.nn.dropout called without arguments", errors[0])

  def testDropoutExpr(self):
    # [FUNCTION DOC]
    # Scenario: dropout whose keep probability is an arithmetic expression.
    # Asserts the expression is wrapped in parentheses in the rate computation.
    text = "tf.nn.dropout(x, 1 - func(3 + 4.), name=\"foo\")\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.nn.dropout(x, rate=1 - (1 - func(3 + 4.)), name=\"foo\")\n",
    )

  def testContribL1(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the report variable is named as unused yet it is read by the assertions; the name
    #   is misleading.
    # Scenario: L1 regularizer from contrib and slim with and without a scope argument.
    # Asserts conversion to the Keras L1 regularizer, argument renamed in the keyword case, layout
    #   comments kept, and the report mentions dropping scope only when scope was given.
    text = "tf.contrib.layers.l1_regularizer(scale)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l1(scale)\n",
    )
    self.assertNotIn("Dropping scope", unused_report)

    text = "tf.contrib.layers.l1_regularizer(scale, scope)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l1(scale)\n",
    )
    self.assertIn("Dropping scope", unused_report)

    text = (
        "slim.l1_regularizer(  # Stuff before\n"
        "                    scale=.4,"
        "                    scope=\"foo\")\n"
    )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l1(  # Stuff before\n"
        "                    l=.4)\n",
    )
    self.assertIn("Dropping scope", unused_report)

  def testContribL2(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the report variable is named as unused yet it is read by the assertions; the name
    #   is misleading.
    # Scenario: L2 regularizer from contrib and slim with and without scope.
    # Asserts conversion to the Keras L2 regularizer with the scale halved in parentheses, and the
    #   dropping-scope message only when scope was given.
    text = "tf.contrib.layers.l2_regularizer(scale)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l2(0.5 * (scale))\n",
    )
    self.assertNotIn("Dropping scope", unused_report)

    text = "tf.contrib.layers.l2_regularizer(scale, scope)\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l2(0.5 * (scale))\n",
    )
    self.assertIn("Dropping scope", unused_report)

    text = (
        "slim.l2_regularizer(  # Stuff before\n"
        "                    scale=.4,"
        "                    scope=\"foo\")\n"
    )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l2(  # Stuff before\n"
        "                    l=0.5 * (.4))\n",
    )
    self.assertIn("Dropping scope", unused_report)

  def testContribL2Expr(self):
    # [FUNCTION DOC]
    # Scenario: L2 regularizer with an expression as scale plus a scope.
    # Asserts the expression is parenthesized inside the halving and the scope removed.
    text = "tf.contrib.layers.l2_regularizer(1 - func(3 + 4.), scope=\"foo\")\n"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(
        new_text,
        "tf.keras.regularizers.l2(0.5 * (1 - func(3 + 4.)))\n",
    )

  def testMathCountNonZeroChanges(self):
    # [FUNCTION DOC]
    # Scenario: nonzero counting in the math namespace with old argument names.
    # Asserts input tensor, reduction indices and keep dims are renamed to input, axis and keepdims.
    text = (
        "tf.math.count_nonzero(input_tensor=input, dtype=dtype, name=name, "
        "reduction_indices=axis, keep_dims=keepdims)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.math.count_nonzero(input=input, dtype=dtype, name=name, "
        "axis=axis, keepdims=keepdims)\n"
        )
    self.assertEqual(new_text, expected_text)

  def testCountNonZeroChanges(self):
    # [FUNCTION DOC]
    # Scenario: the same old nonzero count at top level.
    # Asserts it moves to the math namespace and arguments are renamed.
    text = (
        "tf.count_nonzero(input_tensor=input, dtype=dtype, name=name, "
        "reduction_indices=axis, keep_dims=keepdims)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.math.count_nonzero(input=input, dtype=dtype, name=name, "
        "axis=axis, keepdims=keepdims)\n"
        )
    self.assertEqual(new_text, expected_text)

  def testRandomMultinomialToRandomCategorical(self):
    # [FUNCTION DOC]
    # Scenario: random multinomial in two spellings with positional arguments.
    # Asserts both become the categorical sampler with seed and name as keywords and the output type
    #   renamed to dtype.
    text = (
        "tf.random.multinomial(logits, samples, seed, name, output_dtype)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.random.categorical(logits, samples, seed=seed, name=name, "
        "dtype=output_dtype)\n"
        )
    self.assertEqual(new_text, expected_text)

    text = (
        "tf.multinomial(logits, samples, seed, name, output_dtype)\n"
        )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.random.categorical(logits, samples, seed=seed, name=name, "
        "dtype=output_dtype)\n"
        )
    self.assertEqual(new_text, expected_text)

  def testRandomPoissonConversion(self):
    # [FUNCTION DOC]
    # Scenario: random Poisson in the old and new spellings.
    # Asserts both produce the new name with all three arguments as keywords.
    text1 = "tf.random_poisson(lam, shape, dtype)"
    text2 = "tf.random.poisson(lam, shape, dtype)"
    expected_text = "tf.random.poisson(lam=lam, shape=shape, dtype=dtype)"
    _, unused_report, unused_errors, new_text1 = self._upgrade(text1)
    self.assertEqual(new_text1, expected_text)
    _, unused_report, unused_errors, new_text2 = self._upgrade(text2)
    self.assertEqual(new_text2, expected_text)

  def testConvolutionOpUpdate(self):
    # [FUNCTION DOC]
    # Scenario: convolution with seven positional arguments.
    # Asserts filter becomes filters and dilation rate becomes dilations, others become keywords.
    text = (
        "tf.nn.convolution(input, filter, padding, strides, dilation_rate, "
        "name, data_format)"
    )
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    expected_text = (
        "tf.nn.convolution(input, filters=filter, padding=padding, "
        "strides=strides, dilations=dilation_rate, name=name, "
        "data_format=data_format)"
    )
    self.assertEqual(new_text, expected_text)

  def test_substr(self):
    # [FUNCTION DOC]
    # Scenario: substring op with positional arguments.
    # Asserts it moves to the strings namespace, name and unit become keywords, and there are no
    #   errors.
    text = "tf.substr(input, pos, len, name, unit)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual("tf.strings.substr(input, pos, len, name=name, "
                     "unit=unit)\n", new_text)
    self.assertEqual(errors, [])

  def testColocateGradientsWithOps(self):
    # [FUNCTION DOC]
    # Scenario: gradients calls: unrelated keyword (unchanged, no errors), the removed colocate
    #   argument (dropped with a report note), and a positional call where the colocate argument is
    #   dropped and gate becomes a keyword.
    text = "tf.gradients(yx=a, foo=False)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)
    self.assertEqual(errors, [])

    text = "tf.gradients(yx=a, colocate_gradients_with_ops=False)\n"
    _, report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual("tf.gradients(yx=a)\n", new_text)
    self.assertIn("tf.gradients no longer takes", report)

    text = "tf.gradients(y, x, grad_ys, name, colocate, gate)\n"
    expected = ("tf.gradients(y, x, grad_ys, name, gate_gradients=gate)\n")
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def testColocateGradientsWithOpsMinimize(self):
    # [FUNCTION DOC]
    # Scenario: optimizer minimize with an unrelated keyword (unchanged, no errors) and with the
    #   removed colocate argument.
    # Asserts the argument is dropped and the report notes minimize no longer takes it.
    text = "optimizer.minimize(a, foo=False)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)
    self.assertEqual(errors, [])

    text = "optimizer.minimize(a, colocate_gradients_with_ops=False)\n"
    _, report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual("optimizer.minimize(a)\n", new_text)
    self.assertIn("Optimizer.minimize no longer takes", report)

  def testColocateGradientsWithOpsComputeGradients(self):
    # [FUNCTION DOC]
    # Scenario: optimizer compute-gradients with an unrelated keyword (unchanged) and with the
    #   removed colocate argument.
    # Asserts it is dropped and the report notes the change.
    text = "optimizer.compute_gradients(a, foo=False)\n"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(text, new_text)
    self.assertEqual(errors, [])

    text = "optimizer.compute_gradients(a, colocate_gradients_with_ops=False)\n"
    _, report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual("optimizer.compute_gradients(a)\n", new_text)
    self.assertIn("Optimizer.compute_gradients no longer takes", report)

  def testColocateGradientsWithHessians(self):
    # [FUNCTION DOC]
    # Scenario: hessians with the removed colocate argument.
    # Asserts it is dropped and the report notes the change.
    text = "tf.hessians(ys=a, xs=b, colocate_gradients_with_ops=False)\n"
    _, report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual("tf.hessians(ys=a, xs=b)\n", new_text)
    self.assertIn("tf.hessians no longer takes", report)

  def testArgmin(self):
    # [FUNCTION DOC]
    # Scenario: argmin and the old arg-min name, with dimension keyword or positional arguments.
    # Asserts dimension becomes axis, the third positional becomes name, and the old name is
    #   replaced.
    text = "tf.argmin(input, name=n, dimension=1, output_type=type)"
    expected_text = "tf.argmin(input, name=n, axis=1, output_type=type)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.argmin(input, 0, n)"
    expected_text = "tf.argmin(input, 0, name=n)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.arg_min(input, 0)"
    expected_text = "tf.argmin(input, 0)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.arg_min(input, dimension=0)"
    expected_text = "tf.argmin(input, axis=0)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testArgmax(self):
    # [FUNCTION DOC]
    # Scenario: argmax and the old arg-max name, mirroring the argmin test.
    # Asserts the same renames.
    text = "tf.argmax(input, name=n, dimension=1, output_type=type)"
    expected_text = "tf.argmax(input, name=n, axis=1, output_type=type)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.argmax(input, 0, n)"
    expected_text = "tf.argmax(input, 0, name=n)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.arg_max(input, 0)"
    expected_text = "tf.argmax(input, 0)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.arg_max(input, dimension=0)"
    expected_text = "tf.argmax(input, axis=0)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testAutograph(self):
    # [FUNCTION DOC]
    # Scenario: autograph to-graph and to-code with argument value/type and indentation arguments.
    # Asserts those removed arguments are dropped.
    text = "tf.autograph.to_graph(f, True, arg_values=None, arg_types=None)"
    expected_text = "tf.autograph.to_graph(f, True)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = ("tf.autograph.to_code"
            "(f, False, arg_values=None, arg_types=None, indentation=' ')")
    expected_text = "tf.autograph.to_code(f, False)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testBatchToSpace(self):
    # [FUNCTION DOC]
    # Scenario: batch-to-space variants.
    # Asserts the n-dimensional forms are renamed to the plain name and the old block-size form gets
    #   keywords crops and block shape.
    text = "tf.batch_to_space_nd(input, block_shape, crops, name)"
    expected_text = "tf.batch_to_space(input, block_shape, crops, name)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.batch_to_space(input, crops, block_size, name)"
    expected_text = (
        "tf.batch_to_space(input, crops=crops, block_shape=block_size, "
        "name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.manip.batch_to_space_nd(input, block_shape, crops, name)"
    expected_text = "tf.batch_to_space(input, block_shape, crops, name)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testExtractImagePatches(self):
    # [FUNCTION DOC]
    # Scenario: the old extract-image-patches call.
    # Asserts it becomes the image extract-patches function with ksizes renamed to sizes.
    text = (
        "tf.extract_image_patches(images, ksizes=ksizes, strides=strides,"
        "rates=rates, padding=padding, name=name)")
    expected_text = (
        "tf.image.extract_patches(images, sizes=ksizes, strides=strides,"
        "rates=rates, padding=padding, name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testKerasSavedModel(self):
    # [FUNCTION DOC]
    # Scenario: contrib Keras saved model save and load calls.
    # Asserts they become compat v1 experimental export and load functions and the report suggests
    #   using model save.
    text = (
        "tf.contrib.saved_model.save_keras_model(model, './saved_models')\n"
        "tf.contrib.saved_model.load_keras_model(saved_model_path)\n")
    expected_text = (
        "tf.compat.v1.keras.experimental.export_saved_model(model, "
        "'./saved_models')\ntf.compat.v1.keras.experimental."
        "load_from_saved_model(saved_model_path)\n"
    )
    _, report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    expected_info = "Please use model.save"
    self.assertIn(expected_info, report)

  def testStatelessMultinomial(self):
    # [FUNCTION DOC]
    # Scenario: stateless multinomial with output dtype keyword.
    # Asserts renamed to stateless categorical with dtype.
    text = (
        "tf.random.stateless_multinomial(logits, num_samples, seed, "
        "output_dtype=dtype, name=name)")
    expected_text = (
        "tf.random.stateless_categorical(logits, num_samples, seed, "
        "dtype=dtype, name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSoftMaxCrossEntropyWithLogitsV2(self):
    # [FUNCTION DOC]
    # Scenario: the v2 softmax cross entropy name with a dim argument.
    # Asserts it becomes the plain name with axis and there are no errors.
    text = (
        "tf.nn.softmax_cross_entropy_with_logits_v2("
        "labels=labels, logits=logits, dim=2)")
    expected_text = (
        "tf.nn.softmax_cross_entropy_with_logits("
        "labels=labels, logits=logits, axis=2)")
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    self.assertFalse(errors)

  def testSoftMaxCrossEntropyWithLogits(self):
    # [FUNCTION DOC]
    # Scenario: the old softmax cross entropy with labels given as a name or a call.
    # Asserts labels are wrapped in a stop-gradient call and dim becomes axis.
    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=labels, logits=logits, dim=2)")
    expected_text = (
        "tf.nn.softmax_cross_entropy_with_logits("
        "labels=tf.stop_gradient(labels), logits=logits, axis=2)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=foo(bar))")
    expected_text = ("tf.nn.softmax_cross_entropy_with_logits("
                     "labels=tf.stop_gradient(foo(bar)))")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testSoftMaxCrossEntropyWithLogitsDoesntNest(self):
    # [FUNCTION DOC]
    # Scenario: labels already wrapped in stop-gradient, or arbitrary call results.
    # Asserts no double wrapping occurs for existing wrappers, while calls and call chains get
    #   wrapped once.
    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=tf.stop_gradient(labels), logits=logits, dim=2)")
    expected_text = (
        "tf.nn.softmax_cross_entropy_with_logits("
        "labels=tf.stop_gradient(labels), logits=logits, axis=2)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=tf.stop_gradient(foo(bar)))")
    expected_text = ("tf.nn.softmax_cross_entropy_with_logits("
                     "labels=tf.stop_gradient(foo(bar)))")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=foo())")
    expected_text = ("tf.nn.softmax_cross_entropy_with_logits("
                     "labels=tf.stop_gradient(foo()))")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = ("tf.nn.softmax_cross_entropy_with_logits("
            "labels=foo().zz())")
    expected_text = ("tf.nn.softmax_cross_entropy_with_logits("
                     "labels=tf.stop_gradient(foo().zz()))")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testSparseMatmul(self):
    # [FUNCTION DOC]
    # Scenario: sparse matmul with seven positional arguments.
    # Asserts it becomes the linalg matmul with sparse flags and name as keywords.
    text = ("tf.sparse_matmul(a, b, c, d, e, f, g)\n")
    expected_text = ("tf.linalg.matmul(a, b, c, d, a_is_sparse=e, "
                     "b_is_sparse=f, name=g)\n")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testWeightedMoments(self):
    # [FUNCTION DOC]
    # Scenario: weighted moments with five positional arguments.
    # Asserts name and keepdims become keywords.
    text = "tf.nn.weighted_moments(x, axes, freq, name, kd)"
    expected_text = (
        "tf.nn.weighted_moments(x, axes, freq, name=name, keepdims=kd)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSparseAdd(self):
    # [FUNCTION DOC]
    # Scenario: sparse add with thresh keyword.
    # Asserts it is renamed to threshold.
    text = "tf.sparse.add(a, b, thresh=t)"
    expected_text = "tf.sparse.add(a, b, threshold=t)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSparseConcat(self):
    # [FUNCTION DOC]
    # Scenario: sparse concat with five positional arguments.
    # Asserts the last three become name, expand-nonconcat-dims and axis keywords.
    text = "tf.sparse.concat(ax, inp, name, exp, concat)"
    expected_text = (
        "tf.sparse.concat(ax, inp, name=name, expand_nonconcat_dims=exp, "
        "axis=concat)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSeparableConv2D(self):
    # [FUNCTION DOC]
    # Scenario: separable convolution with eight positional arguments.
    # Asserts rate becomes dilations and the last two become keywords.
    text = "tf.nn.separable_conv2d(inp, d, pt, strides, pad, rate, name, fmt)"
    expected_text = (
        "tf.nn.separable_conv2d(inp, d, pt, strides, pad, dilations=rate, "
        "name=name, data_format=fmt)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testConv2D(self):
    # [FUNCTION DOC]
    # Scenario: conv2d with the cudnn flag positional and keyword.
    # Asserts filter becomes filters, the cudnn flag is dropped, and data format becomes a keyword.
    text = (
        "tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu, "
        "data_format)")
    expected_text = (
        "tf.nn.conv2d(input, filters=filter, strides=strides, padding=padding, "
        "data_format=data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = (
        "tf.nn.conv2d(input, filter=filter, strides=strides, padding=padding, "
        "use_cudnn_on_gpu=use_cudnn_on_gpu)")
    expected_text = ("tf.nn.conv2d(input, filters=filter, strides=strides, "
                     "padding=padding)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testConv2DBackpropFilter(self):
    # [FUNCTION DOC]
    # Scenario: conv2d backprop filter.
    # Asserts the call moves to compat v1 with arguments unchanged.
    text = (
        "tf.nn.conv2d_backprop_filter(input, filter_sizes, out_backprop, "
        "strides, padding, use_cudnn_on_gpu, data_format)")
    expected_text = (
        "tf.compat.v1.nn.conv2d_backprop_filter(input, filter_sizes, "
        "out_backprop, strides, padding, use_cudnn_on_gpu, data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testConv2DBackpropInput(self):
    # [FUNCTION DOC]
    # Scenario: conv2d backprop input.
    # Asserts it becomes conv2d transpose with keywords and the cudnn flag removed.
    text = (
        "tf.nn.conv2d_backprop_input(input_sizes, filter, out_backprop, "
        "strides, padding, use_cudnn_on_gpu, data_format)")
    expected_text = (
        "tf.nn.conv2d_transpose(output_shape=input_sizes, filters=filter, "
        "input=out_backprop, strides=strides, padding=padding, "
        "data_format=data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSpacetoBatch(self):
    # [FUNCTION DOC]
    # Scenario: n-dimensional space-to-batch and the nn version with block size.
    # Asserts rename and keywords.
    text = "tf.space_to_batch_nd(input, shape, paddings, name)"
    expected_text = "tf.space_to_batch(input, shape, paddings, name)"
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.nn.space_to_batch(input, paddings, block_size, name)"
    expected_text = (
        "tf.space_to_batch(input, paddings=paddings, block_shape=block_size, "
        "name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testInTopK(self):
    # [FUNCTION DOC]
    # Scenario: in-top-k in the math namespace with positional arguments.
    # Asserts they become predictions, targets, k and name keywords.
    text = "tf.math.in_top_k(a, b, c, n)"
    expected_text = (
        "tf.math.in_top_k(predictions=a, targets=b, k=c, name=n)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testDepthToSpace(self):
    # [FUNCTION DOC]
    # Scenario: depth-to-space with four positional arguments.
    # Asserts the last two become keywords.
    text = "tf.nn.depth_to_space(input, block_size, name, data_format)"
    expected_text = (
        "tf.nn.depth_to_space(input, block_size, name=name, "
        "data_format=data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testEmbeddingLookup(self):
    # [FUNCTION DOC]
    # Scenario: embedding lookup with six positional arguments.
    # Asserts the validate indices argument is dropped and others become keywords.
    text = ("tf.nn.embedding_lookup(params, ids, partition_strategy, name, "
            "validate_indices, max_norm)")
    expected_text = ("tf.nn.embedding_lookup(params, ids, "
                     "partition_strategy=partition_strategy, name=name, "
                     "max_norm=max_norm)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testEmbeddingLookupSparse(self):
    # [FUNCTION DOC]
    # Scenario: sparse embedding lookup with seven positional arguments.
    # Asserts the last four become keywords.
    text = ("tf.nn.embedding_lookup_sparse(params, sp_ids, sp_weights, "
            "partition_strategy, name, combiner, max_norm)")
    expected_text = ("tf.nn.embedding_lookup_sparse(params, sp_ids, "
                     "sp_weights, partition_strategy=partition_strategy, "
                     "name=name, combiner=combiner, max_norm=max_norm)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testNnInTopK(self):
    # [FUNCTION DOC]
    # Scenario: nn in-top-k with positional arguments.
    # Asserts all become keywords.
    text = "tf.nn.in_top_k(predictions, targets, k, name)"
    expected_text = ("tf.nn.in_top_k(predictions=predictions, "
                     "targets=targets, k=k, name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testSpaceToDepth(self):
    # [FUNCTION DOC]
    # Scenario: space-to-depth with four positional arguments.
    # Asserts the last two become keywords.
    text = "tf.nn.space_to_depth(input, block_size, name, data_format)"
    expected_text = ("tf.nn.space_to_depth(input, block_size, name=name, "
                     "data_format=data_format)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testPrint(self):
    # [FUNCTION DOC]
    # Scenario: print function calls preceded by the future import for print.
    # Asserts the text is unchanged (the print call is left alone).
    # tf.print() cannot be parsed unless we import print_function
    text = """from __future__ import print_function
tf.print()
tf.print('abc')
"""
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, text)  # Text should stay the same

  def testSparseSplit(self):
    # [FUNCTION DOC]
    # Scenario: sparse split in old and new spellings with split_dim.
    # Asserts rename to sparse split and split_dim to axis, with unchanged keywords left as-is.
    text = (
        "tf.sparse_split(sp_input=sp_input, num_split=num_split, axis=axis, "
        "name=name)")
    expected_text = (
        "tf.sparse.split(sp_input=sp_input, num_split=num_split, axis=axis, "
        "name=name)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = (
        "tf.sparse_split(sp_input=sp_input, num_split=num_split, "
        "name=name, split_dim=axis)")
    expected_text = (
        "tf.sparse.split(sp_input=sp_input, num_split=num_split, "
        "name=name, axis=axis)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = (
        "tf.sparse.split(sp_input=sp_input, num_split=num_split, "
        "name=name, split_dim=axis)")
    expected_text = (
        "tf.sparse.split(sp_input=sp_input, num_split=num_split, "
        "name=name, axis=axis)")
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testIterators(self):
    # [FUNCTION DOC]
    # Scenario: a table of dataset iterator creation calls: method style on arbitrary expressions,
    #   top-level data functions, and already-compat calls.
    # Asserts method style becomes the compat v1 data function taking the dataset first, and already
    #   compat or top-level forms keep arguments. Checks each pair.
    for (text, expected) in [
        ("(expr + yielding(data)).make_one_shot_iterator()",
         "tf.compat.v1.data.make_one_shot_iterator((expr + yielding(data)))"),
        ("dataset.make_one_shot_iterator()",
         "tf.compat.v1.data.make_one_shot_iterator(dataset)"),
        ("dataset.make_one_shot_iterator(shared_name=foo)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, shared_name=foo)"),
        ("dataset.make_one_shot_iterator(x, y, z)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, x, y, z)"),
        ("dataset.make_initializable_iterator()",
         "tf.compat.v1.data.make_initializable_iterator(dataset)"),
        ("ds.make_initializable_iterator(shared_name=foo)",
         "tf.compat.v1.data.make_initializable_iterator(ds, shared_name=foo)"),
        ("dataset.make_initializable_iterator(x, y, z)",
         "tf.compat.v1.data.make_initializable_iterator(dataset, x, y, z)"),
        ("tf.data.make_one_shot_iterator(dataset)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset)"),
        ("tf.data.make_one_shot_iterator(dataset, shared_name=foo)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, shared_name=foo)"),
        ("tf.data.make_one_shot_iterator(dataset, x, y, z)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, x, y, z)"),
        ("tf.data.make_initializable_iterator(dataset)",
         "tf.compat.v1.data.make_initializable_iterator(dataset)"),
        ("tf.data.make_initializable_iterator(ds, shared_name=foo)",
         "tf.compat.v1.data.make_initializable_iterator(ds, shared_name=foo)"),
        ("tf.data.make_initializable_iterator(dataset, x, y, z)",
         "tf.compat.v1.data.make_initializable_iterator(dataset, x, y, z)"),
        ("tf.compat.v1.data.make_one_shot_iterator(dataset)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset)"),
        ("tf.compat.v1.data.make_one_shot_iterator(dataset, shared_name=foo)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, shared_name=foo)"),
        ("tf.compat.v1.data.make_one_shot_iterator(dataset, x, y, z)",
         "tf.compat.v1.data.make_one_shot_iterator(dataset, x, y, z)"),
        ("tf.compat.v1.data.make_initializable_iterator(dataset)",
         "tf.compat.v1.data.make_initializable_iterator(dataset)"),
        ("tf.compat.v1.data.make_initializable_iterator(ds, shared_name=foo)",
         "tf.compat.v1.data.make_initializable_iterator(ds, shared_name=foo)"),
        ("tf.compat.v1.data.make_initializable_iterator(dataset, x, y, z)",
         "tf.compat.v1.data.make_initializable_iterator(dataset, x, y, z)")]:
      _, unused_report, unused_errors, actual = self._upgrade(text)
      self.assertEqual(actual, expected)

  def testStructure(self):
    # [FUNCTION DOC]
    # Scenario: names of data structure classes.
    # Asserts each maps to its spec class name.
    for (text, expected) in [
        ("tf.data.experimental.DatasetStructure", "tf.data.DatasetSpec"),
        ("tf.data.experimental.OptionalStructure", "tf.OptionalSpec"),
        ("tf.data.experimental.RaggedTensorStructure", "tf.RaggedTensorSpec"),
        ("tf.data.experimental.SparseTensorStructure", "tf.SparseTensorSpec"),
        ("tf.data.experimental.Structure", "tf.TypeSpec"),
        ("tf.data.experimental.TensorArrayStructure", "tf.TensorArraySpec"),
        ("tf.data.experimental.TensorStructure", "tf.TensorSpec"),
    ]:
      _, unused_report, unused_errors, actual = self._upgrade(text)
      self.assertEqual(actual, expected)

  def testMapAndBatch(self):
    # [FUNCTION DOC]
    # Scenario: map-and-batch with legacy function in the experimental data namespace.
    # Asserts it is redirected to compat v1.
    suffix = ".data.experimental.map_and_batch_with_legacy_function(args)"
    text = "tf" + suffix
    expected = "tf.compat.v1" + suffix
    _, unused_report, unused_errors, actual = self._upgrade(text)
    self.assertEqual(actual, expected)

  def testCast(self):
    # [FUNCTION DOC]
    # Scenario: legacy cast helpers for several types with a name keyword.
    # Asserts conversion to cast with a data type keyword (float and double map to 32 and 64 bit).
    for (name, dtype) in [("int32", "int32"),
                          ("int64", "int64"),
                          ("float", "float32"),
                          ("double", "float64"),
                          ("complex64", "complex64"),
                          ("complex128", "complex128"),
                          ("bfloat16", "bfloat16")]:
      text = "tf.to_%s(x, name='test')" % name
      expected_text = "tf.cast(x, name='test', dtype=tf.%s)" % dtype
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

  def testCastPositionalSecondArgument(self):
    # [FUNCTION DOC]
    # Scenario: the same cast helpers with name given positionally.
    # Asserts it becomes a name keyword and the data type keyword is added.
    for (name, dtype) in [("int32", "int32"),
                          ("int64", "int64"),
                          ("float", "float32"),
                          ("double", "float64"),
                          ("complex64", "complex64"),
                          ("complex128", "complex128"),
                          ("bfloat16", "bfloat16")]:
      text = "tf.to_%s(x, 'test')" % name
      expected_text = "tf.cast(x, name='test', dtype=tf.%s)" % dtype
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

  def testImageResize(self):
    # [FUNCTION DOC]
    # Scenario: four resize functions (bilinear, area, bicubic, nearest neighbor).
    # Asserts conversion to resize with the matching resize method constant.
    for method in ["bilinear", "area", "bicubic", "nearest_neighbor"]:
      text = "tf.image.resize_%s(i, s)" % method
      expected_text = ("tf.image.resize(i, s, "
                       "method=tf.image.ResizeMethod.%s)" % method.upper())
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

  def testImageResizeExtraPositionalArgs(self):
    # [FUNCTION DOC]
    # Scenario: the same resize functions with extra positional arguments.
    # Asserts the output contains the resize prefix, the preserve-aspect-ratio keyword and the
    #   method constant.
    for method in ["bilinear", "area", "bicubic", "nearest_neighbor"]:
      text = "tf.image.resize_%s(i, s, a, p)" % method
      expected_text = [
          "tf.image.resize(i, s, ", "preserve_aspect_ratio=p, ",
          "method=tf.image.ResizeMethod.%s)" % method.upper()
      ]
      _, unused_report, unused_errors, new_text = self._upgrade(text)
      for s in expected_text:
        self.assertIn(s, new_text)

  def testCond(self):
    # [FUNCTION DOC]
    # Scenario: cond with a deprecated positional argument.
    # Asserts the argument is dropped, name becomes keyword, and an error says the call requires
    #   manual check.
    text = "tf.cond(a, b, c, True, d)"
    expected_text = "tf.cond(a, b, c, name=d)"
    _, unused_report, errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)
    self.assertIn("tf.cond", errors[0])
    self.assertIn("requires manual check", errors[0])

  def testParens(self):
    # [FUNCTION DOC]
    # Scenario: a multi-line function source (held in a string) whose argument is a parenthesized
    #   expression.
    # Asserts the first argument becomes a keyword with parentheses and layout preserved.
    text = """
def _log_prob(self, x):
  return tf.debugging.assert_all_finite(
      (self.mixture_distribution.logits + self.distribution.log_prob(
          x[..., tf.newaxis])),
          message='Nans or Infs found')"""
    expected_text = """
def _log_prob(self, x):
  return tf.debugging.assert_all_finite(
      x=(self.mixture_distribution.logits + self.distribution.log_prob(
          x[..., tf.newaxis])),
          message='Nans or Infs found')"""
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testAssertStatements(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the list of names repeats two entries (assert less and assert greater), so those
    #   are checked twice.
    # Scenario: a list of assertion functions at top level and in the debugging namespace.
    # Asserts each is redirected to compat v1 and the report says it has been changed.
    for name in [
        "assert_greater", "assert_equal", "assert_none_equal", "assert_less",
        "assert_negative", "assert_positive", "assert_non_negative",
        "assert_non_positive", "assert_near", "assert_less",
        "assert_less_equal", "assert_greater", "assert_greater_equal",
        "assert_scalar"
    ]:
      text = "tf.%s(a)" % name
      expected_text = "tf.compat.v1.%s(a)" % name
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)
      self.assertIn("%s has been" % name, report)

      text = "tf.debugging.%s(a)" % name
      expected_text = "tf.compat.v1.debugging.%s(a)" % name
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)
      self.assertIn("%s has been" % name, report)

  def testAssertRankStatements(self):
    # [FUNCTION DOC]
    # Scenario: the three rank assertion functions at top level and in debugging.
    # Asserts redirect to compat v1 and report notes.
    for name in ["assert_rank", "assert_rank_at_least", "assert_rank_in"]:
      text = "tf.%s(a)" % name
      expected_text = "tf.compat.v1.%s(a)" % name
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)
      self.assertIn("%s has been" % name, report)

      text = "tf.debugging.%s(a)" % name
      expected_text = "tf.compat.v1.debugging.%s(a)" % name
      _, report, unused_errors, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)
      self.assertIn("%s has been" % name, report)

  def test_assert_equal_graph_def(self):
    # [FUNCTION DOC]
    # Scenario: graph-def equality assertion with removed keywords.
    # Asserts removed arguments dropped and a and b renamed to actual and expected.
    text = ("tf.test.assert_equal_graph_def(a, b, checkpoint_v2=x, "
            "hash_table_shared_name=y)")
    expected = "tf.test.assert_equal_graph_def(actual=a, expected=b)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_is_tensor_upgrade(self):
    # [FUNCTION DOC]
    # Scenario: contrib framework is-tensor.
    # Asserts it maps to the top-level is-tensor.
    text = "tf.contrib.framework.is_tensor(x)"
    expected = "tf.is_tensor(x)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_is_tensor_direct_import_upgrade(self):
    # [FUNCTION DOC]
    # Scenario: the direct-import alias form of is-tensor.
    # Asserts the same mapping.
    text = "contrib_framework.is_tensor(x)"
    expected = "tf.is_tensor(x)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_CriticalSection_upgrade(self):
    # [FUNCTION DOC]
    # Scenario: contrib CriticalSection with a shared name.
    # Asserts it maps to the top-level class.
    text = "tf.contrib.framework.CriticalSection(shared_name='blah')"
    expected = "tf.CriticalSection(shared_name='blah')"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_sample_distorted_bounding_box(self):
    # [FUNCTION DOC]
    # Scenario: bounding box sampling with ten positional arguments.
    # Asserts the first three stay positional, the seed-related ones are removed and the rest become
    #   keywords.
    text = ("tf.image.sample_distorted_bounding_box(a, b, c, d, e, f, g, h, i, "
            "j)")
    expected = ("tf.image.sample_distorted_bounding_box(a, b, c, "
                "min_object_covered=e, aspect_ratio_range=f, area_range=g, "
                "max_attempts=h, use_image_if_no_bounding_boxes=i, name=j)")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_initialize(self):
    # [FUNCTION DOC]
    # Scenario: contrib summary initialize reference.
    # Asserts it maps to the summary initialize in compat v1.
    text = "tf.contrib.summary.initialize"
    expected = "tf.compat.v1.summary.initialize"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_framework_argsort(self):
    # [FUNCTION DOC]
    # Scenario: contrib argsort reference.
    # Asserts it maps to top-level argsort.
    text = "tf.contrib.framework.argsort"
    expected = "tf.argsort"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_flags_bare(self):
    # [FUNCTION DOC]
    # Scenario: bare flags module reference.
    # Asserts the first error says flags have been removed.
    _, _, errors, _ = self._upgrade("tf.flags")
    self.assertIn("tf.flags and tf.app.flags have been removed", errors[0])

  def test_flags_flags(self):
    # [FUNCTION DOC]
    # Scenario: the FLAGS object reference.
    # Asserts the same removal error.
    _, _, errors, _ = self._upgrade("tf.flags.FLAGS")
    self.assertIn("tf.flags and tf.app.flags have been removed", errors[0])

  def test_contrib_layers_layer_norm_deprecation(self):
    # [FUNCTION DOC]
    # Scenario: layer norm through two contrib aliases.
    # Asserts the report notes deprecation.
    for contrib_alias in ["tf.contrib.", "contrib_"]:
      _, report, _, _ = self._upgrade(contrib_alias + "layers.layer_norm")
      self.assertIn(
          "`tf.contrib.layers.layer_norm` has been deprecated", report)

  def test_contrib_rnn_deprecation(self):
    # [FUNCTION DOC]
    # Scenario: contrib rnn module reference.
    # Asserts the report notes deprecation.
    _, report, _, _ = self._upgrade("tf.contrib.rnn")
    self.assertIn("tf.contrib.rnn.* has been deprecated", report)

  def test_contrib_cudnn_rnn_deprecation(self):
    # [FUNCTION DOC]
    # Scenario: contrib cudnn rnn reference.
    # Asserts the report notes deprecation.
    _, report, _, _ = self._upgrade("tf.contrib.cudnn_rnn")
    self.assertIn("tf.contrib.cudnn_rnn.* has been deprecated", report)

  def test_max_pool_2d(self):
    # [FUNCTION DOC]
    # Scenario: max pool with value keyword.
    # Asserts rename to the 2D function with input keyword.
    text = "tf.nn.max_pool(value=4)"
    expected_text = "tf.nn.max_pool2d(input=4)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_contrib_rnn_cell(self):
    # [FUNCTION DOC]
    # Scenario: a list of contrib rnn cell classes.
    # Asserts each maps to the compat v1 nn rnn cell namespace.
    api_symbols = ["RNNCell", "BasicLSTMCell", "BasicRNNCell", "GRUCell",
                   "LSTMCell", "MultiRNNCell"]
    for symbol in api_symbols:
      text = "tf.contrib.rnn." + symbol
      expected_text = "tf.compat.v1.nn.rnn_cell." + symbol
      _, _, _, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

  def test_contrib_rnn_function(self):
    # [FUNCTION DOC]
    # Scenario: static rnn functions in contrib.
    # Asserts each maps to compat v1 nn.
    api_symbols = ["static_rnn", "static_state_saving_rnn",
                   "static_bidirectional_rnn"]
    for symbol in api_symbols:
      text = "tf.contrib.rnn." + symbol
      expected_text = "tf.compat.v1.nn." + symbol
      _, _, _, new_text = self._upgrade(text)
      self.assertEqual(expected_text, new_text)

  def test_contrib_summary_generic(self):
    # [FUNCTION DOC]
    # Scenario: generic contrib summary with name, value, metadata, family and step.
    # Asserts conversion to the v2 summary write function with keywords, and three errors in
    #   alphabetical argument order (family, name, generic note).
    text = "tf.contrib.summary.generic('foo', myval, meta, 'fam', 42)"
    expected = ("tf.compat.v2.summary.write(tag='foo', data=myval, "
                "metadata=meta, step=42)")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    # Arg errors come in alphabetical order of arguments, not appearance order.
    self.assertIn("'family' argument", errors[0])
    self.assertIn("'name' argument", errors[1])
    self.assertIn("tf.compat.v2.summary.*", errors[2])

  def test_contrib_summary_audio(self):
    # [FUNCTION DOC]
    # Scenario: contrib audio summary with all arguments.
    # Asserts conversion to v2 audio with keywords and errors about family and the general note.
    text = "tf.contrib.summary.audio('foo', myval, 44100, 3, 'fam', 42)"
    expected = ("tf.compat.v2.summary.audio(name='foo', data=myval, "
                "sample_rate=44100, max_outputs=3, step=42)")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'family' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_histogram(self):
    # [FUNCTION DOC]
    # Scenario: contrib histogram summary.
    # Asserts conversion to v2 histogram, family dropped with an error, and the general note.
    text = "tf.contrib.summary.histogram('foo', myval, 'fam', 42)"
    expected = ("tf.compat.v2.summary.histogram(name='foo', data=myval, "
                "step=42)")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'family' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_image(self):
    # [FUNCTION DOC]
    # Scenario: contrib image summary with bad color, max outputs, family and step.
    # Asserts conversion to v2 image and three errors (bad color, family, general note).
    text = "tf.contrib.summary.image('foo', myval, red, 3, 'fam', 42)"
    expected = ("tf.compat.v2.summary.image(name='foo', data=myval, "
                "max_outputs=3, step=42)")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'bad_color' argument", errors[0])
    self.assertIn("'family' argument", errors[1])
    self.assertIn("tf.compat.v2.summary.*", errors[2])

  def test_contrib_summary_scalar(self):
    # [FUNCTION DOC]
    # Scenario: contrib scalar summary.
    # Asserts conversion to v2 scalar and errors on family and the general note.
    text = "tf.contrib.summary.scalar('foo', myval, 'fam', 42)"
    expected = ("tf.compat.v2.summary.scalar(name='foo', data=myval, "
                "step=42)")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'family' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_generic_nostep(self):
    # [FUNCTION DOC]
    # Scenario: generic summary without a step.
    # Asserts a global-step default is added, with errors for name, step and the general note.
    text = "tf.contrib.summary.generic('foo', myval)"
    expected = ("tf.compat.v2.summary.write(tag='foo', data=myval, "
                "step=tf.compat.v1.train.get_or_create_global_step())")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'name' argument", errors[0])
    self.assertIn("'step' argument", errors[1])
    self.assertIn("tf.compat.v2.summary.*", errors[2])

  def test_contrib_summary_audio_nostep(self):
    # [FUNCTION DOC]
    # Scenario: audio summary without a step.
    # Asserts a default global step and errors for step and the general note.
    text = "tf.contrib.summary.audio('foo', myval, 44100)"
    expected = ("tf.compat.v2.summary.audio(name='foo', data=myval, "
                "sample_rate=44100, "
                "step=tf.compat.v1.train.get_or_create_global_step())")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'step' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_histogram_nostep(self):
    # [FUNCTION DOC]
    # Scenario: histogram summary without a step.
    # Same assertions as the audio variant.
    text = "tf.contrib.summary.histogram('foo', myval)"
    expected = ("tf.compat.v2.summary.histogram(name='foo', data=myval, "
                "step=tf.compat.v1.train.get_or_create_global_step())")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'step' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_image_nostep(self):
    # [FUNCTION DOC]
    # Scenario: image summary without a step.
    # Same assertions as the audio variant.
    text = "tf.contrib.summary.image('foo', myval)"
    expected = ("tf.compat.v2.summary.image(name='foo', data=myval, "
                "step=tf.compat.v1.train.get_or_create_global_step())")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'step' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_scalar_nostep(self):
    # [FUNCTION DOC]
    # Scenario: scalar summary without a step.
    # Same assertions as the audio variant.
    text = "tf.contrib.summary.scalar('foo', myval)"
    expected = ("tf.compat.v2.summary.scalar(name='foo', data=myval, "
                "step=tf.compat.v1.train.get_or_create_global_step())")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'step' argument", errors[0])
    self.assertIn("tf.compat.v2.summary.*", errors[1])

  def test_contrib_summary_graph(self):
    # [FUNCTION DOC]
    # Scenario: contrib summary graph.
    # Asserts the first error mentions the v2 trace function.
    text = "tf.contrib.summary.graph(my_graph)"
    _, _, errors, _ = self._upgrade(text)
    expected_error = "tf.compat.v2.summary.trace"
    self.assertIn(expected_error, errors[0])

  def test_contrib_summary_import_event(self):
    # [FUNCTION DOC]
    # Scenario: contrib summary import event.
    # Asserts the first error mentions the raw proto write function.
    text = "tf.contrib.summary.import_event(my_event)"
    _, _, errors, _ = self._upgrade(text)
    expected_error = "tf.compat.v2.summary.experimental.write_raw_pb"
    self.assertIn(expected_error, errors[0])

  def test_contrib_summary_flush(self):
    # [FUNCTION DOC]
    # Scenario: contrib summary flush with writer keyword.
    # Asserts it maps to v2 flush.
    text = "tf.contrib.summary.flush(writer=foo)"
    expected = "tf.compat.v2.summary.flush(writer=foo)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_summary_create_file_writer(self):
    # [FUNCTION DOC]
    # Scenario: contrib file writer creation with five positional arguments.
    # Asserts keywords are added, the name argument is dropped with an error, and a second error
    #   says event files are no longer reused.
    text = ("tf.contrib.summary.create_file_writer('my_logdir', 0, 1000, "
            "'.foo', 'shared-name')")
    expected = ("tf.compat.v2.summary.create_file_writer(logdir='my_logdir', "
                "max_queue=0, flush_millis=1000, filename_suffix='.foo')")
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("'name' argument", errors[0])
    self.assertIn("no longer re-uses existing event files", errors[1])

  def test_contrib_summary_always_record_summaries(self):
    # [FUNCTION DOC]
    # Scenario: always-record call.
    # Asserts it becomes record-if with true.
    text = "tf.contrib.summary.always_record_summaries()"
    expected = "tf.compat.v2.summary.record_if(True)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_summary_never_record_summaries(self):
    # [FUNCTION DOC]
    # Scenario: never-record call.
    # Asserts it becomes record-if with false.
    text = "tf.contrib.summary.never_record_summaries()"
    expected = "tf.compat.v2.summary.record_if(False)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_summary_record_summaries_every_n_global_steps(self):
    # [FUNCTION DOC]
    # Scenario: every-n-steps recording.
    # Asserts an error saying it must be replaced by a record-if call.
    text = "tf.contrib.summary.record_summaries_every_n_global_steps(10)"
    _, _, errors, _ = self._upgrade(text)
    expected_error = "replaced by a call to tf.compat.v2.summary.record_if()"
    self.assertIn(expected_error, errors[0])

  def test_contrib_summary_all_summary_ops(self):
    # [FUNCTION DOC]
    # Scenario: all-summary-ops call.
    # Asserts it maps to the compat v1 all-v2-summary-ops function.
    text = "tf.contrib.summary.all_summary_ops()"
    expected = "tf.compat.v1.summary.all_v2_summary_ops()"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_contrib_summary_full_example(self):
    # [FUNCTION DOC]
    # Scenario: a multi-line program (held in a string, indented then stripped with a small local
    #   helper) using eager mode, a file writer, a context manager with record-always, and scalar,
    #   histogram and flush calls.
    # Asserts the whole converted program matches the expected text, including default global step
    #   for the scalar.
    deindent = lambda n, s: "\n".join(line[n:] for line in s.split("\n"))
    text = deindent(4, """
    import tensorflow as tf
    tf.enable_eager_execution()
    writer = tf.contrib.summary.create_file_writer(
        "/tmp/migration_test", flush_millis=1000)
    with writer.as_default(), tf.contrib.summary.always_record_summaries():
      tf.contrib.summary.scalar("loss", 0.42)
      tf.contrib.summary.histogram("weights", [1.0, 2.0], step=7)
      tf.contrib.summary.flush()
    """)
    expected = deindent(4, """
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    writer = tf.compat.v2.summary.create_file_writer(
        logdir="/tmp/migration_test", flush_millis=1000)
    with writer.as_default(), tf.compat.v2.summary.record_if(True):
      tf.compat.v2.summary.scalar(name="loss", data=0.42, step=tf.compat.v1.train.get_or_create_global_step())
      tf.compat.v2.summary.histogram(name="weights", data=[1.0, 2.0], step=7)
      tf.compat.v2.summary.flush()
    """)
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_summary_api_warning(self):
    # [FUNCTION DOC]
    # Scenario: a v1 summary scalar call.
    # Asserts the report says the v1 summary API cannot be migrated automatically.
    text = "tf.summary.scalar('foo', 42)"
    _, report, _, _ = self._upgrade(text)
    expected_info = "TF 1.x summary API cannot be automatically migrated"
    self.assertIn(expected_info, report)

  def test_avg_pool_2d(self):
    # [FUNCTION DOC]
    # Scenario: average pool with value keyword.
    # Asserts rename to the 2D function with input keyword.
    text = "tf.nn.avg_pool(value=4)"
    expected_text = "tf.nn.avg_pool2d(input=4)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_saved_model_load(self):
    # [FUNCTION DOC]
    # Scenario: saved model load with a session.
    # Asserts redirect to compat v1 and a report note that load works differently.
    text = "tf.saved_model.load(sess, ['foo_graph'])"
    expected = "tf.compat.v1.saved_model.load(sess, ['foo_graph'])"
    _, report, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    expected_info = "tf.saved_model.load works differently in 2.0"
    self.assertIn(expected_info, report)

  def test_saved_model_loader_load(self):
    # [FUNCTION DOC]
    # Scenario: loader-namespace load.
    # Asserts redirect to compat v1 load and the same note.
    text = "tf.saved_model.loader.load(sess, ['foo_graph'])"
    expected = "tf.compat.v1.saved_model.load(sess, ['foo_graph'])"
    _, report, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    expected_info = "tf.saved_model.load works differently in 2.0"
    self.assertIn(expected_info, report)

  def test_saved_model_load_v2(self):
    # [FUNCTION DOC]
    # Scenario: load-v2.
    # Asserts it becomes the compat v2 load.
    text = "tf.saved_model.load_v2('/tmp/blah')"
    expected = "tf.compat.v2.saved_model.load('/tmp/blah')"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_app_flags(self):
    # [FUNCTION DOC]
    # Scenario: assignment of the app flags module.
    # Asserts it goes to compat v1.
    text = "flags = tf.app.flags"
    expected = "flags = tf.compat.v1.app.flags"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_uniform_unit_scaling_initializer(self):
    # [FUNCTION DOC]
    # Scenario: uniform unit scaling initializer in two spellings.
    # Asserts both become Keras variance scaling with a uniform distribution.
    text = "tf.uniform_unit_scaling_initializer(0.5)"
    expected_text = ("tf.compat.v1.keras.initializers.VarianceScaling("
                     "scale=0.5, distribution=\"uniform\")")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.initializers.uniform_unit_scaling(0.5)"
    expected_text = ("tf.compat.v1.keras.initializers.VarianceScaling("
                     "scale=0.5, distribution=\"uniform\")")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_name_scope(self):
    # [FUNCTION DOC]
    # Scenario: name scope with the old arguments in several forms.
    # Asserts positional and default-name forms become a name keyword, a call with both name and
    #   default name goes to compat v1 with a report note, and a call with neither name nor default
    #   name raises an error entry.
    text = "tf.name_scope(None, default_name, [some, values])"
    expected_text = "tf.name_scope(name=default_name)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.name_scope(default_name=default_name, values=stuff)"
    expected_text = "tf.name_scope(name=default_name)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.name_scope(name=n, default_name=d, values=s)"
    expected_text = "tf.compat.v1.name_scope(name=n, default_name=d, values=s)"
    _, report, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)
    self.assertIn("`name` passed to `name_scope`", report)

    text = "tf.name_scope(name=None, values=stuff)"
    _, _, errors, _ = self._upgrade(text)
    self.assertIn("name_scope call with neither name nor default_name",
                  errors[0])

  @parameterized.parameters(
      # Rename parameter: delimiter -> sep and add .to_sparse()
      ["tf.string_split('test', delimiter=' ')",
       "tf.strings.split(input='test', sep=' ').to_sparse()"],
      # Rename parameter: source -> input
      ["tf.strings.split(source='test1')",
       "tf.strings.split(input='test1').to_sparse()"],
      # Use compat.v1 for skip_empty parameter.
      ["tf.string_split('test', ' ', True)",
       "tf.compat.v1.string_split(source='test', sep=' ', skip_empty=True)"],
      ["tf.string_split('test', ' ', skip_empty=False)",
       "tf.strings.split(input='test', sep=' ').to_sparse()"],
      # Split behavior for sep=None changed.  (In particular, it now splits on
      # all whitespace, not just the space character)
      ["tf.string_split(x)",
       "tf.compat.v1.string_split(source=x)"],
      # Split behavior for sep='' changed:
      ["tf.string_split(x, '')",
       "tf.strings.bytes_split(input=x).to_sparse()"],
      ["tf.string_split(x, sep='')",
       "tf.strings.bytes_split(input=x).to_sparse()"],
      ["tf.string_split(x, delimiter='')",
       "tf.strings.bytes_split(input=x).to_sparse()"],
      ["tf.string_split(x, '', result_type='RaggedTensor')",
       "tf.strings.bytes_split(input=x)"],
      # If sep is a variable, we can't tell if it's empty:
      ["tf.string_split(x, sep)",
       "tf.compat.v1.string_split(source=x, sep=sep)"],
      # If sep is a non-empty string literal, then we don't need compat.v1.
      ["tf.string_split(x, 'non-empty-sep')",
       "tf.strings.split(input=x, sep='non-empty-sep').to_sparse()"],
      # Add to_sparse unless result_type is RaggedTensor:
      ["tf.string_split(x, ' ')",
       "tf.strings.split(input=x, sep=' ').to_sparse()"],
      ["tf.string_split(x, ' ', result_type='SparseTensor')",
       "tf.strings.split(input=x, sep=' ').to_sparse()"],
      ["tf.string_split(x, ' ', result_type='RaggedTensor')",
       "tf.strings.split(input=x, sep=' ')"],
      ["tf.string_split(x, ' ', result_type=x)",
       "tf.compat.v1.string_split(source=x, sep=' ', result_type=x)"],
  )  # pyformat: disable
  # TODO(b/129398290)
  def DISABLED_test_string_split(self, text, expected_text):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name does not start with the test prefix, so the runner never executes it;
    #   the TODO shows it is intentionally disabled.
    # Data-driven test (parameters: input text and expected text) for string splitting conversions:
    #   renames, empty separator to bytes split, compat v1 fallback for unknown separator or result
    #   type, and adding a sparse conversion unless the result type is ragged.
    # Would assert the upgraded text equals the expected text.
    """Tests for transforming from tf.string_split."""
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  @parameterized.parameters(
      # Add to_sparse unless result_type is RaggedTensor:
      ["tf.strings.split(x, sep)",
       "tf.strings.split(x, sep).to_sparse()"],
      ["tf.strings.split(x, sep, result_type='SparseTensor')",
       "tf.strings.split(x, sep).to_sparse()"],
      ["tf.strings.split(x, sep, result_type='RaggedTensor')",
       "tf.strings.split(x, sep)"],
      ["tf.strings.split(x, sep, result_type=x)",
       "tf.compat.v1.strings.split(x, sep, result_type=x)"],
  )  # pyformat: disable
  def test_strings_split(self, text, expected_text):
    # [FUNCTION DOC]
    # Data-driven test (parameters: input text and expected text) for splitting in the strings
    #   namespace.
    # Asserts a sparse conversion is appended unless the result type is ragged, and a non-literal
    #   result type falls back to compat v1.
    """Tests for transforming from tf.strings.split."""
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_sdca_to_raw_ops(self):
    # [FUNCTION DOC]
    # Scenario: SDCA train functions (fingerprint, shrink, optimizer).
    # Asserts each is replaced by its raw op with unchanged arguments.
    text = "tf.train.sdca_fprint(input_tensor)"
    expected_text = "tf.raw_ops.SdcaFprint(input_tensor)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.train.sdca_fprint(input, name=n)"
    expected_text = "tf.raw_ops.SdcaFprint(input, name=n)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = "tf.train.sdca_shrink_l1(w, l, ll)"
    expected_text = "tf.raw_ops.SdcaShrinkL1(w, l, ll)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

    text = (
        "tf.train.sdca_optimizer(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)")
    expected_text = (
        "tf.raw_ops.SdcaOptimizer(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o)")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_contrib_to_addons_move(self):
    # [FUNCTION DOC]
    # Scenario: four contrib layer functions with a mapping to addons replacements.
    # Asserts the report mentions each replacement name.
    small_mapping = {
        "tf.contrib.layers.poincare_normalize":
            "tfa.layers.PoincareNormalize",
        "tf.contrib.layers.maxout":
            "tfa.layers.Maxout",
        "tf.contrib.layers.group_norm":
            "tfa.layers.GroupNormalization",
        "tf.contrib.layers.instance_norm":
            "tfa.layers.InstanceNormalization",
    }
    for symbol, replacement in small_mapping.items():
      text = "{}('stuff', *args, **kwargs)".format(symbol)
      _, report, _, _ = self._upgrade(text)
      self.assertIn(replacement, report)

  def testXlaExperimental(self):
    # [FUNCTION DOC]
    # Scenario: jit scope and compile in the xla experimental namespace.
    # Asserts they are left unchanged.
    text = "tf.xla.experimental.jit_scope(0)"
    expected_text = "tf.xla.experimental.jit_scope(0)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    text = "tf.xla.experimental.compile(0)"
    expected_text = "tf.xla.experimental.compile(0)"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testNnErosion2d(self):
    # [FUNCTION DOC]
    # Scenario: erosion with five positional arguments.
    # Asserts the data format keyword is appended.
    text = "tf.nn.erosion2d(v, k, s, r, p)"
    expected_text = "tf.nn.erosion2d(v, k, s, r, p, data_format='NHWC')"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testNnDilation2d(self):
    # [FUNCTION DOC]
    # Scenario: dilation with five positional arguments.
    # Asserts the data format keyword is appended.
    text = "tf.nn.dilation2d(v, k, s, r, p)"
    expected_text = "tf.nn.dilation2d(v, k, s, r, p, data_format='NHWC')"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

  def testPywrapTensorflowWarning(self):
    # [FUNCTION DOC]
    # Scenario: use of the pywrap module.
    # Asserts the text is unchanged and the first error says it will not be distributed.
    text = "tf.pywrap_tensorflow.foo()"
    expected = "tf.pywrap_tensorflow.foo()"
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("`tf.pywrap_tensorflow` will not be distributed", errors[0])

  def testKerasSaveModelFormat(self):
    # [FUNCTION DOC]
    # Scenario: Keras save-model function call, then a plain model save.
    # Asserts the first gets a save format of h5 and no default-format note, and the second report
    #   has the note about the SavedModel default.
    text = "tf.keras.models.save_model(model, path)"
    expected_text = "tf.keras.models.save_model(model, path, save_format='h5')"
    _, report, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertNotIn(
        "saves to the Tensorflow SavedModel format by default", report)

    _, report, _, _ = self._upgrade("model.save(path)")
    self.assertIn(
        "saves to the Tensorflow SavedModel format by default", report)

  def test_distribute_strategy(self):
    # [FUNCTION DOC]
    # Scenario: contrib distribute names and the new strategy name.
    # Asserts a cross device ops reference is moved, mirrored and TPU strategies are left with
    #   migration errors, the new mirrored strategy gets an API-changed report, and other contrib
    #   distribute names get a general migration note.
    text = "tf.contrib.distribute.CrossDeviceOps()"
    expected = "tf.distribute.CrossDeviceOps()"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

    text = "tf.contrib.distribute.MirroredStrategy"
    expected = "tf.contrib.distribute.MirroredStrategy"
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("migrated to tf.distribute.MirroredStrategy", errors[0])

    text = "tf.distribute.MirroredStrategy"
    expected = "tf.distribute.MirroredStrategy"
    _, report, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("tf.distribute.MirroredStrategy API has changed", report)
    self.assertIn("make_dataset_iterator->experimental_distribute_dataset",
                  report)

    text = "tf.contrib.distribute.TPUStrategy"
    expected = "tf.contrib.distribute.TPUStrategy"
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("migrated to tf.distribute.TPUStrategy",
                  errors[0])

    text = "tf.contrib.distribute.foo"
    expected = "tf.contrib.distribute.foo"
    _, report, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)
    self.assertIn("tf.contrib.distribute.* have been migrated", report)

  def test_decode_raw(self):
    # [FUNCTION DOC]
    # Scenario: raw decode with bytes keyword.
    # Asserts it is renamed to input bytes.
    text = "tf.io.decode_raw(bytes=[1,2,3], output_dtype=tf.int32)"
    expected_text = (
        "tf.io.decode_raw(input_bytes=[1,2,3], output_dtype=tf.int32)")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def testRecomputeGrad(self):
    # [FUNCTION DOC]
    # Scenario: contrib recompute grad.
    # Asserts it maps to the top-level function.
    text = "tf.contrib.layers.recompute_grad()"
    expected = "tf.recompute_grad()"
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected, new_text)

  def test_load_variable(self):
    # [FUNCTION DOC]
    # Scenario: contrib load variable, positional and with checkpoint dir keyword.
    # Asserts mapping to the train function with keyword renamed.
    text = "tf.contrib.framework.load_variable('a')"
    expected_text = (
        "tf.train.load_variable('a')")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)
    text = "tf.contrib.framework.load_variable(checkpoint_dir='a')"
    expected_text = (
        "tf.train.load_variable(ckpt_dir_or_file='a')")
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(expected_text, new_text)

  def test_import_rename_analysis(self):
    # [FUNCTION DOC]
    # Scenario: several import styles combined with the rename option: plain, multi-name, alongside
    #   compat v1 and v2 aliases, from-imports, star imports and submodule imports, plus the compat
    #   v1 upgrade option with and without renaming.
    # Asserts import lines are rewritten to the compat v2 module when appropriate and left alone for
    #   explicit compat aliases, and the old symbol is renamed.
    old_symbol = "tf.conj(a)"
    new_symbol = "tf.math.conj(a)"

    import_header = "import tensorflow as tf\n"
    text = import_header + old_symbol
    expected_text = "import tensorflow.compat.v2 as tf\n" + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(
        text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = "import tensorflow as tf, other_import as y\n"
    text = import_header + old_symbol
    new_import_header = "import tensorflow.compat.v2 as tf, other_import as y\n"
    expected_text = new_import_header + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(
        text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = ("import tensorflow as tf\n"
                     "import tensorflow.compat.v1 as tf_v1\n"
                     "import tensorflow.compat.v2 as tf_v2\n")
    text = import_header + old_symbol
    expected_header = ("import tensorflow.compat.v2 as tf\n"
                       "import tensorflow.compat.v1 as tf_v1\n"
                       "import tensorflow.compat.v2 as tf_v2\n")
    expected_text = expected_header + new_symbol
    _, _, _, new_text = self._upgrade(text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = ("import tensorflow.compat.v1 as tf\n"
                     "import tensorflow.compat.v1 as tf_v1\n"
                     "import tensorflow.compat.v2 as tf_v2\n")
    text = import_header + old_symbol
    expected_header = ("import tensorflow.compat.v2 as tf\n"
                       "import tensorflow.compat.v1 as tf_v1\n"
                       "import tensorflow.compat.v2 as tf_v2\n")
    expected_text = expected_header + new_symbol
    _, _, _, new_text = self._upgrade(
        text, import_rename=True, upgrade_compat_v1_import=True)
    self.assertEqual(new_text, expected_text)

    import_header = ("import tensorflow.compat.v1 as tf\n"
                     "import tensorflow.compat.v1 as tf_v1\n"
                     "import tensorflow.compat.v2 as tf_v2\n")
    text = import_header + old_symbol
    expected_header = ("import tensorflow as tf\n"
                       "import tensorflow.compat.v1 as tf_v1\n"
                       "import tensorflow.compat.v2 as tf_v2\n")
    expected_text = expected_header + new_symbol
    _, _, _, new_text = self._upgrade(
        text, import_rename=False, upgrade_compat_v1_import=True)
    self.assertEqual(new_text, expected_text)

    import_header = "from tensorflow import foo\n"
    text = import_header + old_symbol
    expected_text = "from tensorflow.compat.v2 import foo\n" + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(
        text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = "from tensorflow import *\n"
    text = import_header + old_symbol
    expected_text = "from tensorflow.compat.v2 import *\n" + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(
        text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = "from tensorflow.foo import bar\n"
    text = import_header + old_symbol
    expected_text = "from tensorflow.compat.v2.foo import bar\n" + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(
        text, import_rename=True)
    self.assertEqual(new_text, expected_text)

    import_header = ("from tensorflow import foo as tf\n"
                     "from tensorflow.compat import v1 as tf_v1\n"
                     "from tensorflow.compat import v2 as tf_v2\n")
    text = import_header + old_symbol
    expected_header = ("from tensorflow.compat.v2 import foo as tf\n"
                       "from tensorflow.compat import v1 as tf_v1\n"
                       "from tensorflow.compat import v2 as tf_v2\n")
    expected_text = expected_header + new_symbol
    _, _, _, new_text = self._upgrade(text, import_rename=True)
    self.assertEqual(new_text, expected_text)

  def test_import_analysis(self):
    # [FUNCTION DOC]
    # Scenario: import styles without the rename option.
    # Asserts plain tensorflow aliased as tf is upgraded; unaliased import gives an error and no
    #   change; explicit compat v1 or v2 imports as tf give a report note, no change and no errors.
    old_symbol = "tf.conj(a)"
    new_symbol = "tf.math.conj(a)"

    # We upgrade the base un-versioned tensorflow aliased as tf
    import_header = "import tensorflow as tf\n"
    text = import_header + old_symbol
    expected_text = import_header + new_symbol
    _, unused_report, unused_errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    import_header = ("import tensorflow as tf\n"
                     "import tensorflow.compat.v1 as tf_v1\n"
                     "import tensorflow.compat.v2 as tf_v2\n")
    text = import_header + old_symbol
    expected_text = import_header + new_symbol
    _, _, _, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)

    # We don't handle unaliased tensorflow imports currently,
    # So the upgrade script show log errors
    import_header = "import tensorflow\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, _, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("unaliased `import tensorflow`", "\n".join(errors))

    # Upgrading explicitly-versioned tf code is unsafe, but we don't
    # need to throw errors when we detect explicitly-versioned tf.
    import_header = "import tensorflow.compat.v1 as tf\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("`tensorflow.compat.v1` was directly imported as `tf`",
                  report)
    self.assertEmpty(errors)

    import_header = "from tensorflow.compat import v1 as tf\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("`tensorflow.compat.v1` was directly imported as `tf`",
                  report)
    self.assertEmpty(errors)

    import_header = "from tensorflow.compat import v1 as tf, v2 as tf2\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("`tensorflow.compat.v1` was directly imported as `tf`",
                  report)
    self.assertEmpty(errors)

    import_header = "import tensorflow.compat.v2 as tf\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("`tensorflow.compat.v2` was directly imported as `tf`",
                  report)
    self.assertEmpty(errors)

    import_header = "from tensorflow.compat import v1 as tf1, v2 as tf\n"
    text = import_header + old_symbol
    expected_text = import_header + old_symbol
    _, report, errors, new_text = self._upgrade(text)
    self.assertEqual(new_text, expected_text)
    self.assertIn("`tensorflow.compat.v2` was directly imported as `tf`",
                  report)
    self.assertEmpty(errors)

  @parameterized.parameters(
      [False,
       "import tensorflow.compat.v2 as tf\ntf.conj(a)",
       "import tensorflow.compat.v2 as tf\ntf.conj(a)",
       "tf.conj(a)",
       "tf.math.conj(a)"],
      [False,
       "import tensorflow.compat.v2 as tf\ntf.to_int32(x)",
       "import tensorflow.compat.v2 as tf\ntf.to_int32(x)",
       "tf.to_int32(x)",
       "tf.cast(x, dtype=tf.int32)"],
      # Verify that upgrade_compat_v1_import option persists between files
      [True,
       "import tensorflow.compat.v1 as tf\ntf.conj(a)",
       "import tensorflow.compat.v2 as tf\ntf.math.conj(a)",
       "import tensorflow.compat.v1 as tf\ntf.to_int32(x)",
       "import tensorflow.compat.v2 as tf\ntf.cast(x, dtype=tf.int32)"],
  )  # pyformat: disable
  def test_api_spec_reset_between_files(self,
                                        upgrade_compat_v1_import,
                                        text_a, expected_text_a,
                                        text_b, expected_text_b):
    # [FUNCTION DOC]
    # Data-driven test (parameters: the compat v1 upgrade flag, two source texts and their expected
    #   outputs) that processes two files with one upgrader.
    # Asserts each output matches, showing that the API spec state is reset between files while the
    #   option persists.
    results = self._upgrade_multiple(upgrade_compat_v1_import, [text_a, text_b])
    result_a, result_b = results[0], results[1]
    self.assertEqual(result_a[3], expected_text_a)
    self.assertEqual(result_b[3], expected_text_b)

  def test_keras_experimental_export_warning(self):
    # [FUNCTION DOC]
    # Scenario: Keras experimental export of a saved model.
    # Asserts the report suggests using model save.
    text = "tf.keras.experimental.export_saved_model"
    _, report, _, _ = self._upgrade(text)
    expected_info = "Please use model.save"
    self.assertIn(expected_info, report)


class TestUpgradeFiles(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests of the upgrader's file-level entry point. Each test writes a temporary file, upgrades it
  #   in place and checks the contents. No shared fixtures.

  def testInplace(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the file is read with a handle that is never closed and the temporary file is
    #   only deleted when the assertion passes.
    # Scenario: a temporary file holding an old call, processed with the same path as input and
    #   output.
    # Asserts the file ends up with the upgraded text (no race), then deletes it.
    """Check to make sure we don't have a file system race."""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "tf.conj(a)\n"
    upgraded = "tf.math.conj(a)\n"
    temp_file.write(original)
    temp_file.close()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    upgrader.process_file(temp_file.name, temp_file.name)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)

  def testInplaceNoOutputChangeOnErrorHandling(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: same unclosed read handle and cleanup-on-pass-only as the first file test.
    # Scenario: a temporary file with invalid source, processed in place with the option that keeps
    #   the output unchanged on error.
    # Asserts the content is untouched; then deletes the file.
    """In place file should not be modified when parsing error is handled."""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "print 'a' \n"
    upgraded = "print 'a' \n"
    temp_file.write(original)
    temp_file.close()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    upgrader.process_file(
        temp_file.name, temp_file.name, no_change_to_outfile_on_error=True)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)

  def testInplaceEmptyOutputOnError(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: same unclosed read handle and cleanup-on-pass-only as the first file test.
    # Scenario: a temporary file with invalid source, processed in place without the error option.
    # Asserts the file becomes empty; then deletes it.
    """In place file becomes empty when parsing error is not handled."""
    temp_file = tempfile.NamedTemporaryFile("w", delete=False)
    original = "print 'a' \n"
    upgraded = ""
    temp_file.write(original)
    temp_file.close()
    upgrader = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
    upgrader.process_file(temp_file.name, temp_file.name)
    self.assertAllEqual(open(temp_file.name).read(), upgraded)
    os.unlink(temp_file.name)


if __name__ == "__main__":
  test_lib.main()
