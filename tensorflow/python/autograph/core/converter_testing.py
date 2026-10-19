# [MODULE DOC]
# HELPER: shared test utilities for the AutoGraph converter tests (not a test module itself).
# Provides a function to mark a callable as belonging to an allowlisted fake module (altering the
# global conversion rules), a stack inspection helper telling whether the caller is generated
# code, a transpiler subclass that applies only a chosen list of converters (and can override
# members of the generated-code namespace), and a test case base class that runs each test in a
# fresh graph and offers print capturing and a function-transformation helper.
# Side effects at use time: allowlist registers a module in the interpreter's module table and
# prepends a rule to the global conversion rules.
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
"""Base class for tests in this module."""

import contextlib
import inspect
import io
import sys
import types

from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def allowlist(f):
  # [FUNCTION DOC]
  # HELPER: lets tests mark a callable as exempt from AutoGraph conversion.
  # Input: a callable f. On the first call in the process, creates an empty fake module named
  # for testing, registers it in the interpreter's module table, and prepends a do-not-convert
  # rule for that module name to the global conversion rules (module-level state change that
  # persists). Always sets the callable's module attribute to the fake module name. Returns
  # nothing. The docstring misspells the word allowlisted.
  """Helper that marks a callable as whtelitisted."""
  if 'allowlisted_module_for_testing' not in sys.modules:
    allowlisted_mod = types.ModuleType('allowlisted_module_for_testing')
    sys.modules['allowlisted_module_for_testing'] = allowlisted_mod
    config.CONVERSION_RULES = (
        (config.DoNotConvert('allowlisted_module_for_testing'),) +
        config.CONVERSION_RULES)

  f.__module__ = 'allowlisted_module_for_testing'


def is_inside_generated_code():
  # [FUNCTION DOC]
  # HELPER: test utility for checking whether code is running from AutoGraph-generated code.
  # Inspects the call stack starting at the caller's frame, skipping frames of the internal
  # converted-call and unconverted-call functions, and returns true if the first remaining frame
  # has the generated-code namespace variable among its local variables; false if the stack is
  # exhausted or no such variable is present. Releases the frame reference when done.
  """Tests whether the caller is generated code. Implementation-specific."""
  frame = inspect.currentframe()
  try:
    frame = frame.f_back

    internal_stack_functions = ('converted_call', '_call_unconverted')
    # Walk up the stack until we're out of the internal functions.
    while (frame is not None and
           frame.f_code.co_name in internal_stack_functions):
      frame = frame.f_back
    if frame is None:
      return False

    return 'ag__' in frame.f_locals
  finally:
    del frame


class TestingTranspiler(api.PyToTF):
  # [CLASS DOC]
  # HELPER: transpiler used by the converter tests. Extends the real PyToTF transpiler but, after
  # the initial analysis, applies only the converter modules it was given instead of the full
  # pipeline. Can replace members of the generated-code namespace with test versions. Keeps the
  # transformed AST and the transform context of the last transformation for inspection.
  """Testing version that only applies given transformations."""

  def __init__(self, converters, ag_overrides):
    # [FUNCTION DOC]
    # Constructor. Inputs: converters, one converter module or a list/tuple of them (a single one
    # is wrapped in a tuple), and ag_overrides, a mapping from names to replacement values for
    # the generated-code namespace (may be nothing). Initializes the parent, stores both, and
    # sets the saved transformed AST to nothing.
    super(TestingTranspiler, self).__init__()
    if isinstance(converters, (list, tuple)):
      self._converters = converters
    else:
      self._converters = (converters,)
    self.transformed_ast = None
    self._ag_overrides = ag_overrides

  def get_extra_locals(self):
    # [FUNCTION DOC]
    # Returns the extra names made available to generated code (a mapping). Takes the parent's
    # mapping and, when overrides were given, replaces its generated-code namespace entry with a
    # copy of the original namespace updated with the overrides, so tests can substitute runtime
    # functions without altering the real module.
    retval = super(TestingTranspiler, self).get_extra_locals()
    if self._ag_overrides:
      modified_ag = types.ModuleType('fake_autograph')
      modified_ag.__dict__.update(retval['ag__'].__dict__)
      modified_ag.__dict__.update(self._ag_overrides)
      retval['ag__'] = modified_ag
    return retval

  def transform_ast(self, node, ctx):
    # [FUNCTION DOC]
    # Inputs: the function's AST node and the transformation context. Runs the parent's initial
    # static analysis, then applies the transform function of each configured converter module in
    # order. Stores the resulting AST and the context on the instance and returns the AST.
    node = self.initial_analysis(node, ctx)

    for c in self._converters:
      node = c.transform(node, ctx)

    self.transformed_ast = node
    self.transform_ctx = ctx
    return node


class TestCase(test.TestCase):
  # [CLASS DOC]
  # HELPER: base class for the converter unit tests. Runs every test inside a fresh default graph
  # (graph mode, needed to test control flow), and supplies a context manager to capture and
  # check printed output and a helper to transform functions with chosen converters.
  """Base class for unit tests in this module. Contains relevant utilities."""

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: creates a new graph and enters it as the default graph; the graph
    # context is kept to be exited in tear-down. Does not call the parent setup.
    # AutoGraph tests must run in graph mode to properly test control flow.
    self.graph = ops.Graph().as_default()
    self.graph.__enter__()

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: exits the default-graph context entered in setup. Does not call the
    # parent teardown.
    self.graph.__exit__(None, None, None)

  @contextlib.contextmanager
  def assertPrints(self, expected_result):
    # [FUNCTION DOC]
    # Context manager for tests. Input: expected_result, the exact text expected to be written to
    # standard output inside the managed block. Redirects standard output to an in-memory buffer,
    # runs the block, then asserts the captured text equals the expectation. Always restores the
    # original standard output, even on error. Yields nothing.
    try:
      out_capturer = io.StringIO()
      sys.stdout = out_capturer
      yield
      self.assertEqual(out_capturer.getvalue(), expected_result)
    finally:
      sys.stdout = sys.__stdout__

  def transform(
      self, f, converter_module, include_ast=False, ag_overrides=None):
    # [FUNCTION DOC]
    # HELPER: converts a function applying only the given converters.
    # Inputs: f (the function to convert), converter_module (one converter module or a list/tuple
    # of modules), include_ast (boolean, default false) and ag_overrides (optional mapping of
    # generated-code namespace names to replacement values). Builds a recursive program context,
    # runs the testing transpiler on f. Returns the converted function; if include_ast is true
    # returns a triple of the converted function, the transformed AST and the transform context.
    program_ctx = converter.ProgramContext(
        options=converter.ConversionOptions(recursive=True),
        autograph_module=api)

    tr = TestingTranspiler(converter_module, ag_overrides)
    transformed, _, _ = tr.transform_function(f, program_ctx)

    if include_ast:
      return transformed, tr.transformed_ast, tr.transform_ctx

    return transformed
