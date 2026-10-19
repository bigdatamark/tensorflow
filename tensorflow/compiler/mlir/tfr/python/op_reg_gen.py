# [MODULE DOC]
# Generates C++ op registration source code from Python composite op functions. It transpiles the
# source of functions decorated with the Composite decorator: the decorator arguments (op name,
# inputs, attributes, derived attributes, outputs) are turned into a registration statement for
# the op, skipping ops that are already registered. Exposes the public functions for generating
# code from a single function and from all matching functions in a module or class. Used by the
# composite op definition template and its tests. No import-time side effects beyond imports.
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
"""op_reg_gen: Generate op registration code from composite op code."""

# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=g-direct-tensorflow-import

import gast as ast

from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.framework import op_def_registry
from tensorflow.python.util import tf_inspect

_COMPOSITE_ARG_LIST = ['op_name', 'inputs', 'attrs', 'derived_attrs', 'outputs']


class OpRegGenImpl(transformer.CodeGenerator):
  # [CLASS DOC]
  # HELPER: Internal syntax tree visitor used by the op registration generator.
  # Code generator that walks the syntax tree of a composite function and emits a C++ op
  # registration statement into its code buffer. Most visit methods evaluate small nodes (names,
  # constants, keywords, lists, argument lists) into plain Python values that are consumed by the
  # function definition visitor. Holds the transformation context in an attribute.
  """Visit the AST and generate C++ op registration functions."""

  def __init__(self, ctx):
    # [FUNCTION DOC]
    # Constructor. ctx is the transformation context supplied by the transpiler; it is passed to
    # the base class and also kept as an attribute.
    super(OpRegGenImpl, self).__init__(ctx)
    self.ctx = ctx

  def visit_Name(self, node):
    # [FUNCTION DOC]
    # Visitor for a name node. Returns the identifier string.
    return node.id

  def visit_Constant(self, node):
    # [FUNCTION DOC]
    # Visitor for a literal constant node. Returns the Python value of the constant.
    return node.value

  def visit_keyword(self, node):
    # [FUNCTION DOC]
    # Visitor for a keyword argument of a call. Returns a pair of the keyword name and the visited
    # value of the argument.
    return node.arg, self.visit(node.value)

  def visit_List(self, node):
    # [FUNCTION DOC]
    # Visitor for a list literal. Returns a list holding the visited value of each element.
    return [self.visit(cst) for cst in node.elts]

  def visit_arguments(self, node):
    # [FUNCTION DOC]
    # Visitor for a function's argument list. Returns the list of visited positional argument
    # nodes (for names, their identifier strings). Keyword-only, variadic and default arguments
    # are ignored.
    return [self.visit(arg) for arg in node.args]

  def visit_FunctionDef(self, node):
    # [FUNCTION DOC]
    # Visitor for a function definition; does all the real work. Looks through the decorators for
    # exactly one call to the Composite decorator (as a plain name or an attribute). Functions
    # without it are silently skipped (nothing emitted). Raises a key error if more than one such
    # decorator exists. Positional decorator arguments are mapped by position to op name, inputs,
    # attributes, derived attributes and outputs; keyword arguments are added on top. Raises a key
    # error if a name is given both positionally and by keyword. If the op name is already
    # registered in the global op definition registry, emits nothing, but raises a value error if
    # any argument beyond the op name was given. Otherwise checks that the number of function
    # parameters equals the number of declared inputs plus attributes (key error on mismatch; only
    # counts, not names, are compared), then emits a C++ registration statement listing the op
    # name, inputs, attributes (double quotes replaced by single quotes), derived attributes and
    # outputs, ending with a semicolon. Returns nothing; the output goes to the code buffer.
    # TODO(fengliuai): create one utility method to match different apis and
    # shared it with the tfr_gen.py module.
    compose_dec = []
    for dec in node.decorator_list:
      if isinstance(dec, ast.Call):
        if isinstance(dec.func, ast.Attribute) and dec.func.attr == 'Composite':
          compose_dec.append(dec)
        if isinstance(dec.func, ast.Name) and dec.func.id == 'Composite':
          compose_dec.append(dec)

    if not compose_dec:
      # skip a non-composition function
      return
    elif len(compose_dec) > 1:
      raise KeyError('More than one TF ops decomposes for.')

    all_dec_args = {}
    for arg_name, arg_value in zip(_COMPOSITE_ARG_LIST, compose_dec[0].args):
      all_dec_args[arg_name] = self.visit(arg_value)

    kw_dec_args = dict([self.visit(kw) for kw in compose_dec[0].keywords])

    if all_dec_args.keys() & kw_dec_args.keys():
      raise KeyError('More arguments than expected.')

    all_dec_args.update(kw_dec_args)

    op_name = all_dec_args['op_name']
    op_def = op_def_registry.get(op_name)
    if op_def:
      if len(all_dec_args) > 1:
        # Op has been registered, so it is a user error to specify op def.
        raise ValueError('Op has been registered: ' + op_name)
      else:
        # Op has been registered, then we don't need to generate register code.
        return

    # Validates the function inputs match what are in the decorator.
    inputs = all_dec_args.get('inputs', [])
    attrs = all_dec_args.get('attrs', [])
    expected_args = [arg.split(':')[0] for arg in inputs + attrs]
    all_func_args = self.visit(node.args)

    if len(expected_args) != len(all_func_args):
      raise KeyError(
          'Composition arguments for {} do not match the registration. {} vs {}'
          .format(op_name, expected_args, all_func_args))

    cxx_reg_code = ['\nREGISTER_OP("{}")'.format(op_name)]
    for input_ in inputs:
      cxx_reg_code.append('.Input("{}")'.format(input_))
    for attr in attrs:
      py_str = attr.replace('"', "'")
      cxx_reg_code.append('.Attr("{}")'.format(py_str))
    for attr in all_dec_args.get('derived_attrs', []):
      py_str = attr.replace('"', "'")
      cxx_reg_code.append('.Attr("{}")'.format(py_str))
    for output_ in all_dec_args.get('outputs', []):
      cxx_reg_code.append('.Output("{}")'.format(output_))
    cxx_reg_code[-1] += ';\n'
    self.emit('\n    '.join(cxx_reg_code))


class OpRegGen(transpiler.GenericTranspiler):
  # [CLASS DOC]
  # KNOWN ISSUE: Docstring says it produces TFR MLIR source, but it uses the registration visitor
  # and so produces C++ op registration source code.
  # Transpiler that turns a Python function object into C++ op registration text by running the
  # registration visitor on its syntax tree. Stateless; used by the module level generator
  # function.
  """Transforms Python objects into TFR MLIR source code."""

  def transform_ast(self, node, ctx):
    # [FUNCTION DOC]
    # Transpiler hook. Takes the syntax tree node of the function and the transformation context,
    # runs the registration visitor over the node and returns the text accumulated in the
    # visitor's code buffer.
    gen = OpRegGenImpl(ctx)
    gen.visit(node)
    return gen.code_buffer


def op_reg_gen(func):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Docstring mentions emitting TFR functions, but the result is C++ op registration
  # code for the composite op declared by the function's decorator.
  # Takes a Python function object, transpiles it and returns the generated text (a string, empty
  # for functions that are not composites or whose op is already registered). Raises key or value
  # errors from the visitor on malformed decorator usage.
  """Parse a function and emit the TFR functions."""
  op_reg_code, _ = OpRegGen().transform(func, None)
  return op_reg_code


def gen_register_op(source, method_prefix=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: Docstring mentions emitting TFR functions from a class, but it emits a C++ source
  # fragment with op registrations, and the source can be a module or class (anything whose
  # members can be inspected).
  # Public entry point. source is a module or class; method_prefix is an optional string (default
  # nothing) so that only functions whose names start with it are processed (all functions when
  # empty). Generates registration text for each selected function, joins them and wraps the
  # result in an include of the op header and a namespace block. Returns the full C++ text as a
  # string.
  """Parse a python code and emit the TFR functions from a target class."""
  mlir_funcs = [
      op_reg_gen(func)
      for name, func in tf_inspect.getmembers(source, tf_inspect.isfunction)
      if not method_prefix or name.startswith(method_prefix)
  ]
  headers = r"""
#include "tensorflow/core/framework/op.h"

namespace tensorflow {
  """
  code = '\n'.join(mlir_funcs)
  return headers + code + '}  // namespace tensorflow\n'
