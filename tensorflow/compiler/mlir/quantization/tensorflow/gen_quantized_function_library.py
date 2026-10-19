# [MODULE DOC]
# HELPER: build-time code generator for the quantized function library C++ header.
# Command-line tool (abseil flags and app runner). It reads one or more MLIR source files whose
# names contain the prefix quantized_function_library, expands text templates inside them
# (for-loop templates, parameterized function templates, quantized and implementation function
# name placeholders), and writes a single C++ header holding each expanded module as a constant
# character array of string literals inside the requested namespace.
# Flags: output_file (required, path to write), src (required, space separated source paths),
# namespace (default mlir::quant, double-colon separated).
# Import-time side effects: defines the three flags and marks output_file and src as required.
# A TODO notes that this generator has no test.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Generates the quantized function library contained header file."""

import ast
import re
import string
from typing import Sequence

from absl import app
from absl import flags

# TODO(b/263048929): Create a test for gen_quantized_function_library.

_OUTPUT_FILE = flags.DEFINE_string('output_file', None, 'output file location')
_SRCS = flags.DEFINE_string('src', None, 'source file locations')
_NAMESPACE = flags.DEFINE_string('namespace', 'mlir::quant',
                                 'namespace in the generated file')

flags.mark_flags_as_required(['output_file', 'src'])


def _substitute_for_loop_template(module: str) -> str:
  # [FUNCTION DOC]
  # HELPER: template expansion step used by the library generator's main function.
  # Repeatedly finds blocks written as a for line naming a variable and a bracketed list of
  # values, followed by a braced body and an end-for marker line. Each block is replaced by one
  # copy of the body per list value, with the variable substituted (unknown placeholders are left
  # untouched) and with parameterized function templates in each copy expanded. Copies are
  # separated by a newline escape.
  # Input: module, the MLIR text. Returns the text with all loop blocks expanded.
  # Raises a value error if the variable, value list or body cannot be parsed.
  """Substitutes the for loop templates in the given module."""
  compiled_regex = re.compile(
      r'^\s*for\s(.*?)\sin\s(\[.*?\])\s\{(.*?)\}\s//\send\sfor\n',
      re.MULTILINE | re.DOTALL)
  while True:
    func_match = re.search(compiled_regex, module)
    if func_match is None:
      break

    try:
      arg_name = func_match.group(1)
      arg_values = ast.literal_eval(func_match.group(2))
      loop_template = string.Template(func_match.group(3))
    except Exception as e:  # pylint: disable=broad-except
      raise ValueError('The loop template is in wrong format') from e

    replacement_text = ''
    for arg_value in arg_values:
      arg_dict = {arg_name: arg_value}
      replacement_text += '\\n'
      replacement_text += _substitute_parameterization_template(
          loop_template.safe_substitute(arg_dict))
    module = re.sub(compiled_regex, replacement_text, module, count=1)

  return module


def _substitute_parameterization_template(module: str) -> str:
  # [FUNCTION DOC]
  # HELPER: template expansion step used by the library generator's main function.
  # Finds a parameters line holding a bracketed list of mappings, immediately followed by a
  # function definition ending in a return and closing brace. The function text is instantiated
  # once per mapping, replacing placeholder names by the mapping values. Values are converted to
  # text with single quotes turned into double quotes (single quotes are invalid in MLIR). The
  # dtype attribute marker followed by a dollar sign is escaped so dtype attributes survive.
  # Side effect: the mapping objects in the parsed list are modified (values become text).
  # Input: module, the MLIR text. Returns the expanded text. Raises a value error when the list
  # or template cannot be parsed; a mapping missing a placeholder key raises a key error.
  """Substitutes all the function templates in the given module."""
  compiled_regex = re.compile(
      r'^\s*parameters(\[.*?\])\n?(^\s*(?:func\.)+func.*?\{.*?(?:func\.)+return.*?\}\n)',
      re.MULTILINE | re.DOTALL)
  while True:
    func_match = re.search(compiled_regex, module)
    if func_match is None:
      break

    try:
      value_list = ast.literal_eval(func_match.group(1))
      # Escapes template $-based substitutions for attributes containing $.
      # $$ is replaced with a single $.
      func_template = string.Template(
          func_match.group(2).replace('tfdtype$DT_', 'tfdtype$$DT_'))
    except Exception as e:  # pylint: disable=broad-except
      raise ValueError('The function template is in wrong format') from e

    replacement_text = ''
    for value_dict in value_list:
      for key, value in value_dict.items():
        # Replace single quote to double quote since single quote around a
        # string are not valid in the MLIR representation.
        value_dict[key] = str(value).replace("'", '"')
      replacement_text += '\\n'
      replacement_text += func_template.substitute(value_dict)
    module = re.sub(compiled_regex, replacement_text, module, count=1)
  return module


def _format_snake_case_op_name(s):
  # [FUNCTION DOC]
  # HELPER: name formatting for the generated function name placeholders.
  # Converts a camel case op name to snake case: 2D and 3D become lower case, every upper case
  # letter becomes an underscore plus its lower case form (leading underscore removed), then
  # mat_mul is shortened to matmul and bias_add to bias.
  # Input: s, the op name text. Returns the converted text. No side effects.
  """Formats the op name to snake case."""
  s = s.replace('2D', '2d').replace('3D', '3d')
  snake_case = ''.join(['_' + i.lower() if i.isupper() else i for i in s
                       ]).lstrip('_')
  return snake_case.replace('mat_mul', 'matmul').replace('bias_add', 'bias')


def _substitute_impl_function_name_template(module: str) -> str:
  # [FUNCTION DOC]
  # HELPER: template expansion step used by the library generator's main function.
  # Replaces every generate-impl-function-name placeholder holding an op name with the text
  # internal, underscore, the snake case op name, underscore, fn.
  # Input: module, the MLIR text. Returns the text with all such placeholders replaced.
  """Generates the op-specific implementation function name."""
  compiled_regex = re.compile(r'GenerateImplFunctionName\(([\w\s]+)\)')
  while True:
    func_match = re.search(compiled_regex, module)
    if func_match is None:
      break

    text = func_match.group(1)
    function_name = 'internal_{}_fn'.format(_format_snake_case_op_name(text))
    module = re.sub(compiled_regex, function_name, module, count=1)
  return module


def _substitute_quantized_function_name_template(module: str) -> str:
  # [FUNCTION DOC]
  # HELPER: template expansion step used by the library generator's main function.
  # KNOWN ISSUE: the second check on more than one op name duplicates the first one; harmless
  # redundancy (the slice of remaining ops is simply empty otherwise).
  # Replaces each generate-quantized-function-name placeholder with a function name. The
  # placeholder argument is a list of op names, optionally followed by an output type text. The
  # name is quantized, underscore, first op in snake case; a second op adds with, underscore, op;
  # each further op adds and, underscore, op. The suffix is fn, or float output fn when the output
  # type is f32.
  # Input: module, the MLIR text. Returns the rewritten text.
  # Raises a value error when there are zero or more than two arguments or the op list is empty.
  """Generates the quantized function name."""
  compiled_regex = re.compile(
      r'GenerateQuantizedFunctionName(\([\w\s\'\"\[\],]+\))')
  while True:
    func_match = re.search(compiled_regex, module)
    if func_match is None:
      break

    # Make sure the string ends with ",)" so the parsed value is a tuple.
    argument_string = func_match.group(1)
    if not argument_string.endswith(',)'):
      argument_string = argument_string[:-1] + ',)'
    arguments = ast.literal_eval(argument_string)

    if len(arguments) < 1 or len(arguments) > 2:
      raise ValueError(
          'Wrong number of arguments to GenerateQuantizedFunctionName')

    quantized_ops = arguments[0]
    if not quantized_ops:
      raise ValueError('The quantized_ops list must not be empty')

    # Add op names to the function name.
    function_name = 'quantized_{}'.format(
        _format_snake_case_op_name(quantized_ops[0]))
    if len(quantized_ops) > 1:
      function_name += '_with_{}'.format(
          _format_snake_case_op_name(quantized_ops[1]))
    if len(quantized_ops) > 1:
      for quantized_op in quantized_ops[2:]:
        function_name += '_and_{}'.format(
            _format_snake_case_op_name(quantized_op))

    # Add suffix based on output type.
    suffix = '_fn'
    if len(arguments) > 1 and arguments[1] == 'f32':
      suffix = '_float_output_fn'
    function_name += suffix

    module = re.sub(compiled_regex, function_name, module, count=1)
  return module


def main(_: Sequence[str]) -> None:
  # [FUNCTION DOC]
  # Entry point run by the application runner; the positional arguments are ignored.
  # Reads the flags: splits the namespace flag on double colons and the src flag on spaces. For
  # each source file it reads the text, extracts the module block (dropping the copyright), checks
  # that the file name contains the library prefix exactly once, takes the tag from the text after
  # the prefix minus its last five characters (the mlir extension), and applies the for-loop,
  # parameterization, quantized-name and implementation-name expansions in that order.
  # Then writes the output file: license text, include guard start, nested namespaces, one
  # constant character array per module (named with a fixed prefix plus the upper-cased tag)
  # made of quoted lines with escaped double quotes and newline escapes, closing namespaces and
  # the include guard end (without trailing newline).
  # Raises a value error when no module block is found or the file name lacks the prefix.
  # Side effects: reads the source files and overwrites the output file. Returns nothing.
  namespaces = _NAMESPACE.value.split('::')
  src_files = _SRCS.value.split(' ')
  file_prefix = 'quantized_function_library'
  module_prefix = 'kQuantizedFunctionLibraryInMLIR'

  modules = []

  for src_file in src_files:
    with open(src_file, 'r') as f:
      content = f.read()

      # Skip the copyright in the source file.
      module_match = re.search(r'(^module\s\{)(.*)(^\})', content,
                               re.MULTILINE | re.DOTALL)
      if module_match is None:
        raise ValueError("Couldn't find module in the function library")
      module = module_match.group()

      # Substitute all the function templates.
      out = re.split(file_prefix, src_file)
      if len(out) != 2:
        raise ValueError('The file name must start with {}'.format(file_prefix))
      tag = out[1][:-5]  # the last five values = ".mlir"
      module = _substitute_for_loop_template(module)
      module = _substitute_parameterization_template(module)
      module = _substitute_quantized_function_name_template(module)
      module = _substitute_impl_function_name_template(module)
      modules.append((tag, module))

  with open(_OUTPUT_FILE.value, 'w') as f:
    f.write("""/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_QUANTIZED_FUNCTION_LIBRARY_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_QUANTIZED_FUNCTION_LIBRARY_H_
""")

    for namespace in namespaces:
      f.write('namespace {0} {{\n'.format(namespace))

    for tag, module in modules:
      f.write('constexpr char {0}[] ='.format(module_prefix + tag.upper()))

      for line in module.splitlines():
        f.write('\n  "')
        f.write(line.rstrip().replace('"', r'\"'))
        f.write('\\n"')

      f.write(';\n')

    for namespace in reversed(namespaces):
      f.write('}}  // namespace {0}\n'.format(namespace))

    f.write(
        '#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_QUANTIZED_FUNCTION_LIBRARY_H_'
    )


if __name__ == '__main__':
  app.run(main)
