# [MODULE DOC]
# Implementation of the saved_model_cli command-line tool, which inspects and executes graphs
# stored in a SavedModel directory. It provides sub-commands: show (list tag-sets, signature
# keys, inputs/outputs, ops and concrete functions), run (feed inputs loaded from files, literal
# expressions or tf.Example descriptions and print or save the outputs), scan (report ops that
# appear on a denylist), convert (TensorRT conversion), freeze_model and aot_compile_cpu
# (ahead-of-time compilation preparation through the saved_model_aot_compile module).
# Import-time side effects: defines a large set of absl command-line flags (dir, tag_set,
# signature_def, inputs, input_exprs, outdir, and many more), chooses the default target triple
# depending on whether the machine is s390x, and builds module-level tables such as the default
# op denylist and the required-flags-per-command mapping. The main entry point builds an
# argument parser with one sub-parser per command and runs it through absl.
# Many helpers here are private and shared between the sub-command implementations.
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
"""Command-line interface to inspect and execute a graph in a SavedModel.

For detailed usages and examples, please refer to:
https://www.tensorflow.org/guide/saved_model#cli_to_inspect_and_execute_savedmodel

"""

import argparse
import platform

import ast
import os
import re

from absl import app  # pylint: disable=unused-import
from absl import flags
from absl.flags import argparse_flags
import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.wrappers import local_cli_wrapper
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun
from tensorflow.python.framework import meta_graph as meta_graph_lib
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.tools import saved_model_aot_compile
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.tpu import tpu
from tensorflow.python.util.compat import collections_abc


_XLA_DEBUG_OPTIONS_URL = (
    'https://github.com/tensorflow/tensorflow/blob/master/'
    'tensorflow/compiler/xla/debug_options_flags.cc')


# Set of ops to denylist.
_OP_DENYLIST = set(['WriteFile', 'ReadFile', 'PrintV2'])


# Custom SavedModel CLI flags
_SMCLI_DIR = flags.DEFINE_string(
    name='dir', default=None, help='Directory containing the SavedModel.')

_SMCLI_ALL = flags.DEFINE_bool(
    name='all', default=False,
    help='If set, outputs all available information in the given SavedModel.')

_SMCLI_TAG_SET = flags.DEFINE_string(
    name='tag_set', default=None,
    help='Comma-separated set of tags that identify variant graphs in the '
    'SavedModel.')

_SMCLI_SIGNATURE_DEF = flags.DEFINE_string(
    name='signature_def', default=None,
    help='Specifies a SignatureDef (by key) within the SavedModel to display '
    'input(s) and output(s) for.')

_SMCLI_LIST_OPS = flags.DEFINE_bool(
    name='list_ops', default=False,
    help='If set, will output ops used by a MetaGraphDef specified by tag_set.')

_SMCLI_INPUTS = flags.DEFINE_string(
    name='inputs', default='',
    help='Specifies input data files to pass to numpy.load(). Format should be '
    '\'<input_key>=<filename>\' or \'<input_key>=<filename>[<variable_name>]\','
    ' separated by \';\'. File formats are limited to .npy, .npz, or pickle.')

_SMCLI_INPUT_EXPRS = flags.DEFINE_string(
    name='input_exprs', default='',
    help='Specifies Python literal expressions or numpy functions. Format '
    'should be "<input_key>=\'<python_expression>\'", separated by \';\'. Numpy'
    ' can be accessed with \'np\'. Note that expressions are passed to '
    'literal_eval(), making this flag susceptible to code injection. Overrides '
    'duplicate input keys provided with the --inputs flag.')

_SMCLI_INPUT_EXAMPLES = flags.DEFINE_string(
    name='input_examples', default='',
    help='Specifies tf.train.Example objects as inputs. Format should be '
    '\'<input_key>=[{{feature0:value_list,feature1:value_list}}]\', where input'
    ' keys are separated by \';\'. Overrides duplicate input keys provided with'
    ' the --inputs and --input_exprs flags.')

_SMCLI_OUTDIR = flags.DEFINE_string(
    name='outdir', default=None,
    help='If specified, writes CLI output to the given directory.')

_SMCLI_OVERWRITE = flags.DEFINE_bool(
    name='overwrite', default=False,
    help='If set, overwrites output file if it already exists.')

_SMCLI_TF_DEBUG = flags.DEFINE_bool(
    name='tf_debug', default=False,
    help='If set, uses the Tensorflow Debugger (tfdbg) to watch intermediate '
    'Tensors and runtime GraphDefs while running the SavedModel.')

_SMCLI_WORKER = flags.DEFINE_string(
    name='worker', default=None,
    help='If specified, runs the session on the given worker (bns or gRPC '
    'path).')

_SMCLI_INIT_TPU = flags.DEFINE_bool(
    name='init_tpu', default=False,
    help='If set, calls tpu.initialize_system() on the session. '
                  'Should only be set if the specified worker is a TPU job.')

_SMCLI_USE_TFRT = flags.DEFINE_bool(
    name='use_tfrt', default=False,
    help='If set, runs a TFRT session, instead of a TF1 session.')

_SMCLI_OP_DENYLIST = flags.DEFINE_string(
    name='op_denylist', default=None,
    help='If specified, detects and reports the given ops. List of ops should '
    'be comma-separated. If not specified, the default list of ops is '
    '[WriteFile, ReadFile, PrintV2]. To specify an empty list, pass in the '
    'empty string.')

_SMCLI_OUTPUT_DIR = flags.DEFINE_string(
    name='output_dir', default=None,
    help='Output directory for the SavedModel.')

_SMCLI_MAX_WORKSPACE_SIZE_BYTES = flags.DEFINE_integer(
    name='max_workspace_size_bytes', default=2 << 20,
    help='The maximum temporary GPU memory which the TensorRT engine can use at'
    ' execution time.')

_SMCLI_PRECISION_MODE = flags.DEFINE_enum(
    name='precision_mode', default='FP32', enum_values=['FP32', 'FP16', 'INT8'],
    help='TensorRT data precision. One of FP32, FP16, or INT8.')

_SMCLI_MINIMUM_SEGMENT_SIZE = flags.DEFINE_integer(
    name='minimum_segment_size', default=3,
    help='The minimum number of nodes required for a subgraph to be replaced in'
    ' a TensorRT node.')

_SMCLI_CONVERT_TF1_MODEL = flags.DEFINE_bool(
    name='convert_tf1_model', default=False,
    help='Support TensorRT conversion for TF1 models.')

_SMCLI_OUTPUT_PREFIX = flags.DEFINE_string(
    name='output_prefix', default=None,
    help='Output directory + filename prefix for the resulting header(s) and '
    'object file(s).')

_SMCLI_SIGNATURE_DEF_KEY = flags.DEFINE_string(
    name='signature_def_key',
    default=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    help='SavedModel SignatureDef key to use.')

_SMCLI_CHECKPOINT_PATH = flags.DEFINE_string(
    name='checkpoint_path', default=None,
    help='Custom checkpoint to use. Uses SavedModel variables by default.')

_SMCLI_VARIABLES_TO_FEED = flags.DEFINE_string(
    name='variables_to_feed', default='',
    help='Names of the variables that will be fed into the SavedModel graph. '
    'Pass in \'\' to feed no variables, \'all\' to feed all variables, or a '
    'comma-separated list of variable names. Variables not fed will be frozen. '
    '*NOTE* Variables passed here must be set *by the user*. These variables '
    'will NOT be frozen, and their values will be uninitialized in the compiled'
    ' object.')

if platform.machine() == 's390x':
  _SMCLI_TARGET_TRIPLE = flags.DEFINE_string(
      name='target_triple',
      default='',
      help=(
          'Triple identifying a target variation, containing information suchas'
          ' processor architecture, vendor, operating system, and environment.'
          " Defaults to ''."
      ),
  )
else:
  _SMCLI_TARGET_TRIPLE = flags.DEFINE_string(
      name='target_triple',
      default='x86_64-pc-linux',
      help=(
          'Triple identifying a target variation, containing information such'
          ' as processor architecture, vendor, operating system, and'
          " environment. Defaults to 'x86_64-pc-linux'."
      ),
  )

_SMCLI_TARGET_CPU = flags.DEFINE_string(
    name='target_cpu', default='',
    help='Target CPU name for LLVM during AOT compilation. Examples include '
    '\'x86_64\', \'skylake\', \'haswell\', \'westmere\', \'\' (unknown).')

_SMCLI_CPP_CLASS = flags.DEFINE_string(
    name='cpp_class', default=None,
    help='The name of the generated C++ class, wrapping the generated function.'
    ' Format should be [[<optional_namespace>::],...]<class_name>, i.e. the '
    'same syntax as C++ for specifying a class. This class will be generated in'
    ' the given namespace(s), or, if none are specified, the global namespace.')

_SMCLI_MULTITHREADING = flags.DEFINE_string(
    name='multithreading', default='False',
    help='Enable multithreading in the compiled computation. Note that with '
    'this flag enabled, the resulting object files may have external '
    'dependencies on multithreading libraries, such as \'Abseil\'.')

command_required_flags = {
    'show': ['dir'],
    'run': ['dir', 'tag_set', 'signature_def'],
    'scan': ['dir'],
    'convert': ['dir', 'output_dir', 'tag_set'],
    'freeze_model': ['dir', 'output_prefix', 'tag_set'],
    'aot_compile_cpu': ['cpp_class'],
}


def _sanitize_nonempty_str_list(xs, name: str):
  # [FUNCTION DOC]
  # HELPER: cleans comma-split user input (tag sets, variable names) for the sub-commands.
  # Takes an iterable of items (xs) and a human-readable field name (name) used in the error.
  # Skips null items, converts every other item to text and strips surrounding whitespace, and
  # keeps only the non-empty results in their original order. Returns the new list of strings.
  # Raises a value error naming the field when nothing non-empty remains. No side effects.
  """Trim items, drop empties/None, require at least one non-empty string."""
  out = []
  for x in xs:
    if x is None:
      continue
    s = str(x).strip()
    if s:
      out.append(s)
  if not out:
    raise ValueError(f'{name} must contain at least one non-empty item.')
  return out


def _show_tag_sets(saved_model_dir):
  # [FUNCTION DOC]
  # HELPER: implements the default output of the show command.
  # Reads every tag-set of the SavedModel found in saved_model_dir (a directory path) and prints
  # a header line followed by one line per tag-set, with tags sorted inside each set and the
  # sets sorted. Returns nothing; its only effect is printing to standard output. Errors from
  # reading the SavedModel propagate.
  """Prints the tag-sets stored in SavedModel directory.

  Prints all the tag-sets for MetaGraphs stored in SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
  """
  tag_sets = saved_model_utils.get_saved_model_tag_sets(saved_model_dir)
  print('The given SavedModel contains the following tag-sets:')
  for tag_set in sorted(tag_sets):
    print('%r' % ', '.join(sorted(tag_set)))


def _get_ops_in_metagraph(meta_graph_def):
  # [FUNCTION DOC]
  # HELPER: shared by the show command variants that list ops.
  # Takes a MetaGraphDef message and returns a set of the op type names used by its graph
  # definition (as computed by the meta-graph library). Pure function, no side effects.
  """Returns a set of the ops in the MetaGraph.

  Returns the set of all the ops used in the MetaGraphDef indicated by the
  tag_set stored in SavedModel directory.

  Args:
    meta_graph_def: MetaGraphDef to list the ops of.

  Returns:
    A set of ops.
  """
  return set(meta_graph_lib.ops_used_by_graph_def(meta_graph_def.graph_def))


def _show_ops_in_metagraph_mgd(meta_graph_def):
  # [FUNCTION DOC]
  # HELPER: printing half of the op listing, used by the show command and the show-all path.
  # Takes an already loaded MetaGraphDef, collects its op set and prints one line containing
  # the meta graph's tags followed by the set of op names. Returns nothing; prints to standard
  # output. Has no docstring of its own.
  all_ops_set = _get_ops_in_metagraph(meta_graph_def)
  print(
      'The MetaGraph with tag set %s contains the following ops:'
      % meta_graph_def.meta_info_def.tags,
      all_ops_set,
  )


def _show_ops_in_metagraph(saved_model_dir, tag_set):
  # [FUNCTION DOC]
  # HELPER: implements the list-ops option of the show command.
  # Takes the SavedModel directory and a comma-separated tag-set string. Splits and sanitizes
  # the tags (trimming whitespace and dropping empty items), loads the matching MetaGraphDef and
  # prints its ops. Returns nothing. Raises a value error if no non-empty tag remains; errors
  # from loading (for example a missing tag-set) propagate. Prints to standard output.
  """Prints the ops in the MetaGraph.

  Prints all the ops used in the MetaGraphDef indicated by the tag_set stored in
  SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
    tag_set: Group of tag(s) of the MetaGraphDef in string format, separated by
      ','. For tag-set contains multiple tags, all tags must be passed in.
  """
  tags = _sanitize_nonempty_str_list(tag_set.split(','), 'tag_set')
  meta_graph_def = saved_model_utils.get_meta_graph_def(
      saved_model_dir, ','.join(tags)
  )
  _show_ops_in_metagraph_mgd(meta_graph_def)


def _show_signature_def_map_keys(saved_model_dir, tag_set):
  # [FUNCTION DOC]
  # HELPER: implements the show command when a tag-set but no signature is given.
  # Loads the signature map for the tag-set (comma-separated string, used as given without
  # sanitizing) in the SavedModel directory and prints a header plus one line per signature
  # key, in sorted order. Returns nothing; prints to standard output. Loading errors propagate.
  """Prints the keys for each SignatureDef in the SignatureDef map.

  Prints the list of SignatureDef keys from the SignatureDef map specified by
  the given tag-set and SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
    tag_set: Group of tag(s) of the MetaGraphDef to get SignatureDef map from,
        in string format, separated by ','. For tag-set contains multiple tags,
        all tags must be passed in.
  """
  signature_def_map = get_signature_def_map(saved_model_dir, tag_set)
  print('The given SavedModel MetaGraphDef contains SignatureDefs with the '
        'following keys:')
  for signature_def_key in sorted(signature_def_map.keys()):
    print('SignatureDef key: \"%s\"' % signature_def_key)


def _get_inputs_tensor_info_from_meta_graph_def(meta_graph_def,
                                                signature_def_key):
  # [FUNCTION DOC]
  # HELPER: looks up input descriptions for the show and run paths.
  # Takes a MetaGraphDef and a signature key string. Returns the mapping from input names to
  # TensorInfo messages of that signature. Raises a value error listing the available keys if
  # the signature key is not present. No side effects.
  """Gets TensorInfo for all inputs of the SignatureDef.

  Returns a dictionary that maps each input key to its TensorInfo for the given
  signature_def_key in the meta_graph_def

  Args:
    meta_graph_def: MetaGraphDef protocol buffer with the SignatureDef map to
        look up SignatureDef key.
    signature_def_key: A SignatureDef key string.

  Returns:
    A dictionary that maps input tensor keys to TensorInfos.

  Raises:
    ValueError if `signature_def_key` is not found in the MetaGraphDef.
  """
  if signature_def_key not in meta_graph_def.signature_def:
    raise ValueError(
        f'Could not find signature "{signature_def_key}". Please choose from: '
        f'{", ".join(meta_graph_def.signature_def.keys())}')
  return meta_graph_def.signature_def[signature_def_key].inputs


def _get_outputs_tensor_info_from_meta_graph_def(meta_graph_def,
                                                 signature_def_key):
  # [FUNCTION DOC]
  # HELPER: looks up output descriptions for the show and run paths.
  # KNOWN ISSUE: the docstring documents only meta_graph_def and omits the signature_def_key
  # parameter, and unlike the input counterpart it does no validation, so an unknown key
  # surfaces as a raw key error (or empty default from the protobuf map) rather than a friendly
  # value error.
  # Takes a MetaGraphDef and a signature key string and returns the mapping from output names
  # to TensorInfo messages of that signature. No side effects.
  """Gets TensorInfos for all outputs of the SignatureDef.


  Returns a dictionary that maps each output key to its TensorInfo for the given
  signature_def_key in the meta_graph_def.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer with the SignatureDefmap to
    look up signature_def_key.
  Returns:
    A dictionary that maps output tensor keys to TensorInfos.
  """
  return meta_graph_def.signature_def[signature_def_key].outputs


def _show_inputs_outputs_mgd(meta_graph_def, signature_def_key, indent):
  # [FUNCTION DOC]
  # HELPER: printing core of the show command for one signature, also used by the show-all path.
  # Takes a loaded MetaGraphDef, a signature key and an indent level (each level is two spaces).
  # Prints the inputs (sorted by key) with their dtype, shape and name, then the outputs the
  # same way, then the signature's method name. Raises a value error if the signature key is
  # unknown (from the input lookup). Returns nothing; prints to standard output.
  # Contains a nested printing helper.
  """Prints input and output TensorInfos.

  Prints the details of input and output TensorInfos for the SignatureDef mapped
  by the given signature_def_key.

  Args:
    meta_graph_def: MetaGraphDef to inspect.
    signature_def_key: A SignatureDef key string.
    indent: How far (in increments of 2 spaces) to indent each line of output.
  """
  inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key
  )
  outputs_tensor_info = _get_outputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key)

  indent_str = '  ' * indent
  def in_print(s):
    # [FUNCTION DOC]
    # HELPER: prints one line of output prefixed by the enclosing call's indentation string.
    # Takes the text s; returns nothing; prints to standard output.
    print(indent_str + s)

  in_print('The given SavedModel SignatureDef contains the following input(s):')
  for input_key, input_tensor in sorted(inputs_tensor_info.items()):
    in_print('  inputs[\'%s\'] tensor_info:' % input_key)
    _print_tensor_info(input_tensor, indent+1)

  in_print('The given SavedModel SignatureDef contains the following '
           'output(s):')
  for output_key, output_tensor in sorted(outputs_tensor_info.items()):
    in_print('  outputs[\'%s\'] tensor_info:' % output_key)
    _print_tensor_info(output_tensor, indent+1)

  in_print('Method name is: %s' %
           meta_graph_def.signature_def[signature_def_key].method_name)


def _show_inputs_outputs(saved_model_dir, tag_set, signature_def_key, indent=0):
  # [FUNCTION DOC]
  # HELPER: implements the show command when both tag-set and signature are given.
  # Inputs: SavedModel directory, comma-separated tag-set string, signature key, and optional
  # indent level (default zero). Sanitizes the tags, loads the MetaGraphDef and delegates to
  # the printing routine. Returns nothing. Raises a value error for an empty tag list or an
  # unknown signature; loading errors propagate. Prints to standard output.
  """Prints input and output TensorInfos.

  Prints the details of input and output TensorInfos for the SignatureDef mapped
  by the given signature_def_key.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
    tag_set: Group of tag(s) of the MetaGraphDef, in string format, separated by
      ','. For tag-set contains multiple tags, all tags must be passed in.
    signature_def_key: A SignatureDef key string.
    indent: How far (in increments of 2 spaces) to indent each line of output.
  """
  tags = _sanitize_nonempty_str_list(tag_set.split(','), 'tag_set')
  meta_graph_def = saved_model_utils.get_meta_graph_def(
      saved_model_dir, ','.join(tags)
  )
  _show_inputs_outputs_mgd(meta_graph_def, signature_def_key, indent)


def _show_defined_functions(saved_model_dir, meta_graphs):
  # [FUNCTION DOC]
  # HELPER: prints the object-graph section of the show-all output.
  # Takes the SavedModel directory and the list of its MetaGraphDef messages. Returns early,
  # printing nothing, if no meta graph carries an object graph. Otherwise loads the SavedModel
  # in a fresh graph, skipping checkpoint restoration; if loading fails it prints "N/A" with a
  # reason (custom ops not registered, or unknown reasons) and returns, swallowing the error.
  # On success it walks the root object's children sorted by name, and for each concrete or
  # polymorphic function prints its name and, per concrete function, the numbered option with
  # its positional and named arguments via the argument printer. Non-function children are
  # skipped. Uses private framework APIs. Side effects: prints, and loads the model.
  """Prints the callable concrete and polymorphic functions of the Saved Model.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
    meta_graphs: Already-extracted MetaGraphDef of the SavedModel.
  """
  has_object_graph_def = False

  for meta_graph_def in meta_graphs:
    has_object_graph_def |= meta_graph_def.HasField('object_graph_def')
  if not has_object_graph_def:
    return
  print('\nConcrete Functions:', end='')
  try:
    with ops_lib.Graph().as_default():
      trackable_object = load.load(
          saved_model_dir,
          options=load_options.LoadOptions(experimental_skip_checkpoint=True))
  except Exception as e:  # pylint: disable=broad-exception-caught
    if 'Op type not registered' in str(e):
      error = 'the existence of custom ops in the SavedModel'
    else:
      error = 'unknown reasons'
    print(f' N/A (could not be listed due to {error})')
    return

  children = list(
      save._AugmentedGraphView(trackable_object)  # pylint: disable=protected-access
      .list_children(trackable_object))
  children = sorted(children, key=lambda x: x.name)
  for name, child in children:
    concrete_functions = []
    if isinstance(child, defun.ConcreteFunction):
      concrete_functions.append(child)
    elif isinstance(child, def_function.Function):
      concrete_functions.extend(
          child._list_all_concrete_functions_for_serialization())  # pylint: disable=protected-access
    else:
      continue
    print('\n  Function Name: \'%s\'' % name)
    concrete_functions = sorted(concrete_functions, key=lambda x: x.name)
    for index, concrete_function in enumerate(concrete_functions, 1):
      args, kwargs = None, None
      if concrete_function.structured_input_signature:
        args, kwargs = concrete_function.structured_input_signature
      elif concrete_function._arg_keywords:  # pylint: disable=protected-access
        # For pure ConcreteFunctions we might have nothing better than
        # _arg_keywords.
        args = concrete_function._arg_keywords  # pylint: disable=protected-access
      if args:
        print('    Option #%d' % index)
        print('      Callable with:')
        _print_args(args, indent=4)
      if kwargs:
        _print_args(kwargs, 'Named Argument', indent=4)


def _print_args(arguments, argument_type='Argument', indent=0):
  # [FUNCTION DOC]
  # HELPER: formats function argument descriptions for _show_defined_functions.
  # Takes a sequence of arguments (strings, tensor specs, iterables, dictionaries or anything
  # else), a label such as "Argument" or "Named Argument", and an indent level. For each element
  # it prints, when the indent level is exactly four, a numbered heading; then strings as-is,
  # tensor specs as name and representation, iterables and dictionaries as type name plus the
  # listed values, and other values as type name plus text. Returns nothing; prints to standard
  # output, using backspace characters to erase the trailing separator.
  # Contains two nested helpers.
  """Formats and prints the argument of the concrete functions defined in the model.

  Args:
    arguments: Arguments to format print.
    argument_type: Type of arguments.
    indent: How far (in increments of 2 spaces) to indent each line of
     output.
  """
  indent_str = '  ' * indent

  def _maybe_add_quotes(value):
    # [FUNCTION DOC]
    # HELPER: renders a value as text, wrapping it in single quotes only when it is a string.
    # Takes any value and returns the resulting text. No side effects.
    is_quotes = '\'' * isinstance(value, str)
    return is_quotes + str(value) + is_quotes

  def in_print(s, end='\n'):
    # [FUNCTION DOC]
    # HELPER: prints text s prefixed by the indentation string, ending with the given
    # terminator (default newline). Returns nothing; prints to standard output.
    print(indent_str + s, end=end)

  for index, element in enumerate(arguments, 1):
    if indent == 4:
      in_print('%s #%d' % (argument_type, index))
    if isinstance(element, str):
      in_print('  %s' % element)
    elif isinstance(element, tensor_spec.TensorSpec):
      print((indent + 1) * '  ' + '%s: %s' % (element.name, repr(element)))
    elif (isinstance(element, collections_abc.Iterable) and
          not isinstance(element, dict)):
      in_print('  DType: %s' % type(element).__name__)
      in_print('  Value: [', end='')
      for value in element:
        print('%s' % _maybe_add_quotes(value), end=', ')
      print('\b\b]')
    elif isinstance(element, dict):
      in_print('  DType: %s' % type(element).__name__)
      in_print('  Value: {', end='')
      for (key, value) in element.items():
        print('\'%s\': %s' % (str(key), _maybe_add_quotes(value)), end=', ')
      print('\b\b}')
    else:
      in_print('  DType: %s' % type(element).__name__)
      in_print('  Value: %s' % str(element))


def _print_tensor_info(tensor_info, indent=0):
  # [FUNCTION DOC]
  # HELPER: prints one tensor description for the signature listing.
  # Takes a TensorInfo message and an indent level (two spaces per level, default zero).
  # Prints three lines: dtype (symbolic type name looked up from the numeric enum), shape (a
  # parenthesized comma-separated list of dimension sizes, or the word unknown_rank), and the
  # tensor name. Returns nothing; prints to standard output. Contains a nested print helper.
  """Prints details of the given tensor_info.

  Args:
    tensor_info: TensorInfo object to be printed.
    indent: How far (in increments of 2 spaces) to indent each line output
  """
  indent_str = '  ' * indent
  def in_print(s):
    # [FUNCTION DOC]
    # HELPER: prints text s prefixed by the enclosing call's indentation string.
    # Returns nothing; prints to standard output.
    print(indent_str + s)

  in_print('    dtype: ' +
           {value: key
            for (key, value) in types_pb2.DataType.items()}[tensor_info.dtype])
  # Display shape as tuple.
  if tensor_info.tensor_shape.unknown_rank:
    shape = 'unknown_rank'
  else:
    dims = [str(dim.size) for dim in tensor_info.tensor_shape.dim]
    shape = ', '.join(dims)
    shape = '(' + shape + ')'
  in_print('    shape: ' + shape)
  in_print('    name: ' + tensor_info.name)


def _show_all(saved_model_dir):
  # [FUNCTION DOC]
  # HELPER: implements the all option of the show command.
  # Reads the whole SavedModel in saved_model_dir and, for every meta graph sorted by its tags,
  # prints the tag-set heading, then for each signature key (sorted) the signature's inputs and
  # outputs at indent level one, then the op listing; finally prints the defined concrete
  # functions section. Returns nothing; prints to standard output. Read errors propagate.
  """Prints tag-set, ops, SignatureDef, and Inputs/Outputs of SavedModel.

  Prints all tag-set, ops, SignatureDef and Inputs/Outputs information stored in
  SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect.
  """
  saved_model = saved_model_utils.read_saved_model(saved_model_dir)
  for meta_graph_def in sorted(
      saved_model.meta_graphs,
      key=lambda meta_graph_def: list(meta_graph_def.meta_info_def.tags),
  ):
    tag_set = meta_graph_def.meta_info_def.tags
    print("\nMetaGraphDef with tag-set: '%s' "
          "contains the following SignatureDefs:" % ', '.join(tag_set))

    tag_set = ','.join(tag_set)
    signature_def_map = meta_graph_def.signature_def
    for signature_def_key in sorted(signature_def_map.keys()):
      print('\nsignature_def[\'' + signature_def_key + '\']:')
      _show_inputs_outputs_mgd(meta_graph_def, signature_def_key, indent=1)
    _show_ops_in_metagraph_mgd(meta_graph_def)
  _show_defined_functions(saved_model_dir, saved_model.meta_graphs)


def get_meta_graph_def(saved_model_dir, tag_set):
  # [FUNCTION DOC]
  # Deprecated public wrapper kept for backward compatibility. Forwards both arguments (the
  # SavedModel directory and a comma-separated tag-set string) unchanged to the saved-model
  # utilities function of the same name and returns the matching MetaGraphDef. Errors (for
  # example a missing tag-set) come from the delegate, not from this function. No side effects.
  """DEPRECATED: Use saved_model_utils.get_meta_graph_def instead.

  Gets MetaGraphDef from SavedModel. Returns the MetaGraphDef for the given
  tag-set and SavedModel directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect or execute.
    tag_set: Group of tag(s) of the MetaGraphDef to load, in string format,
        separated by ','. For tag-set contains multiple tags, all tags must be
        passed in.

  Raises:
    RuntimeError: An error when the given tag-set does not exist in the
        SavedModel.

  Returns:
    A MetaGraphDef corresponding to the tag-set.
  """
  return saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)


def get_signature_def_map(saved_model_dir, tag_set):
  # [FUNCTION DOC]
  # Loads the MetaGraphDef for the comma-separated tag_set from the SavedModel directory
  # (without trimming or validating the tag string) and returns its signature map, a mapping
  # from signature key strings to SignatureDef messages. Loading errors propagate.
  # No side effects beyond reading the model.
  """Gets SignatureDef map from a MetaGraphDef in a SavedModel.

  Returns the SignatureDef map for the given tag-set in the SavedModel
  directory.

  Args:
    saved_model_dir: Directory containing the SavedModel to inspect or execute.
    tag_set: Group of tag(s) of the MetaGraphDef with the SignatureDef map, in
        string format, separated by ','. For tag-set contains multiple tags, all
        tags must be passed in.

  Returns:
    A SignatureDef map that maps from string keys to SignatureDefs.
  """
  meta_graph = saved_model_utils.get_meta_graph_def(saved_model_dir, tag_set)
  return meta_graph.signature_def


def _get_op_denylist_set(op_denylist):
  # [FUNCTION DOC]
  # HELPER: parses the value of the op denylist flag for the scan command.
  # Takes a comma-separated string of op names, strips whitespace from each item and discards
  # empty items, so an empty string means an empty denylist. Returns a set of op names.
  # Has no docstring of its own; pure function.
  # Note: Discard empty ops so that "" can mean the empty denylist set.
  set_of_denylisted_ops = {
      op.strip() for op in op_denylist.split(',') if op and op.strip()
  }
  return set_of_denylisted_ops


def scan_meta_graph_def(meta_graph_def, op_denylist):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the "does not contain" message always says "default denylisted ops" even when
  # the caller supplied a custom denylist.
  # Computes the ops used by the meta graph's graph definition, intersects them with the given
  # set of denylisted op names, and prints either the offending ops (with the meta graph's tags)
  # or a success message that echoes the whole denylist. Returns nothing; prints to standard
  # output. A TODO in the body notes that richer warnings are not yet printed.
  """Scans meta_graph_def and reports if there are ops on denylist.

  Print ops if they are on denylist, or print success if no denylisted ops
  found.

  Args:
    meta_graph_def: MetaGraphDef protocol buffer.
    op_denylist: set of ops to scan for.
  """
  ops_in_metagraph = set(
      meta_graph_lib.ops_used_by_graph_def(meta_graph_def.graph_def))
  denylisted_ops = op_denylist & ops_in_metagraph
  if denylisted_ops:
    # TODO(yifeif): print more warnings
    print(
        'MetaGraph with tag set %s contains the following denylisted ops:' %
        meta_graph_def.meta_info_def.tags, denylisted_ops)
  else:
    print(
        'MetaGraph with tag set %s does not contain the default denylisted ops:'
        % meta_graph_def.meta_info_def.tags, op_denylist)


def run_saved_model_with_feed_dict(saved_model_dir,
                                   tag_set,
                                   signature_def_key,
                                   input_tensor_key_feed_dict,
                                   outdir,
                                   overwrite_flag,
                                   worker=None,
                                   init_tpu=False,
                                   use_tfrt=False,
                                   tf_debug=False):
  # [FUNCTION DOC]
  # Executes one signature of a SavedModel in a TF1-style session and reports the results.
  # Inputs: saved_model_dir (directory), tag_set (comma-separated string, sanitized here),
  # signature_def_key, input_tensor_key_feed_dict (mapping from input keys to array values),
  # outdir (directory for saved outputs, or empty/null to skip saving), overwrite_flag (allow
  # replacing existing output files), and optional worker (session target, default none),
  # init_tpu (initialize the TPU system first, default false), use_tfrt (request a TFRT session
  # through the config, default false) and tf_debug (wrap the session in the command-line
  # debugger, default false).
  # Behavior: loads the meta graph, verifies every supplied input key exists in the signature,
  # translates keys to tensor names, loads the SavedModel into a new session on a fresh graph,
  # runs all outputs (ordered by sorted output key) and prints each result. If outdir is set it
  # creates the directory when needed and writes one array file per output named after the key.
  # Raises a value error for an invalid input key, and a runtime error when an output file
  # exists and overwrite is not enabled (earlier outputs may already have been written).
  # Side effects: creates a session, may initialize TPUs, prints, logs and writes files.
  """Runs SavedModel and fetch all outputs.


  Runs the input dictionary through the MetaGraphDef within a SavedModel
  specified by the given tag_set and SignatureDef. Also save the outputs to file
  if outdir is not None.

  Args:
    saved_model_dir: Directory containing the SavedModel to execute.
    tag_set: Group of tag(s) of the MetaGraphDef with the SignatureDef map, in
        string format, separated by ','. For tag-set contains multiple tags, all
        tags must be passed in.
    signature_def_key: A SignatureDef key string.
    input_tensor_key_feed_dict: A dictionary maps input keys to numpy ndarrays.
    outdir: A directory to save the outputs to. If the directory doesn't exist,
        it will be created.
    overwrite_flag: A boolean flag to allow overwrite output file if file with
        the same name exists.
    worker: If provided, the session will be run on the worker.  Valid worker
        specification is a bns or gRPC path.
    init_tpu: If true, the TPU system will be initialized after the session
        is created.
    use_tfrt: If true, TFRT session will be used.
    tf_debug: A boolean flag to use TensorFlow Debugger (TFDBG) to observe the
        intermediate Tensor values and runtime GraphDefs while running the
        SavedModel.

  Raises:
    ValueError: When any of the input tensor keys is not valid.
    RuntimeError: An error when output file already exists and overwrite is not
    enabled.
  """
  # Get a list of output tensor names.
  tags = _sanitize_nonempty_str_list(tag_set.split(','), 'tag_set')
  meta_graph_def = saved_model_utils.get_meta_graph_def(
      saved_model_dir, ','.join(tags)
  )

  # Re-create feed_dict based on input tensor name instead of key as session.run
  # uses tensor name.
  inputs_tensor_info = _get_inputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key)

  # Check if input tensor keys are valid.
  for input_key_name in input_tensor_key_feed_dict.keys():
    if input_key_name not in inputs_tensor_info:
      raise ValueError(
          '"%s" is not a valid input key. Please choose from %s, or use '
          '--show option.' %
          (input_key_name, '"' + '", "'.join(inputs_tensor_info.keys()) + '"'))

  inputs_feed_dict = {
      inputs_tensor_info[key].name: tensor
      for key, tensor in input_tensor_key_feed_dict.items()
  }
  # Get outputs
  outputs_tensor_info = _get_outputs_tensor_info_from_meta_graph_def(
      meta_graph_def, signature_def_key)
  # Sort to preserve order because we need to go from value to key later.
  output_tensor_keys_sorted = sorted(outputs_tensor_info.keys())
  output_tensor_names_sorted = [
      outputs_tensor_info[tensor_key].name
      for tensor_key in output_tensor_keys_sorted
  ]

  config = None
  if use_tfrt:
    logging.info('Using TFRT session.')
    config = config_pb2.ConfigProto(
        experimental=config_pb2.ConfigProto.Experimental(use_tfrt=True))
  with session.Session(worker, graph=ops_lib.Graph(), config=config) as sess:
    if init_tpu:
      print('Initializing TPU System ...')
      # This is needed for freshly started worker, or if the job
      # restarts after a preemption.
      sess.run(tpu.initialize_system())

    loader.load(sess, tags, saved_model_dir)

    if tf_debug:
      sess = local_cli_wrapper.LocalCLIDebugWrapperSession(sess)

    outputs = sess.run(output_tensor_names_sorted, feed_dict=inputs_feed_dict)

    for i, output in enumerate(outputs):
      output_tensor_key = output_tensor_keys_sorted[i]
      print('Result for output key %s:\n%s' % (output_tensor_key, output))

      # Only save if outdir is specified.
      if outdir:
        # Create directory if outdir does not exist
        if not os.path.isdir(outdir):
          os.makedirs(outdir)
        output_full_path = os.path.join(outdir, output_tensor_key + '.npy')

        # If overwrite not enabled and file already exist, error out
        if not overwrite_flag and os.path.exists(output_full_path):
          raise RuntimeError(
              'Output file %s already exists. Add \"--overwrite\" to overwrite'
              ' the existing output files.' % output_full_path)

        np.save(output_full_path, output)
        print('Output %s is saved to %s' % (output_tensor_key,
                                            output_full_path))


def preprocess_inputs_arg_string(inputs_str):
  # [FUNCTION DOC]
  # Parses the value of the inputs flag. The string holds entries separated by semicolons, each
  # either key equals filename or key equals filename followed by a bracketed variable name.
  # Empty entries are skipped. Returns a mapping from input key to a pair of filename and
  # variable name (the variable name is null when no brackets were given). Raises a runtime
  # error naming the entry when it matches neither format. Pure; touches no files.
  """Parses input arg into dictionary that maps input to file/variable tuple.

  Parses input string in the format of, for example,
  "input1=filename1[variable_name1],input2=filename2" into a
  dictionary looks like
  {'input_key1': (filename1, variable_name1),
   'input_key2': (file2, None)}
  , which maps input keys to a tuple of file name and variable name(None if
  empty).

  Args:
    inputs_str: A string that specified where to load inputs. Inputs are
    separated by semicolons.
        * For each input key:
            '<input_key>=<filename>' or
            '<input_key>=<filename>[<variable_name>]'
        * The optional 'variable_name' key will be set to None if not specified.

  Returns:
    A dictionary that maps input keys to a tuple of file name and variable name.

  Raises:
    RuntimeError: An error when the given input string is in a bad format.
  """
  input_dict = {}
  inputs_raw = inputs_str.split(';')
  for input_raw in filter(bool, inputs_raw):  # skip empty strings
    # Format of input=filename[variable_name]'
    match = re.match(r'([^=]+)=([^\[\]]+)\[([^\[\]]+)\]$', input_raw)

    if match:
      input_dict[match.group(1)] = match.group(2), match.group(3)
    else:
      # Format of input=filename'
      match = re.match(r'([^=]+)=([^\[\]]+)$', input_raw)
      if match:
        input_dict[match.group(1)] = match.group(2), None
      else:
        raise RuntimeError(
            '--inputs "%s" format is incorrect. Please follow'
            '"<input_key>=<filename>", or'
            '"<input_key>=<filename>[<variable_name>]"' % input_raw)

  return input_dict


def preprocess_input_exprs_arg_string(input_exprs_str, safe=True):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the format check tests whether an equals sign occurs anywhere in the whole
  # input string rather than in the current entry, so an entry lacking one is not rejected when
  # another entry has one; the split then fails with an unhelpful unpacking error. Also, the
  # flag help promises numpy access, but the only callers in this file use the default safe
  # mode, which accepts literals only.
  # Parses semicolon-separated key equals expression entries (empty entries skipped). Splits each
  # entry at the first equals sign. With safe true (default) the expression is evaluated as a
  # literal only; failures raise a runtime error chained to the cause. With safe false the
  # expression is evaluated with arbitrary code execution (numpy available), which is a
  # code-injection risk. Returns a mapping from input key to evaluated value. Raises a runtime
  # error when the string lacks any equals sign.
  """Parses input arg into dictionary that maps input key to python expression.

  Parses input string in the format of 'input_key=<python expression>' into a
  dictionary that maps each input_key to its python expression.

  Args:
    input_exprs_str: A string that specifies python expression for input keys.
      Each input is separated by semicolon. For each input key:
        'input_key=<python expression>'
    safe: Whether to evaluate the python expression as literals or allow
      arbitrary calls (e.g. numpy usage).

  Returns:
    A dictionary that maps input keys to their values.

  Raises:
    RuntimeError: An error when the given input string is in a bad format.
  """
  input_dict = {}

  for input_raw in filter(bool, input_exprs_str.split(';')):
    if '=' not in input_exprs_str:
      raise RuntimeError('--input_exprs "%s" format is incorrect. Please follow'
                         '"<input_key>=<python expression>"' % input_exprs_str)
    input_key, expr = input_raw.split('=', 1)
    if safe:
      try:
        input_dict[input_key] = ast.literal_eval(expr)
      except Exception as exc:
        raise RuntimeError(
            f'Expression "{expr}" is not a valid python literal.') from exc
    else:
      # ast.literal_eval does not work with numpy expressions
      input_dict[input_key] = eval(expr)  # pylint: disable=eval-used
  return input_dict


def preprocess_input_examples_arg_string(input_examples_str):
  # [FUNCTION DOC]
  # Parses the input examples flag. Reuses the expression parser in safe literal mode to turn
  # each key's text into a list of feature dictionaries, then converts every dictionary into a
  # serialized tf.Example byte string. Returns a mapping from input key to list of serialized
  # examples. Raises a value error when a key's value is not a list; errors for bad feature
  # values come from the example builder. Pure apart from building protocol buffers.
  """Parses input into dict that maps input keys to lists of tf.Example.

  Parses input string in the format of 'input_key1=[{feature_name:
  feature_list}];input_key2=[{feature_name:feature_list}];' into a dictionary
  that maps each input_key to its list of serialized tf.Example.

  Args:
    input_examples_str: A string that specifies a list of dictionaries of
    feature_names and their feature_lists for each input.
    Each input is separated by semicolon. For each input key:
      'input=[{feature_name1: feature_list1, feature_name2:feature_list2}]'
      items in feature_list can be the type of float, int, long or str.

  Returns:
    A dictionary that maps input keys to lists of serialized tf.Example.

  Raises:
    ValueError: An error when the given tf.Example is not a list.
  """
  input_dict = preprocess_input_exprs_arg_string(input_examples_str)
  for input_key, example_list in input_dict.items():
    if not isinstance(example_list, list):
      raise ValueError(
          'tf.Example input must be a list of dictionaries, but "%s" is %s' %
          (example_list, type(example_list)))
    input_dict[input_key] = [
        _create_example_string(example) for example in example_list
    ]
  return input_dict


def _create_example_string(example_dict):
  # [FUNCTION DOC]
  # HELPER: builds the serialized examples for the input examples flag.
  # Takes a mapping from feature names to non-empty lists of values. The type of the first
  # element of each list picks the feature kind: float gives a float list, text gives a byte
  # list (UTF-8 encoded), bytes gives a byte list, integer gives an int64 list. Returns the
  # serialized tf.Example bytes. Raises a value error when a value is not a list or the first
  # element has an unsupported type; an empty list causes an index error. Note that booleans
  # count as integers.
  """Create a serialized tf.example from feature dictionary."""
  example = example_pb2.Example()
  for feature_name, feature_list in example_dict.items():
    if not isinstance(feature_list, list):
      raise ValueError('feature value must be a list, but %s: "%s" is %s' %
                       (feature_name, feature_list, type(feature_list)))
    if isinstance(feature_list[0], float):
      example.features.feature[feature_name].float_list.value.extend(
          feature_list)
    elif isinstance(feature_list[0], str):
      example.features.feature[feature_name].bytes_list.value.extend(
          [f.encode('utf8') for f in feature_list])
    elif isinstance(feature_list[0], bytes):
      example.features.feature[feature_name].bytes_list.value.extend(
          feature_list)
    elif isinstance(feature_list[0], int):
      example.features.feature[feature_name].int64_list.value.extend(
          feature_list)
    else:
      raise ValueError(
          'Type %s for value %s is not supported for tf.train.Feature.' %
          (type(feature_list[0]), feature_list[0]))
  return example.SerializeToString()


def load_inputs_from_input_arg_string(inputs_str, input_exprs_str,
                                      input_examples_str):
  # [FUNCTION DOC]
  # Builds the feed mapping for the run command from the three input flags: inputs_str (files
  # to load with numpy, optionally selecting a variable inside an archive or pickle),
  # input_exprs_str (literal expressions) and input_examples_str (feature lists to convert to
  # serialized examples). Files are opened through the TensorFlow file layer with pickle loading
  # allowed, which can execute code from untrusted files. A variable name given for a single
  # array file is ignored with a warning; for archives or pickles it selects the entry, and a
  # missing entry raises a runtime error. Without a variable name an archive must contain
  # exactly one array, otherwise a runtime error is raised. Expressions override file inputs and
  # examples override both, each logging a warning on conflict. Returns a mapping from input key
  # to value. Parsing errors from the helper parsers propagate. Reads files and logs warnings.
  """Parses input arg strings and create inputs feed_dict.


  Parses '--inputs' string for inputs to be loaded from file, and parses
  '--input_exprs' string for inputs to be evaluated from python expression.
  '--input_examples' string for inputs to be created from tf.example feature
  dictionary list.

  Args:
    inputs_str: A string that specified where to load inputs. Each input is
        separated by semicolon.
        * For each input key:
            '<input_key>=<filename>' or
            '<input_key>=<filename>[<variable_name>]'
        * The optional 'variable_name' key will be set to None if not specified.
        * File specified by 'filename' will be loaded using numpy.load. Inputs
            can be loaded from only .npy, .npz or pickle files.
        * The "[variable_name]" key is optional depending on the input file type
            as descripted in more details below.
        When loading from a npy file, which always contains a numpy ndarray, the
        content will be directly assigned to the specified input tensor. If a
        variable_name is specified, it will be ignored and a warning will be
        issued.
        When loading from a npz zip file, user can specify which variable within
        the zip file to load for the input tensor inside the square brackets. If
        nothing is specified, this function will check that only one file is
        included in the zip and load it for the specified input tensor.
        When loading from a pickle file, if no variable_name is specified in the
        square brackets, whatever that is inside the pickle file will be passed
        to the specified input tensor, else SavedModel CLI will assume a
        dictionary is stored in the pickle file and the value corresponding to
        the variable_name will be used.
    input_exprs_str: A string that specifies python expressions for inputs.
        * In the format of: '<input_key>=<python expression>'.
        * numpy module is available as np.
    input_examples_str: A string that specifies tf.Example with dictionary.
        * In the format of: '<input_key>=<[{feature:value list}]>'

  Returns:
    A dictionary that maps input tensor keys to numpy ndarrays.

  Raises:
    RuntimeError: An error when a key is specified, but the input file contains
        multiple numpy ndarrays, none of which matches the given key.
    RuntimeError: An error when no key is specified, but the input file contains
        more than one numpy ndarrays.
  """
  tensor_key_feed_dict = {}

  inputs = preprocess_inputs_arg_string(inputs_str)
  input_exprs = preprocess_input_exprs_arg_string(input_exprs_str)
  input_examples = preprocess_input_examples_arg_string(input_examples_str)

  for input_tensor_key, (filename, variable_name) in inputs.items():
    data = np.load(file_io.FileIO(filename, mode='rb'), allow_pickle=True)  # pylint: disable=unexpected-keyword-arg

    # When a variable_name key is specified for the input file
    if variable_name:
      # if file contains a single ndarray, ignore the input name
      if isinstance(data, np.ndarray):
        logging.warn(
            'Input file %s contains a single ndarray. Name key \"%s\" ignored.'
            % (filename, variable_name))
        tensor_key_feed_dict[input_tensor_key] = data
      else:
        if variable_name in data:
          tensor_key_feed_dict[input_tensor_key] = data[variable_name]
        else:
          raise RuntimeError(
              'Input file %s does not contain variable with name \"%s\".' %
              (filename, variable_name))
    # When no key is specified for the input file.
    else:
      # Check if npz file only contains a single numpy ndarray.
      if isinstance(data, np.lib.npyio.NpzFile):
        variable_name_list = data.files
        if len(variable_name_list) != 1:
          raise RuntimeError(
              'Input file %s contains more than one ndarrays. Please specify '
              'the name of ndarray to use.' % filename)
        tensor_key_feed_dict[input_tensor_key] = data[variable_name_list[0]]
      else:
        tensor_key_feed_dict[input_tensor_key] = data

  # When input is a python expression:
  for input_tensor_key, py_expr_evaluated in input_exprs.items():
    if input_tensor_key in tensor_key_feed_dict:
      logging.warn(
          'input_key %s has been specified with both --inputs and --input_exprs'
          ' options. Value in --input_exprs will be used.' % input_tensor_key)
    tensor_key_feed_dict[input_tensor_key] = py_expr_evaluated

  # When input is a tf.Example:
  for input_tensor_key, example in input_examples.items():
    if input_tensor_key in tensor_key_feed_dict:
      logging.warn(
          'input_key %s has been specified in multiple options. Value in '
          '--input_examples will be used.' % input_tensor_key)
    tensor_key_feed_dict[input_tensor_key] = example
  return tensor_key_feed_dict


def show():
  # [FUNCTION DOC]
  # Handler for the show command; takes no arguments and reads the parsed flags. With the all
  # flag it prints everything. Otherwise, without a tag-set it prints all tag-sets (and a hint
  # if list-ops was requested); with a tag-set it optionally prints the ops, then either the
  # signature keys (no signature given) or the inputs and outputs of the named signature.
  # Returns nothing; prints to standard output; errors from the helpers propagate.
  """Function triggered by show command."""
  # If all tag is specified, display all information.
  if _SMCLI_ALL.value:
    _show_all(_SMCLI_DIR.value)
  else:
    # If no tag is specified, display all tag_sets.
    # If a tag set is specified:
    # # If list_ops is set, display all ops in the specified MetaGraphDef.
    # # If no signature_def key is specified, display all SignatureDef keys.
    # # If a signature_def is specified, show its corresponding input output
    # # tensor information.
    if _SMCLI_TAG_SET.value is None:
      if _SMCLI_LIST_OPS.value:
        print('--list_ops must be paired with a tag-set or with --all.')
      _show_tag_sets(_SMCLI_DIR.value)
    else:
      if _SMCLI_LIST_OPS.value:
        _show_ops_in_metagraph(_SMCLI_DIR.value, _SMCLI_TAG_SET.value)
      if _SMCLI_SIGNATURE_DEF.value is None:
        _show_signature_def_map_keys(_SMCLI_DIR.value, _SMCLI_TAG_SET.value)
      else:
        _show_inputs_outputs(
            _SMCLI_DIR.value, _SMCLI_TAG_SET.value, _SMCLI_SIGNATURE_DEF.value)


def run():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the error occurs when neither inputs nor input expressions
  # are passed, but the code also accepts input examples as sufficient.
  # Handler for the run command; takes no arguments and reads flags. Raises an attribute error if
  # none of the inputs, input expressions or input examples flags is set. Otherwise builds the
  # feed mapping from those flags and executes the model with the directory, tag-set, signature,
  # output directory, overwrite, worker, TPU, TFRT and debugger flags. Returns nothing; side
  # effects are those of the execution routine (session, prints, files).
  """Function triggered by run command.

  Raises:
    AttributeError: An error when neither --inputs nor --input_exprs is passed
    to run command.
  """
  if (
      not _SMCLI_INPUTS.value
      and not _SMCLI_INPUT_EXPRS.value
      and not _SMCLI_INPUT_EXAMPLES.value
  ):
    raise AttributeError(
        'At least one of --inputs, --input_exprs or --input_examples must be '
        'required')
  tensor_key_feed_dict = load_inputs_from_input_arg_string(
      _SMCLI_INPUTS.value,
      _SMCLI_INPUT_EXPRS.value,
      _SMCLI_INPUT_EXAMPLES.value)
  run_saved_model_with_feed_dict(
      _SMCLI_DIR.value,
      _SMCLI_TAG_SET.value,
      _SMCLI_SIGNATURE_DEF.value,
      tensor_key_feed_dict,
      _SMCLI_OUTDIR.value,
      _SMCLI_OVERWRITE.value,
      worker=_SMCLI_WORKER.value,
      init_tpu=_SMCLI_INIT_TPU.value,
      use_tfrt=_SMCLI_USE_TFRT.value,
      tf_debug=_SMCLI_TF_DEBUG.value)


def scan():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the denylist flag help says an empty string means an empty denylist, but the
  # code tests the flag value for truthiness, so an empty string is treated like an unset flag
  # and the default denylist is used.
  # Handler for the scan command; takes no arguments and reads flags. If a tag-set is given
  # (sanitized and rejoined) only that meta graph is scanned, otherwise every meta graph in the
  # SavedModel is scanned. The denylist is the parsed flag value when non-empty, else the
  # default set (write file, read file, print). Returns nothing; prints findings. Errors from
  # loading or sanitizing propagate.
  """Function triggered by scan command."""
  if _SMCLI_TAG_SET.value and _SMCLI_OP_DENYLIST.value:
    tags = ','.join(
        _sanitize_nonempty_str_list(_SMCLI_TAG_SET.value.split(','), 'tag_set')
    )
    scan_meta_graph_def(
        saved_model_utils.get_meta_graph_def(_SMCLI_DIR.value, tags),
        _get_op_denylist_set(_SMCLI_OP_DENYLIST.value),
    )
  elif _SMCLI_TAG_SET.value:
    tags = ','.join(
        _sanitize_nonempty_str_list(_SMCLI_TAG_SET.value.split(','), 'tag_set')
    )
    scan_meta_graph_def(
        saved_model_utils.get_meta_graph_def(_SMCLI_DIR.value, tags),
        _OP_DENYLIST,
    )
  else:
    saved_model = saved_model_utils.read_saved_model(_SMCLI_DIR.value)
    if _SMCLI_OP_DENYLIST.value:
      for meta_graph_def in saved_model.meta_graphs:
        scan_meta_graph_def(meta_graph_def,
                            _get_op_denylist_set(_SMCLI_OP_DENYLIST.value))
    else:
      for meta_graph_def in saved_model.meta_graphs:
        scan_meta_graph_def(meta_graph_def, _OP_DENYLIST)


def convert_with_tensorrt():
  # [FUNCTION DOC]
  # Handler for the convert tensorrt command; takes no arguments and reads flags. Imports the
  # TensorRT conversion module lazily because it fails when TensorRT is missing. For a TF2
  # model (default) it builds conversion parameters from the workspace size, precision and
  # minimum segment size flags, converts with the V2 converter and saves to the output
  # directory; any conversion failure is re-raised as a runtime error suggesting the TF1 flag.
  # For a TF1 model it calls the legacy inference graph creation with dynamic ops and batch size
  # one. Tags come from the sanitized tag-set flag (a null tag-set would fail). Returns nothing;
  # writes a converted SavedModel.
  """Function triggered by 'convert tensorrt' command."""
  # Import here instead of at top, because this will crash if TensorRT is
  # not installed
  from tensorflow.python.compiler.tensorrt import trt_convert as trt  # pylint: disable=g-import-not-at-top

  if not _SMCLI_CONVERT_TF1_MODEL.value:
    params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
        max_workspace_size_bytes=_SMCLI_MAX_WORKSPACE_SIZE_BYTES.value,
        precision_mode=_SMCLI_PRECISION_MODE.value,
        minimum_segment_size=_SMCLI_MINIMUM_SEGMENT_SIZE.value)
    try:
      converter = trt.TrtGraphConverterV2(
          input_saved_model_dir=_SMCLI_DIR.value,
          input_saved_model_tags=_sanitize_nonempty_str_list(
              _SMCLI_TAG_SET.value.split(','), 'tag_set'
          ),
          **params._asdict(),
      )
      converter.convert()
    except Exception as exc:
      raise RuntimeError(
          '{}. Try passing "--convert_tf1_model=True".'.format(exc)) from exc
    converter.save(output_saved_model_dir=_SMCLI_OUTPUT_DIR.value)
  else:
    trt.create_inference_graph(
        None,
        None,
        max_batch_size=1,
        max_workspace_size_bytes=_SMCLI_MAX_WORKSPACE_SIZE_BYTES.value,
        precision_mode=_SMCLI_PRECISION_MODE.value,
        minimum_segment_size=_SMCLI_MINIMUM_SEGMENT_SIZE.value,
        is_dynamic_op=True,
        input_saved_model_dir=_SMCLI_DIR.value,
        input_saved_model_tags=_sanitize_nonempty_str_list(
            _SMCLI_TAG_SET.value.split(','), 'tag_set'
        ),
        output_saved_model_dir=_SMCLI_OUTPUT_DIR.value,
    )


def freeze_model():
  # [FUNCTION DOC]
  # Handler for the freeze_model command; takes no arguments and reads flags. Chooses the
  # checkpoint path (flag, or the model's variables file by default) and the variables to feed:
  # none when the flag is empty, all (represented as null, resolved later) when it is the word
  # all in any case, otherwise the sanitized comma-separated list. Loads the meta graph for the
  # sanitized tag-set and asks the AOT compile module to freeze it for the signature key,
  # writing outputs under the output prefix. Returns nothing; writes files.
  """Function triggered by freeze_model command."""
  checkpoint_path = (
      _SMCLI_CHECKPOINT_PATH.value
      or os.path.join(_SMCLI_DIR.value, 'variables/variables'))
  if not _SMCLI_VARIABLES_TO_FEED.value:
    variables_to_feed = []
  elif _SMCLI_VARIABLES_TO_FEED.value.lower() == 'all':
    variables_to_feed = None  # We will identify them after.
  else:
    variables_to_feed = _sanitize_nonempty_str_list(
        _SMCLI_VARIABLES_TO_FEED.value.split(','), 'variables_to_feed'
    )

  saved_model_aot_compile.freeze_model(
      checkpoint_path=checkpoint_path,
      meta_graph_def=saved_model_utils.get_meta_graph_def(
          _SMCLI_DIR.value,
          ','.join(
              _sanitize_nonempty_str_list(
                  _SMCLI_TAG_SET.value.split(','), 'tag_set'
              )
          ),
      ),
      signature_def_key=_SMCLI_SIGNATURE_DEF_KEY.value,
      variables_to_feed=variables_to_feed,
      output_prefix=_SMCLI_OUTPUT_PREFIX.value,
  )


def aot_compile_cpu():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the required-flags table only demands the C++ class name for this command, yet
  # the directory, tag-set and output prefix are used unconditionally; omitting them fails later
  # with an unhelpful error.
  # Handler for the aot_compile_cpu command; takes no arguments and reads flags. Resolves the
  # checkpoint path and variables to feed exactly like the freeze handler, then calls the AOT
  # compile module with the meta graph for the sanitized tag-set, signature key, output prefix,
  # target triple, target CPU, C++ class name and a multithreading boolean (false when the flag
  # text is f, false or 0 in any case). Returns nothing; writes generated headers and objects.
  """Function triggered by aot_compile_cpu command."""
  checkpoint_path = (
      _SMCLI_CHECKPOINT_PATH.value
      or os.path.join(_SMCLI_DIR.value, 'variables/variables'))
  if not _SMCLI_VARIABLES_TO_FEED.value:
    variables_to_feed = []
  elif _SMCLI_VARIABLES_TO_FEED.value.lower() == 'all':
    variables_to_feed = None  # We will identify them after.
  else:
    variables_to_feed = _sanitize_nonempty_str_list(
        _SMCLI_VARIABLES_TO_FEED.value.split(','), 'variables_to_feed'
    )

  saved_model_aot_compile.aot_compile_cpu_meta_graph_def(
      checkpoint_path=checkpoint_path,
      meta_graph_def=saved_model_utils.get_meta_graph_def(
          _SMCLI_DIR.value,
          ','.join(
              _sanitize_nonempty_str_list(
                  _SMCLI_TAG_SET.value.split(','), 'tag_set'
              )
          ),
      ),
      signature_def_key=_SMCLI_SIGNATURE_DEF_KEY.value,
      variables_to_feed=variables_to_feed,
      output_prefix=_SMCLI_OUTPUT_PREFIX.value,
      target_triple=_SMCLI_TARGET_TRIPLE.value,
      target_cpu=_SMCLI_TARGET_CPU.value,
      cpp_class=_SMCLI_CPP_CLASS.value,
      multithreading=(
          _SMCLI_MULTITHREADING.value.lower() not in ('f', 'false', '0')
      ),
  )


def add_show_subparser(subparsers):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the help text says multiple tags are separated by semicolons, but the code
  # splits the tag-set on commas (and the example itself uses a comma).
  # Registers the show sub-parser on the given sub-parsers object, with a raw-text usage
  # description, and binds the show handler as its default function. Returns nothing.
  """Add parser for `show`."""
  show_msg = (
      'Usage examples:\n'
      'To show all tag-sets in a SavedModel:\n'
      '$saved_model_cli show --dir /tmp/saved_model\n\n'
      'To show all available SignatureDef keys in a '
      'MetaGraphDef specified by its tag-set:\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve\n\n'
      'For a MetaGraphDef with multiple tags in the tag-set, all tags must be '
      'passed in, separated by \';\':\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve,gpu\n\n'
      'To show all inputs and outputs TensorInfo for a specific'
      ' SignatureDef specified by the SignatureDef key in a'
      ' MetaGraph.\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve'
      ' --signature_def serving_default\n\n'
      'To show all ops in a MetaGraph.\n'
      '$saved_model_cli show --dir /tmp/saved_model --tag_set serve'
      ' --list_ops\n\n'
      'To show all available information in the SavedModel:\n'
      '$saved_model_cli show --dir /tmp/saved_model --all')
  parser_show = subparsers.add_parser(
      'show',
      description=show_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_show.set_defaults(func=show)


def add_run_subparser(subparsers):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the usage example invokes the show command instead of run, and separates
  # multiple inputs with a comma although the parser splits on semicolons.
  # Registers the run sub-parser on the given sub-parsers object with its raw-text usage
  # description and binds the run handler as default function. Returns nothing.
  """Add parser for `run`."""
  run_msg = ('Usage example:\n'
             'To run input tensors from files through a MetaGraphDef and save'
             ' the output tensors to files:\n'
             '$saved_model_cli show --dir /tmp/saved_model --tag_set serve \\\n'
             '   --signature_def serving_default \\\n'
             '   --inputs input1_key=/tmp/124.npz[x],input2_key=/tmp/123.npy '
             '\\\n'
             '   --input_exprs \'input3_key=np.ones(2)\' \\\n'
             '   --input_examples '
             '\'input4_key=[{"id":[26],"weights":[0.5, 0.5]}]\' \\\n'
             '   --outdir=/out\n\n'
             'For more information about input file format, please see:\n'
             'https://www.tensorflow.org/guide/saved_model_cli\n')
  parser_run = subparsers.add_parser(
      'run', description=run_msg, formatter_class=argparse.RawTextHelpFormatter)
  parser_run.set_defaults(func=run)


def add_scan_subparser(subparsers):
  # [FUNCTION DOC]
  # Registers the scan sub-parser on the given sub-parsers object with its raw-text usage
  # description and binds the scan handler as default function. Returns nothing.
  """Add parser for `scan`."""
  scan_msg = ('Usage example:\n'
              'To scan for default denylisted ops in SavedModel:\n'
              '$saved_model_cli scan --dir /tmp/saved_model\n'
              'To scan for a specific set of ops in SavedModel:\n'
              '$saved_model_cli scan --dir /tmp/saved_model --op_denylist '
              'OpName,OpName,OpName\n'
              'To scan a specific MetaGraph, pass in --tag_set\n')
  parser_scan = subparsers.add_parser(
      'scan',
      description=scan_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_scan.set_defaults(func=scan)


def add_convert_subparser(subparsers):
  # [FUNCTION DOC]
  # Registers the convert sub-parser on the given sub-parsers object, and nested under it a
  # tensorrt conversion method sub-parser whose default function is the TensorRT handler.
  # Only the nested parser binds a handler, so convert without a method has no function.
  # Returns nothing.
  """Add parser for `convert`."""
  convert_msg = ('Usage example:\n'
                 'To convert the SavedModel to one that have TensorRT ops:\n'
                 '$saved_model_cli convert \\\n'
                 '   --dir /tmp/saved_model \\\n'
                 '   --tag_set serve \\\n'
                 '   --output_dir /tmp/saved_model_trt \\\n'
                 '   tensorrt \n')
  parser_convert = subparsers.add_parser(
      'convert',
      description=convert_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  convert_subparsers = parser_convert.add_subparsers(
      title='conversion methods',
      description='valid conversion methods',
      help='the conversion to run with the SavedModel')
  parser_convert_with_tensorrt = convert_subparsers.add_parser(
      'tensorrt',
      description='Convert the SavedModel with Tensorflow-TensorRT integration',
      formatter_class=argparse.RawTextHelpFormatter)
  parser_convert_with_tensorrt.set_defaults(func=convert_with_tensorrt)


def add_freeze_model_subparser(subparsers):
  # [FUNCTION DOC]
  # Registers the freeze_model sub-parser on the given sub-parsers object with a usage
  # description (built by joining lines) and binds the freeze handler. Returns nothing.
  """Add parser for `freeze_model`."""
  compile_msg = '\n'.join(
      ['Usage example:',
       'To freeze a SavedModel in preparation for tfcompile:',
       '$saved_model_cli freeze_model \\',
       '   --dir /tmp/saved_model \\',
       '   --tag_set serve \\',
       '   --output_prefix /tmp/saved_model_xla_aot',
      ])

  parser_compile = subparsers.add_parser(
      'freeze_model',
      description=compile_msg,
      formatter_class=argparse.RawTextHelpFormatter)
  parser_compile.set_defaults(func=freeze_model)


def add_aot_compile_cpu_subparser(subparsers):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the usage example passes an output directory flag, but the command actually
  # reads the output prefix flag.
  # Registers the aot_compile_cpu sub-parser on the given sub-parsers object, with a usage
  # description that also explains XLA compilation options and the XLA flags environment
  # variable, and binds the AOT compile handler. Returns nothing.
  """Add parser for `aot_compile_cpu`."""
  compile_msg = '\n'.join(
      ['Usage example:',
       'To compile a SavedModel signature via (CPU) XLA AOT:',
       '$saved_model_cli aot_compile_cpu \\',
       '   --dir /tmp/saved_model \\',
       '   --tag_set serve \\',
       '   --output_dir /tmp/saved_model_xla_aot',
       '', '',
       'Note: Additional XLA compilation options are available by setting the ',
       'XLA_FLAGS environment variable.  See the XLA debug options flags for ',
       'all the options: ',
       '  {}'.format(_XLA_DEBUG_OPTIONS_URL),
       '',
       'For example, to disable XLA fast math when compiling:',
       '',
       'XLA_FLAGS="--xla_cpu_enable_fast_math=false" $saved_model_cli ',
       'aot_compile_cpu ...',
       '',
       'Some possibly useful flags:',
       '  --xla_cpu_enable_fast_math=false',
       '  --xla_force_host_platform_device_count=<num threads>',
       '    (useful in conjunction with disabling multi threading)'
      ])

  parser_compile = subparsers.add_parser(
      'aot_compile_cpu',
      description=compile_msg,
      formatter_class=argparse.RawTextHelpFormatter)

  parser_compile.set_defaults(func=aot_compile_cpu)


def create_parser():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a namespace of parsed arguments, but it returns
  # the argument parser object itself without parsing anything.
  # Builds the top-level absl-aware argument parser (conflicting options resolved in favor of
  # later ones, version option 0.1.0) and registers the show, run, scan, convert, aot_compile_cpu
  # and freeze_model sub-parsers. Takes no arguments; returns the parser.
  """Creates a parser that parse the command line arguments.

  Returns:
    A namespace parsed from command line arguments.
  """
  parser = argparse_flags.ArgumentParser(
      description='saved_model_cli: Command-line interface for SavedModel',
      conflict_handler='resolve')
  parser.add_argument('-v', '--version', action='version', version='0.1.0')

  subparsers = parser.add_subparsers(
      title='commands', description='valid commands', help='additional help')

  # show command
  add_show_subparser(subparsers)

  # run command
  add_run_subparser(subparsers)

  # scan command
  add_scan_subparser(subparsers)

  # tensorrt convert command
  add_convert_subparser(subparsers)

  # aot_compile_cpu command
  add_aot_compile_cpu_subparser(subparsers)

  # freeze_model command
  add_freeze_model_subparser(subparsers)
  return parser


def main():
  # [FUNCTION DOC]
  # Program entry point. Sets the logging verbosity to info and runs the absl application with
  # the nested main routine. Takes no arguments; the process exits through absl.
  logging.set_verbosity(logging.INFO)

  def smcli_main(argv):
    # [FUNCTION DOC]
    # Real command dispatcher run by absl with the command-line argument list. Builds the
    # parser, errors out when fewer than two arguments are present, marks the flags required by
    # the chosen command (the second argument; an unknown command name causes a key error) as
    # required, parses the arguments and calls the selected handler. Returns nothing.
    parser = create_parser()
    if len(argv) < 2:
      parser.error('Too few arguments.')
    flags.mark_flags_as_required(command_required_flags[argv[1]])
    args = parser.parse_args()
    args.func()

  app.run(smcli_main)


if __name__ == '__main__':
  main()
