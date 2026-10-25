# [MODULE DOC]
# Command-line tool for inspecting checkpoint files: lists tensor names, shapes and types, prints
# the values of one or all tensors, and reports the total parameter count (optionally excluding
# tensors whose names match a pattern). Exposes the printing function and a parser for numpy
# print options. Keeps a module-level flags holder that is filled only when run as a script.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A simple script for inspect checkpoint files."""
import argparse
import re
import sys

from absl import app
import numpy as np

from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import flags
from tensorflow.python.training import py_checkpoint_reader

FLAGS = None


def _count_total_params(reader, count_exclude_pattern=""):
  # [FUNCTION DOC]
  # HELPER: computes the parameter total reported by the checkpoint printing function.
  # Inputs: reader, a checkpoint reader object; count_exclude_pattern, optional regular expression
  # string (default empty, meaning no exclusion). Gets the variable-name to shape mapping; if a
  # pattern is given, tensors whose names match are dropped and a line reporting how many tensors
  # and parameters were excluded is printed. Returns the sum of element counts (product of each
  # shape) of the remaining tensors as an integer. Raises a regex error for an invalid pattern.
  """Count total number of variables."""
  var_to_shape_map = reader.get_variable_to_shape_map()

  # Filter out tensors that we don't want to count
  if count_exclude_pattern:
    regex_pattern = re.compile(count_exclude_pattern)
    new_var_to_shape_map = {}
    exclude_num_tensors = 0
    exclude_num_params = 0
    for v in var_to_shape_map:
      if regex_pattern.search(v):
        exclude_num_tensors += 1
        exclude_num_params += np.prod(var_to_shape_map[v])
      else:
        new_var_to_shape_map[v] = var_to_shape_map[v]
    var_to_shape_map = new_var_to_shape_map
    print("# Excluding %d tensors (%d params) that match %s when counting." % (
        exclude_num_tensors, exclude_num_params, count_exclude_pattern))

  var_sizes = [np.prod(var_to_shape_map[v]) for v in var_to_shape_map]
  return np.sum(var_sizes, dtype=int)


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors,
                                     all_tensor_names=False,
                                     count_exclude_pattern=""):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the whole body is wrapped in a catch-all handler that prints the error text
  # instead of raising, so failures (missing file, corrupted data) never propagate to callers.
  # KNOWN ISSUE: in the data-loss handler, the loop variable of the generator shadows the caught
  # exception name; harmless here but confusing.
  # Opens the checkpoint at file_name. If all_tensors or all_tensor_names is true, prints every
  # tensor's name, type and shape in sorted order, plus its value when all_tensors is true
  # (values that cannot convert to a numpy type print a placeholder message). Else if tensor_name
  # is empty, prints the reader's debug string. Else prints the single named tensor's name, type,
  # shape and value, or a not-found message and returns early (without the parameter count).
  # Afterwards prints the total parameter count, honoring count_exclude_pattern (regex string,
  # default empty). Any exception is printed; hints are added for compressed-block errors and, for
  # data-loss errors on names ending in index/meta/data, a suggestion to use the V2 filename
  # prefix. Returns nothing.
  """Prints tensors in a checkpoint file.

  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.

  If `tensor_name` is provided, prints the content of the tensor.

  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
    all_tensor_names: Boolean indicating whether to print all tensor names.
    count_exclude_pattern: Regex string, pattern to exclude tensors from count.
  """
  try:
    reader = py_checkpoint_reader.NewCheckpointReader(file_name)
    if all_tensors or all_tensor_names:
      var_to_shape_map = reader.get_variable_to_shape_map()
      var_to_dtype_map = reader.get_variable_to_dtype_map()
      for key, value in sorted(var_to_shape_map.items()):
        print("tensor: %s (%s) %s" % (key, var_to_dtype_map[key].name, value))
        if all_tensors:
          try:
            print(reader.get_tensor(key))
          except errors_impl.InternalError:
            print("<not convertible to a numpy dtype>")
    elif not tensor_name:
      print(reader.debug_string().decode("utf-8", errors="ignore"))
    else:
      if not reader.has_tensor(tensor_name):
        print("Tensor %s not found in checkpoint" % tensor_name)
        return

      var_to_shape_map = reader.get_variable_to_shape_map()
      var_to_dtype_map = reader.get_variable_to_dtype_map()
      print("tensor: %s (%s) %s" %
            (tensor_name, var_to_dtype_map[tensor_name].name,
             var_to_shape_map[tensor_name]))
      print(reader.get_tensor(tensor_name))

    # Count total number of parameters
    print("# Total number of params: %d" % _count_total_params(
        reader, count_exclude_pattern=count_exclude_pattern))
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed "
            "with SNAPPY.")
    if ("Data loss" in str(e) and
        any(e in file_name for e in [".index", ".meta", ".data"])):
      proposed_file = ".".join(file_name.split(".")[0:-1])
      v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
      print(v2_file_error_template.format(proposed_file))


def parse_numpy_printoption(kv_str):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the value-conversion error handler reads a message attribute of the exception,
  # which does not exist in Python 3, so a bad value raises an attribute error instead of the
  # intended argument-type error.
  # Argument type converter for the printoptions flag. Splits the text at the first equals sign
  # into an option name and value string, looks the option up in numpy's current print options,
  # converts the value to that option's type (booleans use the flags boolean parser), and applies
  # it globally with the numpy print-options setter. Raises an argument-type error if the text is
  # not in key=value form, the key is unknown, or the option's current type is the null type
  # (unsupported). Returns nothing; mutates global numpy print settings.
  """Sets a single numpy printoption from a string of the form 'x=y'.

  See documentation on numpy.set_printoptions() for details about what values
  x and y can take. x can be any option listed there other than 'formatter'.

  Args:
    kv_str: A string of the form 'x=y', such as 'threshold=100000'

  Raises:
    argparse.ArgumentTypeError: If the string couldn't be used to set any
        numpy printoption.
  """
  k_v_str = kv_str.split("=", 1)
  if len(k_v_str) != 2 or not k_v_str[0]:
    raise argparse.ArgumentTypeError("'%s' is not in the form k=v." % kv_str)
  k, v_str = k_v_str
  printoptions = np.get_printoptions()
  if k not in printoptions:
    raise argparse.ArgumentTypeError("'%s' is not a valid printoption." % k)
  v_type = type(printoptions[k])
  if v_type is type(None):
    raise argparse.ArgumentTypeError(
        "Setting '%s' from the command line is not supported." % k)
  try:
    v = (
        v_type(v_str)
        if v_type is not bool else flags.BooleanParser().parse(v_str))
  except ValueError as e:
    raise argparse.ArgumentTypeError(e.message)
  np.set_printoptions(**{k: v})


def main(unused_argv):
  # [FUNCTION DOC]
  # Application main; the argument is ignored. If the file_name flag is empty, prints a usage
  # message and exits with status 1. Otherwise calls the checkpoint printing function with the
  # file name, tensor name, all-tensors and all-tensor-names flags and the count exclusion pattern.
  # Relies on the module-level flags holder being set by the script block.
  if not FLAGS.file_name:
    print(
        "Usage: inspect_checkpoint --file_name=checkpoint_file_name "
        "[--tensor_name=tensor_to_print] "
        "[--all_tensors] "
        "[--all_tensor_names] "
        "[--count_exclude_pattern] "
        "[--printoptions]"
    )
    sys.exit(1)
  else:
    print_tensors_in_checkpoint_file(
        FLAGS.file_name, FLAGS.tensor_name,
        FLAGS.all_tensors, FLAGS.all_tensor_names,
        count_exclude_pattern=FLAGS.count_exclude_pattern)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--file_name",
      type=str,
      default="",
      help="Checkpoint filename. "
      "Note, if using Checkpoint V2 format, file_name is the "
      "shared prefix between all files in the checkpoint.")
  parser.add_argument(
      "--tensor_name",
      type=str,
      default="",
      help="Name of the tensor to inspect")
  parser.add_argument(
      "--count_exclude_pattern",
      type=str,
      default="",
      help="Pattern to exclude tensors, e.g., from optimizers, when counting.")
  parser.add_argument(
      "--all_tensors",
      nargs="?",
      const=True,
      type="bool",
      default=False,
      help="If True, print the names and values of all the tensors.")
  parser.add_argument(
      "--all_tensor_names",
      nargs="?",
      const=True,
      type="bool",
      default=False,
      help="If True, print the names of all the tensors.")
  parser.add_argument(
      "--printoptions",
      nargs="*",
      type=parse_numpy_printoption,
      help="Argument for numpy.set_printoptions(), in the form 'k=v'.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
