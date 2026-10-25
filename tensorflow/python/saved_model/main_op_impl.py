# [MODULE DOC]
# Implementation of the deprecated v1 SavedModel main op helpers: functions that build a single
# grouped op initializing global variables, local variables and tables, optionally followed by a
# restore op. Both functions are deprecated and exported only through the v1 API; the public
# facade module re-exports them. Import-time effect: registers the API exports.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""SavedModel main op implementation."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

_DEPRECATION_MSG = (
    'This API was designed for TensorFlow v1. See '
    'https://www.tensorflow.org/guide/migrate for instructions on how to '
    'migrate your code to TensorFlow v2.')


@tf_export(v1=['saved_model.main_op.main_op'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def main_op():
  # [FUNCTION DOC]
  # Builds and returns a single grouped op that runs the global variable initializer, the local
  # variable initializer and the table initializer of the default graph. Takes no inputs.
  # Side effect: adds the initializer ops to the current default graph. Deprecated and exported
  # only in the v1 API (calling it logs a deprecation warning).
  """Returns a main op to init variables and tables.

  Returns the main op including the group of ops that initializes all
  variables, initializes local variables and initialize all tables.

  Returns:
    The set of ops to be run as part of the main op upon the load operation.
  """
  init = variables.global_variables_initializer()
  init_local = variables.local_variables_initializer()
  init_tables = lookup_ops.tables_initializer()
  return control_flow_ops.group(init, init_local, init_tables)


# TODO(sukritiramesh): Integrate with Saver for complete restore functionality.
@tf_export(v1=['saved_model.main_op_with_restore',
               'saved_model.main_op.main_op_with_restore'])
@deprecation.deprecated(None, _DEPRECATION_MSG)
def main_op_with_restore(restore_op_name):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says restore_op_name is the name of an op, but the value is passed
  # straight to the grouping function, which expects op or tensor objects, so a plain string
  # name is not resolved to an op here. The local variable also shadows the function name.
  # Builds a grouped op that depends on the main op (all variable, local variable and table
  # initializers) and then groups the given restore op. Input: restore_op_name, the restore op
  # (see issue above). Returns the grouped op. Side effect: adds ops to the default graph.
  # Deprecated and exported only in the v1 API.
  """Returns a main op to init variables, tables and restore the graph.

  Returns the main op including the group of ops that initializes all
  variables, initialize local variables, initialize all tables and the restore
  op name.

  Args:
    restore_op_name: Name of the op to use to restore the graph.

  Returns:
    The set of ops to be run as part of the main op upon the load operation.
  """
  with ops.control_dependencies([main_op()]):
    main_op_with_restore = control_flow_ops.group(restore_op_name)
  return main_op_with_restore
