# [MODULE DOC]
# Turns syntax trees or source text into live Python modules. Exposes load_source (write source
# text to a uniquely named temporary file, import it as a module and register it in the global
# module table) and load_ast (unparse a syntax tree to text, load it via load_source, optionally
# build a source map back to the original code). Writing to a real file keeps the generated code
# readable by debuggers and the source inspection facilities. Private helper removes the file.
# Import-time side effects: none beyond imports; temporary files and module-table entries are
# created only when the functions are called.
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
"""Converting AST to code and Python entities.

Adapted from Tangent.
"""

import atexit
import errno
import importlib
import os
import sys
import tempfile

from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser


def _remove_file(file_name):
  # [FUNCTION DOC]
  # HELPER: cleanup callback for load_source's temporary files.
  # Deletes the file with the given name. A "file not found" error is silently ignored (the file
  # may already have been cleaned up); any other operating-system error is re-raised. Returns nothing.
  """Remove a file, if it exists."""
  try:
    os.remove(file_name)
  except OSError as e:
    if e.errno == errno.ENOENT:
      # The file disappeared. Ignore this. Temporary files might get
      # cleaned up, especially if they reside in /tmp.
      pass
    else:
      raise


def load_source(source, delete_on_exit):
  # [FUNCTION DOC]
  # KNOWN ISSUE: only the top-level importlib package is imported, yet its util submodule is used;
  # this works only because another import has already loaded the submodule.
  # Writes the given source text (string) to a new temporary file (UTF-8, ".py" suffix, name
  # prefixed with a generated-file marker, not auto-deleted), imports that file as a module whose
  # name is the file's base name, and stores the module in the global module table under that
  # name (a TODO notes this should become a garbage-collected cache). If delete_on_exit is true,
  # registers an exit-time callback that removes the file. Executing the source may raise any
  # error the code raises. Returns a pair: the loaded module and the temporary file's path.
  """Loads the given source code as a Python module."""
  with tempfile.NamedTemporaryFile(
      mode='w',
      suffix='.py',
      prefix='__autograph_generated_file',
      delete=False,
      encoding='utf-8') as f:
    module_name = os.path.basename(f.name[:-3])
    file_name = f.name
    f.write(source)

  if delete_on_exit:
    atexit.register(lambda: _remove_file(file_name))

  spec = importlib.util.spec_from_file_location(module_name, file_name)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  # TODO(mdan): Use our own garbage-collected cache instead of sys.modules.
  sys.modules[module_name] = module
  return module, file_name


def load_ast(nodes,
             indentation='  ',
             include_source_map=False,
             delete_on_exit=True):
  # [FUNCTION DOC]
  # Converts a syntax tree (a single node or a list/tuple of nodes) to source text using the
  # given indentation string (default two spaces), loads that text as a module via load_source
  # (delete_on_exit, default true, controls removal of the temporary file at exit) and returns a
  # triple: the module, the generated source text and a source map. The source map is built from
  # the nodes, the text and the module's file only when include_source_map (default false) is
  # true; otherwise it is the null value. A single non-sequence node is wrapped into a tuple first.
  # Side effects: creates a temporary file and registers a module in the global module table.
  """Loads the given AST as a Python module.

  Compiling the AST code this way ensures that the source code is readable by
  e.g. `pdb` or `inspect`.

  Args:
    nodes: Union[ast.AST, Iterable[ast.AST]], the code to compile, as an AST
      object.
    indentation: Text, the string to use for indentation.
    include_source_map: bool, whether return a source map.
    delete_on_exit: bool, whether to delete the temporary file used for
      compilation on exit.

  Returns:
    Tuple[module, Text, Dict[LineLocation, OriginInfo]], containing:
    the module containing the unparsed nodes, the source code corresponding to
    nodes, and the source map. Is include_source_map is False, the source map
    will be None.
  """
  if not isinstance(nodes, (list, tuple)):
    nodes = (nodes,)

  source = parser.unparse(nodes, indentation=indentation)
  module, _ = load_source(source, delete_on_exit)

  if include_source_map:
    source_map = origin_info.create_source_map(nodes, source, module.__file__)
  else:
    source_map = None

  # TODO(mdan): Return a structured object.
  return module, source, source_map
