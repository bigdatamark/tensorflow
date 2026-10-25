# [MODULE DOC]
# Generic program entry point for TensorFlow scripts (exported as the v1 app.run API).
# Wraps the absl application runner so that command-line flags unknown to the registered flag
# definitions are left in argv instead of crashing the program. Importing it has no side effects
# beyond importing the flags module and the export decorator.
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

"""Generic entry point script."""
import sys as _sys

from absl.app import run as _run

from tensorflow.python.platform import flags
from tensorflow.python.util.tf_export import tf_export


def _parse_flags_tolerate_undef(argv):
  # [FUNCTION DOC]
  # HELPER: Flag parser callback handed to the absl runner by run.
  # Parses the command line with the global flags object in known-only mode. Takes a list of
  # argument strings, or nothing/null to use the process argument list. Returns the argument
  # list with recognized flags consumed and unknown flags retained. Updates global flag values.
  """Parse args, returning any unknown flags (ABSL defaults to crashing)."""
  return flags.FLAGS(_sys.argv if argv is None else argv, known_only=True)


@tf_export(v1=['app.run'])
def run(main=None, argv=None):
  # [FUNCTION DOC]
  # Runs a program through the absl runner. main is an optional callable taking the argument
  # list; when omitted it defaults to the main function of the top-level script module (an
  # attribute error is raised if that module has none). argv is an optional list of strings,
  # defaulting to the process arguments. Unknown flags are tolerated and passed through.
  # Does not return normally: the absl runner exits the process with the status of main.
  """Runs the program with an optional 'main' function and 'argv' list."""

  main = main or _sys.modules['__main__'].main

  _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)
