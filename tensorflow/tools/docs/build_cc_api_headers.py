# [MODULE DOC]
# Command line script that builds the C++ API headers for the TensorFlow.org reference docs.
# Defines the required output directory flag and path constants derived from this file's location.
# Runs configure with default answers and builds the C++ ops target with bazel, then copies the
# build output directory into the output folder. Import-time side effect: defines a command line
# flag. The entry point runs only when executed as a script.
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
"""Generate C++ reference docs for TensorFlow.org."""
import os
import pathlib
import subprocess

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# These flags are required by infrastructure, not all of them are used.
flags.DEFINE_string('output_dir', None,
                    ("Use this branch as the root version and don't"
                     ' create in version directory'))

# __file__ is the path to this file
DOCS_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
TENSORFLOW_ROOT = DOCS_TOOLS_DIR.parents[2]


def build_headers(output_dir):
  # [FUNCTION DOC]
  # Builds the generated C++ headers and copies them.
  # Input: output_dir, a path object (the slash operator is used on it, so a plain string fails).
  # Creates the directory if needed, runs the configure script in the TensorFlow root feeding it
  # default answers, builds the C++ ops bazel target, then copies the dereferenced bazel output
  # tree to a subfolder named for generated files. Returns nothing. Raises a called-process error
  # if the build or copy fails. Side effects: subprocesses, file system writes.
  # KNOWN ISSUE: the exit status of the configure step is ignored and the helper process that
  # supplies default answers is never waited on.
  """Builds the headers files for TF."""
  os.makedirs(output_dir, exist_ok=True)

  # `$ yes | configure`
  yes = subprocess.Popen(['yes', ''], stdout=subprocess.PIPE)
  configure = subprocess.Popen([TENSORFLOW_ROOT / 'configure'],
                               stdin=yes.stdout,
                               cwd=TENSORFLOW_ROOT)
  configure.communicate()

  subprocess.check_call(['bazel', 'build', 'tensorflow/cc:cc_ops'],
                        cwd=TENSORFLOW_ROOT)
  subprocess.check_call(
      ['cp', '--dereference', '-r', 'bazel-bin', output_dir / 'bazel-genfiles'],
      cwd=TENSORFLOW_ROOT)


def main(argv):
  # [FUNCTION DOC]
  # Entry point: ignores the command line arguments and builds the headers into the directory given
  # by the output directory flag. Returns nothing.
  del argv
  build_headers(pathlib.Path(FLAGS.output_dir))


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)
