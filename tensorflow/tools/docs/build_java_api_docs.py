# [MODULE DOC]
# Command line script that generates the Java reference docs for TensorFlow.org. Defines flags for
# output directory (required), site path prefix, an unused code URL prefix, an unused search hints
# switch, and a switch controlling whether bazel-generated op sources are included. Defines path
# constants for the Java sources and the bazel output of generated ops. Import-time side effect:
# defines command line flags. The entry point runs only when executed as a script.
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
"""Generate Java reference docs for TensorFlow.org."""
import pathlib
import shutil
import subprocess
import tempfile

from absl import app
from absl import flags

from tensorflow_docs.api_generator import gen_java

FLAGS = flags.FLAGS

# These flags are required by infrastructure, not all of them are used.
flags.DEFINE_string('output_dir', None,
                    ("Use this branch as the root version and don't"
                     ' create in version directory'))

flags.DEFINE_string('site_path', 'api_docs/java',
                    'Path prefix in the _toc.yaml')

flags.DEFINE_string('code_url_prefix', None,
                    '[UNUSED] The url prefix for links to code.')

flags.DEFINE_bool(
    'search_hints', True,
    '[UNUSED] Include metadata search hints in the generated files')

# Use this flag to disable bazel generation if you're not setup for it.
flags.DEFINE_bool('gen_ops', True, 'enable/disable bazel-generated ops')

# __file__ is the path to this file
DOCS_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
TENSORFLOW_ROOT = DOCS_TOOLS_DIR.parents[2]
SOURCE_PATH = TENSORFLOW_ROOT / 'tensorflow/java/src/main/java'
OP_SOURCE_PATH = (
    TENSORFLOW_ROOT /
    'bazel-bin/tensorflow/java/ops/src/main/java/org/tensorflow/op')


def main(unused_argv):
  # [FUNCTION DOC]
  # Entry point: ignores its argument. Copies the Java sources into a fresh temporary directory;
  # when the generate-ops flag is on, runs configure with default answers, builds the Java op
  # generation bazel target and copies the generated ops into the temporary tree. Then calls the
  # Java doc generator for the org.tensorflow package, writing to the output directory flag with
  # the site path flag. Returns nothing. Raises a called-process error if the bazel build fails,
  # and a file error if the sources are missing.
  # KNOWN ISSUE: the temporary directory is never removed, and the code URL prefix and search
  # hints flags are accepted but unused.
  merged_source = pathlib.Path(tempfile.mkdtemp())
  shutil.copytree(SOURCE_PATH, merged_source / 'java')

  if FLAGS.gen_ops:
    # `$ yes | configure`
    yes = subprocess.Popen(['yes', ''], stdout=subprocess.PIPE)
    configure = subprocess.Popen([TENSORFLOW_ROOT / 'configure'],
                                 stdin=yes.stdout,
                                 cwd=TENSORFLOW_ROOT)
    configure.communicate()

    subprocess.check_call(
        ['bazel', 'build', '//tensorflow/java:java_op_gen_sources'],
        cwd=TENSORFLOW_ROOT)
    shutil.copytree(OP_SOURCE_PATH, merged_source / 'java/org/tensorflow/ops')

  gen_java.gen_java_docs(
      package='org.tensorflow',
      source_path=merged_source / 'java',
      output_dir=pathlib.Path(FLAGS.output_dir),
      site_path=pathlib.Path(FLAGS.site_path))


if __name__ == '__main__':
  flags.mark_flags_as_required(['output_dir'])
  app.run(main)
