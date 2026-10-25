# [MODULE DOC]
# Test that a crashing process prints a native stack trace. The same script plays two roles
# chosen by a command-line option named child: in the parent role the test launches itself as a
# subprocess in the child role, where one test aborts the process with the abort signal; the
# parent captures output and checks a stack trace marker appears. Module state: a global flags
# holder, set only when run as a script, after parsing the child option with the standard
# argument parser and stripping it from the argument list before the test main runs.

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
# =============================================================================
"""Test to make sure stack trace is generated in case of test failures."""

import argparse
import os
import signal
import subprocess
import sys

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


# FLAGS defined at the bottom:
# child (bool) set to true if we are running in the child process.
FLAGS = None

_CHILD_FLAG_HELP = 'Boolean. Set to true if this is the child process.'


class StacktraceHandlerTest(test.TestCase):
  # [CLASS DOC]
  # Test case with two tests that cooperate across two processes. Depends on the module-level
  # flags holder being set (only true when the file is run as a script, otherwise the tests fail
  # with an attribute error on the null value).

  def testChildProcessKillsItself(self):
    # [FUNCTION DOC]
    # Only does something in the child role: sends the abort signal to the current process, which
    # should make the stack trace handler print a trace. In the parent role it does nothing and
    # passes. KNOWN ISSUE: if run in the child role it never reports success; it relies on the
    # parent test inspecting the output.
    if FLAGS.child:
      os.kill(os.getpid(), signal.SIGABRT)

  def testGeneratesStacktrace(self):
    # [FUNCTION DOC]
    # Returns immediately in the child role. In the parent role: launches the script again as a
    # subprocess with the child option true (using the Python executable if known, otherwise the
    # script directly), in the current working directory with piped output; waits and collects
    # both output streams, logs them, and asserts the combined bytes contain the interpreter
    # frame-evaluation marker, which shows a native stack trace was printed by the aborted child.
    if FLAGS.child:
      return

    # Subprocess sys.argv[0] with --child=True
    if sys.executable:
      child_process = subprocess.Popen(
          [sys.executable, sys.argv[0], '--child=True'], cwd=os.getcwd(),
          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
      child_process = subprocess.Popen(
          [sys.argv[0], '--child=True'], cwd=os.getcwd(),
          stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Capture its output. capture both stdout and stderr and append them.
    # We are not worried about timing or order of messages in this test.
    child_stdout, child_stderr = child_process.communicate()
    child_output = child_stdout + child_stderr

    # Make sure the child process is dead before we proceed.
    child_process.wait()

    logging.info('Output from the child process:')
    logging.info(child_output)

    # Verify a stack trace is printed.
    self.assertIn(b'PyEval_EvalFrame', child_output)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--child', type=bool, default=False, help=_CHILD_FLAG_HELP)
  FLAGS, unparsed = parser.parse_known_args()

  # Now update argv, so that unittest library does not get confused.
  sys.argv = [sys.argv[0]] + unparsed
  test.main()
