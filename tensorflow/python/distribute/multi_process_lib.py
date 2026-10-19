# [MODULE DOC]
# HELPER: helper module for multi-process distribute tests.
# Provides a process class and entry points for running tests in spawned subprocesses. On
# platforms other than Windows it builds an absl-compatible forkserver multiprocessing context
# and replaces the module-level multiprocessing name with it; on Windows the process class just
# skips the test. Also offers a test main that fixes the executable path for spawned children,
# handles being a spawned child, and then runs the normal test main. Import-time side effects:
# rebinds the module's multiprocessing name and defines the process class; keeps module state
# recording whether the test main was called.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Library for multi-process testing."""

import multiprocessing
import os
import platform
import sys
import unittest
from absl import app
from absl import logging

from tensorflow.python.eager import test


def is_oss():
  # [FUNCTION DOC]
  # Returns true when the first command-line argument contains the text bazel, taken as a sign
  # the test runs under the open-source build. Takes no inputs. Reads the global argument list.
  # It is a loose heuristic on the program name only.
  """Returns whether the test is run under OSS."""
  return len(sys.argv) >= 1 and 'bazel' in sys.argv[0]


def _is_enabled():
  # [FUNCTION DOC]
  # HELPER: decides whether multi-process support is available on this platform.
  # KNOWN ISSUE: the Python 3.8 on Linux check compares the full version info object with a
  # two-number tuple, which is never equal, so that branch is dead code.
  # Returns false if running under the open-source build with a TPU command-line argument, or on
  # Windows; true otherwise. Reads the argument list and platform info only.
  # Note that flags may not be parsed at this point and simply importing the
  # flags module causes a variety of unusual errors.
  tpu_args = [arg for arg in sys.argv if arg.startswith('--tpu')]
  if is_oss() and tpu_args:
    return False
  if sys.version_info == (3, 8) and platform.system() == 'Linux':
    return False  # TODO(b/171242147)
  return sys.platform != 'win32'


class _AbslProcess:
  # [CLASS DOC]
  # HELPER: mixin combined with a multiprocessing process class so that the child process runs
  # its target through the absl application runner (which parses flags and sets up logging).
  """A process that runs using absl.app.run."""

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Cooperative constructor: forwards all positional and keyword arguments to the next class in
    # the inheritance chain, then replaces the instance's run method with the absl-wrapped one,
    # keeping the original run method saved. The replacement is carried into the spawned process
    # by pickling.
    super(_AbslProcess, self).__init__(*args, **kwargs)
    # Monkey-patch that is carried over into the spawned process by pickle.
    self._run_impl = getattr(self, 'run')
    self.run = self._run_with_absl

  def _run_with_absl(self):
    # [FUNCTION DOC]
    # HELPER: replacement run method; invokes the absl application runner with a main callback
    # that ignores its arguments and calls the saved original run method. Returns nothing; the
    # runner may exit the process when it finishes.
    app.run(lambda _: self._run_impl())


if _is_enabled():

  class AbslForkServerProcess(_AbslProcess,
                              multiprocessing.context.ForkServerProcess):
    # [CLASS DOC]
    # Forkserver process type whose run goes through the absl application runner. Defined only
    # when multi-process support is enabled (not on Windows).
    """An absl-compatible Forkserver process.

    Note: Forkserver is not available in windows.
    """

  class AbslForkServerContext(multiprocessing.context.ForkServerContext):
    # [CLASS DOC]
    # Forkserver multiprocessing context named absl_forkserver whose process type is the
    # absl-compatible forkserver process. An instance replaces the module-level multiprocessing
    # name so that all processes created here use it.
    _name = 'absl_forkserver'
    Process = AbslForkServerProcess  # pylint: disable=invalid-name

  multiprocessing = AbslForkServerContext()
  Process = multiprocessing.Process

else:

  class Process(object):
    # [CLASS DOC]
    # Placeholder process type for unsupported platforms (Windows): constructing it never makes a
    # process, it skips the running test instead.
    """A process that skips test (until windows is supported)."""

    def __init__(self, *args, **kwargs):
      # [FUNCTION DOC]
      # Discards all arguments and raises a unit-test skip signal saying Windows is not supported
      # by the multi-process runner.
      del args, kwargs
      raise unittest.SkipTest(
          'TODO(b/150264776): Windows is not supported in MultiProcessRunner.')


_test_main_called = False


def _set_spawn_exe_path():
  # [FUNCTION DOC]
  # HELPER: prepares multiprocessing so spawned children launch the right binary.
  # If the program path is a Python file, guesses the real test binary under the bazel output
  # tree (trying the bzlmod, TensorFlow and Keras workspace roots, using the test target
  # environment variable), requiring it to be executable, and overwrites the program path with
  # it. Then sets the executable on the default multiprocessing context. Raises a runtime error if
  # no binary can be found. Side effects: logs, modifies the global argument list and context.
  """Set the path to the executable for spawned processes.

  This utility searches for the binary the parent process is using, and sets
  the executable of multiprocessing's context accordingly.

  Raises:
    RuntimeError: If the binary path cannot be determined.
  """
  # TODO(b/150264776): This does not work with Windows. Find a solution.
  if sys.argv[0].endswith('.py'):
    def guess_path(package_root):
      # [FUNCTION DOC]
      # HELPER: guesses the test binary path for a given workspace root name. Only when the
      # program path contains the bazel-out marker and the root name: derives the base from the
      # program path, builds the binary path from the test target environment variable, logs it,
      # and returns it if executable, otherwise returns nothing. In all other cases returns
      # nothing. May fail with a key error if the target variable is unset.
      # If all we have is a python module path, we'll need to make a guess for
      # the actual executable path.
      if 'bazel-out' in sys.argv[0] and package_root in sys.argv[0]:
        # Guess the binary path under bazel. For target
        # //tensorflow/python/distribute:input_lib_test_multiworker_gpu, the
        # argv[0] is in the form of
        # /.../tensorflow/python/distribute/input_lib_test.py
        # and the binary is
        # /.../tensorflow/python/distribute/input_lib_test_multiworker_gpu
        package_root_base = sys.argv[0][: sys.argv[0].rfind(package_root)]
        binary = os.environ['TEST_TARGET'][2:].replace(':', '/', 1)
        possible_path = os.path.join(package_root_base, package_root, binary)
        logging.info('Guessed test binary path: %s', possible_path)
        if os.access(possible_path, os.X_OK):
          return possible_path
        return None
    path = guess_path('_main')  # bzlmod
    if not path:
      path = guess_path('org_tensorflow')
    if not path:
      path = guess_path('org_keras')
    if path is None:
      logging.error(
          'Cannot determine binary path. sys.argv[0]=%s os.environ=%s',
          sys.argv[0], os.environ)
      raise RuntimeError('Cannot determine binary path')
    sys.argv[0] = path
  # Note that this sets the executable for *all* contexts.
  multiprocessing.get_context().set_executable(sys.argv[0])


def _if_spawn_run_and_exit():
  # [FUNCTION DOC]
  # HELPER: lets the test binary act as the spawned child of multiprocessing.
  # Detects a spawned process by a dash-c argument whose command starts with importing from the
  # multiprocessing package. If not spawned, does nothing. Otherwise trims the argument list to
  # the program name, runs that command, and exits with status zero. Side effects: executes
  # arbitrary command text supplied on the command line, rewrites the argument list, exits.
  """If spawned process, run requested spawn task and exit. Else a no-op."""

  # `multiprocessing` module passes a script "from multiprocessing.x import y"
  # to subprocess, followed by a main function call. We use this to tell if
  # the process is spawned. Examples of x are "forkserver" or
  # "semaphore_tracker".
  is_spawned = ('-c' in sys.argv[1:] and
                sys.argv[sys.argv.index('-c') +
                         1].startswith('from multiprocessing.'))

  if not is_spawned:
    return
  cmd = sys.argv[sys.argv.index('-c') + 1]
  # As a subprocess, we disregarding all other interpreter command line
  # arguments.
  sys.argv = sys.argv[0:1]

  # Run the specified command - this is expected to be one of:
  # 1. Spawn the process for semaphore tracker.
  # 2. Spawn the initial process for forkserver.
  # 3. Spawn any process as requested by the "spawn" method.
  exec(cmd)  # pylint: disable=exec-used
  sys.exit(0)  # Semaphore tracker doesn't explicitly sys.exit.


def test_main():
  # [FUNCTION DOC]
  # Entry point for the main section of multi-process test files. Marks the module as having been
  # called, sets the environment variable enabling GPU memory growth, and, when multi-process
  # support is enabled, fixes the spawn executable path and handles the spawned-child case (which
  # exits). Finally runs the regular test main. Takes no inputs, returns nothing.
  """Main function to be called within `__main__` of a test file."""
  global _test_main_called
  _test_main_called = True

  os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

  if _is_enabled():
    _set_spawn_exe_path()
    _if_spawn_run_and_exit()

  # Only runs test.main() if not spawned process.
  test.main()


def initialized():
  # [FUNCTION DOC]
  # Returns true once the test main entry point has been called in this process, otherwise
  # false. Takes no inputs; reads module state.
  """Returns whether the module is initialized."""
  return _test_main_called
