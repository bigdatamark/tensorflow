# [MODULE DOC]
# Public testing API surface. Re-exports the TensorFlow test case class (as TestCase), graph
# comparison, local cluster, GPU query and gradient checking helpers, the mock library, the
# benchmark base class and the stub-out helper; defines the test main (which also installs the
# native stack trace handler), temp directory and source directory path helpers, build-feature
# queries (CUDA, ROCm, any GPU, XLA, CPU target) and a decorator factory that skips tests when a
# predicate holds. Import-time side effects: registers the mock library and the functions under
# public API names through the export decorator.

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

"""Testing."""

import functools

from unittest import mock

# pylint: disable=g-bad-import-order
from tensorflow.python.framework import test_util as _test_util
from tensorflow.python.platform import googletest as _googletest

# pylint: disable=unused-import
from tensorflow.python.framework.test_util import assert_equal_graph_def
from tensorflow.python.framework.test_util import create_local_cluster
from tensorflow.python.framework.test_util import TensorFlowTestCase as TestCase
from tensorflow.python.framework.test_util import gpu_device_name
from tensorflow.python.framework.test_util import is_gpu_available

from tensorflow.python.ops.gradient_checker import compute_gradient_error
from tensorflow.python.ops.gradient_checker import compute_gradient
# pylint: enable=unused-import,g-bad-import-order


from tensorflow.python.util.tf_export import tf_export

tf_export(v1=['test.mock'])(mock)

# Import Benchmark class
Benchmark = _googletest.Benchmark  # pylint: disable=invalid-name

# Import StubOutForTesting class
StubOutForTesting = _googletest.StubOutForTesting  # pylint: disable=invalid-name


@tf_export('test.main')
def main(argv=None):
  # [FUNCTION DOC]
  # Entry point for test binaries. Installs the native stack trace handler (side effect: crashes
  # then print a trace), then delegates to the googletest main with the optional argument list
  # (defaults to the process arguments). Returns whatever that main returns; normally the process
  # exits after the tests run.
  """Runs all unit tests."""
  _test_util.InstallStackTraceHandler()
  return _googletest.main(argv)


@tf_export(v1=['test.get_temp_dir'])
def get_temp_dir():
  # [FUNCTION DOC]
  # Returns the per-process temporary directory path string from the googletest helper, creating
  # it on first use (side effect: directory creation and an exit-time cleanup registration).
  # Exported in v1 only. No inputs.
  """Returns a temporary directory for use during tests.

  There is no need to delete the directory after the test.

  @compatibility(TF2)
  This function is removed in TF2. Please use `TestCase.get_temp_dir` instead
  in a test case.
  Outside of a unit test, obtain a temporary directory through Python's
  `tempfile` module.
  @end_compatibility

  Returns:
    The temporary directory.
  """
  return _googletest.GetTempDir()


@tf_export(v1=['test.test_src_dir_path'])
def test_src_dir_path(relative_path):
  # [FUNCTION DOC]
  # Public wrapper (v1 only) that forwards the relative path string to the googletest helper and
  # returns the absolute path inside the test runfiles. Raises a key error if the runfiles
  # environment variables are not set.
  """Creates an absolute test srcdir path given a relative path.

  Args:
    relative_path: a path relative to tensorflow root.
      e.g. "core/platform".

  Returns:
    An absolute path to the linked in runfiles.
  """
  return _googletest.test_src_dir_path(relative_path)


@tf_export('test.is_built_with_cuda')
def is_built_with_cuda():
  # [FUNCTION DOC]
  # Returns true if this TensorFlow build includes CUDA support, as reported by the test utility
  # module. No inputs; no side effects.
  """Returns whether TensorFlow was built with CUDA (GPU) support.

  This method should only be used in tests written with `tf.test.TestCase`. A
  typical usage is to skip tests that should only run with CUDA (GPU).

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_gpu(self):
  ...     if not tf.test.is_built_with_cuda():
  ...       self.skipTest("test is only applicable on GPU")
  ...
  ...     with tf.device("GPU:0"):
  ...       self.assertEqual(tf.math.add(1.0, 2.0), 3.0)

  TensorFlow official binary is built with CUDA.
  """
  return _test_util.IsGoogleCudaEnabled()


@tf_export('test.is_built_with_rocm')
def is_built_with_rocm():
  # [FUNCTION DOC]
  # Returns true if this TensorFlow build includes ROCm support, as reported by the test utility
  # module. No inputs; no side effects.
  """Returns whether TensorFlow was built with ROCm (GPU) support.

  This method should only be used in tests written with `tf.test.TestCase`. A
  typical usage is to skip tests that should only run with ROCm (GPU).

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_gpu(self):
  ...     if not tf.test.is_built_with_rocm():
  ...       self.skipTest("test is only applicable on GPU")
  ...
  ...     with tf.device("GPU:0"):
  ...       self.assertEqual(tf.math.add(1.0, 2.0), 3.0)

  TensorFlow official binary is NOT built with ROCm.
  """
  return _test_util.IsBuiltWithROCm()


@tf_export('test.disable_with_predicate')
def disable_with_predicate(pred, skip_message):
  # [FUNCTION DOC]
  # Decorator factory for test methods. Inputs: pred, a callable with no arguments returning a
  # truth value, evaluated each time the test runs; skip_message, the reason text. Returns a
  # decorator which wraps a test method so that, when the predicate is true, the test is skipped
  # with the message, and otherwise the method runs normally.
  """Disables the test if pred is true."""

  def decorator_disable_with_predicate(func):
    # [FUNCTION DOC]
    # HELPER: the actual decorator returned by the factory. Takes a test method and returns a
    # wrapped version that preserves the original's metadata.

    @functools.wraps(func)
    def wrapper_disable_with_predicate(self, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: the wrapped test method. Evaluates the predicate; if true, calls the test case's
      # skip method with the message (which raises the framework's skip exception) and returns
      # nothing; otherwise calls the original method with the same arguments and returns its
      # result.
      if pred():
        self.skipTest(skip_message)
      else:
        return func(self, *args, **kwargs)

    return wrapper_disable_with_predicate

  return decorator_disable_with_predicate


@tf_export('test.is_built_with_gpu_support')
def is_built_with_gpu_support():
  # [FUNCTION DOC]
  # Returns true if the build has CUDA or ROCm support (logical or of the two queries in this
  # module). No inputs; no side effects.
  """Returns whether TensorFlow was built with GPU (CUDA or ROCm) support.

  This method should only be used in tests written with `tf.test.TestCase`. A
  typical usage is to skip tests that should only run with GPU.

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_gpu(self):
  ...     if not tf.test.is_built_with_gpu_support():
  ...       self.skipTest("test is only applicable on GPU")
  ...
  ...     with tf.device("GPU:0"):
  ...       self.assertEqual(tf.math.add(1.0, 2.0), 3.0)

  TensorFlow official binary is built with CUDA GPU support.
  """
  return is_built_with_cuda() or is_built_with_rocm()


@tf_export('test.is_built_with_xla')
def is_built_with_xla():
  # [FUNCTION DOC]
  # Returns true if this TensorFlow build includes XLA compiler support, as reported by the test
  # utility module. No inputs; no side effects.
  """Returns whether TensorFlow was built with XLA support.

  This method should only be used in tests written with `tf.test.TestCase`. A
  typical usage is to skip tests that should only run with XLA.

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_xla(self):
  ...     if not tf.test.is_built_with_xla():
  ...       self.skipTest("test is only applicable on XLA")

  ...     @tf.function(jit_compile=True)
  ...     def add(x, y):
  ...       return tf.math.add(x, y)
  ...
  ...     self.assertEqual(add(tf.ones(()), tf.ones(())), 2.0)

  TensorFlow official binary is built with XLA.
  """
  return _test_util.IsBuiltWithXLA()


@tf_export('test.is_cpu_target_available')
def is_cpu_target_available(target):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring example for a CPU-target check is a copy of the XLA example and
  # compiles a function with just-in-time compilation; the function itself only queries a flag.
  # Returns true if TensorFlow was built with support for the named CPU target (string, e.g. an
  # architecture name), as reported by the test utility module. No side effects.
  """Indicates whether TensorFlow was built with support for a given CPU target.

  Args:
    target: The name of the CPU target whose support to check for.

  Returns:
    A boolean indicating whether TensorFlow was built with support for the
    given CPU target.

  This method should only be used in tests written with `tf.test.TestCase`. A
  typical usage is to skip tests that should only run with a given target.

  >>> class MyTest(tf.test.TestCase):
  ...
  ...   def test_add_on_aarch64(self):
  ...     if not tf.test.is_cpu_target_available('aarch64'):
  ...       self.skipTest("test is only applicable on AArch64")

  ...     @tf.function(jit_compile=True)
  ...     def add(x, y):
  ...       return tf.math.add(x, y)
  ...
  ...     self.assertEqual(add(tf.ones(()), tf.ones(())), 2.0)
  """
  return _test_util.IsCPUTargetAvailable(target)
