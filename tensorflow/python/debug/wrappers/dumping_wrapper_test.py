# [MODULE DOC]
# Unit tests for the dumping debug wrapper session and the dumping debug hook. Verifies argument
# validation of the constructor, the layout and naming of per-run dump directories, dumped
# tensor values, recorded fetches and feed keys, watch callbacks (legacy tuple form and the
# watch-options form, stateful callbacks), thread name filtering and empty fetches. Graph-mode
# only. No import-time side effects.
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
"""Unit Tests for classes in dumping_wrapper.py."""
import glob
import os
import tempfile
import threading

from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import dumping_wrapper
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.debug.wrappers import hooks
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.training import monitored_session


@test_util.run_v1_only("b/120545219")
class DumpingDebugWrapperSessionTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the dumping wrapper session and dumping hook. Fixture: temporary session root, a
  # variable (10.0), constants delta (1.0) and eta (-1.4), increment and decrement assign-add
  # ops, a scalar placeholder with an assign-add using it, and a session with the variable
  # initialized.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the temporary session root directory, the variable, constants, increment and
    # decrement ops, the placeholder and its assign-add op, opens a session and initializes the
    # variable. Stores all on the test instance.
    self.session_root = tempfile.mkdtemp()

    self.v = variable_v1.VariableV1(10.0, dtype=dtypes.float32, name="v")
    self.delta = constant_op.constant(1.0, dtype=dtypes.float32, name="delta")
    self.eta = constant_op.constant(-1.4, dtype=dtypes.float32, name="eta")
    self.inc_v = state_ops.assign_add(self.v, self.delta, name="inc_v")
    self.dec_v = state_ops.assign_add(self.v, self.eta, name="dec_v")

    self.ph = array_ops.placeholder(dtypes.float32, shape=(), name="ph")
    self.inc_w_ph = state_ops.assign_add(self.v, self.ph, name="inc_w_ph")

    self.sess = session.Session()
    self.sess.run(self.v.initializer)

  def tearDown(self):
    # [FUNCTION DOC]
    # Resets the default graph and recursively deletes the session root directory if it exists.
    ops.reset_default_graph()
    if os.path.isdir(self.session_root):
      file_io.delete_recursively(self.session_root)

  def _assert_correct_run_subdir_naming(self, run_subdir):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the dump tests.
    # Asserts a run directory name starts with the run prefix, contains exactly two underscores,
    # and that the middle part (timestamp) parses as a positive integer. Input: the directory's
    # base name.
    self.assertStartsWith(run_subdir, "run_")
    self.assertEqual(2, run_subdir.count("_"))
    self.assertGreater(int(run_subdir.split("_")[1]), 0)

  def testConstructWrapperWithExistingNonEmptyRootDirRaisesException(self):
    # [FUNCTION DOC]
    # Puts a subdirectory inside the session root so it is non-empty; asserts constructing the
    # wrapper on it raises a value error saying the path points to a non-empty directory.
    dir_path = os.path.join(self.session_root, "foo")
    os.mkdir(dir_path)
    self.assertTrue(os.path.isdir(dir_path))

    with self.assertRaisesRegex(
        ValueError, "session_root path points to a non-empty directory"):
      dumping_wrapper.DumpingDebugWrapperSession(
          session.Session(), session_root=self.session_root)

  def testConstructWrapperWithExistingFileDumpRootRaisesException(self):
    # [FUNCTION DOC]
    # Creates a regular file and uses it as the session root; asserts the constructor raises a
    # value error saying the path points to a file.
    file_path = os.path.join(self.session_root, "foo")
    open(file_path, "a").close()  # Create the file
    self.assertTrue(gfile.Exists(file_path))
    self.assertFalse(gfile.IsDirectory(file_path))
    with self.assertRaisesRegex(ValueError,
                                "session_root path points to a file"):
      dumping_wrapper.DumpingDebugWrapperSession(
          session.Session(), session_root=file_path)

  def testConstructWrapperWithNonexistentSessionRootCreatesDirectory(self):
    # [FUNCTION DOC]
    # Constructs the wrapper with a nonexistent session root path and asserts the directory was
    # created. Cleans it up at the end.
    new_dir_path = os.path.join(tempfile.mkdtemp(), "new_dir")
    dumping_wrapper.DumpingDebugWrapperSession(
        session.Session(), session_root=new_dir_path)
    self.assertTrue(gfile.IsDirectory(new_dir_path))
    # Cleanup.
    gfile.DeleteRecursively(new_dir_path)

  def testDumpingOnASingleRunWorks(self):
    # [FUNCTION DOC]
    # Runs the increment op once through the wrapper. Asserts one run directory exists with a
    # valid name, the variable's dumped value is 10.0, the recorded fetches match the op's textual
    # form and the recorded feed keys is the textual form of nothing/null.
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root)
    sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))

    self._assert_correct_run_subdir_naming(os.path.basename(dump_dirs[0]))
    dump = debug_data.DebugDumpDir(dump_dirs[0])
    self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))

    self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
    self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testDumpingOnASingleRunWorksWithRelativePathForDebugDumpDir(self):
    # [FUNCTION DOC]
    # After one wrapped run, changes the working directory to the session root and loads the dump
    # using a relative path; asserts the variable's dumped value is 10.0. Restores the original
    # working directory in all cases.
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root)
    sess.run(self.inc_v)
    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    cwd = os.getcwd()
    try:
      os.chdir(self.session_root)
      dump = debug_data.DebugDumpDir(
          os.path.relpath(dump_dirs[0], self.session_root))
      self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))
    finally:
      os.chdir(cwd)

  def testDumpingOnASingleRunWithFeedDictWorks(self):
    # [FUNCTION DOC]
    # Runs the placeholder-based op once with a feed mapping. Asserts one valid run directory, the
    # dumped variable value 10.0, recorded fetches matching the op and recorded feed keys equal to
    # the textual form of the feed mapping's keys.
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root)
    feed_dict = {self.ph: 3.2}
    sess.run(self.inc_w_ph, feed_dict=feed_dict)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))

    self._assert_correct_run_subdir_naming(os.path.basename(dump_dirs[0]))
    dump = debug_data.DebugDumpDir(dump_dirs[0])
    self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))

    self.assertEqual(repr(self.inc_w_ph), dump.run_fetches_info)
    self.assertEqual(repr(feed_dict.keys()), dump.run_feed_keys_info)

  def testDumpingOnMultipleRunsWorks(self):
    # [FUNCTION DOC]
    # Runs the increment op three times. Sorts the run directories by timestamp and asserts there
    # are three, each validly named, with dumped variable values 10, 11, 12 and the expected
    # fetches and (null) feed keys metadata.
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root)
    for _ in range(3):
      sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    dump_dirs = sorted(
        dump_dirs, key=lambda x: int(os.path.basename(x).split("_")[1]))
    self.assertEqual(3, len(dump_dirs))
    for i, dump_dir in enumerate(dump_dirs):
      self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
      dump = debug_data.DebugDumpDir(dump_dir)
      self.assertAllClose([10.0 + 1.0 * i],
                          dump.get_tensors("v", 0, "DebugIdentity"))
      self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
      self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testUsingNonCallableAsWatchFnRaisesTypeError(self):
    # [FUNCTION DOC]
    # Passes a string as the watch callback; asserts construction raises a type error saying it is
    # not callable.
    bad_watch_fn = "bad_watch_fn"
    with self.assertRaisesRegex(TypeError, "watch_fn is not callable"):
      dumping_wrapper.DumpingDebugWrapperSession(
          self.sess,
          session_root=self.session_root,
          watch_fn=bad_watch_fn)

  def testDumpingWithLegacyWatchFnOnFetchesWorks(self):
    # [FUNCTION DOC]
    # Uses a legacy-style watch callback that depends on the fetch name. Alternately runs
    # increment and decrement three times each (six runs). Asserts even-indexed (increment) runs
    # dump data with expected variable values, and odd-indexed (decrement) runs dump nothing;
    # metadata fetches are as expected and feed keys null.
    """Use a watch_fn that returns different allowlists for different runs."""

    def watch_fn(fetches, feeds):
      # [FUNCTION DOC]
      # HELPER: watch callback for this test. Ignores feeds. If the fetch is the increment op's
      # output returns the debug op with match-everything patterns; otherwise returns patterns
      # that match nothing.
      del feeds
      # A watch_fn that picks fetch name.
      if fetches.name == "inc_v:0":
        # If inc_v, watch everything.
        return "DebugIdentity", r".*", r".*"
      else:
        # If dec_v, watch nothing.
        return "DebugIdentity", r"$^", r"$^"

    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess,
        session_root=self.session_root,
        watch_fn=watch_fn)

    for _ in range(3):
      sess.run(self.inc_v)
      sess.run(self.dec_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    dump_dirs = sorted(
        dump_dirs, key=lambda x: int(os.path.basename(x).split("_")[1]))
    self.assertEqual(6, len(dump_dirs))

    for i, dump_dir in enumerate(dump_dirs):
      self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
      dump = debug_data.DebugDumpDir(dump_dir)
      if i % 2 == 0:
        self.assertGreater(dump.size, 0)
        self.assertAllClose([10.0 - 0.4 * (i / 2)],
                            dump.get_tensors("v", 0, "DebugIdentity"))
        self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
        self.assertEqual(repr(None), dump.run_feed_keys_info)
      else:
        self.assertEqual(0, dump.size)
        self.assertEqual(repr(self.dec_v), dump.run_fetches_info)
        self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testDumpingWithLegacyWatchFnWithNonDefaultDebugOpsWorks(self):
    # [FUNCTION DOC]
    # Uses a legacy watch callback returning two debug ops (identity and numeric summary) and
    # watching everything. After one run asserts one run directory, the identity dump is 10.0,
    # and the numeric summary dump has 14 elements.
    """Use a watch_fn that specifies non-default debug ops."""

    def watch_fn(fetches, feeds):
      # [FUNCTION DOC]
      # HELPER: watch callback for this test. Ignores its arguments; returns a list of two debug
      # op names with match-everything node and op type patterns.
      del fetches, feeds
      return ["DebugIdentity", "DebugNumericSummary"], r".*", r".*"

    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess,
        session_root=self.session_root,
        watch_fn=watch_fn)

    sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))
    dump = debug_data.DebugDumpDir(dump_dirs[0])

    self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))
    self.assertEqual(14,
                     len(dump.get_tensors("v", 0, "DebugNumericSummary")[0]))

  def testDumpingWithWatchFnWithNonDefaultDebugOpsWorks(self):
    # [FUNCTION DOC]
    # Uses a watch callback returning a watch-options object with two debug ops, node names
    # starting with the variable's name, all op types and reference-type tensors. After one run
    # asserts the identity dump is 10.0, the numeric summary has 14 elements, and the increment and
    # delta nodes were not dumped.
    """Use a watch_fn that specifies non-default debug ops."""

    def watch_fn(fetches, feeds):
      # [FUNCTION DOC]
      # HELPER: watch callback for this test. Ignores its arguments; returns a watch-options
      # object with the two debug ops, node-name allowlist for names starting with the variable,
      # all op types, and tensor dtype pattern for reference types.
      del fetches, feeds
      return framework.WatchOptions(
          debug_ops=["DebugIdentity", "DebugNumericSummary"],
          node_name_regex_allowlist=r"^v.*",
          op_type_regex_allowlist=r".*",
          tensor_dtype_regex_allowlist=".*_ref")

    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess,
        session_root=self.session_root,
        watch_fn=watch_fn)

    sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))
    dump = debug_data.DebugDumpDir(dump_dirs[0])

    self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))
    self.assertEqual(14,
                     len(dump.get_tensors("v", 0, "DebugNumericSummary")[0]))

    dumped_nodes = [dump.node_name for dump in dump.dumped_tensor_data]
    self.assertNotIn("inc_v", dumped_nodes)
    self.assertNotIn("delta", dumped_nodes)

  def testDumpingDebugHookWithoutWatchFnWorks(self):
    # [FUNCTION DOC]
    # Uses a dumping debug hook with no watch callback inside a hooked session and runs the
    # increment op once. Asserts one valid run directory, dumped variable value 10.0, and expected
    # fetches and null feed keys metadata.
    dumping_hook = hooks.DumpingDebugHook(self.session_root)
    mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
    mon_sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))

    self._assert_correct_run_subdir_naming(os.path.basename(dump_dirs[0]))
    dump = debug_data.DebugDumpDir(dump_dirs[0])
    self.assertAllClose([10.0], dump.get_tensors("v", 0, "DebugIdentity"))

    self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
    self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testDumpingDebugHookWithStatefulWatchFnWorks(self):
    # [FUNCTION DOC]
    # Uses a dumping hook whose watch callback counts calls. Four runs: odd runs watch all
    # reference-type tensors, even runs watch nothing. Asserts four run directories; even-indexed
    # directories contain the variable's dumped values (10, 12) without the delta node, odd-indexed
    # ones are empty; fetches and null feed keys metadata correct for all.
    watch_fn_state = {"run_counter": 0}

    def counting_watch_fn(fetches, feed_dict):
      # [FUNCTION DOC]
      # HELPER: stateful watch callback for this test. Ignores its arguments, increments a shared
      # run counter, and returns a watch-options object: on odd calls, identity debug op with
      # reference-type tensors; on even calls, patterns matching nothing.
      del fetches, feed_dict
      watch_fn_state["run_counter"] += 1
      if watch_fn_state["run_counter"] % 2 == 1:
        # If odd-index run (1-based), watch every ref-type tensor.
        return framework.WatchOptions(
            debug_ops="DebugIdentity", tensor_dtype_regex_allowlist=".*_ref")
      else:
        # If even-index run, watch nothing.
        return framework.WatchOptions(
            debug_ops="DebugIdentity",
            node_name_regex_allowlist=r"^$",
            op_type_regex_allowlist=r"^$")

    dumping_hook = hooks.DumpingDebugHook(
        self.session_root, watch_fn=counting_watch_fn)
    mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
    for _ in range(4):
      mon_sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    dump_dirs = sorted(
        dump_dirs, key=lambda x: int(os.path.basename(x).split("_")[1]))
    self.assertEqual(4, len(dump_dirs))

    for i, dump_dir in enumerate(dump_dirs):
      self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
      dump = debug_data.DebugDumpDir(dump_dir)
      if i % 2 == 0:
        self.assertAllClose([10.0 + 1.0 * i],
                            dump.get_tensors("v", 0, "DebugIdentity"))
        self.assertNotIn("delta",
                         [datum.node_name for datum in dump.dumped_tensor_data])
      else:
        self.assertEqual(0, dump.size)

      self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
      self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testDumpingDebugHookWithStatefulLegacyWatchFnWorks(self):
    # [FUNCTION DOC]
    # Same scenario as the stateful watch-options test but with a legacy tuple-returning
    # callback: odd runs watch everything, even runs watch nothing. Asserts four run directories,
    # dumped values present for even-indexed ones and empty dumps for odd-indexed ones, and the
    # expected metadata.
    watch_fn_state = {"run_counter": 0}

    def counting_watch_fn(fetches, feed_dict):
      # [FUNCTION DOC]
      # HELPER: stateful legacy watch callback. Ignores its arguments, increments a shared run
      # counter, returns the debug op with match-everything patterns on odd calls and
      # match-nothing patterns on even calls.
      del fetches, feed_dict
      watch_fn_state["run_counter"] += 1
      if watch_fn_state["run_counter"] % 2 == 1:
        # If odd-index run (1-based), watch everything.
        return "DebugIdentity", r".*", r".*"
      else:
        # If even-index run, watch nothing.
        return "DebugIdentity", r"$^", r"$^"

    dumping_hook = hooks.DumpingDebugHook(
        self.session_root, watch_fn=counting_watch_fn)
    mon_sess = monitored_session._HookedSession(self.sess, [dumping_hook])
    for _ in range(4):
      mon_sess.run(self.inc_v)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    dump_dirs = sorted(
        dump_dirs, key=lambda x: int(os.path.basename(x).split("_")[1]))
    self.assertEqual(4, len(dump_dirs))

    for i, dump_dir in enumerate(dump_dirs):
      self._assert_correct_run_subdir_naming(os.path.basename(dump_dir))
      dump = debug_data.DebugDumpDir(dump_dir)
      if i % 2 == 0:
        self.assertAllClose([10.0 + 1.0 * i],
                            dump.get_tensors("v", 0, "DebugIdentity"))
      else:
        self.assertEqual(0, dump.size)

      self.assertEqual(repr(self.inc_v), dump.run_fetches_info)
      self.assertEqual(repr(None), dump.run_feed_keys_info)

  def testDumpingFromMultipleThreadsObeysThreadNameFilter(self):
    # [FUNCTION DOC]
    # Wraps with a thread name filter allowing only the main thread. Runs delta on the main thread
    # and eta on a child thread; asserts both return correct values but only one run directory
    # exists, with a single dumped tensor from the delta node.
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root,
        thread_name_filter=r"MainThread$")

    self.assertAllClose(1.0, sess.run(self.delta))
    child_thread_result = []
    def child_thread_job():
      # [FUNCTION DOC]
      # HELPER: body of the child thread. Runs the eta constant through the wrapped session and
      # appends the result to the shared result list. Returns nothing.
      child_thread_result.append(sess.run(self.eta))

    thread = threading.Thread(name="ChildThread", target=child_thread_job)
    thread.start()
    thread.join()
    self.assertAllClose([-1.4], child_thread_result)

    dump_dirs = glob.glob(os.path.join(self.session_root, "run_*"))
    self.assertEqual(1, len(dump_dirs))
    dump = debug_data.DebugDumpDir(dump_dirs[0])
    self.assertEqual(1, dump.size)
    self.assertEqual("delta", dump.dumped_tensor_data[0].node_name)

  def testDumpingWrapperWithEmptyFetchWorks(self):
    # [FUNCTION DOC]
    # Runs the wrapped session with an empty list of fetches and expects it to complete without
    # error.
    sess = dumping_wrapper.DumpingDebugWrapperSession(
        self.sess, session_root=self.session_root)
    sess.run([])


if __name__ == "__main__":
  googletest.main()
