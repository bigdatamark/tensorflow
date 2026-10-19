# [MODULE DOC]
# Unit tests for cli_shared: byte and time readable-string formatting, the run-start intro and
# short run description (for various fetch/feed shapes and tensor filters), and the run-time
# error intro. The run-related test classes only run under graph-mode (v1) sessions.
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
"""Unit tests for the shared functions and classes for tfdbg CLI."""
from collections import namedtuple

from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class BytesToReadableStrTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for cli_shared.bytes_to_readable_str across size ranges and the optional B suffix.

  def testNoneSizeWorks(self):
    # [FUNCTION DOC]
    # A missing size must be rendered as the string form of nothing.
    self.assertEqual(str(None), cli_shared.bytes_to_readable_str(None))

  def testSizesBelowOneKiloByteWorks(self):
    # [FUNCTION DOC]
    # Sizes 0, 500 and 1023 are rendered as plain integers without a suffix.
    self.assertEqual("0", cli_shared.bytes_to_readable_str(0))
    self.assertEqual("500", cli_shared.bytes_to_readable_str(500))
    self.assertEqual("1023", cli_shared.bytes_to_readable_str(1023))

  def testSizesBetweenOneKiloByteandOneMegaByteWorks(self):
    # [FUNCTION DOC]
    # Sizes from 1024 up to just under a megabyte are rendered with two decimals and a k suffix.
    self.assertEqual("1.00k", cli_shared.bytes_to_readable_str(1024))
    self.assertEqual("2.40k", cli_shared.bytes_to_readable_str(int(1024 * 2.4)))
    self.assertEqual("1023.00k", cli_shared.bytes_to_readable_str(1024 * 1023))

  def testSizesBetweenOneMegaByteandOneGigaByteWorks(self):
    # [FUNCTION DOC]
    # Sizes from one megabyte up to just under a gigabyte are rendered with two decimals and an
    # M suffix.
    self.assertEqual("1.00M", cli_shared.bytes_to_readable_str(1024**2))
    self.assertEqual("2.40M",
                     cli_shared.bytes_to_readable_str(int(1024**2 * 2.4)))
    self.assertEqual("1023.00M",
                     cli_shared.bytes_to_readable_str(1024**2 * 1023))

  def testSizeAboveOneGigaByteWorks(self):
    # [FUNCTION DOC]
    # Sizes of a gigabyte and above are rendered with two decimals and a G suffix (no larger
    # unit exists, so 2000 gigabytes stays in G).
    self.assertEqual("1.00G", cli_shared.bytes_to_readable_str(1024**3))
    self.assertEqual("2000.00G",
                     cli_shared.bytes_to_readable_str(1024**3 * 2000))

  def testReadableStrIncludesBAtTheEndOnRequest(self):
    # [FUNCTION DOC]
    # With the include-B option on, a B is appended to every unit (plain, k, M and G).
    self.assertEqual("0B", cli_shared.bytes_to_readable_str(0, include_b=True))
    self.assertEqual(
        "1.00kB", cli_shared.bytes_to_readable_str(
            1024, include_b=True))
    self.assertEqual(
        "1.00MB", cli_shared.bytes_to_readable_str(
            1024**2, include_b=True))
    self.assertEqual(
        "1.00GB", cli_shared.bytes_to_readable_str(
            1024**3, include_b=True))


class TimeToReadableStrTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for cli_shared.time_to_readable_str: automatic unit choice and forced units.

  def testNoneTimeWorks(self):
    # [FUNCTION DOC]
    # A missing time value is rendered as "0".
    self.assertEqual("0", cli_shared.time_to_readable_str(None))

  def testMicrosecondsTime(self):
    # [FUNCTION DOC]
    # A value of 40 microseconds is rendered with the microsecond unit.
    self.assertEqual("40us", cli_shared.time_to_readable_str(40))

  def testMillisecondTime(self):
    # [FUNCTION DOC]
    # A value of 40 thousand microseconds is rendered as 40 milliseconds.
    self.assertEqual("40ms", cli_shared.time_to_readable_str(40e3))

  def testSecondTime(self):
    # [FUNCTION DOC]
    # A value of 40 million microseconds is rendered as 40 seconds.
    self.assertEqual("40s", cli_shared.time_to_readable_str(40e6))

  def testForceTimeUnit(self):
    # [FUNCTION DOC]
    # Forces each unit (seconds, milliseconds, microseconds) on the same value and checks the
    # scaled output, including a small value in exponent form and zero staying "0". An unknown
    # unit must raise a value error mentioning an invalid time unit.
    self.assertEqual("40s",
                     cli_shared.time_to_readable_str(
                         40e6, force_time_unit=cli_shared.TIME_UNIT_S))
    self.assertEqual("40000ms",
                     cli_shared.time_to_readable_str(
                         40e6, force_time_unit=cli_shared.TIME_UNIT_MS))
    self.assertEqual("40000000us",
                     cli_shared.time_to_readable_str(
                         40e6, force_time_unit=cli_shared.TIME_UNIT_US))
    self.assertEqual("4e-05s",
                     cli_shared.time_to_readable_str(
                         40, force_time_unit=cli_shared.TIME_UNIT_S))
    self.assertEqual("0",
                     cli_shared.time_to_readable_str(
                         0, force_time_unit=cli_shared.TIME_UNIT_S))

    with self.assertRaisesRegex(ValueError, r"Invalid time unit: ks"):
      cli_shared.time_to_readable_str(100, force_time_unit="ks")


@test_util.run_v1_only("tfdbg CLI is for tf.Session only")
class GetRunStartIntroAndDescriptionTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for cli_shared.get_run_start_intro and get_run_short_description. Fixtures: three
  # float constants named a, b and c, and a small sparse tensor. Graph-mode (v1) only.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the constant tensors a, b, c (values 11, 22, 33) and a sparse tensor fixture.
    self.const_a = constant_op.constant(11.0, name="a")
    self.const_b = constant_op.constant(22.0, name="b")
    self.const_c = constant_op.constant(33.0, name="c")

    self.sparse_d = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 1]], values=[1.0, 2.0], dense_shape=[3, 3])

  def tearDown(self):
    # [FUNCTION DOC]
    # Resets the default graph so tests do not share graph state.
    ops.reset_default_graph()

  def testSingleFetchNoFeeds(self):
    # [FUNCTION DOC]
    # Builds the intro for one fetch and no feeds. Checks the call-number line, the fetch name
    # line, the "(Empty)" feeds line, the four recommended commands with their bold/link font
    # attributes, the short description, and the main menu with run and exit items.
    run_start_intro = cli_shared.get_run_start_intro(12, self.const_a, None, {})

    # Verify line about run() call number.
    self.assertTrue(run_start_intro.lines[1].endswith("run() call #12:"))

    # Verify line about fetch.
    const_a_name_line = run_start_intro.lines[4]
    self.assertEqual(self.const_a.name, const_a_name_line.strip())

    # Verify line about feeds.
    feeds_line = run_start_intro.lines[7]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify lines about possible commands and their font attributes.
    self.assertEqual("run:", run_start_intro.lines[11][2:])
    annot = run_start_intro.font_attr_segs[11][0]
    self.assertEqual(2, annot[0])
    self.assertEqual(5, annot[1])
    self.assertEqual("run", annot[2][0].content)
    self.assertEqual("bold", annot[2][1])
    annot = run_start_intro.font_attr_segs[13][0]
    self.assertEqual(2, annot[0])
    self.assertEqual(8, annot[1])
    self.assertEqual("run -n", annot[2][0].content)
    self.assertEqual("bold", annot[2][1])
    self.assertEqual("run -t <T>:", run_start_intro.lines[15][2:])
    self.assertEqual([(2, 12, "bold")], run_start_intro.font_attr_segs[15])
    self.assertEqual("run -f <filter_name>:", run_start_intro.lines[17][2:])
    self.assertEqual([(2, 22, "bold")], run_start_intro.font_attr_segs[17])

    # Verify short description.
    description = cli_shared.get_run_short_description(12, self.const_a, None)
    self.assertEqual("run #12: 1 fetch (a:0); 0 feeds", description)

    # Verify the main menu associated with the run_start_intro.
    self.assertIn(debugger_cli_common.MAIN_MENU_KEY,
                  run_start_intro.annotations)
    menu = run_start_intro.annotations[debugger_cli_common.MAIN_MENU_KEY]
    self.assertEqual("run", menu.caption_to_item("run").content)
    self.assertEqual("exit", menu.caption_to_item("exit").content)

  def testSparseTensorAsFeedShouldHandleNoNameAttribute(self):
    # [FUNCTION DOC]
    # Uses a sparse tensor (which has no name attribute) as a feed key; expects the intro and the
    # short description to fall back to its string form.
    sparse_feed_val = ([[0, 0], [1, 1]], [10.0, 20.0])
    run_start_intro = cli_shared.get_run_start_intro(
        1, self.sparse_d, {self.sparse_d: sparse_feed_val}, {})
    self.assertEqual(str(self.sparse_d), run_start_intro.lines[7].strip())

    short_description = cli_shared.get_run_short_description(
        1, self.sparse_d, {self.sparse_d: sparse_feed_val})
    self.assertEqual(
        "run #1: 1 fetch; 1 feed (%s)" % self.sparse_d, short_description)

  def testSparseTensorAsFetchShouldHandleNoNameAttribute(self):
    # [FUNCTION DOC]
    # Uses a sparse tensor as the fetch; expects the fetch line to show its string form.
    run_start_intro = cli_shared.get_run_start_intro(1, self.sparse_d, None, {})
    self.assertEqual(str(self.sparse_d), run_start_intro.lines[4].strip())

  def testTwoFetchesListNoFeeds(self):
    # [FUNCTION DOC]
    # Two fetches given as a list: both names are listed, feeds show "(Empty)", and the short
    # description reports 2 fetches and 0 feeds.
    fetches = [self.const_a, self.const_b]
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    const_a_name_line = run_start_intro.lines[4]
    const_b_name_line = run_start_intro.lines[5]
    self.assertEqual(self.const_a.name, const_a_name_line.strip())
    self.assertEqual(self.const_b.name, const_b_name_line.strip())

    feeds_line = run_start_intro.lines[8]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 2 fetches; 0 feeds", description)

  def testNestedListAsFetches(self):
    # [FUNCTION DOC]
    # Fetches given as a nested list are flattened in order in the intro; the short description
    # reports 3 fetches.
    fetches = [self.const_c, [self.const_a, self.const_b]]
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    # Verify lines about the fetches.
    self.assertEqual(self.const_c.name, run_start_intro.lines[4].strip())
    self.assertEqual(self.const_a.name, run_start_intro.lines[5].strip())
    self.assertEqual(self.const_b.name, run_start_intro.lines[6].strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 3 fetches; 0 feeds", description)

  def testNestedDictAsFetches(self):
    # [FUNCTION DOC]
    # Fetches given as a nested mapping are flattened; since key order is not guaranteed, the
    # test compares the set of listed names. The short description reports 3 fetches.
    fetches = {"c": self.const_c, "ab": {"a": self.const_a, "b": self.const_b}}
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    # Verify lines about the fetches. The ordering of the dict keys is
    # indeterminate.
    fetch_names = set()
    fetch_names.add(run_start_intro.lines[4].strip())
    fetch_names.add(run_start_intro.lines[5].strip())
    fetch_names.add(run_start_intro.lines[6].strip())

    self.assertEqual({"a:0", "b:0", "c:0"}, fetch_names)

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 3 fetches; 0 feeds", description)

  def testTwoFetchesAsTupleNoFeeds(self):
    # [FUNCTION DOC]
    # Same as the list case but fetches are a tuple; expects two listed names and 2 fetches.
    fetches = (self.const_a, self.const_b)
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    const_a_name_line = run_start_intro.lines[4]
    const_b_name_line = run_start_intro.lines[5]
    self.assertEqual(self.const_a.name, const_a_name_line.strip())
    self.assertEqual(self.const_b.name, const_b_name_line.strip())

    feeds_line = run_start_intro.lines[8]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 2 fetches; 0 feeds", description)

  def testTwoFetchesAsNamedTupleNoFeeds(self):
    # [FUNCTION DOC]
    # Same as the list case but fetches are a named tuple; expects two listed names and 2 fetches.
    fetches_namedtuple = namedtuple("fetches", "x y")
    fetches = fetches_namedtuple(self.const_b, self.const_c)
    run_start_intro = cli_shared.get_run_start_intro(1, fetches, None, {})

    const_b_name_line = run_start_intro.lines[4]
    const_c_name_line = run_start_intro.lines[5]
    self.assertEqual(self.const_b.name, const_b_name_line.strip())
    self.assertEqual(self.const_c.name, const_c_name_line.strip())

    feeds_line = run_start_intro.lines[8]
    self.assertEqual("(Empty)", feeds_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, fetches, None)
    self.assertEqual("run #1: 2 fetches; 0 feeds", description)

  def testWithFeedDict(self):
    # [FUNCTION DOC]
    # Two feeds with one fetch: expects the fetch line, both feed names listed, and a short
    # description of 1 fetch (named) and 2 feeds.
    feed_dict = {
        self.const_a: 10.0,
        self.const_b: 20.0,
    }

    run_start_intro = cli_shared.get_run_start_intro(1, self.const_c, feed_dict,
                                                     {})

    const_c_name_line = run_start_intro.lines[4]
    self.assertEqual(self.const_c.name, const_c_name_line.strip())

    # Verify lines about the feed dict.
    feed_a_line = run_start_intro.lines[7]
    feed_b_line = run_start_intro.lines[8]
    self.assertEqual(self.const_a.name, feed_a_line.strip())
    self.assertEqual(self.const_b.name, feed_b_line.strip())

    # Verify short description.
    description = cli_shared.get_run_short_description(1, self.const_c,
                                                       feed_dict)
    self.assertEqual("run #1: 1 fetch (c:0); 2 feeds", description)

  def testTensorFilters(self):
    # [FUNCTION DOC]
    # Passes two registered tensor filters. Expects both names to be listed in the intro, each
    # with a link to run with that filter, and a short description with one named feed.
    feed_dict = {self.const_a: 10.0}
    tensor_filters = {
        "filter_a": lambda x: True,
        "filter_b": lambda x: False,
    }

    run_start_intro = cli_shared.get_run_start_intro(1, self.const_c, feed_dict,
                                                     tensor_filters)

    # Verify the listed names of the tensor filters.
    filter_names = set()
    filter_names.add(run_start_intro.lines[20].split(" ")[-1])
    filter_names.add(run_start_intro.lines[21].split(" ")[-1])

    self.assertEqual({"filter_a", "filter_b"}, filter_names)

    # Verify short description.
    description = cli_shared.get_run_short_description(1, self.const_c,
                                                       feed_dict)
    self.assertEqual("run #1: 1 fetch (c:0); 1 feed (a:0)", description)

    # Verify the command links for the two filters.
    command_set = set()
    annot = run_start_intro.font_attr_segs[20][0]
    command_set.add(annot[2].content)
    annot = run_start_intro.font_attr_segs[21][0]
    command_set.add(annot[2].content)
    self.assertEqual({"run -f filter_a", "run -f filter_b"}, command_set)

  def testGetRunShortDescriptionWorksForTensorFeedKey(self):
    # [FUNCTION DOC]
    # A tensor used as the single feed key is named by its tensor name in the description.
    short_description = cli_shared.get_run_short_description(
        1, self.const_a, {self.const_a: 42.0})
    self.assertEqual("run #1: 1 fetch (a:0); 1 feed (a:0)", short_description)

  def testGetRunShortDescriptionWorksForUnicodeFeedKey(self):
    # [FUNCTION DOC]
    # A text string used as the single feed key is shown as-is in the description.
    short_description = cli_shared.get_run_short_description(
        1, self.const_a, {u"foo": 42.0})
    self.assertEqual("run #1: 1 fetch (a:0); 1 feed (foo)", short_description)


@test_util.run_v1_only("tfdbg CLI is for tf.Session only")
class GetErrorIntroTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for cli_shared.get_error_intro using a variable fixture. Graph-mode (v1) only.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates a float variable named a used as the source of a fake failing op.
    self.var_a = variables.Variable(42.0, name="a")

  def tearDown(self):
    # [FUNCTION DOC]
    # Resets the default graph so tests do not share graph state.
    ops.reset_default_graph()

  def testShapeError(self):
    # [FUNCTION DOC]
    # Builds an op error attached to the variable's initializer op and checks the intro: the
    # blinking banner line, the three suggested commands with bold link attributes, and the
    # op name, error type and details lines.
    tf_error = errors.OpError(None, self.var_a.initializer, "foo description",
                              None)

    error_intro = cli_shared.get_error_intro(tf_error)

    self.assertEqual("!!! An error occurred during the run !!!",
                     error_intro.lines[1])
    self.assertEqual([(0, len(error_intro.lines[1]), "blink")],
                     error_intro.font_attr_segs[1])

    self.assertEqual(2, error_intro.lines[4].index("ni -a -d -t a/Assign"))
    self.assertEqual(2, error_intro.font_attr_segs[4][0][0])
    self.assertEqual(22, error_intro.font_attr_segs[4][0][1])
    self.assertEqual("ni -a -d -t a/Assign",
                     error_intro.font_attr_segs[4][0][2][0].content)
    self.assertEqual("bold", error_intro.font_attr_segs[4][0][2][1])

    self.assertEqual(2, error_intro.lines[6].index("li -r a/Assign"))
    self.assertEqual(2, error_intro.font_attr_segs[6][0][0])
    self.assertEqual(16, error_intro.font_attr_segs[6][0][1])
    self.assertEqual("li -r a/Assign",
                     error_intro.font_attr_segs[6][0][2][0].content)
    self.assertEqual("bold", error_intro.font_attr_segs[6][0][2][1])

    self.assertEqual(2, error_intro.lines[8].index("lt"))
    self.assertEqual(2, error_intro.font_attr_segs[8][0][0])
    self.assertEqual(4, error_intro.font_attr_segs[8][0][1])
    self.assertEqual("lt", error_intro.font_attr_segs[8][0][2][0].content)
    self.assertEqual("bold", error_intro.font_attr_segs[8][0][2][1])

    self.assertStartsWith(error_intro.lines[11], "Op name:")
    self.assertTrue(error_intro.lines[11].endswith("a/Assign"))

    self.assertStartsWith(error_intro.lines[12], "Error type:")
    self.assertTrue(error_intro.lines[12].endswith(str(type(tf_error))))

    self.assertEqual("Details:", error_intro.lines[14])
    self.assertStartsWith(error_intro.lines[15], "foo description")

  def testGetErrorIntroForNoOpName(self):
    # [FUNCTION DOC]
    # For an op error with no op, the intro must contain the warning that the op name cannot be
    # determined.
    tf_error = errors.OpError(None, None, "Fake OpError", -1)
    error_intro = cli_shared.get_error_intro(tf_error)
    self.assertIn("Cannot determine the name of the op", error_intro.lines[3])


if __name__ == "__main__":
  googletest.main()
