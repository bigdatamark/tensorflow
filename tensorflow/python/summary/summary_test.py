# [MODULE DOC]
# Test module for the API surface of the version 1 summary operations (scalar, image, histogram,
# audio, text). Checks tags, operation names, family handling, name sanitization, dtype
# acceptance and meta graph export/import; it deliberately does not verify full serialized
# values for complex summaries (those live in kernel tests). Run as a test binary.
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
"""Tests for the API surface of the V1 tf.summary ops.

These tests don't check the actual serialized proto summary value for the
more complex summaries (e.g. audio, image).  Those test live separately in
tensorflow/python/kernel_tests/summary_v1_*.py.
"""


from tensorflow.core.framework import summary_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib


class SummaryTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the version 1 summary-creating functions. Most tests run in graph mode with a
  # cached session, evaluate the summary tensor, parse the resulting serialized summary message
  # and assert on tags and values. No shared fixtures.

  @test_util.run_deprecated_v1
  def testScalarSummary(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): creates a scalar summary of the constant three inside a name scope,
    # evaluates it, and asserts one value whose tag includes the scope prefix and whose simple
    # value is three.
    with self.cached_session() as s:
      i = constant_op.constant(3)
      with ops.name_scope('outer'):
        im = summary_lib.scalar('inner', i)
      summary_str = s.run(im)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 1)
    self.assertEqual(values[0].tag, 'outer/inner')
    self.assertEqual(values[0].simple_value, 3.0)

  @test_util.run_deprecated_v1
  def testScalarSummaryWithFamily(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): creates two scalar summaries with the same name and a family inside a
    # name scope. Asserts the operation names include the family and get a uniquifying suffix on
    # the second, and that the tags are prefixed by the family, with the value seven.
    with self.cached_session() as s:
      i = constant_op.constant(7)
      with ops.name_scope('outer'):
        im1 = summary_lib.scalar('inner', i, family='family')
        self.assertEqual(im1.op.name, 'outer/family/inner')
        im2 = summary_lib.scalar('inner', i, family='family')
        self.assertEqual(im2.op.name, 'outer/family/inner_1')
      sm1, sm2 = s.run([im1, im2])
    summary = summary_pb2.Summary()

    summary.ParseFromString(sm1)
    values = summary.value
    self.assertEqual(len(values), 1)
    self.assertEqual(values[0].tag, 'family/outer/family/inner')
    self.assertEqual(values[0].simple_value, 7.0)

    summary.ParseFromString(sm2)
    values = summary.value
    self.assertEqual(len(values), 1)
    self.assertEqual(values[0].tag, 'family/outer/family/inner_1')
    self.assertEqual(values[0].simple_value, 7.0)

  @test_util.run_deprecated_v1
  def testSummarizingVariable(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): creates a variable from a constant forty-two, initializes it,
    # summarizes it as a scalar and asserts the tag is the plain name and the value is 42.
    with self.cached_session() as s:
      c = constant_op.constant(42.0)
      v = variables.Variable(c)
      ss = summary_lib.scalar('summary', v)
      init = variables.global_variables_initializer()
      s.run(init)
      summ_str = s.run(ss)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summ_str)
    self.assertEqual(len(summary.value), 1)
    value = summary.value[0]
    self.assertEqual(value.tag, 'summary')
    self.assertEqual(value.simple_value, 42.0)

  @test_util.run_deprecated_v1
  def testImageSummary(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): summarizes a batch of five all-ones images with at most three
    # outputs in a name scope, and asserts exactly three values with tags numbered zero to two
    # under the scoped name plus an image segment.
    with self.cached_session() as s:
      i = array_ops.ones((5, 4, 4, 3))
      with ops.name_scope('outer'):
        im = summary_lib.image('inner', i, max_outputs=3)
      summary_str = s.run(im)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted('outer/inner/image/{}'.format(i) for i in range(3))
    self.assertEqual(tags, expected)

  @test_util.run_deprecated_v1
  def testImageSummaryWithFamily(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): as the plain image test but with a family; asserts the operation name
    # contains the family and the three tags are prefixed by the family.
    with self.cached_session() as s:
      i = array_ops.ones((5, 2, 3, 1))
      with ops.name_scope('outer'):
        im = summary_lib.image('inner', i, max_outputs=3, family='family')
        self.assertEqual(im.op.name, 'outer/family/inner')
      summary_str = s.run(im)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted(
        'family/outer/family/inner/image/{}'.format(i) for i in range(3))
    self.assertEqual(tags, expected)

  @test_util.run_deprecated_v1
  def testHistogramSummary(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): creates a histogram summary of an all-ones tensor in a name scope and
    # asserts one value with the scoped tag.
    with self.cached_session() as s:
      i = array_ops.ones((5, 4, 4, 3))
      with ops.name_scope('outer'):
        summ_op = summary_lib.histogram('inner', i)
      summary_str = s.run(summ_op)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    self.assertEqual(len(summary.value), 1)
    self.assertEqual(summary.value[0].tag, 'outer/inner')

  @test_util.run_deprecated_v1
  def testHistogramSummaryWithFamily(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): histogram summary with a family; asserts the operation name includes
    # the family and the single tag is prefixed by the family.
    with self.cached_session() as s:
      i = array_ops.ones((5, 4, 4, 3))
      with ops.name_scope('outer'):
        summ_op = summary_lib.histogram('inner', i, family='family')
        self.assertEqual(summ_op.op.name, 'outer/family/inner')
      summary_str = s.run(summ_op)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    self.assertEqual(len(summary.value), 1)
    self.assertEqual(summary.value[0].tag, 'family/outer/family/inner')

  def testHistogramSummaryTypes(self):
    # [FUNCTION DOC]
    # Test: builds a histogram summary from a constant of each of several integer and floating
    # point types (8, 16 and 32 bit integers, unsigned 8 bit, 32 and 64 bit floats). Passes if
    # none raises; nothing is evaluated or asserted beyond construction succeeding.
    for dtype in (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.int32,
                  dtypes.float32, dtypes.float64):
      const = constant_op.constant(10, dtype=dtype)
      summary_lib.histogram('h', const)

  @test_util.run_deprecated_v1
  def testAudioSummary(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): summarizes a batch of five all-ones audio clips at sample rate 0.2
    # with at most three outputs in a name scope; asserts three values tagged zero to two under
    # the scoped name plus an audio segment.
    with self.cached_session() as s:
      i = array_ops.ones((5, 3, 4))
      with ops.name_scope('outer'):
        aud = summary_lib.audio('inner', i, 0.2, max_outputs=3)
      summary_str = s.run(aud)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted('outer/inner/audio/{}'.format(i) for i in range(3))
    self.assertEqual(tags, expected)

  @test_util.run_deprecated_v1
  def testAudioSummaryWithFamily(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): audio summary with a family; asserts the operation name includes the
    # family and the three tags are prefixed by the family.
    with self.cached_session() as s:
      i = array_ops.ones((5, 3, 4))
      with ops.name_scope('outer'):
        aud = summary_lib.audio('inner', i, 0.2, max_outputs=3, family='family')
        self.assertEqual(aud.op.name, 'outer/family/inner')
      summary_str = s.run(aud)
    summary = summary_pb2.Summary()
    summary.ParseFromString(summary_str)
    values = summary.value
    self.assertEqual(len(values), 3)
    tags = sorted(v.tag for v in values)
    expected = sorted(
        'family/outer/family/inner/audio/{}'.format(i) for i in range(3))
    self.assertEqual(tags, expected)

  def testAudioSummaryWithInvalidSampleRate(self):
    # [FUNCTION DOC]
    # Test: builds and evaluates an audio summary whose sample rate is a two-element list rather
    # than a scalar, and expects an invalid-argument error.
    with self.assertRaises(errors.InvalidArgumentError):
      invalid_sample_rate = [22000.0, 22000.0]
      self.evaluate(summary_lib.audio('', [[1.0]], invalid_sample_rate))

  @test_util.run_deprecated_v1
  def testTextSummary(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): a text summary of a numeric tensor raises a value error; a text
    # summary of a vector of strings and of a scalar string both succeed and produce a
    # tensor-summary operation of the version 2 kind.
    with self.cached_session():
      with self.assertRaises(ValueError):
        num = array_ops.constant(1)
        summary_lib.text('foo', num)

      # The API accepts vectors.
      arr = array_ops.constant(['one', 'two', 'three'])
      summ = summary_lib.text('foo', arr)
      self.assertEqual(summ.op.type, 'TensorSummaryV2')

      # the API accepts scalars
      summ = summary_lib.text('foo', array_ops.constant('one'))
      self.assertEqual(summ.op.type, 'TensorSummaryV2')

  @test_util.run_deprecated_v1
  def testSummaryNameConversion(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): scalar summaries whose names contain spaces, illegal characters or a
    # leading slash get operation names with illegal characters replaced by underscores and the
    # leading slash removed; asserts the exact sanitized operation names.
    c = constant_op.constant(3)
    s = summary_lib.scalar('name with spaces', c)
    self.assertEqual(s.op.name, 'name_with_spaces')

    s2 = summary_lib.scalar('name with many $#illegal^: characters!', c)
    self.assertEqual(s2.op.name, 'name_with_many___illegal___characters_')

    s3 = summary_lib.scalar('/name/with/leading/slash', c)
    self.assertEqual(s3.op.name, 'name/with/leading/slash')

  @test_util.run_deprecated_v1
  def testSummaryWithFamilyMetaGraphExport(self):
    # [FUNCTION DOC]
    # Test (version 1 mode): creates summaries with and without family inside a name scope,
    # exports the meta graph for that scope, and imports it into a new graph under a different
    # scope. Asserts the summary tensors exist under the new scope names but that the evaluated
    # tags still carry the original scope and family, since tags are not renamed on import.
    with ops.name_scope('outer'):
      i = constant_op.constant(11)
      summ = summary_lib.scalar('inner', i)
      self.assertEqual(summ.op.name, 'outer/inner')
      summ_f = summary_lib.scalar('inner', i, family='family')
      self.assertEqual(summ_f.op.name, 'outer/family/inner')

    metagraph_def, _ = meta_graph.export_scoped_meta_graph(export_scope='outer')

    with ops.Graph().as_default() as g:
      meta_graph.import_scoped_meta_graph(metagraph_def, graph=g,
                                          import_scope='new_outer')
      # The summaries should exist, but with outer scope renamed.
      new_summ = g.get_tensor_by_name('new_outer/inner:0')
      new_summ_f = g.get_tensor_by_name('new_outer/family/inner:0')

      # However, the tags are unaffected.
      with self.cached_session() as s:
        new_summ_str, new_summ_f_str = s.run([new_summ, new_summ_f])
        new_summ_pb = summary_pb2.Summary()
        new_summ_pb.ParseFromString(new_summ_str)
        self.assertEqual('outer/inner', new_summ_pb.value[0].tag)
        new_summ_f_pb = summary_pb2.Summary()
        new_summ_f_pb.ParseFromString(new_summ_f_str)
        self.assertEqual('family/outer/family/inner',
                         new_summ_f_pb.value[0].tag)


if __name__ == '__main__':
  test.main()
