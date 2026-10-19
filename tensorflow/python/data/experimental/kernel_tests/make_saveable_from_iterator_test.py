# [MODULE DOC]
# Integration tests for making a saveable object from a dataset iterator and saving/restoring
# it with a v1 Saver in graph mode (concurrent saves of many pipelines, and saving an
# uninitialized iterator). Test module only; no import-time side effects.
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
"""Integration test for dataset serialization."""
import os

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import iterator_ops as contrib_iterator_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib


class MakeSaveableFromIteratorTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering saver-based checkpointing of dataset iterators. Provides helpers that
  # build pipelines and a checkpoint path in the test temp directory.

  def _build_input_pipeline(self, name, num_outputs):
    # [FUNCTION DOC]
    # HELPER: builds one iterator pipeline for the save/restore tests.
    # Inside a name scope of the given name, builds a shuffled (no reshuffle) and prefetched
    # range dataset with num_outputs elements, makes an initializable iterator, registers its
    # saveable in the saveable-objects graph collection. Returns the iterator initializer op
    # and the get-next tensor.
    with ops.name_scope(name):
      ds = dataset_ops.Dataset.range(num_outputs).shuffle(
          10, reshuffle_each_iteration=False).prefetch(10)
      iterator = ds.make_initializable_iterator()
      saveable = contrib_iterator_ops.make_saveable_from_iterator(iterator)
      ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
      return iterator.initializer, iterator.get_next()

  def _build_graph(self, num_pipelines, num_outputs):
    # [FUNCTION DOC]
    # HELPER: builds the whole graph for the save/restore tests.
    # Creates num_pipelines input pipelines (each with num_outputs elements) and a v1 Saver
    # after them. Returns the list of initializer ops, the list of get-next tensors and the
    # saver.
    init_ops = []
    get_next_ops = []
    for i in range(num_pipelines):
      name = "input_pipeline_%d" % i
      init_op, get_next_op = self._build_input_pipeline(name, num_outputs)
      init_ops.append(init_op)
      get_next_ops.append(get_next_op)
    saver = saver_lib.Saver()
    return init_ops, get_next_ops, saver

  def _ckpt_path(self):
    # [FUNCTION DOC]
    # HELPER: returns the checkpoint file path (named "iterator") inside the test temp dir.
    return os.path.join(self.get_temp_dir(), "iterator")

  @combinations.generate(combinations.combine(tf_api_version=1, mode=["graph"]))
  def testConcurrentSaves(self):
    # [FUNCTION DOC]
    # Graph-mode (v1) test. Builds 10 pipelines of 10 elements, consumes 10 elements from each
    # and saves a checkpoint; then in a fresh graph restores it and consumes the remainder
    # (none, since break point equals the output count). Asserts each pipeline's collected
    # outputs, sorted, equal the full range.
    num_pipelines = 10
    num_outputs = 10
    break_point = 10
    all_outputs = [[] for _ in range(num_pipelines)]
    with ops.Graph().as_default() as g:
      init_ops, get_next_ops, saver = self._build_graph(num_pipelines,
                                                        num_outputs)
      with self.session(graph=g) as sess:
        self.evaluate(init_ops)
        for _ in range(break_point):
          output = self.evaluate(get_next_ops)
          for i in range(num_pipelines):
            all_outputs[i].append(output[i])
        saver.save(sess, self._ckpt_path())

    with ops.Graph().as_default() as g:
      init_ops, get_next_ops, saver = self._build_graph(num_pipelines,
                                                        num_outputs)
      with self.session(graph=g) as sess:
        self.evaluate(init_ops)
        saver.restore(sess, self._ckpt_path())
        for _ in range(num_outputs - break_point):
          output = self.evaluate(get_next_ops)
          for i in range(num_pipelines):
            all_outputs[i].append(output[i])

    for output in all_outputs:
      self.assertSequenceEqual(sorted(output), range(num_outputs))

  @combinations.generate(combinations.combine(tf_api_version=1, mode=["graph"]))
  def testUninitializedIterator(self):
    # [FUNCTION DOC]
    # Graph-mode (v1) test. Builds one pipeline but never runs its initializer, then saves;
    # asserts saving raises a failed-precondition error.
    num_pipelines = 1
    num_outputs = 1
    with ops.Graph().as_default() as g:
      _, _, saver = self._build_graph(num_pipelines, num_outputs)
      with self.session(graph=g) as sess:
        with self.assertRaises(errors.FailedPreconditionError):
          saver.save(sess, self._ckpt_path())


if __name__ == "__main__":
  test.main()
