# [MODULE DOC]
# HELPER: shared test library for the profiler tests (model analyzer and related tests).
# Provides functions that build small TensorFlow graphs (a conv model, a conv plus RNN plus
# optimizer model, and a splittable two-branch conv model), a tree search utility for profile
# nodes, a context manager that loads a profiler from a profile file, and a helper that strips
# the documentation header from profile output. Importing it has no side effects beyond imports
# (the gradient modules are imported only to register gradients).
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
"""A test lib that defines some models."""
import contextlib

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import variable_scope
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import _pywrap_tfprof as print_mdl
from tensorflow.python.util import compat


def BuildSmallModel():
  # [FUNCTION DOC]
  # HELPER: builds a model fragment for profiler tests.
  # Creates a zero image of shape 2x6x6x3, a scalar variable ScalarW (unused in the output) and
  # two strided same-padded convolutions with variables DW and DW2 created via the variable
  # scope (so names clash if called twice in one scope without reuse). Returns the output
  # tensor of the second convolution. Adds ops and variables to the default graph.
  """Build a small forward conv model."""
  image = array_ops.zeros([2, 6, 6, 3])
  _ = variable_scope.get_variable(
      'ScalarW', [],
      dtypes.float32,
      initializer=init_ops.random_normal_initializer(stddev=0.001))
  kernel = variable_scope.get_variable(
      'DW', [3, 3, 3, 6],
      dtypes.float32,
      initializer=init_ops.random_normal_initializer(stddev=0.001))
  x = nn_ops.conv2d(image, kernel, [1, 2, 2, 1], padding='SAME')
  kernel = variable_scope.get_variable(
      'DW2', [2, 2, 6, 12],
      dtypes.float32,
      initializer=init_ops.random_normal_initializer(stddev=0.001))
  x = nn_ops.conv2d(x, kernel, [1, 2, 2, 1], padding='SAME')
  return x


def BuildFullModel():
  # [FUNCTION DOC]
  # HELPER: builds a larger model for profiler tests.
  # Builds four small conv models in separate variable scopes, flattens and concatenates them
  # into a sequence, runs a basic RNN cell with 16 units over it, computes an L2 loss of the
  # mean difference to an all-ones target, and builds a gradient descent train op with learning
  # rate 0.01. Returns that training operation. Adds many ops and variables to the default graph.
  """Build the full model with conv,rnn,opt."""
  seq = []
  for i in range(4):
    with variable_scope.variable_scope('inp_%d' % i):
      seq.append(array_ops.reshape(BuildSmallModel(), [2, 1, -1]))

  cell = rnn_cell.BasicRNNCell(16)
  out = rnn.dynamic_rnn(
      cell, array_ops.concat(seq, axis=1), dtype=dtypes.float32)[0]

  target = array_ops.ones_like(out)
  loss = nn_ops.l2_loss(math_ops.reduce_mean(target - out))
  sgd_op = gradient_descent.GradientDescentOptimizer(1e-2)
  return sgd_op.minimize(loss)


def BuildSplittableModel():
  # [FUNCTION DOC]
  # HELPER: builds a small model whose two branches can be fetched separately in different steps.
  # Creates a zero image and two convolutions on it (variables DW and DW2 with different
  # kernel shapes), plus their sum. Returns a tuple of the first branch, second branch and sum
  # tensors. Adds ops and variables to the default graph.
  """Build a small model that can be run partially in each step."""
  image = array_ops.zeros([2, 6, 6, 3])

  kernel1 = variable_scope.get_variable(
      'DW', [3, 3, 3, 6],
      dtypes.float32,
      initializer=init_ops.random_normal_initializer(stddev=0.001))
  r1 = nn_ops.conv2d(image, kernel1, [1, 2, 2, 1], padding='SAME')

  kernel2 = variable_scope.get_variable(
      'DW2', [2, 3, 3, 6],
      dtypes.float32,
      initializer=init_ops.random_normal_initializer(stddev=0.001))
  r2 = nn_ops.conv2d(image, kernel2, [1, 2, 2, 1], padding='SAME')

  r3 = r1 + r2
  return r1, r2, r3


def SearchTFProfNode(node, name):
  # [FUNCTION DOC]
  # HELPER: lookup utility for profile node trees.
  # Depth-first search starting at the given node for a node whose name equals the given name,
  # visiting children in order. Returns the first matching node or nothing if none matches.
  """Search a node in the tree."""
  if node.name == name:
    return node
  for c in node.children:
    r = SearchTFProfNode(c, name)
    if r: return r
  return None


@contextlib.contextmanager
def ProfilerFromFile(profile_file):
  # [FUNCTION DOC]
  # HELPER: context manager giving tests a profiler object backed by a profile file.
  # KNOWN ISSUE: the cleanup call that deletes the native profiler is not in a finally block,
  # so if the body of the with statement raises, the native profiler is never deleted.
  # Loads the file (path converted to bytes) into the native profiler, creates a bare Profiler
  # instance without running its constructor, and yields it. After the body completes it
  # deletes the native profiler. Side effect: global native profiler state.
  """Initialize a profiler from profile file."""
  print_mdl.ProfilerFromFile(compat.as_bytes(profile_file))
  profiler = model_analyzer.Profiler.__new__(model_analyzer.Profiler)
  yield profiler
  print_mdl.DeleteProfiler()


def CheckAndRemoveDoc(profile):
  # [FUNCTION DOC]
  # HELPER: strips the documentation header from profile text in tests.
  # KNOWN ISSUE: the cut position is a hard-coded offset of nine characters after the start of
  # the profile marker, and if the marker is missing the search result of minus one yields an
  # offset of eight, silently returning a wrong slice.
  # Asserts the text contains the documentation marker (assertion error otherwise) and returns
  # the text following the profile marker plus its trailing newline and space.
  assert 'Doc:' in profile
  start_pos = profile.find('Profile:')
  return profile[start_pos + 9:]
