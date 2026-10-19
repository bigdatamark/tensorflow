# [MODULE DOC]
# HELPER: Helper module with a shared test base class for tests of composite op definitions.
# Provides a test case class whose assertion helper compares a real op (with its registered
# gradient) against its composite Python implementation, for both values and gradients.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test utils for composite op definition."""
from tensorflow.python.eager import backprop
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class OpsDefsTest(test.TestCase):
  # [CLASS DOC]
  # HELPER: Base test case for composite op tests, providing the op-versus-composite comparison.
  # Has no state; subclasses call the assertion helper from their test methods.
  """Test utils."""

  def _assertOpAndComposite(self, vars_, compute_op, compute_composite, kwargs,
                            op_kwargs=None):
    # [FUNCTION DOC]
    # HELPER: Assertion helper shared by composite op tests.
    # Parameters: vars_ is a list of tensors/variables to differentiate with respect to;
    # compute_op is a callable that runs the real op; compute_composite is a callable that runs the
    # composite implementation; kwargs is a mapping of keyword arguments for the composite
    # callable; op_kwargs is an optional mapping for the op callable (defaults to kwargs).
    # Skips the test when the oneDNN custom op build is enabled. Otherwise computes the op result
    # and its gradients with respect to vars_ under a gradient tape, does the same for the
    # composite, then asserts element-wise closeness of each output and each gradient. Both
    # callables must return sequences of outputs. Returns nothing; fails through test assertions.
    if op_kwargs is None:
      op_kwargs = kwargs
    if test_util.IsMklEnabled():
      self.skipTest("Not compatible with oneDNN custom ops.")

    # compute with op.
    with backprop.GradientTape() as gt:
      for var_ in vars_:
        gt.watch(var_)
      y = compute_op(**op_kwargs)  # uses op and decomposites by the graph pass.
      grads = gt.gradient(y, vars_)  # uses registered gradient function.

    # compute with composition
    with backprop.GradientTape() as gt:
      for var_ in vars_:
        gt.watch(var_)
      re_y = compute_composite(**kwargs)  # uses composite function.
      re_grads = gt.gradient(re_y, vars_)  # uses gradients compposite function.

    for v, re_v in zip(y, re_y):
      self.assertAllClose(v, re_v)
    for g, re_g in zip(grads, re_grads):
      self.assertAllClose(g, re_g)
