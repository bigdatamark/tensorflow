# [MODULE DOC]
# Tests for the delegating-trackable mixin. Defines a small inner Trackable holding one variable,
# a wrapper combining the mixin with Trackable, and a parameterized test case that checks
# checkpoint round trips (with and without async checkpointing) and saved-model export/load
# through the wrapper. Test-only module; no import-time side effects.

# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for base_delegate."""
import os

from absl.testing import parameterized

from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import base
from tensorflow.python.trackable import base_delegate


class Inner(base.Trackable):
  # [CLASS DOC]
  # HELPER: test fixture for the delegating-mixin tests.
  # Simple Trackable that stores one variable as attribute v and registers it as a dependency
  # named "v". Overrides the CPU-copy hook so async checkpointing can copy it.

  def __init__(self, v):
    # [FUNCTION DOC]
    # Constructor: stores the given variable (any trackable value) as attribute v and tracks it
    # as a dependency named "v". Returns nothing; mutates the new instance's dependency state.
    self.v = v
    self._track_trackable(v, "v")

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # Hook used by async checkpointing. Inputs: object_map, a mapping from original objects to
    # their CPU copies, updated in place. If this object has no entry yet, adds a new Inner
    # sharing the same variable; then asks the variable to add its own CPU copy. Returns nothing.
    if self not in object_map:
      object_map[self] = Inner(self.v)
    self.v._copy_trackable_to_cpu(object_map)


class Wrapper(base_delegate.DelegatingTrackableMixin, base.Trackable):
  # [CLASS DOC]
  # HELPER: test fixture for the delegating-mixin tests.
  # Wraps an Inner object; all Trackable behavior is delegated to it by the mixin, so only the
  # inner object appears in checkpoints and saved models. Exposes the inner variable as v.

  def __init__(self, inner):
    # [FUNCTION DOC]
    # Constructor: keeps the inner object as attribute inner and calls the mixin constructor with
    # it so that all Trackable calls are delegated to the inner object. Returns nothing.
    self.inner = inner
    super(Wrapper, self).__init__(inner)

  @property
  def v(self):
    # [FUNCTION DOC]
    # Property returning the variable held by the inner object. No side effects.
    return self.inner.v


@test_util.run_all_in_graph_and_eager_modes
class BaseDelegateTest(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Tests that a Wrapper using the delegating mixin saves and restores exactly like its inner
  # object. Runs every test in both graph and eager modes; parameterized for async checkpointing.

  @parameterized.named_parameters(
      ("_enable_async_ckpt", True),
      ("_disable_async_ckpt", False)
  )
  def test_checkpoint(self, enable_async_ckpt):
    # [FUNCTION DOC]
    # Input: enable_async_ckpt, whether the checkpoint options request asynchronous saving.
    # Scenario: two wrapped variables (15 and -15) are initialized and saved through a checkpoint
    # of the wrappers under the test temp dir. Both are then overwritten (-3 and 12). A second
    # checkpoint object built from the inner objects (no wrappers) restores the saved prefix, after
    # syncing when async is on, and must consume everything; values must return to 15 and -15.
    # Side effect: writes checkpoint files to the temp directory.
    a = Wrapper(Inner(variables_lib.Variable(15.0)))
    b = Wrapper(Inner(variables_lib.Variable(-15.0)))
    self.evaluate([a.v.initializer, b.v.initializer])

    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    ckpt = util.Checkpoint(a=a, b=b)
    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_enable_async_checkpoint=enable_async_ckpt)
    prefix_tensor = ckpt.save(prefix, options=ckpt_options)

    self.assertEqual([15, -15], self.evaluate([a.v, b.v]))
    self.evaluate(a.v.assign(-3))
    self.evaluate(b.v.assign(12))
    self.assertEqual([-3, 12], self.evaluate([a.v, b.v]))

    # Test that the model can be saved with the wrapper and loaded without it.
    ckpt2 = util.Checkpoint(a=a.inner, b=b.inner)
    if enable_async_ckpt:
      ckpt.sync()
    ckpt2.restore(prefix_tensor).assert_consumed().run_restore_ops()
    self.assertEqual([15, -15], self.evaluate([a.v, b.v]))

  def test_saved_model(self):
    # [FUNCTION DOC]
    # Scenario: a wrapped variable with value -15 is initialized and exported with the saved-model
    # saver to the temp dir, then loaded back. The loaded object (the inner trackable, wrapper
    # disappears) must expose variable v, which after initialization evaluates to -15.
    # Side effect: writes a saved model to the temp directory.
    a = Wrapper(Inner(variables_lib.Variable(-15.0)))
    self.evaluate([a.v.initializer])
    self.assertEqual([-15], self.evaluate([a.v]))

    test_dir = self.get_temp_dir()
    saved_model_path = os.path.join(test_dir, "saved_model")
    save.save(a, saved_model_path)

    loaded = load.load(saved_model_path)
    self.evaluate([loaded.v.initializer])
    self.assertEqual([-15], self.evaluate([loaded.v]))


if __name__ == "__main__":
  test.main()
