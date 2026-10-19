# [MODULE DOC]
# Unit tests for TrackableView child lookup and descendant listing on a tiny two-node graph.
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
"""Tests for the trackable view."""

from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.eager import test
from tensorflow.python.trackable import base


class TrackableViewTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the trackable view using a root trackable with one tracked leaf child.

  def test_children(self):
    # [FUNCTION DOC]
    # Tracks a leaf under a root with the name leaf, then expects the class-level children lookup
    # on the root to return exactly one entry: name leaf mapped to the same leaf object.
    root = base.Trackable()
    leaf = base.Trackable()
    root._track_trackable(leaf, name="leaf")
    (current_name,
     current_dependency), = trackable_view.TrackableView.children(root).items()
    self.assertIs(leaf, current_dependency)
    self.assertEqual("leaf", current_name)

  def test_descendants(self):
    # [FUNCTION DOC]
    # Same root and leaf graph; expects the view's descendants to be a two-element list with the
    # root first and the leaf second.
    root = base.Trackable()
    leaf = base.Trackable()
    root._track_trackable(leaf, name="leaf")
    descendants = trackable_view.TrackableView(root).descendants()
    self.assertIs(2, len(descendants))
    self.assertIs(root, descendants[0])
    self.assertIs(leaf, descendants[1])


if __name__ == "__main__":
  test.main()
