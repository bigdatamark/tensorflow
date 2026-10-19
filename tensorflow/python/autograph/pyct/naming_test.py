# [MODULE DOC]
# Unit tests for the autograph naming module's Namer class: tracking of generated names,
# avoidance of duplicates, and avoidance of conflicts with global and reserved local names.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for naming module."""

from tensorflow.python.autograph.pyct import naming
from tensorflow.python.platform import test


class NamerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Namer symbol generator; each test creates its own namer, no shared fixtures.

  def test_new_symbol_tracks_names(self):
    # [FUNCTION DOC]
    # With an empty global namespace, requests a name for "temp" and asserts it is returned
    # unchanged and recorded in the namer's set of generated names.
    namer = naming.Namer({})
    self.assertEqual('temp', namer.new_symbol('temp', set()))
    self.assertItemsEqual(('temp',), namer.generated_names)

  def test_new_symbol_avoids_duplicates(self):
    # [FUNCTION DOC]
    # Requests "temp" twice from the same namer and asserts the first result is "temp", the
    # second is "temp_1", and both are recorded as generated.
    namer = naming.Namer({})
    self.assertEqual('temp', namer.new_symbol('temp', set()))
    self.assertEqual('temp_1', namer.new_symbol('temp', set()))
    self.assertItemsEqual(('temp', 'temp_1'), namer.generated_names)

  def test_new_symbol_avoids_conflicts(self):
    # [FUNCTION DOC]
    # With "temp" present in the global namespace, asserts the first request yields "temp_1";
    # then, with "temp_2" reserved as a local, asserts the next request yields "temp_3" (skipping
    # the reserved and already generated names); both generated names are recorded.
    namer = naming.Namer({'temp': 1})
    # temp is reserved in the global namespace
    self.assertEqual('temp_1', namer.new_symbol('temp', set()))
    # temp_2 is reserved in the local namespace
    self.assertEqual('temp_3', namer.new_symbol('temp', set(('temp_2',))))
    self.assertItemsEqual(('temp_1', 'temp_3'), namer.generated_names)


if __name__ == '__main__':
  test.main()
