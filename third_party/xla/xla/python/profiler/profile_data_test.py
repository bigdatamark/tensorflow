# [MODULE DOC]
# Unit tests for the JAX profiler's ProfileData wrapper (parsing of XSpace profiles).
# Each test builds a profile from a text-format protocol buffer string (or a serialized file),
# then walks the planes, lines, events and statistics objects and checks names, counts, timestamps
# and statistic values. Runs under absltest; no import-time side effects.
# Copyright 2025 The OpenXLA Authors.
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
"""Unit tests for profile_data.py."""

from absl.testing import absltest
from jax import profiler as profile_data


class ProfileDataTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case covering the ProfileData profile-reading API: finding planes by name, iterating
  # planes, lines and events, reading statistics, and loading a profile from a serialized file.
  # No shared fixtures; every test builds its own profile from text-format protocol buffer input.

  def test_find_plane_with_name(self):
    # [FUNCTION DOC]
    # Builds a profile with two planes named a and b and checks that looking up a plane by name
    # returns the plane with that name for a and b, and returns nothing for the unknown name c.
    profile = profile_data.ProfileData.from_text_proto("""
        planes { name: "a" }
        planes { name: "b" }
        """)
    self.assertEqual(profile.find_plane_with_name('a').name, 'a')
    self.assertEqual(profile.find_plane_with_name('b').name, 'b')
    self.assertIsNone(profile.find_plane_with_name('c'))

  def test_visit_space(self):
    # [FUNCTION DOC]
    # Builds a profile with two planes and checks that iterating the planes yields their names in
    # order (a then b).
    space = profile_data.ProfileData.from_text_proto("""
        planes { name: "a" }
        planes { name: "b" }
        """)
    self.assertEqual([plane.name for plane in space.planes], ['a', 'b'])

  def test_visit_empty_space(self):
    # [FUNCTION DOC]
    # Parses an empty text profile and asserts that the resulting space has no planes.
    space = profile_data.ProfileData.from_text_proto('')
    self.assertEmpty(list(space.planes))

  def test_visit_plane(self):
    # [FUNCTION DOC]
    # Builds a plane p0 with two lines, one string statistic referencing a named statistic
    # metadata entry. Checks the plane name, the ordered line names (t1, t2), and that the
    # statistics, viewed as a mapping, translate the metadata name hello to the value world.
    profile = profile_data.ProfileData.from_text_proto("""
        planes {
          name: "p0"
          lines { name: "t1" }
          lines { name: "t2" }
          stats { metadata_id: 1 str_value: "world" }
          stat_metadata {
            key: 1
            value { name: "hello" }
          }
        }
        """)
    plane = profile.find_plane_with_name('p0')
    self.assertEqual(plane.name, 'p0')
    self.assertEqual([line.name for line in plane.lines], ['t1', 't2'])
    self.assertEqual(dict(plane.stats), {'hello': 'world'})

  def test_visit_empty_plane(self):
    # [FUNCTION DOC]
    # Parses a profile containing one completely empty plane and asserts that its name is empty
    # and it has no lines.
    profile = profile_data.ProfileData.from_text_proto('planes {}')

    plane = next(profile.planes)
    self.assertEmpty(plane.name)
    self.assertEmpty(list(plane.lines))

  def test_visit_line(self):
    # [FUNCTION DOC]
    # Builds a plane with one line t100 holding two events whose names come from the plane's
    # event metadata (foo and bar). Asserts there is exactly one line, its name is t100, and its
    # events are listed in order with the resolved names.
    profile = profile_data.ProfileData.from_text_proto("""
        planes {
          name: "p0"
          lines {
            name: "t100"
            events { metadata_id: 1 }
            events { metadata_id: 2 }
          }
          event_metadata {
            key: 1
            value { name: "foo" }
          }
          event_metadata {
            key: 2
            value { name: "bar" }
          }
        }
        """)
    plane = next(profile.planes)
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    self.assertEqual(line.name, 't100')
    self.assertListEqual([event.name for event in line.events], ['foo', 'bar'])

  def test_visit_empty_line(self):
    # [FUNCTION DOC]
    # Parses a plane that has a single empty line and asserts the line exists, has an empty name
    # and contains no events.
    profile = profile_data.ProfileData.from_text_proto("""
        planes {
          name: "p0"
          lines {}
        }
        """)
    plane = next(profile.planes)
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    self.assertEmpty(line.name)
    self.assertEmpty(list(line.events))

  def test_visit_event(self):
    # [FUNCTION DOC]
    # Builds one line (start timestamp 1000 ns) with one event of offset 500000 ps and duration
    # 600000 ps plus three statistics (a floating point, an unsigned integer and a reference to
    # another statistic metadata name). Asserts start 1500 ns, duration 600 ns, end 2100 ns,
    # the event name from metadata (hlo), and that statistics map to flops 400.0, bytes 1024 and
    # the referenced name for the provenance statistic.
    profile = profile_data.ProfileData.from_text_proto("""
        planes {
          name: "p0"
          lines {
            timestamp_ns: 1000
            events {
              metadata_id: 1
              offset_ps: 500000
              duration_ps: 600000
              stats { metadata_id: 1 double_value: 400.0 }
              stats { metadata_id: 2 uint64_value: 1024 }
              stats { metadata_id: 3 ref_value: 4 }
            }
          }
          event_metadata {
            key: 1
            value { name: "hlo" }
          }
          stat_metadata {
            key: 1
            value { name: "flops" }
          }
          stat_metadata {
            key: 2
            value { name: "bytes" }
          }
          stat_metadata {
            key: 3
            value { name: "provenance" }
          }
          stat_metadata {
            key: 4
            value { name: "tf_op" }
          }
        }
        """)
    plane = next(profile.planes)
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    events = list(line.events)
    self.assertLen(events, 1)
    event = events[0]
    self.assertEqual(event.start_ns, 1500.0)
    self.assertEqual(event.duration_ns, 600.0)
    self.assertEqual(event.end_ns, 2100.0)
    self.assertEqual(event.name, 'hlo')
    self.assertEqual(
        dict(event.stats),
        {'flops': 400.0, 'bytes': 1024, 'provenance': 'tf_op'},
    )

  def test_visit_event_missing_metadata(self):
    # [FUNCTION DOC]
    # Builds an event whose event metadata entry is absent and whose statistics partly lack
    # metadata (no entry for the bytes statistic, and the reference target is missing). Asserts
    # the event name falls back to an empty string and that, after dropping entries with a null
    # key, the statistics mapping contains flops 400.0 and provenance as an empty string.
    profile = profile_data.ProfileData.from_text_proto("""
        planes {
          name: "p0"
          lines {
            timestamp_ns: 1000
            events {
              metadata_id: 1
              stats { metadata_id: 1 double_value: 400.0 }
              stats { metadata_id: 2 uint64_value: 1024 }
              stats { metadata_id: 3 ref_value: 4 }
            }
          }
          stat_metadata {
            key: 1
            value { name: "flops" }
          }
          stat_metadata {
            key: 3
            value { name: "provenance" }
          }
        }
        """)
    plane = next(profile.planes)
    lines = list(plane.lines)
    self.assertLen(lines, 1)
    line = lines[0]
    events = list(line.events)
    self.assertLen(events, 1)
    event = events[0]
    self.assertEqual(event.name, '')
    self.assertEqual(
        dict(filter(lambda x: x[0] is not None, event.stats)),
        {'flops': 400.0, 'provenance': ''},
    )

  def test_create_profile_data_from_file(self):
    # [FUNCTION DOC]
    # Serializes a two-plane text profile to bytes, writes it to a temporary file created by the
    # test framework (side effect: a temp file on disk), loads it back through the from-file
    # constructor and asserts the plane names a and b are preserved in order.
    serialized = profile_data.ProfileData.text_proto_to_serialized_xspace("""
        planes { name: "a" }
        planes { name: "b" }
        """)
    tmp_file = self.create_tempfile().full_path
    with open(tmp_file, 'wb') as f:
      f.write(serialized)
    profile = profile_data.ProfileData.from_file(tmp_file)
    self.assertEqual([plane.name for plane in profile.planes], ['a', 'b'])


if __name__ == '__main__':
  absltest.main()
