# [MODULE DOC]
# Tests for the profile data reader exposed by the jax profiler module. Each test builds an
# XSpace profile from a text-format protocol buffer (or from a serialized file) and checks the
# Python navigation API: planes, lines, events, their names, timing in nanoseconds and the
# statistics mappings, including empty and missing-metadata cases. Run directly it starts the
# absltest runner.
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

from absl.testing import absltest
from jax import profiler


class ProfileDataTest(absltest.TestCase):
  # [CLASS DOC]
  # Test case for the profile data wrapper. Has no fixtures; each test builds its own profile from
  # text and asserts on the planes, lines, events and stats the API returns.

  def test_find_plane_with_name(self):
    # [FUNCTION DOC]
    # Test: profile with planes named a and b; finding a plane by name returns the right plane for
    # a and b and returns the null value for the unknown name c.
    # Takes no arguments and returns nothing; failed assertions signal a failure.
    profile = profiler.ProfileData.from_text_proto("""
        planes { name: "a" }
        planes { name: "b" }
        """)
    self.assertEqual(profile.find_plane_with_name('a').name, 'a')
    self.assertEqual(profile.find_plane_with_name('b').name, 'b')
    self.assertIsNone(profile.find_plane_with_name('c'))

  def test_visit_space(self):
    # [FUNCTION DOC]
    # Test: profile with two planes; iterating over the planes yields their names in order, a then
    # b.
    # Takes no arguments and returns nothing; failed assertions signal a failure.
    space = profiler.ProfileData.from_text_proto("""
        planes { name: "a" }
        planes { name: "b" }
        """)
    self.assertEqual([plane.name for plane in space.planes], ['a', 'b'])

  def test_visit_empty_space(self):
    # [FUNCTION DOC]
    # Test: profile parsed from empty text; the planes iterator yields nothing.
    # Takes no arguments and returns nothing; failed assertions signal a failure.
    space = profiler.ProfileData.from_text_proto('')
    self.assertEmpty(list(space.planes))

  def test_visit_plane(self):
    # [FUNCTION DOC]
    # Test: one plane with two lines and one string statistic whose metadata name is hello;
    # asserts the plane name, the line names in order, and that the statistics mapping equals
    # hello mapped to world.
    # Takes no arguments and returns nothing; failed assertions signal a failure.
    profile = profiler.ProfileData.from_text_proto("""
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
    # Test: profile with a single empty plane; asserts the name is empty and the lines list is
    # empty.
    # Takes no arguments and returns nothing; failed assertions signal a failure.
    profile = profiler.ProfileData.from_text_proto('planes {}')

    plane = next(profile.planes)
    self.assertEmpty(plane.name)
    self.assertEmpty(list(plane.lines))

  def test_visit_line(self):
    # [FUNCTION DOC]
    # Test: plane with one line holding two events whose metadata names are foo and bar; asserts
    # there is exactly one line named t100 and its event names in order are foo then bar.
    # Takes no arguments and returns nothing; failed assertions signal a failure.
    profile = profiler.ProfileData.from_text_proto("""
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
    # Test: plane with one empty line; asserts exactly one line exists, its name is empty and it
    # has no events.
    # Takes no arguments and returns nothing; failed assertions signal a failure.
    profile = profiler.ProfileData.from_text_proto("""
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
    # Test: line with timestamp 1000 ns and an event at offset 500000 ps lasting 600000 ps with a
    # double, an unsigned and a reference statistic; asserts start 1500, duration 600, end 2100
    # (ns as floats), name hlo, and the statistics mapping flops to 400.0, bytes to 1024 and
    # provenance to the referenced statistic name tf_op.
    # Takes no arguments and returns nothing; failed assertions signal a failure.
    profile = profiler.ProfileData.from_text_proto("""
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
    # Test: event referencing metadata ids absent from the tables; asserts the event name is
    # empty, and after filtering out statistics without a name, the mapping has flops 400.0 and
    # provenance as an empty string (the reference resolves to an empty string because the
    # referenced metadata is missing).
    # Takes no arguments and returns nothing; failed assertions signal a failure.
    profile = profiler.ProfileData.from_text_proto("""
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
    # Test: serializes a text-format profile with planes a and b to bytes, writes them to a
    # temporary file, loads the profile from that file path and asserts the plane names are a and
    # b. Side effect: creates a temporary file managed by the test framework.
    # Takes no arguments and returns nothing; failed assertions signal a failure.
    serialized = profiler.ProfileData.text_proto_to_serialized_xspace("""
        planes { name: "a" }
        planes { name: "b" }
        """)
    tmp_file = self.create_tempfile().full_path
    with open(tmp_file, 'wb') as f:
      f.write(serialized)
    profile = profiler.ProfileData.from_file(tmp_file)
    self.assertEqual([plane.name for plane in profile.planes], ['a', 'b'])


if __name__ == '__main__':
  absltest.main()
