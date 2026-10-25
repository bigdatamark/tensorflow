# [MODULE DOC]
# HELPER: utility module shared by the proto splitter modules (splitters, joiners and tests).
# Provides byte-size formatting, navigation of protobuf messages by a path of field names, field
# numbers, map keys and list indices (get_field), generation of the matching field-index protos
# (get_field_tag), and a repeated-field check. Defines the FieldTypes type alias and private
# lookup tables. No import-time side effects other than defining these names.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for Proto Splitter modules."""

from collections.abc import Sequence
from typing import Any, Optional, Union

from google.protobuf import descriptor
from google.protobuf import message
from tensorflow.tools.proto_splitter import chunk_pb2

_BYTE_UNITS = [(1, "B"), (1 << 10, "KiB"), (1 << 20, "MiB"), (1 << 30, "GiB")]


def format_bytes(b: int) -> str:
  # [FUNCTION DOC]
  # HELPER: formats sizes for messages (for example error text about oversized nodes).
  # Input: b, a number of bytes (an integer; floats also work). Output: a string with the value
  # scaled to the largest fitting unit among bytes, KiB, MiB and GiB (values of a GiB or more
  # stay in GiB), shown with at most two decimals and with trailing zeros and a trailing decimal
  # point removed, with no space before the unit. No exceptions; no side effects.
  """Formats bytes into a human-readable string."""
  for i in range(1, len(_BYTE_UNITS)):
    if b < _BYTE_UNITS[i][0]:
      n = f"{b / _BYTE_UNITS[i-1][0]:.2f}"
      units = _BYTE_UNITS[i - 1][1]
      break
  else:
    n = f"{b / _BYTE_UNITS[-1][0]:.2f}"
    units = _BYTE_UNITS[-1][1]
  n = n.rstrip("0").rstrip(".")
  return f"{n}{units}"


FieldTypes = Union[str, int, bool, Sequence[Union[str, int, bool]]]


def get_field(
    proto: message.Message, fields: FieldTypes
) -> tuple[Any, Optional[descriptor.FieldDescriptor]]:
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the field is None if none are found, but a missing map key or
  # out-of-range list index yields None only for the value, while an empty path returns the
  # proto itself together with a missing (None) descriptor.
  # Follows a path through a proto and returns the field value and its descriptor. Inputs:
  # proto, the starting message; fields, a single name/number or a list of field names or
  # numbers, with map keys and list indices following the map or repeated field they apply to.
  # Returns a pair of the value reached by the final step and the descriptor of the last field
  # (descriptor is nothing for an empty path). Raises key errors for unknown names or numbers,
  # value errors if the path goes below a non-message field, and a type error from the walker.
  """Returns the field and field descriptor from the proto.

  Args:
    proto: Parent proto of any message type.
    fields: List of string/int/map key fields, e.g. ["nodes", "attr", "value"]
      can represent `proto.nodes.attr["value"]`.

  Returns:
    Tuple of (
      Field in the proto or `None` if none are found,
      Field descriptor
    )
  """
  field_proto = proto
  field_desc = None
  for field_proto, field_desc, _, _ in _walk_fields(proto, fields):
    pass
  return field_proto, field_desc


def get_field_tag(
    proto: message.Message, fields: FieldTypes
) -> Sequence[chunk_pb2.FieldIndex]:
  # [FUNCTION DOC]
  # Builds the list of field-index protos describing a path into a proto. Inputs: proto, the
  # starting message; fields, a path in the same form accepted by get_field. For every field
  # step it emits an entry holding the field number, followed by an extra entry for a map key
  # (typed by the map's key type) or a list index when one is present in the path.
  # Returns the list of entries (longer than the path when map keys or indices are used).
  # Raises the same key, value and type errors as the field walker. No side effects.
  """Generates FieldIndex proto for a nested field within a proto.

  Args:
    proto: Parent proto of any message type.
    fields: List of string/int/map key fields, e.g. ["nodes", "attr", "value"]
      can represent `proto.nodes.attr["value"]`.

  Returns:
    A list of FieldIndex protos with the same length as `fields`.
  """
  field_tags = []
  for _, field_desc, map_key, list_index in _walk_fields(proto, fields):
    field_tags.append(chunk_pb2.FieldIndex(field=field_desc.number))
    if map_key is not None:
      key_type = field_desc.message_type.fields_by_name["key"].type
      field_tags.append(
          chunk_pb2.FieldIndex(map_key=_map_key_proto(key_type, map_key))
      )
    elif list_index is not None:
      field_tags.append(chunk_pb2.FieldIndex(index=list_index))
  return field_tags


def _walk_fields(proto: message.Message, fields: FieldTypes):
  # [FUNCTION DOC]
  # HELPER: shared path walker behind get_field and get_field_tag.
  # KNOWN ISSUE: the type error for a boolean found in the field list can effectively never fire
  # for a boolean in field position because booleans count as integers and are treated as field
  # numbers (then fail with a key error). Also the docstring says None is yielded when not found,
  # which only holds for map keys and list indices that are missing.
  # Generator. Inputs: proto, starting message; fields, a single entry or list of names/numbers,
  # with map keys and list indices placed after their map/repeated field. For each field step it
  # yields a tuple of the value reached (nothing if a map key or index was missing), the field
  # descriptor, the map key used (or nothing) and the list index used (or nothing).
  # Raises a value error if the path continues below a field that is not a message, and key
  # errors for names or numbers missing from the message type. Does not modify the proto.
  """Yields fields in a proto.

  Args:
    proto: Parent proto of any message type.
    fields: List of string/int/map key fields, e.g. ["nodes", "attr", "value"]
      can represent `proto.nodes.attr["value"]`.

  Yields:
    Tuple of (
      Field in the proto or `None` if none are found,
      Field descriptor,
      Key into this map field (or None),
      Index into this repeated field (or None))
  """
  if not isinstance(fields, list):
    fields = [fields]

  field_proto = proto
  parent_desc = proto.DESCRIPTOR
  i = 0
  while i < len(fields):
    field = fields[i]
    field_desc = None
    map_key = None
    index = None

    if parent_desc is None:
      raise ValueError(
          f"Unable to find fields: {fields} in proto of type {type(proto)}."
      )

    if isinstance(field, int):
      try:
        field_desc = parent_desc.fields_by_number[field]
      except KeyError:
        raise KeyError(  # pylint:disable=raise-missing-from
            f"Unable to find field number {field} in {parent_desc.full_name}. "
            f"Valid field numbers: {parent_desc.fields_by_number.keys()}"
        )
    elif isinstance(field, str):
      try:
        field_desc = parent_desc.fields_by_name[field]
      except KeyError:
        raise KeyError(  # pylint:disable=raise-missing-from
            f"Unable to find field '{field}' in {parent_desc.full_name}. "
            f"Valid field names: {parent_desc.fields_by_name.keys()}"
        )
    else:  # bool (only expected as map key)
      raise TypeError("Unexpected bool found in field list.")

    i += 1
    parent_desc = field_desc.message_type
    if field_proto is not None:
      field_proto = getattr(field_proto, field_desc.name)

    # Handle special fields types (map key and list index).
    if _is_map(parent_desc) and i < len(fields):
      # Next field is the map key.
      map_key = fields[i]

      try:
        field_proto = field_proto[map_key] if field_proto is not None else None
      except KeyError:
        field_proto = None
      i += 1

      if i < len(fields):
        # The next field must be from the Value Message.
        value_desc = parent_desc.fields_by_name["value"]
        assert value_desc.message_type is not None
        parent_desc = value_desc.message_type

    elif is_repeated(field_desc) and i < len(fields):
      # The next field is the index within the list.
      index = fields[i]
      try:
        field_proto = field_proto[index] if field_proto is not None else None
      except IndexError:
        field_proto = None
      i += 1

    yield field_proto, field_desc, map_key, index


def _is_map(desc: descriptor.Descriptor) -> bool:
  # [FUNCTION DOC]
  # HELPER: used by the path walker. Returns true if the message descriptor is a generated map
  # entry type; false for non-map descriptors and for a missing descriptor.
  return desc.GetOptions().map_entry if desc is not None else False


def is_repeated(field_desc: descriptor.FieldDescriptor) -> bool:
  # [FUNCTION DOC]
  # Returns true if the field descriptor describes a repeated field (including maps). The
  # commented-out lines are source-transformation markers for an internal build; the active
  # code compares the field label with the repeated label.
  # copybara:uncomment_begin(google-only)
  # return field_desc.is_repeated
  # copybara:uncomment_end_and_comment_begin
  return field_desc.label == descriptor.FieldDescriptor.LABEL_REPEATED
  # copybara:comment_end


_FIELD_DESC = descriptor.FieldDescriptor
_MAP_KEY = {
    _FIELD_DESC.TYPE_STRING: lambda key: chunk_pb2.FieldIndex.MapKey(s=key),
    _FIELD_DESC.TYPE_BOOL: lambda key: chunk_pb2.FieldIndex.MapKey(boolean=key),
    _FIELD_DESC.TYPE_UINT32: lambda key: chunk_pb2.FieldIndex.MapKey(ui32=key),
    _FIELD_DESC.TYPE_UINT64: lambda key: chunk_pb2.FieldIndex.MapKey(ui64=key),
    _FIELD_DESC.TYPE_INT32: lambda key: chunk_pb2.FieldIndex.MapKey(i32=key),
    _FIELD_DESC.TYPE_INT64: lambda key: chunk_pb2.FieldIndex.MapKey(i64=key),
}


def _map_key_proto(key_type, key):
  # [FUNCTION DOC]
  # HELPER: used by get_field_tag. Inputs: key_type, the protobuf field type of the map key;
  # key, the key value. Returns a map-key proto with the matching typed member set (string,
  # bool, 32/64-bit signed or unsigned integer). Raises a key error for unsupported key types.
  """Returns MapKey proto for a key of key_type."""
  return _MAP_KEY[key_type](key)
