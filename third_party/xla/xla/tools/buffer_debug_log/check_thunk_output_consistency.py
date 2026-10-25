# [MODULE DOC]
# Command-line tool that reads one or more GPU buffer debug log protocol buffers (binary or text
# format) produced with checksum tracing on thunks, optionally a thunk metadata file, and prints
# whether identical thunk executions (same input checksums) produced different output checksums.
# Uses the checksum_mismatch_report module for the analysis and only does input parsing and
# printing itself. Defines one command-line flag (the metadata file path) at import time and runs
# the application main function when executed as a script.
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
"""A tool to analyze buffer debug logs.

To generate the log files, run the HLO with
--xla_gpu_experimental_enable_checksum_tracing_on_thunks.
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from google.protobuf import message
from google.protobuf import text_format

from xla.backends.gpu.runtime import buffer_debug_log_pb2
from xla.backends.gpu.runtime import thunk_pb2
from xla.tools.buffer_debug_log import checksum_mismatch_report


def parse_binary_or_text_proto(data: bytes, proto_type):
  # [FUNCTION DOC]
  # HELPER: Input-reading utility for the tool's main function.
  # Parses raw bytes into a message of the given protocol buffer type. First tries the binary wire
  # format; if decoding fails with a decode error, falls back to parsing the bytes as text format.
  # Returns the parsed message. A text parse failure propagates as the text parser's error; only
  # a decode error (not other errors) triggers the text fallback.
  """Parses a binary or text proto."""
  try:
    return proto_type.FromString(data)
  except message.DecodeError:
    pass
  return text_format.Parse(data, proto_type())


_METADATA_FILE = flags.DEFINE_string(
    "metadata-file", None, "Path to the thunk metadata proto file."
)


def _print_formatted_report(
    report: checksum_mismatch_report.ChecksumMismatchReport,
):
  # [FUNCTION DOC]
  # HELPER: Output formatter for the tool's main function.
  # Prints the given report to standard output. If there are no mismatches prints a single
  # success line. Otherwise prints a header and, for each thunk with mismatches, a description line
  # (thunk id plus kind and profile annotation when metadata is available, or a metadata-missing
  # note) followed, for each distinct input checksum set in sorted order, by one line per buffer
  # index listing the differing output checksums. Returns nothing.
  """Prints a ChecksumMismatchReport to stdout in a human-readable format."""

  if not report.mismatches:
    print("\N{WHITE HEAVY CHECK MARK} All results are perfectly consistent.")
    return

  print(
      "\N{OCTAGONAL SIGN} Different outputs detected among identical"
      " thunk executions:"
  )
  for thunk_id, mismatches_by_inputs in report.mismatches.items():
    if not mismatches_by_inputs:
      continue

    def describe_thunk(thunk_id: checksum_mismatch_report.ThunkId):
      # [FUNCTION DOC]
      # HELPER: Label builder for the report printer.
      # Returns a text line beginning with the thunk id, followed by either the thunk kind and
      # profile annotation (when the report has metadata for this thunk) or a note that metadata is
      # missing. Reads the enclosing report; no side effects.
      result = f"In outputs of thunk {thunk_id}"
      metadata = " (metadata missing)"
      if report.thunk_metadata:
        thunk_metadata = report.thunk_metadata.get(thunk_id)
        if thunk_metadata:
          metadata = f" (kind: {thunk_metadata.thunk_kind}, profile_annotation:"
          metadata += f" {thunk_metadata.profile_annotation})"
      return result + metadata

    print(describe_thunk(thunk_id))
    for _, mismatches_by_buffer_idx in sorted(mismatches_by_inputs.items()):
      for buffer_idx, checksums in mismatches_by_buffer_idx.items():
        print(f"  buffer {buffer_idx}: checksums={checksums}")


def main(argv: Sequence[str]) -> None:
  # [FUNCTION DOC]
  # KNOWN ISSUE: The usage message names a program called buffer-debug.py, not this file.
  # Application entry point. Input: argv, the program name followed by one or more paths of buffer
  # debug log files. Raises a usage error when no log path is given. Each file is read and parsed
  # (binary or text) and keyed by its position as the module execution id; any read or parse error
  # gets a note naming the file and is re-raised. If the metadata flag is set, that file is
  # parsed likewise; otherwise an empty metadata message is used. Builds the mismatch report and
  # prints it. Returns nothing.
  if len(argv) < 2:
    raise app.UsageError(
        "Usage: buffer-debug.py [--metadata-file METADATA_PROTO_PATH]"
        " LOG_PROTO_PATHS..."
    )

  log_protos = {}
  for module_id, arg in enumerate(argv[1:]):
    try:
      with open(arg, "rb") as f:
        log_protos[module_id] = parse_binary_or_text_proto(
            f.read(), buffer_debug_log_pb2.BufferDebugLogProto
        )
    except Exception as e:
      e.add_note(f"when reading {arg}")
      raise

  if _METADATA_FILE.value:
    try:
      with open(_METADATA_FILE.value, "rb") as f:
        metadata_proto = parse_binary_or_text_proto(
            f.read(), thunk_pb2.ThunkMetadataListProto
        )
    except Exception as e:
      e.add_note(f"when reading {_METADATA_FILE.value}")
      raise
  else:
    metadata_proto = thunk_pb2.ThunkMetadataListProto()

  report = checksum_mismatch_report.ChecksumMismatchReport.from_protos(
      log_protos, metadata_proto
  )
  _print_formatted_report(report)


if __name__ == "__main__":
  app.run(main)
