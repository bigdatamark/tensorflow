# [MODULE DOC]
# Build-time test script (not a unit-test class) that checks a built Python wheel conforms to the
# manylinux platform tag expected for the current CPU architecture. It runs the wheel auditing
# tool's "show" command in-process, capturing its standard output, and searches the report for
# the required platform tag (accepting the alternative name used for the 2014 tag). Takes the
# wheel path and the compliance tags for aarch64, x86_64 and ppc64le on the command line (an
# argument file is allowed). Raises a runtime error for an unsupported machine or a
# non-compliant wheel. Import-time side effect: imports the auditwheel package. Runs the check
# only when executed as a script.
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

import argparse
import io
import pathlib
import platform
import re
import sys
from auditwheel import main_show


def parse_args():
  # [FUNCTION DOC]
  # HELPER: command-line parser for the compliance test.
  # Takes no inputs; reads the process arguments. Required options: the wheel path and one
  # compliance tag each for aarch64, x86_64 and ppc64le. Arguments may be supplied via an
  # at-prefixed argument file. Returns the parsed namespace; missing options make the parser exit.
  """Arguments parser."""
  parser = argparse.ArgumentParser(
      description="Helper for manylinux compliance verification",
      fromfile_prefix_chars="@",
  )
  parser.add_argument(
      "--wheel-path", required=True, help="Path of the wheel, mandatory"
  )
  parser.add_argument(
      "--aarch64-compliance-tag",
      required=True,
      help="ManyLinux compliance tag for aarch64",
  )
  parser.add_argument(
      "--x86_64-compliance-tag",
      required=True,
      help="ManyLinux compliance tag for x86_64",
  )
  parser.add_argument(
      "--ppc64le-compliance-tag",
      required=True,
      help="ManyLinux compliance tag for ppc64le",
  )
  return parser.parse_args()


def get_auditwheel_output(wheel_path: str) -> None:
  # [FUNCTION DOC]
  # HELPER: runs the wheel audit tool for the compliance test and captures its report.
  # KNOWN ISSUE: the return annotation says it returns nothing, but it actually returns the
  # captured report text as a string. Also, standard output is not restored if the audit
  # command raises, since no cleanup guard is used.
  # Input: path of the wheel file. Temporarily redirects standard output to an in-memory buffer,
  # builds the tool's parser and invokes its show command with the ISA extension check disabled
  # and verbose level one, then restores standard output. Returns the captured text. Exceptions
  # from the audit tool propagate.
  """Run "auditwheel show" on the wheel and return the output.

  Args:
    wheel_path: path of the wheel file

  Returns:
    "auditwheel show" output
  """
  stringio = io.StringIO()
  previous_stdout = sys.stdout
  sys.stdout = stringio

  auditwheel_parser = argparse.ArgumentParser(
      description="Cross-distro Python wheels."
  )
  sub_parsers = auditwheel_parser.add_subparsers(metavar="command", dest="cmd")
  main_show.configure_parser(sub_parsers)
  auditwheel_args = argparse.Namespace(
      WHEEL_FILE=pathlib.Path(wheel_path),
      DISABLE_ISA_EXT_CHECK=True,
      verbose=1,
  )

  main_show.execute(auditwheel_args, auditwheel_parser)

  sys.stdout = previous_stdout
  return stringio.getvalue()


def verify_manylinux_compliance(
    auditwheel_log: str,
    compliance_tag: str,
) -> None:
  # [FUNCTION DOC]
  # HELPER: assertion step of the compliance test.
  # KNOWN ISSUE: the pattern is written as a plain string with a backslash-s escape, which is an
  # invalid escape sequence warning in recent Python versions; the tag is also inserted into the
  # pattern without escaping.
  # Inputs: the audit report text and the expected manylinux tag. Searches the report for the
  # phrase naming the supported platform tag equal to the expected tag, also trying a variant in
  # which the 2014 tag is replaced by its equivalent glibc-versioned name. Returns nothing.
  # Raises a runtime error containing the tag and the full report if neither pattern matches.
  """Verify manylinux compliance.

  Args:
    auditwheel_log: "auditwheel show" execution results
    compliance_tag: manyLinux compliance tag

  Raises:
    RuntimeError: if the wheel is not manyLinux compliant.
  """
  regex = 'following platform tag:\s+"{}"'.format(compliance_tag)
  alt_regex = regex.replace("2014", "_2_17")
  if not (
      re.search(regex, auditwheel_log) or re.search(alt_regex, auditwheel_log)
  ):
    raise RuntimeError(
        ("The wheel is not compliant with the tag {tag}.\n{result}").format(
            tag=compliance_tag, result=auditwheel_log
        )
    )


def test_manylinux_compliance(args):
  # [FUNCTION DOC]
  # Test entry point (a plain function, run from the script main block).
  # Input: parsed arguments holding the wheel path and the three per-architecture compliance
  # tags. Determines the machine type of the host; raises a runtime error if it is not x86_64,
  # aarch64 or ppc64le. Picks the tag for that architecture, produces the audit report for the
  # wheel and verifies compliance, raising a runtime error on failure. Returns nothing.
  machine_type = platform.uname().machine
  supported_machine_types = ["x86_64", "aarch64", "ppc64le"]
  if machine_type not in supported_machine_types:
    raise RuntimeError(
        "Unsupported machine type {machine_type}. The supported are:"
        " {supported_types}".format(
            machine_type=machine_type, supported_types=supported_machine_types
        )
    )
  if machine_type == "x86_64":
    compliance_tag = args.x86_64_compliance_tag
  elif machine_type == "aarch64":
    compliance_tag = args.aarch64_compliance_tag
  else:
    compliance_tag = args.ppc64le_compliance_tag
  auditwheel_output = get_auditwheel_output(args.wheel_path)
  verify_manylinux_compliance(
      auditwheel_output,
      compliance_tag,
  )


if __name__ == "__main__":
  test_manylinux_compliance(parse_args())
