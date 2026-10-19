# [MODULE DOC]
# CI utility script that scans a Bazel build log for "Streaming build results to" ResultStore
# links, works out for each Bazel invocation whether it passed, failed to build, or had failing
# tests, tries to recover the Bazel command that produced it, and then prints a summary and/or
# writes a JUnit-style XML file with one test case per invocation. Run as a command-line script
# taking a log path, an optional XML output path, a print flag and a verbose flag.
# Defines status constants, regular expressions and a result mapping type at import time; no
# other import-time side effects.
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

"""Extracts ResultStore links from a log containing Bazel invocations.

The links and the invocations' status can then be printed out, or output in the
form of JUnit-based XML.
"""
import argparse
import datetime
import os
import re
from typing import Dict, Union
import xml.etree.ElementTree as ElemTree


ResultDictType = Dict[str, Dict[str, Union[str, int]]]

RESULT_STORE_LINK_RE = re.compile(
    r'^INFO: Streaming build results to: (https://[\w./\-]+)')
FAILED_BUILD_LINE = 'FAILED: Build did NOT complete successfully'
BUILD_STATUS_LINE = 'INFO: Build'
TESTS_FAILED_RE = re.compile(r'^INFO: Build completed, \d+ tests? FAILED')
BAZEL_COMMAND_RE = re.compile(
    r'(^| )(?P<command>bazel (.*? )?(?P<type>test|build) .+)')


class InvokeStatus:
  # [CLASS DOC]
  # HELPER: namespace of string constants naming the outcome of a Bazel invocation: tests failed,
  # build failed, or passed. Never instantiated; the values are stored in the result mapping.
  tests_failed = 'tests_failed'
  build_failed = 'build_failed'
  passed = 'passed'


def parse_args() -> argparse.Namespace:
  # [FUNCTION DOC]
  # Parses command-line arguments: a required build log path, optional XML output path, a print
  # flag and a verbose flag. Returns the parsed namespace. Raises a type error (rather than the
  # usual usage error) when neither the print flag nor the XML output path is given.
  """Parses the commandline args."""
  parser = argparse.ArgumentParser(
      description='Extracts ResultStore links from a build log.\n'
                  'These can be then printed out, and/or output into a '
                  'JUnit-based XML file inside a specified directory.')

  parser.add_argument('build_log',
                      help='Path to a build log.')
  parser.add_argument('--xml-out-path',
                      required=False,
                      help='Path to which to output '
                           'the JUnit-based XML with ResultStore links.')
  parser.add_argument('--print',
                      action='store_true', dest='print', default=False,
                      help='Whether to print out a short summary with the '
                           'found ResultStore links (if any).')
  parser.add_argument('-v', '--verbose',
                      action='store_true', dest='verbose', default=False,
                      help='Prints out lines helpful for debugging.')
  parsed_args = parser.parse_args()
  if not parsed_args.print and not parsed_args.xml_out_path:
    raise TypeError('`--print` or `--xml-out-path` must be specified')

  return parsed_args


def parse_log(file_path: str,
              verbose: bool = False) -> ResultDictType:
  # [FUNCTION DOC]
  # Scans a log file for ResultStore links and classifies each invocation.
  # Inputs: file_path, path to the log (read as UTF-8, undecodable bytes ignored); verbose,
  # print details of failed invocations. Returns a mapping from each link to a record holding the
  # start line, optional end line, optional next-link line, status, and, when found, a log
  # fragment of the failure plus the bazel command text and its type (test or build).
  # Pass 1 records, for every link, the lines where it first and last appears (an invocation prints
  # its link at start and end). Pass 2 walks backwards from each invocation's last line looking
  # for a build-failed line or a "tests failed" build summary; on a hit it saves the preceding
  # twenty lines as a fragment and sets the status accordingly, otherwise the status stays
  # passed. It then walks further backwards for a one-line bazel build/test command.
  # KNOWN ISSUE: the command search continues from the failure line (or the start line when
  # nothing failed) instead of from the invocation start, so for failed invocations it can miss
  # or mis-assign the command; a single occurrence of a link has no end line.
  """Finds ResultStore links, and tries to determine their status."""
  with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    log_lines = f.read().splitlines()

  result_store_links: ResultDictType = {}
  current_url = None
  for i in range(len(log_lines)):
    line = log_lines[i]
    result_store_line_match = re.search(RESULT_STORE_LINK_RE, line)
    if not result_store_line_match:
      continue

    url = result_store_line_match.group(1)
    url_lines = result_store_links.setdefault(url, {})
    # Each bazel RBE invocation should produce two
    # 'Streaming build results to: ...' lines, one at the start, and one at the
    # end of the invocation.
    # If there's a failure message, it will be found in-between these two.

    if not current_url:
      url_lines['start'] = i
    elif current_url == url:
      url_lines['end'] = i
    else:
      result_store_links[current_url]['next_url'] = i
      url_lines['start'] = i
    current_url = url

  previous_end_line = None
  for url, lines in result_store_links.items():
    lines['status'] = InvokeStatus.passed  # default to passed
    start_line = lines['start']
    end_line = lines.get('end', lines.get('next_url', len(log_lines))) - 1
    k = end_line
    while k > start_line:
      backtrack_line = log_lines[k]
      build_failed = backtrack_line.startswith(FAILED_BUILD_LINE)
      if build_failed or not backtrack_line.startswith(BUILD_STATUS_LINE):
        tests_failed = False
      else:
        tests_failed = re.search(TESTS_FAILED_RE, backtrack_line)
      if build_failed or tests_failed:
        log_fragment = '\n'.join(
            log_lines[max(k - 20, 0):min(end_line + 1, len(log_lines) - 1)])
        lines['log_fragment'] = log_fragment
        lines['status'] = (InvokeStatus.build_failed if build_failed
                           else InvokeStatus.tests_failed)
        if verbose:
          print(f'Found failed invocation: {url.rsplit("/")[-1]}\n'
                f'Log fragment:\n'
                f'```\n{log_fragment}\n```\n'
                f'{"=" * 140}')
        break
      k -= 1

    # A low-effort attempt to find the bazel command that triggered the
    # invocation.
    bazel_comm_min_line_i = (previous_end_line if previous_end_line is not None
                             else 0)
    while k > bazel_comm_min_line_i:
      backtrack_line = log_lines[k]
      # Don't attempt to parse multi-line commands broken up by backslashes
      if 'bazel ' in backtrack_line and not backtrack_line.endswith('\\'):
        bazel_line = BAZEL_COMMAND_RE.search(backtrack_line)
        if bazel_line:
          lines['command'] = bazel_line.group('command')
          lines['command_type'] = bazel_line.group('type')
          break
      k -= 1
      continue
    previous_end_line = lines.get('end') or start_line

  return result_store_links


def indent_xml(elem, level=0) -> None:
  # [FUNCTION DOC]
  # HELPER: pretty-printing helper for the XML writer.
  # Recursively fills in whitespace text and tail values of an XML element tree in place so the
  # output is indented two spaces per level. Inputs: elem, an XML element; level, current depth
  # (default 0). Returns nothing.
  # KNOWN ISSUE: the loop over children reuses the name of the element parameter, so after the
  # loop the tail fix-up applies to the last child rather than to the element itself.
  """Indents and newlines the XML for better output."""
  indent_str = '\n' + level * '  '
  if len(elem):  # pylint: disable=g-explicit-length-test  # `if elem` not valid
    if not elem.text or not elem.text.strip():
      elem.text = indent_str + '  '
    if not elem.tail or not elem.tail.strip():
      elem.tail = indent_str
    for elem in elem:
      indent_xml(elem, level + 1)
    if not elem.tail or not elem.tail.strip():
      elem.tail = indent_str
  else:
    if level and (not elem.tail or not elem.tail.strip()):
      elem.tail = indent_str


def create_xml_file(result_store_dict: ResultDictType,
                    output_path: str,
                    verbose: bool = False):
  # [FUNCTION DOC]
  # Writes a JUnit-style XML report of Bazel invocations.
  # Inputs: result_store_dict, the mapping produced by the log parser; output_path, file to
  # write (its parent directory is created if needed); verbose, print progress. One test case per
  # invocation is created, named by command type and invocation id (last part of the link). A
  # tests-failed invocation gets a failure element, a build-failed one an error element, each with
  # text containing the command (or a note that it was not found), the link and the log fragment.
  # Passing ones get properties with a success description and the command. Suite attributes
  # carry the total, error and failure counts and a UTC timestamp; the tree is indented and
  # written as a binary file with an XML declaration. Returns nothing.
  # KNOWN ISSUE: the suite attributes and indentation are recomputed inside the loop, and with an
  # empty mapping they are never set; the parent-directory creation fails if the output path has
  # no directory part; the UTC timestamp call is deprecated in newer Python versions; the
  # no-command case name has a leading space.
  """Creates a JUnit-based XML file, with each invocation as a testcase."""
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  failure_count = 0
  error_count = 0

  date_time = datetime.datetime
  attrib = {'name': 'Bazel Invocations', 'time': '0.0',
            'timestamp': date_time.isoformat(date_time.utcnow())}
  testsuites = ElemTree.Element('testsuites')
  testsuite = ElemTree.SubElement(testsuites, 'testsuite')
  for url, invocation_results in result_store_dict.items():
    invocation_id = url.rsplit('/')[-1]
    if verbose:
      print(f'Creating testcase for invocation {invocation_id}')
    status = invocation_results['status']
    command = invocation_results.get('command')
    command_type = invocation_results.get('command_type')

    case_attrib = attrib.copy()
    if command_type:
      command_type = command_type.title()
      case_name = f'{command_type} invocation {invocation_id}'
    else:
      case_name = f' Invocation {invocation_id}'
    case_attrib.update({'name': case_name,
                        'status': 'run', 'result': 'completed'})

    testcase = ElemTree.SubElement(testsuite, 'testcase', attrib=case_attrib)
    if status in (InvokeStatus.tests_failed, InvokeStatus.build_failed):
      if status == InvokeStatus.tests_failed:
        failure_count += 1
        elem_name = 'failure'
      else:
        error_count += 1
        elem_name = 'error'
      if command:
        failure_msg = (f'\nThe command was:\n\n'
                       f'{command}\n\n')
      else:
        failure_msg = ('\nCouldn\'t parse a bazel command '
                       'matching the invocation, inside the log. '
                       'Please look for it in the build log.\n\n')
      failure_msg += (
          f'See the ResultStore link for a detailed view of failed targets:\n'
          f'{url}\n\n')
      failure_msg += (
          f'Here\'s a fragment of the log containing the failure:\n\n'
          f'[ ... TRUNCATED ... ]\n\n'
          f'{invocation_results["log_fragment"]}\n'
          f'\n[ ... TRUNCATED ... ]\n'
      )
      failure = ElemTree.SubElement(
          testcase, elem_name,
          message=f'Bazel invocation {invocation_id} failed.')
      failure.text = failure_msg
    else:
      properties = ElemTree.SubElement(testcase, 'properties')
      success_msg = 'Build completed successfully.\n' f'See {url} for details.'
      ElemTree.SubElement(properties, 'property',
                          name='description',
                          value=success_msg)
      if command:
        ElemTree.SubElement(properties, 'property',
                            name='bazel_command',
                            value=command)

    suite_specific = {'tests': str(len(result_store_dict)),
                      'errors': str(error_count),
                      'failures': str(failure_count)}
    suite_attrib = attrib.copy()
    suite_attrib.update(suite_specific)
    testsuites.attrib = suite_attrib
    testsuite.attrib = suite_attrib
    indent_xml(testsuites)

  tree = ElemTree.ElementTree(testsuites)
  file_path = os.path.join(output_path)
  with open(file_path, 'wb') as f:
    f.write(b'<?xml version="1.0"?>\n')
    tree.write(f)
    if verbose:
      print(f'\nWrote XML with Bazel invocation results to {file_path}')


def print_invocation_results(result_store_dict: ResultDictType):
  # [FUNCTION DOC]
  # Prints a human-readable summary. Input: result_store_dict, the parsed result mapping.
  # Prints a header (either a no-links message, or the count plus an explanation of ResultStore),
  # then for each invocation a numbered entry with its status, the bazel command (or a note that
  # it could not be parsed) and the link. Returns nothing.
  """Prints out a short summary of the found ResultStore links (if any)."""
  print()
  if not result_store_dict:
    print('Found no ResultStore links for Bazel build/test invocations.')
  else:
    print(f'Found {len(result_store_dict)} ResultStore link(s) for '
          f'Bazel invocations.\n'
          f'ResultStore contains individual representations of each target '
          f'that were run/built during the invocation.\n'
          f'These results are generally easier to read than looking through '
          f'the entire build log:\n')
  i = 1
  for url, invocation_results in result_store_dict.items():
    line_str = f'Invocation #{i} ({invocation_results["status"]}):\n'
    command = invocation_results.get('command')
    if command:
      line_str += command
    else:
      line_str += ('Couldn\'t parse the bazel command, '
                   'check inside the build log instead')
    line_str += f'\n{url}\n'
    print(line_str)
    i += 1


def main():
  # [FUNCTION DOC]
  # Script entry point. Parses arguments, expands environment variables in the log path, parses
  # the log, then writes the XML report if an output path was given (also expanding environment
  # variables) and prints the summary if requested. Returns nothing.
  args = parse_args()
  verbose = args.verbose
  build_log_path = os.path.expandvars(args.build_log)
  links = parse_log(build_log_path, verbose=verbose)

  if args.xml_out_path:
    output_path = os.path.expandvars(args.xml_out_path)
    create_xml_file(links, output_path, verbose=verbose)
  if args.print:
    print_invocation_results(links)


if __name__ == '__main__':
  main()
