#!/usr/bin/python
# [MODULE DOC]
# KNOWN ISSUE: the readme update uses a POSIX character class that Python's regular expression
# engine does not understand, so that substitution likely never matches as intended; the nightly
# option help text and the main description text are copy-pasted and unrelated to the behavior.
# Command line tool (run from the TensorFlow source root) that bumps the TensorFlow version in
# the source tree. Reads the current version from the bazel configuration file and the version
# bzl file, computes a new version from either an explicit version option or the nightly option
# (current version with a development suffix holding today's date), rewrites the version bzl,
# the bazel configuration (wheel type, build date, version suffix) and the readme, prints the
# major, minor and patch transition, and finally greps the source directory for lingering
# references to the old version strings. Defines module constants for file paths and for the two
# version kinds (snapshot and nightly), plus a small version value class. Helper tool for release
# and CI automation; no import-time side effects.
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
#
# Automatically update TensorFlow version in source files
#
# Usage:
#           ./tensorflow/tools/ci_build/update_version.py --version 1.4.0-rc1
#           ./tensorflow/tools/ci_build/update_version.py --nightly
#
"""Update version of TensorFlow script."""

# pylint: disable=superfluous-parens

import argparse
import os
import re
import subprocess
import time

# File parameters.
TF_SRC_DIR = "tensorflow"
README_MD = "./README.md"
TF_VERSION_BZL = "%s/tf_version.bzl" % TF_SRC_DIR
BAZEL_RC = "./.bazelrc"
RELEVANT_FILES = [
    TF_SRC_DIR,
    TF_VERSION_BZL,
    README_MD,
    BAZEL_RC,
]

# Version type parameters.
NIGHTLY_VERSION = 1
SNAPSHOT_VERSION = 0


def check_existence(filename):
  # [FUNCTION DOC]
  # HELPER: precondition check for the version update script.
  # Raises a runtime error telling the user to run from the TensorFlow source root if the given
  # file or directory path does not exist; otherwise returns nothing.
  """Check the existence of file or dir."""
  if not os.path.exists(filename):
    raise RuntimeError("%s not found. Are you under the TensorFlow source root"
                       " directory?" % filename)


def check_all_files():
  # [FUNCTION DOC]
  # HELPER: runs the existence check for every relevant path (source directory, version bzl,
  # readme and bazel configuration file). Raises a runtime error for the first missing one.
  # Takes no inputs, returns nothing.
  """Check all relevant files necessary for upgrade."""
  for file_name in RELEVANT_FILES:
    check_existence(file_name)


def replace_string_in_line(search, replace, filename):
  # [FUNCTION DOC]
  # HELPER: in-place regular expression replacement in a text file.
  # KNOWN ISSUE: the docstring mentions sed, but the code uses Python's regular expression
  # substitution; the search text is a Python pattern, not a sed one. Also the file is truncated
  # before the substitution result is written, so an error in between would lose its contents.
  # Inputs: search (pattern), replace (replacement text, may use group references) and filename.
  # Reads the whole file, then rewrites it with every match replaced. Returns nothing.
  """Replace with sed when regex is required."""
  with open(filename, "r") as source:
    content = source.read()
  with open(filename, "w") as source:
    source.write(re.sub(search, replace, content))


class Version(object):
  # [CLASS DOC]
  # Value object for a semantic version of TensorFlow. Holds major, minor and patch as text, an
  # identifier string (such as a release candidate or development suffix, including its leading
  # dash or empty), the version kind (snapshot or nightly) and a cached full string of the form
  # major.minor.patch plus identifier. Created through the constructor or the static parser from
  # text; the identifier can be replaced afterwards, which refreshes the cached string.
  """Version class object that stores SemVer version information."""

  def __init__(self, major, minor, patch, identifier_string, version_type):
    # [FUNCTION DOC]
    # Constructor. Stores the major, minor and patch texts, the identifier suffix text and the
    # version kind constant, then computes the cached full version string. No validation.
    """Constructor.

    Args:
      major: major string eg. (1)
      minor: minor string eg. (3)
      patch: patch string eg. (1)
      identifier_string: extension string eg. (-rc0)
      version_type: version parameter ((SNAPSHOT|NIGHTLY)_VERSION)
    """
    self.major = major
    self.minor = minor
    self.patch = patch
    self.identifier_string = identifier_string
    self.version_type = version_type
    self._update_string()

  def _update_string(self):
    # [FUNCTION DOC]
    # HELPER: recomputes the cached full version string as major.minor.patch followed by the
    # identifier string, storing it on the instance. Returns nothing.
    self.string = "%s.%s.%s%s" % (self.major,
                                  self.minor,
                                  self.patch,
                                  self.identifier_string)

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the cached full version string (major.minor.patch plus identifier).
    return self.string

  def set_identifier_string(self, identifier_string):
    # [FUNCTION DOC]
    # Replaces the identifier suffix text (for example a development date suffix) and refreshes
    # the cached full version string. Returns nothing.
    self.identifier_string = identifier_string
    self._update_string()

  @property
  def pep_440_str(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: for non-snapshot (nightly) versions the patch number is left out and the
    # identifier is used in its place (major.minor.identifier), which looks like a bug or at least
    # a surprising format.
    # Read-only property returning a package-index friendly version text with all dashes removed.
    # For snapshot versions the form is major.minor.patch plus identifier; for other kinds it is
    # major.minor plus identifier (no patch).
    if self.version_type == SNAPSHOT_VERSION:
      return_string = "%s.%s.%s%s" % (self.major,
                                      self.minor,
                                      self.patch,
                                      self.identifier_string)
      return return_string.replace("-", "")
    else:
      return_string = "%s.%s.%s" % (self.major,
                                    self.minor,
                                    self.identifier_string)
      return return_string.replace("-", "")

  @staticmethod
  def parse_from_string(string, version_type):
    # [FUNCTION DOC]
    # Static factory. Inputs: version text such as major.minor.patch optionally followed by a dash
    # and an identifier, and the version kind constant. Validates loosely (a search, not a full
    # match, for digits, digits, alphanumerics) and raises a runtime error naming the string if
    # invalid. Splits into major, minor and the rest, separates the patch from the identifier at
    # the first dash (the identifier keeps its leading dash, or is empty) and returns a new
    # version object. Note a string with fewer than two dots can still pass the check only if the
    # pattern appears; otherwise unpacking fails with a value error.
    """Returns version object from Semver string.

    Args:
      string: version string
      version_type: version parameter

    Raises:
      RuntimeError: If the version string is not valid.
    """
    # Check validity of new version string.
    if not re.search(r"[0-9]+\.[0-9]+\.[a-zA-Z0-9]+", string):
      raise RuntimeError("Invalid version string: %s" % string)

    major, minor, extension = string.split(".", 2)

    # Isolate patch and identifier string if identifier string exists.
    extension_split = extension.split("-", 1)
    patch = extension_split[0]
    if len(extension_split) == 2:
      identifier_string = "-" + extension_split[1]
    else:
      identifier_string = ""

    return Version(major, minor, patch, identifier_string, version_type)


def _get_regex_match(line, regex, is_last_match=False):
  # [FUNCTION DOC]
  # HELPER: single-line regular expression lookup used when reading the bazel and bzl files.
  # Inputs: a text line, a pattern with one capture group, and a flag saying whether a match
  # means the scan is finished. Returns a pair: the first captured group and the flag when the
  # pattern is found anywhere in the line, otherwise a null value and false.
  match = re.search(regex, line)
  if match:
    return (match.group(1), is_last_match)
  return (None, False)


def get_current_semver_version():
  # [FUNCTION DOC]
  # KNOWN ISSUE: both input files are opened and never closed; if the version line is not found
  # the version text stays empty or null and splitting it into three parts fails with an error;
  # when the suffix line is never found the suffix variable becomes null (no match resets it).
  # Reads the current version of the source tree. Scans the bazel configuration file for the wheel
  # type, wheel build date and wheel version suffix settings (stopping at the suffix line), then
  # scans the version bzl file for the quoted three-part version number. The kind is nightly when
  # the wheel type is nightly, otherwise snapshot. The identifier is a development marker with the
  # build date for nightly; for snapshot it is the optional development marker with the date plus
  # the suffix. Returns a version object; takes no inputs.
  """Returns a Version object of current version.

  Returns:
    version: Version object of current SemVer string based on information from
    .bazelrc and tf_version.bzl files.
  """

  # Get current version information.
  bazel_rc_file = open(BAZEL_RC, "r")

  wheel_type = ""
  wheel_build_date = ""
  wheel_version_suffix = ""

  for line in bazel_rc_file:
    wheel_type = (
        _get_regex_match(line, '^common --repo_env=ML_WHEEL_TYPE="(.+)"')[0]
        or wheel_type
    )
    wheel_build_date = (
        _get_regex_match(
            line, '^common --repo_env=ML_WHEEL_BUILD_DATE="([0-9]*)"'
        )[0]
        or wheel_build_date
    )
    (wheel_version_suffix, is_matched) = _get_regex_match(
        line,
        '^common --repo_env=ML_WHEEL_VERSION_SUFFIX="(.*)"',
        is_last_match=True,
    )
    if is_matched:
      break

  tf_version_bzl_file = open(TF_VERSION_BZL, "r")
  wheel_version = ""
  for line in tf_version_bzl_file:
    (wheel_version, is_matched) = _get_regex_match(
        line, '^TF_VERSION = "([0-9.]+)"', is_last_match=True
    )
    if is_matched:
      break
  (old_major, old_minor, old_patch_num) = wheel_version.split(".")

  if wheel_type == "nightly":
    version_type = NIGHTLY_VERSION
  else:
    version_type = SNAPSHOT_VERSION

  old_extension = ""
  if wheel_type == "nightly":
    old_extension = "-dev{}".format(wheel_build_date)
  else:
    if wheel_build_date:
      old_extension += "-dev{}".format(wheel_build_date)
    if wheel_version_suffix:
      old_extension += wheel_version_suffix

  return Version(old_major,
                 old_minor,
                 old_patch_num,
                 old_extension,
                 version_type)


def update_readme(old_version, new_version):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the search pattern uses a POSIX bracket class for alphanumerics, which Python's
  # regular expression engine treats as an ordinary character set (not alphanumerics), so the
  # readme is probably not updated as intended.
  # Rewrites version references in the readme file: replaces text of the form old major.old
  # minor.something followed by a dash with the new version's package-index style string followed
  # by a dash. Inputs: old and new version objects. Modifies the readme file; returns nothing.
  """Update README."""
  pep_440_str = new_version.pep_440_str
  replace_string_in_line(r"%s\.%s\.([[:alnum:]]+)-" % (old_version.major,
                                                       old_version.minor),
                         "%s-" % pep_440_str, README_MD)


def _get_mmp(version):
  # [FUNCTION DOC]
  # HELPER: returns the major.minor.patch text (no identifier) of a version object.
  return "%s.%s.%s" % (version.major, version.minor, version.patch)


def _get_wheel_type(version):
  # [FUNCTION DOC]
  # HELPER: maps a version object to the wheel type setting text: nightly for the nightly kind,
  # snapshot for anything else.
  if version.version_type == NIGHTLY_VERSION:
    return "nightly"
  else:
    return "snapshot"


def _get_wheel_build_date(version):
  # [FUNCTION DOC]
  # HELPER: extracts the eight digit build date that follows the development marker in the
  # identifier string of a version object. Returns the date text, or an empty text if absent.
  date_match = re.search(".*dev([0-9]{8}).*", version.identifier_string)
  if date_match:
    return date_match.group(1)
  return ""


def _get_wheel_version_suffix(version):
  # [FUNCTION DOC]
  # HELPER: returns the identifier string of a version object with the development marker and
  # its build date removed (what remains is the wheel version suffix, possibly empty).
  return version.identifier_string.replace(
      "-dev{}".format(_get_wheel_build_date(version)), ""
  )


def update_tf_version_bzl(old_version, new_version):
  # [FUNCTION DOC]
  # Rewrites the version bzl file, replacing the assignment of the old major.minor.patch with the
  # new one. Inputs: old and new version objects. Modifies the file in place; returns nothing.
  # The old version text is used as a regular expression (dots match any character).
  """Update tf_version.bzl."""
  old_mmp = _get_mmp(old_version)
  new_mmp = _get_mmp(new_version)
  replace_string_in_line(
      'TF_VERSION = "%s"' % old_mmp,
      'TF_VERSION = "%s"' % new_mmp,
      TF_VERSION_BZL,
  )


def update_bazelrc(old_version, new_version):
  # [FUNCTION DOC]
  # Rewrites three repository environment settings in the bazel configuration file: the wheel
  # type, the wheel build date and the wheel version suffix, each replaced from the old version's
  # value to the new version's value (derived from the version kinds and identifier strings).
  # Inputs: old and new version objects. Modifies the file in place (three passes); returns
  # nothing.
  """Update .bazelrc."""
  old_wheel_type = _get_wheel_type(old_version)
  new_wheel_type = _get_wheel_type(new_version)
  replace_string_in_line(
      'common --repo_env=ML_WHEEL_TYPE="%s"' % old_wheel_type,
      'common --repo_env=ML_WHEEL_TYPE="%s"' % new_wheel_type,
      BAZEL_RC,
  )

  old_wheel_build_date = _get_wheel_build_date(old_version)
  new_wheel_build_date = _get_wheel_build_date(new_version)
  replace_string_in_line(
      'common --repo_env=ML_WHEEL_BUILD_DATE="%s"' % old_wheel_build_date,
      'common --repo_env=ML_WHEEL_BUILD_DATE="%s"' % new_wheel_build_date,
      BAZEL_RC,
  )

  old_wheel_suffix = _get_wheel_version_suffix(old_version)
  new_wheel_suffix = _get_wheel_version_suffix(new_version)
  replace_string_in_line(
      'common --repo_env=ML_WHEEL_VERSION_SUFFIX="%s"' % old_wheel_suffix,
      'common --repo_env=ML_WHEEL_VERSION_SUFFIX="%s"' % new_wheel_suffix,
      BAZEL_RC,
  )


def major_minor_change(old_version, new_version):
  # [FUNCTION DOC]
  # Returns true if the major or the minor component differs between the two version objects,
  # false otherwise (patch and identifier are ignored). No side effects.
  """Check if a major or minor change occurred."""
  major_mismatch = old_version.major != new_version.major
  minor_mismatch = old_version.minor != new_version.minor
  if major_mismatch or minor_mismatch:
    return True
  return False


def check_for_lingering_string(lingering_string):
  # [FUNCTION DOC]
  # Searches the source directory recursively (with the external grep program) for the given old
  # version text, with dots escaped, and prints the result. If grep finds nothing (it exits with a
  # failure status) the list is empty. When matches exist, prints a warning followed by each
  # match line (file, line number and matched text); the output contains a trailing empty entry.
  # Otherwise prints a confirmation. Input: the text to search for. Returns nothing.
  """Check for given lingering strings."""
  formatted_string = lingering_string.replace(".", r"\.")
  try:
    linger_str_output = subprocess.check_output(
        ["grep", "-rnoH", formatted_string, TF_SRC_DIR])
    linger_strs = linger_str_output.decode("utf8").split("\n")
  except subprocess.CalledProcessError:
    linger_strs = []

  if linger_strs:
    print("WARNING: Below are potentially instances of lingering old version "
          "string \"%s\" in source directory \"%s/\" that are not "
          "updated by this script. Please check them manually!"
          % (lingering_string, TF_SRC_DIR))
    for linger_str in linger_strs:
      print(linger_str)
  else:
    print("No lingering old version strings \"%s\" found in source directory"
          " \"%s/\". Good." % (lingering_string, TF_SRC_DIR))


def check_for_old_version(old_version, new_version):
  # [FUNCTION DOC]
  # Looks for leftover references to the old version: checks the old full version string and its
  # package-index style string, and, when the major or minor part changed, also the old release
  # branch name (r, major, dot, minor). Inputs: old and new version objects. Prints results via
  # the lingering string check; returns nothing.
  """Check for old version references."""
  for old_ver in [old_version.string, old_version.pep_440_str]:
    check_for_lingering_string(old_ver)

  if major_minor_change(old_version, new_version):
    old_r_major_minor = "r%s.%s" % (old_version.major, old_version.minor)
    check_for_lingering_string(old_r_major_minor)


def main():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the argument parser description (cherry picking automation) and the help text
  # of the nightly option (disable the service provisioning step) are copy-pasted and do not
  # describe this script; without either option the empty version text fails validation; the
  # docstring says one of version or nightly is required but nothing enforces exclusivity.
  # Script entry point. Parses the optional version text and nightly flag, verifies the relevant
  # files exist, reads the current version, and builds the new one: with the nightly flag it is
  # the explicit version (if given) or the current major, minor and patch, with a development
  # suffix of today's date and the nightly kind; otherwise it is the parsed explicit version as a
  # snapshot. Then updates the version bzl, bazel configuration and readme files, prints the
  # major, minor and patch transitions, and checks for lingering old version strings.
  # Raises a runtime error for invalid or missing files or version text. Returns nothing.
  """This script updates all instances of version in the tensorflow directory.

  Requirements:
    version: The version tag
    OR
    nightly: Create a nightly tag with current date

  Raises:
    RuntimeError: If the script is not being run from tf source dir
  """

  parser = argparse.ArgumentParser(description="Cherry picking automation.")

  # Arg information
  parser.add_argument("--version",
                      help="<new_major_ver>.<new_minor_ver>.<new_patch_ver>",
                      default="")
  parser.add_argument("--nightly",
                      help="disable the service provisioning step",
                      action="store_true")

  args = parser.parse_args()

  check_all_files()
  old_version = get_current_semver_version()

  if args.nightly:
    if args.version:
      new_version = Version.parse_from_string(args.version, NIGHTLY_VERSION)
      new_version.set_identifier_string("-dev" + time.strftime("%Y%m%d"))
    else:
      new_version = Version(old_version.major,
                            str(old_version.minor),
                            old_version.patch,
                            "-dev" + time.strftime("%Y%m%d"),
                            NIGHTLY_VERSION)
  else:
    new_version = Version.parse_from_string(args.version, SNAPSHOT_VERSION)

  update_tf_version_bzl(old_version, new_version)
  update_bazelrc(old_version, new_version)
  update_readme(old_version, new_version)

  # Print transition details.
  print("Major: %s -> %s" % (old_version.major, new_version.major))
  print("Minor: %s -> %s" % (old_version.minor, new_version.minor))
  print("Patch: %s -> %s\n" % (old_version.patch, new_version.patch))

  check_for_old_version(old_version, new_version)


if __name__ == "__main__":
  main()
