# [MODULE DOC]
# HELPER: helper module for wheel build tooling that fills in a setup file template.
# Exposes one function that rewrites the text of a setup script by replacing placeholder lines
# (the CUDA major version, the wheel name suffix and per-package NVIDIA wheel version lines)
# with concrete values taken from a requirements-style text listing NVIDIA wheels. Also defines
# a compiled pattern that splits a requirement line into a package name and a version constraint.
# Pure text processing with no import-time side effects beyond compiling the pattern.
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Utility function for updating setup.py with NVIDIA wheel versions.

The content of the setup.py file is updated with the NVIDIA wheel versions
provided in the nvidia_wheel_versions_data string.

The setup.py file is expected to have the following lines:

```
# Mandatory placeholders
cuda_version = 0  # placeholder
cuda_wheel_suffix = ''  # placeholder

# Optional placeholders (add only those that are needed)
nvidia_cublas_version = ''  # placeholder

EXTRA_PACKAGES = {
    'and-cuda': [
        f'nvidia-cublas{cuda_wheel_suffix}{nvidia_cublas_version}',
        # add more wheels here
    ],
}
```
"""

import re

# Regex to capture wheel name and its version constraint
# Example: "nvidia-cublas-cu12>=12.1.3.1 ; sys_platform == 'linux'"
NVIDIA_WHEEL_VERSIONS_PATTERN = re.compile(r"^([a-z0-9_-]+)(\W*[0-9\.]*.*)$")


def get_setup_py_content_with_nvidia_wheel_versions(
    setup_py_content: str, cuda_version: str, nvidia_wheel_versions_data: str
) -> str:
  # [FUNCTION DOC]
  # HELPER: template filler for the wheel setup script generation.
  # KNOWN ISSUE: only CUDA versions 12 and 13 are supported; any other version string makes the
  # lookup fail with a key error. The wheel suffix is set to the 12 suffix only for version 12 and
  # to empty otherwise. Replacement uses exact text matching, so placeholders that differ in
  # whitespace are silently left unchanged.
  # Inputs: the setup script text, the CUDA major version as a string, and a multi-line text of
  # requirement entries. Each entry's name is normalized (dashes to underscores); names ending in
  # the 12 or 13 suffix are filed under that version, names without suffix under 13, and the
  # suffix is dropped and a version suffix appended to form the variable name, with the
  # version constraint (trimmed) as value. Then replaces the CUDA version placeholder, the wheel
  # suffix placeholder and each version placeholder for the selected CUDA version. Returns the
  # updated script text; inputs are not modified.
  nvidia_wheel_versions = {"12": {}, "13": {}}
  for line in nvidia_wheel_versions_data.splitlines():
    match = NVIDIA_WHEEL_VERSIONS_PATTERN.match(line)
    if match:
      wheel_name = match.group(1).replace("-", "_")
      for suffix, version in {"_cu12": "12", "_cu13": "13", "": "13"}.items():
        if not wheel_name.endswith(suffix):
          continue
        wheel_name = wheel_name.replace(suffix, "") + "_version"
        nvidia_wheel_versions[version][wheel_name] = match.group(2).strip()
        break

  setup_py_content = setup_py_content.replace(
      "cuda_version = 0  # placeholder", f"cuda_version = {cuda_version}"
  )
  setup_py_content = setup_py_content.replace(
      "cuda_wheel_suffix = ''  # placeholder",
      "cuda_wheel_suffix = '-cu12'" if cuda_version == "12" else "cuda_wheel_suffix = ''",
  )
  for version_name, version_value in nvidia_wheel_versions[
      str(cuda_version)
  ].items():
    setup_py_content = setup_py_content.replace(
        f"{version_name} = ''  # placeholder",
        f"{version_name} = '{version_value}'",
    )

  return setup_py_content
