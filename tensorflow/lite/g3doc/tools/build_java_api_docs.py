# [MODULE DOC]
# HELPER: Documentation build tool (command line script) for the TensorFlow Lite Java reference docs.
# Command line script that generates the TensorFlow Lite Java API reference docs for tensorflow.org.
# It defines command line flags (output directory, site path, and two unused flags required by
# the infrastructure), locates the sibling source trees (core, GPU, support library, metadata,
# ODML image library, Android SDK API listing) by walking up from this file's location, merges the
# Java sources into one temporary directory and calls the external docs generator for Java.
# Import time side effects: registers absl command line flags in the global flag registry.
# Running it as a program starts the absl app with the main function.
# Copyright 2024 The AI Edge Authors. All Rights Reserved.
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
"""Generate TensorFlow Lite Java reference docs for TensorFlow.org."""
import os
import pathlib
import shutil
import tempfile
from typing import Iterable, Union

from absl import app
from absl import flags

from tensorflow_docs.api_generator import gen_java

FLAGS = flags.FLAGS

# These flags are required by infrastructure, not all of them are used.
_OUT_DIR = flags.DEFINE_string('output_dir', '/tmp/lite_api/',
                               "Use this branch as the root version and don't "
                               'create in version directory')

_SITE_PATH = flags.DEFINE_string('site_path', 'lite/api_docs/java',
                                 'Path prefix in the _toc.yaml')

flags.DEFINE_string('code_url_prefix', None,
                    '[UNUSED] The url prefix for links to code.')

flags.DEFINE_bool(
    'search_hints', True,
    '[UNUSED] Include metadata search hints in the generated files')

# This tool expects to live within a directory structure that hosts multiple
# repositories, like so:
# /root (name not important)
#   /tensorflow (github.com/tensorflow/tensorflow - home of this script)
#   /tensorflow_lite_support (github.com/tensorflow/tflite-support)
#   /android/sdk (android.googlesource.com/platform/prebuilts/sdk
#     - Note that this needs a branch with an api/ dir, such as *-release)
# Internally, both the monorepo and the external build system do this for you.
SOURCE_PATH_CORE = pathlib.Path('tensorflow/lite/java/src/main/java')
SOURCE_PATH_GPU = pathlib.Path(
    'tensorflow/lite/delegates/gpu/java/src/main/java')
SOURCE_PATH_SUPPORT = pathlib.Path('tensorflow_lite_support/java/src/java')
SOURCE_PATH_METADATA = pathlib.Path(
    'tensorflow_lite_support/metadata/java/src/java')
SOURCE_PATH_ODML = pathlib.Path('tensorflow_lite_support/odml/java/image/src')
SOURCE_PATH_ANDROID_SDK = pathlib.Path('android/sdk/api/26.txt')

# This (key) ordering is preserved in the TOC output.
SECTION_LABELS = {
    'org.tensorflow.lite': 'Core',
    'org.tensorflow.lite.support': 'Support Library',
    'org.tensorflow.lite.gpu': 'Delegates',
    # If we ever need other ODML packages, drop the `.image` here.
    'com.google.android.odml.image': 'ODML',
}

EXTERNAL_APIS = {'https://developer.android.com': SOURCE_PATH_ANDROID_SDK}


def overlay(from_root: Union[str, os.PathLike[str]],
            to_root: Union[str, os.PathLike[str]]) -> None:
  # [FUNCTION DOC]
  # HELPER: Merges several Java source trees into one directory for build_java_api_docs main.
  # Recursively copies everything under the source directory into the destination directory,
  # allowing the destination to already exist (existing files are overwritten).
  # Inputs: from_root and to_root, both paths given as text or path objects.
  # Returns nothing. Raises the usual file system errors if the source is missing.
  """Recursively copy from_root/* into to_root/."""
  shutil.copytree(from_root, to_root, dirs_exist_ok=True)


def resolve_nested_dir(path: pathlib.Path, root: pathlib.Path) -> pathlib.Path:
  # [FUNCTION DOC]
  # HELPER: Locates source directories across differing GitHub and internal directory layouts.
  # Inputs: path, a relative path; root, the directory to search under.
  # Tries the path directly under root, then a variant where the path's first component is
  # repeated as an extra directory level (for example foo/bar becomes foo/foo/bar).
  # Returns the first variant that exists. Raises a value error if neither exists.
  """Returns the path that exists, out of foo/... and foo/foo/..., with root."""
  nested = path.parts[0] / path
  if (root_path := root / path).exists():
    return root_path
  elif (root_nested_path := root / nested).exists():
    return root_nested_path
  raise ValueError(f'Could not find {path} or {nested}')


def exists_maybe_nested(paths: Iterable[pathlib.Path],
                        root: pathlib.Path) -> bool:
  # [FUNCTION DOC]
  # HELPER: Checks that all required source directories can be found under a candidate root.
  # Inputs: paths, an iterable of relative paths; root, the directory to search under.
  # Returns true only if every path resolves, either as given or with its first component
  # repeated; returns false as soon as one cannot be resolved. No side effects.
  """Evaluates whether all paths exist, either as-is, or nested."""
  # Due to differing directory structures between GitHub & Google, we need to
  # check if a path exists as-is, or with the first section repeated.
  for path in paths:
    try:
      resolve_nested_dir(path, root)
    except ValueError:
      return False
  return True


def main(unused_argv):
  # [FUNCTION DOC]
  # Entry point of the Java docs build. The command line arguments are ignored.
  # Starting at this file's resolved path, walks up parent directories until one hosts the core,
  # support library and ODML source trees (stops at the file system root).
  # Raises a file-not-found error if no such root exists.
  # Then, inside a temporary directory, overlays the core, GPU, support, metadata and ODML
  # sources into one tree, and, if the GMS_PATH environment variable is set, overlays that
  # Play Services tree under the task gms package directory.
  # Finally calls the Java docs generator for the TensorFlow Lite and ODML packages, writing to
  # the output directory flag, using the site path flag, the section labels, and the Android SDK
  # API listing as federated external docs. Returns nothing; writes generated docs to disk.
  # Note: the GPU and metadata trees are overlaid but not part of the existence check.
  root = pathlib.Path(__file__).resolve()
  all_deps = [SOURCE_PATH_CORE, SOURCE_PATH_SUPPORT, SOURCE_PATH_ODML]
  # Keep searching upwards for a root that hosts the various dependencies. We
  # test `root.name` to ensure we haven't hit /.
  while not (exists := exists_maybe_nested(all_deps, root)) and root.name:
    root = root.parent
  if not exists:
    raise FileNotFoundError('Could not find dependencies.')

  with tempfile.TemporaryDirectory() as merge_tmp_dir:
    # Merge the combined API sources into a single location.
    merged_temp_dir = pathlib.Path(merge_tmp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_CORE, root), merged_temp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_GPU, root), merged_temp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_SUPPORT, root), merged_temp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_METADATA, root), merged_temp_dir)
    overlay(resolve_nested_dir(SOURCE_PATH_ODML, root), merged_temp_dir)

    if gms_path := os.getenv('GMS_PATH'):
      # Play Services code needs to be massaged into a Java directory structure.
      overlay(gms_path, merged_temp_dir / 'org/tensorflow/lite/task/gms')

    gen_java.gen_java_docs(
        package=['org.tensorflow.lite', 'com.google.android.odml'],
        source_path=merged_temp_dir,
        output_dir=pathlib.Path(_OUT_DIR.value),
        site_path=pathlib.Path(_SITE_PATH.value),
        section_labels=SECTION_LABELS,
        federated_docs={k: root / v for k, v in EXTERNAL_APIS.items()})


if __name__ == '__main__':
  app.run(main)
