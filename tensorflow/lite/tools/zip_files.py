# [MODULE DOC]
# Small command line tool that packs the Java source files found under a directory into a zip
# archive. Defines two string flags: the output zip path and the directory to scan. Run as a
# script through the absl application runner. Import-time side effect: flag definitions.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Creates a zip package of the files passed in."""
import os
import zipfile

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("export_zip_path", None, "Path to zip file.")
flags.DEFINE_string("file_directory", None, "Path to the files to be zipped.")


def main(_):
  # [FUNCTION DOC]
  # Entry point; the positional arguments are ignored. Creates (overwriting) the zip file named
  # by the export path flag and walks the directory given by the file directory flag recursively,
  # adding every file whose name ends with the Java extension. Entries are stored under the
  # path as built from the walked directory. Returns nothing. Fails if either flag is unset.
  with zipfile.ZipFile(FLAGS.export_zip_path, mode="w") as zf:
    for root, _, files in os.walk(FLAGS.file_directory):
      for f in files:
        if f.endswith(".java"):
          zf.write(os.path.join(root, f))


if __name__ == "__main__":
  app.run(main)
