# [MODULE DOC]
# KNOWN ISSUE: the argument parser description mentions generating a metrics model for mobilenet
# v1, which does not describe this script (it copies associated files between models).
# Command line tool that copies the 'associated files' (a zip archive appended to a TFLite model
# with metadata) from one model to a copy of another model. Creates the argument parser at
# import time. Used while building validation models for the mini benchmark.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""TFLite model associated files copier.

A TFLite model with metadata may have 'associated files': extra data stored as a
zip file appended to the model. This script copies such files from one model to
another. This is used as part of constructing validation models.

See https://www.tensorflow.org/lite/convert/metadata for description of models
with metadata.

If there are no associated files in the provided file, the model is output
as-is.
"""

import argparse
import sys
import zipfile

parser = argparse.ArgumentParser(
    description='Script to generate a metrics model for mobilenet v1.')
parser.add_argument('model', help='Input model filepath')
parser.add_argument(
    'copy_associated_files_from',
    help='Model with potential associated files filepath')
parser.add_argument('output', help='Output filepath')


def main(model_path, associated_files_path, output_path):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the zip archives are opened and never closed, so the appended entries may not be
  # fully flushed until the objects are collected.
  # Inputs: path of the model to copy, path of the model whose associated files are wanted, and
  # the output path. Copies the first model's bytes to the output; then, if the second file is a
  # zip archive, appends every entry of it to the output (as an archive in append mode). If it is
  # not a zip, the output is the plain copy. Returns nothing. Side effect: writes the output file.
  with open(model_path, 'rb') as input_file:
    with open(output_path, 'wb') as output_file:
      output_file.write(input_file.read())
  if zipfile.is_zipfile(associated_files_path):
    zip_src = zipfile.ZipFile(associated_files_path, 'r')
    zip_tgt = zipfile.ZipFile(output_path, 'a')
    for info in zip_src.infolist():
      zip_tgt.writestr(info, zip_src.read(info))


if __name__ == '__main__':
  flags, unparsed = parser.parse_known_args()

  if unparsed:
    parser.print_usage()
    sys.stderr.write('\nGot the following unparsed args, %r please fix.\n' %
                     unparsed)
    exit(1)
  else:
    main(flags.model, flags.copy_associated_files_from, flags.output)
    exit(0)
