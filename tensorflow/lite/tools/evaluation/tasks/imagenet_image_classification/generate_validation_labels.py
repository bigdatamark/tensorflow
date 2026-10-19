# [MODULE DOC]
# Command line script that converts the ILSVRC 2012 devkit validation ground truth (a list of
# synset ids, one per validation image) into a text file with one human readable label (the
# first word of the synset) per line, for use by the accuracy evaluation tools. Reads the devkit
# metadata and validation files from fixed relative locations. Run as a script; all functions
# except main are private helpers.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tool to convert ILSVRC devkit validation ground truth to synset labels."""

import argparse
from os import path
import sys
import scipy.io

_SYNSET_ARRAYS_RELATIVE_PATH = 'data/meta.mat'
_VALIDATION_FILE_RELATIVE_PATH = 'data/ILSVRC2012_validation_ground_truth.txt'


def _synset_to_word(filepath):
  # [FUNCTION DOC]
  # HELPER: loads the synset metadata for label generation.
  # Reads the devkit metadata matrix file at the given path and builds a mapping from integer
  # synset id to the first comma-separated word of its description. Returns that mapping.
  # Raises file or format errors from the matrix loader.
  """Returns synset to word dictionary by reading sysnset arrays."""
  mat = scipy.io.loadmat(filepath)
  entries = mat['synsets']
  # These fields are listed in devkit readme.txt
  fields = [
      'synset_id', 'WNID', 'words', 'gloss', 'num_children', 'children',
      'wordnet_height', 'num_train_images'
  ]
  synset_index = fields.index('synset_id')
  words_index = fields.index('words')
  synset_to_word = {}
  for entry in entries:
    entry = entry[0]
    synset_id = int(entry[synset_index][0])
    first_word = entry[words_index][0].split(',')[0]
    synset_to_word[synset_id] = first_word
  return synset_to_word


def _validation_file_path(ilsvrc_dir):
  # [FUNCTION DOC]
  # HELPER: path builder. Returns the path of the validation ground truth file inside the given
  # devkit directory.
  return path.join(ilsvrc_dir, _VALIDATION_FILE_RELATIVE_PATH)


def _synset_array_path(ilsvrc_dir):
  # [FUNCTION DOC]
  # HELPER: path builder. Returns the path of the synset metadata matrix file inside the given
  # devkit directory.
  return path.join(ilsvrc_dir, _SYNSET_ARRAYS_RELATIVE_PATH)


def _generate_validation_labels(ilsvrc_dir, output_file):
  # [FUNCTION DOC]
  # HELPER: core of the script. Loads the synset-to-word mapping from the devkit directory, then
  # reads the validation ground truth file line by line (one integer synset id per line) and
  # writes the matching word on its own line to the output file (overwritten). Returns nothing.
  # Raises a key error for an unknown synset id and a value error for a non-integer line.
  synset_to_word = _synset_to_word(_synset_array_path(ilsvrc_dir))
  with open(_validation_file_path(ilsvrc_dir), 'r') as synset_id_file, open(
      output_file, 'w') as output:
    for synset_id in synset_id_file:
      synset_id = int(synset_id)
      output.write('%s\n' % synset_to_word[synset_id])


def _check_arguments(args):
  # [FUNCTION DOC]
  # HELPER: validates parsed command line arguments. Raises a value error if the output path is
  # missing, if the devkit directory is missing or not a directory, or if the validation file or
  # synset metadata file is not found inside it. Returns nothing when all checks pass.
  if not args.validation_labels_output:
    raise ValueError('Invalid path to output file.')
  ilsvrc_dir = args.ilsvrc_devkit_dir
  if not ilsvrc_dir or not path.isdir(ilsvrc_dir):
    raise ValueError('Invalid path to ilsvrc_dir')
  if not path.exists(_validation_file_path(ilsvrc_dir)):
    raise ValueError('Invalid path to ilsvrc_dir, cannot find validation file.')
  if not path.exists(_synset_array_path(ilsvrc_dir)):
    raise ValueError(
        'Invalid path to ilsvrc_dir, cannot find synset arrays file.')


def main():
  # [FUNCTION DOC]
  # Script entry point. Parses two options (output path for labels and devkit directory path),
  # validates them, and on validation failure prints usage and an error message to the error
  # stream and exits with status one. Otherwise generates the labels file. Returns nothing.
  parser = argparse.ArgumentParser(
      description='Converts ILSVRC devkit validation_ground_truth.txt to synset'
      ' labels file that can be used by the accuracy script.')
  parser.add_argument(
      '--validation_labels_output',
      type=str,
      help='Full path for outputting validation labels.')
  parser.add_argument(
      '--ilsvrc_devkit_dir',
      type=str,
      help='Full path to ILSVRC 2012 devkit directory.')
  args = parser.parse_args()
  try:
    _check_arguments(args)
  except ValueError as e:
    parser.print_usage()
    file_name = path.basename(sys.argv[0])
    sys.stderr.write('{0}: error: {1}\n'.format(file_name, str(e)))
    sys.exit(1)
  _generate_validation_labels(args.ilsvrc_devkit_dir,
                              args.validation_labels_output)


if __name__ == '__main__':
  main()
