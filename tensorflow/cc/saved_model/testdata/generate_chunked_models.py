# [MODULE DOC]
# HELPER: Test-data generator script for the saved model merger tests.
# Executable that, given a path flag to a test data directory, writes a regular (non-chunked)
# saved model containing a large constant and then a chunked version of it produced by the
# proto splitter (alongside a text form of the proto). Defines one string flag named path.
# Run as a main program it enables TF2 behavior and runs the application main.
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

"""Generates GraphDef test data for Merger.

Constructs chunked protos test data containing GraphDefs with lots of nodes and
large nodes for Merger::Read and Merger::Merge.
"""

from collections.abc import Sequence

import os

from absl import app
from absl import flags
import numpy as np

from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_options
from tensorflow.python.util import compat
from tensorflow.tools.proto_splitter import constants
from tensorflow.tools.proto_splitter.python import saved_model as proto_splitter

SPLITTER_TESTDATA_PATH = flags.DEFINE_string(
    "path", None, help="Path to testdata directory.")


def generate_non_chunked_model(non_chunked_dir: str):
  # [FUNCTION DOC]
  # non_chunked_dir: output directory for a regular saved model. Builds a module with a random
  # 150 by 150 floating point constant and a function returning it, lowers the splitter's
  # maximum chunk size to 80000 bytes (global side effect on the splitter constants, which makes
  # later splitting create chunks), and saves the module with the function as signature, with
  # the image format option disabled. Returns nothing; writes files. Output is random each run.
  root = module.Module()
  root.c = constant_op.constant(np.random.random_sample([150, 150]))
  constants.debug_set_max_size(80000)
  root.get_c = def_function.function(lambda: root.c)
  signatures = root.get_c.get_concrete_function()
  save.save(root, non_chunked_dir, signatures=signatures,
            options=save_options.SaveOptions(experimental_image_format=False))


def generate_chunked_model(non_chunked_dir: str, chunked_dir: str):
  # [FUNCTION DOC]
  # non_chunked_dir: directory holding an existing saved model to read. chunked_dir: output
  # directory. Parses the saved model proto, writes its text form to a text-proto file named with
  # the saved model prefix in the output directory, and writes the chunked (split) form using
  # the proto splitter under the same prefix. Returns nothing; writes files.
  saved_model = loader_impl.parse_saved_model(non_chunked_dir)
  prefix = file_io.join(compat.as_str(chunked_dir), "saved_model")
  file_io.write_string_to_file(f"{prefix}.pbtxt", str(saved_model))
  proto_splitter.SavedModelSplitter(saved_model).write(prefix)


def main(argv: Sequence[str]) -> None:
  # [FUNCTION DOC]
  # Program entry point. argv: command-line arguments; raises a usage error if there is anything
  # beyond the program name. Uses the path flag as the root, creates the chunked saved model
  # directory layout beneath it (a non-chunked model directory and a chunked model directory),
  # generates the non-chunked model then the chunked one. Returns nothing.
  # The path flag defaults to nothing, so running without it fails when joining paths.
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  main_dir = os.path.join(SPLITTER_TESTDATA_PATH.value, "chunked_saved_model")
  non_chunked_dir = os.path.join(main_dir, "non_chunked_model")
  generate_non_chunked_model(non_chunked_dir)
  chunked_dir = os.path.join(main_dir, "chunked_model")
  generate_chunked_model(non_chunked_dir, chunked_dir)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  app.run(main)
