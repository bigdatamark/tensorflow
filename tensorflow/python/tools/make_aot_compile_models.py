# [MODULE DOC]
# Script that generates two small SavedModels (a small and a large matrix multiplication) used by
# ahead-of-time compilation tests. Import-time side effect: defines the required out_dir command
# line flag. When run as a script it marks that flag required and runs main, which writes the
# models under the output directory.
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
"""Generate some SavedModels for use by AOT compilation tests."""

import os

from absl import app
from absl import flags

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import autotrackable


flags.DEFINE_string('out_dir', None,
                    'Directory to output saved models to.')

FLAGS = flags.FLAGS


def create_large_matmul_savedmodel(out_dir):
  # [FUNCTION DOC]
  # Builds a trackable root object holding a compiled function that multiplies two float32
  # matrices of fixed shapes 3000x5000 and 5000x4000, calls it once with zero inputs so it is
  # traced, and saves it under a subdirectory named x_matmul_y_large inside out_dir (a directory
  # path string) with that function as the serving signature. Also creates an empty variables index
  # file in the saved model so the build rule has the expected output. Returns nothing.
  """Create a SavedModel that performs a large matmul."""
  root = autotrackable.AutoTrackable()
  root.f = def_function.function(
      lambda x, y: math_ops.matmul(x, y),  # pylint: disable=unnecessary-lambda
      input_signature=[tensor_spec.TensorSpec([3000, 5000], dtypes.float32),
                       tensor_spec.TensorSpec([5000, 4000], dtypes.float32),])
  root.f(x=array_ops.zeros((3000, 5000)),
         y=array_ops.zeros((5000, 4000)))
  save_dir = os.path.join(out_dir, 'x_matmul_y_large')
  save.save(root, save_dir, root.f)
  # This simple SavedModel lacks any variables, but we need to create a
  # variables.index file to make bazel genrule happy.
  with open(os.path.join(save_dir, 'variables', 'variables.index'), 'w'):
    pass


def create_small_matmul_savedmodel(out_dir):
  # [FUNCTION DOC]
  # Same as the large variant but with float32 shapes 3x5 and 5x4: saves a matmul function as a
  # SavedModel in the x_matmul_y_small subdirectory of out_dir and creates an empty variables
  # index file there. Returns nothing; writes files.
  """Create a SavedModel that performs a small matmul."""
  root = autotrackable.AutoTrackable()
  root.f = def_function.function(
      lambda x, y: math_ops.matmul(x, y),  # pylint: disable=unnecessary-lambda
      input_signature=[tensor_spec.TensorSpec([3, 5], dtypes.float32),
                       tensor_spec.TensorSpec([5, 4], dtypes.float32),])
  root.f(x=array_ops.zeros((3, 5)),
         y=array_ops.zeros((5, 4)))
  save_dir = os.path.join(out_dir, 'x_matmul_y_small')
  save.save(root, save_dir, root.f)
  # This simple SavedModel lacks any variables, but we need to create a
  # variables.index file to make bazel genrule happy.
  with open(os.path.join(save_dir, 'variables', 'variables.index'), 'w'):
    pass


def main(unused_args):
  # [FUNCTION DOC]
  # Application main; the argument is ignored. Creates the small and then the large matmul
  # SavedModels in the directory given by the out_dir flag. Returns nothing; writes files.
  create_small_matmul_savedmodel(FLAGS.out_dir)
  create_large_matmul_savedmodel(FLAGS.out_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('out_dir')
  app.run(main)
