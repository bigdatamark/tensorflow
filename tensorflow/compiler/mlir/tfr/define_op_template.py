# [MODULE DOC]
# Command line template script that defines composite ops and generates either the C++ op
# registration source or the TFR MLIR source for them. It defines a required output path flag and a
# boolean flag selecting the mode. Composite functions are found in this module by the name prefix
# for composites. Only a placeholder test random op is defined here; users copy the template and
# add their own composite functions. Defines flags at import time; run as a program.
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
"""A template to define composite ops."""

# pylint: disable=g-direct-tensorflow-import

import os
import sys

from absl import app
from tensorflow.compiler.mlir.tfr.python.composite import Composite
from tensorflow.compiler.mlir.tfr.python.op_reg_gen import gen_register_op
from tensorflow.compiler.mlir.tfr.python.tfr_gen import tfr_gen_from_module
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'output', None,
    'Path to write the genereated register op file and MLIR file.')

flags.DEFINE_bool('gen_register_op', True,
                  'Generate register op cc file or tfr mlir file.')

flags.mark_flag_as_required('output')


@Composite('TestRandom', derived_attrs=['T: numbertype'], outputs=['o: T'])
def _composite_random_op():
  # [FUNCTION DOC]
  # HELPER: Placeholder composite op definition used by this template to give the generator
  # something to process.
  # Decorated as the composition for the op named test random, with a derived attribute of numeric
  # type and one output of that type. Takes no arguments, has an empty body and returns nothing.
  pass


def main(_):
  # [FUNCTION DOC]
  # Program entry point (the single argument is the unused list of leftover command line
  # arguments). If the register op flag is true, requires the output path to end with the C++
  # source suffix and generates C++ op registration code for all functions in this module whose
  # names start with the composite prefix; otherwise requires the output path to end with the MLIR
  # suffix and generates TFR MLIR text for them. Creates the output directory if missing and
  # writes the generated text to the output file. Raises an assertion error for a wrong suffix.
  # Note: when the output path has no directory component, the empty directory name makes the
  # directory creation call fail.
  if FLAGS.gen_register_op:
    assert FLAGS.output.endswith('.cc')
    generated_code = gen_register_op(sys.modules[__name__], '_composite_')
  else:
    assert FLAGS.output.endswith('.mlir')
    generated_code = tfr_gen_from_module(sys.modules[__name__], '_composite_')

  dirname = os.path.dirname(FLAGS.output)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  with open(FLAGS.output, 'w') as f:
    f.write(generated_code)


if __name__ == '__main__':
  app.run(main=main)
