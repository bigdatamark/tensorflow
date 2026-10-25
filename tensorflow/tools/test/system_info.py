# [MODULE DOC]
# Small command-line entry point that gathers the machine configuration (via the system_info_lib
# helper module) and prints it to standard output. Exposes only a main function; running it as a
# script starts the absl application runner. No import-time side effects.
# KNOWN ISSUE: the module docstring says "Library for getting system information" but this file is
# a thin executable wrapper, the actual library is system_info_lib.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Library for getting system information during TensorFlow tests."""

from absl import app
from tensorflow.tools.test import system_info_lib


def main(unused_args):
  # [FUNCTION DOC]
  # Application entry point. Ignores the command-line arguments passed in by the app runner,
  # asks the system information library to collect the machine configuration (CPU, GPU, platform
  # and similar details) and prints the resulting record to standard output. Returns nothing.
  # Any error raised while gathering the configuration propagates to the caller.
  config = system_info_lib.gather_machine_configuration()
  print(config)


if __name__ == "__main__":
  app.run()  # pylint: disable=no-value-for-parameter
