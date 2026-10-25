# [MODULE DOC]
# Executable entry point (binary) for the API generator. When run as a script it hands the
# generator library's main function to the absl application runner, which parses the command-line
# flags defined by the generator module and passes the remaining arguments (API mapping files).
# Importing it has no side effects beyond importing the generator and absl. Thin wrapper module.
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
# =============================================================================
"""Binary for generating the API for tensorflow."""

from absl import app

from tensorflow.python.tools.api.generator2.generator import generator

if __name__ == '__main__':
  app.run(generator.main)
