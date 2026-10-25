# [MODULE DOC]
# Tiny script that starts a standard interactive Python console. Used in the pip package to give
# a prompt where TensorFlow can be imported (the console itself does not import it). Run as a
# script; its exit status is the value returned by the entry function.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Start a simple interactive console with TensorFlow available."""

import code
import sys


def main(_):
  # [FUNCTION DOC]
  # Entry point. The single argument (the command line list) is ignored. Starts a blocking
  # interactive console on the terminal until the user exits it, then returns zero as the exit
  # status.
  """Run an interactive console."""
  code.interact()
  return 0


if __name__ == '__main__':
  sys.exit(main(sys.argv))
