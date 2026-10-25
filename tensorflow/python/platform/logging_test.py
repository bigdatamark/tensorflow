# [MODULE DOC]
# Minimal smoke test for the logging wrapper module: logs one error message and expects no
# exception. Uses the googletest router for the test case class and main. Runs the test main
# when executed as a script.

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

from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging


class EventLoaderTest(googletest.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the class name (event loader) is a leftover; it actually tests the logging
  # wrapper, not an event loader.
  # Test case holding the single logging smoke test; no fixtures.

  def test_log(self):
    # [FUNCTION DOC]
    # Emits one error-level log message through the TensorFlow logging wrapper. The only
    # expectation is that no exception is raised; nothing is asserted about the output.
    # Just check that logging works without raising an exception.
    logging.error("test log message")


if __name__ == "__main__":
  googletest.main()
