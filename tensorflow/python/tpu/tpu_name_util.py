# [MODULE DOC]
# HELPER: helper module providing TPU device-name utilities. Currently exposes a single function
# that builds the device name of a virtual core in a replicated TPU computation (exported for v1).
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
# ======================================
"""Helper functions for TPU device names."""

from typing import Text
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["tpu.core"])
def core(num: int) -> Text:
  # [FUNCTION DOC]
  # Builds the device name string of the replicated TPU core with the given virtual core number
  # (an integer within each replica). Returns the name, suitable for a device scope. No
  # validation of num and no side effects.
  """Returns the device name for a core in a replicated TPU computation.

  Args:
    num: the virtual core number within each replica to which operators should
    be assigned.
  Returns:
    A device name, suitable for passing to `tf.device()`.
  """
  return "device:TPU_REPLICATED_CORE:{}".format(num)
