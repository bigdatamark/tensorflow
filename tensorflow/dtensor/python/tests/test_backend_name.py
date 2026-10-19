# [MODULE DOC]
# HELPER: small helper module that names the accelerator backend a DTensor test is running on.
# Defines an enumeration of known test backends (unspecified, cpu, gpu, two-gpu, several tpu
# flavors) and a module-level constant holding the backend chosen for this process.
# Import-time side effect: reads the environment variable for the test backend (defaulting to
# the unspecified value) and converts it to an enumeration member; an unrecognized value makes
# the import fail with a value error. Used by the other DTensor test utility modules.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""List of test backend names."""

import enum
import os


# LINT.IfChange(backend_name)
class DTensorTestUtilBackend(enum.Enum):
  # [CLASS DOC]
  # HELPER: enumeration of backend names used by DTensor test utilities.
  # Enumeration whose values are the strings accepted in the test backend environment variable:
  # unspecified, cpu, gpu, 2gpus, tpu, tpu_se, tpu_v3_2x2 and tpu_v4_2x2. Holds no state beyond
  # the members; members are looked up by value.
  """DTensor backend the test is being run on."""
  UNSPECIFIED = 'unspecified'
  CPU = 'cpu'
  GPU = 'gpu'
  GPU_2DEVS_BACKEND = '2gpus'
  TPU = 'tpu'
  TPU_STREAM_EXECUTOR = 'tpu_se'
  TPU_V3_DONUT_BACKEND = 'tpu_v3_2x2'
  TPU_V4_DONUT_BACKEND = 'tpu_v4_2x2'


DTENSOR_TEST_UTIL_BACKEND = DTensorTestUtilBackend(
    os.getenv('DTENSOR_TEST_UTIL_BACKEND', default='unspecified')
)

# LINT.ThenChange()
