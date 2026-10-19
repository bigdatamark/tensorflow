# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Defines NetworkSavedModelSaver, the saver for Keras functional networks. It reuses the
# model saver and only changes the object identifier. No import-time side effects.
"""Classes and functions implementing to Network SavedModel serialization."""

from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import model_serialization


# FunctionalModel serialization is pretty much the same as Model serialization.
class NetworkSavedModelSaver(model_serialization.ModelSavedModelSaver):
  # [CLASS DOC]
  # Model saver for functional networks; behaves like the model saver apart from the
  # identifier it reports.
  """Network serialization."""

  @property
  def object_identifier(self):
    # [FUNCTION DOC]
    # Returns the constant string marking a Keras network in the SavedModel.
    return constants.NETWORK_IDENTIFIER
