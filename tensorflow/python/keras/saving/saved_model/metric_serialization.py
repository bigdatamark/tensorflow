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
# [MODULE DOC]
# Defines MetricSavedModelSaver, the saver that serializes Keras metric objects into a
# SavedModel, built on the layer saver. Saves metric metadata and the metric variables;
# functions are not saved. No import-time side effects.
"""Classes and functions implementing Metrics SavedModel serialization."""

from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.trackable import data_structures


class MetricSavedModelSaver(layer_serialization.LayerSavedModelSaver):
  # [CLASS DOC]
  # Layer saver specialization for metrics. Uses the metric identifier, a smaller metadata
  # mapping and saves only the metric's variables as extra objects.
  """Metric serialization."""

  @property
  def object_identifier(self):
    # [FUNCTION DOC]
    # Returns the constant string marking a metric in the SavedModel.
    return constants.METRIC_IDENTIFIER

  def _python_properties_internal(self):
    # [FUNCTION DOC]
    # HELPER: overrides the layer metadata builder for metrics.
    # Builds a mapping with the registered class name, the metric name and its dtype, merges
    # in the serialized config, and adds the build input shape when known. Returns it.
    metadata = dict(
        class_name=generic_utils.get_registered_name(type(self.obj)),
        name=self.obj.name,
        dtype=self.obj.dtype)
    metadata.update(layer_serialization.get_serialized(self.obj))
    if self.obj._build_input_shape is not None:  # pylint: disable=protected-access
      metadata['build_input_shape'] = self.obj._build_input_shape  # pylint: disable=protected-access
    return metadata

  def _get_serialized_attributes_internal(self, unused_serialization_cache):
    # [FUNCTION DOC]
    # HELPER: overrides the layer builder for metrics. The cache argument is unused.
    # Returns a pair: an objects mapping holding the metric's variables wrapped as a
    # trackable structure, and an empty functions mapping.
    # KNOWN ISSUE: functions are not saved yet (see the bug reference in the code), so custom
    # metrics cannot be revived from saved functions.
    return (dict(variables=data_structures.wrap_or_unwrap(self.obj.variables)),
            dict())  # TODO(b/135550038): save functions to enable saving
                     # custom metrics.
