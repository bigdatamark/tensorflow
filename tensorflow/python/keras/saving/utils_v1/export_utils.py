# [MODULE DOC]
# HELPER: utilities for exporting Keras models to SavedModel, re-exported by the utils_v1 package.
# Defines the tables mapping each mode to MetaGraph tags (EXPORT_TAG_MAP) and default signature keys
# (SIGNATURE_KEY_MAP), default input names, and functions to build signature definitions for all
# export outputs, construct the export output mapping for a mode, validate or default serving
# outputs, log a report of included and excluded signatures, and compute timestamped and temporary
# export directory names. Mirrored by a copy in the saved_model model_utils package. Import-time
# side effects: builds the two mode maps and the method-name table.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# LINT.IfChange
"""Utilities for creating SavedModels."""

import collections
import os
import time

from tensorflow.python.keras.saving.utils_v1 import export_output as export_output_lib
from tensorflow.python.keras.saving.utils_v1 import mode_keys
from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.keras.saving.utils_v1.mode_keys import KerasModeKeys as ModeKeys
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat


# Mapping of the modes to appropriate MetaGraph tags in the SavedModel.
EXPORT_TAG_MAP = mode_keys.ModeKeyMap(**{
    ModeKeys.PREDICT: [tag_constants.SERVING],
    ModeKeys.TRAIN: [tag_constants.TRAINING],
    ModeKeys.TEST: [unexported_constants.EVAL]})

# For every exported mode, a SignatureDef map should be created using the
# functions `export_outputs_for_mode` and `build_all_signature_defs`. By
# default, this map will contain a single Signature that defines the input
# tensors and output predictions, losses, and/or metrics (depending on the mode)
# The default keys used in the SignatureDef map are defined below.
SIGNATURE_KEY_MAP = mode_keys.ModeKeyMap(**{
    ModeKeys.PREDICT: signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    ModeKeys.TRAIN: unexported_constants.DEFAULT_TRAIN_SIGNATURE_DEF_KEY,
    ModeKeys.TEST: unexported_constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY})

# Default names used in the SignatureDef input map, which maps strings to
# TensorInfo protos.
SINGLE_FEATURE_DEFAULT_NAME = 'feature'
SINGLE_RECEIVER_DEFAULT_NAME = 'input'
SINGLE_LABEL_DEFAULT_NAME = 'label'

### Below utilities are specific to SavedModel exports.


def build_all_signature_defs(receiver_tensors,
                             export_outputs,
                             receiver_tensors_alternatives=None,
                             serving_only=True):
  # [FUNCTION DOC]
  # Builds the mapping of signature names to signature definitions for a set of export outputs.
  # Inputs: receiver_tensors, a tensor or mapping of names to tensors (a lone tensor is stored under
  # the default receiver name); export_outputs, mapping from keys to export output objects;
  # receiver_tensors_alternatives, optional mapping of alternative names to further receiver
  # tensors (each yields extra signatures named alternative-name colon output-key); serving_only,
  # when true (default) drops signatures that fail the serving validity check. Output objects whose
  # signature creation raises a value error are excluded and recorded. Logs a report of included and
  # excluded signatures. Raises a value error if export_outputs is null or not a mapping. Returns
  # the signature map; a null or empty key is rendered as the text "None".
  """Build `SignatureDef`s for all export outputs.

  Args:
    receiver_tensors: a `Tensor`, or a dict of string to `Tensor`, specifying
      input nodes where this receiver expects to be fed by default.  Typically,
      this is a single placeholder expecting serialized `tf.Example` protos.
    export_outputs: a dict of ExportOutput instances, each of which has
      an as_signature_def instance method that will be called to retrieve
      the signature_def for all export output tensors.
    receiver_tensors_alternatives: a dict of string to additional
      groups of receiver tensors, each of which may be a `Tensor` or a dict of
      string to `Tensor`.  These named receiver tensor alternatives generate
      additional serving signatures, which may be used to feed inputs at
      different points within the input receiver subgraph.  A typical usage is
      to allow feeding raw feature `Tensor`s *downstream* of the
      tf.io.parse_example() op.  Defaults to None.
    serving_only: boolean; if true, resulting signature defs will only include
      valid serving signatures. If false, all requested signatures will be
      returned.

  Returns:
    signature_def representing all passed args.

  Raises:
    ValueError: if export_outputs is not a dict
  """
  if not isinstance(receiver_tensors, dict):
    receiver_tensors = {SINGLE_RECEIVER_DEFAULT_NAME: receiver_tensors}
  if export_outputs is None or not isinstance(export_outputs, dict):
    raise ValueError('export_outputs must be a dict and not'
                     '{}'.format(type(export_outputs)))

  signature_def_map = {}
  excluded_signatures = {}
  for output_key, export_output in export_outputs.items():
    signature_name = '{}'.format(output_key or 'None')
    try:
      signature = export_output.as_signature_def(receiver_tensors)
      signature_def_map[signature_name] = signature
    except ValueError as e:
      excluded_signatures[signature_name] = str(e)

  if receiver_tensors_alternatives:
    for receiver_name, receiver_tensors_alt in (
        receiver_tensors_alternatives.items()):
      if not isinstance(receiver_tensors_alt, dict):
        receiver_tensors_alt = {
            SINGLE_RECEIVER_DEFAULT_NAME: receiver_tensors_alt
        }
      for output_key, export_output in export_outputs.items():
        signature_name = '{}:{}'.format(receiver_name or 'None', output_key or
                                        'None')
        try:
          signature = export_output.as_signature_def(receiver_tensors_alt)
          signature_def_map[signature_name] = signature
        except ValueError as e:
          excluded_signatures[signature_name] = str(e)

  _log_signature_report(signature_def_map, excluded_signatures)

  # The above calls to export_output_lib.as_signature_def should return only
  # valid signatures; if there is a validity problem, they raise a ValueError,
  # in which case we exclude that signature from signature_def_map above.
  # The is_valid_signature check ensures that the signatures produced are
  # valid for serving, and acts as an additional sanity check for export
  # signatures produced for serving. We skip this check for training and eval
  # signatures, which are not intended for serving.
  if serving_only:
    signature_def_map = {
        k: v
        for k, v in signature_def_map.items()
        if signature_def_utils.is_valid_signature(v)
    }
  return signature_def_map


_FRIENDLY_METHOD_NAMES = {
    signature_constants.CLASSIFY_METHOD_NAME: 'Classify',
    signature_constants.REGRESS_METHOD_NAME: 'Regress',
    signature_constants.PREDICT_METHOD_NAME: 'Predict',
    unexported_constants.SUPERVISED_TRAIN_METHOD_NAME: 'Train',
    unexported_constants.SUPERVISED_EVAL_METHOD_NAME: 'Eval',
}


def _log_signature_report(signature_def_map, excluded_signatures):
  # [FUNCTION DOC]
  # HELPER: reporting step of build_all_signature_defs.
  # KNOWN ISSUE: the inline comment speaks of three standard methods, but five method names
  # (classify, regress, predict, train, eval) are pre-seeded; the excluded-signature message also
  # contains a doubled word ("be be").
  # Logs, at info level, which signature names were included per method (every known
  # method gets a line, "None" when empty), then any excluded signatures with their error text, and
  # a warning if no signatures exist or the default serving key is missing. Inputs: signature_def_map,
  # mapping of names to signature definitions; excluded_signatures, mapping of names to messages.
  # Returns nothing.
  """Log a report of which signatures were produced."""
  sig_names_by_method_name = collections.defaultdict(list)

  # We'll collect whatever method_names are present, but also we want to make
  # sure to output a line for each of the three standard methods even if they
  # have no signatures.
  for method_name in _FRIENDLY_METHOD_NAMES:
    sig_names_by_method_name[method_name] = []

  for signature_name, sig in signature_def_map.items():
    sig_names_by_method_name[sig.method_name].append(signature_name)

  # TODO(b/67733540): consider printing the full signatures, not just names
  for method_name, sig_names in sig_names_by_method_name.items():
    if method_name in _FRIENDLY_METHOD_NAMES:
      method_name = _FRIENDLY_METHOD_NAMES[method_name]
    logging.info('Signatures INCLUDED in export for {}: {}'.format(
        method_name, sig_names if sig_names else 'None'))

  if excluded_signatures:
    logging.info('Signatures EXCLUDED from export because they cannot be '
                 'be served via TensorFlow Serving APIs:')
    for signature_name, message in excluded_signatures.items():
      logging.info('\'{}\' : {}'.format(signature_name, message))

  if not signature_def_map:
    logging.warning('Export includes no signatures!')
  elif (signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY not in
        signature_def_map):
    logging.warning('Export includes no default signature!')


# When we create a timestamped directory, there is a small chance that the
# directory already exists because another process is also creating these
# directories. In this case we just wait one second to get a new timestamp and
# try again. If this fails several times in a row, then something is seriously
# wrong.
MAX_DIRECTORY_CREATION_ATTEMPTS = 10


def get_timestamped_export_dir(export_dir_base):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the result is a byte string (paths are converted to bytes), not a text string as
  # the docstring's wording suggests.
  # Computes a path for a new export subdirectory named by the current time in whole seconds. Input:
  # export_dir_base, directory path text. If the path exists already, logs a warning, sleeps one
  # second and retries, up to the maximum attempts constant (ten). Does not create the directory.
  # Returns the byte-string path; raises a runtime error when all attempts collide.
  """Builds a path to a new subdirectory within the base directory.

  Each export is written into a new subdirectory named using the
  current time.  This guarantees monotonically increasing version
  numbers even across multiple runs of the pipeline.
  The timestamp used is the number of seconds since epoch UTC.

  Args:
    export_dir_base: A string containing a directory to write the exported
        graph and checkpoints.
  Returns:
    The full path of the new subdirectory (which is not actually created yet).

  Raises:
    RuntimeError: if repeated attempts fail to obtain a unique timestamped
      directory name.
  """
  attempts = 0
  while attempts < MAX_DIRECTORY_CREATION_ATTEMPTS:
    timestamp = int(time.time())

    result_dir = os.path.join(
        compat.as_bytes(export_dir_base), compat.as_bytes(str(timestamp)))
    if not gfile.Exists(result_dir):
      # Collisions are still possible (though extremely unlikely): this
      # directory is not actually created yet, but it will be almost
      # instantly on return from this function.
      return result_dir
    time.sleep(1)
    attempts += 1
    logging.warning(
        'Directory {} already exists; retrying (attempt {}/{})'.format(
            compat.as_str(result_dir), attempts,
            MAX_DIRECTORY_CREATION_ATTEMPTS))
  raise RuntimeError('Failed to obtain a unique export directory name after '
                     '{} attempts.'.format(MAX_DIRECTORY_CREATION_ATTEMPTS))


def get_temp_export_dir(timestamped_export_dir):
  # [FUNCTION DOC]
  # Computes a sibling path whose last component is prefixed with temp- (ignored by the serving
  # system because it is not an integer). Input: timestamped_export_dir, text or bytes path. Always
  # returns a byte string. No filesystem access.
  """Builds a directory name based on the argument but starting with 'temp-'.

  This relies on the fact that TensorFlow Serving ignores subdirectories of
  the base directory that can't be parsed as integers.

  Args:
    timestamped_export_dir: the name of the eventual export directory, e.g.
      /foo/bar/<timestamp>

  Returns:
    A sister directory prefixed with 'temp-', e.g. /foo/bar/temp-<timestamp>.
  """
  (dirname, basename) = os.path.split(timestamped_export_dir)
  if isinstance(basename, bytes):
    str_name = basename.decode('utf-8')
  else:
    str_name = str(basename)
  temp_export_dir = os.path.join(
      compat.as_bytes(dirname),
      compat.as_bytes('temp-{}'.format(str_name)))
  return temp_export_dir


def export_outputs_for_mode(
    mode, serving_export_outputs=None, predictions=None, loss=None,
    metrics=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: any mode that is not predict or train (including an unrecognized one that is
  # nonetheless a key of the signature map) is treated as eval; and the docstring's loss and metrics
  # inputs are ignored for predict mode.
  # Builds the export outputs mapping for a mode. Inputs: mode, a mode name string; the remaining
  # inputs are optional: serving_export_outputs (mapping of serving outputs), predictions, loss and
  # metrics. Predict mode validates or defaults the serving outputs from the first two inputs; train
  # and eval modes return a one-entry mapping from the mode's default signature key to a train or
  # eval output built from predictions, loss and metrics. Raises a value error when the mode is not
  # in the signature key map.
  """Util function for constructing a `ExportOutput` dict given a mode.

  The returned dict can be directly passed to `build_all_signature_defs` helper
  function as the `export_outputs` argument, used for generating a SignatureDef
  map.

  Args:
    mode: A `ModeKeys` specifying the mode.
    serving_export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving. Should be a dict or None.
    predictions: A dict of Tensors or single Tensor representing model
        predictions. This argument is only used if serving_export_outputs is not
        set.
    loss: A dict of Tensors or single Tensor representing calculated loss.
    metrics: A dict of (metric_value, update_op) tuples, or a single tuple.
      metric_value must be a Tensor, and update_op must be a Tensor or Op

  Returns:
    Dictionary mapping the key to an `ExportOutput` object.
    The key is the expected SignatureDef key for the mode.

  Raises:
    ValueError: if an appropriate ExportOutput cannot be found for the mode.
  """
  if mode not in SIGNATURE_KEY_MAP:
    raise ValueError(
        'Export output type not found for mode: {}. Expected one of: {}.\n'
        .format(mode, SIGNATURE_KEY_MAP.keys()))
  signature_key = SIGNATURE_KEY_MAP[mode]
  if mode_keys.is_predict(mode):
    return get_export_outputs(serving_export_outputs, predictions)
  elif mode_keys.is_train(mode):
    return {signature_key: export_output_lib.TrainOutput(
        loss=loss, predictions=predictions, metrics=metrics)}
  else:
    return {signature_key: export_output_lib.EvalOutput(
        loss=loss, predictions=predictions, metrics=metrics)}


def get_export_outputs(export_outputs, predictions):
  # [FUNCTION DOC]
  # Validates serving export outputs or creates a default one. Inputs: export_outputs, a mapping of
  # keys to export output objects, or null; predictions, a tensor or mapping used to build a default
  # predict output (under the default serving key) when export_outputs is null. Raises a type error
  # if the outputs are not a mapping or contain non-ExportOutput values. Then ensures a default
  # serving entry exists (may mutate the given mapping). Returns the mapping.
  """Validate export_outputs or create default export_outputs.

  Args:
    export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving. Should be a dict or None.
    predictions:  Predictions `Tensor` or dict of `Tensor`.

  Returns:
    Valid export_outputs dict

  Raises:
    TypeError: if export_outputs is not a dict or its values are not
      ExportOutput instances.
  """
  if export_outputs is None:
    default_output = export_output_lib.PredictOutput(predictions)
    export_outputs = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: default_output}

  if not isinstance(export_outputs, dict):
    raise TypeError('export_outputs must be dict, given: {}'.format(
        export_outputs))
  for v in export_outputs.values():
    if not isinstance(v, export_output_lib.ExportOutput):
      raise TypeError(
          'Values in export_outputs must be ExportOutput objects. '
          'Given: {}'.format(export_outputs))

  _maybe_add_default_serving_output(export_outputs)

  return export_outputs


def _maybe_add_default_serving_output(export_outputs):
  # [FUNCTION DOC]
  # HELPER: guarantees a default serving signature key. Input: export_outputs, a mutable mapping.
  # If it has a single entry under another key, that same output is also added under the default
  # serving key (mutating the mapping, and keeping the original key). If it has several entries and
  # none uses the default key, raises a value error. Returns the same mapping.
  """Add a default serving output to the export_outputs if not present.

  Args:
    export_outputs: Describes the output signatures to be exported to
      `SavedModel` and used during serving. Should be a dict.

  Returns:
    export_outputs dict with default serving signature added if necessary

  Raises:
    ValueError: if multiple export_outputs were provided without a default
      serving key.
  """
  if len(export_outputs) == 1:
    (key, value), = export_outputs.items()
    if key != signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
      export_outputs[
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = value
  if len(export_outputs) > 1:
    if (signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        not in export_outputs):
      raise ValueError(
          'Multiple export_outputs were provided, but none of them is '
          'specified as the default.  Do this by naming one of them with '
          'signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY.')

  return export_outputs
# LINT.ThenChange(//tensorflow/python/saved_model/model_utils/export_utils.py)
