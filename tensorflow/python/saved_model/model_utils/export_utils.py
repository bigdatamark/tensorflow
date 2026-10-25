# [MODULE DOC]
# Utilities for creating SavedModels from estimator-style export outputs. Provides the default
# mappings from mode keys to export tags and signature keys, the default receiver/feature/label
# names, construction of the full signature-definition mapping from receiver tensors and export
# outputs (with reporting of what was included or excluded), timestamped and temporary export
# directory naming, and helpers that choose or validate export outputs for a mode.
# Import-time side effects: builds the module-level mode-keyed mappings and a friendly-name table
# only; no registration. Part of the model_utils package.
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
"""Utilities for creating SavedModels."""

import collections
import os
import time

from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import op_selector
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model.model_utils import export_output as export_output_lib
from tensorflow.python.saved_model.model_utils import mode_keys
from tensorflow.python.saved_model.model_utils.mode_keys import KerasModeKeys as ModeKeys
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity


# Mapping of the modes to appropriate MetaGraph tags in the SavedModel.
EXPORT_TAG_MAP = mode_keys.ModeKeyMap(**{
    ModeKeys.PREDICT: [tag_constants.SERVING],
    ModeKeys.TRAIN: [tag_constants.TRAINING],
    ModeKeys.TEST: [tag_constants.EVAL]})

# For every exported mode, a SignatureDef map should be created using the
# functions `export_outputs_for_mode` and `build_all_signature_defs`. By
# default, this map will contain a single Signature that defines the input
# tensors and output predictions, losses, and/or metrics (depending on the mode)
# The default keys used in the SignatureDef map are defined below.
SIGNATURE_KEY_MAP = mode_keys.ModeKeyMap(**{
    ModeKeys.PREDICT: signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
    ModeKeys.TRAIN: signature_constants.DEFAULT_TRAIN_SIGNATURE_DEF_KEY,
    ModeKeys.TEST: signature_constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY})

# Default names used in the SignatureDef input map, which maps strings to
# TensorInfo protos.
SINGLE_FEATURE_DEFAULT_NAME = 'feature'
SINGLE_RECEIVER_DEFAULT_NAME = 'input'
SINGLE_LABEL_DEFAULT_NAME = 'label'

### Below utilities are specific to SavedModel exports.


def _must_be_fed(op):
  # [FUNCTION DOC]
  # HELPER: predicate used by the servability check to find operations that need external input.
  # Takes a graph operation and returns true when its type is a placeholder, false otherwise.
  return op.type == 'Placeholder'


def _ensure_servable(input_tensors, names_to_output_tensor_infos):
  # [FUNCTION DOC]
  # HELPER: validation step used by the signature-definition builder.
  # Takes an iterable of signature input tensors (composite tensors are expanded into their
  # components) and a mapping from output names to tensor-info messages. Finds the single graph of
  # the inputs, resolves the outputs in that graph, walks backward from the outputs stopping at
  # the inputs, and checks every placeholder reached is among the fed inputs. Returns nothing.
  # Raises a value error naming the inputs, output keys, offending operation and a dependency
  # path if an output depends on a placeholder that is not provided as an input.
  """Check that the signature outputs don't depend on unreachable placeholders.

  Args:
    input_tensors: An iterable of `Tensor`s specified as the signature's inputs.
    names_to_output_tensor_infos: An mapping from output names to respective
      `TensorInfo`s corresponding to the signature's output tensors.

  Raises:
    ValueError: If any of the signature's outputs depend on placeholders not
      provided as signature's inputs.
  """
  plain_input_tensors = nest.flatten(input_tensors, expand_composites=True)

  graph = op_selector.get_unique_graph(plain_input_tensors)

  output_tensors = [
      utils.get_tensor_from_tensor_info(tensor, graph=graph)
      for tensor in names_to_output_tensor_infos.values()
  ]
  plain_output_tensors = nest.flatten(output_tensors, expand_composites=True)

  dependency_ops = op_selector.get_backward_walk_ops(
      plain_output_tensors, stop_at_ts=plain_input_tensors)

  fed_tensors = object_identity.ObjectIdentitySet(plain_input_tensors)
  for dependency_op in dependency_ops:
    if _must_be_fed(dependency_op) and (not all(
        output in fed_tensors for output in dependency_op.outputs)):
      input_tensor_names = [tensor.name for tensor in plain_input_tensors]
      output_tensor_keys = list(names_to_output_tensor_infos.keys())
      output_tensor_names = [tensor.name for tensor in plain_output_tensors]
      dependency_path = op_selector.show_path(dependency_op,
                                              plain_output_tensors,
                                              plain_input_tensors)
      raise ValueError(
          f'The signature\'s input tensors {input_tensor_names} are '
          f'insufficient to compute its output keys {output_tensor_keys} '
          f'(respectively, tensors {output_tensor_names}) because of the '
          f'dependency on `{dependency_op.name}` which is not given as '
          'a signature input, as illustrated by the following dependency path: '
          f'{dependency_path}')


def build_all_signature_defs(receiver_tensors,
                             export_outputs,
                             receiver_tensors_alternatives=None,
                             serving_only=True):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns "signature_def representing all passed args", but
  # the function returns a mapping from signature name to signature definition.
  # Builds the mapping of signature names to signature definitions for all export outputs.
  # receiver_tensors: a tensor or mapping of names to tensors; a bare tensor is wrapped under the
  # default receiver name. export_outputs: required mapping of keys to export output objects;
  # a null value or non-mapping raises a value error. receiver_tensors_alternatives: optional
  # mapping of alternative names to a tensor or mapping of tensors; each non-mapping value is
  # wrapped under the default receiver name. serving_only: default true; when true, signatures
  # failing the serving validity check are dropped from the result.
  # For each output key (a falsy key is named None) the signature is created from the receivers,
  # checked for servability, and stored; a value error from either step excludes that signature
  # and records the message. Alternatives produce entries named alternative-name colon output-key.
  # Side effect: logs a report of included and excluded signatures via the internal reporter.
  # Returns the mapping of signature names to signature definitions.
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
    raise ValueError('`export_outputs` must be a dict. Received '
                     f'{export_outputs} with type '
                     f'{type(export_outputs).__name__}.')

  signature_def_map = {}
  excluded_signatures = {}
  input_tensors = receiver_tensors.values()
  for output_key, export_output in export_outputs.items():
    signature_name = '{}'.format(output_key or 'None')
    try:
      signature = export_output.as_signature_def(receiver_tensors)
      _ensure_servable(input_tensors, signature.outputs)
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
      alt_input_tensors = receiver_tensors_alt.values()
      for output_key, export_output in export_outputs.items():
        signature_name = '{}:{}'.format(receiver_name or 'None', output_key or
                                        'None')
        try:
          signature = export_output.as_signature_def(receiver_tensors_alt)
          _ensure_servable(alt_input_tensors, signature.outputs)
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
    signature_constants.SUPERVISED_TRAIN_METHOD_NAME: 'Train',
    signature_constants.SUPERVISED_EVAL_METHOD_NAME: 'Eval',
}


def _log_signature_report(signature_def_map, excluded_signatures):
  # [FUNCTION DOC]
  # HELPER: logging step of the signature-definition builder.
  # Takes the mapping of included signature names to definitions and the mapping of excluded
  # signature names to their error messages. Groups included names by method name (always
  # listing the five standard methods even if empty, using friendly names), logs one info line
  # per method, logs the excluded signatures with reasons, and warns when no signatures exist or
  # when the default serving key is missing. Returns nothing; only emits log messages.
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
    logging.warn('Export includes no signatures!')
  elif (signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY not in
        signature_def_map):
    logging.warn('Export includes no default signature!')


# When we create a timestamped directory, there is a small chance that the
# directory already exists because another process is also creating these
# directories. In this case we just wait one second to get a new timestamp and
# try again. If this fails several times in a row, then something is seriously
# wrong.
MAX_DIRECTORY_CREATION_ATTEMPTS = 10


def get_timestamped_export_dir(export_dir_base):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a path, but the result is a byte string, because
  # the base and the timestamp are converted to bytes before joining.
  # Builds (without creating) a new subdirectory path under the base directory named by the current
  # time in whole seconds since the epoch. If that path already exists it sleeps one second and
  # retries, logging a warning each time, up to the module maximum of attempts (ten).
  # Input: export_dir_base, a string or bytes path. Returns the byte-string path. Raises a runtime
  # error if no unique name was obtained within the allowed attempts.
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

    result_dir = file_io.join(
        compat.as_bytes(export_dir_base), compat.as_bytes(str(timestamp)))
    if not gfile.Exists(result_dir):
      # Collisions are still possible (though extremely unlikely): this
      # directory is not actually created yet, but it will be almost
      # instantly on return from this function.
      return result_dir
    time.sleep(1)
    attempts += 1
    logging.warn('Directory {} already exists; retrying (attempt {}/{})'.format(
        compat.as_str(result_dir), attempts, MAX_DIRECTORY_CREATION_ATTEMPTS))
  raise RuntimeError('Failed to obtain a unique export directory name after '
                     f'{MAX_DIRECTORY_CREATION_ATTEMPTS} attempts.')


def get_temp_export_dir(timestamped_export_dir):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring implies a plain path is returned, but the result is a byte string
  # because the inputs are converted to bytes before joining.
  # Splits the given timestamped export directory into parent and final component (decoding a
  # byte-string component as UTF-8) and returns a sibling path whose final component is the
  # original one prefixed with temp-. Pure path manipulation: no
  # file system access and no exceptions of its own.
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
  temp_export_dir = file_io.join(
      compat.as_bytes(dirname), compat.as_bytes('temp-{}'.format(str_name)))
  return temp_export_dir


def export_outputs_for_mode(
    mode, serving_export_outputs=None, predictions=None, loss=None,
    metrics=None):
  # [FUNCTION DOC]
  # Builds the mapping of signature key to export output appropriate for a mode.
  # Inputs: mode, one of the mode keys (predict, train, test); serving_export_outputs, optional
  # mapping used only in predict mode; predictions, loss and metrics, optional tensors or mappings
  # (metrics as value/update-op pairs). In predict mode it delegates to the validating helper that
  # supplies a default prediction output when none is given. In train mode it returns a single
  # entry with a train output, otherwise a single entry with an eval output, keyed by the default
  # signature key for the mode.
  # Raises a value error when the mode is not one of the known keys. No side effects.
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
        f'Export output type not found for `mode`: {mode}. Expected one of: '
        f'{list(SIGNATURE_KEY_MAP.keys())}.')
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
  # Validates the given export outputs or creates the default ones.
  # Inputs: export_outputs, a mapping of keys to export output objects or nothing; predictions,
  # a tensor or mapping of tensors used to build a default prediction output when export_outputs
  # is nothing. Raises a type error if export_outputs is not a mapping or any value is not an
  # export output object. Then ensures a default serving key exists via the helper (which may
  # add an alias entry to, or raise for, the given mapping). Returns the mapping, which is the
  # same object as the argument when one was given (so the argument may be mutated).
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
    raise TypeError(
        f'`export_outputs` must be dict, received: {export_outputs}.')
  for v in export_outputs.values():
    if not isinstance(v, export_output_lib.ExportOutput):
      raise TypeError(
          'Values in `export_outputs` must be ExportOutput objects, '
          f'received: {export_outputs}.')

  _maybe_add_default_serving_output(export_outputs)

  return export_outputs


def _maybe_add_default_serving_output(export_outputs):
  # [FUNCTION DOC]
  # HELPER: normalization step for the export-output validation function.
  # KNOWN ISSUE: the error message for multiple outputs lacks a space before the quoted key name
  # (the strings are concatenated without one).
  # Takes a mapping of keys to export outputs and mutates it in place: if it has exactly one entry
  # under a non-default key, the same output is also registered under the default serving key.
  # If it has several entries and none uses the default serving key, raises a value error.
  # Returns the same (possibly modified) mapping.
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
          'Multiple `export_outputs` were provided, but none of them are '
          'specified as the default. Use'
          '`tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY` to '
          'specify a default.')

  return export_outputs
