# [MODULE DOC]
# Defines classes describing the outputs of a model that can be served: an abstract base class,
# classification, regression and prediction outputs, and a supervised base class with train and
# eval outputs. Each class validates and wraps its tensors and can convert itself into a
# signature definition for inclusion in a meta graph, using the signature definition utilities.
# Re-exported by the model_utils package. No import-time side effects.
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
"""Classes for different types of export output."""

import abc


from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import signature_def_utils


class ExportOutput:
  # [CLASS DOC]
  # KNOWN ISSUE: Uses the Python 2 style metaclass attribute, which has no effect in Python 3,
  # so the abstract method marker is not enforced and the class can be instantiated directly.
  # Abstract base for all servable model outputs. Subclasses implement conversion to a
  # signature definition given receiver tensors. Provides shared helpers to validate output keys
  # and to wrap raw tensors into a mapping of names to tensors. Holds a key separator character.
  """Represents an output of a model that can be served.

  These typically correspond to model heads.
  """

  __metaclass__ = abc.ABCMeta

  _SEPARATOR_CHAR = '/'

  @abc.abstractmethod
  def as_signature_def(self, receiver_tensors):
    # [FUNCTION DOC]
    # Abstract method: subclasses build and return a signature definition that uses the given
    # receiver tensors (a tensor or mapping of names to tensors) as inputs. The base version does
    # nothing and returns nothing.
    """Generate a SignatureDef proto for inclusion in a MetaGraphDef.

    The SignatureDef will specify outputs as described in this ExportOutput,
    and will use the provided receiver_tensors as inputs.

    Args:
      receiver_tensors: a `Tensor`, or a dict of string to `Tensor`, specifying
        input nodes that will be fed.
    """
    pass

  def _check_output_key(self, key, error_label):
    # [FUNCTION DOC]
    # HELPER: Key validation shared by the output wrappers.
    # Takes an output key and a label for error messages. A tuple key (multi-head models) is
    # joined into a single string with the separator character. Raises a value error if the
    # result is not a string. Returns the string key.
    # For multi-head models, the key can be a tuple.
    if isinstance(key, tuple):
      key = self._SEPARATOR_CHAR.join(key)

    if not isinstance(key, str):
      raise ValueError(
          '{} output key must be a string; got {}.'.format(error_label, key))
    return key

  def _wrap_and_check_outputs(
      self, outputs, single_output_default_name, error_label=None):
    # [FUNCTION DOC]
    # HELPER: Normalizes output tensors for subclasses.
    # Takes a tensor or mapping of keys to tensors, the key to use when a single raw tensor is
    # given, and an optional error label (defaults to that key). Builds a new mapping with
    # validated string keys (tuple keys joined). Raises a value error for bad keys or for values
    # that are not tensors. Returns the new mapping; the input is not modified.
    """Wraps raw tensors as dicts and checks type.

    Note that we create a new dict here so that we can overwrite the keys
    if necessary.

    Args:
      outputs: A `Tensor` or a dict of string to `Tensor`.
      single_output_default_name: A string key for use in the output dict
        if the provided `outputs` is a raw tensor.
      error_label: descriptive string for use in error messages. If none,
        single_output_default_name will be used.

    Returns:
      A dict of tensors

    Raises:
      ValueError: if the outputs dict keys are not strings or tuples of strings
        or the values are not Tensors.
    """
    if not isinstance(outputs, dict):
      outputs = {single_output_default_name: outputs}

    output_dict = {}
    for key, value in outputs.items():
      error_name = error_label or single_output_default_name
      key = self._check_output_key(key, error_name)
      if not isinstance(value, tensor.Tensor):
        raise ValueError(
            '{} output value must be a Tensor; got {}.'.format(
                error_name, value))

      output_dict[key] = value
    return output_dict


class ClassificationOutput(ExportOutput):
  # [CLASS DOC]
  # Export output for a classification head. Holds an optional scores tensor (floating point)
  # and an optional classes tensor (strings); at least one must be given. Converts to a
  # classification signature definition taking a single string examples input.
  """Represents the output of a classification head.

  Either classes or scores or both must be set.

  The classes `Tensor` must provide string labels, not integer class IDs.

  If only classes is set, it is interpreted as providing top-k results in
  descending order.

  If only scores is set, it is interpreted as providing a score for every class
  in order of class ID.

  If both classes and scores are set, they are interpreted as zipped, so each
  score corresponds to the class at the same index.  Clients should not depend
  on the order of the entries.
  """

  def __init__(self, scores=None, classes=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Error text and docstring say float32, but the check accepts any floating
    # point dtype for scores.
    # Takes optional scores (floating tensor) and optional classes (string tensor). Raises a
    # value error if scores is given but is not a floating tensor, if classes is given but is not
    # a string tensor, or if both are missing. Stores them privately.
    """Constructor for `ClassificationOutput`.

    Args:
      scores: A float `Tensor` giving scores (sometimes but not always
          interpretable as probabilities) for each class.  May be `None`, but
          only if `classes` is set.  Interpretation varies-- see class doc.
      classes: A string `Tensor` giving predicted class labels.  May be `None`,
          but only if `scores` is set.  Interpretation varies-- see class doc.

    Raises:
      ValueError: if neither classes nor scores is set, or one of them is not a
          `Tensor` with the correct dtype.
    """
    if (scores is not None
        and not (isinstance(scores, tensor.Tensor)
                 and scores.dtype.is_floating)):
      raise ValueError('Classification scores must be a float32 Tensor; '
                       'got {}'.format(scores))
    if (classes is not None
        and not (isinstance(classes, tensor.Tensor)
                 and dtypes.as_dtype(classes.dtype) == dtypes.string)):
      raise ValueError('Classification classes must be a string Tensor; '
                       'got {}'.format(classes))
    if scores is None and classes is None:
      raise ValueError('Cannot create a ClassificationOutput with empty '
                       'arguments. At least one of `scores` and `classes` '
                       'must be defined.')
    self._scores = scores
    self._classes = classes

  @property
  def scores(self):
    # [FUNCTION DOC]
    # Property returning the scores tensor, or nothing if not set.
    return self._scores

  @property
  def classes(self):
    # [FUNCTION DOC]
    # Property returning the classes tensor, or nothing if not set.
    return self._classes

  def as_signature_def(self, receiver_tensors):
    # [FUNCTION DOC]
    # Builds a classification signature definition. Takes the receiver tensors mapping, which
    # must contain exactly one entry whose tensor has string dtype; otherwise raises a value
    # error explaining how to structure the serving input. Returns the signature definition built
    # from that examples tensor, the classes and the scores.
    if len(receiver_tensors) != 1:
      raise ValueError(
          'Classification signatures can only accept a single tensor input of '
          'type tf.string. Please check to make sure that you have structured '
          'the serving_input_receiver_fn so that it creates a single string '
          'placeholder. If your model function expects multiple inputs, then '
          'use `tf.io.parse_example()` to parse the string into multiple '
          f'tensors.\n Received: {receiver_tensors}')
    (_, examples), = receiver_tensors.items()
    if dtypes.as_dtype(examples.dtype) != dtypes.string:
      raise ValueError(
          'Classification signatures can only accept a single tensor input of '
          'type tf.string. Please check to make sure that you have structured '
          'the serving_input_receiver_fn so that it creates a single string '
          'placeholder. If your model function expects multiple inputs, then '
          'use `tf.io.parse_example()` to parse the string into multiple '
          f'tensors.\n Received: {receiver_tensors}')
    return signature_def_utils.classification_signature_def(
        examples, self.classes, self.scores)


class RegressionOutput(ExportOutput):
  # [CLASS DOC]
  # Export output for a regression head. Holds one floating point value tensor and converts to a
  # regression signature definition taking a single string examples input.
  """Represents the output of a regression head."""

  def __init__(self, value):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Error text and docstring say float32, but any floating dtype is accepted.
    # Takes the required predicted values tensor. Raises a value error if it is not a floating
    # point tensor. Stores it privately.
    """Constructor for `RegressionOutput`.

    Args:
      value: a float `Tensor` giving the predicted values.  Required.

    Raises:
      ValueError: if the value is not a `Tensor` with dtype tf.float32.
    """
    if not (isinstance(value, tensor.Tensor) and value.dtype.is_floating):
      raise ValueError('Regression output value must be a float32 Tensor; '
                       'got {}'.format(value))
    self._value = value

  @property
  def value(self):
    # [FUNCTION DOC]
    # Property returning the stored predicted values tensor.
    return self._value

  def as_signature_def(self, receiver_tensors):
    # [FUNCTION DOC]
    # Builds a regression signature definition. The receiver tensors mapping must contain exactly
    # one entry whose tensor has string dtype, otherwise a value error is raised. Returns the
    # signature definition from that examples tensor and the stored value.
    if len(receiver_tensors) != 1:
      raise ValueError(
          'Regression signatures can only accept a single tensor input of '
          'type tf.string. Please check to make sure that you have structured '
          'the serving_input_receiver_fn so that it creates a single string '
          'placeholder. If your model function expects multiple inputs, then '
          'use `tf.io.parse_example()` to parse the string into multiple '
          f'tensors.\n Received: {receiver_tensors}')
    (_, examples), = receiver_tensors.items()
    if dtypes.as_dtype(examples.dtype) != dtypes.string:
      raise ValueError(
          'Regression signatures can only accept a single tensor input of '
          'type tf.string. Please check to make sure that you have structured '
          'the serving_input_receiver_fn so that it creates a single string '
          'placeholder. If your model function expects multiple inputs, then '
          'use `tf.io.parse_example()` to parse the string into multiple '
          f'tensors.\n Received: {receiver_tensors}')
    return signature_def_utils.regression_signature_def(examples, self.value)


class PredictOutput(ExportOutput):
  # [CLASS DOC]
  # Export output for a generic prediction head. Holds a mapping of output names to tensors
  # (a single raw tensor is stored under the default name "output") and converts to a predict
  # signature definition accepting arbitrary receiver tensors.
  """Represents the output of a generic prediction head.

  A generic prediction need not be either a classification or a regression.

  Named outputs must be provided as a dict from string to `Tensor`,
  """
  _SINGLE_OUTPUT_DEFAULT_NAME = 'output'

  def __init__(self, outputs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Docstring says a non-dict raises an error, but a single tensor is accepted and
    # wrapped under the default name.
    # Takes a tensor or a mapping of string keys to tensors, validates and wraps it (error label
    # "Prediction"), raising a value error for bad keys or non-tensor values, and stores it.
    """Constructor for PredictOutput.

    Args:
      outputs: A `Tensor` or a dict of string to `Tensor` representing the
        predictions.

    Raises:
      ValueError: if the outputs is not dict, or any of its keys are not
          strings, or any of its values are not `Tensor`s.
    """

    self._outputs = self._wrap_and_check_outputs(
        outputs, self._SINGLE_OUTPUT_DEFAULT_NAME, error_label='Prediction')

  @property
  def outputs(self):
    # [FUNCTION DOC]
    # Property returning the validated mapping of output names to tensors.
    return self._outputs

  def as_signature_def(self, receiver_tensors):
    # [FUNCTION DOC]
    # Takes the receiver tensors (tensor or mapping of names to tensors) and returns a predict
    # signature definition combining them with the stored outputs. No input checks are made here.
    return signature_def_utils.predict_signature_def(receiver_tensors,
                                                     self.outputs)


class _SupervisedOutput(ExportOutput):
  # [CLASS DOC]
  # HELPER: Shared base for the train and eval outputs.
  # KNOWN ISSUE: Uses the Python 2 style metaclass attribute, so the abstract marker on the
  # signature-function getter is not enforced in Python 3.
  # Holds optional loss, predictions and metrics mappings. Loss and predictions keys are prefixed
  # with their category name; metrics become value and update-op tensor entries. Subclasses
  # choose which supervised signature builder is used. Class constants hold the category names
  # and the value and update-op suffixes; class-level defaults for the three mappings are nothing.
  """Represents the output of a supervised training or eval process."""
  __metaclass__ = abc.ABCMeta

  LOSS_NAME = 'loss'
  PREDICTIONS_NAME = 'predictions'
  METRICS_NAME = 'metrics'

  METRIC_VALUE_SUFFIX = 'value'
  METRIC_UPDATE_SUFFIX = 'update_op'

  _loss = None
  _predictions = None
  _metrics = None

  def __init__(self, loss=None, predictions=None, metrics=None):
    # [FUNCTION DOC]
    # Takes optional loss, predictions (each a tensor or mapping of names to tensors) and optional
    # metrics (mapping of names to metric objects or value/update-op pairs, or a single pair).
    # Each provided argument is validated and stored with prefixed keys; metrics go through the
    # metrics wrapper. Raises a value error for non-string keys or non-tensor values. Arguments
    # left as nothing leave the corresponding stored attribute unset (nothing).
    """Constructor for SupervisedOutput (ie, Train or Eval output).

    Args:
      loss: dict of Tensors or single Tensor representing calculated loss.
      predictions: dict of Tensors or single Tensor representing model
        predictions.
      metrics: Dict of metric results keyed by name.
        The values of the dict can be one of the following:
        (1) instance of `Metric` class.
        (2) (metric_value, update_op) tuples, or a single tuple.
        metric_value must be a Tensor, and update_op must be a Tensor or Op.

    Raises:
      ValueError: if any of the outputs' dict keys are not strings or tuples of
        strings or the values are not Tensors (or Operations in the case of
        update_op).
    """

    if loss is not None:
      loss_dict = self._wrap_and_check_outputs(loss, self.LOSS_NAME)
      self._loss = self._prefix_output_keys(loss_dict, self.LOSS_NAME)
    if predictions is not None:
      pred_dict = self._wrap_and_check_outputs(
          predictions, self.PREDICTIONS_NAME)
      self._predictions = self._prefix_output_keys(
          pred_dict, self.PREDICTIONS_NAME)
    if metrics is not None:
      self._metrics = self._wrap_and_check_metrics(metrics)

  def _prefix_output_keys(self, output_dict, output_name):
    # [FUNCTION DOC]
    # HELPER: Key prefixing for loss and predictions.
    # Takes a validated mapping of names to tensors and a prefix name. Returns a new mapping in
    # which each key not already starting with the prefix gets the prefix and separator prepended;
    # values are unchanged.
    """Prepend output_name to the output_dict keys if it doesn't exist.

    This produces predictable prefixes for the pre-determined outputs
    of SupervisedOutput.

    Args:
      output_dict: dict of string to Tensor, assumed valid.
      output_name: prefix string to prepend to existing keys.

    Returns:
      dict with updated keys and existing values.
    """

    new_outputs = {}
    for key, val in output_dict.items():
      key = self._prefix_key(key, output_name)
      new_outputs[key] = val
    return new_outputs

  def _prefix_key(self, key, output_name):
    # [FUNCTION DOC]
    # HELPER: Prepends the output name and separator to a key unless the key already starts with
    # the output name. Note it is a plain starts-with test, so a key like "lossy" is left alone.
    # Returns the resulting key string.
    if key.find(output_name) != 0:
      key = output_name + self._SEPARATOR_CHAR + key
    return key

  def _wrap_and_check_metrics(self, metrics):
    # [FUNCTION DOC]
    # HELPER: Converts metrics into a flat mapping of tensors for the signature.
    # Takes a mapping of names to metric objects or (value, update_op) tuples; a non-mapping is
    # treated as a single entry under the name "metrics". For a metric object, the value is its
    # result and its single update op is used (an assertion fails if it does not have exactly one
    # update). Keys are validated and prefixed with "metrics"; outputs are stored under key/value
    # and key/update_op. Raises a value error if a value is not a tensor or an update op is not a
    # tensor or operation. Non-tensor update ops are wrapped by an empty constant created under a
    # control dependency on the op. Returns the flat mapping.
    """Handle the saving of metrics.

    Metrics is either a tuple of (value, update_op), or a dict of such tuples.
    Here, we separate out the tuples and create a dict with names to tensors.

    Args:
      metrics: Dict of metric results keyed by name.
        The values of the dict can be one of the following:
        (1) instance of `Metric` class.
        (2) (metric_value, update_op) tuples, or a single tuple.
        metric_value must be a Tensor, and update_op must be a Tensor or Op.

    Returns:
      dict of output_names to tensors

    Raises:
      ValueError: if the dict key is not a string, or the metric values or ops
        are not tensors.
    """
    if not isinstance(metrics, dict):
      metrics = {self.METRICS_NAME: metrics}

    outputs = {}
    for key, value in metrics.items():
      if isinstance(value, tuple):
        metric_val, metric_op = value
      else:  # value is a keras.Metrics object
        metric_val = value.result()
        assert len(value.updates) == 1  # We expect only one update op.
        metric_op = value.updates[0]
      key = self._check_output_key(key, self.METRICS_NAME)
      key = self._prefix_key(key, self.METRICS_NAME)

      val_name = key + self._SEPARATOR_CHAR + self.METRIC_VALUE_SUFFIX
      op_name = key + self._SEPARATOR_CHAR + self.METRIC_UPDATE_SUFFIX
      if not isinstance(metric_val, tensor.Tensor):
        raise ValueError(
            '{} output value must be a Tensor; got {}.'.format(
                key, metric_val))
      if not (tensor_util.is_tf_type(metric_op) or
              isinstance(metric_op, ops.Operation)):
        raise ValueError(
            '{} update_op must be a Tensor or Operation; got {}.'.format(
                key, metric_op))

      # We must wrap any ops (or variables) in a Tensor before export, as the
      # SignatureDef proto expects tensors only. See b/109740581
      metric_op_tensor = metric_op
      if not isinstance(metric_op, tensor.Tensor):
        with ops.control_dependencies([metric_op]):
          metric_op_tensor = constant_op.constant([], name='metric_op_wrapper')

      outputs[val_name] = metric_val
      outputs[op_name] = metric_op_tensor

    return outputs

  @property
  def loss(self):
    # [FUNCTION DOC]
    # Property returning the prefixed loss mapping, or nothing if none was given.
    return self._loss

  @property
  def predictions(self):
    # [FUNCTION DOC]
    # Property returning the prefixed predictions mapping, or nothing if none was given.
    return self._predictions

  @property
  def metrics(self):
    # [FUNCTION DOC]
    # Property returning the flattened metrics mapping, or nothing if none was given.
    return self._metrics

  @abc.abstractmethod
  def _get_signature_def_fn(self):
    # [FUNCTION DOC]
    # HELPER: Hook for subclasses. Abstract (though not enforced); should return the function
    # that builds a supervised signature definition. The base version returns nothing.
    """Returns a function that produces a SignatureDef given desired outputs."""
    pass

  def as_signature_def(self, receiver_tensors):
    # [FUNCTION DOC]
    # Takes the receiver tensors, obtains the subclass's signature builder and returns the
    # signature definition it produces from the receiver tensors, loss, predictions and metrics.
    signature_def_fn = self._get_signature_def_fn()
    return signature_def_fn(
        receiver_tensors, self.loss, self.predictions, self.metrics)


class TrainOutput(_SupervisedOutput):
  # [CLASS DOC]
  # Export output for a supervised training process; uses the supervised train signature builder.
  """Represents the output of a supervised training process.

  This class generates the appropriate signature def for exporting
  training output by type-checking and wrapping loss, predictions, and metrics
  values.
  """

  def _get_signature_def_fn(self):
    # [FUNCTION DOC]
    # Returns the function that builds a supervised train signature definition.
    return signature_def_utils.supervised_train_signature_def


class EvalOutput(_SupervisedOutput):
  # [CLASS DOC]
  # Export output for a supervised evaluation process; uses the supervised eval signature builder.
  """Represents the output of a supervised eval process.

  This class generates the appropriate signature def for exporting
  eval output by type-checking and wrapping loss, predictions, and metrics
  values.
  """

  def _get_signature_def_fn(self):
    # [FUNCTION DOC]
    # Returns the function that builds a supervised eval signature definition.
    return signature_def_utils.supervised_eval_signature_def
