# [MODULE DOC]
# Implementation of SignatureDef helper functions for SavedModel. Provides builders that create
# SignatureDef protocol buffers from tensors (generic, regression, classification, prediction,
# supervised train and eval), validators that decide whether a signature is servable by TensorFlow
# Serving, and helpers to create an op-based signature and load the op back from a graph. Public
# functions are exported under the v1 saved_model API names, some with deprecated endpoints. It is
# re-exported by signature_def_utils. Importing it has no side effects besides API registration.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""SignatureDef utility functions implementation."""


from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils_impl as utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export(
    v1=[
        'saved_model.build_signature_def',
        'saved_model.signature_def_utils.build_signature_def'
    ])
@deprecation.deprecated_endpoints(
    'saved_model.signature_def_utils.build_signature_def')
def build_signature_def(
    inputs=None, outputs=None, method_name=None, defaults=None
):
  # [FUNCTION DOC]
  # Builds a SignatureDef protocol buffer. inputs and outputs are optional mappings from names to
  # tensor-info protos and are copied into the result; method_name is an optional string; defaults
  # is an optional mapping from argument names to tensors. Eager tensors are converted to constant
  # tensor protos from their numpy value; graph tensors are accepted only when produced by a
  # constant operation, whose value attribute is copied. Returns the new SignatureDef.
  # Raises a value error if a default is a non-constant graph tensor. No other side effects.
  """Utility function to build a SignatureDef protocol buffer.

  Args:
    inputs: Inputs of the SignatureDef defined as a proto map of string to
      tensor info.
    outputs: Outputs of the SignatureDef defined as a proto map of string to
      tensor info.
    method_name: Method name of the SignatureDef as a string.
    defaults: Defaults of the SignatureDef defined as a proto map of string to
      TensorProto.

  Returns:
    A SignatureDef protocol buffer constructed based on the supplied arguments.
  """
  signature_def = meta_graph_pb2.SignatureDef()
  if inputs is not None:
    for item in inputs:
      signature_def.inputs[item].CopyFrom(inputs[item])
  if outputs is not None:
    for item in outputs:
      signature_def.outputs[item].CopyFrom(outputs[item])
  if method_name is not None:
    signature_def.method_name = method_name
  if defaults is not None:
    for arg_name, default in defaults.items():
      if isinstance(default, ops.EagerTensor):
        signature_def.defaults[arg_name].CopyFrom(
            tensor_util.make_tensor_proto(default.numpy())
        )
      else:
        if default.op.type == 'Const':
          signature_def.defaults[arg_name].CopyFrom(
              default.op.get_attr('value')
          )
        else:
          raise ValueError(
              f'Unable to convert object {str(default)} of type {type(default)}'
              ' to TensorProto.'
          )
  return signature_def


@tf_export(
    v1=[
        'saved_model.regression_signature_def',
        'saved_model.signature_def_utils.regression_signature_def'
    ])
@deprecation.deprecated_endpoints(
    'saved_model.signature_def_utils.regression_signature_def')
def regression_signature_def(examples, predictions):
  # [FUNCTION DOC]
  # Builds a regression signature for TensorFlow Serving from an examples tensor and a predictions
  # tensor. Validates that examples is not null, is a tensor, and has string dtype, and that
  # predictions is not null and has float dtype. Returns a SignatureDef with the regress input and
  # output keys and the regress method name. Raises a value error on any failed check (the
  # docstring mentions only the null examples case; type checks also raise).
  """Creates regression signature from given examples and predictions.

  This function produces signatures intended for use with the TensorFlow Serving
  Regress API (tensorflow_serving/apis/prediction_service.proto), and so
  constrains the input and output types to those allowed by TensorFlow Serving.

  Args:
    examples: A string `Tensor`, expected to accept serialized tf.Examples.
    predictions: A float `Tensor`.

  Returns:
    A regression-flavored signature_def.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('Regression `examples` cannot be None.')
  if not isinstance(examples, tensor_lib.Tensor):
    raise ValueError('Expected regression `examples` to be of type Tensor. '
                     f'Found `examples` of type {type(examples)}.')
  if predictions is None:
    raise ValueError('Regression `predictions` cannot be None.')

  input_tensor_info = utils.build_tensor_info(examples)
  if input_tensor_info.dtype != types_pb2.DT_STRING:
    raise ValueError('Regression input tensors must be of type string. '
                     f'Found tensors with type {input_tensor_info.dtype}.')
  signature_inputs = {signature_constants.REGRESS_INPUTS: input_tensor_info}

  output_tensor_info = utils.build_tensor_info(predictions)
  if output_tensor_info.dtype != types_pb2.DT_FLOAT:
    raise ValueError('Regression output tensors must be of type float. '
                     f'Found tensors with type {output_tensor_info.dtype}.')
  signature_outputs = {signature_constants.REGRESS_OUTPUTS: output_tensor_info}

  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.REGRESS_METHOD_NAME)

  return signature_def


@tf_export(
    v1=[
        'saved_model.classification_signature_def',
        'saved_model.signature_def_utils.classification_signature_def'
    ])
@deprecation.deprecated_endpoints(
    'saved_model.signature_def_utils.classification_signature_def')
def classification_signature_def(examples, classes, scores):
  # [FUNCTION DOC]
  # Builds a classification signature for TensorFlow Serving. examples must be a string tensor;
  # classes (string tensor) and scores (float tensor) are each optional but not both null. Returns a
  # SignatureDef with the classify input key, the classes and/or scores output keys and the classify
  # method name. Raises a value error when examples is null or not a tensor, both outputs are null,
  # or any dtype is wrong.
  # KNOWN ISSUE: the classes dtype error message ends with a stray backtick character.
  """Creates classification signature from given examples and predictions.

  This function produces signatures intended for use with the TensorFlow Serving
  Classify API (tensorflow_serving/apis/prediction_service.proto), and so
  constrains the input and output types to those allowed by TensorFlow Serving.

  Args:
    examples: A string `Tensor`, expected to accept serialized tf.Examples.
    classes: A string `Tensor`.  Note that the ClassificationResponse message
      requires that class labels are strings, not integers or anything else.
    scores: a float `Tensor`.

  Returns:
    A classification-flavored signature_def.

  Raises:
    ValueError: If examples is `None`.
  """
  if examples is None:
    raise ValueError('Classification `examples` cannot be None.')
  if not isinstance(examples, tensor_lib.Tensor):
    raise ValueError('Classification `examples` must be a string Tensor. '
                     f'Found `examples` of type {type(examples)}.')
  if classes is None and scores is None:
    raise ValueError('Classification `classes` and `scores` cannot both be '
                     'None.')

  input_tensor_info = utils.build_tensor_info(examples)
  if input_tensor_info.dtype != types_pb2.DT_STRING:
    raise ValueError('Classification input tensors must be of type string. '
                     f'Found tensors of type {input_tensor_info.dtype}')
  signature_inputs = {signature_constants.CLASSIFY_INPUTS: input_tensor_info}

  signature_outputs = {}
  if classes is not None:
    classes_tensor_info = utils.build_tensor_info(classes)
    if classes_tensor_info.dtype != types_pb2.DT_STRING:
      raise ValueError('Classification classes must be of type string Tensor. '
                       f'Found tensors of type {classes_tensor_info.dtype}.`')
    signature_outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES] = (
        classes_tensor_info)
  if scores is not None:
    scores_tensor_info = utils.build_tensor_info(scores)
    if scores_tensor_info.dtype != types_pb2.DT_FLOAT:
      raise ValueError('Classification scores must be a float Tensor.')
    signature_outputs[signature_constants.CLASSIFY_OUTPUT_SCORES] = (
        scores_tensor_info)

  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.CLASSIFY_METHOD_NAME)

  return signature_def


@tf_export(
    v1=[
        'saved_model.predict_signature_def',
        'saved_model.signature_def_utils.predict_signature_def'
    ])
@deprecation.deprecated_endpoints(
    'saved_model.signature_def_utils.predict_signature_def')
def predict_signature_def(inputs, outputs):
  # [FUNCTION DOC]
  # Builds a prediction signature for TensorFlow Serving from a mapping of names to input tensors
  # and a mapping of names to output tensors, converting each tensor to tensor info. Both mappings
  # must be non-empty. Returns a SignatureDef with the predict method name. Raises a value error
  # if either mapping is null or empty.
  """Creates prediction signature from given inputs and outputs.

  This function produces signatures intended for use with the TensorFlow Serving
  Predict API (tensorflow_serving/apis/prediction_service.proto). This API
  imposes no constraints on the input and output types.

  Args:
    inputs: dict of string to `Tensor`.
    outputs: dict of string to `Tensor`.

  Returns:
    A prediction-flavored signature_def.

  Raises:
    ValueError: If inputs or outputs is `None`.
  """
  if inputs is None or not inputs:
    raise ValueError('Prediction `inputs` cannot be None or empty.')
  if outputs is None or not outputs:
    raise ValueError('Prediction `outputs` cannot be None or empty.')

  signature_inputs = {key: utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}
  signature_outputs = {key: utils.build_tensor_info(tensor)
                       for key, tensor in outputs.items()}

  signature_def = build_signature_def(
      signature_inputs, signature_outputs,
      signature_constants.PREDICT_METHOD_NAME)

  return signature_def


def supervised_train_signature_def(
    inputs, loss, predictions=None, metrics=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: has no docstring; loss is a required parameter here although the shared helper
  # treats it as optional.
  # Builds a supervised training signature by delegating to the shared supervised helper with the
  # supervised training method name. inputs and loss are required mappings of names to tensors;
  # predictions and metrics are optional mappings. Returns a SignatureDef; raises a value error
  # if inputs is null or empty.
  return _supervised_signature_def(
      signature_constants.SUPERVISED_TRAIN_METHOD_NAME, inputs, loss=loss,
      predictions=predictions, metrics=metrics)


def supervised_eval_signature_def(
    inputs, loss, predictions=None, metrics=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: has no docstring.
  # Builds a supervised evaluation signature by delegating to the shared supervised helper with the
  # supervised eval method name. inputs and loss are required mappings of names to tensors;
  # predictions and metrics are optional mappings. Returns a SignatureDef; raises a value error if
  # inputs is null or empty.
  return _supervised_signature_def(
      signature_constants.SUPERVISED_EVAL_METHOD_NAME, inputs, loss=loss,
      predictions=predictions, metrics=metrics)


def _supervised_signature_def(
    method_name, inputs, loss=None, predictions=None,
    metrics=None):
  # [FUNCTION DOC]
  # HELPER: shared implementation for the supervised train and eval signature builders.
  # KNOWN ISSUE: the docstring says it raises when inputs or outputs is null, but only inputs is
  # checked; loss, predictions and metrics may all be null, giving a signature with no outputs.
  # Builds a SignatureDef from method_name (string), inputs (required non-empty mapping of names to
  # tensors) and optional loss, predictions and metrics mappings, whose entries are merged into one
  # outputs mapping (later sets overwrite same-named keys). Raises a value error if inputs is null
  # or empty. Returns the SignatureDef.
  """Creates a signature for training and eval data.

  This function produces signatures that describe the inputs and outputs
  of a supervised process, such as training or evaluation, that
  results in loss, metrics, and the like. Note that this function only requires
  inputs to be not None.

  Args:
    method_name: Method name of the SignatureDef as a string.
    inputs: dict of string to `Tensor`.
    loss: dict of string to `Tensor` representing computed loss.
    predictions: dict of string to `Tensor` representing the output predictions.
    metrics: dict of string to `Tensor` representing metric ops.

  Returns:
    A train- or eval-flavored signature_def.

  Raises:
    ValueError: If inputs or outputs is `None`.
  """
  if inputs is None or not inputs:
    raise ValueError(f'{method_name} `inputs` cannot be None or empty.')

  signature_inputs = {key: utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}

  signature_outputs = {}
  for output_set in (loss, predictions, metrics):
    if output_set is not None:
      sig_out = {key: utils.build_tensor_info(tensor)
                 for key, tensor in output_set.items()}
      signature_outputs.update(sig_out)

  signature_def = build_signature_def(
      signature_inputs, signature_outputs, method_name)

  return signature_def


@tf_export(
    v1=[
        'saved_model.is_valid_signature',
        'saved_model.signature_def_utils.is_valid_signature'
    ])
@deprecation.deprecated_endpoints(
    'saved_model.signature_def_utils.is_valid_signature')
def is_valid_signature(signature_def):
  # [FUNCTION DOC]
  # Returns whether a SignatureDef is servable by TensorFlow Serving, i.e. is a valid
  # classification, regression or predict signature. A null signature returns false.
  # Takes a SignatureDef proto; returns a boolean; no side effects.
  """Determine whether a SignatureDef can be served by TensorFlow Serving."""
  if signature_def is None:
    return False
  return (_is_valid_classification_signature(signature_def) or
          _is_valid_regression_signature(signature_def) or
          _is_valid_predict_signature(signature_def))


def _is_valid_predict_signature(signature_def):
  # [FUNCTION DOC]
  # HELPER: validity check used by the public servable-signature check.
  # Returns true if the SignatureDef has the predict method name and at least one input and one
  # output; otherwise false. Pure function of the proto.
  """Determine whether the argument is a servable 'predict' SignatureDef."""
  if signature_def.method_name != signature_constants.PREDICT_METHOD_NAME:
    return False
  if not signature_def.inputs.keys():
    return False
  if not signature_def.outputs.keys():
    return False
  return True


def _is_valid_regression_signature(signature_def):
  # [FUNCTION DOC]
  # HELPER: validity check used by the public servable-signature check.
  # Returns true only if the method name is the regress one, the inputs consist of exactly the
  # regress input key with string dtype, and the outputs consist of exactly the regress output key
  # with float dtype; otherwise false. Pure function of the proto.
  """Determine whether the argument is a servable 'regress' SignatureDef."""
  if signature_def.method_name != signature_constants.REGRESS_METHOD_NAME:
    return False

  if (set(signature_def.inputs.keys())
      != set([signature_constants.REGRESS_INPUTS])):
    return False
  if (signature_def.inputs[signature_constants.REGRESS_INPUTS].dtype !=
      types_pb2.DT_STRING):
    return False

  if (set(signature_def.outputs.keys())
      != set([signature_constants.REGRESS_OUTPUTS])):
    return False
  if (signature_def.outputs[signature_constants.REGRESS_OUTPUTS].dtype !=
      types_pb2.DT_FLOAT):
    return False

  return True


def _is_valid_classification_signature(signature_def):
  # [FUNCTION DOC]
  # HELPER: validity check used by the public servable-signature check.
  # Returns true only if the method name is the classify one, the inputs are exactly the classify
  # input key with string dtype, outputs are non-empty and limited to the classes and scores keys,
  # classes (if present) is string dtype and scores (if present) is float dtype; else false.
  """Determine whether the argument is a servable 'classify' SignatureDef."""
  if signature_def.method_name != signature_constants.CLASSIFY_METHOD_NAME:
    return False

  if (set(signature_def.inputs.keys())
      != set([signature_constants.CLASSIFY_INPUTS])):
    return False
  if (signature_def.inputs[signature_constants.CLASSIFY_INPUTS].dtype !=
      types_pb2.DT_STRING):
    return False

  allowed_outputs = set([signature_constants.CLASSIFY_OUTPUT_CLASSES,
                         signature_constants.CLASSIFY_OUTPUT_SCORES])

  if not signature_def.outputs.keys():
    return False
  if set(signature_def.outputs.keys()) - allowed_outputs:
    return False
  if (signature_constants.CLASSIFY_OUTPUT_CLASSES in signature_def.outputs
      and
      signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES].dtype
      != types_pb2.DT_STRING):
    return False
  if (signature_constants.CLASSIFY_OUTPUT_SCORES in signature_def.outputs
      and
      signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_SCORES].dtype !=
      types_pb2.DT_FLOAT):
    return False

  return True


def op_signature_def(op, key):
  # [FUNCTION DOC]
  # Creates a SignatureDef whose single output, stored under the given key (string), refers to the
  # given graph element by name (an operation, or possibly a tensor). Returns the SignatureDef.
  # No other side effects.
  """Creates a signature def with the output pointing to an op.

  Note that op isn't strictly enforced to be an Op object, and may be a Tensor.
  It is recommended to use the build_signature_def() function for Tensors.

  Args:
    op: An Op (or possibly Tensor).
    key: Key to graph element in the SignatureDef outputs.

  Returns:
    A SignatureDef with a single output pointing to the op.
  """
  # Use build_tensor_info_from_op, which creates a TensorInfo from the element's
  # name.
  return build_signature_def(outputs={key: utils.build_tensor_info_from_op(op)})


def load_op_from_signature_def(signature_def, key, import_scope=None):
  # [FUNCTION DOC]
  # Loads the graph element (operation or tensor) referenced by the output stored under key in a
  # SignatureDef, using the default graph and an optional import_scope prefix. Returns the element.
  # Raises a not-found error if the name is not present in the graph.
  """Load an Op from a SignatureDef created by op_signature_def().

  Args:
    signature_def: a SignatureDef proto
    key: string key to op in the SignatureDef outputs.
    import_scope: Scope used to import the op

  Returns:
    Op (or possibly Tensor) in the graph with the same name as saved in the
      SignatureDef.

  Raises:
    NotFoundError: If the op could not be found in the graph.
  """
  tensor_info = signature_def.outputs[key]
  try:
    # The init and train ops are not strictly enforced to be operations, so
    # retrieve any graph element (can be either op or tensor).
    return utils.get_element_from_tensor_info(
        tensor_info, import_scope=import_scope)
  except KeyError:
    raise errors.NotFoundError(
        None, None,
        f'The key "{key}" could not be found in the graph. Please make sure the'
        ' SavedModel was created by the internal _SavedModelBuilder. If you '
        'are using the public API, please make sure the SignatureDef in the '
        f'SavedModel does not contain the key "{key}".')
