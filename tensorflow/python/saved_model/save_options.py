# [MODULE DOC]
# Defines the public option objects for saving SavedModels: the variable policy enumeration and
# the save options container, both exported to the public saved_model API through the export
# decorator (import-time registration). Also holds a private validator for the namespace
# whitelist and a module flag marking the open-source build (rewritten by an internal tool).
# Consumed by the saving code and the save context.
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
"""Options for saving SavedModels."""

import enum

from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


is_oss = True  # Updated by copybara.


@tf_export("saved_model.experimental.VariablePolicy")
class VariablePolicy(enum.Enum):
  # [CLASS DOC]
  # Enumeration of how variables are treated on save: no policy (null value), save variable
  # devices, or expand distributed variables into components. Members carry a string value
  # (or nothing for the default). Public via export decorator; created by conversion from user
  # input with the from_obj helper.
  """Enum defining options for variable handling when saving.

  NONE
    No policy applied: Distributed variables are saved as one variable, with no
    device attached.

  SAVE_VARIABLE_DEVICES
    When saving variables, also save their device assignment.
    This is useful if one wants to hardcode devices in saved models, but it also
    makes them non-portable if soft device placement is disabled (more details
    in `tf.config.set_soft_device_placement`). This is currently not
    fully supported by `saved_model.load`, and is mainly intended to be used
    when one will be reading the saved model at a lower API level. In the
    example below, the graph saved by the call to `saved_model.save` will have
    the variable devices correctly specified:
    ```python
    exported = tf.train.Checkpoint()
    with tf.device('/GPU:0'):
      exported.x_gpu = tf.Variable(1.0)
    with tf.device('/CPU:0'):
      exported.x_cpu = tf.Variable(1.0)
    tf.saved_model.save(exported, export_dir,
        options = tf.saved_model.SaveOptions(
            experimental_variable_policy=
              tf.saved_model.experimental.VariablePolicy.SAVE_VARIABLE_DEVICES))
    ```
    Distributed variables are still saved as one variable under this policy.

  EXPAND_DISTRIBUTED_VARIABLES
    Distributed variables will be saved with information about their components,
    allowing for their restoration on load. Also, the saved graph will contain
    references to those variables. This is useful when one wants to use the
    model for training in environments where the original distribution strategy
    is not available.
  """

  NONE = None

  SAVE_VARIABLE_DEVICES = "save_variable_devices"

  EXPAND_DISTRIBUTED_VARIABLES = "expand_distributed_variables"

  def _save_variable_devices(self):
    # [FUNCTION DOC]
    # HELPER: used by the saving code to decide whether to record variable devices.
    # Returns true for any policy other than the default none policy (so also for expanding
    # distributed variables), false for none.
    """Checks whether variable devices should be saved."""
    return self != VariablePolicy.NONE

  def _expand_distributed_variables(self):
    # [FUNCTION DOC]
    # HELPER: used by the saving code to decide whether distributed variables get component info.
    # Returns true only for the expand-distributed-variables policy, false otherwise.
    """Checks whether distributed variables should be expanded."""
    return self == VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES

  @staticmethod
  def from_obj(obj):
    # [FUNCTION DOC]
    # Static converter. Input: nothing, a policy member, or a string (any case; non-strings are
    # turned into text first). Returns the matching policy: none for nothing, the same member if
    # already a policy, otherwise the member whose string value equals the lower-cased text.
    # Raises a value error for unrecognized input. No side effects.
    """Tries to convert `obj` to a VariablePolicy instance."""
    if obj is None:
      return VariablePolicy.NONE
    if isinstance(obj, VariablePolicy):
      return obj
    key = str(obj).lower()
    for policy in VariablePolicy:
      if key == policy.value:
        return policy
    raise ValueError(f"Received invalid VariablePolicy value: {obj}.")


@tf_export("saved_model.SaveOptions")
class SaveOptions:
  # [CLASS DOC]
  # Plain container of user-selectable settings for saving a SavedModel (namespace whitelist,
  # debug info, function aliases, debug stripper, IO device, variable policy, custom gradients,
  # image format, skip-saver, sharding callback, extra tags). Attributes are fixed by a slots
  # declaration so no other attributes can be added. Created by users and passed as the options
  # argument of the save functions; stored in the save context during a save.
  """Options for saving to SavedModel.

  This function may be used in the `options` argument in functions that
  save a SavedModel (`tf.saved_model.save`, `tf.keras.models.save_model`).
  """

  # Define object attributes in __slots__ for improved memory and performance.
  __slots__ = (
      "namespace_whitelist",
      "save_debug_info",
      "function_aliases",
      "experimental_debug_stripper",
      "experimental_io_device",
      "experimental_variable_policy",
      "experimental_custom_gradients",
      "experimental_image_format",
      "experimental_skip_saver",
      "experimental_sharding_callback",
      "extra_tags",
  )

  def __init__(
      self,
      namespace_whitelist=None,
      save_debug_info=False,
      function_aliases=None,
      experimental_debug_stripper=False,
      experimental_io_device=None,
      experimental_variable_policy=None,
      experimental_custom_gradients=True,
      experimental_image_format=False,
      experimental_skip_saver=False,
      experimental_sharding_callback=None,
      extra_tags=None,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message for a wrong sharding callback type is built from adjacent
    # string pieces without spaces, so it reads "optionmust" and "providedwas".
    # Stores all options as attributes. Inputs (all optional): namespace whitelist (list of
    # strings or nothing, validated), debug-info flag, function aliases (mapping; nothing becomes
    # an empty mapping), debug stripper flag, IO device string, variable policy (converted to a
    # policy member), custom gradients flag (default true), image format flag, skip-saver flag,
    # sharding callback (must be a sharding callback instance or nothing), extra tags (stored
    # as given). Raises a value error if the image format is enabled in the open-source build
    # or the sharding callback has the wrong type; the whitelist validator may raise type or
    # value errors. Returns nothing.
    """Creates an object that stores options for SavedModel saving.

    Args:
      namespace_whitelist: List of strings containing op namespaces to whitelist
        when saving a model. Saving an object that uses namespaced ops must
        explicitly add all namespaces to the whitelist. The namespaced ops must
        be registered into the framework when loading the SavedModel. If no
        whitelist is provided, all namespaced ops will be allowed.
      save_debug_info: Boolean indicating whether debug information is saved. If
        True, then a debug/saved_model_debug_info.pb file will be written with
        the contents of a GraphDebugInfo binary protocol buffer containing stack
        trace information for all ops and functions that are saved.
      function_aliases: Python dict. Mapping from string to object returned by
        @tf.function. A single tf.function can generate many ConcreteFunctions.
        If a downstream tool wants to refer to all concrete functions generated
        by a single tf.function you can use the `function_aliases` argument to
        store a map from the alias name to all concrete function names. E.g. >>>
        class Adder(tf.Module): ...   @tf.function ...   def double(self, x):
        ...     return x + x  >>> model = Adder() >>>
        model.double.get_concrete_function( ...   tf.TensorSpec(shape=[],
        dtype=tf.float32, name="float_input")) >>>
        model.double.get_concrete_function( ...   tf.TensorSpec(shape=[],
        dtype=tf.string, name="string_input"))  >>> options =
        tf.saved_model.SaveOptions( ...   function_aliases={'double':
        model.double}) >>> tf.saved_model.save(model, '/tmp/adder',
        options=options)
      experimental_debug_stripper: bool. If set to True, this strips the debug
        nodes from the graph, from both the nodes and the function defs. Note
        that this currently only strips the `Assert` nodes from the graph and
        converts them into `NoOp`s instead.
      experimental_io_device: string. Applies in a distributed setting.
        Tensorflow device to use to access the filesystem. If `None` (default)
        then for each variable the filesystem is accessed from the CPU:0 device
        of the host where that variable is assigned. If specified, the
        filesystem is instead accessed from that device for all variables.  This
        is for example useful if you want to save to a local directory, such as
        "/tmp" when running in a distributed setting. In that case pass a device
        for the host where the "/tmp" directory is accessible.
      experimental_variable_policy: The policy to apply to variables when
        saving. This is either a `saved_model.experimental.VariablePolicy` enum
        instance or one of its value strings (case is not important). See that
        enum documentation for details. A value of `None` corresponds to the
        default policy.
      experimental_custom_gradients: Boolean. When True, will save traced
        gradient functions for the functions decorated by `tf.custom_gradient`.
        Defaults to `True`.
      experimental_image_format: New (highly) experimental format that is
        capable of saving models larger than the 2GB protobuf limit. Enabling
        this option will likely break compatibility with downstream consumers.
        This option is currently disabled in OSS.
      experimental_skip_saver: If True, will prevent SavedModel from creating
        its native checkpointing ops - this is for models that do not use
        SavedModel's native checkpointing functionality to avoid the costs
        associated with creating and serializing those ops.
      experimental_sharding_callback: `tf.train.experimental.ShardingCallback`.
        A pre-made or custom callback that determines how checkpoints are
        sharded on disk. Pre-made callback options are
        `tf.train.experimental.ShardByDevicePolicy` and
        `tf.train.experimental.MaxShardSizePolicy`. You may also write a custom
        callback, see `tf.train.experimental.ShardingCallback`.
      extra_tags: Extra tags to be saved with the MetaGraph in the SavedModel.
    """
    self.namespace_whitelist = _validate_namespace_whitelist(
        namespace_whitelist
    )
    self.save_debug_info = save_debug_info
    self.function_aliases = function_aliases if function_aliases else dict()
    self.experimental_custom_gradients = experimental_custom_gradients
    self.experimental_debug_stripper = experimental_debug_stripper
    self.experimental_io_device = experimental_io_device
    self.experimental_variable_policy = VariablePolicy.from_obj(
        experimental_variable_policy
    )
    self.experimental_skip_saver = experimental_skip_saver

    # TODO(b/277279153): Enable image format in OSS after proto splitter is
    #  public.
    if experimental_image_format and is_oss:
      raise ValueError(
          "The option `experimental_image_format` is disabled in OSS."
      )
    self.experimental_image_format = experimental_image_format

    if experimental_sharding_callback is not None:
      if not isinstance(
          experimental_sharding_callback, sharding_util.ShardingCallback
      ):
        raise ValueError(
            "The experimental_sharding_callback checkpoint option"
            "must be of type ShardingCallback. The option provided"
            f"was of type {type(experimental_sharding_callback)}."
        )
    self.experimental_sharding_callback = experimental_sharding_callback
    self.extra_tags = extra_tags


def _validate_namespace_whitelist(namespace_whitelist):
  # [FUNCTION DOC]
  # HELPER: argument validation for the save options constructor.
  # Input: nothing or a list of strings. Returns nothing for nothing, otherwise a new list of
  # the entries converted to text. Raises a type error if the input is not a list and a value
  # error if any entry is not a string.
  """Validates namespace whitelist argument."""
  if namespace_whitelist is None:
    return None
  if not isinstance(namespace_whitelist, list):
    raise TypeError(
        "`namespace_whitelist` must be a list of strings. Got: "
        f"{namespace_whitelist} with type "
        f"{type(namespace_whitelist)}."
    )

  processed = []
  for namespace in namespace_whitelist:
    if not isinstance(namespace, str):
      raise ValueError(
          "Whitelisted namespace must be a string. Got: "
          f"{namespace} of type {type(namespace)}."
      )
    processed.append(compat.as_str(namespace))
  return processed
