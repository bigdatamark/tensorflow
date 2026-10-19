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
# [MODULE DOC]
# Defines CheckpointOptions (exported as the public train checkpoint-options class), a small
# slotted value object holding settings for saving/restoring checkpoints: IO device, async
# saving, write callbacks, sharding callback and slot-variable skipping. No import-time side
# effects beyond public API export registration.
# ==============================================================================
"""Options for saving Checkpoints."""

import copy
import inspect

from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export


@tf_export("train.CheckpointOptions")
class CheckpointOptions(object):
  # [CLASS DOC]
  # Plain options container passed as the options argument to checkpoint save/restore. Uses
  # fixed slots for the attributes: io device, (deprecated) async flag, write callbacks, async
  # flag, sharding callback, and skip-slot-variables flag. Registered as a public API symbol.
  """Options for constructing a Checkpoint.

  Used as the `options` argument to either `tf.train.Checkpoint.save()` or
  `tf.train.Checkpoint.restore()` methods to adjust how variables are
  saved/restored.

  Example: Run IO ops on "localhost" while saving a checkpoint:

  ```
  step = tf.Variable(0, name="step")
  checkpoint = tf.train.Checkpoint(step=step)
  options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
  checkpoint.save("/tmp/ckpt", options=options)
  ```
  """

  # Define object attributes in __slots__ for improved memory and performance.
  __slots__ = (
      "experimental_io_device",
      "experimental_enable_async_checkpoint",
      "experimental_write_callbacks",
      "enable_async",
      "experimental_sharding_callback",
      "experimental_skip_slot_variables",
  )

  @deprecated_args(
      None, "Use enable_async instead", "experimental_enable_async_checkpoint"
  )
  def __init__(
      self,
      experimental_io_device=None,
      experimental_enable_async_checkpoint=False,
      experimental_write_callbacks=None,
      enable_async=False,
      experimental_skip_slot_variables=False,
      experimental_sharding_callback=None
  ):
    # [FUNCTION DOC]
    # Stores the options. All inputs are optional: io device (string, default none), deprecated
    # async flag (boolean, default false), write callbacks (list of callables, default none; each
    # must accept at most one parameter, checked with an assertion), async flag (boolean, default
    # false), slot-variable skip flag (boolean, default false), and sharding callback (default
    # none; if given must be a ShardingCallback, otherwise raises a value error). The async
    # attributes are both set to true if either flag is true. Deprecated-args decorator warns when
    # the old async argument is used. Returns nothing.
    """Creates an object that stores options for a Checkpoint.

    Args:
      experimental_io_device: string. Applies in a distributed setting.
        Tensorflow device to use to access the filesystem. If `None` (default)
        then for each variable the filesystem is accessed from the CPU:0 device
        of the host where that variable is assigned. If specified, the
        filesystem is instead accessed from that device for all variables.  This
        is for example useful if you want to save to a local directory, such as
        "/tmp" when running in a distributed setting. In that case pass a device
        for the host where the "/tmp" directory is accessible.
      experimental_enable_async_checkpoint: bool Type. Deprecated, please use
        the enable_async option.
      experimental_write_callbacks: List[Callable]. A list of callback functions
        that will be executed after each saving event finishes (i.e. after
        `save()` or `write()`). For async checkpoint, the callbacks will be
        executed only after the async thread finishes saving.  The return values
        of the callback(s) will be ignored. The callback(s) can optionally take
        the `save_path` (the result of `save()` or `write()`) as an argument.
        The callbacks will be executed in the same order of this list after the
        checkpoint has been written.
      enable_async: bool Type. Indicates whether async checkpointing is enabled.
        Default is False, i.e., no async checkpoint.  Async checkpoint moves the
        checkpoint file writing off the main thread, so that the model can
        continue to train while the checkpoint file writing runs in the
        background. Async checkpoint reduces TPU device idle cycles and speeds
        up model training process, while memory consumption may increase.
      experimental_skip_slot_variables: bool Type. If true, ignores slot
        variables during restore. Context: TPU Embedding layers for Serving do
        not properly restore slot variables. This option is a way to omit
        restoring slot variables which are not required for Serving usecase
        anyways.(b/315912101)
      experimental_sharding_callback: `tf.train.experimental.ShardingCallback`.
        A pre-made or custom callback that determines how checkpoints are
        sharded on disk. Pre-made callback options are
        `tf.train.experimental.ShardByDevicePolicy` and
        `tf.train.experimental.MaxShardSizePolicy`. You may also write a custom
        callback, see `tf.train.experimental.ShardingCallback`.
    """
    self.experimental_io_device = experimental_io_device
    self.enable_async = experimental_enable_async_checkpoint or enable_async
    self.experimental_enable_async_checkpoint = self.enable_async
    # Ensure that each callback only has either 0 or 1 parameter
    if experimental_write_callbacks is not None:
      for callback in experimental_write_callbacks:
        assert len(inspect.signature(callback).parameters) <= 1
    self.experimental_write_callbacks = experimental_write_callbacks
    if experimental_sharding_callback is not None:
      if not isinstance(
          experimental_sharding_callback, sharding_util.ShardingCallback):
        raise ValueError("The experimental_sharding_callback checkpoint option"
                         "must be of type ShardingCallback. The option provided"
                         f"was of type {type(experimental_sharding_callback)}.")
    self.experimental_sharding_callback = experimental_sharding_callback
    self.experimental_skip_slot_variables = experimental_skip_slot_variables

  def __copy__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: copies a super proxy object rather than the instance itself, so the result is
    # likely not a faithful CheckpointOptions copy (the other attributes may not carry over).
    # Intended behavior: shallow copy where the write-callbacks list is copied but its callbacks
    # are not. Returns the copy; assigns a copied callbacks list to it.
    # Only `experimental_write_callbacks` needs special treatment to Ensure that
    # the list is deep-copied, but the callbacks are not deep-copied.
    result = copy.copy(super())  # First invoke the non-overridden copy method.
    result.experimental_write_callbacks = copy.copy(
        self.experimental_write_callbacks
    )
    return result
