# [MODULE DOC]
# Keras-side (TF v2 compatible) RNN cell wrappers: a shared base class plus the dropout,
# residual and device wrappers, exported under the tf.nn namespace. They are ported here from
# the legacy cell implementation to avoid a cyclic dependency with layer serialization. Each
# public wrapper combines a shared "Base" behavior class from the legacy wrapper implementation
# module with the V2 base class defined here, and is marked deprecated in favor of the Keras RNN
# layer. Importing registers the tf.nn export names; serialization.py picks these classes up
# when TF2 behavior is enabled.
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
"""Module implementing for RNN wrappers for TF v2."""

# Note that all the APIs under this module are exported as tf.nn.*. This is due
# to the fact that those APIs were from tf.nn.rnn_cell_impl. They are ported
# here to avoid the cyclic dependency issue for serialization. These APIs will
# probably be deprecated and removed in future since similar API is available in
# existing Keras RNN API.


from tensorflow.python.keras.layers import recurrent
from tensorflow.python.keras.layers.legacy_rnn import rnn_cell_wrapper_impl
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export


class _RNNCellWrapperV2(recurrent.AbstractRNNCell):
  # [CLASS DOC]
  # HELPER: internal base class for the public V2 RNN cell wrappers in this module.
  # Abstract Keras RNN cell that wraps another cell, stored in the attribute cell. It also records
  # whether the wrapped cell's call accepts a training argument (explicitly or through
  # arbitrary keyword arguments). Provides call, build, config serialization and restoration that
  # delegate to the wrapped cell. Subclasses mix in a legacy "Base" wrapper behavior class.
  """Base class for cells wrappers V2 compatibility.

  This class along with `rnn_cell_impl._RNNCellWrapperV1` allows to define
  wrappers that are compatible with V1 and V2, and defines helper methods for
  this purpose.
  """

  def __init__(self, cell, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Takes the cell to wrap (any RNN cell object with a call method) plus extra
    # positional and keyword arguments forwarded to the parent cell constructor. Stores the cell
    # and computes the private flag saying whether the cell's call takes a training argument
    # (true if named in its arguments or if it accepts arbitrary keyword arguments). Inspects the
    # cell's call signature; returns nothing.
    super(_RNNCellWrapperV2, self).__init__(*args, **kwargs)
    self.cell = cell
    cell_call_spec = tf_inspect.getfullargspec(cell.call)
    self._expects_training_arg = ("training" in cell_call_spec.args) or (
        cell_call_spec.varkw is not None
    )

  def call(self, inputs, state, **kwargs):
    # [FUNCTION DOC]
    # Runs one step of the wrapped cell through the wrapper-specific step helper, passing the
    # wrapped cell's own call as the function to invoke. Inputs: the step input tensor, the
    # state (tensor or tuple of tensors) and extra keyword arguments forwarded to the cell.
    # Returns the pair of output and new state. Assumes the wrapper has already been built.
    # No exceptions of its own; the helper comes from the mixed-in legacy base class.
    """Runs the RNN cell step computation.

    When `call` is being used, we assume that the wrapper object has been built,
    and therefore the wrapped cells has been built via its `build` method and
    its `call` method can be used directly.

    This allows to use the wrapped cell and the non-wrapped cell equivalently
    when using `call` and `build`.

    Args:
      inputs: A tensor with wrapped cell's input.
      state: A tensor or tuple of tensors with wrapped cell's state.
      **kwargs: Additional arguments passed to the wrapped cell's `call`.

    Returns:
      A pair containing:

      - Output: A tensor with cell's output.
      - New state: A tensor or tuple of tensors with new wrapped cell's state.
    """
    return self._call_wrapped_cell(
        inputs, state, cell_call_fn=self.cell.call, **kwargs)

  def build(self, inputs_shape):
    # [FUNCTION DOC]
    # Builds the wrapped cell by calling its build with the given input shape, then marks this
    # wrapper as built. Side effect: creates the wrapped cell's weights. Returns nothing.
    """Builds the wrapped cell."""
    self.cell.build(inputs_shape)
    self.built = True

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a configuration mapping: the parent layer config merged with an entry for the
    # wrapped cell holding the cell's class name and its own config. Takes no arguments, no side
    # effects. Note the cell entry overrides any same-named key from the base config.
    config = {
        "cell": {
            "class_name": self.cell.__class__.__name__,
            "config": self.cell.get_config()
        },
    }
    base_config = super(_RNNCellWrapperV2, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # Class method that rebuilds a wrapper from a config mapping. Copies the config (the input is
    # not mutated), removes the cell entry, deserializes it into a cell object with the layer
    # deserializer (optionally using a mapping of custom object names to classes), and calls the
    # constructor with the cell and the remaining config entries. The deserializer is imported
    # lazily to avoid a circular import. Returns the new wrapper instance.
    config = config.copy()
    from tensorflow.python.keras.layers.serialization import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    cell = deserialize_layer(config.pop("cell"), custom_objects=custom_objects)
    return cls(cell, **config)


@deprecated(None, "Please use tf.keras.layers.RNN instead.")
@tf_export("nn.RNNCellDropoutWrapper", v1=[])
class DropoutWrapper(rnn_cell_wrapper_impl.DropoutWrapperBase,
                     _RNNCellWrapperV2):
  # [CLASS DOC]
  # Public V2 wrapper (exported as the nn dropout cell wrapper, deprecated) that applies dropout
  # to the input, state and output of the wrapped cell. Behavior comes from the legacy dropout
  # base class; the V2 base supplies the Keras cell plumbing.
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Constructor forwarding all positional and keyword arguments to the base classes (cell and
    # dropout settings as documented on the legacy dropout base). Raises a value error if the
    # wrapped cell is a Keras LSTM cell, because dropout must be configured on that cell itself.
    # The constructor's documentation text is copied from the legacy base class.
    super(DropoutWrapper, self).__init__(*args, **kwargs)
    if isinstance(self.cell, recurrent.LSTMCell):
      raise ValueError("keras LSTM cell does not work with DropoutWrapper. "
                       "Please use LSTMCell(dropout=x, recurrent_dropout=y) "
                       "instead.")

  __init__.__doc__ = rnn_cell_wrapper_impl.DropoutWrapperBase.__init__.__doc__


@deprecated(None, "Please use tf.keras.layers.RNN instead.")
@tf_export("nn.RNNCellResidualWrapper", v1=[])
class ResidualWrapper(rnn_cell_wrapper_impl.ResidualWrapperBase,
                      _RNNCellWrapperV2):
  # [CLASS DOC]
  # Public V2 wrapper (exported as the nn residual cell wrapper, deprecated) that adds the cell
  # input to the cell output. Behavior comes from the legacy residual base class.
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Constructor that only forwards all positional and keyword arguments to the base classes
    # (the cell and optional residual function). Returns nothing; its documentation text is
    # copied from the legacy base class.
    super(ResidualWrapper, self).__init__(*args, **kwargs)

  __init__.__doc__ = rnn_cell_wrapper_impl.ResidualWrapperBase.__init__.__doc__


@deprecated(None, "Please use tf.keras.layers.RNN instead.")
@tf_export("nn.RNNCellDeviceWrapper", v1=[])
class DeviceWrapper(rnn_cell_wrapper_impl.DeviceWrapperBase,
                    _RNNCellWrapperV2):
  # [CLASS DOC]
  # Public V2 wrapper (exported as the nn device cell wrapper, deprecated) that runs the wrapped
  # cell under a given device scope. Behavior comes from the legacy device base class.
  """Operator that ensures an RNNCell runs on a particular device."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    # [FUNCTION DOC]
    # Constructor that only forwards all positional and keyword arguments to the base classes
    # (the cell and the device specification). Returns nothing; its documentation text is copied
    # from the legacy base class.
    super(DeviceWrapper, self).__init__(*args, **kwargs)

  __init__.__doc__ = rnn_cell_wrapper_impl.DeviceWrapperBase.__init__.__doc__
