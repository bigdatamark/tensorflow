# [MODULE DOC]
# Defines the Composite decorator used to mark a Python function as the composition (decomposition
# into other ops) of a TensorFlow operator. The decorator only tags the function with the op name;
# the other declared metadata (inputs, attributes, derived attributes, outputs) is read from the
# decorator call syntax by the op registration generator and the TFR generator, not from the
# tagged function. No import-time side effects.
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
"""Op composition registration."""


# TODO(fengliuai): add the tf_export decrator
class Composite(object):
  # [CLASS DOC]
  # Decorator class (instantiated with arguments, then applied to a function). Stores the op name
  # and optional declarations of inputs, attributes, derived attributes and outputs in private
  # fields; applying it to a function sets a marker attribute holding the op name on that function
  # and returns the same function. Only the op name is used at runtime; the remaining fields are
  # stored but otherwise unused (generators parse the syntax tree of the decorator call instead).
  """A decorator to register a function as a composition for an TF operator.

  The argument to the decorator must be the name of a TF raw operator the
  function composites for. Decorated function must take positional arguments
  which corresponds to the input and attributes in OpDef of the TF operation.
  # TODO(fengliuai): more documents here.

  Example:
    @composite.Composite('AddN')
    def _compose_add_n(inputs, N):
      if N == 1:
        ....
  """

  # TODO(fengliuai): support input_binding and output_binding so the arguments
  # are not positional.
  def __init__(self,
               op_name,
               inputs=None,
               attrs=None,
               derived_attrs=None,
               outputs=None):
    # [FUNCTION DOC]
    # Constructor. op_name is the string name of the TensorFlow raw op being composed. inputs,
    # attrs, derived_attrs and outputs are optional (default nothing) lists of declaration strings
    # for the op's inputs, attributes, derived attributes and outputs. All values are just stored
    # in private fields; no validation is done.
    self._op_name = op_name
    self._inputs = inputs
    self._attrs = attrs
    self._derived_attrs = derived_attrs
    self._outputs = outputs

  def __call__(self, compose_fn):
    # [FUNCTION DOC]
    # Applies the decorator. Takes the composition function compose_fn, sets a marker attribute
    # recording the op name on it (mutating the function), and returns the same function object.
    # No validation of the function arguments against the declared inputs and attributes is done.
    # TODO(fengliuai): more sanity check of the input function and make sure
    # the bounded arguments of the function matches the 'inputs' and 'attrs'.
    setattr(compose_fn, '_tfr_op_name', self._op_name)
    return compose_fn
