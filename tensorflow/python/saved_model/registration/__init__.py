# [PACKAGE DOC]
# Package for the SavedModel/checkpoint registration API used for advanced objects. Re-exports
# from the registration submodule the lookup functions (registered class, class name, saver name,
# save function, restore function, strict-predicate flag), the experimental registration
# decorators/functions for serializable classes and checkpoint savers, the registered-saver record
# type and the restore-function validator. Adds two convenience wrappers that register under the
# fixed package name tf. Importing has no side effects beyond importing the submodule; the global
# registries live in the registration submodule.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for registering the saving/loading steps for advanced objects."""

from tensorflow.python.saved_model.registration.registration import get_registered_class
from tensorflow.python.saved_model.registration.registration import get_registered_class_name
from tensorflow.python.saved_model.registration.registration import get_registered_saver_name
from tensorflow.python.saved_model.registration.registration import get_restore_function
from tensorflow.python.saved_model.registration.registration import get_save_function
from tensorflow.python.saved_model.registration.registration import get_strict_predicate_restore

# These are currently an evolving feature. Use with care.
from tensorflow.python.saved_model.registration.registration import register_checkpoint_saver
from tensorflow.python.saved_model.registration.registration import register_serializable

from tensorflow.python.saved_model.registration.registration import RegisteredSaver
from tensorflow.python.saved_model.registration.registration import validate_restore_function


def register_tf_serializable(name=None, predicate=None):
  # [FUNCTION DOC]
  # Convenience wrapper that registers a serializable class under the fixed package name tf.
  # Inputs: name, optional serialization name (defaults to the class name); predicate, optional
  # callable deciding whether an object uses this registration. Returns the class decorator
  # produced by the underlying registration function, which registers the decorated class in the
  # global class registry (side effect) and raises when invalid or duplicate names are used.
  """See the docstring for `register_serializable`."""
  return register_serializable(package="tf", name=name, predicate=predicate)


def register_tf_checkpoint_saver(name=None,
                                 predicate=None,
                                 save_fn=None,
                                 restore_fn=None,
                                 strict_predicate_restore=True):
  # [FUNCTION DOC]
  # Convenience wrapper that registers a checkpoint saver under the fixed package name tf by
  # forwarding all arguments to the underlying registration function. Inputs: name, predicate,
  # save function, restore function (all effectively required) and the strict-predicate-restore
  # flag (default true). Registers into the global saver registry (side effect); raises a type
  # error if the functions are not callable and errors for invalid or duplicate names. Returns
  # whatever the underlying function returns, which is nothing.
  """See the docstring for `register_checkpoint_saver`."""
  return register_checkpoint_saver(
      package="tf",
      name=name,
      predicate=predicate,
      save_fn=save_fn,
      restore_fn=restore_fn,
      strict_predicate_restore=strict_predicate_restore)
