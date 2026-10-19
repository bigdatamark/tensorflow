# [MODULE DOC]
# HELPER: helper module for Keras callbacks and model saving under multi-worker strategies.
# Path utilities so that only the chief worker writes to the real output location. Non-chief
# workers are redirected to a per-task temporary subdirectory (named after the worker task id)
# that they later delete. Exposes functions for directory paths and file paths (get the write
# location, remove the temporary location). Experimental. Imports os, distribute_lib and file_io;
# no import-time side effects.
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
"""Utilities that help manage directory path in distributed settings.

In multi-worker training, the need to write a file to distributed file
location often requires only one copy done by one worker despite many workers
that are involved in training. The option to only perform saving by chief is
not feasible for a couple of reasons: 1) Chief and workers may each contain
a client that runs the same piece of code and it's preferred not to make
any distinction between the code run by chief and other workers, and 2)
saving of model or model's related information may require SyncOnRead
variables to be read, which needs the cooperation of all workers to perform
all-reduce.

This set of utility is used so that only one copy is written to the needed
directory, by supplying a temporary write directory path for workers that don't
need to save, and removing the temporary directory once file writing is done.

Example usage:
```
# Before using a directory to write file to.
self.log_write_dir = write_dirpath(self.log_dir, get_distribution_strategy())
# Now `self.log_write_dir` can be safely used to write file to.

...

# After the file is written to the directory.
remove_temp_dirpath(self.log_dir, get_distribution_strategy())

```

Experimental. API is subject to change.
"""

import os

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.lib.io import file_io


def _get_base_dirpath(strategy):
  # [FUNCTION DOC]
  # HELPER: builds the temporary directory name for write_dirpath and remove_temp_dirpath.
  # Input: strategy, a tf.distribute strategy whose extended object has a private task id.
  # Output: a string made of the prefix for worker temp directories followed by the task id.
  # Raises an attribute error if the strategy has no task id. No side effects.
  task_id = strategy.extended._task_id  # pylint: disable=protected-access
  return 'workertemp_' + str(task_id)


def _is_temp_dir(dirpath, strategy):
  # [FUNCTION DOC]
  # HELPER: lets _get_temp_dir avoid appending the temporary name twice.
  # Returns true if dirpath already ends with this worker's temporary directory name.
  # Inputs: dirpath (string path), strategy (tf.distribute strategy). Output: boolean.
  # No side effects.
  return dirpath.endswith(_get_base_dirpath(strategy))


def _get_temp_dir(dirpath, strategy):
  # [FUNCTION DOC]
  # HELPER: computes and creates the per-worker temporary directory for write_dirpath and
  # remove_temp_dirpath.
  # If dirpath already is the temporary directory it is used as is; otherwise the worker's
  # temporary name is joined onto dirpath. Inputs: dirpath (string), strategy (tf.distribute
  # strategy). Output: the temporary directory path string.
  # Side effect: recursively creates that directory on the file system (also when it is only
  # about to be deleted). File-system errors from the creation propagate.
  if _is_temp_dir(dirpath, strategy):
    temp_dir = dirpath
  else:
    temp_dir = os.path.join(dirpath, _get_base_dirpath(strategy))
  file_io.recursive_create_dir_v2(temp_dir)
  return temp_dir


def write_dirpath(dirpath, strategy):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says dirpath is created if it does not exist, but the code only
  # creates the temporary worker directory; when the original dirpath is returned (no strategy,
  # not multi-worker, or chief) nothing is created.
  # Chooses the directory a worker should write to. Inputs: dirpath (original directory path
  # string) and strategy (tf.distribute strategy, or nothing to use the current strategy from
  # the distribute library). Returns dirpath unchanged when no strategy is available, when not
  # in multi-worker mode, or when this worker should checkpoint (is chief); otherwise returns the
  # path of a per-worker temporary directory under dirpath, which it creates.
  # Side effect: may create a directory. Raises only if the strategy lacks expected attributes.
  """Returns the writing dir that should be used to save file distributedly.

  `dirpath` would be created if it doesn't exist.

  Args:
    dirpath: Original dirpath that would be used without distribution.
    strategy: The tf.distribute strategy object currently used.

  Returns:
    The writing dir path that should be used to save with distribution.
  """
  if strategy is None:
    # Infer strategy from `distribute_lib` if not given.
    strategy = distribute_lib.get_strategy()
  if strategy is None:
    # If strategy is still not available, this is not in distributed training.
    # Fallback to original dirpath.
    return dirpath
  if not strategy.extended._in_multi_worker_mode():  # pylint: disable=protected-access
    return dirpath
  if strategy.extended.should_checkpoint:
    return dirpath
  # If this worker is not chief and hence should not save file, save it to a
  # temporary directory to be removed later.
  return _get_temp_dir(dirpath, strategy)


def remove_temp_dirpath(dirpath, strategy):
  # [FUNCTION DOC]
  # Cleans up after writing: on a non-chief worker in multi-worker mode, recursively deletes the
  # worker's temporary directory for dirpath. Inputs: dirpath (original directory path string),
  # strategy (tf.distribute strategy, or nothing to use the current strategy). Returns nothing.
  # Does nothing when no strategy is available, when not in multi-worker mode, or when this
  # worker is the one that checkpoints. Side effects: the temporary directory is created if
  # missing (by the path helper) and then deleted; file-system errors propagate.
  """Removes the temp path after writing is finished.

  Args:
    dirpath: Original dirpath that would be used without distribution.
    strategy: The tf.distribute strategy object currently used.
  """
  if strategy is None:
    # Infer strategy from `distribute_lib` if not given.
    strategy = distribute_lib.get_strategy()
  if strategy is None:
    # If strategy is still not available, this is not in distributed training.
    # Fallback to no-op.
    return
  # TODO(anjalisridhar): Consider removing the check for multi worker mode since
  # it is redundant when used with the should_checkpoint property.
  if (strategy.extended._in_multi_worker_mode() and  # pylint: disable=protected-access
      not strategy.extended.should_checkpoint):
    # If this worker is not chief and hence should not save file, remove
    # the temporary directory.
    file_io.delete_recursively(_get_temp_dir(dirpath, strategy))


def write_filepath(filepath, strategy):
  # [FUNCTION DOC]
  # File-level wrapper around write_dirpath. Splits filepath into directory and base name,
  # obtains the directory to write to for this worker, and joins the base name back on.
  # Inputs: filepath (original file path string), strategy (tf.distribute strategy or nothing).
  # Output: the file path to actually write to (the original for the chief or non-distributed
  # runs, otherwise inside the worker's temporary directory). Side effect: may create the
  # temporary directory.
  """Returns the writing file path to be used to save file distributedly.

  Directory to contain `filepath` would be created if it doesn't exist.

  Args:
    filepath: Original filepath that would be used without distribution.
    strategy: The tf.distribute strategy object currently used.

  Returns:
    The writing filepath that should be used to save file with distribution.
  """
  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  return os.path.join(write_dirpath(dirpath, strategy), base)


def remove_temp_dir_with_filepath(filepath, strategy):
  # [FUNCTION DOC]
  # File-level wrapper around remove_temp_dirpath: takes the directory part of filepath and
  # removes the worker's temporary directory for it when applicable (non-chief worker in
  # multi-worker mode). Inputs: filepath (original file path string), strategy (tf.distribute
  # strategy or nothing). Returns nothing. Side effect: may delete a directory recursively.
  """Removes the temp path for file after writing is finished.

  Args:
    filepath: Original filepath that would be used without distribution.
    strategy: The tf.distribute strategy object currently used.
  """
  remove_temp_dirpath(os.path.dirname(filepath), strategy)
