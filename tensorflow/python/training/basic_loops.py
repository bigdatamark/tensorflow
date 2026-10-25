# [MODULE DOC]
# Provides basic_train_loop, a minimal training driver for the legacy Supervisor workflow. It is
# exported publicly as the v1 train.basic_train_loop symbol. Only the exported decorator
# registration happens at import time.
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
"""Basic loop for training."""
from tensorflow.python.framework import errors
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["train.basic_train_loop"])
def basic_train_loop(supervisor,
                     train_step_fn,
                     args=None,
                     kwargs=None,
                     master=""):
  # [FUNCTION DOC]
  # Runs a training loop under a Supervisor. Inputs: supervisor (training services object),
  # train_step_fn (callable invoked with the session followed by args and kwargs), args (optional
  # list of positional arguments, defaults to empty), kwargs (optional mapping of keyword
  # arguments, defaults to empty) and master (address used to create the session, empty string
  # means a local in-process session). Repeatedly opens a managed session and calls the step
  # function until the supervisor says to stop. If an aborted error is raised (a distributed
  # server restarted) the whole managed session is recreated and the loop starts again; any other
  # exception propagates. Returns nothing. Side effects: whatever the step function and
  # supervisor do (checkpointing, summaries, threads).
  """Basic loop to train a model.

  Calls `train_step_fn` in a loop to train a model.  The function is called as:

  ```python
  train_step_fn(session, *args, **kwargs)
  ```

  It is passed a `tf.compat.v1.Session` in addition to `args` and `kwargs`.  The
  function
  typically runs one training step in the session.

  Args:
    supervisor: `tf.compat.v1.train.Supervisor` to run the training services.
    train_step_fn: Callable to execute one training step.  Called repeatedly as
      `train_step_fn(session, *args **kwargs)`.
    args: Optional positional arguments passed to `train_step_fn`.
    kwargs: Optional keyword arguments passed to `train_step_fn`.
    master: Master to use to create the training session.  Defaults to `""`
      which causes the session to be created in the local process.
  """
  if args is None:
    args = []
  if kwargs is None:
    kwargs = {}
  should_retry = True
  while should_retry:
    try:
      should_retry = False
      with supervisor.managed_session(master) as sess:
        while not supervisor.should_stop():
          train_step_fn(sess, *args, **kwargs)
    except errors.AbortedError:
      # Always re-run on AbortedError as it indicates a restart of one of the
      # distributed tensorflow servers.
      should_retry = True
