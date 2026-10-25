# [MODULE DOC]
# Defines SyncReplicasOptimizer (deprecated, registered under the v1 train namespace), which
# turns any optimizer into a synchronous multi-replica one using conditional accumulators and a
# token queue, plus an internal session hook that performs chief and worker initialization.
# Imports are only used for graph construction; no import-time side effects beyond the export
# registration. Superseded by distribution strategies.
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

"""Synchronize replicas for training."""
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import optimizer
from tensorflow.python.training import queue_runner
from tensorflow.python.training import session_manager
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


# Please note that the gradients from replicas are averaged instead of summed
# (as in the old sync_replicas_optimizer) so you need to increase the learning
# rate according to the number of replicas. This change is introduced to be
# consistent with how gradients are aggregated (averaged) within a batch in a
# replica.
@tf_export(v1=["train.SyncReplicasOptimizer"])
class SyncReplicasOptimizer(optimizer.Optimizer):
  # [CLASS DOC]
  # Deprecated optimizer wrapper for synchronous data-parallel training in graph mode. It wraps
  # another optimizer (held in the opt attribute), routes each replica's gradients into
  # per-variable conditional accumulators, lets the chief average a configured number of them,
  # applies the average through the wrapped optimizer, and releases replicas through a shared
  # token queue holding global step values, acting as a barrier. State held: the wrapped
  # optimizer, replicas to aggregate, total replica count, tokens per step, optional variable
  # averager and variable list, the global step, the token queue, the chief queue runner, a
  # list of (accumulator, device) pairs, and a flag recording whether gradients were applied.
  # After applying gradients it also exposes the local step initialization op, the chief
  # initialization op and a readiness op. Created with a wrapped optimizer and replica counts;
  # used via the usual minimize/compute/apply flow plus a session run hook for chief setup.
  # Subclass of the base Optimizer; delegates gradient computation, slots and variables.
  # Creating an instance logs a deprecation warning (once) through the deprecation decorator.
  """Class to synchronize, aggregate gradients and pass them to the optimizer.

  This class is deprecated. For synchronous training, please use [Distribution
  Strategies](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute).

  In a typical asynchronous training environment, it's common to have some
  stale gradients. For example, with a N-replica asynchronous training,
  gradients will be applied to the variables N times independently. Depending
  on each replica's training speed, some gradients might be calculated from
  copies of the variable from several steps back (N-1 steps on average). This
  optimizer avoids stale gradients by collecting gradients from all replicas,
  averaging them, then applying them to the variables in one shot, after
  which replicas can fetch the new variables and continue.

  The following accumulators/queue are created:

  * N `gradient accumulators`, one per variable to train. Gradients are pushed
    to them and the chief worker will wait until enough gradients are collected
    and then average them before applying to variables. The accumulator will
    drop all stale gradients (more details in the accumulator op).
  * 1 `token` queue where the optimizer pushes the new global_step value after
    all variables are updated.

  The following local variable is created:
  * `sync_rep_local_step`, one per replica. Compared against the global_step in
    each accumulator to check for staleness of the gradients.

  The optimizer adds nodes to the graph to collect gradients and pause the
  trainers until variables are updated.
  For the Parameter Server job:

  1. An accumulator is created for each variable, and each replica pushes the
     gradients into the accumulators instead of directly applying them to the
     variables.
  2. Each accumulator averages once enough gradients (replicas_to_aggregate)
     have been accumulated.
  3. Apply the averaged gradients to the variables.
  4. Only after all variables have been updated, increment the global step.
  5. Only after step 4, pushes `global_step` in the `token_queue`, once for
     each worker replica. The workers can now fetch the global step, use it to
     update its local_step variable and start the next batch. Please note that
     some workers can consume multiple minibatches, while some may not consume
     even one. This is because each worker fetches minibatches as long as
     a token exists. If one worker is stuck for some reason and does not
     consume a token, another worker can use it.

  For the replicas:

  1. Start a step: fetch variables and compute gradients.
  2. Once the gradients have been computed, push them into gradient
     accumulators. Each accumulator will check the staleness and drop the stale.
  3. After pushing all the gradients, dequeue an updated value of global_step
     from the token queue and record that step to its local_step variable. Note
     that this is effectively a barrier.
  4. Start the next batch.

  ### Usage

  ```python
  # Create any optimizer to update the variables, say a simple SGD:
  opt = GradientDescentOptimizer(learning_rate=0.1)

  # Wrap the optimizer with sync_replicas_optimizer with 50 replicas: at each
  # step the optimizer collects 50 gradients before applying to variables.
  # Note that if you want to have 2 backup replicas, you can change
  # total_num_replicas=52 and make sure this number matches how many physical
  # replicas you started in your job.
  opt = tf.compat.v1.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=50,
                                 total_num_replicas=50)

  # Some models have startup_delays to help stabilize the model but when using
  # sync_replicas training, set it to 0.

  # Now you can call `minimize()` or `compute_gradients()` and
  # `apply_gradients()` normally
  training_op = opt.minimize(total_loss, global_step=self.global_step)


  # You can create the hook which handles initialization and queues.
  sync_replicas_hook = opt.make_session_run_hook(is_chief)
  ```

  In the training program, every worker will run the train_op as if not
  synchronized.

  ```python
  with training.MonitoredTrainingSession(
      master=workers[worker_id].target, is_chief=is_chief,
      hooks=[sync_replicas_hook]) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(training_op)
  ```
  """

  @deprecation.deprecated(
      None, "The `SyncReplicaOptimizer` class is deprecated. For synchronous "
      "training, please use [Distribution Strategies](https://github.com/"
      "tensorflow/tensorflow/tree/master/tensorflow/contrib/distribute).",
      warn_once=True)
  def __init__(self,
               opt,
               replicas_to_aggregate,
               total_num_replicas=None,
               variable_averages=None,
               variables_to_average=None,
               use_locking=False,
               name="sync_replicas"):
    # [FUNCTION DOC]
    # Constructor. Inputs: opt, the wrapped optimizer; replicas_to_aggregate, how many gradients
    # are averaged per update; total_num_replicas, optional worker count (defaults to
    # replicas_to_aggregate when null); variable_averages, optional moving-average object;
    # variables_to_average, list of variables for that averager; use_locking, passed to the base
    # class; name, optional operation name. Stores all of these, computes tokens per step as the
    # larger of the two replica counts, and initializes the global step, token queue and chief
    # queue runner to nothing and the accumulator list to empty. Logs the replica counts at
    # info level. Returns nothing and raises nothing itself. Wrapped by a deprecation decorator
    # that warns once.
    """Construct a sync_replicas optimizer.

    Args:
      opt: The actual optimizer that will be used to compute and apply the
        gradients. Must be one of the Optimizer classes.
      replicas_to_aggregate: number of replicas to aggregate for each variable
        update.
      total_num_replicas: Total number of tasks/workers/replicas, could be
        different from replicas_to_aggregate.
        If total_num_replicas > replicas_to_aggregate: it is backup_replicas +
        replicas_to_aggregate.
        If total_num_replicas < replicas_to_aggregate: Replicas compute
        multiple batches per update to variables.
      variable_averages: Optional `ExponentialMovingAverage` object, used to
        maintain moving averages for the variables passed in
        `variables_to_average`.
      variables_to_average: a list of variables that need to be averaged. Only
        needed if variable_averages is passed in.
      use_locking: If True use locks for update operation.
      name: string. Optional name of the returned operation.
    """
    if total_num_replicas is None:
      total_num_replicas = replicas_to_aggregate

    super(SyncReplicasOptimizer, self).__init__(use_locking, name)
    logging.info(
        "SyncReplicasV2: replicas_to_aggregate=%s; total_num_replicas=%s",
        replicas_to_aggregate, total_num_replicas)
    self._opt = opt
    self._replicas_to_aggregate = replicas_to_aggregate
    self._gradients_applied = False
    self._variable_averages = variable_averages
    self._variables_to_average = variables_to_average
    self._total_num_replicas = total_num_replicas
    self._tokens_per_step = max(total_num_replicas, replicas_to_aggregate)
    self._global_step = None
    self._sync_token_queue = None

    # The synchronization op will be executed in a queue runner which should
    # only be executed by one of the replicas (usually the chief).
    self._chief_queue_runner = None

    # Remember which accumulator is on which device to set the initial step in
    # the accumulator to be global step. This list contains list of the
    # following format: (accumulator, device).
    self._accumulator_list = []

  def compute_gradients(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Pass-through to the wrapped optimizer. Takes any positional and keyword arguments and
    # returns the list of (gradient, variable) pairs from the wrapped optimizer unchanged.
    # Aggregation across replicas is deferred to apply_gradients. No side effects of its own.
    """Compute gradients of "loss" for the variables in "var_list".

    This simply wraps the compute_gradients() from the real optimizer. The
    gradients will be aggregated in the apply_gradients() so that user can
    modify the gradients like clipping with per replica global norm if needed.
    The global norm with aggregated gradients can be bad as one replica's huge
    gradients can hurt the gradients from other replicas.

    Args:
      *args: Arguments for compute_gradients().
      **kwargs: Keyword arguments for compute_gradients().

    Returns:
      A list of (gradient, variable) pairs.
    """
    return self._opt.compute_gradients(*args, **kwargs)

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name argument is accepted but never used, so the documented default of
    # the optimizer name is not applied here (the name scope uses the optimizer's own name).
    # KNOWN ISSUE: the docstring says global step is optional, but a missing global step raises.
    # Builds the synchronous update graph. Inputs: grads_and_vars, non-empty list of
    # (gradient, variable) pairs where a gradient may be nothing, a dense tensor or sparse
    # indexed slices; global_step, a required variable; name, unused. For each variable on its
    # device it creates a conditional accumulator (dense or sparse) named from the variable,
    # pushes the gradient with the local step, and takes the average once the required number of
    # replicas has contributed; null gradients pass through. The averaged pairs are applied by
    # the wrapped optimizer on the global step device, which also increments the global step. A
    # FIFO token queue is created; the returned train op dequeues a token (after all pushes)
    # and assigns it to the local step, acting as a barrier. A sync op enqueues tokens equal to
    # tokens per step after the update (and after variable averaging if configured) and is
    # wrapped in a chief queue runner. Side effects: creates a non-trainable local-variable
    # local step colocated with a placeholder op, sets attributes for the local step init op,
    # chief init op (local step plus accumulator global steps) and readiness op, records
    # accumulators, and marks gradients as applied. Returns the train op. Raises a value error if
    # the list is empty, the global step is missing, or a gradient has an unknown type.
    """Apply gradients to variables.

    This contains most of the synchronization implementation and also wraps the
    apply_gradients() from the real optimizer.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
      global_step: Optional Variable to increment by one after the
        variables have been updated.
      name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

    Returns:
      train_op: The op to dequeue a token so the replicas can exit this batch
      and start the next one. This is executed by each replica.

    Raises:
      ValueError: If the grads_and_vars is empty.
      ValueError: If global step is not provided, the staleness cannot be
        checked.
    """
    if not grads_and_vars:
      raise ValueError("Must supply at least one variable")

    if global_step is None:
      raise ValueError("Global step is required to check staleness")

    self._global_step = global_step
    train_ops = []
    aggregated_grad = []
    var_list = []

    # local_anchor op will be placed on this worker task by default.
    local_anchor = control_flow_ops.no_op()
    # Colocating local_step variable prevents it being placed on the PS.
    distribution_strategy = distribute_lib.get_strategy()
    with distribution_strategy.extended.colocate_vars_with(local_anchor):
      self._local_step = variable_v1.VariableV1(
          initial_value=0,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          dtype=global_step.dtype.base_dtype,
          name="sync_rep_local_step")

    self.local_step_init_op = state_ops.assign(self._local_step, global_step)
    chief_init_ops = [self.local_step_init_op]
    self.ready_for_local_init_op = variables.report_uninitialized_variables(
        variables.global_variables())

    with ops.name_scope(None, self._name):
      for grad, var in grads_and_vars:
        var_list.append(var)
        with ops.device(var.device):
          # Dense gradients.
          if grad is None:
            aggregated_grad.append(None)  # pass-through.
            continue
          elif isinstance(grad, tensor.Tensor):
            grad_accum = data_flow_ops.ConditionalAccumulator(
                grad.dtype,
                shape=var.get_shape(),
                shared_name=var.name + "/grad_accum")
            train_ops.append(grad_accum.apply_grad(
                grad, local_step=self._local_step))
            aggregated_grad.append(grad_accum.take_grad(
                self._replicas_to_aggregate))
          else:
            if not isinstance(grad, indexed_slices.IndexedSlices):
              raise ValueError("Unknown grad type!")
            grad_accum = data_flow_ops.SparseConditionalAccumulator(
                grad.dtype, shape=(), shared_name=var.name + "/grad_accum")
            train_ops.append(grad_accum.apply_indexed_slices_grad(
                grad, local_step=self._local_step))
            aggregated_grad.append(grad_accum.take_indexed_slices_grad(
                self._replicas_to_aggregate))

          self._accumulator_list.append((grad_accum, var.device))

      aggregated_grads_and_vars = zip(aggregated_grad, var_list)

      # sync_op will be assigned to the same device as the global step.
      with ops.device(global_step.device), ops.name_scope(""):
        update_op = self._opt.apply_gradients(aggregated_grads_and_vars,
                                              global_step)

      # Create token queue.
      with ops.device(global_step.device), ops.name_scope(""):
        sync_token_queue = (
            data_flow_ops.FIFOQueue(-1,
                                    global_step.dtype.base_dtype,
                                    shapes=(),
                                    name="sync_token_q",
                                    shared_name="sync_token_q"))
        self._sync_token_queue = sync_token_queue

      with ops.device(global_step.device), ops.name_scope(""):
        # Replicas have to wait until they can get a token from the token queue.
        with ops.control_dependencies(train_ops):
          token = sync_token_queue.dequeue()
        train_op = state_ops.assign(self._local_step, token)

        with ops.control_dependencies([update_op]):
          # Sync_op needs to insert tokens to the token queue at the end of the
          # step so the replicas can fetch them to start the next step.
          tokens = array_ops.fill([self._tokens_per_step], global_step)
          sync_op = sync_token_queue.enqueue_many((tokens,))

        if self._variable_averages is not None:
          with ops.control_dependencies([sync_op]), ops.name_scope(""):
            sync_op = self._variable_averages.apply(
                self._variables_to_average)

        self._chief_queue_runner = queue_runner.QueueRunner(
            sync_token_queue, [sync_op])
      for accum, dev in self._accumulator_list:
        with ops.device(dev):
          chief_init_ops.append(
              accum.set_global_step(
                  global_step, name="SetGlobalStep"))
      self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
      self._gradients_applied = True
      return train_op

  def get_chief_queue_runner(self):
    # [FUNCTION DOC]
    # Returns the queue runner the chief must start, which runs the sync op that applies the
    # aggregated update and inserts tokens. Takes no inputs. Raises a value error if
    # apply_gradients has not been called yet. No side effects.
    """Returns the QueueRunner for the chief to execute.

    This includes the operations to synchronize replicas: aggregate gradients,
    apply to variables, increment global step, insert tokens to token queue.

    Note that this can only be called after calling apply_gradients() which
    actually generates this queuerunner.

    Returns:
      A `QueueRunner` for chief to execute.

    Raises:
      ValueError: If this is called before apply_gradients().
    """
    if self._gradients_applied is False:
      raise ValueError("Should be called after apply_gradients().")

    return self._chief_queue_runner

  def get_slot(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Pass-through to the wrapped optimizer's slot lookup. Takes any positional and keyword
    # arguments (normally a variable and a slot name). Returns the slot variable or nothing.
    """Return a slot named "name" created for "var" by the Optimizer.

    This simply wraps the get_slot() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      The `Variable` for the slot if it was created, `None` otherwise.
    """
    return self._opt.get_slot(*args, **kwargs)

  def variables(self):
    # [FUNCTION DOC]
    # Returns the list of variables of the wrapped optimizer. Takes no inputs. Does not include
    # this class's own local step variable. No side effects.
    """Fetches a list of optimizer variables in the default graph.

    This wraps `variables()` from the actual optimizer. It does not include
    the `SyncReplicasOptimizer`'s local step.

    Returns:
      A list of variables.
    """
    return self._opt.variables()

  def get_slot_names(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Pass-through to the wrapped optimizer's slot-name listing. Takes any positional and
    # keyword arguments and returns the list of slot name strings. No side effects.
    """Return a list of the names of slots created by the `Optimizer`.

    This simply wraps the get_slot_names() from the actual optimizer.

    Args:
      *args: Arguments for get_slot().
      **kwargs: Keyword arguments for get_slot().

    Returns:
      A list of strings.
    """
    return self._opt.get_slot_names(*args, **kwargs)

  def get_init_tokens_op(self, num_tokens=-1):
    # [FUNCTION DOC]
    # Builds the op that seeds the token queue with initial tokens (copies of the global step).
    # Input num_tokens: count to add; the default of minus one means use replicas_to_aggregate.
    # Returns the enqueue op, or a no-op named for no initial tokens when the count is zero or
    # less. Raises a value error if called before apply_gradients, or if an explicit count is
    # below replicas_to_aggregate minus total_num_replicas. Creates graph ops on the global step
    # device.
    """Returns the op to fill the sync_token_queue with the tokens.

    This is supposed to be executed in the beginning of the chief/sync thread
    so that even if the total_num_replicas is less than replicas_to_aggregate,
    the model can still proceed as the replicas can compute multiple steps per
    variable update. Make sure:
    `num_tokens >= replicas_to_aggregate - total_num_replicas`.

    Args:
      num_tokens: Number of tokens to add to the queue.

    Returns:
      An op for the chief/sync replica to fill the token queue.

    Raises:
      ValueError: If this is called before apply_gradients().
      ValueError: If num_tokens are smaller than replicas_to_aggregate -
        total_num_replicas.
    """
    if self._gradients_applied is False:
      raise ValueError(
          "get_init_tokens_op() should be called after apply_gradients().")

    tokens_needed = self._replicas_to_aggregate - self._total_num_replicas
    if num_tokens == -1:
      num_tokens = self._replicas_to_aggregate
    elif num_tokens < tokens_needed:
      raise ValueError(
          "Too few tokens to finish the first step: %d (given) vs %d (needed)" %
          (num_tokens, tokens_needed))

    if num_tokens > 0:
      with ops.device(self._global_step.device), ops.name_scope(""):
        tokens = array_ops.fill([num_tokens], self._global_step)
        init_tokens = self._sync_token_queue.enqueue_many((tokens,))
    else:
      init_tokens = control_flow_ops.no_op(name="no_init_tokens")

    return init_tokens

  def make_session_run_hook(self, is_chief, num_tokens=-1):
    # [FUNCTION DOC]
    # Factory for the session hook that initializes this optimizer's local step, queue runner
    # and initial tokens. Inputs: is_chief, whether this worker is the chief; num_tokens, initial
    # token count (default minus one meaning the standard amount). Returns the hook object.
    # No other side effects; validation happens when the hook starts.
    """Creates a hook to handle SyncReplicasHook ops such as initialization."""
    return _SyncReplicasOptimizerHook(self, is_chief, num_tokens)


class _SyncReplicasOptimizerHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # HELPER: internal hook created by SyncReplicasOptimizer.make_session_run_hook.
  # Session hook that, once a session exists, runs the local-step initialization (chief or
  # non-chief variant), runs the initial token op on the chief, and starts the chief queue
  # runner threads as daemons. Holds the optimizer, chief flag, token count, and the init ops
  # and queue runner chosen in begin.
  """A SessionRunHook handles ops related to SyncReplicasOptimizer."""

  def __init__(self, sync_optimizer, is_chief, num_tokens):
    # [FUNCTION DOC]
    # HELPER: constructor for the internal sync-replicas hook.
    # Stores the synchronized optimizer, whether this is the chief, and the initial token count.
    # Returns nothing; no graph or session effects.
    """Creates hook to handle SyncReplicasOptimizer initialization ops.

    Args:
      sync_optimizer: `SyncReplicasOptimizer` which this hook will initialize.
      is_chief: `Bool`, whether is this a chief replica or not.
      num_tokens: Number of tokens to add to the queue.
    """
    self._sync_optimizer = sync_optimizer
    self._is_chief = is_chief
    self._num_tokens = num_tokens

  def begin(self):
    # [FUNCTION DOC]
    # HELPER: hook start callback selecting which init ops to use.
    # Raises a value error if the optimizer's gradients were not applied yet. On the chief it
    # picks the chief init op, the readiness op, the chief queue runner and the initial tokens
    # op (built with the stored token count); on other workers it picks the local step init op
    # and the readiness op and sets the queue runner and tokens op to nothing. Stores them as
    # attributes; returns nothing.
    if self._sync_optimizer._gradients_applied is False:  # pylint: disable=protected-access
      raise ValueError(
          "SyncReplicasOptimizer.apply_gradient should be called before using "
          "the hook.")
    if self._is_chief:
      self._local_init_op = self._sync_optimizer.chief_init_op
      self._ready_for_local_init_op = (
          self._sync_optimizer.ready_for_local_init_op)
      self._q_runner = self._sync_optimizer.get_chief_queue_runner()
      self._init_tokens_op = self._sync_optimizer.get_init_tokens_op(
          self._num_tokens)
    else:
      self._local_init_op = self._sync_optimizer.local_step_init_op
      self._ready_for_local_init_op = (
          self._sync_optimizer.ready_for_local_init_op)
      self._q_runner = None
      self._init_tokens_op = None

  def after_create_session(self, session, coord):
    # [FUNCTION DOC]
    # HELPER: hook callback run after the session is created.
    # Inputs: session, the new session; coord, the coordinator for threads. Checks that the
    # readiness op reports the model ready for local initialization, raising a runtime error
    # (including the init op name and message) otherwise. Then runs the local init op, runs the
    # initial tokens op if present (chief only), and starts queue runner threads as daemons
    # tied to the coordinator if present. Returns nothing.
    """Runs SyncReplicasOptimizer initialization ops."""
    local_init_success, msg = session_manager._ready(  # pylint: disable=protected-access
        self._ready_for_local_init_op, session,
        "Model is not ready for SyncReplicasOptimizer local init.")
    if not local_init_success:
      raise RuntimeError(
          "Init operations did not make model ready for SyncReplicasOptimizer "
          "local_init. Init op: %s, error: %s" %
          (self._local_init_op.name, msg))
    session.run(self._local_init_op)
    if self._init_tokens_op is not None:
      session.run(self._init_tokens_op)
    if self._q_runner is not None:
      self._q_runner.create_threads(
          session, coord=coord, daemon=True, start=True)
