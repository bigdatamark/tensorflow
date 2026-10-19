# [MODULE DOC]
# Framework of debug wrapper sessions. Defines the request/response/action value types exchanged
# with the callbacks of a wrapper session (session init, run start, run end), the abstract base
# class that wraps a normal or monitored session and intercepts its run calls to add tensor
# watches or profiling, a watch-options value type, and a base class for non-interactive wrapper
# sessions driven by an optional watch callback. Concrete wrappers (for example the dumping
# wrapper) subclass these. The remote instruction loop is declared but not implemented. No
# import-time side effects.
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
"""Framework of debug wrapper sessions.

A debug wrapper session is a wrapper around a TensorFlow Python Session.
The wrapper preserves the Session interface, most importantly the run() method,
while providing abilities to:
a) Intercept a run() call to a wrapped session and insert debug tensor watches
   according to externally-specified debug URLs.

b) Release control to an external (i.e., non-Session) object before and after
   the run() call, so that the external object can perform actions such as
   launching a UI to let users inspect the intermediate tensors and partition
   graphs from the run() call.

c) (To be implemented in a future CL) Enter an instruction loop to let an
   external object (e.g., remote client) launch run() and cont() calls
   remotely.

*** The lifetime of a debug wrapper session: ***

1) The wrapper session is created by calling the constructor with a
   wrapped (normal) session as the argument:
     wrapper = FooDebugWrapperSession(sess)
   wherein FooDebugWrapperSession is a concrete subclass implementing the
   abstract BaseDebugWrapperSession class below.

2) Near the end of the constructor call, the on_session_init() callback is
   invoked, with a OnSessionInitRequest object as the argument. The object
   carries the wrapped (normal) session object.

3) The callback handles the request and returns a OnSessionInitResponse
   object with an action field, directing the wrapper session what to do next.

If the action field in the OnSessionInitResponse is PROCEED, the constructor
returns. Control is released back to the caller of the constructor, which can
invoke run() method of wrapper session with the same syntax as a non-wrapped
session, e.g.,:
  wrapper.run(fetches, feed_dict=feeds, options=run_options)

Below, A1 - A2 is the lifetime of a wrapper run() call if the action is
PROCEED:

A1) Right at the start of each run() call, the on_run_start() callback is
    invoked, with an OnRunStartRequest object carrying information such as
    the fetches, the feed dict, the run options and run metadata used in
    this run call, along with a count of how many run calls has occurred
    on this wrapper session. The callback then returns an OnRunStartResponse
    object, of which the action field directs what the wrapper session
    actually will do of the run() call.

    If the action is DEBUG_RUN, a debugged (tensor-watched) run will ensue,
    with the debug URLs supplied in the debug_urls field of the response.
    These can be file:// or grpc:// URLs, for example.

    If the action is NON_DEBUG_RUN, a non-debug (normal) run will ensue.

A2) Right before the run() returns, the on_run_end() callback is invoked,
    with an OnRunEndRequest object as the argument, which carries information
    including the actual action performed in the wrapper run() call and the
    run_metadata from the run() call.

However, if the action field in OnSessionInitResponse is
REMOTE_INSTR_LOOP, the constructor will automatically invoke an instruction loop
that gives the control to a remote caller.

In the remote instruction loop, the following steps will happen:

B1) Callback on_instr_start() is invoked. The callback will return an
    OnInstrStartResponse object with an action field which can order one of
    the following actions:
        i) a run() call with fetches, feeds and debug_urls specified.
       ii) exit the instruction loop.

B2) The wrapper session carries out the action specified above.

B3) If still in the instruction loop, the wrapper session invokes the
    on_instr_end() callback. After the on_instr_end() callback returns, jump
    back to B1.

TODO(cais): Implemented the instruction loop in B1 - B3.

"""

import abc
import re
import threading

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import errors
from tensorflow.python.framework import stack
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import monitored_session
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc


# Helper function.
def _check_type(obj, expected_types):
  # [FUNCTION DOC]
  # HELPER: type validation used by the request/response classes and wrapper sessions.
  # Inputs: obj, any value; expected_types, one type or a tuple of types. Raises a type error
  # naming expected and actual types when the value is not an instance. Returns nothing.
  """Check if an object is of the expected type.

  Args:
    obj: The object being checked.
    expected_types: (`type` or an iterable of `type`s) The expected `type`(s)
      of obj.

  Raises:
      TypeError: If obj is not an instance of expected_type.
  """
  if not isinstance(obj, expected_types):
    raise TypeError("Expected type %s; got type %s" %
                    (expected_types, type(obj)))


class OnSessionInitRequest:
  # [CLASS DOC]
  # Value object passed to the on-session-init callback of a wrapper session. Holds the wrapped
  # session in a session attribute.
  """Request to an on-session-init callback.

  This callback is invoked during the __init__ call to a debug-wrapper session.
  """

  def __init__(self, sess):
    # [FUNCTION DOC]
    # Stores the wrapped session. Input: sess, a regular or monitored session; raises a type
    # error otherwise. Returns nothing.
    """Constructor.

    Args:
      sess: A tensorflow Session object.
    """

    _check_type(sess, (session.BaseSession, monitored_session.MonitoredSession))
    self.session = sess


class OnSessionInitAction:
  # [CLASS DOC]
  # Enum-like holder of string constants for what a wrapper does after session init: proceed
  # normally, or enter a remote instruction loop (not implemented by the base class).
  """Enum-like values for possible action to take on session init."""

  # Proceed, without special actions, in the wrapper session initialization.
  # What action the wrapper session performs next is determined by the caller
  # of the wrapper session. E.g., it can call run().
  PROCEED = "proceed"

  # Instead of letting the caller of the wrapper session determine what actions
  # the wrapper session will perform next, enter a loop to receive instructions
  # from a remote client.
  # For example, TensorBoard visual debugger can use this action so that it can
  # launch session.run() calls remotely.
  REMOTE_INSTR_LOOP = "remote_instr_loop"


class OnSessionInitResponse:
  # [CLASS DOC]
  # Value object returned by the on-session-init callback; carries the action telling the
  # wrapper what to do next.
  """Response from an on-session-init callback."""

  def __init__(self, action):
    # [FUNCTION DOC]
    # Stores the action. Input: action, a string (one of the session-init action constants);
    # raises a type error if not a string. Returns nothing.
    """Constructor.

    Args:
      action: (`OnSessionInitAction`) Debugger action to take on session init.
    """
    _check_type(action, str)
    self.action = action


class OnRunStartRequest:
  # [CLASS DOC]
  # Value object passed to the on-run-start callback. Holds the fetches, feed mapping, run
  # options, run metadata, the one-based run call count and whether a callable runner is used.
  """Request to an on-run-start callback.

  This callback is invoked during a run() call of the debug-wrapper
  session, immediately after the run() call counter is incremented.
  """

  def __init__(self, fetches, feed_dict, run_options, run_metadata,
               run_call_count, is_callable_runner=False):
    # [FUNCTION DOC]
    # Stores its arguments as attributes without validation. Inputs: fetches, feed_dict,
    # run_options, run_metadata (same as a normal run call); run_call_count, one-based integer
    # count of runs including this one; is_callable_runner, boolean, default false, whether a
    # runner made by make-callable is being run. Returns nothing.
    """Constructor of `OnRunStartRequest`.

    Args:
      fetches: Fetch targets of the run() call.
      feed_dict: The feed dictionary to the run() call.
      run_options: RunOptions input to the run() call.
      run_metadata: RunMetadata input to the run() call.
        The above four arguments are identical to the input arguments to the
        run() method of a non-wrapped TensorFlow session.
      run_call_count: 1-based count of how many run calls (including this one)
        has been invoked.
      is_callable_runner: (bool) whether a runner returned by
        Session.make_callable is being run.
    """
    self.fetches = fetches
    self.feed_dict = feed_dict
    self.run_options = run_options
    self.run_metadata = run_metadata
    self.run_call_count = run_call_count
    self.is_callable_runner = is_callable_runner


class OnRunStartAction:
  # [CLASS DOC]
  # Enum-like holder of string constants for what a wrapper does with a run call: debug run with
  # tensor watches, profiled run, or normal run without debugging.
  """Enum-like values for possible action to take on start of a run() call."""

  # Run once with debug tensor-watching.
  DEBUG_RUN = "debug_run"

  # Run once with profiler.
  PROFILE_RUN = "profile_run"

  # Run without debug tensor-watching.
  NON_DEBUG_RUN = "non_debug_run"



class OnRunStartResponse:
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring calls this a "Request"; it is the response returned by the
  # on-run-start callback.
  # Value object telling the wrapper what to do for a run call: the action, the debug URLs, debug
  # ops and the node-name, op-type and tensor-dtype allowlists plus a tolerance flag.
  """Request from an on-run-start callback.

  The caller of the callback can use this response object to specify what
  action the debug-wrapper session actually takes on the run() call.
  """

  def __init__(self,
               action,
               debug_urls,
               debug_ops="DebugIdentity",
               node_name_regex_allowlist=None,
               op_type_regex_allowlist=None,
               tensor_dtype_regex_allowlist=None,
               tolerate_debug_op_creation_failures=False):
    # [FUNCTION DOC]
    # Stores the response fields. Inputs: action, string run-start action (type-checked, raises a
    # type error otherwise); debug_urls, list of strings (type-checked as list); debug_ops, string
    # or list of strings, default is the identity debug op; node_name_regex_allowlist,
    # op_type_regex_allowlist, tensor_dtype_regex_allowlist, optional regular-expression strings;
    # tolerate_debug_op_creation_failures, boolean default false. Returns nothing.
    """Constructor of `OnRunStartResponse`.

    Args:
      action: (`OnRunStartAction`) the action actually taken by the wrapped
        session for the run() call.
      debug_urls: (`list` of `str`) debug_urls used in watching the tensors
        during the run() call.
      debug_ops: (`str` or `list` of `str`) Debug op(s) to be used by the
        debugger.
      node_name_regex_allowlist: Regular-expression allowlist for node
        name.
      op_type_regex_allowlist: Regular-expression allowlist for op type.
      tensor_dtype_regex_allowlist: Regular-expression allowlist for tensor
        dtype.
      tolerate_debug_op_creation_failures: Whether debug op creation failures
        are to be tolerated.
    """

    _check_type(action, str)
    self.action = action

    _check_type(debug_urls, list)
    self.debug_urls = debug_urls

    self.debug_ops = debug_ops

    self.node_name_regex_allowlist = node_name_regex_allowlist
    self.op_type_regex_allowlist = op_type_regex_allowlist
    self.tensor_dtype_regex_allowlist = tensor_dtype_regex_allowlist
    self.tolerate_debug_op_creation_failures = (
        tolerate_debug_op_creation_failures)


class OnRunEndRequest:
  # [CLASS DOC]
  # Value object passed to the on-run-end callback. Holds the action actually performed, the run
  # metadata, the client-side graph definition and any op error captured during the run.
  """Request to an on-run-end callback.

  The callback is invoked immediately before the wrapped run() call ends.
  """

  def __init__(self,
               performed_action,
               run_metadata=None,
               client_graph_def=None,
               tf_error=None):
    # [FUNCTION DOC]
    # Stores the fields. Inputs: performed_action, string (type-checked); run_metadata, optional
    # run-metadata message (type-checked when given); client_graph_def, optional graph definition
    # from the Python front end; tf_error, optional op error raised during the run. Raises a type
    # error on wrong types. Returns nothing.
    """Constructor for `OnRunEndRequest`.

    Args:
      performed_action: (`OnRunStartAction`) Actually-performed action by the
        debug-wrapper session.
      run_metadata: run_metadata output from the run() call (if any).
      client_graph_def: (GraphDef) GraphDef from the client side, i.e., from
        the python front end of TensorFlow. Can be obtained with
        session.graph.as_graph_def().
      tf_error: (errors.OpError subtypes) TensorFlow OpError that occurred
        during the run (if any).
    """

    _check_type(performed_action, str)
    self.performed_action = performed_action

    if run_metadata is not None:
      _check_type(run_metadata, config_pb2.RunMetadata)
    self.run_metadata = run_metadata
    self.client_graph_def = client_graph_def
    self.tf_error = tf_error


class OnRunEndResponse:
  # [CLASS DOC]
  # Placeholder response from the on-run-end callback. Carries no data; no action is taken on it.
  """Response from an on-run-end callback."""

  def __init__(self):
    # [FUNCTION DOC]
    # Does nothing; the response is currently only a placeholder.

    # Currently only a placeholder.
    pass


class BaseDebugWrapperSession(session.SessionInterface, metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base of debug wrapper sessions. Wraps a regular or monitored session, forwards the
  # session interface to it, and intercepts run calls: it invokes the run-start callback, then
  # runs normally, with debug tensor watches, or with profiling depending on the response, then
  # invokes the run-end callback. Can be limited to threads whose names match a pattern. Holds
  # the wrapped session, thread filter pattern, pass-through-error flag, run call counter, a
  # default-session context manager and a cache of callables built from callable options.
  # Subclasses must implement on_session_init, on_run_start and on_run_end.
  """Base class of debug-wrapper session classes.

  Concrete classes that inherit from this class need to implement the abstract
  methods such as on_session_init, on_run_start and on_run_end.
  """

  def __init__(self, sess, thread_name_filter=None,
               pass_through_operrors=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it raises not-implemented for a non-direct session; the code
    # actually raises not-implemented only when the init callback requests the remote instruction
    # loop, and no session-kind check beyond the type check is done.
    # Wraps a session. Inputs: sess, regular or monitored session (type error otherwise);
    # thread_name_filter, optional regular expression matched at the start of thread names where
    # the wrapper is active (all threads if none); pass_through_operrors, boolean, whether op
    # errors propagate (default false). Compiles the filter, resets the run counter, calls the
    # on-session-init callback and checks its response: proceed continues, remote loop raises
    # not-implemented, anything else raises a value error. Initializes the cache of callables.
    """Constructor of `BaseDebugWrapperSession`.

    Args:
      sess: An (unwrapped) TensorFlow session instance. It should be a subtype
        of `BaseSession` or `tf.MonitoredSession`.
      thread_name_filter: Regular-expression filter (allowlist) for name(s) of
        thread(s) on which the wrapper session will be active. This regular
        expression is used in a start-anchored fashion on the thread name, i.e.,
        by applying the `match` method of the compiled pattern. The default
        `None` means that the wrapper session will be active on all threads.
        E.g., r"MainThread$", r"QueueRunnerThread.*".
      pass_through_operrors: If True, all captured OpErrors will be
        propagated.  By default this captures all OpErrors.

    Raises:
      ValueError: On invalid `OnSessionInitAction` value.
      NotImplementedError: If a non-DirectSession sess object is received.
    """

    _check_type(sess, (session.BaseSession, monitored_session.MonitoredSession))

    # The session being wrapped.
    self._sess = sess
    self._thread_name_filter_pattern = (re.compile(thread_name_filter)
                                        if thread_name_filter else None)
    # TODO(cais/kstevens): Unittest this pass through feature.
    self._pass_through_operrors = pass_through_operrors

    # Keeps track of number of run calls that have been performed on this
    # debug-wrapper session. The count can be used for purposes such as
    # displaying the state of the Session in a UI and determining a run
    # number-dependent debug URL.
    self._run_call_count = 0

    # Invoke on-session-init callback.
    response = self.on_session_init(OnSessionInitRequest(self._sess))
    _check_type(response, OnSessionInitResponse)

    if response.action == OnSessionInitAction.PROCEED:
      pass
    elif response.action == OnSessionInitAction.REMOTE_INSTR_LOOP:
      # TODO(cais): Implement REMOTE_INSTR_LOOP
      raise NotImplementedError(
          "OnSessionInitAction REMOTE_INSTR_LOOP has not been "
          "implemented.")
    else:
      raise ValueError(
          "Invalid OnSessionInitAction value: %s" % response.action)

    self._default_session_context_manager = None

    # A cache for callables created from CallableOptions.
    self._cached_callables_from_options = {}

  @property
  def graph(self):
    # [FUNCTION DOC]
    # Returns the graph of the wrapped session.
    return self._sess.graph

  @property
  def graph_def(self):
    # [FUNCTION DOC]
    # Returns the graph definition of the wrapped session.
    return self._sess.graph_def

  @property
  def sess_str(self):
    # [FUNCTION DOC]
    # Returns the connection target string of the wrapped session.
    return self._sess.sess_str

  @property
  def session(self):
    # [FUNCTION DOC]
    # Returns the wrapped (underlying) session object.
    return self._sess

  def run(self,
          fetches,
          feed_dict=None,
          options=None,
          run_metadata=None,
          callable_runner=None,
          callable_runner_args=None,
          callable_options=None):
    # [FUNCTION DOC]
    # Debug-aware replacement for the session run call. Inputs: fetches, feed_dict, options,
    # run_metadata as in a normal run; callable_runner, optional callable from make-callable (then
    # fetches and feed_dict must be empty); callable_runner_args, optional argument list for the
    # runner or callable options; callable_options, optional callable-options message, mutually
    # exclusive with callable_runner. Raises a value error for conflicting arguments or an unknown
    # run-start action. Increments the run counter. If the current thread is filtered out or the
    # fetches are empty (possibly nested), runs the wrapped session directly (logging for empty
    # fetches). Otherwise calls the on-run-start callback and, by the returned action, runs with
    # debugging, with profiling, or normally, then calls the on-run-end callback. Returns the
    # wrapped session's result (or the captured op error object when an op error was captured in
    # a debug run).
    """Wrapper around Session.run() that inserts tensor watch options.

    Args:
      fetches: Same as the `fetches` arg to regular `Session.run()`.
      feed_dict: Same as the `feed_dict` arg to regular `Session.run()`.
      options: Same as the `options` arg to regular `Session.run()`.
      run_metadata: Same as the `run_metadata` arg to regular `Session.run()`.
      callable_runner: A `callable` returned by `Session.make_callable()`.
        If not `None`, `fetches` and `feed_dict` must both be `None`.
        Mutually exclusive with `callable_options`.
      callable_runner_args: An optional list of arguments to `callable_runner`
        or for `callable_options`.
      callable_options: An instance of `config_pb2.CallableOptions`, to be
        used with `Session._make_callable_from_options()`. Mutually exclusive
        with `callable_runner`.

    Returns:
      Simply forwards the output of the wrapped `Session.run()` call.

    Raises:
      ValueError: On invalid `OnRunStartAction` value. Or if `callable_runner`
        is not `None` and either or both of `fetches` and `feed_dict` is `None`.
    """
    if callable_runner and callable_options:
      raise ValueError(
          "callable_runner and callable_options are mutually exclusive, but "
          "are both specified in this call to BaseDebugWrapperSession.run().")

    if callable_runner and (fetches or feed_dict):
      raise ValueError(
          "callable_runner and fetches/feed_dict are mutually exclusive, "
          "but are used simultaneously.")
    elif callable_options and (fetches or feed_dict):
      raise ValueError(
          "callable_options and fetches/feed_dict are mutually exclusive, "
          "but are used simultaneously.")

    self.increment_run_call_count()

    def is_empty(x):
      # [FUNCTION DOC]
      # HELPER: for the run method. Returns true if the input is a nested structure (list, tuple,
      # mapping) whose leaves are all empty containers, recursing into mapping values and
      # items; a non-nested value (even none) yields false.
      """Check whether a possibly nested structure is empty."""
      if not nest.is_nested(x):
        return False
      if isinstance(x, collections_abc.Mapping):
        return is_empty(list(x.values()))
      for item in x:
        if not is_empty(item):
          return False
      return True

    empty_fetches = is_empty(fetches)
    if empty_fetches:
      tf_logging.info(
          "Due to empty fetches, tfdbg Session wrapper is letting a "
          "Session.run pass through without any debugging actions.")
    if self._is_disabled_thread() or empty_fetches:
      if callable_runner:
        return callable_runner(*callable_runner_args)
      elif callable_options:
        # pylint:disable=protected-access
        return self._sess._make_callable_from_options(
            callable_options)(*callable_runner_args)
        # pylint:enable=protected-access
      else:
        return self._sess.run(fetches,
                              feed_dict=feed_dict,
                              options=options,
                              run_metadata=run_metadata)

    # Invoke on-run-start callback and obtain response.
    run_start_resp = self.on_run_start(
        OnRunStartRequest(fetches, feed_dict, options, run_metadata,
                          self._run_call_count,
                          is_callable_runner=bool(callable_runner)))
    _check_type(run_start_resp, OnRunStartResponse)

    if run_start_resp.action == OnRunStartAction.DEBUG_RUN:
      retvals, run_end_req = self._run_with_debugging(
          run_start_resp, fetches, feed_dict, options, run_metadata,
          callable_runner, callable_runner_args, callable_options)
    elif run_start_resp.action == OnRunStartAction.PROFILE_RUN:
      retvals, run_end_req = self._run_with_profiling(
          run_start_resp, fetches, feed_dict, options, run_metadata,
          callable_runner, callable_runner_args, callable_options)
    elif run_start_resp.action == OnRunStartAction.NON_DEBUG_RUN:
      # Invoke run() method of the wrapped session.
      if callable_runner:
        retvals = callable_runner(*callable_runner_args)
      elif callable_options:
        # pylint:disable=protected-access
        callable_object = self._sess._make_callable_from_options(
            callable_options)
        # pylint:enable=protected-access
        retvals = callable_object(*callable_runner_args)
      else:
        retvals = self._sess.run(
            fetches,
            feed_dict=feed_dict,
            options=options,
            run_metadata=run_metadata)

      # Prepare arg for the on-run-end callback.
      run_end_req = OnRunEndRequest(run_start_resp.action)
    else:
      raise ValueError(
          "Invalid OnRunStartAction value: %s" % run_start_resp.action)

    # Invoke on-run-end callback and obtain response.
    run_end_resp = self.on_run_end(run_end_req)
    _check_type(run_end_resp, OnRunEndResponse)
    # Currently run_end_resp is only a placeholder. No action is taken on it.

    return retvals

  def _run_with_debugging(self,
                          run_start_resp,
                          fetches,
                          feed_dict,
                          options,
                          run_metadata,
                          callable_runner,
                          callable_runner_args,
                          callable_options):
    # [FUNCTION DOC]
    # HELPER: debug-run implementation for the run method.
    # KNOWN ISSUE: with callable options already seen (cached), no new options copy is made and
    # the decorated run options stay empty, so no watches are added for the cached callable.
    # Inputs: run_start_resp, the run-start response (debug URLs, ops, allowlists); the remaining
    # arguments are those of the run method. Prepares run options (copying callable options to
    # avoid mutating them), decorates them with tensor watches, then runs the callable runner, the
    # cached or newly made callable, or the wrapped session. Op errors are re-raised if the
    # pass-through flag is set, otherwise captured. Returns a pair: the results (or the error
    # object) and a run-end request holding action, metadata, client graph definition and error.
    """Perform a session.run() or callable with debugging."""
    # Decorate RunOption to fill in debugger tensor watch specifications.
    decorated_run_options = None
    if callable_options:
      callable_options_id = id(callable_options)
      if callable_options_id not in self._cached_callables_from_options:
        # Make a copy of callable_options to avoid mutating it.
        new_callable_options = config_pb2.CallableOptions()
        new_callable_options.CopyFrom(callable_options)
        decorated_run_options = new_callable_options.run_options
    else:
      decorated_run_options = options or config_pb2.RunOptions()

    run_metadata = run_metadata or config_pb2.RunMetadata()

    if decorated_run_options:
      self._decorate_run_options_for_debug(
          decorated_run_options,
          run_start_resp.debug_urls,
          debug_ops=run_start_resp.debug_ops,
          node_name_regex_allowlist=(run_start_resp.node_name_regex_allowlist),
          op_type_regex_allowlist=run_start_resp.op_type_regex_allowlist,
          tensor_dtype_regex_allowlist=(
              run_start_resp.tensor_dtype_regex_allowlist),
          tolerate_debug_op_creation_failures=(
              run_start_resp.tolerate_debug_op_creation_failures))

    # Invoke the run() method of the wrapped Session. Catch any TensorFlow
    # runtime errors.
    tf_error = None
    try:
      if callable_runner:
        retvals = callable_runner(*callable_runner_args,
                                  options=decorated_run_options,
                                  run_metadata=run_metadata)
      elif callable_options:
        # pylint:disable=protected-access
        if callable_options_id in self._cached_callables_from_options:
          callable_object = self._cached_callables_from_options[
              callable_options_id]
        else:
          callable_object = self._sess._make_callable_from_options(
              new_callable_options)
          self._cached_callables_from_options[
              callable_options_id] = callable_object
        # pylint:enable=protected-access
        retvals = callable_object(
            *callable_runner_args, run_metadata=run_metadata)
      else:
        retvals = self._sess.run(fetches,
                                 feed_dict=feed_dict,
                                 options=decorated_run_options,
                                 run_metadata=run_metadata)
    except errors.OpError as op_error:
      if self._pass_through_operrors:
        raise op_error
      tf_error = op_error
      retvals = op_error

    return retvals, OnRunEndRequest(
        run_start_resp.action,
        run_metadata=run_metadata,
        client_graph_def=self._sess.graph.as_graph_def(),
        tf_error=tf_error)

  def _run_with_profiling(self,
                          run_start_resp,
                          fetches,
                          feed_dict,
                          options,
                          run_metadata,
                          callable_runner,
                          callable_runner_args,
                          callable_options):
    # [FUNCTION DOC]
    # HELPER: profile-run implementation for the run method.
    # KNOWN ISSUE: for callable options already cached, decorated run options remain empty (the
    # profile decoration is then applied to nothing) and the options copy used later is undefined,
    # so that path would fail. Also the code comment mentions debugger watches but it only sets
    # full tracing.
    # Inputs as for the debug-run helper. Prepares run options (copy of callable options if
    # given), turns on full tracing, then runs the callable runner, callable or wrapped session.
    # Returns a pair: the results and a run-end request with action, metadata and client graph
    # definition. Op errors are not captured here.
    """Perform a session.run() or callable with profiling."""
    # Decorate RunOption to fill in debugger tensor watch specifications.
    decorated_run_options = None
    if callable_options:
      callable_options_id = id(callable_options)
      if callable_options_id not in self._cached_callables_from_options:
        # Make a copy of callable_options to avoid mutating it.
        new_callable_options = config_pb2.CallableOptions()
        new_callable_options.CopyFrom(callable_options)
        decorated_run_options = new_callable_options.run_options
    else:
      decorated_run_options = options or config_pb2.RunOptions()
    self._decorate_run_options_for_profile(decorated_run_options)

    run_metadata = run_metadata or config_pb2.RunMetadata()
    if callable_runner:
      retvals = callable_runner(*callable_runner_args,
                                options=decorated_run_options,
                                run_metadata=run_metadata)
    elif callable_options:
      # pylint:disable=protected-access
      callable_object = self._sess._make_callable_from_options(
          new_callable_options)
      # pylint:enable=protected-access
      retvals = callable_object(
          *callable_runner_args, run_metadata=run_metadata)
    else:
      retvals = self._sess.run(fetches,
                               feed_dict=feed_dict,
                               options=decorated_run_options,
                               run_metadata=run_metadata)
    return retvals, OnRunEndRequest(
        run_start_resp.action,
        run_metadata=run_metadata,
        client_graph_def=self._sess.graph.as_graph_def())

  def _is_disabled_thread(self):
    # [FUNCTION DOC]
    # HELPER: for the run method. Returns a true-like value if a thread filter is set and the
    # current thread's name does not match it from the start; otherwise false-like (including
    # when no filter is set, in which case it returns nothing/null).
    thread_name = threading.current_thread().name or ""
    return (self._thread_name_filter_pattern and
            not self._thread_name_filter_pattern.match(thread_name))

  def run_step_fn(self, step_fn):
    # [FUNCTION DOC]
    # Runs a step function with a step context built from the wrapped session and this wrapper's
    # run method, so runs inside the step go through the debug logic. Input: step_fn, callable
    # taking a step context. Returns its result.
    return step_fn(
        monitored_session.MonitoredSession.StepContext(self._sess, self.run))

  def partial_run_setup(self, fetches, feeds=None):
    # [FUNCTION DOC]
    # Always raises a not-implemented error: partial runs are unsupported by debug wrappers.
    """Sets up the feeds and fetches for partial runs in the session."""
    raise NotImplementedError(
        "partial_run_setup is not implemented for debug-wrapper sessions.")

  def partial_run(self, handle, fetches, feed_dict=None):
    # [FUNCTION DOC]
    # Always raises a not-implemented error: partial runs are unsupported by debug wrappers.
    raise NotImplementedError(
        "partial_run is not implemented for debug-wrapper sessions.")

  def list_devices(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments to the wrapped session's device listing and returns its result.
    return self._sess.list_devices(*args, **kwargs)

  def reset(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forwards all arguments to the wrapped session's reset and returns its result.
    return self._sess.reset(*args, **kwargs)

  def make_callable(self,
                    fetches,
                    feed_list=None,
                    accept_options=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the accept_options argument is ignored; the underlying callable is always made
    # to accept options so the wrapper can pass watch options.
    # Creates a callable from the wrapped session for the given fetches and feed list, and returns
    # a wrapper function that, when called, goes through this wrapper's run method with the
    # underlying callable as the runner (forwarding optional options and run metadata keywords).
    runner = self._sess.make_callable(
        fetches, feed_list=feed_list, accept_options=True)
    def wrapped_runner(*runner_args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: callable returned by make-callable. Calls the wrapper's run method with no fetches
      # or feeds, passing the underlying runner, the positional arguments, and the options and
      # run-metadata keywords if present. Returns the run result.
      return self.run(None,
                      feed_dict=None,
                      options=kwargs.get("options", None),
                      run_metadata=kwargs.get("run_metadata", None),
                      callable_runner=runner,
                      callable_runner_args=runner_args)
    return wrapped_runner

  def _make_callable_from_options(self, callable_options):
    # [FUNCTION DOC]
    # Returns a function that runs the given callable-options message through this wrapper's run
    # method (so debugging applies). Input: callable_options, a callable-options message.
    def wrapped_runner(*feed_values, **kwargs):
      # [FUNCTION DOC]
      # HELPER: callable returned by the method above. Runs through the wrapper's run method with
      # the feed values as runner arguments and an optional run-metadata keyword. Returns the
      # result.
      return self.run(None,
                      run_metadata=kwargs.get("run_metadata", None),
                      callable_options=callable_options,
                      callable_runner_args=feed_values)
    return wrapped_runner

  @property
  def run_call_count(self):
    # [FUNCTION DOC]
    # Returns the number of run calls performed on this wrapper so far.
    return self._run_call_count

  def increment_run_call_count(self):
    # [FUNCTION DOC]
    # Increments the run call counter by one. Returns nothing.
    self._run_call_count += 1

  def _is_disk_usage_reset_each_run(self):
    # [FUNCTION DOC]
    # HELPER: hook for subclasses. Returns false by default; subclasses that clean up dumped data
    # after every run should override it to return true.
    """Indicates whether disk usage is reset after each Session.run.

    Subclasses that clean up the disk usage after every run should
    override this protected method.

    Returns:
      (`bool`) Whether the disk usage amount is reset to zero after
        each Session.run.
    """
    return False

  def _decorate_run_options_for_debug(
      self,
      run_options,
      debug_urls,
      debug_ops="DebugIdentity",
      node_name_regex_allowlist=None,
      op_type_regex_allowlist=None,
      tensor_dtype_regex_allowlist=None,
      tolerate_debug_op_creation_failures=False):
    # [FUNCTION DOC]
    # HELPER: for the debug-run helper.
    # Mutates the given run options: requests partition graphs and adds tensor watches over the
    # wrapped session's graph using the debug URLs, debug ops and node-name, op-type and
    # tensor-dtype allowlists. Inputs: run_options, run-options message (modified in place);
    # debug_urls, list of strings; debug_ops, string or list (default identity op); the three
    # allowlists, optional regular expressions; tolerate_debug_op_creation_failures, boolean.
    # Disk byte usage is reset on the first run or when the subclass says to reset every run.
    # Returns nothing.
    """Modify a RunOptions object for debug tensor watching.

    Specifies request for outputting partition graphs. Adds
    debug_tensor_watch_opts with proper debug URLs.

    Args:
      run_options: (RunOptions) the modified RunOptions object.
      debug_urls: (list of str) debug URLs to be entered in run_options.
        debug_tensor_watch_opts.
      debug_ops: (str or list of str) debug op(s) to be used by the debugger.
      node_name_regex_allowlist: Regular-expression allowlist for node
        name.
      op_type_regex_allowlist: Regular-expression allowlist for op type.
      tensor_dtype_regex_allowlist: Regular-expression allowlist for tensor
        dtype.
      tolerate_debug_op_creation_failures: Whether debug op creation failures
        are to be tolerated.
    """

    run_options.output_partition_graphs = True
    debug_utils.watch_graph(
        run_options,
        self._sess.graph,
        debug_urls=debug_urls,
        debug_ops=debug_ops,
        node_name_regex_allowlist=node_name_regex_allowlist,
        op_type_regex_allowlist=op_type_regex_allowlist,
        tensor_dtype_regex_allowlist=tensor_dtype_regex_allowlist,
        tolerate_debug_op_creation_failures=tolerate_debug_op_creation_failures,
        reset_disk_byte_usage=(self._run_call_count == 1 or
                               self._is_disk_usage_reset_each_run()))

  def _decorate_run_options_for_profile(self, run_options):
    # [FUNCTION DOC]
    # HELPER: for the profile-run helper. Sets the trace level of the given run options (modified
    # in place) to full tracing. Returns nothing.
    """Modify a RunOptions object for profiling TensorFlow graph execution.

    Args:
      run_options: (RunOptions) the modified RunOptions object.
    """

    run_options.trace_level = config_pb2.RunOptions.FULL_TRACE

  @abc.abstractmethod
  def on_session_init(self, request):
    # [FUNCTION DOC]
    # Abstract callback invoked at the end of construction. Input: an on-session-init request
    # holding the wrapped session. Subclasses return an on-session-init response with an action.
    """Callback invoked during construction of the debug-wrapper session.

    This is a blocking callback.
    The invocation happens right before the constructor ends.

    Args:
      request: (`OnSessionInitRequest`) callback request carrying information
        such as the session being wrapped.

    Returns:
      An instance of `OnSessionInitResponse`.
    """

  @abc.abstractmethod
  def on_run_start(self, request):
    # [FUNCTION DOC]
    # Abstract callback invoked at the start of each (non-bypassed) run. Input: an on-run-start
    # request. Subclasses return an on-run-start response saying which action to take and with
    # what debug URLs and watches.
    """Callback invoked on run() calls to the debug-wrapper session.

    This is a blocking callback.
    The invocation happens after the wrapper's run() call is entered,
    after an increment of run call counter.

    Args:
      request: (`OnRunStartRequest`) callback request object carrying
        information about the run call such as the fetches, feed dict, run
        options, run metadata, and how many `run()` calls to this wrapper
        session have occurred.

    Returns:
      An instance of `OnRunStartResponse`, carrying information to
        debug URLs used to watch the tensors.
    """

  @abc.abstractmethod
  def on_run_end(self, request):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns an on-run-start response; the run method checks
    # for and requires an on-run-end response.
    # Abstract callback invoked right before a (non-bypassed) run returns. Input: an on-run-end
    # request with the performed action, metadata and any captured error. Subclasses return an
    # on-run-end response (currently a placeholder).
    """Callback invoked on run() calls to the debug-wrapper session.

    This is a blocking callback.
    The invocation happens right before the wrapper exits its run() call.

    Args:
      request: (`OnRunEndRequest`) callback request object carrying information
        such as the actual action performed by the session wrapper for the
        run() call.

    Returns:
      An instance of `OnRunStartResponse`.
    """

  def as_default(self):
    # [FUNCTION DOC]
    # Returns a context manager that makes this wrapper the default session.
    return stack.default_session(self)

  def __enter__(self):
    # [FUNCTION DOC]
    # Enters the default-session context for this wrapper (creating the context manager on first
    # use) and returns what that context manager returns.
    if self._default_session_context_manager is None:
      self._default_session_context_manager = self.as_default()
    return self._default_session_context_manager.__enter__()

  def __exit__(self, exec_type, exec_value, exec_tb):
    # [FUNCTION DOC]
    # Exits the default-session context, passing the exception information through. Returns
    # nothing, so exceptions are not suppressed.
    self._default_session_context_manager.__exit__(
        exec_type, exec_value, exec_tb)

  def __del__(self):
    # [FUNCTION DOC]
    # On destruction, calls the wrapped session's own destructor if it has one. Returns nothing.
    if hasattr(self._sess, "__del__"):
      self._sess.__del__()

  def close(self):
    # [FUNCTION DOC]
    # Closes the wrapped session.
    self._sess.close()

  # TODO(cais): Add _node_name_regex_allowlist and
  #   _node_op_type_regex_allowlist.

  def should_stop(self):
    # [FUNCTION DOC]
    # Forwards the stop query to the wrapped session and returns its answer. Raises a value error
    # if the wrapped session lacks a should-stop method (for example a plain session rather than
    # a monitored session).
    if hasattr(self._sess, "should_stop"):
      return self._sess.should_stop()
    else:
      raise ValueError(
          "The wrapped session %r does not have a method called 'should_stop'. "
          "Do you intend to wrap a tf.MonitoredSession instead?" % self._sess)


class WatchOptions:
  # [CLASS DOC]
  # Value type returned by watch callbacks: debug ops plus allowlists (node name, op type, tensor
  # dtype; combined with logical and) and a flag for tolerating debug-op creation failures.
  """Type for return values of watch_fn."""

  def __init__(self,
               debug_ops=None,
               node_name_regex_allowlist=None,
               op_type_regex_allowlist=None,
               tensor_dtype_regex_allowlist=None,
               tolerate_debug_op_creation_failures=False):
    # [FUNCTION DOC]
    # Stores the options. Inputs: debug_ops, string or list of strings; if empty or none, defaults
    # to a list holding the identity debug op. The three allowlists are optional regular
    # expression strings; tolerate_debug_op_creation_failures is a boolean, default false.
    # Returns nothing.
    """Constructor of WatchOptions: Debug watch options.

    Used as return values of `watch_fn`s.

    Args:
      debug_ops: (`str` or `list of str`) Debug ops to be used.
      node_name_regex_allowlist: Regular-expression allowlist for node_name,
        e.g., `"(weight_[0-9]+|bias_.*)"`
      op_type_regex_allowlist: Regular-expression allowlist for the op type of
        nodes, e.g., `"(Variable|Add)"`.
        If both `node_name_regex_allowlist` and `op_type_regex_allowlist`
        are set, the two filtering operations will occur in a logical `AND`
        relation. In other words, a node will be included if and only if it
        hits both allowlists.
      tensor_dtype_regex_allowlist: Regular-expression allowlist for Tensor
        data type, e.g., `"^int.*"`.
        This allowlist operates in logical `AND` relations to the two allowlists
        above.
      tolerate_debug_op_creation_failures: (`bool`) whether debug op creation
        failures (e.g., due to dtype incompatibility) are to be tolerated by not
        throwing exceptions.
    """
    if debug_ops:
      self.debug_ops = debug_ops
    else:
      self.debug_ops = ["DebugIdentity"]
    self.node_name_regex_allowlist = node_name_regex_allowlist
    self.op_type_regex_allowlist = op_type_regex_allowlist
    self.tensor_dtype_regex_allowlist = tensor_dtype_regex_allowlist
    self.tolerate_debug_op_creation_failures = (
        tolerate_debug_op_creation_failures)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a readable string listing all five option values.
    return ("WatchOptions(debug_ops=%r, node_name_regex_allowlist=%r, "
            "op_type_regex_allowlist=%r, tensor_dtype_regex_allowlist=%r, "
            "tolerate_debug_op_creation_failures=%r)" %
            (self.debug_ops, self.node_name_regex_allowlist,
             self.op_type_regex_allowlist, self.tensor_dtype_regex_allowlist,
             self.tolerate_debug_op_creation_failures))


class NonInteractiveDebugWrapperSession(BaseDebugWrapperSession):
  # [CLASS DOC]
  # Base class for wrapper sessions without a user interface. Every run is a debug run: it asks
  # the subclass for run-specific debug URLs and consults an optional watch callback (returning
  # watch options or a legacy tuple) to decide what to watch. Subclasses implement the method
  # that prepares debug URLs. Holds the optional watch callback.
  """Base class for non-interactive (i.e., non-CLI) debug wrapper sessions."""

  def __init__(self, sess, watch_fn=None, thread_name_filter=None,
               pass_through_operrors=False):
    # [FUNCTION DOC]
    # Constructs the session. Inputs: sess, session to wrap; watch_fn, optional callback taking
    # fetches and feeds and returning watch options (or a legacy tuple of options); thread name
    # filter and pass-through flag as in the base class (defaults none and false). Calls the base
    # constructor, then stores the callback. Raises a type error if a callback is given but is not
    # callable.
    """Constructor of NonInteractiveDebugWrapperSession.

    Args:
      sess: The TensorFlow `Session` object being wrapped.
      watch_fn: (`Callable`) A Callable that maps the fetches and feeds of a
        debugged `Session.run()` call to `WatchOptions.`
        * Args:
          * `fetches`: the fetches to the `Session.run()` call.
          * `feeds`: the feeds to the `Session.run()` call.

        * Returns:
         (`tf_debug.WatchOptions`) An object containing debug options including
           the debug ops to use, the node names, op types and/or tensor data
           types to watch, etc. See the documentation of `tf_debug.WatchOptions`
           for more details.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
      pass_through_operrors: If true, all captured OpErrors will be
        propagated.  By default this captures all OpErrors.
    Raises:
       TypeError: If a non-None `watch_fn` is specified and it is not callable.
    """

    BaseDebugWrapperSession.__init__(
        self, sess, thread_name_filter=thread_name_filter,
        pass_through_operrors=pass_through_operrors)

    self._watch_fn = None
    if watch_fn is not None:
      if not callable(watch_fn):
        raise TypeError("watch_fn is not callable")
      self._watch_fn = watch_fn

  def on_session_init(self, request):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring refers to the on-run-start method; it should refer to the
    # on-session-init callback.
    # Ignores the request and returns a session-init response with the proceed action.
    """See doc of BaseDebugWrapperSession.on_run_start."""

    return OnSessionInitResponse(OnSessionInitAction.PROCEED)

  @abc.abstractmethod
  def prepare_run_debug_urls(self, fetches, feed_dict):
    # [FUNCTION DOC]
    # Abstract method for subclasses. Inputs: fetches and feed_dict of the run call. Returns the
    # run-specific debug URL(s), a string or list of strings.
    """Abstract method to be implemented by concrete subclasses.

    This method prepares the run-specific debug URL(s).

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`
      feed_dict: Same as the `feed_dict` argument to `Session.run()`

    Returns:
      debug_urls: (`str` or `list` of `str`) Debug URLs to be used in
        this `Session.run()` call.
    """

  def on_run_start(self, request):
    # [FUNCTION DOC]
    # Gets the debug URLs and watch options for the run's fetches and feeds, and returns a
    # run-start response with the debug-run action, those URLs and the watch options' debug ops,
    # allowlists and tolerance flag.
    """See doc of BaseDebugWrapperSession.on_run_start."""

    debug_urls, watch_opts = self._prepare_run_watch_config(
        request.fetches, request.feed_dict)

    return OnRunStartResponse(
        OnRunStartAction.DEBUG_RUN,
        debug_urls,
        debug_ops=watch_opts.debug_ops,
        node_name_regex_allowlist=watch_opts.node_name_regex_allowlist,
        op_type_regex_allowlist=watch_opts.op_type_regex_allowlist,
        tensor_dtype_regex_allowlist=watch_opts.tensor_dtype_regex_allowlist,
        tolerate_debug_op_creation_failures=(
            watch_opts.tolerate_debug_op_creation_failures))

  def _prepare_run_watch_config(self, fetches, feed_dict):
    # [FUNCTION DOC]
    # HELPER: for the on-run-start callback. Obtains debug URLs from the subclass method, then
    # watch options: defaults if no callback, otherwise the callback's result (a legacy tuple is
    # converted into watch options). Returns a pair: debug URLs and watch options.
    """Get the debug_urls, and node/op allowlists for the current run() call.

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`.
      feed_dict: Same as the `feed_dict argument` to `Session.run()`.

    Returns:
      debug_urls: (str or list of str) Debug URLs for the current run() call.
        Currently, the list consists of only one URL that is a file:// URL.
      watch_options: (WatchOptions) The return value of a watch_fn, containing
        options including debug_ops, and allowlists.
    """

    debug_urls = self.prepare_run_debug_urls(fetches, feed_dict)
    if self._watch_fn is None:
      watch_options = WatchOptions()
    else:
      watch_options = self._watch_fn(fetches, feed_dict)
      if isinstance(watch_options, tuple):
        # For legacy return type (tuples).
        watch_options = WatchOptions(*watch_options)

    return debug_urls, watch_options

  def on_run_end(self, request):
    # [FUNCTION DOC]
    # Ignores the request and returns an empty run-end response.
    """See doc of BaseDebugWrapperSession.on_run_end."""

    return OnRunEndResponse()
