# [MODULE DOC]
# Implements ProfileContext, a context manager that monkey-patches the graph-mode Session class so
# that every session run is counted, selectively traced with full tracing, fed into a profiler
# object, automatically profiled at chosen steps and dumped to a profile directory.
# Exposes the ProfileContext class plus the module constants for the warm-up step count and the
# maximum number of traced steps, and two private replacement functions for session init and run.
# Import-time side effects: none beyond imports; patching of the session class happens only when a
# context is entered and is undone on exit.
# Relies on the model analyzer Profiler and on the native tfprof wrapper for deleting the profiler.
# Public entry point is exposed to users through the profiler package.
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
"""A Context that captures profile and performs profiling/dumping.
"""
import contextlib
import os
import random
import sys
import threading

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.util import _pywrap_tfprof as print_mdl
from tensorflow.python.util import compat

WARMUP_STEPS = 10
MAX_TRACED_STEPS = 100


def _profiled_init(self, target='', graph=None, config=None):
  # [FUNCTION DOC]
  # HELPER: Replacement session constructor installed by ProfileContext while a context is active.
  # Simply forwards the target, graph and config arguments to the saved original constructor
  # (stored on the session class under an internal name), so session creation behaves as before.
  # The call is made so that the session object gets its normal initialization; returns nothing.
  # Exists only so the original constructor can be swapped in and out reversibly.
  """Overwrites the session.__init__."""
  self._profiler_init_internal(target, graph, config)  # pylint: disable=protected-access


def _profiled_run(self,
                  fetches,
                  feed_dict=None,
                  options=None,
                  run_metadata=None):
  # [FUNCTION DOC]
  # HELPER: Replacement for the session run method installed by ProfileContext while active.
  # KNOWN ISSUE: On the non-traced slow path the original run is called without the caller's
  # run_metadata argument, so a caller-supplied run_metadata is not filled for those steps.
  # KNOWN ISSUE: When the caller passes no options, the saved "old trace level" is read after the
  # new options were created with full tracing, so restoring it is a no-op (harmless).
  # Wraps each run in the context's step counter. Arguments are the same as the real run:
  # fetches, optional feed mapping, optional run options and optional run metadata.
  # If the step is not on the fast path, decides whether to trace: when tracing, it creates
  # metadata/options if missing, forces full trace level, runs the original run, optionally writes
  # the raw metadata to the profile directory in debug mode, points the profiler at the session
  # graph and adds the step's metadata to it, then restores the caller's trace level.
  # Afterwards it may dump the profile and executes every auto-profiling command scheduled for the
  # step (graph, scope, op or code views), saving each result per command and step in the
  # context's view store. Raises a value error for an unknown profiling command.
  # If the lock is already held (concurrent run) it just calls the original run directly.
  # Returns the fetched values of the original run. Writes debug messages to standard error.
  """Overwrites the session.run()."""
  # pylint: disable=protected-access
  # Count the session steps.
  with self.profile_context._new_step() as state:
    step, locked = state
    # Fast path if no need for profiling.
    if locked and not self.profile_context._is_fast_path(step):
      # Maybe trace this step.
      if self.profile_context._should_trace(step, self.graph, fetches):
        if self.profile_context._debug:
          sys.stderr.write('debug: tracing step: %d\n' % step)
        # Enable tracing, perform auto profiling or auto dump.
        if not run_metadata:
          run_metadata = config_pb2.RunMetadata()

        if not options:
          options = config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE)
          old_trace_level = options.trace_level
        else:
          old_trace_level = options.trace_level
          options.trace_level = config_pb2.RunOptions.FULL_TRACE

        ret = self._profiler_run_internal(
            fetches, feed_dict, options, run_metadata)
        if self.profile_context._debug:
          self.profile_context._dump_file(run_metadata, 'run_meta_%d' % step)

        self.profile_context.profiler._graph = self.graph
        self.profile_context.profiler.add_step(step, run_metadata)
        options.trace_level = old_trace_level
      else:
        ret = self._profiler_run_internal(fetches, feed_dict, options)

      # Maybe dump profile.
      self.profile_context._maybe_dump(step)

      # Maybe profile:
      to_profiles = self.profile_context._profile_candidates()
      for to_prof in to_profiles:
        cmd, opts, _ = to_prof
        saved_views = self.profile_context._views.setdefault(cmd, {})
        if self.profile_context._debug:
          sys.stderr.write('debug: profiling %s step: %d\n' % (cmd, step))
        if cmd == 'graph':
          saved_views[step] = self.profile_context.profiler.profile_graph(opts)
        elif cmd == 'scope':
          saved_views[step] = self.profile_context.profiler.profile_name_scope(
              opts)
        elif cmd == 'op':
          saved_views[step] = self.profile_context.profiler.profile_operations(
              opts)
        elif cmd == 'code':
          saved_views[step] = self.profile_context.profiler.profile_python(opts)
        else:
          raise ValueError('Unknown cmd: %s\n' % cmd)
      return ret
  # Fast no lock path.
  return self._profiler_run_internal(
      fetches, feed_dict, options, run_metadata)
  # pylint: enable=protected-access


class ProfileContext(object):
  # [CLASS DOC]
  # Context manager that enables step-aware tracing and profiling of graph-mode session runs.
  # On entering, it replaces the session class run and constructor with wrappers (and attaches
  # itself to the session class); on exit it undoes this and deletes the native profiler.
  # State: profile directory, debug flag, enabled flag, sets of trace/dump/slow-path steps, an
  # auto-tracing flag (used when no trace steps are given), the list of auto-profiling commands,
  # a lazily created Profiler, a mapping of saved profile views per command and step, a lock
  # guarding concurrent runs, a seeded random generator and a step counter.
  # When disabled, constructor returns early and all public methods become near no-ops.
  """A Context that captures RunMetadata and performs profiling.

  ```python
    # Trace steps 100~200, profile at [150, 200] and dump profile at 200.
    with profile_context.ProfileContext('/tmp/train_dir',
                                        trace_steps=range(100, 200, 3),
                                        dump_steps=[200]) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling('op', opts, [150, 200])
      train_loop().

    # Tracing only.
    with profile_context.tfprof.ProfileContext('/tmp/train_dir') as pctx:
      # Run train/eval loop for at least few hundred steps. Profiles will be
      # dumped to train_dir. Use web UI or command line to do profiling.
      train_loop().

    # When session object is available, do explicit trace, profile and dump.
    with profile_context.ProfileContext('/tmp/train_dir',
                                        trace_steps=[],
                                        dump_steps=[]) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.trace_next_step()
      _ = session.run(train_op)
      pctx.profiler.profile_operations(options=opts)
  ```

  Args:
    profile_dir: Directory to store profiles.
    trace_steps: A list of session run steps to trace. If None, use
        pre-defined steps.
    dump_steps: A list of steps to dump the profile to `profile_dir`. If None,
        use pre-defined steps.
    enabled: If false, everything is disabled with minimal overhead. It allows
        user to only enable profiling when needed.
    debug: If true, also dumps the raw trace RunMetadata text file to
        profile_dir. And print debugging message. Useful for bug report.
  """

  def __init__(self,
               profile_dir,
               trace_steps=None,
               dump_steps=None,
               enabled=True,
               debug=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: When enabled is false the constructor returns before creating most attributes
    # (for example the saved views), so get_profiles on a disabled context fails with an
    # attribute error instead of the documented value error.
    # Creates the context. Inputs: profile_dir (required directory for dumped profiles),
    # trace_steps (optional list of step numbers to trace; nothing means automatic tracing),
    # dump_steps (optional list of steps at which to dump; nothing means one dump at the maximum
    # traced step count), enabled (default true) and debug (default false).
    # Raises a value error if profile_dir is empty or more than the maximum number of trace steps
    # is given. Copies the step lists into sets, initializes counters, lock and views.
    self._enabled = enabled
    if not self._enabled:
      return

    self._debug = debug
    if not profile_dir:
      raise ValueError('Must have a directory for profile.\n')
    self._profiler_dir = profile_dir

    if trace_steps is None:
      self._trace_steps = set()
      self._auto_tracing = True
    else:
      if len(trace_steps) > MAX_TRACED_STEPS:
        raise ValueError('Only support tracing up to 100 steps.\n')
      self._trace_steps = set(trace_steps[:])
      self._auto_tracing = False

    if dump_steps is None:
      self._dump_steps = set([MAX_TRACED_STEPS])
    else:
      self._dump_steps = set(dump_steps[:])

    self._rng = random.Random(111)
    self._fetched = set()
    self._slow_path_steps = self._dump_steps | self._trace_steps
    self._trace_next_step = False
    self._dump_next_step = False
    self._step = 0
    self._traced_steps = 0
    self._auto_profiles = []
    self._profiler = None
    self._views = {}
    self._lock = threading.Lock()

  def get_profiles(self, cmd):
    # [FUNCTION DOC]
    # Looks up saved auto-profiling results for one command. Input: cmd, the command name that
    # was registered through add_auto_profiling. Returns a mapping from step number to the
    # profile result recorded at that step. Raises a value error if no result exists for cmd.
    """Returns profiling results for each step at which `cmd` was run.

    Args:
      cmd: string, profiling command used in an `add_auto_profiling` call.

    Returns:
      dict[int: (MultiGraphNodeProto | GraphNodeProto)]. Keys are steps at which
      the profiling command was run. Values are the outputs of profiling.
      For "code" and "op" commands this will be a `MultiGraphNodeProto`, for
      "scope" and "graph" commands this will be a `GraphNodeProto.

    Raises:
      ValueError: if `cmd` was never run (either because no session.run call was
      made or because there was no `add_auto_profiling` call with the specified
      `cmd`.
    """
    if cmd not in self._views:
      raise ValueError('No autoprofiler for command: {}, was run'.format(cmd))
    return self._views[cmd]

  def add_auto_profiling(self, cmd, options, profile_steps):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring lists python as a command, but the run wrapper only recognizes
    # graph, scope, op and code; any other name raises a value error at run time.
    # Schedules a profiling command. Inputs: cmd (view name), options (profiling options) and
    # profile_steps (list of step numbers). Stores a copy of the steps, and adds them to both the
    # slow-path and trace step sets so those steps are traced. Returns nothing; does nothing
    # when the context is disabled.
    """Traces and profiles at some session run steps.

    Args:
      cmd: The profiling commands. (i.e. scope, op, python, graph)
      options: The profiling options.
      profile_steps: A list/set of integers. The profiling command and options
          will be run automatically at these integer steps. Each step is
          a session.run.
    """
    if not self._enabled:
      return
    self._auto_profiles.append((cmd, options, profile_steps[:]))
    self._slow_path_steps |= set(profile_steps)
    self._trace_steps |= set(profile_steps)

  @property
  def profiler(self):
    # [FUNCTION DOC]
    # Property giving the Profiler object of this context. Returns nothing (null) if the context
    # is disabled; otherwise lazily creates a Profiler for the default graph on first access and
    # returns the same object afterwards. Side effect: creates the profiler on first use.
    """Returns the current profiler object."""
    if not self._enabled:
      return None
    if not self._profiler:
      self._profiler = model_analyzer.Profiler(ops.get_default_graph())
    return self._profiler

  def trace_next_step(self):
    # [FUNCTION DOC]
    # Requests full tracing (and adding the trace to the profiler) of the next session run.
    # Sets a one-shot flag and marks the current step as a slow-path step. No inputs, returns
    # nothing; does nothing if disabled. The flag is cleared when the step finishes.
    """Enables tracing and adds traces to profiler at next step."""
    if not self._enabled:
      return
    self._trace_next_step = True
    self._slow_path_steps.add(self._step)

  def dump_next_step(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says tracing is enabled too, but this only sets the dump flag and
    # slow-path marker; the step is traced only if tracing was otherwise requested.
    # Requests that the profile be dumped to the profile directory after the next session run.
    # No inputs, returns nothing; does nothing if disabled. The flag is cleared after the step.
    """Enable tracing and dump profiles at next step."""
    if not self._enabled:
      return
    self._dump_next_step = True
    self._slow_path_steps.add(self._step)

  def _is_fast_path(self, step):
    # [FUNCTION DOC]
    # HELPER: Used by the patched session run to skip all profiling work cheaply.
    # KNOWN ISSUE: The auto-tracing limit here uses "at most" the maximum traced steps while the
    # should-trace check uses "more than", an off-by-one inconsistency at the boundary.
    # Input: step number. Returns false if the step is explicitly scheduled (slow-path set) or
    # if auto tracing is on, the warm-up steps are over and the traced-step budget is not
    # exhausted; otherwise returns true. No side effects.
    if step in self._slow_path_steps:
      return False
    # When user doesn't set the tracing steps explicitly, auto decide it.
    if (self._auto_tracing and step > WARMUP_STEPS and
        self._traced_steps <= MAX_TRACED_STEPS):
      return False
    return True

  def _should_trace(self, step, graph, fetches):
    # [FUNCTION DOC]
    # HELPER: Decision logic used by the patched session run.
    # Inputs: step number, the session graph and the fetches of the run. Returns true if the step
    # should be traced. False when the traced-step budget is exceeded. True if the step is in the
    # trace set or a next-step trace was requested. In auto mode after warm-up it traces when the
    # combination of fetch names has not been seen before, or randomly (seeded generator) when
    # profiler coverage is below one half. Side effects: increments the traced-step counter,
    # records fetch names, writes debug output to standard error.
    """Whether should do tracing at current step."""
    if self._traced_steps > MAX_TRACED_STEPS:
      return False
    # Check user-set tracing steps.
    if step in self._trace_steps or self._trace_next_step:
      self._traced_steps += 1
      return True

    # If no user-set tracing steps set and passes warm up steps, auto trace.
    if self._auto_tracing and step > WARMUP_STEPS:
      # If the fetches have not been seen before, trace it.
      with graph.as_default():
        fetch_names = [f.name for f in
                       session._FetchMapper.for_fetch(fetches).unique_fetches()]  # pylint: disable=protected-access
      fetch_name = '-'.join(sorted(fetch_names))
      if self._debug:
        sys.stderr.write('debug: trace fetches: %s\n' % fetch_name)
      if fetch_name not in self._fetched:
        self._fetched.add(fetch_name)
        self._traced_steps += 1
        return True
      # If the trace coverage is low, does some random tracing.
      if (self.profiler._coverage < 0.5 and step < MAX_TRACED_STEPS and  # pylint: disable=protected-access
          self._rng.randint(0, 10) < 2):
        self._traced_steps += 1
        return True
    return False

  def _maybe_dump(self, step):
    # [FUNCTION DOC]
    # HELPER: Dump hook called after each slow-path run.
    # Input: step number. If the step is a dump step or a dump was requested, creates the profile
    # directory if needed and writes the profiler state to a file named with the step number in
    # it. Otherwise returns immediately. Returns nothing; writes to the file system.
    """Maybe dump the profile file."""
    if not (step in self._dump_steps or self._dump_next_step):
      return
    if self._debug:
      sys.stderr.write('debug: dumping file at step: %d\n' % step)
    gfile.MakeDirs(self._profiler_dir)

    filename = os.path.join(compat.as_bytes(self._profiler_dir),
                            compat.as_bytes('profile_%d' % step))
    self.profiler._write_profile(filename)  # pylint: disable=protected-access

  def _dump_file(self, pb, basename):
    # [FUNCTION DOC]
    # HELPER: Debug utility writing a raw proto (run metadata) as text.
    # Inputs: pb (the message object) and basename (file name inside the profile directory).
    # Creates the directory if missing and writes the string form of the message to the file.
    # Returns nothing; side effect is a file write.
    gfile.MakeDirs(self._profiler_dir)
    with gfile.Open(os.path.join(self._profiler_dir, basename), 'w') as f:
      f.write('%s' % pb)

  @contextlib.contextmanager
  def _new_step(self):
    # [FUNCTION DOC]
    # HELPER: Context manager bracketing one session run for the patched run function.
    # KNOWN ISSUE: There is no protection against exceptions: if the body raises, the code after
    # the yield (step increment, flag reset, lock release) does not run, so the lock stays held
    # and later runs always take the unprofiled path.
    # Tries to take the lock without blocking, yields the current step number and whether the lock
    # was acquired, then increments the step, clears the one-shot trace/dump flags and releases
    # the lock if it was taken.
    acquired = self._lock.acquire(False)  # pylint: disable=assignment-from-no-return
    yield (self._step, acquired)
    self._step += 1
    self._trace_next_step = False
    self._dump_next_step = False
    if acquired:
      self._lock.release()

  def _profile_candidates(self):
    # [FUNCTION DOC]
    # HELPER: Selects auto-profiling entries due at the current step.
    # Returns a list of the registered (command, options, steps) entries whose step list contains
    # the current step number. No side effects.
    to_profile = []
    for auto_prof in self._auto_profiles:
      _, _, prof_steps = auto_prof
      if self._step in prof_steps:
        to_profile.append(auto_prof)
    return to_profile

  def __enter__(self):
    # [FUNCTION DOC]
    # Enters the context. If enabled, saves the session class original run and constructor, and
    # raises an internal error if either is missing or if a previous context was not cleaned up
    # (the internal-name attributes already exist). Otherwise replaces run and constructor with
    # the profiling wrappers, stores the originals under internal names and attaches this context
    # to the session class. Returns this context (also when disabled, without patching).
    if self._enabled:
      self.old_run = getattr(session.BaseSession, 'run', None)
      self.old_init = getattr(session.BaseSession, '__init__', None)
      if not self.old_run:
        raise errors.InternalError(None, None, 'BaseSession misses run method.')
      elif not self.old_init:
        raise errors.InternalError(None, None,
                                   'BaseSession misses __init__ method.')
      elif getattr(session.BaseSession, '_profiler_run_internal', None):
        raise errors.InternalError(None, None,
                                   'Already in context or context not cleaned.')
      elif getattr(session.BaseSession, '_profiler_init_internal', None):
        raise errors.InternalError(None, None,
                                   'Already in context or context not cleaned.')
      else:
        setattr(session.BaseSession, 'run', _profiled_run)
        setattr(session.BaseSession, '__init__', _profiled_init)
        setattr(session.BaseSession, '_profiler_run_internal', self.old_run)
        setattr(session.BaseSession, '_profiler_init_internal', self.old_init)
        setattr(session.BaseSession, 'profile_context', self)
        return self
    else:
      return self

  def __exit__(self, exec_type, exec_value, exec_tb):
    # [FUNCTION DOC]
    # Leaves the context. Arguments are the usual exception type, value and traceback (ignored).
    # If enabled, deletes the native profiler and restores the original session run and
    # constructor, clears the internal attributes and the attached context. Does not suppress
    # exceptions. Returns nothing.
    if not self._enabled:
      return
    print_mdl.DeleteProfiler()
    setattr(session.BaseSession, 'run', self.old_run)
    setattr(session.BaseSession, '__init__', self.old_init)
    setattr(session.BaseSession, '_profiler_run_internal', None)
    setattr(session.BaseSession, '_profiler_init_internal', None)
    setattr(session.BaseSession, 'profile_context', None)
