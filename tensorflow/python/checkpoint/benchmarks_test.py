# [MODULE DOC]
# Benchmarks measuring checkpoint save and restore cost with many trackable objects that are
# created lazily after restore. Compares baseline (no restore), batch restore, restore-on-create
# and a raw low-level restore operation. When run as a script, enables eager execution and runs
# the benchmark main. No import-time side effects.
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
"""Benchmarks for checkpoint-related APIs."""

import os
import time
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.framework import ops
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.platform import test
from tensorflow.python.trackable import base
from tensorflow.python.training import py_checkpoint_reader


class _TrivialRestore(base.Trackable):
  # [CLASS DOC]
  # HELPER: minimal trackable used as benchmark payload. Saves a single scalar one and restores
  # by doing nothing.

  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # HELPER: returns a mapping from the standard variable-value key to a scalar tensor of one.
    return {base.VARIABLE_VALUE_KEY: array_ops.ones([])}

  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # HELPER: restore hook that ignores the restored tensors and returns a no-op operation.
    return control_flow_ops.no_op()


class _LazyTrivialObjects(module.Module):
  # [CLASS DOC]
  # HELPER: module used by the benchmarks. Holds five trackables created at construction and a
  # list of five more that are created lazily on the first call.

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor: creates the list of five existing trivial trackables and an empty list for
    # the lazy ones. No arguments.
    self.existing = [_TrivialRestore() for _ in range(5)]
    self.lazy = []

  def __call__(self):
    # [FUNCTION DOC]
    # On the first call only, fills the lazy list with five new trivial trackables; later calls
    # do nothing. Returns nothing.
    if not self.lazy:
      self.lazy.extend(_TrivialRestore() for _ in range(5))
    return


def _save_checkpoint():
  # [FUNCTION DOC]
  # HELPER: produces a checkpoint file for the benchmarks. Builds a checkpoint around a lazy
  # module, calls the module so the lazy trackables exist, writes it under the test temporary
  # directory with a fixed prefix, and returns the written path. Side effect: writes files.
  original_checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
  original_checkpoint.m()
  return original_checkpoint.write(os.path.join(test.get_temp_dir(), "ckpt"))


class SavingBenchmarks(test.Benchmark):
  # [CLASS DOC]
  # Benchmark suite for checkpoint restore strategies; each benchmark method times a scenario
  # through the shared run helper and reports mean wall time per iteration.

  def _run(self, func, num_iters, execution_mode=None):
    # [FUNCTION DOC]
    # HELPER: timing harness. Calls func once as warm-up, then num_iters times while timing, and
    # reports a benchmark with the iteration count, mean wall time in microseconds, and an extra
    # iterations-per-second figure. execution_mode is accepted but unused. Returns nothing.
    func()
    start = time.time()
    for _ in range(num_iters):
      func()
    end = time.time()
    mean_us = (end - start) * 1e6 / num_iters
    self.report_benchmark(
        iters=num_iters,
        wall_time=mean_us,
        extras={"examples_per_sec": num_iters / (end - start)})

  def benchmark_baseline_no_restore(self):
    # [FUNCTION DOC]
    # Baseline: times creating a checkpoint around a lazy module and calling the module, with
    # no restore, over three iterations.

    def _create_and_call():
      # [FUNCTION DOC]
      # HELPER: the timed body; builds the checkpoint and calls the module.
      checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
      checkpoint.m()

    self._run(_create_and_call, 3)

  def benchmark_batch_restore(self):
    # [FUNCTION DOC]
    # Saves a checkpoint, then times (three iterations) creating a checkpoint, calling the
    # module so all objects exist, and only then restoring, so values restore in one batch.
    checkpoint_path = _save_checkpoint()

    def _create_and_call():
      # [FUNCTION DOC]
      # HELPER: the timed body; create, call the module, then restore from the saved path.
      checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
      checkpoint.m()
      checkpoint.restore(checkpoint_path)

    self._run(_create_and_call, 3)

  def benchmark_restore_on_create(self):
    # [FUNCTION DOC]
    # Saves a checkpoint, then times (three iterations) restoring before the module is called,
    # so lazily created objects are restored at the moment they are created.
    checkpoint_path = _save_checkpoint()

    def _create_and_call():
      # [FUNCTION DOC]
      # HELPER: the timed body; create, restore from the saved path, then call the module.
      checkpoint = util.Checkpoint(m=_LazyTrivialObjects())
      checkpoint.restore(checkpoint_path)
      checkpoint.m()

    self._run(_create_and_call, 3)

  def benchmark_raw_restore(self):
    # [FUNCTION DOC]
    # Saves a checkpoint, reads its variable names and data types with the checkpoint reader,
    # then times (three iterations) calling the low-level restore operation directly with empty
    # slice specs, as a lower bound for restore cost.
    checkpoint_path = _save_checkpoint()
    all_names, all_dtypes = zip(*py_checkpoint_reader.NewCheckpointReader(
        checkpoint_path).get_variable_to_dtype_map().items())

    def _call_restore_v2():
      # [FUNCTION DOC]
      # HELPER: the timed body; invokes the raw restore operation for all checkpoint variables.
      gen_io_ops.restore_v2(checkpoint_path, all_names, [""] * len(all_names),
                            all_dtypes)

    self._run(_call_restore_v2, 3)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
