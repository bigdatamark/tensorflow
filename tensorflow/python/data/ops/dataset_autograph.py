# [MODULE DOC]
# Autograph support for datasets: defines overrides that let converted code use datasets in for
# loops and in the abs, len, enumerate, zip, map, filter, any and all builtins, plus a function
# that registers them with the autograph operator registries. Importing has no side effects; the
# registration happens only when the registration function is called. Helper module for dataset ops.
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
"""Autograph specific overrides for dataset_ops."""
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


def _general_purpose_scan(ds, init_state, body):
  # [FUNCTION DOC]
  # HELPER: scan variant used by autograph loops.
  # Takes a dataset, an initial state and a scan body function, and returns a scan dataset that
  # does not use the default (data preprocessing) device placement, so it is treated as general
  # computation. Imports the scan module lazily to avoid a circular dependency.
  """Variant of Dataset.scan with semantics of general-purpose computation."""
  # Datasets are typically intended for data preprocessing. However, in
  # autograph loops they usually appear as general-purpose computations (for
  # example, a custom training loop). These two use cases require significantly
  # different optimization policies, the most important of which is the device
  # placement. The flag override for use_default_device below instructs the
  # runtime to treat the computation as general-purpose, rather than data
  # preprocessing.

  # Loaded lazily due to a circular dependency (dataset_ops ->
  # scan_op -> dataset_ops).
  # pylint: disable=g-import-not-at-top,protected-access
  from tensorflow.python.data.ops import scan_op
  return scan_op._ScanDataset(ds, init_state, body, use_default_device=False)
  # pylint: enable=g-import-not-at-top,protected-access


def _tf_ag_dataset_for_stmt(
    ds, extra_test, body, get_state, set_state, symbol_names, opts
):
  # [FUNCTION DOC]
  # HELPER: autograph override of the for-loop statement when iterating over a dataset.
  # Inputs: ds (dataset), extra_test (optional callback giving an early-stop condition), body
  # (callback run per element), get_state and set_state (callbacks reading and writing the loop
  # variables), symbol_names (names of loop variables), opts (loop options).
  # Implements the loop as scan, then an optional take-while, then reduce over the dataset. If
  # there are no loop variables, uses a dummy state. Verifies loop variable initialization and
  # type consistency. Writes the final loop variables via set_state; returns nothing.
  """Overload of _dataset_for_stmt with early stopping. See for_stmt."""
  # Note: This is easier to follow with the insight that the computations in
  # a dataset pipeline are transposed (aka fused).
  # For example, given a pipeline input -> scan -> take_while -> reduce,
  # and a dataset with input [1, 2, 3], the computations occur in the following
  # order:
  #  reduce(take_while(scan(1)))
  #  reduce(take_while(scan(2)))
  #  reduce(take_while(scan(3)))

  init_vars = get_state()
  control_flow.verify_loop_init_vars(init_vars, symbol_names)

  # Workaround for Dataset.reduce not allowing empty state tensors - create
  # a dummy state variable that remains unused.
  # TODO(mdan): reduce should allow and match empty structures.
  if not init_vars:
    init_vars = (constant_op.constant(0),)
    symbol_names = ("<internal dummy>",)

    def dummy_set_state(unused_dummy):
      # [FUNCTION DOC]
      # HELPER: placeholder state setter used when the loop has no variables; does nothing.
      pass

    def dummy_get_state():
      # [FUNCTION DOC]
      # HELPER: placeholder state getter; returns a tuple holding one constant zero tensor.
      return (constant_op.constant(0),)

    get_state, set_state = dummy_get_state, dummy_set_state

  def scan_body(scan_state, scan_inputs):
    # [FUNCTION DOC]
    # HELPER: scan function of the loop translation.
    # Takes the current loop variables and the current dataset element. Restores the loop
    # variables, runs the loop body (only if the early-stop condition, when present, still
    # holds), and returns the new state plus a pair of the new loop variables and the condition.
    """Main body of the Dataset.scan."""
    loop_vars, iterate = scan_state, scan_inputs
    set_state(loop_vars)

    def main_path():
      # [FUNCTION DOC]
      # HELPER: runs the loop body on the current element, reads the updated loop variables,
      # verifies they are consistent with the initial ones (shapes not checked) and returns them.
      body(iterate)
      new_loop_vars = get_state()
      control_flow.verify_tf_loop_vars(
          init_vars,
          loop_vars,
          new_loop_vars,
          symbol_names,
          opts,
          check_shapes=False)
      return new_loop_vars

    if extra_test is not None:
      extra_cond = extra_test()
      new_loop_vars = cond.cond(extra_cond, main_path,
                                lambda: loop_vars)
    else:
      # TODO(mdan): the optimizer should be able to remove an invariant cond?
      extra_cond = (constant_op.constant(True),)  # dummy value, unused
      new_loop_vars = main_path()

    scan_outputs = new_loop_vars, extra_cond
    new_scan_state = new_loop_vars
    return new_scan_state, scan_outputs

  def take_while_predicate(unused_loop_vars, extra_cond):
    # [FUNCTION DOC]
    # HELPER: predicate for the take-while step; ignores the loop variables and returns the
    # early-stop condition.
    return extra_cond

  def reduce_body(unused_reduce_state, scan_outputs):
    # [FUNCTION DOC]
    # HELPER: reduce function of the loop translation; ignores the previous state and returns the
    # loop variables from the scan outputs as the new state.
    output_loop_vars, unused_extra_cond = scan_outputs
    new_reduce_state = output_loop_vars
    return new_reduce_state

  ds = _general_purpose_scan(ds, init_vars, scan_body)
  if extra_test is not None:
    ds = ds.apply(take_while_ops.take_while(take_while_predicate))
  final_loop_vars = ds.reduce(init_vars, reduce_body)
  set_state(final_loop_vars)


def _tf_ag_dataset_abs(ds):
  # [FUNCTION DOC]
  # HELPER: autograph override of the absolute-value builtin for datasets.
  # Returns a dataset mapping absolute value over each element (over every component when the
  # elements are nested), with automatically tuned parallelism.
  specs = nest.flatten(ds.element_spec)
  if len(specs) == 1:
    return ds.map(math_ops.abs, num_parallel_calls=dataset_ops.AUTOTUNE)
  return ds.map(
      lambda *e: nest.map_structure(math_ops.abs, e),
      num_parallel_calls=dataset_ops.AUTOTUNE)


def _tf_ag_dataset_len(s):
  # [FUNCTION DOC]
  # HELPER: autograph override of the length builtin for datasets.
  # Returns the dataset cardinality as a tensor. Adds a graph assertion that fails at run time
  # with a message if the cardinality is infinite or unknown.
  """Autograph override of the builtin len for dataset_ops.DataSetV2."""
  l = s.cardinality()
  msg = gen_string_ops.string_join([
      "len requires dataset with definitive cardinality, got ",
      gen_string_ops.as_string(l),
  ])
  # TODO(yongtang): UNKNOWN is treated as an error.
  # In case there are more UNKNOWN cases for dataset, we could
  # use dataset.reduce() to find out the length (in an expensive way).
  with ops.control_dependencies([
      control_flow_assert.Assert(
          math_ops.logical_and(
              math_ops.not_equal(l, dataset_ops.INFINITE),
              math_ops.not_equal(l, dataset_ops.UNKNOWN)), [msg])
  ]):
    l = array_ops.identity(l)

  return l


def _tf_ag_dataset_enumerate(ds, start=0):
  # [FUNCTION DOC]
  # HELPER: autograph override of the enumerate builtin; returns the dataset enumerated from
  # start (default 0).
  return ds.enumerate(start)


def _tf_ag_dataset_zip(*iterables, strict=False):
  # [FUNCTION DOC]
  # HELPER: autograph override of the zip builtin for datasets.
  # Takes datasets as positional arguments and a strict flag. Raises a value error if strict is
  # true (unsupported). Otherwise returns the zipped dataset of the arguments as a tuple.
  if strict:
    raise ValueError("strict zip not supported by Dataset")
  return dataset_ops.DatasetV2.zip(iterables)


def _tf_ag_dataset_map(fn, *iterables):
  # [FUNCTION DOC]
  # HELPER: autograph override of the map builtin for datasets.
  # Zips the given datasets and returns the dataset obtained by mapping fn over the zipped
  # elements.
  return dataset_ops.DatasetV2.zip(iterables).map(fn)


def _tf_ag_dataset_filter(fn, iterable):
  # [FUNCTION DOC]
  # HELPER: autograph override of the filter builtin; returns the dataset filtered by predicate fn.
  return iterable.filter(fn)


# any() operation is essentially a "if first True element exist".
# For that it could be translated to `filter(True)` to filter out
# only `True` element, and then `take(1)`. This works in tf.data
# as tf.data's filter+take is done in pipeline so it will stop
# as soon as `take(1)` returns.
def _tf_ag_dataset_any(iterable):
  # [FUNCTION DOC]
  # HELPER: autograph override of the any builtin for datasets.
  # Requires a dataset whose elements are single boolean scalars, else raises a value error.
  # Filters to true elements, takes at most one, and reduces to a boolean tensor that is true if
  # one exists (false otherwise). Stops early in the pipeline.
  # check and make sure iterable.element_spec only consists of one
  # element of tf.bool.
  specs = nest.flatten(iterable.element_spec)
  if len(specs) != 1 or specs[0].dtype != dtypes.bool:
    raise ValueError('in graph mode, the "any" builtin only supports datasets '
                     'that return bool scalars; got: {}'.format(
                         iterable.element_spec))
  ds = iterable.filter(lambda x: x)
  ds = ds.take(1)
  ds = ds.reduce(constant_op.constant(False, dtype=dtypes.bool), lambda _, y: y)
  return ds


# all() operation is similar to any() and could be translated
# to `filter(False)` then `take(1)`, and check if `False` exists.
def _tf_ag_dataset_all(iterable):
  # [FUNCTION DOC]
  # HELPER: autograph override of the all builtin for datasets.
  # Requires a dataset whose elements are single boolean scalars, else raises a value error.
  # Filters to false elements, takes at most one, and reduces to a boolean tensor that is false
  # if one exists (true otherwise).
  # check and make sure iterable.element_spec only consists of one
  # element of tf.bool.
  specs = nest.flatten(iterable.element_spec)
  if len(specs) != 1 or specs[0].dtype != dtypes.bool:
    raise ValueError('in graph mode, the "all" builtin only supports datasets '
                     'that return bool scalars; got: {}'.format(
                         iterable.element_spec))
  ds = iterable.filter(math_ops.logical_not)
  ds = ds.take(1)
  ds = ds.reduce(constant_op.constant(True, dtype=dtypes.bool), lambda _, y: y)
  return ds


def register_overrides():
  # [FUNCTION DOC]
  # Registers the dataset overrides (for loop, abs, len, enumerate, zip, map, filter, any, all)
  # in the autograph operator registries for the dataset type. Takes no arguments, returns
  # nothing; mutates the global autograph registries.
  """Registers the autograph specific overrides for dataset_ops."""
  control_flow.for_loop_registry.register(
      dataset_ops.DatasetV2, _tf_ag_dataset_for_stmt
  )
  py_builtins.abs_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_abs)
  py_builtins.len_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_len)
  py_builtins.enumerate_registry.register(
      dataset_ops.DatasetV2, _tf_ag_dataset_enumerate
  )
  py_builtins.zip_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_zip)
  py_builtins.map_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_map)
  py_builtins.filter_registry.register(
      dataset_ops.DatasetV2, _tf_ag_dataset_filter
  )
  py_builtins.any_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_any)
  py_builtins.all_registry.register(dataset_ops.DatasetV2, _tf_ag_dataset_all)
