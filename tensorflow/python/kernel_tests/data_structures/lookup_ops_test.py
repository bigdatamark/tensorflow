# [MODULE DOC]
# Unit tests for the lookup table ops in the TensorFlow Python lookup_ops module. The file covers
# static hash tables, static vocabulary tables, key/value tensor initializers, text-file
# initializers, dense and mutable hash tables, the id-table-with-hash-buckets wrapper, and the
# convenience factories that build index-to-string and string-to-index tables from files or tensors.
# Tests check lookups with default values, dense, sparse and ragged inputs, export/import, error
# conditions, saver/checkpoint/saved-model round trips, multi-session sharing, tracing inside
# functions, and (at the end) simple insert benchmarks for mutable and dense hash tables.
# Many test classes are parameterized by an "is anonymous" flag so each test runs once with named
# and once with anonymous (reference-counted) table resources; anonymous cases are skipped where
# graph-mode TF1 semantics cannot support them. Importing the module has no side effects beyond
# defining the classes and one skip-reason constant; running it as a script starts the test runner.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for lookup ops."""
import os
import tempfile
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.checkpoint import checkpoint as trackable
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import util as checkpoint_util
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat


class BaseLookupTableTest(test.TestCase):
  # [CLASS DOC]
  # HELPER: Shared base class for the parameterized lookup table test classes below.
  # Base test case that offers three small fixtures: choosing the right static hash table class, the
  # right static vocabulary table class, and initializing a table. The choice depends on whether the
  # TF2 behavior is enabled (the plain classes) or not (the V1 classes that need an explicit
  # initializer run). It holds no state of its own and defines no tests.

  def getHashTable(self):
    # [FUNCTION DOC]
    # HELPER: Fixture picker used by the static hash table tests.
    # Returns the static hash table class (not an instance) to construct: the TF2 static hash table
    # when TF2 behavior is enabled, otherwise the TF1-style static hash table. Takes no inputs and has
    # no side effects.
    if tf2.enabled():
      return lookup_ops.StaticHashTable
    else:
      return lookup_ops.StaticHashTableV1

  def getVocabularyTable(self):
    # [FUNCTION DOC]
    # HELPER: Fixture picker used by the static vocabulary table tests.
    # Returns the static vocabulary table class (not an instance): the TF2 version when TF2 behavior is
    # enabled, otherwise the TF1-style version. Takes no inputs and has no side effects.
    if tf2.enabled():
      return lookup_ops.StaticVocabularyTable
    else:
      return lookup_ops.StaticVocabularyTableV1

  def initialize_table(self, table):
    # [FUNCTION DOC]
    # HELPER: Initialization helper used by the static table tests.
    # Input: a table object. When TF2 behavior is not enabled, runs the table's initializer operation in
    # the current graph/session so the table becomes usable; when TF2 is enabled, tables initialize
    # themselves on construction so nothing is done. Returns nothing.
    if not tf2.enabled():
      self.evaluate(table.initializer)


SKIP_ANONYMOUS_IN_TF1_REASON = (
    "In v1 graph mode, each self.evaluate call will execute the handle "
    "creation op (e.g. AnonymousHashTable) which will create a new table "
    "resource unrelated to other self.evaluate calls, so we can't test "
    "anonymous resources with self.evaluate ."
)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class StaticHashTableTest(BaseLookupTableTest, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the static (immutable) hash table, run once with named and once with anonymous table
  # resources through the class-level parameterization. Covers basic lookup and export, high-rank,
  # sparse and ragged inputs, initialization from Python lists and numpy arrays, signature mismatch
  # errors, uninitialized and doubly-initialized tables in graph mode, use inside functions and
  # control flow, shape inference, and saved-model save/restore. Uses the base fixtures to pick the
  # table class and initialize tables.

  def testStaticHashTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: build a string-to-int64 static table from three keys with default value minus one.
    # Checks the anonymous flag is recorded on the table, the size is three, looking up an array with one
    # unknown key yields the mapped values and the default with static shape of three, and that export
    # returns all keys and values (compared without regard to order). Parameter is_anonymous selects the
    # table resource kind; the test is skipped for anonymous tables in TF1 graph mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertEqual(table._is_anonymous, is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

    exported_keys_tensor, exported_values_tensor = table.export()

    self.assertItemsEqual([b"brain", b"salad", b"surgery"],
                          self.evaluate(exported_keys_tensor))
    self.assertItemsEqual([0, 1, 2], self.evaluate(exported_values_tensor))

  def testStaticHashTableFindHighRank(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: look up a two-by-two matrix of string keys in a static table. Expects the result to keep
    # the input shape, with known keys mapped to their values and unknown keys mapped to the default
    # (minus one). Also confirms the table size is three. Skipped for anonymous tables in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"],
                                         ["tank", "tarkus"]])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testStaticHashTableInitWithPythonArrays(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: construct the table initializer from plain Python lists of keys and values (with an
    # explicit value dtype) instead of tensors. Checks the table size and that a lookup returns the
    # mapped values with the default for an unknown key. Skipped for anonymous tables in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = ["brain", "salad", "surgery"]
    values = [0, 1, 2]
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            keys, values, value_dtype=dtypes.int64),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testStaticHashTableInitWithNumPyArrays(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: construct the table initializer from numpy arrays of strings and int64 values. Checks the
    # table size of three and that lookup returns mapped values and the default for the unknown key.
    # Skipped for anonymous tables in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = np.array(["brain", "salad", "surgery"], dtype=np.str_)
    values = np.array([0, 1, 2], dtype=np.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testMultipleStaticHashTables(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: create three independent static tables from the same keys and values, initialize each,
    # and verify each reports size three and gives identical lookup results (values for known keys,
    # default for unknown). Guards against tables interfering with each other. Skipped for anonymous
    # tables in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)

    table1 = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    table2 = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    table3 = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table1)
    self.initialize_table(table2)
    self.initialize_table(table3)
    self.assertAllEqual(3, self.evaluate(table1.size()))
    self.assertAllEqual(3, self.evaluate(table2.size()))
    self.assertAllEqual(3, self.evaluate(table3.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output1 = table1.lookup(input_string)
    output2 = table2.lookup(input_string)
    output3 = table3.lookup(input_string)

    out1, out2, out3 = self.evaluate([output1, output2, output3])
    self.assertAllEqual([0, 1, -1], out1)
    self.assertAllEqual([0, 1, -1], out2)
    self.assertAllEqual([0, 1, -1], out3)

  def testStaticHashTableWithTensorDefault(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: the default value is given as a scalar int64 tensor instead of a Python number. Checks
    # lookup still returns the mapped values with the tensor default for the missing key. Skipped for
    # anonymous tables in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testStaticHashTableGetItem(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: lookup through the indexing syntax of the table rather than the lookup method. Verifies
    # the result equals mapped values with the default for the missing key. Skipped for anonymous tables
    # in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table[input_string]

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testStaticHashTableWithSparseTensorInput(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: look up a sparse tensor of string keys. Checks that the output is a sparse tensor whose
    # indices and dense shape are unchanged and whose values are the mapped ids (default for the unknown
    # key). Skipped for anonymous tables in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_indices = [[0, 0], [0, 1], [1, 0]]
    sp_shape = [2, 2]
    input_tensor = sparse_tensor.SparseTensor(
        constant_op.constant(sp_indices, dtypes.int64),
        constant_op.constant(["brain", "salad", "tank"]),
        constant_op.constant(sp_shape, dtypes.int64))
    output = table.lookup(input_tensor)

    out_indices, out_values, out_shape = self.evaluate(output)

    self.assertAllEqual([0, 1, -1], out_values)
    self.assertAllEqual(sp_indices, out_indices)
    self.assertAllEqual(sp_shape, out_shape)

  def testStaticHashTableWithRaggedTensorInput(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: look up a ragged tensor of string keys built from row splits. Checks the output keeps the
    # same row splits and that the flat values are mapped ids with the default for the unknown key.
    # Skipped for anonymous tables in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    row_splits = [0, 2, 3]
    input_tensor = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant(["brain", "salad", "tank"]),
        constant_op.constant(row_splits, dtypes.int64))
    output = table.lookup(input_tensor)

    out = self.evaluate(output)

    self.assertAllEqual([0, 1, -1], out.values)
    self.assertAllEqual(row_splits, out.row_splits)

  def testSignatureMismatch(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: type mismatches. A variable of string type (a reference-like input) is accepted for
    # lookup and returns the right value. Looking up int64 keys in a string-keyed table must raise a
    # type error, and creating the table with a string default value for int64 values must also raise a
    # type error. Skipped for anonymous tables in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    # Ref types do not produce a lookup signature mismatch.
    input_string_ref = variables.Variable("brain")
    self.evaluate(input_string_ref.initializer)
    self.assertEqual(0, self.evaluate(table.lookup(input_string_ref)))

    input_string = constant_op.constant([1, 2, 3], dtypes.int64)
    with self.assertRaises(TypeError):
      table.lookup(input_string)

    with self.assertRaises(TypeError):
      self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          "UNK",
          experimental_is_anonymous=is_anonymous)

  def testDTypes(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: building a table whose initializer is given a list as key dtype (instead of a single data
    # type) must raise a type error. Only the construction error is asserted; the anonymous flag is
    # passed through to the table.
    default_val = -1
    with self.assertRaises(TypeError):
      self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(["a"], [1], [dtypes.string],
                                               dtypes.int64),
          default_val,
          experimental_is_anonymous=is_anonymous)

  @test_util.run_v1_only("(Cached) Sessions not available in TF2.0")
  def testNotInitialized(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only, graph mode with a cached session): a table that was never initialized is
    # looked up. Evaluating the lookup must raise an op error saying the table is not initialized.
    with self.cached_session():
      default_val = -1
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(["a"], [1],
                                               value_dtype=dtypes.int64),
          default_val,
          experimental_is_anonymous=is_anonymous)

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      with self.assertRaisesOpError("Table not initialized"):
        self.evaluate(output)

  @test_util.run_v1_only("(Cached) Sessions not available in TF2.0")
  def testInitializeTwice(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only): initializing the same table twice must not raise an error. Nothing is
    # asserted besides the absence of an exception.
    with self.cached_session():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery"])
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          default_val,
          experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)
      # Make sure that initializing twice doesn't throw any errors.
      self.initialize_table(table)

  def testInitializationWithInvalidDimensions(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: the keys tensor has three entries but the values tensor has five. Constructing the table
    # must raise a value error in graph mode, or an invalid-argument error when executing eagerly.
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2, 3, 4], dtypes.int64)

    raised_error = ValueError
    if context.executing_eagerly():
      raised_error = errors_impl.InvalidArgumentError
    with self.assertRaises(raised_error):
      self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          default_val,
          experimental_is_anonymous=is_anonymous)

  @test_util.run_v1_only("Sessions not available in TF2.0")
  def testMultipleSessions(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only): start a local gRPC server and open two sessions to it. A named table is
    # initialized in the first session and its size is three; the initializer is then run in the second
    # session, which must not fail with an "already initialized" error, and the size is again three.
    # Skipped for anonymous tables. Side effects: starts an in-process server and two sessions.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    # Start a server
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target)
    session2 = session.Session(server.target)

    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        name="t1",
        experimental_is_anonymous=is_anonymous)

    # Init the table in the first session.
    with session1:
      self.initialize_table(table)
      self.assertAllEqual(3, self.evaluate(table.size()))

    # Init the table in the second session and verify that we do not get a
    # "Table already initialized" error.
    with session2:
      self.evaluate(table.initializer)
      self.assertAllEqual(3, self.evaluate(table.size()))

  @test_util.run_v2_only
  def testImportedHashTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF2 only): build a static table inside a separate graph, export its meta graph, import it
    # in a wrapped function, prune out the table initialization op and run it, then check that the
    # wrapped lookup returns the stored value. Verifies imported graphs can re-initialize a table.
    g = ops.Graph()
    with g.as_default():
      t = lookup_ops.StaticHashTable(
          lookup_ops.KeyValueTensorInitializer(["a"], [1]),
          2)
      init_op = t._init_op
      op = t.lookup(ops.convert_to_tensor(["a"]))
      meta_graph = saver.export_meta_graph()

    def f():
      # [FUNCTION DOC]
      # HELPER: Inner function used by the imported-table test to wrap the imported graph.
      # Imports the previously exported meta graph into the default graph of the wrapped function and
      # returns the tensor of the lookup result looked up by its operation name. Takes no inputs.
      saver.import_meta_graph(meta_graph)
      return ops.get_default_graph().get_tensor_by_name(op.name)

    wrapped = wrap_function.wrap_function(f, [])
    pruned_init_fn = wrapped.prune(
        (), [wrapped.graph.get_operation_by_name(init_op.name)])
    self.evaluate(pruned_init_fn())
    self.assertAllEqual([1], wrapped())

  def testStaticHashTableInt32String(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: int32 keys mapped to string values with the string default "n/a". Verifies lookup of two
    # known keys and one unknown key returns the strings or the default. Skipped for anonymous tables in
    # TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = "n/a"
    keys = constant_op.constant([0, 1, 2], dtypes.int32)
    values = constant_op.constant(["brain", "salad", "surgery"])
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    input_tensor = constant_op.constant([0, 1, -1])
    output = table.lookup(input_tensor)

    result = self.evaluate(output)
    self.assertAllEqual([b"brain", b"salad", b"n/a"], result)

  def testTableUseInFunction(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (eager only): a table created outside a traced function is used inside a traced function
    # that looks up a batch of keys. Calls the function with two different inputs and checks the
    # results, including default for unknown keys. Skipped when not executing eagerly.
    if not context.executing_eagerly():
      self.skipTest("Only Eager mode test.")
    keys = constant_op.constant([0, 1, 2], dtypes.int32)
    values = constant_op.constant(["brain", "salad", "surgery"])
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        "n/a",
        experimental_is_anonymous=is_anonymous)

    @def_function.function
    def lookup_table_func(k):
      # [FUNCTION DOC]
      # HELPER: Traced function used by the use-in-function test.
      # Input: a tensor of integer keys. Returns the lookup result from the outer table. No side effects.
      return table.lookup(k)

    result = lookup_table_func(constant_op.constant([0, 1, -1]))
    self.assertAllEqual([b"brain", b"salad", b"n/a"], result)
    result = lookup_table_func(constant_op.constant([2, -1, 1]))
    self.assertAllEqual([b"surgery", b"n/a", b"salad"], result)

  def testTableCreatedInFunction(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (eager only): a traced function creates the table itself and returns a lookup result.
    # Calls the function twice with different keys and checks the mapped strings and default.
    # Skipped when not executing eagerly.
    if not context.executing_eagerly():
      self.skipTest("Only Eager mode test.")
    keys = constant_op.constant([0, 1, 2], dtypes.int32)
    values = constant_op.constant(["brain", "salad", "surgery"])

    @def_function.function
    def lookup_table_func(k):
      # [FUNCTION DOC]
      # HELPER: Traced function used by the created-in-function test.
      # Input: a tensor of integer keys. Builds a fresh static table inside the function from the outer
      # keys and values, then returns the lookup result for the input keys.
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          "n/a",
          experimental_is_anonymous=is_anonymous)
      return table.lookup(k)

    result = lookup_table_func(constant_op.constant([0, 1, -1]))
    self.assertAllEqual([b"brain", b"salad", b"n/a"], result)
    result = lookup_table_func(constant_op.constant([2, -1, 1]))
    self.assertAllEqual([b"surgery", b"n/a", b"salad"], result)

  def testTwoTablesInControlFlow(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: two different map functions each create their own static table and look up elements,
    # applied with a map-over-elements op. After initializing all tables, both results must equal the
    # expected values (ten, default, five). Verifies tables created inside control flow work.
    # Skipped for anonymous tables in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([1, 2, 3], dtypes.int32)
    values = constant_op.constant([5, 10, 15], dtypes.int32)

    def table_func1(x):
      # [FUNCTION DOC]
      # HELPER: First map function used by the control-flow test.
      # Input: one scalar key. Creates a static table from the outer keys and values and returns the
      # lookup result for that key.
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          -1,
          experimental_is_anonymous=is_anonymous)
      return table.lookup(x)

    elems = np.array([2, 4, 1], dtype=np.int32)
    result1 = map_fn.map_fn(table_func1, elems, dtype=dtypes.int32)

    def table_func2(x):
      # [FUNCTION DOC]
      # HELPER: Second map function used by the control-flow test, identical in behavior to the first.
      # Input: one scalar key. Creates a static table from the outer keys and values and returns the
      # lookup result for that key.
      table = self.getHashTable()(
          lookup_ops.KeyValueTensorInitializer(keys, values),
          -1,
          experimental_is_anonymous=is_anonymous)
      return table.lookup(x)

    elems = np.array([2, 4, 1], dtype=np.int32)
    result2 = map_fn.map_fn(table_func2, elems, dtype=dtypes.int32)

    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([10, -1, 5], self.evaluate(result1))
    self.assertAllEqual([10, -1, 5], self.evaluate(result2))

  @test_util.enable_control_flow_v2
  def testLookupTableInWhileV2(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (control flow v2): the table lookup is used inside a map over elements within a traced
    # function that also takes a trainable variable. Computes the gradient of the loss with respect to
    # the variable and asserts it is nothing (no gradient flows through the table). Anonymous flag is
    # passed to the table.
    lookup = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            constant_op.constant([2, 5], dtype=dtypes.int64),
            constant_op.constant([-10.0, 1], dtype=dtypes.float32)),
        -1,
        experimental_is_anonymous=is_anonymous)

    beta = variables.Variable(1.0, trainable=True)

    @def_function.function
    def get_loss(unused_beta):
      # [FUNCTION DOC]
      # HELPER: Traced loss function for the while-loop test.
      # Input: an unused value standing in for the variable. Returns the map-over-elements result of
      # looking up two int64 keys, producing float values.
      return map_fn.map_fn(
          lookup.lookup,
          constant_op.constant([2, 3], dtype=dtypes.int64),
          dtype=dtypes.float32)

    with backprop.GradientTape() as tape:
      loss = get_loss(beta)

    self.assertIsNone(tape.gradient(loss, beta))

  @test_util.enable_control_flow_v2
  def testLookupTableInCondV2(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (control flow v2): the lookup is used in a conditional inside a traced function whose result
    # is scaled by a variable. After initializing variables and tables, the gradient with respect to the
    # variable must equal minus ten (the looked-up value). Skipped for anonymous tables in TF1 mode.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    lookup = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            constant_op.constant([2, 5], dtype=dtypes.int64),
            constant_op.constant([-10.0, 1], dtype=dtypes.float32)),
        -1,
        experimental_is_anonymous=is_anonymous)

    beta = variables.Variable(1.0, trainable=True)

    @def_function.function
    def get_loss(beta):
      # [FUNCTION DOC]
      # HELPER: Traced loss function for the conditional test.
      # Input: the variable (a scalar float). Defines the two branches, runs a conditional on a constant
      # true value, and returns the variable times the branch result.

      def true_fn():
        # [FUNCTION DOC]
        # HELPER: True branch of the conditional. Takes no inputs; returns the table lookup of key two.
        return lookup.lookup(constant_op.constant(2, dtype=dtypes.int64))

      def false_fn():
        # [FUNCTION DOC]
        # HELPER: False branch of the conditional. Takes no inputs; returns a constant float zero.
        return constant_op.constant(0, dtype=dtypes.float32)

      return beta * cond.cond(
          constant_op.constant(True), true_fn=true_fn, false_fn=false_fn)

    with backprop.GradientTape() as tape:
      loss = get_loss(beta)
    grad = tape.gradient(loss, beta)
    self.evaluate(variables.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual(grad, -10.)

  def testImportShapeInference(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: calling the raw table-import op with a scalar values argument inside a function
    # compiled with just-in-time compilation. Expects a value error complaining that the shape must be
    # at least rank one but is rank zero. Only the error message is asserted.
    v = variables.Variable(1)

    @def_function.function(jit_compile=True)
    def foo():
      # [FUNCTION DOC]
      # HELPER: Compiled function for the import shape test. Takes no inputs and calls the raw import op on
      # a variable handle with two float keys and a scalar value, which triggers the shape error.
      return gen_lookup_ops.lookup_table_import_v2(
          table_handle=v.handle, keys=[1.1, 2.2], values=1
      )

    with self.assertRaisesRegex(
        ValueError, r"Shape must be at least rank 1 but is rank 0"
    ):
      foo()

  def testExportShapeInference(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: shape inference for table export. Compares the shapes of the exported keys and values
    # obtained directly with the shapes inferred while tracing a function that exports the table. Both
    # must have two entries and each inferred shape must be compatible with the actual one.
    table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(
            constant_op.constant([2, 5], dtype=dtypes.int64),
            constant_op.constant([-10.0, 1], dtype=dtypes.float32)),
        -1,
        experimental_is_anonymous=is_anonymous)
    actual_shapes = [t.shape for t in table.export()]
    inferred_shapes = []

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: Traced function for the export shape test. Takes no inputs; exports the outer table and
      # appends the shape of each exported tensor to a list in the enclosing test (side effect). Returns
      # nothing.
      for t in table.export():
        inferred_shapes.append(t.shape)

    f()
    self.assertLen(actual_shapes, 2)
    self.assertLen(inferred_shapes, 2)
    self.assertTrue(inferred_shapes[0].is_compatible_with(actual_shapes[0]))
    self.assertTrue(inferred_shapes[1].is_compatible_with(actual_shapes[1]))

  @test_util.run_v2_only
  def testSavedModelSaveRestore(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF2 only): save and load a saved model that contains a table. Builds a trackable root with
    # a static table and three traced functions (lookup with a scalar key signature, size, and a check of
    # whether the table handle is reference counted), saves it to a temp directory, deletes the root,
    # loads it back and checks size three, lookup of a known key giving one, and unknown key giving the
    # default, and that the reference-counting flag equals the anonymous flag. Side effect: writes
    # files under the test temp directory.
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    root = autotrackable.AutoTrackable()

    default_value = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    root.table = self.getHashTable()(
        lookup_ops.KeyValueTensorInitializer(keys, values),
        default_value,
        experimental_is_anonymous=is_anonymous)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.int64)])
    def lookup(key):
      # [FUNCTION DOC]
      # HELPER: Traced function with a scalar int64 input signature that returns the lookup result of the
      # root's table for the given key. Used by the saved-model test.
      return root.table.lookup(key)

    @def_function.function(input_signature=[])
    def size():
      # [FUNCTION DOC]
      # HELPER: Traced function with an empty signature that returns the size of the root's table. Used by
      # the saved-model test.
      return root.table.size()

    @def_function.function(input_signature=[])
    def is_ref_counting():
      # [FUNCTION DOC]
      # HELPER: Traced function with an empty signature that returns whether the root table's resource
      # handle is reference counted (true for anonymous tables). Used by the saved-model test.
      return test_ops.is_resource_handle_ref_counting(
          root.table.resource_handle)

    root.lookup = lookup
    root.size = size
    root.is_ref_counting = is_ref_counting

    self.assertEqual(root.table.size(), 3)
    self.assertEqual(root.lookup(12), 1)
    self.assertEqual(root.lookup(10), -1)
    self.assertLen(root.table.export()[0], 3)
    self.assertEqual(root.is_ref_counting(), is_anonymous)

    saved_model_save.save(root, save_path)

    del root
    loaded = saved_model_load.load(save_path)
    self.assertEqual(loaded.size(), 3)
    self.assertEqual(loaded.lookup(12), 1)
    self.assertEqual(loaded.lookup(10), -1)
    self.assertEqual(loaded.is_ref_counting(), is_anonymous)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class KeyValueTensorInitializerTest(BaseLookupTableTest):
  # [CLASS DOC]
  # Tests for the key/value tensor initializer, parameterized by the anonymous flag. Checks that tables
  # can be built from string and int64 keys, that multiple tables get unique names in graph mode,
  # and that an int32 key type that does not match the data fails. Uses the base fixtures.

  def test_string(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: initializer with string keys and int64 values, given as Python tuples. Builds a table
    # with default minus one and initializes it. Only asserts that this succeeds without error.
    init = lookup_ops.KeyValueTensorInitializer(
        ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
    table = self.getHashTable()(
        init, default_value=-1, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

  def test_multiple_tables(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: two tables created inside one name scope. In graph mode, checks the first table is named
    # hash table and its resource op is in the scope, and the second gets a numeric suffix of one (both
    # in the table name and the resource op name). Nothing is asserted when running eagerly.
    with ops.name_scope("table_scope"):
      init1 = lookup_ops.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
      table1 = self.getHashTable()(
          init1, default_value=-1, experimental_is_anonymous=is_anonymous)
      if not context.executing_eagerly():
        self.assertEqual("hash_table", table1.name)
        self.assertEqual("table_scope/hash_table",
                         table1.resource_handle.op.name)
      init2 = lookup_ops.KeyValueTensorInitializer(
          ("brain", "salad", "surgery"), (0, 1, 2), dtypes.string, dtypes.int64)
      table2 = self.getHashTable()(
          init2, default_value=-1, experimental_is_anonymous=is_anonymous)
      if not context.executing_eagerly():
        self.assertEqual("hash_table_1", table2.name)
        self.assertEqual("table_scope/hash_table_1",
                         table2.resource_handle.op.name)

  def test_int64(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: initializer with int64 keys and int64 values given as tuples. Builds the table and
    # initializes it; only asserts that it succeeds.
    init = lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                                dtypes.int64, dtypes.int64)
    table = self.getHashTable()(
        init, default_value=-1, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

  def test_int32(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: initializer declared with int32 key dtype while the table is built with the default
    # int64-capable initializer data. Expects an op error when the table is created and initialized.
    init = lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                                dtypes.int32, dtypes.int64)
    with self.assertRaises(errors_impl.OpError):
      table = self.getHashTable()(
          init, default_value=-1, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class InitializeTableFromFileOpTest(BaseLookupTableTest):
  # [CLASS DOC]
  # Tests for initializing hash tables from text files with the text file initializer, parameterized by
  # the anonymous flag. Covers whole-line keys with line-number values and the reverse, multi-column
  # files, invalid column or data type choices, vocab size limits, feeding the file name through a
  # placeholder in TF1, invalid filenames, and the string and id table initializer helpers.
  # Provides a helper to write small vocabulary files into the test temp directory.

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    # [FUNCTION DOC]
    # HELPER: Creates a vocabulary file for the file-based table tests.
    # Inputs: basename (file name inside the test temp directory) and values (sequence of lines,
    # defaults to brain, salad, surgery). Writes one value per line, ending with a newline, and returns
    # the full path of the file. Side effect: creates or overwrites that file.
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def testInitializeStringTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: string keys from whole lines and int64 values from line numbers. Checks the initializer's
    # shared name contains the file name and the column indexes, initializes the table, and verifies
    # lookup returns zero, one, and the default for an unknown key. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = self._createVocabFile("one_column_1.txt")
    default_value = -1
    init = lookup_ops.TextFileInitializer(
        vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
        dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
    self.assertIn("one_column_1.txt_-2_-1", init._shared_name)
    table = self.getHashTable()(
        init, default_value, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    output = table.lookup(constant_op.constant(["brain", "salad", "tank"]))

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testInitializeInt64Table(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: int64 keys read from a file of numbers, values are line numbers. Checks the shared name,
    # then lookup of known and unknown numbers returns line numbers and the default. Runs inside a
    # cached session. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = self._createVocabFile(
        "one_column_int64.txt", values=("42", "1", "-1000"))

    with self.cached_session():
      default_value = -1
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.int64, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertIn("one_column_int64.txt_-2_-1", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)

      output = table.lookup(
          constant_op.constant((42, 1, 11), dtype=dtypes.int64))

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testInitializeIndexTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: keys are line numbers and values are whole-line strings, default "UNK". Checks the shared
    # name encodes the swapped column indexes and that lookup of ids zero to three returns the three
    # words and the default. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = self._createVocabFile("one_column_2.txt")

    with self.cached_session():
      default_value = "UNK"
      key_index = lookup_ops.TextFileIndex.LINE_NUMBER
      value_index = lookup_ops.TextFileIndex.WHOLE_LINE
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.int64, key_index, dtypes.string, value_index)
      self.assertIn("one_column_2.txt_-1_-2", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)

      input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      output = table.lookup(input_values)

      result = self.evaluate(output)
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"], result)

  def testMultiColumn(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: a three-column tab-separated file where column one holds string keys and column two int64
    # values. Checks the shared name and that lookup returns the numeric third-column values.
    # Side effect: writes a file in the temp directory. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.cached_session():
      default_value = -1
      key_index = 1
      value_index = 2

      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
      self.assertIn("three_columns.txt_1_2", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)

      input_string = constant_op.constant(["brain", "salad", "surgery"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([1, 5, 6], result)

  def testInvalidDataTypeInMultiColumn(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: same three-column file, but the key column is the numeric one declared as string and
    # the value column is the text one declared as int64. Table creation and initialization must raise an
    # op error saying the data is not valid. Side effect: writes a file in the temp directory.
    vocabulary_file = os.path.join(self.get_temp_dir(), "three_columns.txt")
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(["0\tbrain\t1", "1\tsalad\t5", "2\tsurgery\t6"]) + "\n")

    with self.cached_session():
      default_value = -1
      key_index = 2
      value_index = 1
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
      self.assertIn("three_columns.txt_2_1", init._shared_name)
      with self.assertRaisesOpError("is not a valid"):
        table = self.getHashTable()(
            init, default_value, experimental_is_anonymous=is_anonymous)
        self.initialize_table(table)

  def testInvalidDataType(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: whole-line string keys with line-number values are declared with swapped data types
    # (int64 keys and string values). Constructing the initializer or table must raise a value error.
    vocabulary_file = self._createVocabFile("one_column_3.txt")

    with self.cached_session():
      default_value = "UNK"
      key_index = lookup_ops.TextFileIndex.WHOLE_LINE
      value_index = lookup_ops.TextFileIndex.LINE_NUMBER

      with self.assertRaises(ValueError):
        init = lookup_ops.TextFileInitializer(vocabulary_file, dtypes.int64,
                                              key_index, dtypes.string,
                                              value_index)
        self.assertIn("one_column_3.txt_-2_-1", init._shared_name)
        self.getHashTable()(
            init, default_value, experimental_is_anonymous=is_anonymous)

  def testInvalidIndex(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: the key column index selects the second column of a one-column file. Initializing the
    # table must raise an op error about an invalid number of columns.
    vocabulary_file = self._createVocabFile("one_column_4.txt")
    with self.cached_session():
      default_value = -1
      key_index = 1  # second column of the line
      value_index = lookup_ops.TextFileIndex.LINE_NUMBER
      init = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, key_index, dtypes.int64, value_index)
      self.assertIn("one_column_4.txt_1_-1", init._shared_name)

      with self.assertRaisesOpError("Invalid number of columns"):
        table = self.getHashTable()(
            init, default_value, experimental_is_anonymous=is_anonymous)
        self.initialize_table(table)

  def testInitializeSameTableWithMultipleNodes(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: three tables from three separate but identical initializers over the same file. Checks the
    # shared names, initializes all tables at once with the global tables initializer, and verifies each
    # lookup returns zero, one and default. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = self._createVocabFile("one_column_5.txt")

    with self.cached_session():
      default_value = -1
      init1 = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertIn("one_column_5.txt_-2_-1", init1._shared_name)
      table1 = self.getHashTable()(
          init1, default_value, experimental_is_anonymous=is_anonymous)
      init2 = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertIn("one_column_5.txt_-2_-1", init2._shared_name)
      table2 = self.getHashTable()(
          init2, default_value, experimental_is_anonymous=is_anonymous)
      init3 = lookup_ops.TextFileInitializer(
          vocabulary_file, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertIn("one_column_5.txt_-2_-1", init3._shared_name)
      table3 = self.getHashTable()(
          init3, default_value, experimental_is_anonymous=is_anonymous)

      self.evaluate(lookup_ops.tables_initializer())

      input_string = constant_op.constant(["brain", "salad", "tank"])

      output1 = table1.lookup(input_string)
      output2 = table2.lookup(input_string)
      output3 = table3.lookup(input_string)

      out1, out2, out3 = self.evaluate([output1, output2, output3])
      self.assertAllEqual([0, 1, -1], out1)
      self.assertAllEqual([0, 1, -1], out2)
      self.assertAllEqual([0, 1, -1], out3)

  def testInitializeTableWithNoFilename(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: an initializer is given an empty string as file name. Constructing the table must raise
    # a value error.
    with self.cached_session():
      default_value = -1
      with self.assertRaises(ValueError):
        self.getHashTable()(
            lookup_ops.TextFileInitializer(
                "", dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
            default_value,
            experimental_is_anonymous=is_anonymous)

  def testInitializeWithVocabSize(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: effect of the vocab size argument. With size three on a three-line file the table size is
    # three. With size five on a three-line file initialization must fail with an op error about an
    # invalid vocab size. With size one only one record is read and the size is one. Also checks the
    # shared name includes the size. Side effect: writes several vocabulary files. Skipped for anonymous
    # tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      vocabulary_file1 = self._createVocabFile("one_column6.txt")
      init1 = lookup_ops.TextFileInitializer(
          vocabulary_file1,
          dtypes.string,
          lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64,
          lookup_ops.TextFileIndex.LINE_NUMBER,
          vocab_size=vocab_size)
      self.assertIn("one_column6.txt_3_-2_-1", init1._shared_name)
      table1 = self.getHashTable()(
          init1, default_value, experimental_is_anonymous=is_anonymous)

      # Initialize from file.
      self.initialize_table(table1)
      self.assertEqual(vocab_size, self.evaluate(table1.size()))

      vocabulary_file2 = self._createVocabFile("one_column7.txt")
      vocab_size = 5
      init2 = lookup_ops.TextFileInitializer(
          vocabulary_file2,
          dtypes.string,
          lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64,
          lookup_ops.TextFileIndex.LINE_NUMBER,
          vocab_size=vocab_size)
      self.assertIn("one_column7.txt_5_-2_-1", init2._shared_name)
      with self.assertRaisesOpError("Invalid vocab_size"):
        table2 = self.getHashTable()(
            init2, default_value, experimental_is_anonymous=is_anonymous)
        self.initialize_table(table2)

      vocab_size = 1
      vocabulary_file3 = self._createVocabFile("one_column3.txt")
      init3 = lookup_ops.TextFileInitializer(
          vocabulary_file3,
          dtypes.string,
          lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64,
          lookup_ops.TextFileIndex.LINE_NUMBER,
          vocab_size=vocab_size)
      self.assertIn("one_column3.txt_1_-2_-1", init3._shared_name)
      table3 = self.getHashTable()(
          init3, default_value, experimental_is_anonymous=is_anonymous)

      # Smaller vocab size reads only vocab_size records.
      self.initialize_table(table3)
      self.assertEqual(vocab_size, self.evaluate(table3.size()))

  @test_util.run_v1_only("placeholder usage")
  def testFeedVocabularyName(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only, uses placeholders): the initializer is created with a nonexistent file name,
    # so running the initializer fails with an op error mentioning that file. The file path collection is
    # then fed with the real vocabulary file when running the initializer, after which lookup returns the
    # expected ids and default. Skipped for anonymous tables in TF1. Side effect: writes a file.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocabulary_file = self._createVocabFile("feed_vocabulary.txt")

    with self.cached_session():
      default_value = -1
      init = lookup_ops.TextFileInitializer(
          "old_file.txt", dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
          dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER)
      self.assertIn("old_file.txt_-2_-1", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)

      # Initialize with non existing file (old_file.txt) should fail.
      # TODO(yleon): Update message, which might change per FileSystem.
      with self.assertRaisesOpError("old_file.txt"):
        self.evaluate(table.initializer)

      # Initialize the model feeding the vocabulary file.
      filenames = ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)
      table.initializer.run(feed_dict={filenames[0]: vocabulary_file})

      input_string = constant_op.constant(["brain", "salad", "tank"])
      output = table.lookup(input_string)

      result = self.evaluate(output)
      self.assertAllEqual([0, 1, -1], result)

  def testInvalidFilenames(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: invalid file name arguments. A non-string tensor as file name must raise a value or type
    # error. A tensor containing two file names must raise a value or type error in graph mode and an
    # invalid-argument error when eager.
    vocabulary_file = self._createVocabFile("filename_shape.txt")

    with self.cached_session():
      default_value = -1

      # Invalid data type
      other_type = constant_op.constant(1)
      with self.assertRaises(Exception) as cm:
        self.getHashTable()(
            lookup_ops.TextFileInitializer(
                other_type, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
            default_value,
            experimental_is_anonymous=is_anonymous)
      self.assertIsInstance(cm.exception, (ValueError, TypeError))

      # Non-scalar filename
      filenames = constant_op.constant([vocabulary_file, vocabulary_file])
      if not context.executing_eagerly():
        with self.assertRaises(Exception) as cm:
          self.getHashTable()(
              lookup_ops.TextFileInitializer(
                  filenames, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                  dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
              default_value,
              experimental_is_anonymous=is_anonymous)
        self.assertIsInstance(cm.exception, (ValueError, TypeError))
      else:
        with self.assertRaises(errors_impl.InvalidArgumentError):
          self.getHashTable()(
              lookup_ops.TextFileInitializer(
                  filenames, dtypes.string, lookup_ops.TextFileIndex.WHOLE_LINE,
                  dtypes.int64, lookup_ops.TextFileIndex.LINE_NUMBER),
              default_value,
              experimental_is_anonymous=is_anonymous)

  def testIdToStringTable(self, is_anonymous):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shared-name check uses a true/false assertion on a non-empty string, which is
    # always true, so the shared name is never actually compared with anything.
    # Scenario: the string table initializer reads three keys from a file into an index-to-string table
    # with default "UNK". Initializes the table, looks up ids zero to three and expects the three words
    # and "UNK", and checks the table size equals the vocab size. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    with self.cached_session():
      default_value = "UNK"
      vocab_size = 3
      init = lookup_ops.TextFileStringTableInitializer(
          vocab_file, vocab_size=vocab_size)
      self.assertTrue("feat_to_id_1.txt_3_-1_-2", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)

      self.initialize_table(table)

      input_values = constant_op.constant([0, 1, 2, 3], dtypes.int64)

      out = table.lookup(input_values)
      self.assertAllEqual([b"brain", b"salad", b"surgery", b"UNK"],
                          self.evaluate(out))
      self.assertEqual(vocab_size, self.evaluate(table.size()))

  def testStringToIdTable(self, is_anonymous):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shared-name check uses a true/false assertion on a non-empty string, which is
    # always true, so the shared name is never actually compared with anything.
    # Scenario: the id table initializer maps string keys to line numbers with default minus one. Looks up
    # three words and an unknown word, expects zero, one, two and minus one, and the size equals the
    # vocab size. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_2.txt")
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      init = lookup_ops.TextFileIdTableInitializer(
          vocab_file, vocab_size=vocab_size)
      self.assertTrue("feat_to_id_2.txt_3_-1_-2", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)

      input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

      out = table.lookup(input_string)
      self.assertAllEqual([0, 1, 2, -1], self.evaluate(out))
      self.assertEqual(vocab_size, self.evaluate(table.size()))

  def testInt64ToIdTable(self, is_anonymous):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shared-name check uses a true/false assertion on a non-empty string, which is
    # always true, so the shared name is never actually compared with anything.
    # Scenario: the id table initializer with int64 keys read from a file of numbers. Looks up three known
    # numbers and one unknown, expects zero, one, two and minus one, and table size three.
    # Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile(
        "feat_to_id_3.txt", values=("42", "1", "-1000"))
    with self.cached_session():
      default_value = -1
      vocab_size = 3
      init = lookup_ops.TextFileIdTableInitializer(
          vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64)
      self.assertTrue("feat_to_id_3.txt_3_-1_-2", init._shared_name)
      table = self.getHashTable()(
          init, default_value, experimental_is_anonymous=is_anonymous)
      self.initialize_table(table)

      out = table.lookup(
          constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64))
      self.assertAllEqual((0, 1, 2, -1), self.evaluate(out))
      self.assertEqual(vocab_size, self.evaluate(table.size()))


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class StaticVocabularyTableTest(BaseLookupTableTest):
  # [CLASS DOC]
  # Tests for the static vocabulary table (a static table plus out-of-vocabulary hash buckets),
  # parameterized by the anonymous flag. Covers string, int32 and int64 keys, indexing syntax, a table
  # with no initializer (hash buckets only), sharing one initializer between tables, initialization
  # across sessions, asset tracking, sparse and ragged inputs, a missing inner table, and saved-model
  # round trips. Has its own vocabulary file helper.

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    # [FUNCTION DOC]
    # HELPER: Creates a vocabulary file for the vocabulary table tests.
    # Inputs: basename (file name inside the test temp directory) and values (lines to write, defaults to
    # brain, salad, surgery). Writes one value per line and returns the full path. Side effect: creates
    # or overwrites the file.
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def testStringStaticVocabularyTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: string vocabulary table over a three-word file with one out-of-vocabulary bucket. Looks up
    # three known words and one unknown; expects zero, one, two and three (the bucket id), and a table size
    # of vocab size plus buckets. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    vocab_size = 3
    oov_buckets = 1
    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size),
        oov_buckets,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table)

    input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

    out = table.lookup(input_string)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testStaticVocabularyTableGetItem(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: the same as the basic string vocabulary test but using the indexing syntax for lookup.
    # Expects ids zero to three and size four. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    vocab_size = 3
    oov_buckets = 1
    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size),
        oov_buckets,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table)

    input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

    out = table[input_string]
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testInt32StaticVocabularyTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: int32 lookup keys against an int64 vocabulary file, using the lookup key dtype argument.
    # Expects ids zero, one, two for known numbers and three for an unknown one, and size four.
    # Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_2.txt", ("42", "1", "-1000"))
    vocab_size = 3
    oov_buckets = 1
    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
        oov_buckets,
        lookup_key_dtype=dtypes.int32,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table)

    values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int32)

    out = table.lookup(values)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testInt64StaticVocabularyTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: int64 keys read from a vocabulary file. Expects ids zero, one, two for known numbers and
    # three for the unknown one, and table size four. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_3.txt", ("42", "1", "-1000"))
    vocab_size = 3
    oov_buckets = 1
    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
        oov_buckets,
        experimental_is_anonymous=is_anonymous)

    self.initialize_table(table)

    values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64)

    out = table.lookup(values)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testStringStaticVocabularyTableNoInitializer(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: a vocabulary table created with no initializer and five hash buckets. Every input is
    # mapped by a fingerprint hash modulo five; the test expects three, one and four for the three words
    # and a table size of five. Initializes the table first.
    oov_buckets = 5

    # Set a table that only uses hash buckets, for each input value returns
    # an id calculated by fingerprint("input") mod oov_buckets.
    table = self.getVocabularyTable()(
        None, oov_buckets, experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    values = constant_op.constant(("brain", "salad", "surgery"))

    out = table.lookup(values)
    self.assertAllEqual(
        [
            3,  # fingerprint("brain") mod 5.
            1,  # fingerprint("salad") mod 5.
            4  # fingerprint("surgery") mod 5
        ],
        self.evaluate(out))
    self.assertEqual(oov_buckets, self.evaluate(table.size()))

  def testStaticVocabularyTableWithMultipleInitializers(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: two vocabulary tables share one initializer but have different names and three buckets.
    # After running all table initializers, both give the same ids (five for unknown words including
    # fruit and UNK, zero to two for known words) and both have size six. Skipped for anonymous tables
    # in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    vocab_size = 3
    oov_buckets = 3

    init = lookup_ops.TextFileIdTableInitializer(
        vocab_file, vocab_size=vocab_size)
    table1 = self.getVocabularyTable()(
        init,
        oov_buckets,
        name="table1",
        experimental_is_anonymous=is_anonymous)

    table2 = self.getVocabularyTable()(
        init,
        oov_buckets,
        name="table2",
        experimental_is_anonymous=is_anonymous)

    self.evaluate(lookup_ops.tables_initializer())

    input_string = constant_op.constant(
        ["fruit", "brain", "salad", "surgery", "UNK"])

    out1 = table1.lookup(input_string)
    out2 = table2.lookup(input_string)

    out1, out2 = self.evaluate([out1, out2])
    self.assertAllEqual([5, 0, 1, 2, 5], out1)
    self.assertAllEqual([5, 0, 1, 2, 5], out2)
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table1.size()))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table2.size()))

  def testStaticVocabularyTableInitializationAcrossSessions(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: initialization persists across sessions. A first cached session builds and initializes a
    # table and checks lookups and size. A second cached session builds a new table over the same file
    # without initializing it and still gets correct ids (three for unknown words, one for salad) and
    # size four. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_5.txt")
    with self.cached_session():
      vocab_size = 3
      oov_buckets = 1
      table1 = self.getVocabularyTable()(
          lookup_ops.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          oov_buckets,
          experimental_is_anonymous=is_anonymous)

      self.initialize_table(table1)

      input_string_1 = constant_op.constant(
          ["brain", "salad", "surgery", "UNK"])

      out1 = table1.lookup(input_string_1)

      self.assertAllEqual([0, 1, 2, 3], self.evaluate(out1))
      self.assertEqual(vocab_size + oov_buckets, self.evaluate(table1.size()))

    with self.cached_session():
      vocab_size = 3
      oov_buckets = 1

      # Underlying lookup table already initialized in previous session.
      # No need to initialize table2
      table2 = self.getVocabularyTable()(
          lookup_ops.TextFileIdTableInitializer(
              vocab_file, vocab_size=vocab_size),
          oov_buckets,
          experimental_is_anonymous=is_anonymous)

      input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

      out2 = table2.lookup(input_string_2)

      self.assertAllEqual([3, 1, 3], self.evaluate(out2))
      self.assertEqual(vocab_size + oov_buckets, self.evaluate(table2.size()))

  def testStaticVocabularyTableAssetTracking(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: object-graph asset tracking. Builds a vocabulary table from a file, lists the trackable
    # objects in its object graph, filters those that are assets, and expects exactly one asset whose
    # path equals the vocabulary file. Side effect: writes a vocabulary file.
    vocab_file = self._createVocabFile("vocab.txt")
    vocab_size = 3
    oov_buckets = 1
    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size),
        oov_buckets,
        experimental_is_anonymous=is_anonymous)
    objects = checkpoint_util.list_objects(graph_view.ObjectGraphView(table))
    assets = list(filter(lambda obj: isinstance(obj, asset.Asset), objects))
    self.assertLen(assets, 1)
    self.assertEqual(
        self.evaluate(assets[0].asset_path), compat.as_bytes(vocab_file))

  def testSparseTensor(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: sparse tensor of string keys looked up in a table with one bucket. Checks the output
    # values have static length five and that indices and dense shape are unchanged while the values
    # are zero, one, zero, two, three. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_7.txt")
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                             dtypes.string),
        constant_op.constant(input_shape, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(vocab_file, vocab_size=3),
        1,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testRaggedTensor(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: ragged tensor of string keys looked up in a vocabulary table with one bucket. Checks the
    # flat values have static length five, values are zero, one, zero, two, three, and the row splits
    # are unchanged. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    vocab_file = self._createVocabFile("feat_to_id_7.txt")
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                             dtypes.string),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(vocab_file, vocab_size=3),
        1,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testInt32SparseTensor(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: sparse tensor of int32 keys against a key/value initializer with int64 data, using the
    # lookup key dtype argument. Expects ids zero, one, zero, two, three with indices and shape
    # unchanged. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_shape, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        lookup_key_dtype=dtypes.int32,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt32RaggedTensor(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: ragged tensor of int32 keys against a key/value initializer with int64 data, with the
    # lookup key dtype set to int32. Expects values zero, one, zero, two, three and unchanged row splits.
    # Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        lookup_key_dtype=dtypes.int32,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testInt64SparseTensor(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: sparse tensor of int64 keys with a key/value initializer. Expects ids zero, one, zero, two,
    # three, and unchanged indices and dense shape. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_shape, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt64RaggedTensor(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: ragged tensor of int64 keys with a key/value initializer. Expects values zero, one, zero,
    # two, three and unchanged row splits. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = self.getVocabularyTable()(
        lookup_ops.KeyValueTensorInitializer((42, 1, -1000), (0, 1, 2),
                                             dtypes.int64, dtypes.int64),
        1,
        experimental_is_anonymous=is_anonymous)
    self.initialize_table(table)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testStaticVocabularyTableNoInnerTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: a vocabulary table with no inner table (nothing as initializer, one bucket). Its resource
    # handle must be nothing, since there is no underlying table resource.
    table = self.getVocabularyTable()(
        None, num_oov_buckets=1, experimental_is_anonymous=is_anonymous)
    self.assertIsNone(table.resource_handle)

  @test_util.run_v2_only
  def testSavedModelSaveRestore(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF2 only): saved-model round trip of a vocabulary table with int64 keys from a file
    # with 11, 12, 13 and one bucket. Builds traced lookup, size and reference-counting functions, saves,
    # deletes the root and reloads it. Checks size four, key twelve giving one, unknown key ten giving
    # three, and the reference-counting flag equal to the anonymous flag, before and after loading.
    # Side effect: writes files under the test temp directory.
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    root = autotrackable.AutoTrackable()

    vocab_file = self._createVocabFile("feat_to_id_3.txt", ("11", "12", "13"))
    vocab_size = 3
    oov_buckets = 1
    root.table = self.getVocabularyTable()(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
        oov_buckets,
        experimental_is_anonymous=is_anonymous)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.int64)])
    def lookup(key):
      # [FUNCTION DOC]
      # HELPER: Traced function with a scalar int64 input signature returning the root table lookup of
      # the key. Used by the saved-model test.
      return root.table.lookup(key)

    @def_function.function(input_signature=[])
    def size():
      # [FUNCTION DOC]
      # HELPER: Traced function with empty signature returning the size of the root table. Used by the
      # saved-model test.
      return root.table.size()

    @def_function.function(input_signature=[])
    def is_ref_counting():
      # [FUNCTION DOC]
      # HELPER: Traced function with empty signature returning whether the root table handle is reference
      # counted. Used by the saved-model test.
      return test_ops.is_resource_handle_ref_counting(
          root.table.resource_handle)

    root.lookup = lookup
    root.size = size
    root.is_ref_counting = is_ref_counting

    self.assertEqual(root.table.size(), 4)
    self.assertEqual(root.lookup(12), 1)
    self.assertEqual(root.lookup(10), 3)
    self.assertEqual(root.is_ref_counting(), is_anonymous)

    saved_model_save.save(root, save_path)

    del root
    loaded = saved_model_load.load(save_path)
    self.assertEqual(loaded.size(), 4)
    self.assertEqual(loaded.lookup(12), 1)
    self.assertEqual(loaded.lookup(10), 3)
    self.assertEqual(loaded.is_ref_counting(), is_anonymous)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class DenseHashTableOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the dense hash table (open-addressing table that needs an empty key and a deleted key),
  # parameterized by the anonymous flag. Covers basic insert, remove and lookup, bool, string and float
  # types, vector keys and values, resizing, export, saver and object-based checkpoint save/restore,
  # saved models, reprobing on collisions, custom empty keys, error conditions, resource values, and
  # export shape inference.

  def testBasic(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: int64-to-int64 dense table with empty key zero and deleted key minus one. Starts at size
    # zero, inserts four keys, removes one existing and one missing key, expects size three, and
    # looks up three keys giving zero, minus one, minus one with static shape three. Skipped for anonymous
    # tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
    values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant([12, 15], dtypes.int64)
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([11, 12, 15], dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, -1, -1], result)

  def testGetItem(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: insert four entries and read values through the indexing syntax. Looking up eleven,
    # twelve and an unknown key gives zero, one and minus one with static shape three. Skipped for
    # anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
    values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))

    input_string = constant_op.constant([11, 12, 15], dtypes.int64)
    output = table[input_string]
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testBasicBool(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: int64-to-bool dense table with default false. Inserts four true values, removes one existing
    # and one missing key, expects size three and lookups giving false for the removed key, true for
    # a present key, and false for an unknown key. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
    values = constant_op.constant([True, True, True, True], dtypes.bool)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.bool,
        default_value=False,
        empty_key=0,
        deleted_key=-1,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant([11, 15], dtypes.int64)
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([11, 12, 15], dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([False, True, False], result)

  def testSameEmptyAndDeletedKey(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: empty key and deleted key are both forty-two. Creating the table and evaluating its size
    # must raise an invalid-argument error mentioning that empty and deleted keys must differ.
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Empty and deleted keys"):
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=42,
          deleted_key=42,
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table.size()))

  @test_util.run_v1_only("uses placeholders")
  def testLookupUnknownShape(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only, uses placeholders): lookup with a placeholder of unknown shape. The static
    # output shape is unknown, and running with eleven, twelve and fifteen as the fed keys returns zero,
    # one and minus one. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    with self.cached_session():
      keys = constant_op.constant([11, 12, 13], dtypes.int64)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=0,
          deleted_key=-1,
          experimental_is_anonymous=is_anonymous)

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      placeholder_keys = array_ops.placeholder(dtypes.int64)
      output = table.lookup(placeholder_keys)
      self.assertAllEqual(None, output.get_shape())
      result = output.eval({placeholder_keys: [11, 12, 15]})
      self.assertAllEqual([0, 1, -1], result)

  def testMapStringToFloat(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: string keys to float32 values, with empty key an empty string and deleted key a dollar
    # sign and a tensor default of minus one point five. Inserts four entries, removes one existing and one
    # missing key, expects size three and lookups returning zero, default, three point three, default.
    # Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant(["a", "b", "c", "d"], dtypes.string)
    values = constant_op.constant([0.0, 1.1, 2.2, 3.3], dtypes.float32)
    default_value = constant_op.constant(-1.5, dtypes.float32)
    table = lookup_ops.DenseHashTable(
        dtypes.string,
        dtypes.float32,
        default_value=default_value,
        empty_key="",
        deleted_key="$",
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant(["b", "e"])
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["a", "b", "d", "e"], dtypes.string)
    output = table.lookup(input_string)
    self.assertAllEqual([4], output.get_shape())

    result = self.evaluate(output)
    self.assertAllClose([0, -1.5, 3.3, -1.5], result)

  def testMapInt64ToFloat(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: int64 keys to float values, run for both float32 and float64. For each, inserts four
    # entries, removes one existing and one missing key, checks size three and that lookups return
    # the stored values or the default minus one point five. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    for float_dtype in [dtypes.float32, dtypes.float64]:
      keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
      values = constant_op.constant([0.0, 1.1, 2.2, 3.3], float_dtype)
      default_value = constant_op.constant(-1.5, float_dtype)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          float_dtype,
          default_value=default_value,
          empty_key=0,
          deleted_key=-1,
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table.size()))

      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(4, self.evaluate(table.size()))

      remove_string = constant_op.constant([12, 15], dtypes.int64)
      self.evaluate(table.remove(remove_string))
      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant([11, 12, 14, 15], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([4], output.get_shape())

      result = self.evaluate(output)
      self.assertAllClose([0, -1.5, 3.3, -1.5], result)

  def testVectorValues(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: vector (length four) int64 values with initial bucket count four. Checks the exported
    # key length grows from four to eight after a fourth insert, removing keys does not shrink the table,
    # lookup output shape is four by four, and rows for missing keys equal the default vector.
    # Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]],
                                  dtypes.int64)
    default_value = constant_op.constant([-1, -2, -3, -4], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=0,
        deleted_key=-1,
        initial_num_buckets=4,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))
    self.assertAllEqual(4, len(self.evaluate(table.export()[0])))

    self.evaluate(
        table.insert(
            constant_op.constant([14], dtypes.int64),
            constant_op.constant([[2, 3, 4, 5]], dtypes.int64)))
    self.assertAllEqual(4, self.evaluate(table.size()))
    self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

    remove_string = constant_op.constant([12, 16], dtypes.int64)
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))
    self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

    input_string = constant_op.constant([11, 12, 14, 15], dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([4, 4],
                        output.shape,
                        msg="Saw shape: %s" % output.shape)

    result = self.evaluate(output)
    self.assertAllEqual(
        [[0, 1, 2, 3], [-1, -2, -3, -4], [2, 3, 4, 5], [-1, -2, -3, -4]],
        result)

  def testVectorKeys(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: vector keys of length two (empty key and deleted key are also vectors) with eight
    # initial buckets. Inserts three entries, adds a fourth, removes one existing and one missing key,
    # and verifies sizes, exported length eight, and lookup values ten, default, twelve, default.
    # Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([[0, 1], [1, 2], [1, 3]], dtypes.int64)
    values = constant_op.constant([10, 11, 12], dtypes.int64)
    empty_key = constant_op.constant([0, 3], dtypes.int64)
    deleted_key = constant_op.constant([-1, -1], dtypes.int64)
    default_value = constant_op.constant(-1, dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        initial_num_buckets=8,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    self.evaluate(
        table.insert(
            constant_op.constant([[0, 0]], dtypes.int64),
            constant_op.constant([13], dtypes.int64)))
    self.assertAllEqual(4, self.evaluate(table.size()))
    self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

    remove_string = constant_op.constant([[1, 2], [7, 8]], dtypes.int64)
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))
    self.assertAllEqual(8, len(self.evaluate(table.export()[0])))

    input_string = constant_op.constant([[0, 1], [1, 2], [1, 3], [0, 2]],
                                        dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([4], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([10, -1, 12, -1], result)

  def testResize(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: table resize behavior with four initial buckets. Inserting three keys keeps four buckets;
    # removing a key and a missing key leaves four buckets; inserting five more keys grows the table to
    # sixteen buckets with size six. Final lookup of nine keys returns the expected values or the
    # default. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1,
        initial_num_buckets=4,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))
    self.assertAllEqual(4, len(self.evaluate(table.export()[0])))

    keys2 = constant_op.constant([12, 99], dtypes.int64)
    self.evaluate(table.remove(keys2))
    self.assertAllEqual(2, self.evaluate(table.size()))
    self.assertAllEqual(4, len(self.evaluate(table.export()[0])))

    keys3 = constant_op.constant([13, 14, 15, 16, 17], dtypes.int64)
    values3 = constant_op.constant([3, 4, 5, 6, 7], dtypes.int64)

    self.evaluate(table.insert(keys3, values3))
    self.assertAllEqual(6, self.evaluate(table.size()))
    self.assertAllEqual(16, len(self.evaluate(table.export()[0])))

    keys4 = constant_op.constant([10, 11, 12, 13, 14, 15, 16, 17, 18],
                                 dtypes.int64)
    output = table.lookup(keys4)
    self.assertAllEqual([-1, 0, -1, 3, 4, 5, 6, 7, -1], self.evaluate(output))

  def testExport(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: export of a table with eight buckets, empty key one hundred and deleted key two hundred.
    # After inserting four entries and removing one, exports keys and values, checks both have length
    # eight, pairs and sorts them by key, and expects the three live entries, four empty-slot entries
    # (empty key with value zero) and one deleted-slot entry. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
    values = constant_op.constant([1, 2, 3, 4], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=100,
        deleted_key=200,
        initial_num_buckets=8,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    keys2 = constant_op.constant([12, 15], dtypes.int64)
    self.evaluate(table.remove(keys2))
    self.assertAllEqual(3, self.evaluate(table.size()))

    exported_keys, exported_values = table.export()

    np_keys = self.evaluate(exported_keys)
    np_values = self.evaluate(exported_values)

    self.assertAllEqual(8, len(np_keys))
    self.assertAllEqual(8, len(np_values))

    # pair up keys and values, drop extra added dimension
    pairs = np.dstack((np_keys.flatten(), np_values.flatten()))[0]
    # sort by key
    pairs = pairs[pairs[:, 0].argsort()]
    self.assertAllEqual([[11, 1], [13, 3], [14, 4], [100, 0], [100, 0],
                         [100, 0], [100, 0], [200, 2]], pairs)

  @test_util.run_v1_only("Saver V1 only")
  def testSaveRestore(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only, saver): saves a table with thirty-two buckets and three entries using the
    # standard saver to a temp directory, then in a new graph creates a table with sixty-four buckets and
    # different content, restores, and checks the table is restored to size three with thirty-two buckets
    # and the expected lookups. Skipped for anonymous tables in TF1. Side effect: writes checkpoint
    # files.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      default_value = -1
      empty_key = 0
      deleted_key = -1
      keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver()

      self.assertAllEqual(0, table.size())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([12, 15], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, str)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64,
          experimental_is_anonymous=is_anonymous)
      table.insert(
          constant_op.constant([11, 14], dtypes.int64),
          constant_op.constant([12, 24], dtypes.int64)).run()
      self.assertAllEqual(2, table.size())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, -1, 2, 3], output)

  @test_util.run_v1_only("Saver V1 only")
  def testSaveRestoreOnlyTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only): same as the saver round trip, but the saver is constructed with only the
    # table in its variable list. Verifies restoring gives size three, thirty-two buckets and the
    # expected lookups. Side effect: writes checkpoint files in the temp directory.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      default_value = -1
      empty_key = 0
      deleted_key = -1
      keys = constant_op.constant([11, 12, 13, 14], dtypes.int64)
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver([table])

      self.assertAllEqual(0, table.size())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([12, 15], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, str)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64,
          experimental_is_anonymous=is_anonymous)
      table.insert(
          constant_op.constant([11, 14], dtypes.int64),
          constant_op.constant([12, 24], dtypes.int64)).run()
      self.assertAllEqual(2, table.size())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver([table])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, -1, 2, 3], output)

  @test_util.run_in_graph_and_eager_modes
  def testObjectSaveRestore(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (graph and eager): object-based checkpoint save and restore. Saves a table with
    # thirty-two buckets and three entries via a trackable checkpoint, deletes it, creates a new table
    # with sixty-four buckets and different content, restores, and checks the size three, thirty-two
    # buckets and lookups giving minus one, zero, one, two, minus one. Skipped for anonymous tables
    # in graph mode. Side effect: writes checkpoint files.
    if is_anonymous and not context.executing_eagerly():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_prefix = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    default_value = -1
    empty_key = 0
    deleted_key = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    save_table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=32,
        experimental_is_anonymous=is_anonymous)

    save_checkpoint = trackable.Checkpoint(table=save_table)

    self.assertAllEqual(0, self.evaluate(save_table.size()))
    self.evaluate(save_table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(save_table.size()))
    self.assertAllEqual(32, len(self.evaluate(save_table.export()[0])))

    save_path = save_checkpoint.save(save_prefix)
    del save_table, save_checkpoint

    load_table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=64,
        experimental_is_anonymous=is_anonymous)
    self.evaluate(
        load_table.insert(
            constant_op.constant([11, 14], dtypes.int64),
            constant_op.constant([12, 24], dtypes.int64)))
    self.assertAllEqual(2, self.evaluate(load_table.size()))
    self.assertAllEqual(64, len(self.evaluate(load_table.export()[0])))

    restore_checkpoint = trackable.Checkpoint(table=load_table)

    # Restore the saved values in the parameter nodes.
    restore_checkpoint.restore(save_path).run_restore_ops()

    self.assertAllEqual(3, self.evaluate(load_table.size()))
    self.assertAllEqual(32, len(self.evaluate(load_table.export()[0])))

    input_string = constant_op.constant([10, 11, 12, 13, 14], dtypes.int64)
    output = load_table.lookup(input_string)
    self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_v2_only
  def testSavedModelSaveRestore(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF2 only): saved-model round trip of a dense table with checkpointing enabled. Inserts
    # three entries, checks size, lookups, thirty-two exported keys and the reference-counting flag,
    # saves a root holding the table and traced functions, reloads, and checks size three, lookups and
    # the flag again. Side effect: writes files under the test temp directory.
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    root = autotrackable.AutoTrackable()

    default_value = -1
    empty_key = 0
    deleted_key = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    root.table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        checkpoint=True,
        initial_num_buckets=32,
        experimental_is_anonymous=is_anonymous)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.int64)])
    def lookup(key):
      # [FUNCTION DOC]
      # HELPER: Traced function with a scalar int64 input signature returning the root table lookup of the
      # key. Used by the saved-model test.
      return root.table.lookup(key)

    @def_function.function(input_signature=[])
    def size():
      # [FUNCTION DOC]
      # HELPER: Traced function with empty signature returning the root table size. Used by the
      # saved-model test.
      return root.table.size()

    @def_function.function(input_signature=[])
    def is_ref_counting():
      # [FUNCTION DOC]
      # HELPER: Traced function with empty signature returning whether the root table handle is
      # reference counted. Used by the saved-model test.
      return test_ops.is_resource_handle_ref_counting(
          root.table.resource_handle)

    root.lookup = lookup
    root.size = size
    root.is_ref_counting = is_ref_counting

    self.assertEqual(root.table.size(), 0)
    root.table.insert(keys, values)
    self.assertEqual(root.table.size(), 3)
    self.assertEqual(root.table.lookup(12), 1)
    self.assertEqual(root.table.lookup(10), -1)
    self.assertEqual(len(root.table.export()[0]), 32)
    self.assertEqual(root.is_ref_counting(), is_anonymous)

    saved_model_save.save(root, save_path)

    del root
    loaded = saved_model_load.load(save_path)
    self.assertEqual(loaded.size(), 3)
    self.assertEqual(loaded.lookup(12), 1)
    self.assertEqual(loaded.lookup(10), -1)
    self.assertEqual(loaded.is_ref_counting(), is_anonymous)

  @test_util.run_v1_only("Saver V1 only")
  def testVectorSaveRestore(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only, saver): vector keys and vector values with a saver round trip. Saves a table
    # with four entries minus one removed, then restores into a table with sixty-four buckets and
    # different content, and checks the restored size three, thirty-two buckets and the expected vector
    # lookups. Side effect: writes checkpoint files. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "vector_save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-2, -3], dtypes.int64)
      default_value = constant_op.constant([-1, -2], dtypes.int64)
      keys = constant_op.constant([[11, 12], [11, 14], [12, 13], [13, 14]],
                                  dtypes.int64)
      values = constant_op.constant([[0, 1], [2, 3], [2, 4], [4, 5]],
                                    dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=32,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver()

      self.assertAllEqual(0, table.size())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([[12, 13], [16, 17]], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, str)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-2, -3], dtypes.int64)
      default_value = constant_op.constant([-1, -2], dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t1",
          checkpoint=True,
          initial_num_buckets=64,
          experimental_is_anonymous=is_anonymous)
      table.insert(
          constant_op.constant([[11, 12], [13, 15]], dtypes.int64),
          constant_op.constant([[21, 22], [23, 24]], dtypes.int64)).run()
      self.assertAllEqual(2, table.size())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant(
          [[11, 12], [11, 14], [11, 15], [13, 14], [13, 15]], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([[0, 1], [2, 3], [-1, -2], [4, 5], [-1, -2]],
                          self.evaluate(output))

  @test_util.run_v1_only("Saver V1 only")
  def testVectorScalarSaveRestore(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only, saver): vector keys with scalar values and a saver round trip with a table named
    # t2. Saves, then restores into a larger table with different content, and checks size three,
    # thirty-two buckets and lookups zero, one, minus one, three, minus one. Side effect: writes
    # checkpoint files. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "vector_scalar_save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-1, -1], dtypes.int64)
      default_value = constant_op.constant(-1, dtypes.int64)
      keys = constant_op.constant([[11, 12], [11, 14], [12, 13], [13, 14]],
                                  dtypes.int64)
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t2",
          checkpoint=True,
          initial_num_buckets=32,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver()

      self.assertAllEqual(0, table.size())
      table.insert(keys, values).run()
      self.assertAllEqual(4, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      keys2 = constant_op.constant([[12, 13], [15, 16]], dtypes.int64)
      table.remove(keys2).run()
      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, str)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      empty_key = constant_op.constant([11, 13], dtypes.int64)
      deleted_key = constant_op.constant([-1, -1], dtypes.int64)
      default_value = constant_op.constant(-1, dtypes.int64)
      table = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=default_value,
          empty_key=empty_key,
          deleted_key=deleted_key,
          name="t2",
          checkpoint=True,
          initial_num_buckets=64,
          experimental_is_anonymous=is_anonymous)
      table.insert(
          constant_op.constant([[11, 12], [13, 15]], dtypes.int64),
          constant_op.constant([3, 4], dtypes.int64)).run()
      self.assertAllEqual(2, table.size())
      self.assertAllEqual(64, len(table.export()[0].eval()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)

      self.assertAllEqual(3, table.size())
      self.assertAllEqual(32, len(table.export()[0].eval()))

      input_string = constant_op.constant(
          [[11, 12], [11, 14], [11, 15], [13, 14], [13, 15]], dtypes.int64)
      output = table.lookup(input_string)
      self.assertAllEqual([0, 1, -1, 3, -1], output)

  def testReprobe(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: reprobing. Inserts six keys into a table with eight buckets chosen to collide, then
    # looks up nine keys. Expects size six and that colliding keys are all found, with default minus
    # one for absent keys. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    # Insert 6 keys into a table with 8 buckets.
    # The values are chosen to make sure collisions occur when using GCC STL
    keys = constant_op.constant([11, 12, 13, 19, 20, 21], dtypes.int64)
    values = constant_op.constant([51, 52, 53, 54, 55, 56], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1,
        initial_num_buckets=8,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(6, self.evaluate(table.size()))

    input_string = constant_op.constant([10, 11, 12, 13, 14, 19, 20, 21, 22],
                                        dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([9], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([-1, 51, 52, 53, -1, 54, 55, 56, -1], result)

  def testCustomEmptyKey(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: a custom empty key of twelve allows zero as a normal key. Inserts keys eleven, zero and
    # thirteen, checks size three, and looks up eleven, zero and fifteen expecting zero, one and the
    # default. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    keys = constant_op.constant([11, 0, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=12,
        deleted_key=-1,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([11, 0, 15], dtypes.int64)
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testErrors(self, is_anonymous):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment before the deleted-key lookup says it looks up the empty key, but the
    # code looks up keys that contain the deleted key (copy-paste comment).
    # Scenario: error conditions. Inserting or looking up the empty key or the deleted key must raise an
    # invalid-argument error naming the key; multi-dimensional keys raise an error about the expected key
    # shape; an initial bucket count of twelve (not a power of two), empty and deleted keys of different
    # shapes, and equal empty and deleted keys (scalar or vector) each raise invalid-argument errors on
    # creation.
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1,
        experimental_is_anonymous=is_anonymous)

    # Inserting the empty key returns an error
    keys1 = constant_op.constant([11, 0], dtypes.int64)
    values1 = constant_op.constant([0, 1], dtypes.int64)
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "empty_key"):
      self.evaluate(table.insert(keys1, values1))

    # Looking up the empty key returns an error
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "empty_key"):
      self.evaluate(table.lookup(keys1))

    # Inserting the deleted key returns an error
    keys2 = constant_op.constant([11, -1], dtypes.int64)
    values2 = constant_op.constant([0, 1], dtypes.int64)
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "deleted_key"):
      self.evaluate(table.insert(keys2, values2))

    # Looking up the empty key returns an error
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "deleted_key"):
      self.evaluate(table.lookup(keys2))

    # Arbitrary tensors of keys are not supported
    keys = constant_op.constant([[11, 0], [12, 1]], dtypes.int64)
    values = constant_op.constant([[11, 0], [12, 1]], dtypes.int64)
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Expected key shape"):
      self.evaluate(table.lookup(keys))
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Expected key shape"):
      self.evaluate(table.insert(keys, values))

    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Number of buckets must be"):
      table2 = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=17,
          deleted_key=-1,
          initial_num_buckets=12,
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table2.size()))

    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        "Empty and deleted keys must have same shape"):
      table3 = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=42,
          deleted_key=[1, 2],
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table3.size()))

    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Empty and deleted keys cannot be equal"):
      table4 = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=42,
          deleted_key=42,
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table4.size()))

    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Empty and deleted keys cannot be equal"):
      table5 = lookup_ops.DenseHashTable(
          dtypes.int64,
          dtypes.int64,
          default_value=-1,
          empty_key=[1, 2, 3],
          deleted_key=[1, 2, 3],
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table5.size()))

  @test_util.run_in_graph_and_eager_modes
  def testStringToResource(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (graph and eager): table with string keys and resource-handle values, using variable
    # handles as default and as inserted value. Looking up a missing key and a present key must both yield
    # a scalar shape.
    v = variables.Variable(1.)
    v1 = variables.Variable(1.)
    table = lookup_ops.DenseHashTable(
        dtypes.string,
        dtypes.resource,
        default_value=v.handle,
        empty_key="<empty>",
        deleted_key="<deleted>",
        experimental_is_anonymous=is_anonymous)
    self.assertEqual([], table.lookup("not_found").shape)
    table.insert("v1", v1.handle)
    self.assertEqual([], table.lookup("v1").shape)

  def testExportShapeInference(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: shape inference for table export on a dense table. Compares shapes of exported keys and
    # values taken directly with those inferred while tracing a function; both lists have two entries and
    # each inferred shape is compatible with the actual one.
    default_value = -1
    empty_key = 0
    deleted_key = -1
    table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        experimental_is_anonymous=is_anonymous)
    actual_shapes = [t.shape for t in table.export()]
    inferred_shapes = []

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: Traced function for the export shape test. Takes no inputs; exports the table and
      # appends each exported tensor's shape to a list in the enclosing test (side effect). Returns nothing.
      for t in table.export():
        inferred_shapes.append(t.shape)

    f()
    self.assertLen(actual_shapes, 2)
    self.assertLen(inferred_shapes, 2)
    self.assertTrue(inferred_shapes[0].is_compatible_with(actual_shapes[0]))
    self.assertTrue(inferred_shapes[1].is_compatible_with(actual_shapes[1]))


class IndexTableFromFile(test.TestCase):
  # [CLASS DOC]
  # Tests for the factory that builds a string or numeric index table from a vocabulary file. Covers
  # string, multi-column, custom delimiter, tensor and placeholder file names, int32 and int64 keys,
  # default value, out-of-vocabulary buckets, vocab size validation, hasher validation, and the
  # presence of a table resource handle. Has a vocabulary file helper. Not parameterized.

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    # [FUNCTION DOC]
    # HELPER: Creates a vocabulary file for the index-table-from-file tests.
    # Inputs: basename (file name in the test temp directory) and values (lines, defaults to brain,
    # salad, surgery). Writes one line per value with a trailing newline and returns the full path.
    # Side effect: creates or overwrites the file.
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def test_string_index_table_from_file(self):
    # [FUNCTION DOC]
    # Scenario: index table from a three-word file with one bucket. In graph mode, evaluating the lookup
    # before initialization raises an op error. After running the tables initializer, salad, surgery and
    # an unknown word map to one, two and three.
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")

    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, num_oov_buckets=1)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_string_index_table_from_multicolumn_file(self):
    # [FUNCTION DOC]
    # Scenario: two-column tab-separated file where words are keys and values come from line numbers.
    # Same checks as the basic test: pre-initialization error in graph mode, then ids one, two, three.
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain\t300", "salad\t20", "surgery\t1"))
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=1,
        key_column_index=0,
        value_column_index=lookup_ops.TextFileIndex.LINE_NUMBER)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_string_index_table_from_multicolumn_file_custom_delimiter(self):
    # [FUNCTION DOC]
    # Scenario: two-column file separated by spaces, with a custom delimiter argument. Same checks as
    # the multi-column test: error before initialization in graph mode, then ids one, two, three.
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain 300", "salad 20", "surgery 1"))
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=1,
        key_column_index=0,
        value_column_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        delimiter=" ")
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_string_index_table_from_file_tensor_filename(self):
    # [FUNCTION DOC]
    # Scenario: the vocabulary file name is passed as a string tensor. Checks the same ids as the
    # basic test, and in graph mode that exactly one asset file path is recorded in the graph collection.
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    vocabulary_file = constant_op.constant(vocabulary_file)
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, num_oov_buckets=1)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))
    if not context.executing_eagerly():
      self.assertEqual(1,
                       len(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)))

  @test_util.run_v1_only("placeholder usage")
  def test_string_index_table_from_file_placeholder_filename(self):
    # [FUNCTION DOC]
    # Scenario (TF1 only, placeholders): the vocabulary file name is a string placeholder. Lookup fails
    # before initialization; running the tables initializer with the fed file name works and gives ids
    # one, two, three; no asset file path is recorded in the collection.
    vocabulary_file = self._createVocabFile("f2i_vocab1.txt")
    with self.cached_session():
      vocabulary_placeholder = array_ops.placeholder(dtypes.string, [])
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_placeholder, num_oov_buckets=1)
      ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)

      feed_dict = {vocabulary_placeholder.name: vocabulary_file}
      lookup_ops.tables_initializer().run(feed_dict=feed_dict)
      self.assertAllEqual((1, 2, 3), self.evaluate(ids))
      self.assertEqual(0,
                       len(ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS)))

  def test_int32_index_table_from_file(self):
    # [FUNCTION DOC]
    # Scenario: int32 keys from a file of numbers with one bucket. Lookup of one, minus one thousand and
    # eleven gives one, two and three. Graph mode first checks the pre-initialization error.
    vocabulary_file = self._createVocabFile(
        "f2i_vocab2.txt", values=("42", "1", "-1000"))
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=1,
        key_dtype=dtypes.int32)
    ids = table.lookup(constant_op.constant((1, -1000, 11), dtype=dtypes.int32))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int64_index_table_from_file(self):
    # [FUNCTION DOC]
    # Scenario: int64 keys from a file of numbers with one bucket. Lookup gives one, two and three, with
    # the pre-initialization error check in graph mode.
    vocabulary_file = self._createVocabFile(
        "f2i_vocab3.txt", values=("42", "1", "-1000"))
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file,
        num_oov_buckets=1,
        key_dtype=dtypes.int64)
    ids = table.lookup(constant_op.constant((1, -1000, 11), dtype=dtypes.int64))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_index_table_from_file_with_default_value(self):
    # [FUNCTION DOC]
    # Scenario: a custom default value of minus forty-two and no buckets. Lookup of salad, surgery and an
    # unknown word gives one, two and the custom default. Pre-initialization error checked in graph mode.
    default_value = -42
    vocabulary_file = self._createVocabFile("f2i_vocab4.txt")
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, default_value=default_value)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, default_value), self.evaluate(ids))

  def test_index_table_from_file_with_oov_buckets(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comments in the expected values talk about "mod 300" while the test uses one
    # thousand buckets, so the comments are stale or wrong; the numeric expectations are 867 and 860.
    # Scenario: one thousand out-of-vocabulary buckets. Known words give one and two; two unknown words
    # get hash-derived ids three plus their fingerprint modulo the bucket count (867 and 860).
    # Pre-initialization error is checked in graph mode.
    vocabulary_file = self._createVocabFile("f2i_vocab5.txt")
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, num_oov_buckets=1000)
    ids = table.lookup(
        constant_op.constant(["salad", "surgery", "tarkus", "toccata"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual(
        (
            1,  # From vocabulary file.
            2,  # From vocabulary file.
            867,  # 3 + fingerprint("tarkus") mod 300.
            860),  # 3 + fingerprint("toccata") mod 300.
        self.evaluate(ids))

  def test_index_table_from_file_fails_with_empty_vocabulary_file_name(self):
    # [FUNCTION DOC]
    # Scenario: an empty string as vocabulary file name must make the factory raise a value error.
    self.assertRaises(
        ValueError, lookup_ops.index_table_from_file, vocabulary_file="")

  def test_index_table_from_file_fails_with_empty_vocabulary(self):
    # [FUNCTION DOC]
    # Scenario: nothing as vocabulary file name must make the factory raise a value error.
    self.assertRaises(
        ValueError, lookup_ops.index_table_from_file, vocabulary_file=None)

  def test_index_table_from_file_str_fails_with_zero_size_vocabulary(self):
    # [FUNCTION DOC]
    # Scenario: vocab size zero with a string file name must raise a value error whose message says the
    # vocab size must be greater than zero and names the file.
    vocabulary_file = self._createVocabFile("zero_vocab_str.txt")
    self.assertRaisesRegex(
        ValueError, "`vocab_size` must be greater than 0, got 0 for "
        "vocabulary_file: .*zero_vocab_str.txt",
        lookup_ops.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

  def test_index_table_from_file_tensor_fails_with_zero_size_vocabulary(self):
    # [FUNCTION DOC]
    # Scenario: vocab size zero with a tensor file name must raise a value error whose message says the
    # vocab size must be greater than zero and names the file.
    vocabulary_file = constant_op.constant(
        self._createVocabFile("zero_vocab_tensor.txt"))
    self.assertRaisesRegex(
        ValueError, "`vocab_size` must be greater than 0, got 0 for "
        "vocabulary_file: .*zero_vocab_tensor.txt",
        lookup_ops.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

  def test_index_table_from_file_with_vocab_size_too_small(self):
    # [FUNCTION DOC]
    # Scenario: vocab size two on a three-word file. Only the first two words are loaded: salad maps to
    # one, surgery and tarkus map to the default minus one, and table size is two. Pre-initialization
    # error is checked in graph mode.
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, vocab_size=2)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, -1, -1), self.evaluate(ids))
    self.assertEqual(2, self.evaluate(table.size()))

  def test_index_table_from_file_with_vocab_size_too_large(self):
    # [FUNCTION DOC]
    # Scenario: vocab size four on a three-word file. Running the table initializer must raise an
    # invalid-argument error about the vocab size.
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Invalid vocab_size"):
      table = lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=4)
      self.evaluate(table.initializer)

  def test_index_table_from_file_with_vocab_size(self):
    # [FUNCTION DOC]
    # Scenario: vocab size handling. Size zero raises a value error. Size three loads the whole file:
    # ids one, two and default minus one, and table size three. Pre-initialization error is checked in
    # graph mode.
    vocabulary_file = self._createVocabFile("f2i_vocab8.txt")

    self.assertRaises(
        ValueError,
        lookup_ops.index_table_from_file,
        vocabulary_file=vocabulary_file,
        vocab_size=0)

    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, vocab_size=3)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, -1), self.evaluate(ids))
    self.assertEqual(3, self.evaluate(table.size()))

  def test_index_table_from_file_with_invalid_hashers(self):
    # [FUNCTION DOC]
    # Scenario: invalid hasher specs. A plain number as hasher spec raises a type error. A hasher spec
    # with an unknown hasher name is accepted at creation but a later lookup raises a value error.
    vocabulary_file = self._createVocabFile("invalid_hasher.txt")
    with self.assertRaises(TypeError):
      lookup_ops.index_table_from_file(
          vocabulary_file=vocabulary_file,
          vocab_size=3,
          num_oov_buckets=1,
          hasher_spec=1)

    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file,
        vocab_size=3,
        num_oov_buckets=1,
        hasher_spec=lookup_ops.HasherSpec("my-awesome-hash", None))

    self.assertRaises(ValueError, table.lookup,
                      constant_op.constant(["salad", "surgery", "tarkus"]))

  def test_index_table_from_file_table_ref_with_oov_buckets(self):
    # [FUNCTION DOC]
    # Scenario: a table with one out-of-vocabulary bucket exposes a resource handle that is not nothing.
    vocabulary_file = self._createVocabFile("f2i_vocab9.txt")
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, num_oov_buckets=1)
    self.assertIsNotNone(table.resource_handle)

  def test_index_table_from_file_table_ref_without_oov_buckets(self):
    # [FUNCTION DOC]
    # Scenario: a table with zero buckets exposes a resource handle that is not nothing.
    vocabulary_file = self._createVocabFile("f2i_vocab10.txt")
    table = lookup_ops.index_table_from_file(
        vocabulary_file=vocabulary_file, num_oov_buckets=0)
    self.assertIsNotNone(table.resource_handle)


class IndexTableFromTensor(test.TestCase):
  # [CLASS DOC]
  # Tests for the factory that builds an index table from an in-memory vocabulary list. Covers
  # tensor-style initialization, int32 and int64 vocabularies, default value, a missing or empty
  # vocabulary list, and invalid hasher specs. Not parameterized.

  @test_util.run_in_graph_and_eager_modes
  def test_index_table_from_tensor_with_tensor_init(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager): table from a tuple of words with one bucket. In graph mode, lookup
    # before initialization raises an op error; in eager mode, recreating a table works. After running
    # the tables initializer, salad, surgery and tarkus map to one, two and three.
    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=("brain", "salad", "surgery"), num_oov_buckets=1)

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(
            table.lookup(constant_op.constant(("salad", "surgery", "tarkus"))))
    else:
      # Reinitializing a table in eager should work.
      table = lookup_ops.index_table_from_tensor(
          vocabulary_list=("brain", "salad", "surgery"), num_oov_buckets=1)
    self.evaluate(lookup_ops.tables_initializer())
    ids = table.lookup(constant_op.constant(("salad", "surgery", "tarkus")))
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int32_index_table_from_tensor_with_tensor_init(self):
    # [FUNCTION DOC]
    # Scenario: int32 vocabulary list with int32 dtype argument and one bucket. Lookup gives one, two
    # and three; in graph mode lookup before initialization raises a failed-precondition error.
    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=(42, 1, -1000), num_oov_buckets=1, dtype=dtypes.int32)
    ids = table.lookup(constant_op.constant((1, -1000, 11), dtype=dtypes.int32))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_int64_index_table_from_tensor_with_tensor_init(self):
    # [FUNCTION DOC]
    # Scenario: int64 vocabulary list with int64 dtype and one bucket. Lookup gives one, two and three;
    # in graph mode lookup before initialization raises a failed-precondition error.
    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=(42, 1, -1000), num_oov_buckets=1, dtype=dtypes.int64)
    ids = table.lookup(constant_op.constant((1, -1000, 11), dtype=dtypes.int64))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, 3), self.evaluate(ids))

  def test_index_table_from_tensor_with_default_value(self):
    # [FUNCTION DOC]
    # Scenario: default value of minus forty-two and no buckets. Lookup of two known words and one unknown
    # returns one, two and the default; in graph mode lookup before initialization raises a
    # failed-precondition error.
    default_value = -42
    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=["brain", "salad", "surgery"],
        default_value=default_value)
    ids = table.lookup(constant_op.constant(["salad", "surgery", "tarkus"]))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.FailedPreconditionError):
        self.evaluate(ids)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((1, 2, default_value), self.evaluate(ids))

  def test_index_table_from_tensor_missing_vocabulary_list(self):
    # [FUNCTION DOC]
    # Scenario: factory called with nothing as the vocabulary list must raise a value error saying the
    # list must be specified.
    with self.assertRaisesRegex(ValueError,
                                "`vocabulary_list` must be specified"):
      lookup_ops.index_table_from_tensor(
          vocabulary_list=None, num_oov_buckets=1)

  def test_index_table_from_tensor_empty_vocabulary_list(self):
    # [FUNCTION DOC]
    # Scenario: factory called with an empty string array. Evaluating the table initializer must raise an
    # op error saying keys and values cannot be empty.
    with self.assertRaisesRegex(errors_impl.OpError,
                                "keys and values cannot be empty"):
      _ = lookup_ops.index_table_from_tensor(
          vocabulary_list=np.array([], dtype=np.str_), num_oov_buckets=1)
      self.evaluate(lookup_ops.tables_initializer())

  def test_index_table_from_tensor_with_invalid_hashers(self):
    # [FUNCTION DOC]
    # Scenario: invalid hasher specs for the tensor-based factory. A plain number raises a type error;
    # a hasher spec with an unknown name builds a table but a later lookup raises a value error.
    with self.assertRaises(TypeError):
      lookup_ops.index_table_from_tensor(
          vocabulary_list=["brain", "salad", "surgery"],
          num_oov_buckets=1,
          hasher_spec=1)

    table = lookup_ops.index_table_from_tensor(
        vocabulary_list=["brain", "salad", "surgery"],
        num_oov_buckets=1,
        hasher_spec=lookup_ops.HasherSpec("my-awesome-hash", None))

    self.assertRaises(ValueError, table.lookup,
                      constant_op.constant(["salad", "surgery", "tarkus"]))


class IndexToStringTableFromFileTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the factory that builds an index-to-string table from a vocabulary file. Covers string and
  # tensor file names, multi-column files, custom delimiter, default value, and the vocab size rules.
  # Has its own vocabulary file helper. Not parameterized.

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    # [FUNCTION DOC]
    # HELPER: Creates a vocabulary file for the index-to-string tests.
    # Inputs: basename (file name in the test temp directory) and values (lines, defaults to brain,
    # salad, surgery). Writes one line per value with a trailing newline and returns the full path.
    # Side effect: creates or overwrites the file.
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def test_index_to_string_table(self):
    # [FUNCTION DOC]
    # Scenario: the file name may be given as a string or as a string tensor. For each form, builds the
    # table, looks up ids zero to three, and expects brain, salad, surgery and the default "UNK".
    # In graph mode, lookup before initialization raises an op error.
    vocabulary_path = self._createVocabFile("i2f_vocab1.txt")
    # vocabulary_file supports string and tensor
    type_funcs = [str, constant_op.constant]
    for type_func in type_funcs:
      vocabulary_file = type_func(vocabulary_path)
      table = lookup_ops.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file)
      features = table.lookup(constant_op.constant([0, 1, 2, 3], dtypes.int64))
      if not context.executing_eagerly():
        with self.assertRaises(errors_impl.OpError):
          self.evaluate(features)
      self.evaluate(lookup_ops.tables_initializer())
      self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                          self.evaluate(features))

  def test_index_to_string_table_from_multicolumn_file(self):
    # [FUNCTION DOC]
    # Scenario: two-column tab-separated file where the line number is the key and the first column
    # is the value. Lookup of ids zero to three returns the three words and "UNK". Graph mode checks the
    # pre-initialization error.
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain\t300", "salad\t20", "surgery\t1"))
    table = lookup_ops.index_to_string_table_from_file(
        vocabulary_file=vocabulary_file,
        key_column_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        value_column_index=0)
    features = table.lookup(constant_op.constant([0, 1, 2, 3], dtypes.int64))
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                        self.evaluate(features))

  def test_index_to_string_table_from_multicolumn_file_custom_delimiter(self):
    # [FUNCTION DOC]
    # Scenario: two-column file separated by spaces, using the delimiter argument, otherwise the same as
    # the multi-column test.
    vocabulary_file = self._createVocabFile(
        "f2i_vocab1.txt", values=("brain 300", "salad 20", "surgery 1"))
    table = lookup_ops.index_to_string_table_from_file(
        vocabulary_file=vocabulary_file,
        key_column_index=lookup_ops.TextFileIndex.LINE_NUMBER,
        value_column_index=0,
        delimiter=" ")
    features = table.lookup(constant_op.constant([0, 1, 2, 3], dtypes.int64))
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                        self.evaluate(features))

  def test_index_to_string_table_with_default_value(self):
    # [FUNCTION DOC]
    # Scenario: custom default value of "NONE". Looking up ids one, two and four gives salad, surgery and
    # the custom default. Graph mode checks the pre-initialization error.
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    table = lookup_ops.index_to_string_table_from_file(
        vocabulary_file=vocabulary_file, default_value=default_value)
    features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"salad", b"surgery", default_value),
                        self.evaluate(features))

  def test_index_to_string_table_with_vocab_size_too_small(self):
    # [FUNCTION DOC]
    # Scenario: vocab size two with a custom default. Looking up ids one, two and four gives salad and the
    # default for both missing ids. Graph mode checks the pre-initialization error.
    default_value = b"NONE"
    vocabulary_file = self._createVocabFile("f2i_vocab2.txt")
    table = lookup_ops.index_to_string_table_from_file(
        vocabulary_file=vocabulary_file,
        vocab_size=2,
        default_value=default_value)
    features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"salad", default_value, default_value),
                        self.evaluate(features))

  def test_index_to_string_table_with_vocab_size_too_large(self):
    # [FUNCTION DOC]
    # Scenario: vocab size four on a three-line file. Creating the table and running the tables
    # initializer must raise an invalid-argument error about the vocab size.
    vocabulary_file = self._createVocabFile("f2i_vocab6.txt")
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                "Invalid vocab_size"):
      _ = lookup_ops.index_to_string_table_from_file(
          vocabulary_file=vocabulary_file, vocab_size=4)
      self.evaluate(lookup_ops.tables_initializer())

  def test_index_to_string_table_with_vocab_size(self):
    # [FUNCTION DOC]
    # Scenario: vocab size three on a three-line file. Looking up ids one, two and four gives salad,
    # surgery and "UNK". Graph mode checks the pre-initialization error.
    vocabulary_file = self._createVocabFile("f2i_vocab7.txt")
    table = lookup_ops.index_to_string_table_from_file(
        vocabulary_file=vocabulary_file, vocab_size=3)
    features = table.lookup(constant_op.constant([1, 2, 4], dtypes.int64))

    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"salad", b"surgery", b"UNK"), self.evaluate(features))


class IndexToStringTableFromTensorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the factory that builds an index-to-string table from an in-memory vocabulary list tensor:
  # basic lookup, duplicated entries, and a custom default value. Not parameterized.

  def test_index_to_string_table_from_tensor(self):
    # [FUNCTION DOC]
    # Scenario: builds the table from a string tensor of three words. Looks up ids zero to three and
    # expects the three words and "UNK". Graph mode checks the pre-initialization error.
    vocabulary_list = constant_op.constant(["brain", "salad", "surgery"])
    table = lookup_ops.index_to_string_table_from_tensor(
        vocabulary_list=vocabulary_list)

    indices = constant_op.constant([0, 1, 2, 3], dtypes.int64)
    features = table.lookup(indices)
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual((b"brain", b"salad", b"surgery", b"UNK"),
                        self.evaluate(features))

  def test_duplicate_entries(self):
    # [FUNCTION DOC]
    # Scenario: the vocabulary list contains the same word twice. Lookup of ids zero, one and four gives
    # the word, the word and "UNK", showing duplicates are allowed.
    vocabulary_list = constant_op.constant(["hello", "hello"])
    table = lookup_ops.index_to_string_table_from_tensor(
        vocabulary_list=vocabulary_list)
    indices = constant_op.constant([0, 1, 4], dtypes.int64)
    features = table.lookup(indices)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"hello", b"hello", b"UNK"), self.evaluate(features))

  def test_index_to_string_with_default_value(self):
    # [FUNCTION DOC]
    # Scenario: custom default value "NONE". Looking up ids one, two and four returns salad, surgery
    # and the default. Graph mode checks the pre-initialization error.
    default_value = b"NONE"
    vocabulary_list = constant_op.constant(["brain", "salad", "surgery"])
    table = lookup_ops.index_to_string_table_from_tensor(
        vocabulary_list=vocabulary_list, default_value=default_value)
    indices = constant_op.constant([1, 2, 4], dtypes.int64)
    features = table.lookup(indices)
    if not context.executing_eagerly():
      with self.assertRaises(errors_impl.OpError):
        self.evaluate(features)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertAllEqual((b"salad", b"surgery", default_value),
                        self.evaluate(features))


class IdTableWithHashBucketsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the id table with hash buckets, a wrapper that maps words in a vocabulary to ids and sends
  # other words to hash buckets. Covers string, int32 and int64 keys, tables with only buckets,
  # unsupported key dtypes, two tables sharing an inner table with different hashers, initialization
  # across sessions, different defaults, sparse and ragged inputs, invalid hashers, and a missing inner
  # table. Has its own vocabulary file helper. Not parameterized.

  def _createVocabFile(self, basename, values=("brain", "salad", "surgery")):
    # [FUNCTION DOC]
    # HELPER: Creates a vocabulary file for the id-table-with-hash-buckets tests.
    # Inputs: basename (file name in the test temp directory) and values (lines, defaults to brain,
    # salad, surgery). Writes one line per value with a trailing newline and returns the full path.
    # Side effect: creates or overwrites the file.
    vocabulary_file = os.path.join(self.get_temp_dir(), basename)
    with open(vocabulary_file, "w") as f:
      f.write("\n".join(values) + "\n")
    return vocabulary_file

  def testStringIdTableWithHashBuckets(self):
    # [FUNCTION DOC]
    # Scenario: string keys with a static table inner table (default minus one) and one bucket. After
    # initialization, lookup of three words and an unknown word gives zero, one, two and three; table
    # size equals vocab size plus buckets.
    vocab_file = self._createVocabFile("feat_to_id_1.txt")
    default_value = -1
    vocab_size = 3
    oov_buckets = 1
    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size), default_value),
        oov_buckets)

    self.evaluate(table.initializer)

    input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

    out = table.lookup(input_string)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testInt32IdTableWithHashBuckets(self):
    # [FUNCTION DOC]
    # Scenario: int32 lookup keys with an int64 vocabulary file and one bucket. Lookup of forty-two, one,
    # minus one thousand and eleven gives zero, one, two, three; table size is four.
    vocab_file = self._createVocabFile("feat_to_id_2.txt", ("42", "1", "-1000"))
    default_value = -1
    vocab_size = 3
    oov_buckets = 1
    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
            default_value),
        oov_buckets,
        key_dtype=dtypes.int32)

    self.evaluate(table.initializer)

    values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int32)

    out = table.lookup(values)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testInt64IdTableWithHashBuckets(self):
    # [FUNCTION DOC]
    # Scenario: int64 keys with an int64 vocabulary file and one bucket. Lookup gives zero, one, two,
    # three; table size is four.
    vocab_file = self._createVocabFile("feat_to_id_3.txt", ("42", "1", "-1000"))
    default_value = -1
    vocab_size = 3
    oov_buckets = 1
    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size, key_dtype=dtypes.int64),
            default_value), oov_buckets)

    self.evaluate(table.initializer)

    values = constant_op.constant((42, 1, -1000, 11), dtype=dtypes.int64)

    out = table.lookup(values)
    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table.size()))

  def testStringIdTableWithOnlyHashBucket(self):
    # [FUNCTION DOC]
    # Scenario: no inner table and five buckets for string keys. Each word maps to its fingerprint
    # modulo five (three, one, four) and the table size is five.
    oov_buckets = 5

    # Set a table that only uses hash buckets, for each input value returns
    # an id calculated by fingerprint("input") mod oov_buckets.
    table = lookup_ops.IdTableWithHashBuckets(None, oov_buckets)
    self.evaluate(table.initializer)

    values = constant_op.constant(("brain", "salad", "surgery"))

    out = table.lookup(values)
    self.assertAllEqual(
        [
            3,  # fingerprint("brain") mod 5.
            1,  # fingerprint("salad") mod 5.
            4  # fingerprint("surgery") mod 5
        ],
        self.evaluate(out))
    self.assertEqual(oov_buckets, self.evaluate(table.size()))

  def testInt32IdTableWithOnlyHashBucket(self):
    # [FUNCTION DOC]
    # Scenario: no inner table, five buckets and int32 keys. Numbers forty-two, one and minus one
    # thousand map to fingerprint modulo five (one, four, two); table size is five.
    oov_buckets = 5

    # Set a table that only uses hash buckets, for each input value returns
    # an id calculated by fingerprint("input") mod oov_buckets.
    table = lookup_ops.IdTableWithHashBuckets(
        None, oov_buckets, key_dtype=dtypes.int32)
    self.evaluate(table.initializer)

    input_string = constant_op.constant([42, 1, -1000], dtype=dtypes.int32)

    out = table.lookup(input_string)
    self.assertAllEqual(
        [
            1,  # fingerprint("42") mod 5.
            4,  # fingerprint("1") mod 5.
            2  # fingerprint("-1000") mod 5
        ],
        self.evaluate(out))
    self.assertEqual(oov_buckets, self.evaluate(table.size()))

  def testFloat64IdTableWithOnlyHashBucket(self):
    # [FUNCTION DOC]
    # Scenario: float64 as key dtype with only hash buckets must raise a type error mentioning an invalid
    # key dtype.
    with self.assertRaisesRegex(TypeError, "Invalid `key_dtype`"):
      lookup_ops.IdTableWithHashBuckets(
          None, num_oov_buckets=5, key_dtype=dtypes.float64)

  def testBoolIdTableWithOnlyHashBucket(self):
    # [FUNCTION DOC]
    # Scenario: bool as key dtype with only hash buckets must raise a type error mentioning an invalid
    # key dtype.
    with self.assertRaisesRegex(TypeError, "Invalid `key_dtype`"):
      lookup_ops.IdTableWithHashBuckets(
          None, num_oov_buckets=5, key_dtype=dtypes.bool)

  def testIdTableWithHashBucketsWithMultipleInitializers(self):
    # [FUNCTION DOC]
    # Scenario: two id tables share one inner static table but use a fast hasher and a strong hasher with a
    # key pair, each with three buckets and its own name. After initializing all tables, expects the
    # same ids for words in the vocabulary and, for unknown words, the fast hasher gives five and five
    # while the strong one gives five and three; both sizes are six. In graph mode also asserts the
    # graph holds the fast and strong string-to-bucket ops under the table names.
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    default_value = -1
    vocab_size = 3
    oov_buckets = 3

    vocab_table = lookup_ops.StaticHashTable(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size), default_value)
    table1 = lookup_ops.IdTableWithHashBuckets(
        vocab_table,
        oov_buckets,
        hasher_spec=lookup_ops.FastHashSpec,
        name="table1")

    table2 = lookup_ops.IdTableWithHashBuckets(
        vocab_table,
        oov_buckets,
        hasher_spec=lookup_ops.StrongHashSpec((1, 2)),
        name="table2")

    self.evaluate(lookup_ops.tables_initializer())

    input_string = constant_op.constant(
        ["fruit", "brain", "salad", "surgery", "UNK"])

    out1 = table1.lookup(input_string)
    out2 = table2.lookup(input_string)

    out1, out2 = self.evaluate([out1, out2])
    self.assertAllEqual([5, 0, 1, 2, 5], out1)
    self.assertAllEqual([5, 0, 1, 2, 3], out2)
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table1.size()))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table2.size()))
    if not context.executing_eagerly():
      test_util.assert_ops_in_graph({
          "table1_Lookup/hash_bucket": "StringToHashBucketFast",
          "table2_Lookup/hash_bucket": "StringToHashBucketStrong",
      }, ops.get_default_graph())

  def testIdTableWithHashBucketsInitializationAcrossSessions(self):
    # [FUNCTION DOC]
    # Scenario: initialization across sessions. The first table is initialized and checked (ids zero to
    # three, size four). A second table over the same file is created without initializing and still
    # gives three for unknown words and one for salad, with size four. Within eager execution both
    # work against the shared state.
    vocab_file = self._createVocabFile("feat_to_id_5.txt")
    default_value = -1
    vocab_size = 3
    oov_buckets = 1
    table1 = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size), default_value), oov_buckets)

    self.evaluate(table1.initializer)

    input_string_1 = constant_op.constant(["brain", "salad", "surgery", "UNK"])

    out1 = table1.lookup(input_string_1)

    self.assertAllEqual([0, 1, 2, 3], self.evaluate(out1))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table1.size()))

    default_value = -1
    vocab_size = 3
    oov_buckets = 1

    # Underlying lookup table already initialized in previous session.
    # No need to call self.evaluate(table2.initializer)
    table2 = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size), default_value), oov_buckets)

    input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

    out2 = table2.lookup(input_string_2)

    self.assertAllEqual([3, 1, 3], self.evaluate(out2))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table2.size()))

  def testIdTableWithHashBucketsWithMultipleInitializersDifferentDefault(self):
    # [FUNCTION DOC]
    # Scenario: two id tables over the same vocabulary with different inner default values (minus one and
    # minus two) and zero buckets. Checks the first table returns zero, one, two, minus one and the
    # second returns minus two, one, minus two; both sizes equal the vocab size.
    vocab_file = self._createVocabFile("feat_to_id_6.txt")
    default_value1 = -1
    vocab_size = 3
    oov_buckets = 0
    table1 = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size), default_value1),
        oov_buckets)

    default_value2 = -2
    table2 = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(
                vocab_file, vocab_size=vocab_size), default_value2),
        oov_buckets)

    self.evaluate(lookup_ops.tables_initializer())

    input_string_1 = constant_op.constant(
        ["brain", "salad", "surgery", "UNK"])
    input_string_2 = constant_op.constant(["fruit", "salad", "UNK"])

    out1 = table1.lookup(input_string_1)
    out2 = table2.lookup(input_string_2)

    out1, out2 = self.evaluate([out1, out2])
    self.assertAllEqual([0, 1, 2, -1], out1)
    self.assertAllEqual([-2, 1, -2], out2)
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table1.size()))
    self.assertEqual(vocab_size + oov_buckets, self.evaluate(table2.size()))

  def testSparseTensor(self):
    # [FUNCTION DOC]
    # Scenario: sparse tensor of string keys with a static inner table and one bucket. Output values
    # have static length five and equal zero, one, zero, two, three, with indices and dense shape
    # unchanged.
    vocab_file = self._createVocabFile("feat_to_id_7.txt")
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                             dtypes.string),
        constant_op.constant(input_shape, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(vocab_file, vocab_size=3),
            -1), 1)
    self.evaluate(table.initializer)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testRaggedTensor(self):
    # [FUNCTION DOC]
    # Scenario: ragged tensor of string keys with a static inner table and one bucket. Output flat
    # values have static length five and equal zero, one, zero, two, three, with row splits unchanged.
    vocab_file = self._createVocabFile("feat_to_id_7.txt")
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant(["brain", "salad", "brain", "surgery", "tarkus"],
                             dtypes.string),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.TextFileIdTableInitializer(vocab_file, vocab_size=3),
            -1), 1)
    self.evaluate(table.initializer)

    ragged_ids = table.lookup(ragged_features)
    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testInt32SparseTensor(self):
    # [FUNCTION DOC]
    # Scenario: sparse int32 keys with a key/value initializer and an int32 key dtype argument. Output
    # equals zero, one, zero, two, three with indices and dense shape unchanged.
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_shape, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int32)
    self.evaluate(table.initializer)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt32RaggedTensor(self):
    # [FUNCTION DOC]
    # Scenario: ragged int32 keys (with int32 row splits) with a key/value initializer. Output flat
    # values equal zero, one, zero, two, three and the row splits are unchanged.
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int32),
        constant_op.constant(input_row_splits, dtypes.int32))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int32)
    self.evaluate(table.initializer)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testInt64SparseTensor(self):
    # [FUNCTION DOC]
    # Scenario: sparse int64 keys with a key/value initializer and int64 key dtype. Output equals zero,
    # one, zero, two, three with indices and dense shape unchanged.
    input_indices = [[0, 0], [0, 1], [2, 0], [2, 2], [3, 0]]
    input_shape = [4, 4]
    sp_features = sparse_tensor.SparseTensor(
        constant_op.constant(input_indices, dtypes.int64),
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_shape, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int64)
    self.evaluate(table.initializer)

    sp_ids = table.lookup(sp_features)

    self.assertAllEqual([5], sp_ids.values._shape_as_list())

    sp_ids_ind, sp_ids_val, sp_ids_shape = self.evaluate(
        [sp_ids.indices, sp_ids.values, sp_ids.dense_shape])

    self.assertAllEqual(input_indices, sp_ids_ind)
    self.assertAllEqual([0, 1, 0, 2, 3], sp_ids_val)
    self.assertAllEqual(input_shape, sp_ids_shape)

  def testInt64RaggedTensor(self):
    # [FUNCTION DOC]
    # Scenario: ragged int64 keys with a key/value initializer. Output flat values equal zero, one, zero,
    # two, three and the row splits are unchanged.
    input_row_splits = [0, 2, 4, 5]
    ragged_features = ragged_tensor.RaggedTensor.from_row_splits(
        constant_op.constant([42, 1, 42, -1000, 11], dtypes.int64),
        constant_op.constant(input_row_splits, dtypes.int64))

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_ops.StaticHashTable(
            lookup_ops.KeyValueTensorInitializer(
                (42, 1, -1000), (0, 1, 2), dtypes.int64, dtypes.int64), -1),
        1,
        key_dtype=dtypes.int64)
    self.evaluate(table.initializer)

    ragged_ids = table.lookup(ragged_features)

    self.assertAllEqual([5], ragged_ids.values._shape_as_list())

    ragged_ids_val, ragged_ids_row_splits = self.evaluate(
        [ragged_ids.values, ragged_ids.row_splits])

    self.assertAllEqual([0, 1, 0, 2, 3], ragged_ids_val)
    self.assertAllEqual(input_row_splits, ragged_ids_row_splits)

  def testIdTableWithHashBucketsWithInvalidHashers(self):
    # [FUNCTION DOC]
    # Scenario: invalid hasher specs. A number raises a type error; an unknown-named hasher builds but a
    # later lookup raises a value error; a strong hasher with an empty key list or a three-element key
    # list raises a value error; a strong hasher whose key list contains nothing raises a type error.
    vocab_file = self._createVocabFile("feat_to_id_4.txt")
    default_value = -1
    vocab_size = 3
    oov_buckets = 1
    lookup_table = lookup_ops.StaticHashTable(
        lookup_ops.TextFileIdTableInitializer(
            vocab_file, vocab_size=vocab_size), default_value)

    with self.assertRaises(TypeError):
      lookup_ops.IdTableWithHashBuckets(
          lookup_table, oov_buckets, hasher_spec=1)

    table = lookup_ops.IdTableWithHashBuckets(
        lookup_table,
        oov_buckets,
        hasher_spec=lookup_ops.HasherSpec("my-awesome-hash", None))

    input_string = constant_op.constant(["brain", "salad", "surgery", "UNK"])

    with self.assertRaises(ValueError):
      table.lookup(input_string)

    with self.assertRaises(ValueError):
      table = lookup_ops.IdTableWithHashBuckets(
          lookup_table, oov_buckets, hasher_spec=lookup_ops.StrongHashSpec([]))

    with self.assertRaises(ValueError):
      table = lookup_ops.IdTableWithHashBuckets(
          lookup_table,
          oov_buckets,
          hasher_spec=lookup_ops.StrongHashSpec([1, 2, 3]))

    with self.assertRaises(TypeError):
      table = lookup_ops.IdTableWithHashBuckets(
          lookup_table,
          oov_buckets,
          hasher_spec=lookup_ops.StrongHashSpec([None, 2]))

  def testIdTableWithHashBucketsNoInnerTable(self):
    # [FUNCTION DOC]
    # Scenario: id table with no inner table and one bucket. Its resource handle must be nothing.
    table = lookup_ops.IdTableWithHashBuckets(None, num_oov_buckets=1)
    self.assertIsNone(table.resource_handle)


@parameterized.named_parameters(
    (f"_{is_anonymous}", is_anonymous) for is_anonymous in [False, True])
class MutableHashTableOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the mutable hash table, parameterized by the anonymous flag. Covers insert, remove,
  # lookup and export, imported graphs (an expected failure), saver and checkpoint round trips, saved
  # models, sharing between sessions, vector values, high-rank inputs, dynamic default values,
  # signature mismatches, several key and value type combinations, and export shape inference.

  def testMutableHashTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: string-to-int64 mutable table with default minus one. Inserts four entries, removes one
    # existing and one missing key, checks sizes four then three, looks up three keys giving zero, one,
    # minus one (static shape three), and exports the table (checked after sorting because export order
    # is undefined). Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery", "tarkus"])
    values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant(["tarkus", "tank"])
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)
    self.assertAllEqual([3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

    exported_keys, exported_values = table.export()

    # exported data is in the order of the internal map, i.e. undefined
    sorted_keys = np.sort(self.evaluate(exported_keys))
    sorted_values = np.sort(self.evaluate(exported_values))
    self.assertAllEqual([b"brain", b"salad", b"surgery"], sorted_keys)
    self.assertAllEqual([0, 1, 2], sorted_values)

  # TODO(https://github.com/tensorflow/tensorflow/issues/24439): remove exepectedFailure when fixed
  @unittest.expectedFailure
  @test_util.run_v2_only
  def testImportedHashTable(self, is_anonymous):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is marked as an expected failure because of a tracked upstream bug, so it
    # is not expected to pass; the assertion that the wrapped function returns the looked-up values
    # does not currently hold.
    # Scenario (TF2 only): builds a mutable table in a separate graph, inserts entries, exports the meta
    # graph, imports it in a wrapped function, and expects the lookup to return zero, one and minus one.
    g = ops.Graph()
    with g.as_default():
      default_val = -1
      keys = constant_op.constant(["brain", "salad", "surgery", "tarkus"])
      values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          experimental_is_anonymous=is_anonymous)
      self.evaluate(table.insert(keys, values))
      op = table.lookup(constant_op.constant(["brain", "salad", "tank"]))
      meta_graph = saver.export_meta_graph()

    def f():
      # [FUNCTION DOC]
      # HELPER: Inner function for the imported-table test. Takes no inputs; imports the exported meta
      # graph into the wrapped function's graph and returns the lookup tensor found by name.
      saver.import_meta_graph(meta_graph)
      return ops.get_default_graph().get_tensor_by_name(op.name)

    wrapped = wrap_function.wrap_function(f, [])
    self.assertAllEqual([0, 1, -1], wrapped())

  @test_util.run_v1_only("SaverV1")
  def testSaveRestore(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only, saver): saves a mutable table named t1 with three entries together with two
    # variables, then in a fresh graph builds a table with other content and variables of minus one,
    # restores, and verifies the variables are ten and twenty, the table size is three, and lookups give
    # minus one, zero, one, two, minus one. Side effect: writes checkpoint files. Skipped for anonymous
    # tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = -1
      keys = constant_op.constant(["b", "c", "d"], dtypes.string)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          name="t1",
          checkpoint=True,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver()
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, str)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(-1.0, name="v0")
      v1 = variables.Variable(-1.0, name="v1")
      default_val = -1
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          name="t1",
          checkpoint=True,
          experimental_is_anonymous=is_anonymous)
      self.evaluate(
          table.insert(
              constant_op.constant(["a", "c"], dtypes.string),
              constant_op.constant([12, 24], dtypes.int64)))
      self.assertAllEqual(2, self.evaluate(table.size()))

      save = saver.Saver()

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                          dtypes.string)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_v1_only("SaverV1")
  def testSaveRestoreOnlyTable(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only): same as the saver round trip, but the saver is constructed with just the
    # table. Verifies the restored table has size three and the expected lookups. Side effect: writes
    # checkpoint files in the temp directory.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    with self.session(graph=ops.Graph()) as sess:
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(20.0, name="v1")

      default_val = -1
      keys = constant_op.constant(["b", "c", "d"], dtypes.string)
      values = constant_op.constant([0, 1, 2], dtypes.int64)
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          name="t1",
          checkpoint=True,
          experimental_is_anonymous=is_anonymous)

      save = saver.Saver([table])
      self.evaluate(variables.global_variables_initializer())

      # Check that the parameter nodes have been initialized.
      self.assertEqual(10.0, self.evaluate(v0))
      self.assertEqual(20.0, self.evaluate(v1))

      self.assertAllEqual(0, self.evaluate(table.size()))
      self.evaluate(table.insert(keys, values))
      self.assertAllEqual(3, self.evaluate(table.size()))

      val = save.save(sess, save_path)
      self.assertIsInstance(val, str)
      self.assertEqual(save_path, val)

    with self.session(graph=ops.Graph()) as sess:
      default_val = -1
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          name="t1",
          checkpoint=True,
          experimental_is_anonymous=is_anonymous)
      self.evaluate(
          table.insert(
              constant_op.constant(["a", "c"], dtypes.string),
              constant_op.constant([12, 24], dtypes.int64)))
      self.assertAllEqual(2, self.evaluate(table.size()))

      save = saver.Saver([table])

      # Restore the saved values in the parameter nodes.
      save.restore(sess, save_path)
      # Check that the parameter nodes have been restored.

      self.assertAllEqual(3, self.evaluate(table.size()))

      input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                          dtypes.string)
      output = table.lookup(input_string)
      self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_in_graph_and_eager_modes
  def testObjectSaveRestore(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (graph and eager): object-based checkpoint round trip of a mutable table with two
    # variables. Saves, deletes everything, rebuilds with different content, restores and verifies the
    # variables are ten and twenty, the table size is three and lookups return the saved entries.
    # Side effect: writes checkpoint files. Skipped for anonymous tables in graph mode.
    if is_anonymous and not context.executing_eagerly():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_prefix = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(20.0, name="v1")

    default_val = -1
    keys = constant_op.constant(["b", "c", "d"], dtypes.string)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        name="t1",
        checkpoint=True,
        experimental_is_anonymous=is_anonymous)

    checkpoint = trackable.Checkpoint(table=table, v0=v0, v1=v1)
    self.evaluate([v0.initializer, v1.initializer])

    # Check that the parameter nodes have been initialized.
    self.assertEqual(10.0, self.evaluate(v0))
    self.assertEqual(20.0, self.evaluate(v1))

    self.assertAllEqual(0, self.evaluate(table.size()))
    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    save_path = checkpoint.save(save_prefix)
    del table, checkpoint, v0, v1

    v0 = variables.Variable(-1.0, name="v0")
    v1 = variables.Variable(-1.0, name="v1")
    default_val = -1
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        name="t1",
        checkpoint=True,
        experimental_is_anonymous=is_anonymous)
    self.evaluate(
        table.insert(
            constant_op.constant(["a", "c"], dtypes.string),
            constant_op.constant([12, 24], dtypes.int64)))
    self.assertAllEqual(2, self.evaluate(table.size()))

    checkpoint = trackable.Checkpoint(table=table, v0=v0, v1=v1)

    # Restore the saved values in the parameter nodes.
    checkpoint.restore(save_path).run_restore_ops()
    # Check that the parameter nodes have been restored.
    self.assertEqual(10.0, self.evaluate(v0))
    self.assertEqual(20.0, self.evaluate(v1))

    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["a", "b", "c", "d", "e"],
                                        dtypes.string)
    output = table.lookup(input_string)
    self.assertAllEqual([-1, 0, 1, 2, -1], self.evaluate(output))

  @test_util.run_v2_only
  def testSavedModelSaveRestore(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF2 only): saved-model round trip of a mutable int64 table. Inserts three entries, checks
    # size, lookups, export length three and reference-counting flag, saves a root with traced
    # functions, reloads, and checks size three, lookup results and flag again. Side effect: writes
    # files under the test temp directory.
    save_dir = os.path.join(self.get_temp_dir(), "save_restore")
    save_path = os.path.join(tempfile.mkdtemp(prefix=save_dir), "hash")

    root = autotrackable.AutoTrackable()

    default_value = -1
    keys = constant_op.constant([11, 12, 13], dtypes.int64)
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    root.table = lookup_ops.MutableHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value,
        experimental_is_anonymous=is_anonymous)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.int64)])
    def lookup(key):
      # [FUNCTION DOC]
      # HELPER: Traced function with a scalar int64 input signature returning the root table lookup of the
      # key. Used by the saved-model test.
      return root.table.lookup(key)

    @def_function.function(input_signature=[])
    def size():
      # [FUNCTION DOC]
      # HELPER: Traced function with empty signature returning the root table size. Used by the
      # saved-model test.
      return root.table.size()

    @def_function.function(input_signature=[])
    def is_ref_counting():
      # [FUNCTION DOC]
      # HELPER: Traced function with empty signature returning whether the root table handle is
      # reference counted. Used by the saved-model test.
      return test_ops.is_resource_handle_ref_counting(
          root.table.resource_handle)

    root.lookup = lookup
    root.size = size
    root.is_ref_counting = is_ref_counting

    self.assertEqual(root.table.size(), 0)
    root.table.insert(keys, values)
    self.assertEqual(root.table.size(), 3)
    self.assertEqual(root.table.lookup(12), 1)
    self.assertEqual(root.table.lookup(10), -1)
    self.assertEqual(len(root.table.export()[0]), 3)
    self.assertEqual(root.is_ref_counting(), is_anonymous)

    saved_model_save.save(root, save_path)

    del root
    loaded = saved_model_load.load(save_path)
    self.assertEqual(loaded.size(), 3)
    self.assertEqual(loaded.lookup(12), 1)
    self.assertEqual(loaded.lookup(10), -1)
    self.assertEqual(loaded.is_ref_counting(), is_anonymous)

  @test_util.run_v1_only("Multiple sessions")
  def testSharing(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario (TF1 only): start a gRPC server and two sessions. The first session populates a named
    # table (size zero, then two) and looks up keys giving a, b and the default dash. The second session
    # sees the same size and data. Skipped for anonymous tables. Side effect: starts an in-process server.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    # Start a server to store the table state
    server = server_lib.Server({"local0": ["localhost:0"]},
                               protocol="grpc",
                               start=True)
    # Create two sessions sharing the same state
    session1 = session.Session(server.target)
    session2 = session.Session(server.target)

    table = lookup_ops.MutableHashTable(
        dtypes.int64,
        dtypes.string,
        "-",
        name="t1",
        experimental_is_anonymous=is_anonymous)

    # Populate the table in the first session
    with session1:
      self.assertAllEqual(0, table.size())

      keys = constant_op.constant([11, 12], dtypes.int64)
      values = constant_op.constant(["a", "b"])
      table.insert(keys, values).run()
      self.assertAllEqual(2, table.size())

      output = table.lookup(constant_op.constant([11, 12, 13], dtypes.int64))
      self.assertAllEqual([b"a", b"b", b"-"], output)

    # Verify that we can access the shared data from the second session
    with session2:
      self.assertAllEqual(2, table.size())

      output = table.lookup(constant_op.constant([10, 11, 12], dtypes.int64))
      self.assertAllEqual([b"-", b"a", b"b"], output)

  def testMutableHashTableOfTensors(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: mutable table with vector values of length two and a vector default. Inserts four entries,
    # removes one existing and one missing key, checks sizes four and three, lookup shape three by two
    # with default row for the unknown key, and exports (sorted before comparing). Skipped for anonymous
    # tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant([-1, -1], dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery", "tarkus"])
    values = constant_op.constant([[0, 1], [2, 3], [4, 5], [6, 7]],
                                  dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant(["tarkus", "tank"])
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)
    self.assertAllEqual([3, 2], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([[0, 1], [2, 3], [-1, -1]], result)

    exported_keys, exported_values = table.export()
    # exported data is in the order of the internal map, i.e. undefined
    sorted_keys = np.sort(self.evaluate(exported_keys))
    sorted_values = np.sort(self.evaluate(exported_values), axis=0)
    self.assertAllEqual([b"brain", b"salad", b"surgery"], sorted_keys)
    sorted_expected_values = np.sort([[4, 5], [2, 3], [0, 1]], axis=0)
    self.assertAllEqual(sorted_expected_values, sorted_values)

  def testMutableHashTableExportInsert(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: export from one table and insert into a second. Table one holds three vector-valued entries;
    # exports have three keys and six value elements; a second table is filled from the exported data and
    # gives identical lookup results. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant([-1, -1], dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
    table1 = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table1.size()))
    self.evaluate(table1.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table1.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    expected_output = [[0, 1], [2, 3], [-1, -1]]
    output1 = table1.lookup(input_string)
    self.assertAllEqual(expected_output, self.evaluate(output1))

    exported_keys, exported_values = table1.export()
    self.assertAllEqual(3, self.evaluate(exported_keys).size)
    self.assertAllEqual(6, self.evaluate(exported_values).size)

    # Populate a second table from the exported data
    table2 = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table2.size()))
    self.evaluate(table2.insert(exported_keys, exported_values))
    self.assertAllEqual(3, self.evaluate(table2.size()))

    # Verify lookup result is still the same
    output2 = table2.lookup(input_string)
    self.assertAllEqual(expected_output, self.evaluate(output2))

  def testMutableHashTableOfTensorsInvalidShape(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: inserting values whose shapes do not match the vector value shape (flat, transposed,
    # too few rows, too few columns) must each raise an op error saying a different shape was expected;
    # a correctly shaped insert then succeeds and gives size three. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant([-1, -1], dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    # Shape [6] instead of [3, 2]
    values = constant_op.constant([0, 1, 2, 3, 4, 5], dtypes.int64)
    with self.assertRaisesOpError("Expected shape"):
      self.evaluate(table.insert(keys, values))

    # Shape [2,3] instead of [3, 2]
    values = constant_op.constant([[0, 1, 2], [3, 4, 5]], dtypes.int64)
    with self.assertRaisesOpError("Expected shape"):
      self.evaluate(table.insert(keys, values))

    # Shape [2, 2] instead of [3, 2]
    values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
    with self.assertRaisesOpError("Expected shape"):
      self.evaluate(table.insert(keys, values))

    # Shape [3, 1] instead of [3, 2]
    values = constant_op.constant([[0], [2], [4]], dtypes.int64)
    with self.assertRaisesOpError("Expected shape"):
      self.evaluate(table.insert(keys, values))

    # Valid Insert
    values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

  def testMutableHashTableInvalidDefaultValue(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: creating a table whose default value is a matrix rather than a vector must raise an op
    # error saying the default value must be a vector.
    default_val = constant_op.constant([[-1, -1]], dtypes.int64)
    with self.assertRaisesOpError("Default value must be a vector"):
      table = lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          default_val,
          experimental_is_anonymous=is_anonymous)
      self.assertAllEqual(0, self.evaluate(table.size()))

  def testMutableHashTableDuplicateInsert(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: inserting a batch with a duplicate key (brain twice). The table ends with three entries and
    # the last value (three) wins for the duplicated key. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery", "brain"])
    values = constant_op.constant([0, 1, 2, 3], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([3, 1, -1], result)

  def testMutableHashTableFindHighRank(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: look up a two-by-two matrix of keys in a scalar-valued table. The result keeps shape two
    # by two with default for unknown keys. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"],
                                         ["tank", "tarkus"]])
    output = table.lookup(input_string)
    self.assertAllEqual([2, 2], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([[0, 1], [-1, -1]], result)

  def testMutableHashTableFindWithInvalidShapeDefaultValue(self, is_anonymous):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected-message patterns are plain strings containing backslash-escaped brackets,
    # which are invalid string escapes in Python (they only work by accident and may warn); a raw string
    # would be needed.
    # Scenario: passing a dynamic default value of an invalid shape (four by two, or one by two by two)
    # to lookup must raise a value or invalid-argument error saying the expected shape is two or two,
    # two, two.
    default_val = [-1, -1]
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    input_string = constant_op.constant([["brain", "salad"], ["tank",
                                                              "tarkus"]])

    invalid_default_val = constant_op.constant(
        [[-2, -3], [-4, -5], [-6, -7], [-8, -9]], dtypes.int64)

    with self.assertRaisesRegex(
        (ValueError, errors_impl.InvalidArgumentError),
        "Expected shape \[2\] or \[2,2,2\] for default value, got \[4,2]"):
      self.evaluate(table.lookup(input_string, invalid_default_val))

    invalid_default_val = constant_op.constant([[[-2, -3], [-4, -5]]],
                                               dtypes.int64)
    with self.assertRaisesRegex(
        (ValueError, errors_impl.InvalidArgumentError),
        "Expected shape \[2\] or \[2,2,2\] for default value, got \[1,2,2\]"):
      self.evaluate(table.lookup(input_string, invalid_default_val))

  def testMutableHashTableFindHighRankScalarWithDynamicDefaultValue(
      self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: scalar-valued table looked up with a high-rank key matrix and a dynamic default value
    # matrix. Known keys give their values and unknown keys use the matching element of the dynamic
    # default (minus four, minus five); the output shape is two by two. Skipped for anonymous tables in
    # TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"], ["tank",
                                                              "tarkus"]])

    dynamic_default_val = constant_op.constant([[-2, -3], [-4, -5]],
                                               dtypes.int64)
    output = table.lookup(input_string, dynamic_default_val)
    self.assertAllEqual([2, 2], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([[0, 1], [-4, -5]], result)

  def testMutableHashTableFindHighRankVectorWithDynamicDefaultValue(
      self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: vector-valued table looked up with a high-rank key matrix and a dynamic default tensor of
    # shape two by two by two. Known keys give their vectors and unknown keys use the matching default
    # vectors; output shape is two by two by two. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = [-1, -1]
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([[0, 1], [2, 3], [4, 5]], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"], ["tank",
                                                              "tarkus"]])

    dynamic_default_val = constant_op.constant(
        [[[-2, -3], [-4, -5]], [[-6, -7], [-8, -9]]], dtypes.int64)
    output = table.lookup(input_string, dynamic_default_val)
    self.assertAllEqual([2, 2, 2], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual([[[0, 1], [2, 3]], [[-6, -7], [-8, -9]]], result)

  def testMutableHashTableInsertHighRank(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: inserting with a two-by-two matrix of keys and a matrix of values flattens the insert.
    # Size becomes four and lookups of brain, salad, tank give zero, one, three and an unknown key gives
    # the default. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant([["brain", "salad"], ["surgery", "tank"]])
    values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank", "tarkus"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, 3, -1], result)

  def testMutableHashTableRemoveHighRank(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: removing with a matrix of keys after a matrix insert. Removing salad and a missing key
    # leaves size three and lookups of brain, salad, tank, tarkus give zero, default, three, default.
    # Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant([["brain", "salad"], ["surgery", "tank"]])
    values = constant_op.constant([[0, 1], [2, 3]], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(4, self.evaluate(table.size()))

    remove_string = constant_op.constant(["salad", "tarkus"])
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank", "tarkus"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, -1, 3, -1], result)

  def testMutableHashTableOfTensorsFindHighRank(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: vector-valued table (length three) looked up with a two-by-two key matrix. Output shape
    # two by two by three with the default vector for unknown keys. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant([-1, -1, -1], dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                  dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"],
                                         ["tank", "tarkus"]])
    output = table.lookup(input_string)
    self.assertAllEqual([2, 2, 3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual(
        [[[0, 1, 2], [2, 3, 4]], [[-1, -1, -1], [-1, -1, -1]]], result)

  def testMutableHashTableOfTensorsRemoveHighRank(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: vector-valued table (length three) with remove using a one-by-two key matrix. Removes
    # brain and an unknown key; size becomes two and lookups return default vectors for removed or
    # missing keys. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant([-1, -1, -1], dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([[0, 1, 2], [2, 3, 4], [4, 5, 6]],
                                  dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    remove_string = constant_op.constant([["brain", "tank"]])
    self.evaluate(table.remove(remove_string))
    self.assertAllEqual(2, self.evaluate(table.size()))

    input_string = constant_op.constant([["brain", "salad"],
                                         ["surgery", "tank"]])
    output = table.lookup(input_string)
    self.assertAllEqual([2, 2, 3], output.get_shape())

    result = self.evaluate(output)
    self.assertAllEqual(
        [[[-1, -1, -1], [2, 3, 4]], [[4, 5, 6], [-1, -1, -1]]], result)

  def testMultipleMutableHashTables(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: three independent mutable tables filled with the same data each have size three and give
    # the same lookup output. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)

    table1 = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    table2 = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    table3 = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.evaluate(table1.insert(keys, values))
    self.evaluate(table2.insert(keys, values))
    self.evaluate(table3.insert(keys, values))

    self.assertAllEqual(3, self.evaluate(table1.size()))
    self.assertAllEqual(3, self.evaluate(table2.size()))
    self.assertAllEqual(3, self.evaluate(table3.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output1 = table1.lookup(input_string)
    output2 = table2.lookup(input_string)
    output3 = table3.lookup(input_string)

    out1, out2, out3 = self.evaluate([output1, output2, output3])
    self.assertAllEqual([0, 1, -1], out1)
    self.assertAllEqual([0, 1, -1], out2)
    self.assertAllEqual([0, 1, -1], out3)

  def testMutableHashTableWithTensorDefault(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: the default value is a scalar int64 tensor. Insert three entries and look up two known
    # words and an unknown; expects zero, one and the default. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = constant_op.constant(-1, dtypes.int64)
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual([0, 1, -1], result)

  def testSignatureMismatch(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: type mismatches. Inserting int keys or string values into a string-to-int64 table raises a
    # value error; size stays zero. Variable-based (reference-like) inputs are accepted for insert and
    # lookup. Looking up int64 keys raises a value error. Creating the table with a string default value
    # for int64 values raises a type error. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1, 2], dtypes.int64)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.int64,
        default_val,
        experimental_is_anonymous=is_anonymous)

    # insert with keys of the wrong type
    with self.assertRaises(ValueError):
      self.evaluate(table.insert(constant_op.constant([4, 5, 6]), values))

    # insert with values of the wrong type
    with self.assertRaises(ValueError):
      self.evaluate(table.insert(keys, constant_op.constant(["a", "b", "c"])))

    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string_ref = variables.Variable("brain")
    input_int64_ref = variables.Variable(-1, dtype=dtypes.int64)
    self.evaluate(variables.global_variables_initializer())

    # Ref types do not produce an insert signature mismatch.
    self.evaluate(table.insert(input_string_ref, input_int64_ref))
    self.assertAllEqual(3, self.evaluate(table.size()))

    # Ref types do not produce a lookup signature mismatch.
    self.assertEqual(-1, self.evaluate(table.lookup(input_string_ref)))

    # lookup with keys of the wrong type
    input_string = constant_op.constant([1, 2, 3], dtypes.int64)
    with self.assertRaises(ValueError):
      self.evaluate(table.lookup(input_string))

    # default value of the wrong type
    with self.assertRaises(TypeError):
      lookup_ops.MutableHashTable(
          dtypes.string,
          dtypes.int64,
          "UNK",
          experimental_is_anonymous=is_anonymous)

  def testMutableHashTableStringFloat(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: string keys to float32 values with default minus one point five. Looks up two known
    # words and one unknown word and compares with tolerance. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1.5
    keys = constant_op.constant(["brain", "salad", "surgery"])
    values = constant_op.constant([0, 1.1, 2.2], dtypes.float32)
    table = lookup_ops.MutableHashTable(
        dtypes.string,
        dtypes.float32,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant(["brain", "salad", "tank"])
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllClose([0, 1.1, default_val], result)

  def testMutableHashTableIntFloat(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: int64 keys to float32 values with default minus one. Looks up three keys, two known and
    # one unknown, and compares with tolerance. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = -1.0
    keys = constant_op.constant([3, 7, 0], dtypes.int64)
    values = constant_op.constant([7.5, -1.2, 9.9], dtypes.float32)
    table = lookup_ops.MutableHashTable(
        dtypes.int64,
        dtypes.float32,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([7, 0, 11], dtypes.int64)
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllClose([-1.2, 9.9, default_val], result)

  def testMutableHashTableInt64String(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: int64 keys to string values with default "n/a". Looks up two known keys and one unknown
    # and expects the two words and the default. Skipped for anonymous tables in TF1.
    if is_anonymous and not tf2.enabled():
      self.skipTest(SKIP_ANONYMOUS_IN_TF1_REASON)
    default_val = "n/a"
    keys = constant_op.constant([0, 1, 2], dtypes.int64)
    values = constant_op.constant(["brain", "salad", "surgery"])
    table = lookup_ops.MutableHashTable(
        dtypes.int64,
        dtypes.string,
        default_val,
        experimental_is_anonymous=is_anonymous)
    self.assertAllEqual(0, self.evaluate(table.size()))

    self.evaluate(table.insert(keys, values))
    self.assertAllEqual(3, self.evaluate(table.size()))

    input_string = constant_op.constant([0, 1, 3], dtypes.int64)
    output = table.lookup(input_string)

    result = self.evaluate(output)
    self.assertAllEqual((b"brain", b"salad", b"n/a"), result)

  def testExportShapeInference(self, is_anonymous):
    # [FUNCTION DOC]
    # Scenario: shape inference for table export on a mutable table. Compares shapes of exported keys and
    # values directly and as inferred inside a traced function; both lists have two entries and each
    # inferred shape must be compatible with the actual one.
    default_value = -1
    table = lookup_ops.MutableHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        experimental_is_anonymous=is_anonymous)
    actual_shapes = [t.shape for t in table.export()]
    inferred_shapes = []

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: Traced function for the export shape test. Takes no inputs; exports the table and appends
      # each exported tensor's shape to a list in the enclosing test (side effect). Returns nothing.
      for t in table.export():
        inferred_shapes.append(t.shape)

    f()
    self.assertLen(actual_shapes, 2)
    self.assertLen(inferred_shapes, 2)
    self.assertTrue(inferred_shapes[0].is_compatible_with(actual_shapes[0]))
    self.assertTrue(inferred_shapes[1].is_compatible_with(actual_shapes[1]))


class MutableHashTableBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmarks for inserting into a mutable hash table in graph mode: repeated insert of a scalar key
  # and value, a stream of fresh keys, and batches of thirty-two keys. Each benchmark builds the table
  # through an overridable factory method, runs the insert operation many times inside a session and
  # asserts the final table size. Subclassed by the dense hash table benchmark.

  def _create_table(self):
    # [FUNCTION DOC]
    # HELPER: Factory for the table under benchmark. Returns a new mutable hash table with int64 keys,
    # float32 values and default value zero. Overridden in the dense table benchmark subclass.
    return lookup_ops.MutableHashTable(dtypes.int64, dtypes.float32, 0.0)

  def benchmark_single_repeated_scalar_insert_scalar(self):
    # [FUNCTION DOC]
    # Benchmark: repeatedly inserts the same scalar key (zero) with a variable value. Creates the table
    # and an insert op, runs the benchmark helper for ten burn-in and ten thousand measured iterations
    # in a session, and asserts the table holds exactly one entry. Side effect: reports benchmark results.
    table = self._create_table()
    value = variables.Variable(1.0)
    insert = table.insert(0, value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) == 1

  def benchmark_many_repeated_scalar_insert_scalar(self):
    # [FUNCTION DOC]
    # Benchmark: inserts scalar keys from an incrementing counter so every insert adds a new entry.
    # Runs ten burn-in and ten thousand measured iterations, and asserts the table holds at least ten
    # thousand entries. Side effect: reports benchmark results.
    table = self._create_table()
    c = dataset_ops.make_one_shot_iterator(counter.Counter()).get_next()
    value = variables.Variable(1.0)
    insert = table.insert(c, value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=10000)
      assert sess.run(size) >= 10000

  def benchmark_single_repeated_batch_32_insert_scalar(self):
    # [FUNCTION DOC]
    # Benchmark: repeatedly inserts the same batch of thirty-two keys with a vector of values. Runs ten
    # burn-in and one thousand measured iterations, and asserts the table holds exactly thirty-two
    # entries. Side effect: reports benchmark results.
    table = self._create_table()
    value = variables.Variable([1.0] * 32)
    insert = table.insert(list(range(32)), value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) == 32

  def benchmark_many_repeated_batch_32_insert_scalar(self):
    # [FUNCTION DOC]
    # Benchmark: inserts batches of thirty-two fresh keys derived from an incrementing counter. Runs ten
    # burn-in and one thousand measured iterations, and asserts the table holds at least thirty-two
    # thousand entries. Side effect: reports benchmark results.
    table = self._create_table()
    c = dataset_ops.make_one_shot_iterator(counter.Counter()).get_next()
    value = variables.Variable([1.0] * 32)
    insert = table.insert(32 * c + list(range(32)), value)
    size = table.size()
    with session.Session() as sess:
      sess.run(value.initializer)
      self.run_op_benchmark(sess, insert, burn_iters=10, min_iters=1000)
      assert sess.run(size) >= 1000 * 32


class DenseHashTableBenchmark(MutableHashTableBenchmark):
  # [CLASS DOC]
  # Benchmark class for the dense hash table. Inherits all benchmark methods of the mutable table
  # benchmark and only overrides the factory so the same insert workloads run against a dense table.

  def _create_table(self):
    # [FUNCTION DOC]
    # HELPER: Factory override that returns a dense hash table with int64 keys, float32 values, default
    # value zero, empty key minus one and deleted key minus two. Used by the inherited benchmarks.
    return lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.float32,
        default_value=0.0,
        empty_key=-1,
        deleted_key=-2)


if __name__ == "__main__":
  test.main()
