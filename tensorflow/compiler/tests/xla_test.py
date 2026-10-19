# [MODULE DOC]
# Base test fixture for XLA tests in the compiler test suite. Provides XLATestCase (a TensorFlow
# test case that knows which device and which data types to test, can disable tests through a
# manifest file, and supplies a custom session and device scope), a session subclass that wraps
# operations for TPU rewriting, a manifest parser, and a benchmark helper that runs a graph with or
# without the XLA JIT. Import-time side effect: defines command-line flags for the test device,
# the types to test, a disabled-tests manifest path, and a value for the XLA flags environment
# variable. Used as a base by most test files in this directory.
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
"""Definition of XLA test case."""

from collections.abc import Iterator
import contextlib
import os
import random
import re

import numpy as np

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.xla import jit
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'test_device',
    None,
    'Tensorflow device on which to place operators under test',
)
flags.DEFINE_string('types', None, 'Types to test. Comma-separated list.')
flags.DEFINE_string(
    'disabled_manifest',
    None,
    'Path to a file with a list of tests that should not run.',
)
flags.DEFINE_string(
    'tf_xla_flags',
    None,
    'Value to set the TF_XLA_FLAGS environment variable to',
)


def parse_disabled_manifest(manifest_content):
  # [FUNCTION DOC]
  # HELPER: used by XLATestCase to read the file listing tests that must not run.
  # Input: manifest_content is the whole text of a manifest. Each line may have a trailing
  # comment (from a hash sign to end of line) which is stripped; blank lines are ignored.
  # A line with one word is a test-name regular expression that disables the entire test. A
  # line with two space-separated words is a test name plus a comma-separated list of data type
  # names (protobuf enum names); those types are removed from that test only.
  # Returns a pair: a single regex text made by joining the disabled test names with an
  # alternation separator (empty text if none), and a mapping from test name to the set of
  # numpy data types to exclude for it.
  # Raises a value error for a line with more than two words.
  # KNOWN ISSUE: splitting on a single space means lines with repeated spaces between words
  # yield empty words and are likely rejected as bad entries.
  comments_re = re.compile('#.*$')
  disabled_tests = []
  disabled_method_types = []
  for l in manifest_content.splitlines():
    stripped = comments_re.sub('', l).strip()
    if not stripped:
      continue
    entry = stripped.split(' ')
    if len(entry) == 1:
      disabled_tests.append(entry[0])
    elif len(entry) == 2:
      disabled_method_types.append((entry[0], entry[1].strip().split(',')))
    else:
      raise ValueError('Bad entry in manifest file.')

  disabled_regex = '|'.join(disabled_tests)
  method_types_filter = {}
  for method, types in disabled_method_types:
    method_types_filter[method] = set([
        dtypes.as_dtype(types_pb2.DataType.Value(name)).as_numpy_dtype
        for name in types
    ])
  return disabled_regex, method_types_filter


class TPURewriteSession(session.Session):
  # [CLASS DOC]
  # A session that, when running, wraps any fetched operation (not tensor) in a TPU rewrite
  # before executing. On the first run it also initializes the TPU system and stores the
  # returned topology in the topology attribute. Used by XLATestCase when ops must go through
  # TPU compilation.
  """Tensorflow session that runs tpu.rewrite() on ops on run()."""

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Passes all positional and keyword arguments to the base session constructor and sets the
    # topology attribute to nothing (not yet initialized).
    super().__init__(*args, **kwargs)
    self.topology = None

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    # [FUNCTION DOC]
    # Same inputs as the base session run: fetches (any structure of tensors/operations),
    # optional feed mapping, run options and run metadata. First call initializes the TPU system
    # (running the initialization op) and caches the topology; asserts it is not nothing.
    # Every fetched operation is replaced by its TPU rewrite; tensors are left as they are.
    # Returns the base run results in the same structure as fetches.
    from tensorflow.python.tpu import tpu  # pylint: disable=g-import-not-at-top

    if self.topology is None:
      self.topology = super().run(tpu.initialize_system())
      assert self.topology is not None
    fetch_mapper = session._FetchMapper.for_fetch(fetches)
    new_fetches = []
    for fetch in fetch_mapper.unique_fetches():
      if isinstance(fetch, ops.Operation):
        fetch = tpu.rewrite(lambda fetch=fetch: fetch)
      new_fetches.append(fetch)
    rewritten_fetches = fetch_mapper.build_results(new_fetches)
    return super().run(rewritten_fetches, feed_dict, options, run_metadata)


class XLATestCase(test.TestCase):
  # [CLASS DOC]
  # Base class for XLA op tests. Reads the test device name, the list of data types and the
  # optional disabled-tests manifest from command-line flags, and exposes sets of data types
  # (all, integer, signed/unsigned, float, complex, numeric, quantized) as properties that
  # honor per-test exclusions from the manifest. Also supplies a custom session, a device
  # scope, and an output-check helper. Constructing it may enable XLA devices and set the MLIR
  # bridge option on the global context, and may set an environment variable.
  """XLA test cases are parameterized test cases."""

  def __init__(self, method_name='runTest'):
    # [FUNCTION DOC]
    # Input: method_name, name of the test method to run (default is the standard single-test
    # name). Reads flags: the test device name (must be set, else this fails because it
    # is searched for XLA), the comma-separated type names (must be set), the optional
    # manifest path and optional XLA flags text.
    # Side effects: enables XLA devices on the global context when the device name contains XLA;
    # sets the MLIR bridge option on the global context to true or false only if it was
    # explicitly enabled or disabled; sets the XLA flags environment variable if given.
    # Sets: device, a flag for custom-call support (true only for the XLA CPU device), a
    # rewrite-for-TPU flag (false), many type sets, the disabled-test regex (compiled from
    # the manifest, or nothing) and the per-test type filter mapping.
    # Quantized types are kept only in the TF-type sets as they have no numpy equivalent.
    # Raises: a file error if the manifest cannot be opened, a value error from manifest parsing.
    super(XLATestCase, self).__init__(method_name)
    if 'XLA' in FLAGS.test_device:
      context.context().enable_xla_devices()

    # Check if the mlir bridge has been explicitly enabled or disabled. If
    # is_mlir_bridge_enabled() returns None, the user did not explicitly enable
    # or disable the bridge so do not update enable_mlir_bridge.
    if test_util.is_mlir_bridge_enabled():
      context.context().enable_mlir_bridge = True
    elif test_util.is_mlir_bridge_enabled() is not None:
      context.context().enable_mlir_bridge = False

    self.device = FLAGS.test_device
    self.has_custom_call = self.device == 'XLA_CPU'

    # Some tests (e.g. ftrl_ops) only work if the program goes through the
    # _TPUCompileMLIR op. They will set this flag to True.
    # TODO(kramm): Flip to true (and enable MLIR bridge) for more tests.
    self.rewrite_ops_for_tpu = False

    self._all_tf_types = set([
        dtypes.as_dtype(types_pb2.DataType.Value(name))
        for name in FLAGS.types.split(',')
    ])
    self.int_tf_types = set(
        [dtype for dtype in self._all_tf_types if dtype.is_integer]
    )
    self._float_tf_types = set(
        [dtype for dtype in self._all_tf_types if dtype.is_floating]
    )
    self.complex_tf_types = set(
        [dtype for dtype in self._all_tf_types if dtype.is_complex]
    )
    self._numeric_tf_types = set(
        self.int_tf_types | self._float_tf_types | self.complex_tf_types
    )
    self.quantized_tf_types = set(
        dtype for dtype in self._all_tf_types if dtype.is_quantized
    )

    # Quantized types don't have a numpy equivalent, include them in
    # all_tf_types but not in all_types.
    # TODO(b/115960798): Parametrize tests on TF types instead of numpy types
    # and remove all_types.
    self._all_types = set(
        dtype.as_numpy_dtype
        for dtype in self._all_tf_types
        if not dtype.is_quantized
    )
    self._int_types = set([dtype.as_numpy_dtype for dtype in self.int_tf_types])
    self.signed_int_types = set(
        dtype.as_numpy_dtype
        for dtype in self.int_tf_types
        if not dtype.is_unsigned
    )
    self.unsigned_int_types = set(
        dtype.as_numpy_dtype for dtype in self.int_tf_types if dtype.is_unsigned
    )
    self._float_types = set(
        [dtype.as_numpy_dtype for dtype in self._float_tf_types]
    )
    self.complex_types = set(
        [dtype.as_numpy_dtype for dtype in self.complex_tf_types]
    )
    self._numeric_types = set(
        self._int_types | self._float_types | self.complex_types
    )

    # Parse the manifest file, if any, into a regex identifying tests to
    # disable
    # TODO(xpan): Make it text proto if it doesn't scale.
    # Each line of the manifest file specifies an entry. The entry can be
    # 1) TestNameRegex  // E.g. CumprodTest.* Or
    # 2) TestName TypeName  // E.g. AdamOptimizerTest.testSharing DT_BFLOAT16
    # The 1) disables the entire test. While 2) only filter some numeric types
    # so that they are not used in those tests.
    self.disabled_regex = None
    self._method_types_filter = {}

    if FLAGS.disabled_manifest is not None:
      with open(FLAGS.disabled_manifest, 'r') as manifest_file:
        disabled_regex, self._method_types_filter = parse_disabled_manifest(
            manifest_file.read()
        )
        if disabled_regex:
          self.disabled_regex = re.compile(disabled_regex)

    if FLAGS.tf_xla_flags is not None:
      os.environ['TF_XLA_FLAGS'] = FLAGS.tf_xla_flags

  @property
  def all_tf_types(self):
    # [FUNCTION DOC]
    # Property: set of all TensorFlow data types under test, minus the types the manifest
    # excludes for the current test (looked up by class name and test method name).
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    tf_types = set(
        [dtypes.as_dtype(t) for t in self._method_types_filter.get(name, set())]
    )
    return self._all_tf_types - tf_types

  @property
  def float_types(self):
    # [FUNCTION DOC]
    # Property: set of floating-point numpy types under test, minus those the manifest excludes
    # for the current test.
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    return self._float_types - self._method_types_filter.get(name, set())

  @property
  def float_tf_types(self):
    # [FUNCTION DOC]
    # Property: set of floating-point TensorFlow types under test, minus the manifest exclusions.
    # KNOWN ISSUE: subtracts numpy types from a set of TensorFlow type objects (unlike the
    # sibling properties that convert first), so exclusions likely never take effect.
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    return self._float_tf_types - self._method_types_filter.get(name, set())

  @property
  def int_types(self):
    # [FUNCTION DOC]
    # Property: set of integer numpy types under test, minus the manifest exclusions for the
    # current test.
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    return self._int_types - self._method_types_filter.get(name, set())

  @property
  def numeric_tf_types(self):
    # [FUNCTION DOC]
    # Property: set of numeric (integer, float, complex) TensorFlow types under test, minus the
    # manifest exclusions for the current test (converted to TensorFlow types first).
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    tf_types = set(
        [dtypes.as_dtype(t) for t in self._method_types_filter.get(name, set())]
    )
    return self._numeric_tf_types - tf_types

  @property
  def numeric_types(self):
    # [FUNCTION DOC]
    # Property: set of numeric (integer, float, complex) numpy types under test, minus the
    # manifest exclusions for the current test.
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    return self._numeric_types - self._method_types_filter.get(name, set())

  @property
  def all_types(self):
    # [FUNCTION DOC]
    # Property: set of all non-quantized numpy types under test, minus the manifest exclusions
    # for the current test.
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    return self._all_types - self._method_types_filter.get(name, set())

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent setup, then skips the test (with a message) when its class-and-method name
    # matches the disabled regex from the manifest. Otherwise logs the start and seeds both the
    # Python and numpy random generators with the default graph seed for reproducibility.
    # Returns nothing.
    super(XLATestCase, self).setUp()
    name = '{}.{}'.format(type(self).__name__, self._testMethodName)
    if self.disabled_regex is not None and self.disabled_regex.match(name):
      logging.info('Disabled test case: %s', name)
      self.skipTest('{} is disabled by manifest.'.format(name))
      return
    logging.info('Start test case: %s', name)

    random.seed(random_seed.DEFAULT_GRAPH_SEED)
    np.random.seed(random_seed.DEFAULT_GRAPH_SEED)

  def tearDown(self):
    # [FUNCTION DOC]
    # Runs the parent teardown, then logs that the test method has ended.
    super(XLATestCase, self).tearDown()
    logging.info('End test case: %s', self._testMethodName)

  @contextlib.contextmanager
  def session(self) -> Iterator[session.Session]:
    # [FUNCTION DOC]
    # Context manager (yields one session) replacing the standard test session. Creates a fresh
    # graph, copies the global context config, turns off constant folding in the graph rewriter
    # (it would create variant constants that XLA cannot handle), and opens either a TPU-rewriting
    # session (when the rewrite-for-TPU flag is set) or an ordinary session on that graph, with the
    # graph made default inside the block. Yields the session; closes it on exit.
    # Note: soft placement is not enabled by this method.
    """Custom implementation of session() for XLA tests.

    We override the standard Tensorflow session() since it is too
    specific to CPU and GPU tests. In particular, we want to disable soft
    placement and explicitly assign ops to devices under test.

    Yields:
      A session to use when running a test case.
    """
    graph = ops.Graph()
    config = context.context().config

    # Grappler can constant fold TensorListFromTensor ops into DT_VARIANT
    # constants which XLA does not understand.  So disable constant folding in
    # these tests.
    config.graph_options.rewrite_options.constant_folding = (
        rewriter_config_pb2.RewriterConfig.OFF
    )

    if self.rewrite_ops_for_tpu:
      session_type = TPURewriteSession
    else:
      session_type = session.Session

    with session_type(graph=graph, config=config) as sess, graph.as_default():
      yield sess

  def cached_session(self):
    # [FUNCTION DOC]
    # Unsupported override: always raises a not-implemented error telling the caller to use the
    # session method instead.
    raise NotImplementedError(
        'cached_session not supported on XLATestCase, please use session'
    )

  def test_session(self):
    # [FUNCTION DOC]
    # Unsupported override: always raises a not-implemented error telling the caller to use the
    # session method instead.
    raise NotImplementedError(
        'test_session not supported on XLATestCase, please use session'
    )

  @contextlib.contextmanager
  def device_scope(self):
    # [FUNCTION DOC]
    # Context manager that places operations created inside it on the first device of the
    # configured test device type. Takes no arguments; yields nothing useful.
    """Scope that runs tests on `self.device`.

    Yields:
      A scope to apply to the operators under test.
    """
    with ops.device('device:{}:0'.format(self.device)):
      yield

  def assert_op_output_matches_expected(
      self, op, inp, expected, local_session,
      equality_test=None, rtol=1e-3, atol=1e-5
  ):
    # [FUNCTION DOC]
    # Builds a placeholder matching the input array's type and shape inside the test device scope,
    # applies the operator to it, runs it in the given session feeding the input, and compares.
    # Inputs: op (callable from tensor to tensor), inp (array), expected (array), local_session
    # (an open session), optional equality_test (callable taking result, expected, and tolerances;
    # default uses a type-aware closeness check), rtol and atol tolerances (defaults 1e-3, 1e-5).
    # Without a custom equality test it first asserts the output dtype equals the expected dtype,
    # then checks closeness with a looser relative tolerance for bfloat16. Returns nothing;
    # fails the test on mismatch.
    """Verifies that 'op' produces 'expected' when fed input 'inp' .

    Args:
      op: operator to test
      inp: numpy input array to use as input to 'op'.
      expected: numpy array representing the expected output of 'op'.
      local_session: The session to use for the test.
      equality_test: either None, or a function that tests two numpy arrays for
        equality. If None, self.assertAllClose is used.
      rtol: relative tolerance for equality test.
      atol: absolute tolerance for equality test.
    """
    with self.test_scope():
      pinp = array_ops.placeholder(
          dtypes.as_dtype(inp.dtype), inp.shape, name='a'
      )
      output = op(pinp)
    result = local_session.run(output, {pinp: inp})
    if equality_test is None:
      self.assertEqual(output.dtype, expected.dtype)
      self.assertAllCloseAccordingToType(
          expected, result, rtol=rtol, atol=atol, bfloat16_rtol=0.03
      )
    else:
      equality_test(result, expected, rtol=rtol, atol=atol)

  def test_scope(self):
    # [FUNCTION DOC]
    # HELPER: deprecated alias kept for older tests; simply returns the device scope.
    """Deprecated alias of `device_scope`.

    This should be avoided as the name starts with `test`, so test runners
    treat it as a test. This interferes with class decorators that operate on
    each test method.
    """
    return self.device_scope()


def Benchmark(
    tf_bench, builder_fn, use_xla_jit, device, separate_compiled_gradients=False
):
  # [FUNCTION DOC]
  # HELPER: shared by benchmark files in this directory.
  # Inputs: tf_bench (benchmark object that runs op benchmarks), builder_fn (callable returning a
  # name and a list of fetch tensors), use_xla_jit (bool), device (device name text),
  # separate_compiled_gradients (bool, default false; gradients get their own compilation scope).
  # Builds the graph in a new default graph on the device inside an experimental JIT scope, adds
  # identity ops (outside compilation) after each fetch so XLA cannot drop the work, then in a
  # session with soft placement allowed initializes all global variables and runs the benchmark
  # under a name made of the test name, an XLA marker if enabled, and the device.
  # Returns nothing; benchmark results are recorded by the benchmark object.
  """Build a graph and run benchmarks against it, with or without XLA.

  Args:
    tf_bench: An instance of tf.test.Benchmark, used to run the benchmark.
    builder_fn: A function that builds a graph when invoked, and returns (name,
      fetches), where name is the name of the test, and fetches is a list of
      tensors to fetch as output.
    use_xla_jit: If true compile with the XLA JIT, otherwise use regular TF.
    device: The tensorflow device to run on, e.g. "cpu", "gpu".
    separate_compiled_gradients: If true put each gradient subgraph into a
      separate compilation scope. This gives fine-grained control over which
      portions of the graph will be compiled as a single unit. Compiling
      gradients separately may yield better performance for some graphs. The
      scope is named based on the scope of the forward computation as well as
      the name of the gradients. As a result, the gradients will be compiled in
      a scope that is separate from both the forward computation, and from other
      gradients.
  """

  with ops.Graph().as_default():
    name = None
    targets = []
    with ops.device(device):
      fetches = []
      jit_scope = jit.experimental_jit_scope
      with jit_scope(
          compile_ops=use_xla_jit,
          separate_compiled_gradients=separate_compiled_gradients,
      ):
        name, fetches = builder_fn()

      # We only want to benchmark the operations themselves, and not the data
      # transfer of the result(s).  Non-compiled identity ops ensure XLA
      # doesn't know we're dropping the results, otherwise it might compile
      # away the entire computation.
      for fetch in fetches:
        targets.append(array_ops.identity(fetch).op)

    # TODO(b/132430685):  Should we allow soft placement here?
    config = config_pb2.ConfigProto(allow_soft_placement=True)
    with session.Session(config=config) as sess:
      sess.run(variables.global_variables_initializer())
      xla = 'xla_' if use_xla_jit else ''
      tf_bench.run_op_benchmark(
          sess, targets, name='%s_%s%s' % (name, xla, device)
      )
