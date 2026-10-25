# [MODULE DOC]
# Unit tests for the NCCL collective ops in nccl_ops: all-reduce (sum, product, min, max),
# single-device reduce, broadcast, their gradients, error handling and a combined graph that
# mixes all of them. Tests run over several device sets of virtual GPUs and several dtypes;
# when no GPU is available only graph construction and shape inference are checked. Module-level
# helper functions adapt the ops to a common calling convention used by the shared test
# fixtures. Runs through the test main entry point.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for nccl ops. See also the cc test for nccl_communicator."""

from functools import partial

import numpy as np

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import nccl_ops
from tensorflow.python.platform import test


def _DeviceTensors(tensors, devices):
  # [FUNCTION DOC]
  # HELPER: test utility that places values on specific devices.
  # Pairs each input value with a device and, under that device scope, wraps the value in an
  # identity op. Returns the list of resulting tensors, one per pair (extra items in the longer
  # list are ignored).
  res = []
  for t, d in zip(tensors, devices):
    with ops.device(d):
      res.append(array_ops.identity(t))
  return res


def _NcclAllReduce(nccl_fun, tensors, devices):
  # [FUNCTION DOC]
  # HELPER: adapter used by the tests for the all-reduce variants. Places the tensors on the
  # given devices and applies the supplied all-reduce function, returning its list of outputs.
  return nccl_fun(_DeviceTensors(tensors, devices))


def _NcclReduce(nccl_fun, tensors, devices):
  # [FUNCTION DOC]
  # HELPER: adapter used by the tests for the single-device reduce. Randomly picks one of the
  # devices as the receiver, then under that device places the tensors on their devices and
  # applies the supplied reduce function. Returns a one-element list with the reduced tensor.
  # Uses the global numpy random generator.
  receiver = np.random.randint(0, len(devices))
  with ops.device(devices[receiver]):
    return [nccl_fun(_DeviceTensors(tensors, devices))]


def _NcclBroadcast(tensors, devices):
  # [FUNCTION DOC]
  # HELPER: adapter used by the tests for broadcast. Randomly chooses a sender device, copies the
  # first input tensor there and broadcasts it, then returns one identity of the broadcast result
  # on each of the given devices (so every device receives the sender value). Only the first
  # tensor is used. Uses the global numpy random generator.
  sender = np.random.randint(0, len(devices))
  with ops.device(devices[sender]):
    tensor = array_ops.identity(tensors[0])
    broadcast = nccl_ops.broadcast(tensor)
  return _DeviceTensors([broadcast] * len(devices), devices)


class NcclTestCase(test.TestCase):
  # [CLASS DOC]
  # HELPER: base test case providing shared fixtures for the NCCL tests. It offers a generic
  # forward check comparing an NCCL function against a numpy reduction and a gradient check built
  # on top of it. Holds no state; concrete test classes subclass it and call these methods.

  def _Test(self,
            nccl_reduce,
            numpy_fn,
            device_sets=(['/device:GPU:1', '/device:GPU:2', '/device:GPU:0'],
                         ['/device:GPU:1', '/device:GPU:0'])):
    # [FUNCTION DOC]
    # HELPER: shared check used by the test methods.
    # KNOWN ISSUE: the docstring says the reduce function returns a list of tensors and a list of
    # ops, but the adapters actually return only a list of tensors.
    # For each dtype (half, single, 32-bit and 64-bit integer, double) and each device set,
    # builds random inputs of shape three by four, computes the expected result by folding the
    # numpy function over the inputs, calls the NCCL function, and asserts the result list is not
    # empty and that each result has the expected static shape. If no GPU is available (checked
    # after creating a session) it stops after graph construction; otherwise it evaluates the
    # identity-wrapped results and asserts each is close to the expected array. Inputs: the NCCL
    # adapter, the numpy combiner of two arrays, and a tuple of device lists. Returns nothing.
    """Tests that nccl_reduce does the same as reduction with numpy_fn.

    Args:
      nccl_reduce: A function taking a list of tensors and a list of devices,
          and returns a list of reduced tensors and a list of ops to perform the
          reduction.
      numpy_fn: A function taking two tensors and returning the reduction of the
          two.
      device_sets: Tuple of virtual devices to run test on.
    """
    for dtype in [np.float16, np.float32, np.int32, np.int64, np.float64]:
      # Create session inside outer loop to test use of
      # same communicator across multiple sessions.
      with self.test_session():

        for devices in device_sets:
          shape = (3, 4)
          random = (np.random.random_sample(shape) - .5) * 1024
          tensors = []
          for _ in devices:
            tensors.append(random.astype(dtype))
          np_ans = tensors[0]
          for t in tensors[1:]:
            np_ans = numpy_fn(np_ans, t)

          reduce_tensors = nccl_reduce(tensors, devices)
          self.assertNotEmpty(reduce_tensors)

          # Test shape inference.
          for r in reduce_tensors:
            self.assertEqual(shape, r.get_shape())

          result_tensors = [array_ops.identity(t) for t in reduce_tensors]

          # Check GPU availability *after* creating session, see b/68975239.
          if not test.is_gpu_available():
            # If no GPU is available, only test graph construction.
            continue

          # Test execution and results.
          for t in self.evaluate(result_tensors):
            self.assertAllClose(t, np_ans)

  def _TestGradient(self, nccl_reduce, numpy_fn):
    # [FUNCTION DOC]
    # HELPER: shared gradient check used by the gradient tests.
    # Wraps the NCCL adapter in a function that builds placeholders, applies the adapter, uses
    # device-placed random upstream values as gradient seeds, takes gradients with respect to the
    # placeholders (colocated with their ops) and keeps the non-null ones; then runs the general
    # forward check on that function with the given numpy combiner as the expected gradient.
    # Inputs: the NCCL adapter and the numpy function giving expected gradients. Returns nothing.
    """Tests the gradient of nccl_reduce.

    Args:
      nccl_reduce: A function taking a list of tensors and a list of devices,
          and returns a list of reduced tensors and a list of ops to perform the
          reduction.
      numpy_fn: A function taking two tensors and returning the gradient of the
          reduction of the two.
    """

    def _Gradient(tensors, devices):
      # [FUNCTION DOC]
      # HELPER: adapter passed to the forward check by the gradient test. Creates placeholders
      # matching the tensors, applies the NCCL function, builds device-placed upstream gradient
      # values from the tensors, and returns the list of non-null gradients of the reduced
      # outputs with respect to the placeholders.
      inputs = [array_ops.placeholder(t.dtype, t.shape) for t in tensors]
      reduce_tensors = nccl_reduce(inputs, devices)
      losses = _DeviceTensors(tensors, [t.device for t in reduce_tensors])
      grads = gradients.gradients(
          reduce_tensors, inputs, losses, colocate_gradients_with_ops=True)
      return [g for g in grads if g is not None]

    self._Test(_Gradient, numpy_fn)


class AllReduceTest(NcclTestCase):
  # [CLASS DOC]
  # Tests for the all-reduce functions (sum, product, min, max): forward results, the sum
  # gradient, and argument validation errors. Uses the shared fixtures of the base class.

  def testAllReduce(self):
    # [FUNCTION DOC]
    # Runs the forward check for all-reduce sum, product, minimum and maximum against the
    # matching numpy operations; asserts shapes and (with GPUs) values.
    self._Test(partial(_NcclAllReduce, nccl_ops.all_sum), lambda x, y: x + y)
    self._Test(partial(_NcclAllReduce, nccl_ops.all_prod), lambda x, y: x * y)
    self._Test(partial(_NcclAllReduce, nccl_ops.all_min), np.minimum)
    self._Test(partial(_NcclAllReduce, nccl_ops.all_max), np.maximum)

  def testAllSumGrad(self):
    # [FUNCTION DOC]
    # Runs the gradient check for all-reduce sum; the expected gradient on each device is the sum
    # of the upstream values.
    self._TestGradient(
        partial(_NcclAllReduce, nccl_ops.all_sum), lambda x, y: x + y)

  def testErrors(self):
    # [FUNCTION DOC]
    # Asserts that all-reduce sum raises a value error mentioning a required device assignment
    # when given a tensor with no device, and a value error about needing more than zero tensors
    # when given an empty list.
    with self.assertRaisesRegex(ValueError, 'Device assignment .* required'):
      nccl_ops.all_sum([array_ops.identity(np.random.random_sample((3, 4)))])
    with self.assertRaisesRegex(ValueError, 'Must pass >0 tensors'):
      nccl_ops.all_sum([])


class SingleReduceTest(NcclTestCase):
  # [CLASS DOC]
  # Tests for the single-receiver reduce sum: forward values and gradient, using the shared
  # fixtures of the base class.

  def testSum(self):
    # [FUNCTION DOC]
    # Runs the forward check for reduce sum on a randomly chosen receiver device; the result must
    # equal the numpy sum of all inputs.
    self._Test(partial(_NcclReduce, nccl_ops.reduce_sum), lambda x, y: x + y)

  def testSumGrad(self):
    # [FUNCTION DOC]
    # Runs the gradient check for reduce sum; the expected gradient for every input equals the
    # upstream value (the first of the two compared arrays).
    self._TestGradient(partial(_NcclReduce, nccl_ops.reduce_sum),
                       lambda x, y: x)


class BroadcastTest(NcclTestCase):
  # [CLASS DOC]
  # Tests for the broadcast op, including the single-device case and the unsupported broadcast
  # to a CPU device, using the shared fixtures of the base class.

  def testBroadcast(self):
    # [FUNCTION DOC]
    # Runs the forward check for broadcast; every receiving device must see the first input
    # value.
    self._Test(_NcclBroadcast, lambda x, y: x)

  def testBroadcastSingleDevice(self):
    # [FUNCTION DOC]
    # Runs the broadcast forward check with both devices being the same GPU, which the graph
    # rewrite removes entirely; the values must still match the first input.
    # Broadcasts on a single device are removed completely during rewrite.
    self._Test(_NcclBroadcast, lambda x, y: x,
               (['/device:GPU:0', '/device:GPU:0'],))

  def testBroadcastToCpuError(self):
    # [FUNCTION DOC]
    # Broadcasts from a GPU to a CPU device, which is unsupported. If the forward check raises a
    # not-found error, asserts its message says no receive kernel is registered for CPU devices.
    # If nothing is raised and a GPU is available, the test fails; without a GPU the session is
    # never run so no error is expected.
    try:
      # Broadcasts to CPU is not supported.
      self._Test(_NcclBroadcast, lambda x, y: x,
                 (['/device:GPU:0', '/device:CPU:0'],))
    except errors.NotFoundError as e:
      self.assertRegex(
          str(e), "No registered '_NcclBroadcastRecv' OpKernel for CPU devices")
    else:
      # Session isn't executed when no GPU is available.
      if test.is_gpu_available():
        self.fail("Didn't raise NotFoundError trying to broadcast to CPU")


class CombinedTest(NcclTestCase):
  # [CLASS DOC]
  # Test case that builds all-reduce, single reduce and broadcast in one graph and evaluates
  # them together in a single run, checking they interoperate and agree.
  """Test all-reduce vs. single-reduce plus broadcast in one session.run."""

  def _Combined(self, tensors, devices):
    # [FUNCTION DOC]
    # HELPER: adapter for the combined test. Builds an all-reduce sum, a single-device reduce sum
    # and a broadcast of the reduced value, and returns the concatenated list of all-reduce
    # outputs and broadcast outputs so all are evaluated together.
    all_reduce_tensors = _NcclAllReduce(nccl_ops.all_sum, tensors, devices)
    single_reduce_tensors = _NcclReduce(nccl_ops.reduce_sum, tensors, devices)
    broadcast_tensors = _NcclBroadcast(single_reduce_tensors, devices)
    return all_reduce_tensors + broadcast_tensors

  def testCombined(self):
    # [FUNCTION DOC]
    # Runs the forward check on the combined adapter; both all-reduce and reduce-then-broadcast
    # results must equal the numpy sum of the inputs.
    self._Test(self._Combined, lambda x, y: x + y)


if __name__ == '__main__':
  test.main()
