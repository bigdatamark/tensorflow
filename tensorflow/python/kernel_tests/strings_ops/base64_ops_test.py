# [MODULE DOC]
# Tests for the base64 encode and decode string operations (string_ops.encode_base64 and
# decode_base64), in padded and unpadded modes. Verifies agreement with the Python url-safe
# base64 codec on random data, shape preservation including zero-element inputs, and
# invalid-argument errors for malformed encodings. Graph-mode tests run via the standard test
# main entry point. No import-time side effects. Not a helper module.
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

"""Tests for EncodeBase64 and DecodeBase64."""

import base64

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


@test_util.run_deprecated_v1
class Base64OpsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (graph mode only, via the deprecated-v1 decorator) for the encode and decode
  # base64 string operations. Its fixture holds a string placeholder and the encode/decode
  # graphs both without and with padding. Results are compared against the Python url-safe
  # base64 implementation.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: creates a string placeholder and stores it plus four graph nodes on the instance:
    # encode without padding, decode of that, encode with padding, and decode of that.
    # Takes no arguments, returns nothing.
    self._msg = array_ops.placeholder(dtype=dtypes.string)
    self._encoded_f = string_ops.encode_base64(self._msg, pad=False)
    self._decoded_f = string_ops.decode_base64(self._encoded_f)
    self._encoded_t = string_ops.encode_base64(self._msg, pad=True)
    self._decoded_t = string_ops.decode_base64(self._encoded_t)

  def _RemovePad(self, msg, base64_msg):
    # [FUNCTION DOC]
    # HELPER: lets the round-trip test derive the expected unpadded encoding from the padded one.
    # Given the raw message bytes and its padded base64 encoding, strips two trailing characters
    # when the message length leaves remainder one modulo three, one trailing character when
    # the remainder is two, and nothing otherwise. Returns the trimmed encoding.
    if len(msg) % 3 == 1:
      return base64_msg[:-2]
    if len(msg) % 3 == 2:
      return base64_msg[:-1]
    return base64_msg

  def _RunTest(self, msg, pad):
    # [FUNCTION DOC]
    # HELPER: shared round-trip check used by the encode/decode tests.
    # Takes a single byte string or a list/tuple of them, plus a flag selecting padded or unpadded
    # operations. Feeds the message to the matching prebuilt encode and decode graph in a cached
    # session, wraps single values into one-element lists, then asserts every encoded value
    # equals the Python url-safe base64 encoding (padding removed when requested) and every
    # decoded value equals the original message. Returns nothing.
    with self.cached_session() as sess:
      if pad:
        encoded, decoded = sess.run([self._encoded_t, self._decoded_t],
                                    feed_dict={self._msg: msg})
      else:
        encoded, decoded = sess.run([self._encoded_f, self._decoded_f],
                                    feed_dict={self._msg: msg})

    if not isinstance(msg, (list, tuple)):
      msg = [msg]
      encoded = [encoded]
      decoded = [decoded]

    base64_msg = [base64.urlsafe_b64encode(m) for m in msg]
    if not pad:
      base64_msg = [self._RemovePad(m, b) for m, b in zip(msg, base64_msg)]

    for i in range(len(msg)):
      self.assertEqual(base64_msg[i], encoded[i])
      self.assertEqual(msg[i], decoded[i])

  def testWithPythonBase64(self):
    # [FUNCTION DOC]
    # For both padded and unpadded modes, checks the empty message and one hundred random byte
    # strings of random length below one mebibyte against the Python base64 implementation via
    # the shared round-trip helper. Returns nothing.
    for pad in (False, True):
      self._RunTest(b"", pad=pad)

      for _ in range(100):
        length = np.random.randint(1024 * 1024)
        msg = np.random.bytes(length)
        self._RunTest(msg, pad=pad)

  def testShape(self):
    # [FUNCTION DOC]
    # For both padding modes, round-trips ten random lists of short random byte strings through
    # the helper, then ten times builds an empty byte array with zero rows and a random number
    # of columns, encodes and decodes it, and asserts both outputs keep the input shape.
    # Returns nothing.
    for pad in (False, True):
      for _ in range(10):
        msg = [np.random.bytes(np.random.randint(20))
               for _ in range(np.random.randint(10))]
        self._RunTest(msg, pad=pad)

      # Zero-element, non-trivial shapes.
      for _ in range(10):
        k = np.random.randint(10)
        msg = np.empty((0, k), dtype=bytes)
        encoded = string_ops.encode_base64(msg, pad=pad)
        decoded = string_ops.decode_base64(encoded)

        with self.cached_session() as sess:
          encoded_value, decoded_value = self.evaluate([encoded, decoded])

        self.assertEqual(encoded_value.shape, msg.shape)
        self.assertEqual(decoded_value.shape, msg.shape)

  def testInvalidInput(self):
    # [FUNCTION DOC]
    # Asserts the decode operation raises an invalid-argument error for malformed input: a length
    # that is one modulo four (with a message mentioning that), a single character replaced by a
    # question mark, a non-ascii byte, a plus sign or a slash (the last two being non-url-safe),
    # partial padding, unnecessary padding of two to four equals signs, and padding characters
    # inserted in the middle. Returns nothing.
    def try_decode(enc):
      # [FUNCTION DOC]
      # HELPER: runs the unpadded decode graph on one encoded byte string by feeding it in
      # place of the encode output and evaluating in the default session. Returns nothing;
      # the decode error propagates to the caller.
      self._decoded_f.eval(feed_dict={self._encoded_f: enc})

    with self.cached_session():
      # Invalid length.
      msg = np.random.bytes(99)
      enc = base64.urlsafe_b64encode(msg)
      with self.assertRaisesRegex(errors.InvalidArgumentError, "1 modulo 4"):
        try_decode(enc + b"a")

      # Invalid char used in encoding.
      msg = np.random.bytes(34)
      enc = base64.urlsafe_b64encode(msg)
      for i in range(len(msg)):
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"?" + enc[(i + 1):])
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"\x80" + enc[(i + 1):])  # outside ascii range.
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"+" + enc[(i + 1):])  # not url-safe.
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"/" + enc[(i + 1):])  # not url-safe.

      # Partial padding.
      msg = np.random.bytes(34)
      enc = base64.urlsafe_b64encode(msg)
      with self.assertRaises(errors.InvalidArgumentError):
        # enc contains == at the end. Partial padding is not allowed.
        try_decode(enc[:-1])

      # Unnecessary padding.
      msg = np.random.bytes(33)
      enc = base64.urlsafe_b64encode(msg)
      with self.assertRaises(errors.InvalidArgumentError):
        try_decode(enc + b"==")
      with self.assertRaises(errors.InvalidArgumentError):
        try_decode(enc + b"===")
      with self.assertRaises(errors.InvalidArgumentError):
        try_decode(enc + b"====")

      # Padding in the middle. (Previous implementation was ok with this as long
      # as padding char location was 2 or 3 (mod 4).
      msg = np.random.bytes(33)
      enc = base64.urlsafe_b64encode(msg)
      for i in range(len(msg) - 1):
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"=" + enc[(i + 1):])
      for i in range(len(msg) - 2):
        with self.assertRaises(errors.InvalidArgumentError):
          try_decode(enc[:i] + b"==" + enc[(i + 2):])


if __name__ == "__main__":
  test.main()
