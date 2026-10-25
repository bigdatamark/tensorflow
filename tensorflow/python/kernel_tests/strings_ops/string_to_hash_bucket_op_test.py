# [MODULE DOC]
# Tests for the string-to-hash-bucket operations in string_ops: the fast (fingerprint based),
# legacy and strong keyed variants. Checks that a single bucket maps everything to zero, that
# buckets for known strings match precomputed hash values modulo the bucket count, and that a
# strong-hash key of the wrong length raises an op error. Run via the standard test main entry
# point. No import-time side effects. Not a helper module.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for StringToHashBucket op from string_ops."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class StringToHashBucketOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the three string hashing operations. Graph-mode tests feed strings through a
  # string placeholder; the strong-hash tests use constants. Expected buckets are hard-coded.

  @test_util.run_deprecated_v1
  def testStringToOneHashBucketFast(self):
    # [FUNCTION DOC]
    # Feeds three strings to the fast hash operation with a single bucket and asserts every
    # result is bucket zero. Graph mode only. Returns nothing.
    with self.cached_session():
      input_string = array_ops.placeholder(dtypes.string)
      output = string_ops.string_to_hash_bucket_fast(input_string, 1)
      result = output.eval(feed_dict={input_string: ['a', 'b', 'c']})

      self.assertAllEqual([0, 0, 0], result)

  @test_util.run_deprecated_v1
  def testStringToHashBucketsFast(self):
    # [FUNCTION DOC]
    # Feeds the strings a, b, c, d to the fast hash operation with ten buckets and asserts the
    # buckets equal the 64-bit fingerprints modulo ten (nine, two, two, five). Graph mode only.
    # Returns nothing.
    with self.cached_session():
      input_string = array_ops.placeholder(dtypes.string)
      output = string_ops.string_to_hash_bucket_fast(input_string, 10)
      result = output.eval(feed_dict={input_string: ['a', 'b', 'c', 'd']})

      # Fingerprint64('a') -> 12917804110809363939 -> mod 10 -> 9
      # Fingerprint64('b') -> 11795596070477164822 -> mod 10 -> 2
      # Fingerprint64('c') -> 11430444447143000872 -> mod 10 -> 2
      # Fingerprint64('d') -> 4470636696479570465 -> mod 10 -> 5
      self.assertAllEqual([9, 2, 2, 5], result)

  @test_util.run_deprecated_v1
  def testStringToOneHashBucketLegacyHash(self):
    # [FUNCTION DOC]
    # Feeds three strings to the legacy hash operation with a single bucket and asserts every
    # result is bucket zero. Graph mode only. Returns nothing.
    with self.cached_session():
      input_string = array_ops.placeholder(dtypes.string)
      output = string_ops.string_to_hash_bucket(input_string, 1)
      result = output.eval(feed_dict={input_string: ['a', 'b', 'c']})

      self.assertAllEqual([0, 0, 0], result)

  @test_util.run_deprecated_v1
  def testStringToHashBucketsLegacyHash(self):
    # [FUNCTION DOC]
    # Feeds the strings a, b, c to the legacy hash operation with ten buckets and asserts the
    # buckets equal the legacy 64-bit hash modulo ten (eight, zero, seven). Graph mode only.
    # Returns nothing.
    with self.cached_session():
      input_string = array_ops.placeholder(dtypes.string)
      output = string_ops.string_to_hash_bucket(input_string, 10)
      result = output.eval(feed_dict={input_string: ['a', 'b', 'c']})

      # Hash64('a') -> 2996632905371535868 -> mod 10 -> 8
      # Hash64('b') -> 5795986006276551370 -> mod 10 -> 0
      # Hash64('c') -> 14899841994519054197 -> mod 10 -> 7
      self.assertAllEqual([8, 0, 7], result)

  def testStringToOneHashBucketStrongOneHashBucket(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name repeats the words one hash bucket; it only checks the strong
    # hash with a single bucket.
    # Hashes three constant strings with the strong keyed operation, a two-element key and one
    # bucket, and asserts every result is zero. Returns nothing.
    with self.cached_session():
      input_string = constant_op.constant(['a', 'b', 'c'])
      output = string_ops.string_to_hash_bucket_strong(
          input_string, 1, key=[123, 345])
      self.assertAllEqual([0, 0, 0], self.evaluate(output))

  def testStringToHashBucketsStrong(self):
    # [FUNCTION DOC]
    # Hashes the strings a, b, c with the strong keyed operation, ten buckets and a fixed
    # two-element key, and asserts the buckets equal the precomputed keyed hashes modulo ten
    # (four, two, eight). Returns nothing.
    with self.cached_session():
      input_string = constant_op.constant(['a', 'b', 'c'])
      output = string_ops.string_to_hash_bucket_strong(
          input_string, 10, key=[98765, 132])
      # key = [98765, 132]
      # StrongKeyedHash(key, 'a') -> 7157389809176466784 -> mod 10 -> 4
      # StrongKeyedHash(key, 'b') -> 15805638358933211562 -> mod 10 -> 2
      # StrongKeyedHash(key, 'c') -> 18100027895074076528 -> mod 10 -> 8
      self.assertAllEqual([4, 2, 8], self.evaluate(output))

  def testStringToHashBucketsStrongInvalidKey(self):
    # [FUNCTION DOC]
    # Calls the strong keyed hash with a one-element key and asserts that evaluating raises an op
    # error stating the key must have two elements. Returns nothing.
    with self.cached_session():
      input_string = constant_op.constant(['a', 'b', 'c'])
      with self.assertRaisesOpError('Key must have 2 elements'):
        string_ops.string_to_hash_bucket_strong(
            input_string, 10, key=[98765]).eval()


if __name__ == '__main__':
  test.main()
