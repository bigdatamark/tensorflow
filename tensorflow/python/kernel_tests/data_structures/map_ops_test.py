# [MODULE DOC]
# Test module for the TensorMap ops exposed through the map_ops module (empty map creation, size,
# insert, lookup, erase, has-key and stack-keys) and for their gradient behavior.
# It defines a single test class that runs every test in both graph mode and eager mode.
# It exposes nothing for reuse; running the file as a script starts the standard test runner.
# No import-time side effects beyond imports.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for TensorMap ops."""
from absl.testing import parameterized
from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import map_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class MapOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the TensorMap (tensor-keyed map variant) ops: creating an empty map, querying
  # its size, inserting, looking up, erasing, checking key presence, stacking keys, and the gradients
  # that flow through insert, lookup and erase. The class decorator runs every test method in both
  # graph and eager modes. It holds no state of its own; each test builds its own map and tensors.
  # Derives from the TensorFlow test base case and the parameterized test mixin.

  def testEmptyTensorMapSize(self):
    # [FUNCTION DOC]
    # Test: a freshly created empty tensor map reports a size of zero.
    m = map_ops.empty_tensor_map()
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 0)

  def testTensorMapInsert(self):
    # [FUNCTION DOC]
    # Test: inserting one scalar key and scalar value into an empty map yields a map of size one.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 1)

  def testTensorMapLookup(self):
    # [FUNCTION DOC]
    # Test: after inserting a scalar key and value, looking up that key with the matching float type
    # returns a value close to the inserted value.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)
    l = map_ops.tensor_map_lookup(m, k, dtypes.float32)
    self.assertAllClose(l, v)

  def testTensorMapLookupFromEmptyMapFails(self):
    # [FUNCTION DOC]
    # Test: looking up a key in an empty map must raise an invalid-argument error whose message says the
    # key to look up does not exist. The error is expected when the lookup result is evaluated.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Trying to lookup non-existent key. *"):
      l = map_ops.tensor_map_lookup(m, k, dtypes.float32)
      self.evaluate(l)

  def testTensorMapLookupMissingKeyFails(self):
    # [FUNCTION DOC]
    # Test: with one key inserted, looking up a different key must raise an invalid-argument error
    # saying the key does not exist; the error is expected on evaluating the lookup result.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    k2 = constant_op.constant(2.0)
    v = constant_op.constant(11.0)
    m = map_ops.tensor_map_insert(m, k, v)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Trying to lookup non-existent key. *"):
      l = map_ops.tensor_map_lookup(m, k2, dtypes.float32)
      self.evaluate(l)

  def testTensorMapErase(self):
    # [FUNCTION DOC]
    # Test: after inserting one entry the size is one; erasing that key (given the value type) produces
    # a map whose size is zero.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 1)
    m = map_ops.tensor_map_erase(m, k, v.dtype)
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 0)

  def testTensorMapEraseFromEmptyMapFails(self):
    # [FUNCTION DOC]
    # Test: erasing a key from an empty map must raise an invalid-argument error saying the item to
    # erase does not exist; the error is expected on evaluating the resulting map.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Trying to erase non-existent item. *"):
      m = map_ops.tensor_map_erase(m, k, dtypes.float32)
      self.evaluate(m)

  def testTensorMapEraseMissingKeyFails(self):
    # [FUNCTION DOC]
    # Test: erasing a key that was never inserted (another key is present) must raise an
    # invalid-argument error saying the item to erase does not exist, when the result is evaluated.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    k2 = constant_op.constant(2.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k2, v)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Trying to erase non-existent item. *"):
      m = map_ops.tensor_map_erase(m, k, dtypes.float32)
      self.evaluate(m)

  def testTensorMapHasKey(self):
    # [FUNCTION DOC]
    # Test: the has-key op returns true for an inserted key and false for a key that was not inserted.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    k2 = constant_op.constant(2.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)
    # Check has key.
    b = map_ops.tensor_map_has_key(m, k)
    b2 = map_ops.tensor_map_has_key(m, k2)
    self.assertAllEqual(b, True)
    self.assertAllEqual(b2, False)

  def testIfHasKeyLookup(self):
    # [FUNCTION DOC]
    # Test: a conditional uses the has-key result to choose between a lookup and a zero default value.
    # For a present key the result equals the stored value; for an absent key the result equals the
    # default (the lookup branch is not taken, so no missing-key error occurs).
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    k2 = constant_op.constant(2.0)
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)

    default_value = array_ops.zeros_like(v)
    l = cond.cond(
        map_ops.tensor_map_has_key(m, k),
        lambda: map_ops.tensor_map_lookup(m, k, dtypes.float32),
        lambda: default_value)
    l2 = cond.cond(
        map_ops.tensor_map_has_key(m, k2),
        lambda: map_ops.tensor_map_lookup(m, k, dtypes.float32),
        lambda: default_value)
    self.assertAllClose(l, v)
    self.assertAllClose(l2, default_value)

  def testStackKeys(self):
    # [FUNCTION DOC]
    # Test: stacking the keys of a map into one tensor. With two entries the stacked keys have the shape
    # of a two-element vector and, once sorted (map order is unspecified), equal the two keys. After a
    # third insertion the same check is repeated for three keys.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    k2 = constant_op.constant(2.0)
    k3 = constant_op.constant(3.0)
    v = constant_op.constant(21.0)
    v2 = constant_op.constant(22.0)
    v3 = constant_op.constant(23.0)
    m = map_ops.tensor_map_insert(m, k, v)
    m = map_ops.tensor_map_insert(m, k2, v2)
    keys = map_ops.tensor_map_stack_keys(m, k.dtype)
    expected = constant_op.constant([1.0, 2.0])
    self.assertAllClose(array_ops.shape(keys), array_ops.shape(expected))
    self.assertAllClose(sort_ops.sort(keys), expected)

    m = map_ops.tensor_map_insert(m, k3, v3)
    keys = map_ops.tensor_map_stack_keys(m, k.dtype)
    expected = constant_op.constant([1.0, 2.0, 3.0])
    self.assertAllClose(array_ops.shape(keys), array_ops.shape(expected))
    self.assertAllClose(sort_ops.sort(keys), expected)

  def testStackKeysEmptyMapFails(self):
    # [FUNCTION DOC]
    # Test: stacking keys of an empty map must raise an invalid-argument error stating that the stack
    # keys op cannot be called on an empty map.
    m = map_ops.empty_tensor_map()
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, "TensorMapStackKeys cannot be called "
        "on empty map."):
      keys = map_ops.tensor_map_stack_keys(m, dtypes.float32)
      self.evaluate(keys)

  def testStackKeysIncorrectDtypeFails(self):
    # [FUNCTION DOC]
    # Test: the map holds a string key but keys are requested as floats; evaluation must raise an
    # invalid-argument error saying the key does not match the requested type.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant("key_with_wrong_dtype")
    v = constant_op.constant(2.0)
    m = map_ops.tensor_map_insert(m, k, v)
    simple = "Key does not match requested dtype."
    with self.assertRaisesRegex(errors.InvalidArgumentError, simple):
      keys = map_ops.tensor_map_stack_keys(m, dtypes.float32)
      self.evaluate(keys)

  def testStackKeysIncorrectShapeFails(self):
    # [FUNCTION DOC]
    # Test: the map holds a scalar key and a vector key; stacking must raise an invalid-argument error
    # saying that all keys must have the same shape.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant(1.0)
    k2 = constant_op.constant([1.0, 11.0])
    v = constant_op.constant(2.0)
    v2 = constant_op.constant(22.0)
    m = map_ops.tensor_map_insert(m, k, v)
    m = map_ops.tensor_map_insert(m, k2, v2)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Keys must all have the same shape."):
      keys = map_ops.tensor_map_stack_keys(m, dtypes.float32)
      self.evaluate(keys)

  def testInsertLookupGrad(self):
    # [FUNCTION DOC]
    # Test: gradient through insert and lookup. Inside a gradient tape the value is watched, inserted,
    # looked up and multiplied by five; the gradient of the result with respect to the value must be five.
    with backprop.GradientTape() as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      v = constant_op.constant(11.0)
      tape.watch(v)
      m = map_ops.tensor_map_insert(m, k, v)
      l = map_ops.tensor_map_lookup(m, k, dtypes.float32)
      l *= 5
      g = tape.gradient(l, v)
      self.assertAllEqual(g, 5)

  def testMultipleInsertLookupGrad(self):
    # [FUNCTION DOC]
    # Test: with a persistent gradient tape, three distinct key/value pairs are inserted and looked up;
    # the lookups are scaled by five, six and seven, and each gradient with respect to its own value must
    # equal its scale factor. The tape is deleted at the end to release resources.
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      k3 = constant_op.constant(3.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(12.0)
      v3 = constant_op.constant(13.0)
      tape.watch(v)
      tape.watch(v2)
      tape.watch(v3)
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v2)
      m = map_ops.tensor_map_insert(m, k3, v3)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      l3 = map_ops.tensor_map_lookup(m, k3, v3.dtype)
      g = tape.gradient(l * 5, v)
      g2 = tape.gradient(l2 * 6, v2)
      g3 = tape.gradient(l3 * 7, v3)
      self.assertAllEqual(g, 5)
      self.assertAllEqual(g2, 6)
      self.assertAllEqual(g3, 7)
    del tape

  def testInsertLookupComposeGrad(self):
    # [FUNCTION DOC]
    # Test: gradient through chained map operations. A looked-up value is inserted under a second key and
    # looked up again, then scaled by five; the gradient with respect to the original value must be five.
    with backprop.GradientTape() as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      tape.watch(v)
      m = map_ops.tensor_map_insert(m, k, v)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      m = map_ops.tensor_map_insert(m, k2, l)
      l2 = map_ops.tensor_map_lookup(m, k2, l.dtype)
      g = tape.gradient(l2 * 5, v)
      self.assertAllEqual(g, 5)

  def testReplaceLookupGrad(self):
    # [FUNCTION DOC]
    # Test: replacing the value stored under an existing key. After the first insert the gradient with
    # respect to the first value is five. After the key is overwritten with a second value, the lookup
    # equals the second value, the gradient with respect to the first value is zero, and the gradient
    # with respect to the second value equals the scale factor seven. Persistent tape, deleted at the end.
    # Test using same key and different value.
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      self.assertAllClose(l, v)
      g = tape.gradient(l * 5, v)
      self.assertAllEqual(g, 5)
      # Replace key and lookup.
      m = map_ops.tensor_map_insert(m, k, v2)
      l2 = map_ops.tensor_map_lookup(m, k, v2.dtype)
      self.assertAllClose(l2, v2)
      g2 = tape.gradient(l2 * 6, v)
      self.assertAllClose(g2, array_ops.zeros_like(v))
      g3 = tape.gradient(l2 * 7, v2)
      self.assertAllClose(g3, 7)
    del tape

  def testDiffKeySameValueGrad(self):
    # [FUNCTION DOC]
    # Test: the same value inserted under two different keys. The gradient of the sum of both lookups with
    # respect to that value is two. After one key is overwritten with a second value, the gradient of the
    # sum with respect to the second value is one. Persistent tape, deleted at the end.
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(11.0)
      v = constant_op.constant(2.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      l2 = map_ops.tensor_map_lookup(m, k2, v.dtype)
      g = tape.gradient(l + l2, v)
      self.assertAllEqual(g, 2)
      m = map_ops.tensor_map_insert(m, k2, v2)
      l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      g2 = tape.gradient(l + l2, v2)
      self.assertAllEqual(g2, 1)
    del tape

  def testLookupAddGrad(self):
    # [FUNCTION DOC]
    # Test: gradients of sums of lookups. The gradient of the sum of two lookups with respect to each
    # lookup and with respect to each original value is one; scaling one lookup by four makes the gradient
    # for that value four. Persistent tape, deleted at the end.
    with backprop.GradientTape(persistent=True) as tape:
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.empty_tensor_map()
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v2)
      l1 = map_ops.tensor_map_lookup(m, k, v.dtype)
      l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      g = tape.gradient(l1 + l2, [l1, l2])
      self.assertAllClose(g, [1, 1])
      g2 = tape.gradient(l1 + l2, [v, v2])
      self.assertAllClose(g2, [1, 1])
      g3 = tape.gradient(l1 + l2 * 4, v2)
      self.assertAllEqual(g3, 4)
    del tape

  def testLookupMultiplyGrad(self):
    # [FUNCTION DOC]
    # Test: gradients of products of lookups. The gradient of the product of two lookups with respect to
    # the two values equals the other value in each case; the gradient of a lookup squared equals twice
    # the value. Persistent tape, deleted at the end.
    with backprop.GradientTape(persistent=True) as tape:
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.empty_tensor_map()
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v2)
      l1 = map_ops.tensor_map_lookup(m, k, v.dtype)
      l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      g = tape.gradient(l1 * l2, [v, v2])
      self.assertAllClose(g, [v2, v])
      g2 = tape.gradient(l1 * l1, v)
      self.assertAllClose(g2, 2 * v)
    del tape

  def testEraseFirstInsertGrad(self):
    # [FUNCTION DOC]
    # Test: gradients when the first inserted key is erased. A value is looked up before erasure, then a
    # second entry is inserted and the first key erased. The remaining lookup equals the second value, the
    # earlier lookup still has gradient five with respect to the first value, and the remaining lookup has
    # gradient six. Persistent tape, deleted at the end.
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      m = map_ops.tensor_map_insert(m, k2, v2)
      m = map_ops.tensor_map_erase(m, k, v.dtype)
      l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      self.assertAllClose(l2, v2)
      g = tape.gradient(l * 5, v)
      self.assertAllEqual(g, 5)
      g2 = tape.gradient(l2 * 6, v2)
      self.assertAllEqual(g2, 6)
    del tape

  def testEraseSecondInsertGrad(self):
    # [FUNCTION DOC]
    # Test: gradients when the second inserted key is erased. After erasing it, the first key's lookup
    # still equals the first value and its gradient with respect to the first value is five.
    # Persistent tape, deleted at the end.
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v2)
      m = map_ops.tensor_map_erase(m, k2, v2.dtype)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      self.assertAllClose(l, v)
      g = tape.gradient(l * 5, v)
      self.assertAllEqual(g, 5)
    del tape

  def testEraseInsertComposedGrad(self):
    # [FUNCTION DOC]
    # Test: gradient through lookup, erase and re-insert. A value is looked up, its key erased, and the
    # looked-up tensor inserted under a new key then looked up again and scaled by five; the gradient with
    # respect to the original value must be five. Persistent tape, deleted at the end.
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant(1.0)
      k2 = constant_op.constant(2.0)
      v = constant_op.constant(11.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      m = map_ops.tensor_map_erase(m, k, v.dtype)
      m = map_ops.tensor_map_insert(m, k2, l)
      l2 = map_ops.tensor_map_lookup(m, k2, l.dtype)
      g = tape.gradient(l2 * 5, v)
      self.assertAllEqual(g, 5)
    del tape

  def testStringKeyGrad(self):
    # [FUNCTION DOC]
    # Test: a map with string keys and float values under a persistent tape. Checks size two after two
    # inserts, lookup and gradient (five), replacing a key's value and gradient (six), then erasing a key:
    # size becomes one, has-key is false for the erased key, and the remaining entry's gradient is six.
    # The tape is deleted at the end.
    with backprop.GradientTape(persistent=True) as tape:
      m = map_ops.empty_tensor_map()
      k = constant_op.constant("key")
      k2 = constant_op.constant("key2")
      v = constant_op.constant(2.0)
      v2 = constant_op.constant(22.0)
      tape.watch(v)
      tape.watch(v2)
      m = map_ops.tensor_map_insert(m, k, v)
      m = map_ops.tensor_map_insert(m, k2, v2)
      s = map_ops.tensor_map_size(m)
      self.assertAllEqual(s, 2)
      # Test lookup and gradient.
      l = map_ops.tensor_map_lookup(m, k, v.dtype)
      self.assertAllClose(l, v)
      self.assertAllClose(tape.gradient(l * 5, v), 5)
      # Test replace and gradient.
      m = map_ops.tensor_map_insert(m, k, v2)
      l2 = map_ops.tensor_map_lookup(m, k, v2.dtype)
      self.assertAllClose(l2, v2)
      g = tape.gradient(l2 * 6, v2)
      self.assertAllEqual(g, 6)
      # Test erase, has key, and gradient.
      m = map_ops.tensor_map_erase(m, k, v2.dtype)
      s = map_ops.tensor_map_size(m)
      self.assertAllEqual(s, 1)
      h = map_ops.tensor_map_has_key(m, k)
      self.assertAllEqual(h, False)
      l = map_ops.tensor_map_lookup(m, k2, v2.dtype)
      g2 = tape.gradient(l * 6, v2)
      self.assertAllEqual(g2, 6)
    del tape

  def testStringKeyValue(self):
    # [FUNCTION DOC]
    # Test: a map mixing a string key with a string value and a float key with a float value. Both lookups
    # return their stored values; erasing the string key makes has-key false for it while the float key
    # remains present.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant("key")
    v = constant_op.constant("value")
    k2 = constant_op.constant(1.0)
    v2 = constant_op.constant(2.0)
    # Test insert and lookup on string key-value pair.
    m = map_ops.tensor_map_insert(m, k, v)
    m = map_ops.tensor_map_insert(m, k2, v2)
    l = map_ops.tensor_map_lookup(m, k, v.dtype)
    self.assertAllEqual(l, v)
    # Test lookup on float key-value pair.
    l2 = map_ops.tensor_map_lookup(m, k2, v2.dtype)
    self.assertAllClose(l2, v2)
    # Test erase and has key.
    self.assertAllEqual(map_ops.tensor_map_has_key(m, k), True)
    m = map_ops.tensor_map_erase(m, k, v.dtype)
    self.assertAllEqual(map_ops.tensor_map_has_key(m, k), False)
    self.assertAllEqual(map_ops.tensor_map_has_key(m, k2), True)

  def testVectorValue(self):
    # [FUNCTION DOC]
    # Test: a map with a vector key and a vector value. Insert gives size one and lookup returns the
    # vector; after erasing the key the size is zero and has-key is false.
    m = map_ops.empty_tensor_map()
    k = constant_op.constant([1.0, 2.0])
    v = constant_op.constant([11.0, 22.0])
    # Test insert and lookup.
    m = map_ops.tensor_map_insert(m, k, v)
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 1)
    l = map_ops.tensor_map_lookup(m, k, v.dtype)
    self.assertAllEqual(l, v)
    # Test erase and has key.
    m = map_ops.tensor_map_erase(m, k, v.dtype)
    s = map_ops.tensor_map_size(m)
    self.assertAllEqual(s, 0)
    self.assertAllEqual(map_ops.tensor_map_has_key(m, k), False)

if __name__ == "__main__":
  test.main()
