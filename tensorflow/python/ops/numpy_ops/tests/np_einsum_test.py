# [MODULE DOC]
# Test module comparing the TF NumPy einsum implementation against real NumPy einsum. It holds a
# single test class with many cases covering one, two and three operand contractions, repeated
# indices (diagonals and traces), ellipsis broadcasting, cases that TF einsum does not natively
# support, a parameterized set of equations taken from the dask test suite across several dtypes,
# and a check of the einsum path helper. Some cases are disabled via the test-util disable marker.
# Import-time side effects: parses absl flags through the local test config; running the file
# directly starts the absl test runner.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
from collections import defaultdict  # pylint: disable=g-importing-member
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
from tensorflow.python.ops.numpy_ops.tests.config import config
import tensorflow.python.ops.numpy_ops.tests.np_wrapper as tnp
import tensorflow.python.ops.numpy_ops.tests.test_util as tntu


config.parse_flags_with_absl()


class EinsumTest(tntu.TestCase):
  # [CLASS DOC]
  # Test case class covering the TF NumPy einsum against NumPy's einsum. Inherits the shared test
  # utilities (seeded random generator, tolerant array comparison). Shared fixture: the check
  # helper that evaluates both implementations and compares results with dtype checking.

  def _check(self, s, *ops):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the einsum tests. Evaluates the subscript string s with the
    # given operand arrays using NumPy and the TF NumPy implementation and asserts the results are
    # close (tolerance 1e-4 absolute and relative) including matching dtypes.
    a = np.einsum(s, *ops)
    b = tnp.einsum(s, *ops)
    self.assertAllClose(a, b, check_dtypes=True, atol=1e-4, rtol=1e-4)

  def test_three_operands_1(self):
    # [FUNCTION DOC]
    # Scenario: three vectors of lengths 3, 4, 5 with random values, outer product 'i,j,k->ijk'.
    # Expects the TF NumPy result to match NumPy.
    r = self.rng()
    x = r.randn(3)
    y = r.randn(4)
    z = r.randn(5)
    s = 'i,j,k->ijk'
    self._check(s, x, y, z)

  def test_three_operands_2(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: identical to the first three-operand test (same shapes and equation), so it adds
    # no extra coverage.
    # Scenario: outer product of vectors of lengths 3, 4, 5 via 'i,j,k->ijk'; expects a match
    # with NumPy.
    r = self.rng()
    x = r.randn(3)
    y = r.randn(4)
    z = r.randn(5)
    s = 'i,j,k->ijk'
    self._check(s, x, y, z)

  def test_two_operands_1(self):
    # [FUNCTION DOC]
    # Scenario: matrix (3 by 4) times vector (4) with 'ij,j->i'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 4)
    y = r.randn(4)
    s = 'ij,j->i'
    self._check(s, x, y)

  def test_two_operands_2(self):
    # [FUNCTION DOC]
    # Scenario: rank-3 array (3,4,5) and vector (4) with 'ijk,j->i', where index k is summed out
    # of the first operand alone. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 4, 5)
    y = r.randn(4)
    s = 'ijk,j->i'
    self._check(s, x, y)

  def test_two_operands_3(self):
    # [FUNCTION DOC]
    # Scenario: array (3,4,3) with a repeated index and vector (3), equation 'iji,i->j'. Expects a
    # match with NumPy.
    r = self.rng()
    x = r.randn(3, 4, 3)
    y = r.randn(3)
    s = 'iji,i->j'
    self._check(s, x, y)

  def test_two_operands_4(self):
    # [FUNCTION DOC]
    # Scenario: two (3,4) matrices with 'ij,ij->', a full contraction to a scalar. Expects a match
    # with NumPy.
    r = self.rng()
    x = r.randn(3, 4)
    y = r.randn(3, 4)
    s = 'ij,ij->'
    self._check(s, x, y)

  def test_two_operands_5(self):
    # [FUNCTION DOC]
    # Scenario: batched matrix product, (10,2,3) with (3,4) using 'nij,jk->nik'. Expects a match
    # with NumPy.
    r = self.rng()
    x = r.randn(10, 2, 3)
    y = r.randn(3, 4)
    s = 'nij,jk->nik'
    self._check(s, x, y)

  def test_two_operands_6(self):
    # [FUNCTION DOC]
    # Scenario: regression case from a JAX issue: (2,1) with (2,3,4) using 'sa,shb->shab', mixing
    # a shared batch index with free indices. Expects a match with NumPy.
    # based on https://github.com/google/jax/issues/37#issuecomment-448572187
    r = self.rng()
    x = r.randn(2, 1)
    y = r.randn(2, 3, 4)
    s = 'sa,shb->shab'
    self._check(s, x, y)

  def test_one_operand_1(self):
    # [FUNCTION DOC]
    # Scenario: single (3,4,5) operand reduced with 'ijk->j'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->j'
    self._check(s, x)

  def test_one_operand_2(self):
    # [FUNCTION DOC]
    # Scenario: single (3,4,5) operand permuted with 'ijk->kij'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->kij'
    self._check(s, x)

  def test_one_operand_3(self):
    # [FUNCTION DOC]
    # Scenario: single (3,4,5) operand, transpose plus reduction with 'ijk->ki'. Expects a match
    # with NumPy.
    r = self.rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->ki'
    self._check(s, x)

  def test_one_operand_4(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: duplicate of the previous single-operand test (same shape and equation).
    # Scenario: single (3,4,5) operand with 'ijk->ki'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 4, 5)
    s = 'ijk->ki'
    self._check(s, x)

  def test_one_operand_5(self):
    # [FUNCTION DOC]
    # Scenario: (2,3,4,5) operand with leading ellipsis, '...ijk->...ki'. Expects a match with
    # NumPy.
    r = self.rng()
    x = r.randn(2, 3, 4, 5)
    s = '...ijk->...ki'
    self._check(s, x)

  def test_one_operand_6(self):
    # [FUNCTION DOC]
    # Scenario: (3,4,5) operand with an ellipsis that is summed away, '...ijk->ki'. Expects a
    # match with NumPy.
    r = self.rng()
    x = r.randn(3, 4, 5)
    s = '...ijk->ki'
    self._check(s, x)

  def test_one_operand_7(self):
    # [FUNCTION DOC]
    # Scenario: trace of a (3,3) matrix with 'ii->'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 3)
    s = 'ii->'
    self._check(s, x)

  def test_one_operand_8(self):
    # [FUNCTION DOC]
    # Scenario: sum of all entries of a (3,3) matrix with 'ij->'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 3)
    s = 'ij->'
    self._check(s, x)

  def test_one_operand_9(self):
    # [FUNCTION DOC]
    # Scenario: triple-repeated index on a (3,3,3) array, 'iii->' (sum of the main diagonal).
    # Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 3, 3)
    s = 'iii->'
    self._check(s, x)

  def test_one_operand_10(self):
    # [FUNCTION DOC]
    # Scenario: extract the diagonal of a (3,3) matrix with 'ii->i'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 3)
    s = 'ii->i'
    self._check(s, x)

  def test_one_operand_11(self):
    # [FUNCTION DOC]
    # Scenario: partial diagonal with a summed index on a (3,3,4) array, 'iij->i'. Expects a match
    # with NumPy.
    r = self.rng()
    x = r.randn(3, 3, 4)
    s = 'iij->i'
    self._check(s, x)

  def test_one_operand_12(self):
    # [FUNCTION DOC]
    # Scenario: main diagonal of a (3,3,3) array with 'iii->i'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 3, 3)
    s = 'iii->i'
    self._check(s, x)

  def test_one_operand_13(self):
    # [FUNCTION DOC]
    # Scenario: two separate repeated-index pairs on a (3,3,5,4,4) array, 'iijkk->i'. Expects a
    # match with NumPy.
    r = self.rng()
    x = r.randn(3, 3, 5, 4, 4)
    s = 'iijkk->i'
    self._check(s, x)

  def test_one_operand_14(self):
    # [FUNCTION DOC]
    # Scenario: same (3,3,5,4,4) array with output keeping two diagonals, 'iijkk->ik'. Expects a
    # match with NumPy.
    r = self.rng()
    x = r.randn(3, 3, 5, 4, 4)
    s = 'iijkk->ik'
    self._check(s, x)

  def test_one_operand_15(self):
    # [FUNCTION DOC]
    # Scenario: (3,3,5,4,4) array, one diagonal kept and free indices of different sizes,
    # 'iijkl->il'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 3, 5, 4, 4)
    s = 'iijkl->il'
    self._check(s, x)

  def test_one_operand_16(self):
    # [FUNCTION DOC]
    # Scenario: identity subscript 'ij->ij' on a (3,3) matrix. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 3)
    s = 'ij->ij'
    self._check(s, x)

  def test_tf_unsupported_1(self):
    # [FUNCTION DOC]
    # Scenario: an equation listed in the TF einsum documentation as unsupported by TF (ellipsis
    # in the middle of two operands), 'ij...,jk...->ik...' with (2,3,5,1) and (3,4,5,1) inputs.
    # Expects the TF NumPy version to still match NumPy.
    # from https://www.tensorflow.org/api_docs/python/tf/einsum
    r = self.rng()
    x = r.randn(2, 3, 5, 1)
    y = r.randn(3, 4, 5, 1)
    s = 'ij...,jk...->ik...'
    self._check(s, x, y)

  def test_tf_unsupported_2(self):
    # [FUNCTION DOC]
    # Scenario: documented TF-unsupported case with a repeated index in one operand and an
    # unrelated second operand, 'ijj,k->ik' with (2,3,3) and (4). Expects a match with NumPy.
    # from https://www.tensorflow.org/api_docs/python/tf/einsum
    r = self.rng()
    x = r.randn(2, 3, 3)
    y = r.randn(4)
    s = 'ijj,k->ik'
    self._check(s, x, y)

  def test_tf_unsupported_3(self):
    # [FUNCTION DOC]
    # Scenario: documented TF-unsupported case where an index appears in three operands,
    # 'ij,ij,jk->ik' with (2,3), (2,3), (3,4). Expects a match with NumPy.
    # from https://www.tensorflow.org/api_docs/python/tf/einsum
    r = self.rng()
    x = r.randn(2, 3)
    y = r.randn(2, 3)
    z = r.randn(3, 4)
    s = 'ij,ij,jk->ik'
    self._check(s, x, y, z)

  # these tests are based on https://github.com/dask/dask/pull/3412/files
  @parameterized.named_parameters(
      {'testcase_name': '_{}_dtype={}'.format(einstr, dtype.__name__),  # pylint: disable=g-complex-comprehension
       'einstr': einstr, 'dtype': dtype}
      for einstr in [
          'abc,bad->abcd',
          'abcdef,bcdfg->abcdeg',
          'ea,fb,abcd,gc,hd->efgh',
          'ab,b',
          'aa',
          'a,a->',
          'a,a->a',
          'a,a',
          'a,b',
          'a,b,c',
          'a',
          'ba,b',
          'ba,b->',
          'defab,fedbc->defac',
          'ab...,bc...->ac...',
          'a...a',
          'abc...->cba...',
          '...ab->...a',
          'a...a->a...',
          # Following 2 from # https://stackoverflow.com/a/19203475/1611416
          '...abc,...abcd->...d',
          'ab...,b->ab...',
          # https://github.com/dask/dask/pull/3412#discussion_r182413444
          'aa->a',
          'ab,ab,c->c',
          'aab,bc->ac',
          'aab,bcc->ac',
          'fdf,cdd,ccd,afe->ae',
          'fff,fae,bef,def->abd',
      ]
      # TODO(wangpeng): Add tnp.bool_ to dtype list
      for dtype in [tnp.float32, tnp.int32, tnp.complex64])
  def test_from_dask(self, einstr, dtype):
    # [FUNCTION DOC]
    # Parameterized over a list of equations borrowed from the dask test suite and the dtypes
    # float32, int32 and complex64 (bool is a noted TODO). Builds operand shapes by assigning each
    # index letter a size cycling through 2, 3, 4 (ellipsis dimensions get two extra letters),
    # generates random operands of the dtype and expects TF NumPy einsum to match NumPy.
    r = tntu.rand_default()
    if '->' in einstr:
      input_str, _ = einstr.split('->')
    else:
      input_str = einstr
    input_names = input_str.split(',')

    dims = itertools.cycle([2, 3, 4])
    shapes = defaultdict(lambda: next(dims))
    input_shapes = [tuple(shapes[c] for c in names.replace('...', '01'))
                    for names in input_names]
    operands = [r(shape, dtype) for shape in input_shapes]

    self._check(einstr, *operands)

  def test_ordered_front_batch_dim_case(self):
    # [FUNCTION DOC]
    # Scenario: two all-ones (1,8,20,4) arrays with a leading batch index of size one,
    # 'ijkl,ijml->ijkm'. Expects a match with NumPy.
    x = np.ones((1, 8, 20, 4))
    y = np.ones((1, 8, 20, 4))
    s = 'ijkl,ijml->ijkm'
    self._check(s, x, y)

  # pylint: disable=invalid-name
  def test_einsum_path(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name it exercises only NumPy's own einsum path function, never the
    # TF NumPy one, so it does not test this library.
    # Scenario: examples from the NumPy einsum path documentation with a greedy strategy. Asserts
    # the first path element is the expected contraction list and the report starts with the
    # expected line; also checks a larger call does not crash.
    # just check examples from np.einsum_path docstring
    a = self.rng().rand(2, 2)
    b = self.rng().rand(2, 5)
    c = self.rng().rand(5, 2)

    path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
    self.assertEqual(str(path_info[0]), "['einsum_path', (1, 2), (0, 1)]")
    self.assertEqual(path_info[1].split('\n')[0],
                     '  Complete contraction:  ij,jk,kl->il')

    # check this doesn't crash
    I = self.rng().rand(10, 10, 10, 10)
    C = self.rng().rand(10, 10)
    np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C, optimize='greedy')

  @tntu.disable
  def test_einsum_kpmurphy_example(self):
    # [FUNCTION DOC]
    # Disabled test. Scenario: three-operand contraction 'ntk,kd,dc->nc' from an email example;
    # computes the expected result with explicit loops, obtains an optimal path from the TF NumPy
    # path function and asserts the TF NumPy einsum using that path matches (looser tolerance on
    # TPU, dtype checks off).
    # code from an email with @murphyk
    N = 2
    C = 3
    D = 4
    K = 5
    T = 6
    r = self.rng()
    S = r.randn(N, T, K)
    W = r.randn(K, D)
    V = r.randn(D, C)
    L = np.zeros((N, C))
    for n in range(N):
      for c in range(C):
        s = 0
        for d in range(D):
          for k in range(K):
            for t in range(T):
              s += S[n, t, k] * W[k, d] * V[d, c]
        L[n, c] = s

    path = tnp.einsum_path('ntk,kd,dc->nc', S, W, V, optimize='optimal')[0]
    rtol = 1e-2 if tntu.device_under_test() == 'tpu' else None
    self.assertAllClose(L, tnp.einsum('ntk,kd,dc->nc', S, W, V, optimize=path),
                        check_dtypes=False, rtol=rtol)
  # pylint: enable=invalid-name

  @tntu.disable
  def test_contraction_broadcasting(self):
    # [FUNCTION DOC]
    # Disabled test. Scenario: contraction index of size 5 versus 1 broadcast, shapes (3,4,5) and
    # (3,1,6) with 'cij,cjk->cik'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(3, 4, 5)
    y = r.randn(3, 1, 6)
    s = 'cij,cjk->cik'
    self._check(s, x, y)

  @tntu.disable
  def test_batch_broadcasting(self):
    # [FUNCTION DOC]
    # Disabled test. Scenario: batch dimension of size 1 versus 3, shapes (1,4,5) and (3,5,6) with
    # 'cij,cjk->cik'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(1, 4, 5)
    y = r.randn(3, 5, 6)
    s = 'cij,cjk->cik'
    self._check(s, x, y)

  @tntu.disable
  def test_batch_and_contraction_broadcasting(self):
    # [FUNCTION DOC]
    # Disabled test. Scenario: both batch and contraction dimensions need broadcasting, shapes
    # (1,4,5) and (3,1,6) with 'cij,cjk->cik'. Expects a match with NumPy.
    r = self.rng()
    x = r.randn(1, 4, 5)
    y = r.randn(3, 1, 6)
    s = 'cij,cjk->cik'
    self._check(s, x, y)

  @tntu.disable
  def test_broadcasting_issue_2189(self):
    # [FUNCTION DOC]
    # Disabled test (regression for a JAX issue). Scenario: implicit-output ellipsis equation
    # '...ij,...j' with shapes (2,1,3,3) and (2,4,3). Expects a match with NumPy.
    r = self.rng()
    x = r.randn(2, 1, 3, 3)
    y = r.randn(2, 4, 3)
    s = '...ij,...j'
    self._check(s, x, y)


if __name__ == '__main__':
  absltest.main()
