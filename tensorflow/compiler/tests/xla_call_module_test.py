# [MODULE DOC]
# Tests for the XLA call-module op wrapper, which embeds a serialized StableHLO module in a
# TensorFlow computation and runs it through XLA. Scenarios include simple math modules, tokens,
# multiple arguments and results, shape polymorphism (dynamic dimensions, dynamic iota, reshape,
# gather, slice, update-slice, broadcast, reduce, calls, while loops, and more), multi-platform
# modules with a platform index argument, platform-related error checks, shape assertions, and
# graph construction. Test modules are written as StableHLO text, serialized to a portable
# artifact at the minimum supported version and run with jit compilation on the XLA test device.
# No import-time side effects other than imports; it is run via the standard test main entry.
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
"""Tests for XLA call module op wrapper."""

import os
import re
from typing import Optional, Sequence
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.mlir.stablehlo import stablehlo
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


def serialize(module_str: str) -> tuple[str, int]:
  # [FUNCTION DOC]
  # HELPER: converts StableHLO text into the serialized form consumed by the call-module op.
  # Takes the textual module and serializes it as a portable artifact targeting the minimum
  # supported StableHLO version. Returns a pair: the serialized bytes (as a string) and the maximum
  # call-module version supported by the op wrapper. May raise if the module text is invalid.
  target = stablehlo.get_minimum_version()
  byte_str = stablehlo.serialize_portable_artifact_str(module_str, target)
  return byte_str, xla.call_module_maximum_supported_version()


class XlaCallModuleOpTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the XLA call-module op. Combines the XLA test base (device selection and test
  # scope) with parameterized test support. Holds a class-level StableHLO text module used by the
  # platform-error helper tests. Tests are run on the XLA test device with jit compilation.

  def _assertOpOutputMatchesExpected(self,
                                     op,
                                     args,
                                     expected,
                                     equality_fn=None):
    # [FUNCTION DOC]
    # HELPER: common assertion for the tests in this class.
    # Inside the XLA test scope, wraps the given callable in a jit-compiled function without
    # autograph, calls it with the given argument sequence, and compares the result with the
    # expected value using the equality function (default: approximate closeness with relative
    # tolerance of 1e-3). Errors from the op propagate to the caller. Returns nothing.
    """Asserts op(*args) == expected."""
    with self.test_scope():
      tf_func = def_function.function(op, autograph=False, jit_compile=True)
      result = tf_func(*args)

      if not equality_fn:
        equality_fn = self.assertAllClose
      equality_fn(result, expected, rtol=1e-3)

  def testing_platform(self):
    # [FUNCTION DOC]
    # HELPER: maps the XLA test device name to a platform name used by the call-module op.
    # Returns the string CPU, ROCM or CUDA (for GPU devices, depending on whether the build uses
    # ROCm), or TPU. Fails an assertion for any other device name. Reads the test's device name.
    """Current testing platform, one of CPU, GPU, TPU."""
    if self.device in ['CPU', 'XLA_CPU']:
      return 'CPU'
    elif self.device in ['GPU', 'XLA_GPU']:
      if test.is_built_with_rocm():
        return 'ROCM'
      else:
        return 'CUDA'
    elif self.device in ['TPU', 'XLA_TPU']:
      return 'TPU'
    else:
      assert False, f'Unexpected {self.device=}'

  def test_basic(self):
    # [FUNCTION DOC]
    # Runs a StableHLO module computing sine of cosine on a float vector of three elements and
    # asserts the result matches the same computation done with numpy.
    x = np.array([1., 2., 3.], dtype=np.float32)

    def f(x):
      # [FUNCTION DOC]
      # Serializes the sine-of-cosine module and invokes the call-module op on the input with the
      # current platform; returns the op's output.
      # sin(cos(x))
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.cosine %arg0 : tensor<3xf32>
    %1 = stablehlo.sine %0 : tensor<3xf32>
    return %1 : tensor<3xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module, Tout=[x.dtype], Sout=[x.shape],
                             platforms=[self.testing_platform()])

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_basic_with_token_v8(self):
    # [FUNCTION DOC]
    # Same sine-of-cosine computation but the module takes and returns a leading token and the op
    # is invoked with version 8 and the token input/output flag set. Asserts the numeric output
    # matches numpy.
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def f(x):
      # [FUNCTION DOC]
      # Serializes the token-carrying module and calls the op using version 8 with the token flag;
      # returns the op's output.
      # sin(cos(x))
      module, _ = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: !stablehlo.token, %arg1: tensor<3xf32>) -> (!stablehlo.token, tensor<3xf32>) {
    %0 = stablehlo.cosine %arg1 : tensor<3xf32>
    %1 = stablehlo.sine %0 : tensor<3xf32>
    return %arg0, %1 : !stablehlo.token, tensor<3xf32>
  }
}
""")
      return xla.call_module(
          [x],
          version=8,  # Version 8 uses only one prefix token
          module=module,
          Tout=[x.dtype],
          Sout=[x.shape],
          has_token_input_output=True,  # Version 8 cares about this
          platforms=[self.testing_platform()],
      )

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_basic_with_multiple_tokens(self):
    # [FUNCTION DOC]
    # Runs a module with two leading token arguments (marked by an attribute) and two token
    # results plus the sine-of-cosine tensor. Asserts the numeric output matches numpy.
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def f(x):
      # [FUNCTION DOC]
      # Serializes the two-token module and calls the op with the maximum supported version;
      # returns the op's output.
      # sin(cos(x))
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: !stablehlo.token {jax.token = true}, %arg1: !stablehlo.token {jax.token = true}, %arg2: tensor<3xf32>) -> (!stablehlo.token, !stablehlo.token, tensor<3xf32>) {
    %0 = stablehlo.cosine %arg2 : tensor<3xf32>
    %1 = stablehlo.sine %0 : tensor<3xf32>
    return %arg0, %arg1, %1 : !stablehlo.token, !stablehlo.token, tensor<3xf32>
  }
}
""")
      return xla.call_module(
          [x],
          version=version,
          module=module,
          Tout=[x.dtype],
          Sout=[x.shape],
          platforms=[self.testing_platform()],
      )

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_basic_with_tokens_preceeded_by_other_args(self):
    # [FUNCTION DOC]
    # Runs a module whose first argument is an integer scalar, followed by two token arguments and
    # the tensor. The op is given a zero integer then the vector. Asserts the output matches the
    # numpy sine-of-cosine.
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def f(x):
      # [FUNCTION DOC]
      # Serializes the module and calls the op with an extra leading integer zero argument and the
      # input; returns the op's output.
      # sin(cos(x))
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<i32>, %arg1: !stablehlo.token {jax.token = true}, %arg2: !stablehlo.token {jax.token = true}, %arg3: tensor<3xf32>) -> (!stablehlo.token, !stablehlo.token, tensor<3xf32>) {
    %0 = stablehlo.cosine %arg3 : tensor<3xf32>
    %1 = stablehlo.sine %0 : tensor<3xf32>
    return %arg1, %arg2, %1 : !stablehlo.token, !stablehlo.token, tensor<3xf32>
  }
}
""")
      return xla.call_module(
          [np.int32(0), x],
          version=version,
          module=module,
          Tout=[x.dtype],
          Sout=[x.shape],
          platforms=[self.testing_platform()],
      )

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_compare(self):
    # [FUNCTION DOC]
    # Runs a module that compares an unsigned 32-bit scalar with one (greater or equal) and
    # returns a boolean. Input is two; asserts the result is true.
    x = np.uint32(2)
    res = np.bool_(True)

    def f(x):
      # [FUNCTION DOC]
      # Serializes the comparison module and calls the op, declaring a boolean scalar output;
      # returns the op's output.
      # return x >= 1
      module, version = serialize("""
module @jit_f_jax.0 {
  func.func public @main(%arg0: tensor<ui32>) -> tensor<i1> {
    %0 = stablehlo.constant dense<1> : tensor<ui32>
    %1 = "stablehlo.compare"(%arg0, %0) {compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>
    return %1 : tensor<i1>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_multiple_args_results(self):
    # [FUNCTION DOC]
    # Runs a module with two inputs (float32 vector of three, float64 vector of four) and two
    # outputs: sine of the first and cosine of the second. Asserts both match numpy.
    x = np.array([1., 2., 3.], dtype=np.float32)
    y = np.array([11., 12., 13., 14.], dtype=np.float64)

    def f(x, y):
      # [FUNCTION DOC]
      # Serializes the two-input two-output module and calls the op with both inputs, declaring
      # output types and shapes from the inputs; returns the op's outputs.
      # (sin(x), cos(y))
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<3xf32>, %arg1: tensor<4xf64>) -> (tensor<3xf32>, tensor<4xf64>) {
    %0 = stablehlo.sine %arg0 : tensor<3xf32>
    %1 = stablehlo.cosine %arg1 : tensor<4xf64>
    return %0, %1 : tensor<3xf32>, tensor<4xf64>
  }
}
""")
      return xla.call_module([x, y], version=version,
                             module=module,
                             Tout=[x.dtype, y.dtype],
                             Sout=[x.shape, y.shape],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x, y), (np.sin(x), np.cos(y)))

  # TODO(b/305813026): asan test failure for the i64 test variant.
  @parameterized.named_parameters(
      dict(testcase_name='_' + dim_var_type,
           dim_var_type=dim_var_type)
      for dim_var_type in ('i32',)
  )
  def test_poly_basic(self, *, dim_var_type: str):
    # [FUNCTION DOC]
    # Parameterized (currently only the 32-bit integer variant) test of shape polymorphism. Input
    # is a two-by-three float matrix; the module has a dynamic second dimension, and returns the
    # sine of the input plus the dimension size as a scalar of the given dimension-variable type.
    # Parameter: dim_var_type is the textual integer type name used for the dimension variable.
    # Asserts the outputs equal numpy sine and the size of the second dimension.
    x = np.arange(6, dtype=np.float32).reshape((2, 3))

    def f(x):  # x: f32[2, b]
      # [FUNCTION DOC]
      # Serializes the polymorphic module (built from a template with the dimension type filled in)
      # and calls the op declaring a partially unknown output shape; returns the op's outputs.
      # (sin(x), x.shape[1])
      module, version = serialize(f"""
module @jit_f.0 attributes {{jax.uses_shape_polymorphism = true}} {{
  func.func public @main(%arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<{dim_var_type}>) {{
    %arg0_new_i32 = "stablehlo.get_dimension_size"(%arg1) {{dimension = 1 : i64}} : (tensor<2x?xf32>) -> tensor<i32>
    %arg0_new = stablehlo.convert %arg0_new_i32 : (tensor<i32>) -> tensor<{dim_var_type}>
    %0, %1 = call @dyn_main(%arg0_new, %arg1) : (tensor<{dim_var_type}>, tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<{dim_var_type}>)
    return %0, %1 : tensor<2x?xf32>, tensor<{dim_var_type}>
  }}
  func.func private @dyn_main(%arg0: tensor<{dim_var_type}> {{jax.global_constant = "b"}}, %arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<{dim_var_type}>) {{
    %0 = stablehlo.sine %arg1 : tensor<2x?xf32>
    return %0, %arg0 : tensor<2x?xf32>, tensor<{dim_var_type}>
  }}
}}
""")
      return xla.call_module([x],
                             module=module, version=version,
                             Tout=[x.dtype, np.int32],
                             Sout=[(None, 3), ()],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(x), x.shape[1]))

  def test_wrong_actual_args_errors(self):
    # [FUNCTION DOC]
    # Builds a module expecting a float matrix with unknown rows and two columns plus an unranked
    # int32 tensor, and returns the first. Asserts the correct call works, then that wrong element
    # type for either argument, or a static dimension mismatch, each raise an invalid-argument
    # error with the expected refinement message.
    x = np.arange(6, dtype=np.float32).reshape((3, 2))
    y = np.arange(6, dtype=np.int32).reshape((2, 3))

    # x: f32[a, 2], return x
    module, version = serialize("""
module @jit_f.0 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg0: tensor<?x2xf32>, %arg1: tensor<*xi32>) -> tensor<?x2xf32> {
    return %arg0 : tensor<?x2xf32>
  }
}
""")

    def f(x, y):
      # [FUNCTION DOC]
      # Calls the op with the serialized module on both inputs, declaring the output as the first
      # input's type with an unknown row count and two columns; returns the op's output.
      return xla.call_module(
          [x, y],
          module=module,
          version=version,
          Tout=[x.dtype],
          Sout=[(None, 2)],
          platforms=[self.testing_platform()],
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (x,))

    x_bad_etype = x.astype(np.int32)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        re.escape(
            'invalid refinement for argument 0, refinement element types must'
            ' match in tensor<?x2xf32> -> tensor<3x2xi32>'
        ),
    ):
      self._assertOpOutputMatchesExpected(f, (x_bad_etype, y), (x_bad_etype,))

    y_bad_etype = y.astype(np.float32)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        re.escape(
            'invalid refinement for argument 1, refinement element types must'
            ' match in tensor<*xi32> -> tensor<2x3xf32>'
        ),
    ):
      self._assertOpOutputMatchesExpected(f, (x, y_bad_etype), (x,))

    x_bad_shape = np.arange(15, dtype=np.float32).reshape(5, 3)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        re.escape(
            'invalid refinement for argument 0, refinement dimension sizes must'
            ' match for static dimensions in tensor<?x2xf32> -> tensor<5x3xf32>'
        ),
    ):
      self._assertOpOutputMatchesExpected(f, (x_bad_shape, y), (x_bad_shape,))

  @parameterized.named_parameters(
      dict(testcase_name='_' + platform_idx_type,
           platform_idx_type=platform_idx_type)
      for platform_idx_type in ('i32', 'i64')
  )
  def test_platforms_basic(self, *, platform_idx_type: str):
    # [FUNCTION DOC]
    # Parameterized over the platform index integer type (32-bit or 64-bit). The module selects a
    # constant by platform index (two for CPU, three for GPU, four for TPU) and adds it to a scalar
    # zero. The op is given the platform list CPU, CUDA, ROCM, TPU. Asserts the result is the
    # constant for the current platform.
    x = np.float32(0.)

    #  returns x + 2. on CPU, x + 3. on GPU (CUDA or ROCM) and x + 4. on TPU
    module, version = serialize(f"""
module @jit_f.0 {{
  func.func public @main(%arg_platform_idx: tensor<{platform_idx_type}> {{jax.global_constant = "_platform_index"}}, %arg0: tensor<f32>) -> tensor<f32> {{
    %0 = stablehlo.convert %arg_platform_idx : (tensor<{platform_idx_type}>) -> tensor<i32>
    %to_add = "stablehlo.case"(%0) ({{
      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>
      stablehlo.return %cpu_val : tensor<f32>
    }}, {{
      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>
      stablehlo.return %gpu_val : tensor<f32>
    }}, {{
      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>
      stablehlo.return %tpu_val : tensor<f32>
    }}) : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.add %arg0, %to_add : tensor<f32>
    return %1 : tensor<f32>
  }}
}}
""")

    platforms = ['CPU', 'CUDA', 'ROCM', 'TPU']
    def f(x):
      # [FUNCTION DOC]
      # Calls the op on the input with the multi-platform module and platform list; returns the
      # op's scalar float output.
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[()],
                             platforms=platforms)

    expected_value = (
        x + dict(CPU=2.0, CUDA=3.0, ROCM=3.0, TPU=4.0)[self.testing_platform()]
    )
    self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

  def test_platforms_unknown_custom_call(self):
    # [FUNCTION DOC]
    # Same multi-platform selection idea, but the module has a fourth branch (for ROCM) with a
    # custom call to a non-existent target. Skipped on ROCM. On other platforms asserts the
    # unknown custom call in the unused branch does not break compilation and the result is the
    # constant for the current platform.
    # One of the platform branches ("ROCM") has custom call unknown to other
    # platforms.
    if self.testing_platform() == 'ROCM':
      raise unittest.SkipTest('Not intended for ROCM')
    x = np.float32(0.)

    #  returns x + 2. on CPU, x + 3. on GPU, and x + 4. on TPU
    module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<f32>) -> tensor<f32> {
    %to_add = "stablehlo.case"(%arg_platform_idx) ({
      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>
      stablehlo.return %cpu_val : tensor<f32>
    }, {
      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>
      stablehlo.return %gpu_val : tensor<f32>
    }, {
      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>
      stablehlo.return %tpu_val : tensor<f32>
    }, {
      %rocm_val = stablehlo.custom_call @non_existent_target(%arg0) : (tensor<f32>) -> tensor<f32>
      stablehlo.return %rocm_val : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %0 = stablehlo.add %arg0, %to_add : tensor<f32>
    return %0 : tensor<f32>
  }
}
""")

    platforms = ['CPU', 'CUDA', 'TPU', 'ROCM']
    def f(x):
      # [FUNCTION DOC]
      # Calls the op on the input with the module containing the unknown custom call and the
      # platform list; returns the op's scalar float output.
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[()],
                             platforms=platforms)

    expected_value = (
        x + dict(CPU=2.0, CUDA=3.0, TPU=4.0)[self.testing_platform()]
    )
    self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

  def test_platforms_and_poly(self):
    # [FUNCTION DOC]
    # Combines platform selection with a dynamic-shape vector module: adds a per-platform constant
    # (two CPU, three GPU, four TPU) to each element of a six-element vector via a dynamic
    # broadcast. Asserts the output equals the input plus the current platform's constant.
    x = np.arange(6, dtype=np.float32)
    #  returns x + 2. on CPU, x + 3. on GPU (CUDA or ROCM) and x + 4. on TPU

    module, version = serialize("""
module @jit_f_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<?xf32>) -> (tensor<?xf32>) {
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
    %5 = call @_wrapped_jax_export_main(%arg_platform_idx, %0, %arg0) : (tensor<i32>, tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
    return %5 : tensor<?xf32>
  }

  func.func private @_wrapped_jax_export_main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> (tensor<?xf32>) {
    %to_add = "stablehlo.case"(%arg_platform_idx) ({
      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>
      stablehlo.return %cpu_val : tensor<f32>
    }, {
      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>
      stablehlo.return %gpu_val : tensor<f32>
    }, {
      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>
      stablehlo.return %tpu_val : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.dynamic_broadcast_in_dim %to_add, %1, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<?xf32>
    return %4 : tensor<?xf32>
  }
}
""")
    platforms = ['CPU', 'CUDA', 'ROCM', 'TPU']
    def f(x):
      # [FUNCTION DOC]
      # Calls the op on the input with the polymorphic multi-platform module and platform list;
      # returns the op's output.
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[()],
                             platforms=platforms)

    expected_value = (
        x + dict(CPU=2.0, CUDA=3.0, ROCM=3.0, TPU=4.0)[self.testing_platform()]
    )
    self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

  def test_platforms_and_poly_and_tokens(self):
    # [FUNCTION DOC]
    # Like the platforms-and-polymorphism test but the module also has a token argument that is
    # passed through to a token result. Asserts the tensor output equals the input plus the
    # current platform's constant.
    x = np.arange(6, dtype=np.float32)
    #  returns x + 2. on CPU, x + 3. on GPU (CUDA or ROCM) and x + 4. on TPU

    module, version = serialize("""
module @jit_f_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg_tok: !stablehlo.token {jax.token = true}, %arg0: tensor<?xf32>) -> (!stablehlo.token, tensor<?xf32>) {
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>
    %5:2 = call @_wrapped_jax_export_main(%arg_platform_idx, %0, %arg_tok, %arg0) : (tensor<i32>, tensor<i32>, !stablehlo.token, tensor<?xf32>) -> (!stablehlo.token, tensor<?xf32>)
    return %5#0, %5#1 : !stablehlo.token, tensor<?xf32>
  }

  func.func private @_wrapped_jax_export_main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<i32> {jax.global_constant = "b"}, %arg_tok: !stablehlo.token {jax.token = true}, %arg1: tensor<?xf32>) -> (!stablehlo.token, tensor<?xf32>) {
    %to_add = "stablehlo.case"(%arg_platform_idx) ({
      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>
      stablehlo.return %cpu_val : tensor<f32>
    }, {
      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>
      stablehlo.return %gpu_val : tensor<f32>
    }, {
      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>
      stablehlo.return %tpu_val : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    %1 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.dynamic_broadcast_in_dim %to_add, %1, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
    %4 = stablehlo.add %3, %arg1 : tensor<?xf32>
    return %arg_tok, %4 : !stablehlo.token, tensor<?xf32>
  }
}
""")
    platforms = ['CPU', 'CUDA', 'ROCM', 'TPU']
    def f(x):
      # [FUNCTION DOC]
      # Calls the op on the input with the token-carrying polymorphic multi-platform module and the
      # platform list; returns the op's output.
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[()],
                             platforms=platforms)

    expected_value = (
        x + dict(CPU=2.0, CUDA=3.0, ROCM=3.0, TPU=4.0)[self.testing_platform()]
    )
    self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

  # A module used for testing errors related to use of "platforms".
  platforms_errors_module_str = """
  module @jit_f.0 {
    func.func public @main(%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>) -> tensor<f32> {
      return %arg0 : tensor<f32>
    }
  }
"""

  def platforms_errors_helper(
      self,
      *,
      module_str: str,
      platforms: Sequence[str] = ('CPU', 'CUDA', 'ROCM', 'TPU'),
      disabled_checks: Sequence[str] = (),
      expected_error: Optional[Exception] = None,
      expected_error_message: str = '',
  ):
    # [FUNCTION DOC]
    # HELPER: shared driver for the platform-error scenarios below.
    # Parameters (all keyword-only): module_str is the StableHLO text; platforms is a sequence of
    # platform names (default CPU, CUDA, ROCM, TPU); disabled_checks is a sequence of check names to
    # disable (default empty); expected_error is the error type expected, or nothing if success is
    # expected; expected_error_message is a regular expression the error text must match.
    # Serializes the module and runs the op on a float zero scalar. If no error is expected,
    # asserts the output equals the input; otherwise asserts the given error is raised with a
    # matching message. Returns nothing.
    module, version = serialize(module_str)
    x = np.float32(0.0)

    def f(x):
      # [FUNCTION DOC]
      # Calls the op with the module, platforms and disabled checks from the enclosing helper;
      # returns the op's scalar output.
      return xla.call_module(
          [x],
          version=version,
          module=module,
          Tout=[np.float32],
          Sout=[()],
          platforms=platforms,
          disabled_checks=disabled_checks,
      )

    if expected_error is None:
      self._assertOpOutputMatchesExpected(f, (x,), (x,))
    else:
      with self.assertRaisesRegex(expected_error, expected_error_message):
        self._assertOpOutputMatchesExpected(f, (x,), (x,))

  def platforms_errors_singleton_platform(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name does not start with "test", so the test runner never collects or runs
    # this scenario (the same applies to the other platforms_errors_* methods below).
    # Passes only the current platform as the platform list while the module still has a platform
    # index argument; expects an invalid-argument error about an incorrect number of arguments.
    # With singleton `platforms`, there should be no platform_index argument
    self.platforms_errors_helper(
        module_str=self.platforms_errors_module_str,
        platforms=(self.testing_platform(),),
        expected_error=errors.InvalidArgumentError,
        expected_error_message=(
            'Incorrect number of arguments passed to XlaCallModule = 1. The'
            ' module main function takes 2 arguments of which 0 platform index'
            ' arguments, 0 dimension arguments and 0 token arguments.'
        ),
    )

  def platforms_errors_no_platform_index_arg(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: not collected by the test runner (name lacks the test prefix).
    # Removes all arguments from the module's main function while keeping multiple platforms;
    # expects an invalid-argument error saying a platform index argument is missing.
    module_str = self.platforms_errors_module_str.replace(
        '%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>', ''
    )
    self.platforms_errors_helper(
        module_str=module_str,
        expected_error=errors.InvalidArgumentError,
        expected_error_message=(
            'The module should have a platform index argument but it has no '
            'arguments'
        ),
    )

  def platforms_errors_platform_index_i16(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: not collected by the test runner (name lacks the test prefix).
    # Changes the module's types to 16-bit integers so the platform index has an unsupported type;
    # expects an invalid-argument error about the required scalar 32- or 64-bit integer type.
    module_str = self.platforms_errors_module_str.replace('i32', 'i16')
    self.platforms_errors_helper(
        module_str=module_str,
        expected_error=errors.InvalidArgumentError,
        expected_error_message=(
            'Module argument at index 0 should be a 0-dimensional '
            '32-bit or 64-bit integer-tensor platform index argument '
            '.* has type tensor<i16>'
        ),
    )

  def platforms_errors_platform_index_non_scalar(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: not collected by the test runner (name lacks the test prefix).
    # Makes the platform index a one-element vector instead of a scalar; expects an
    # invalid-argument error requiring a zero-dimensional integer platform index.
    module_str = self.platforms_errors_module_str.replace(
        'tensor<i32>', 'tensor<1xi32>'
    )
    self.platforms_errors_helper(
        module_str=module_str,
        expected_error=errors.InvalidArgumentError,
        expected_error_message=(
            'Module argument at index 0 should be a 0-dimensional '
            '32-bit integer-tensor platform index argument .* has type '
            'tensor<1xi32>'
        ),
    )

  def platforms_errors_platform_index_unranked(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: not collected by the test runner (name lacks the test prefix).
    # Makes the platform index an unranked tensor; expects an invalid-argument error requiring a
    # zero-dimensional integer platform index.
    module_str = self.platforms_errors_module_str.replace(
        'tensor<i32>', 'tensor<*xi32>'
    )
    self.platforms_errors_helper(
        module_str=module_str,
        expected_error=errors.InvalidArgumentError,
        expected_error_message=(
            'Module argument at index 0 should be a 0-dimensional '
            '32-bit integer-tensor platform index argument'
        ),
    )

  def platforms_errors_different_from_current(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: not collected by the test runner (name lacks the test prefix).
    # Supplies two made-up platform names not including the current one. Reads the XLA flags
    # environment variable: if the platform check is disabled there, expects success; otherwise
    # expects a not-found error saying the current platform is not among the platforms.
    platform_check_disabled_by_flags = (
        '--tf_xla_call_module_disabled_checks=platform'
        in os.getenv('TF_XLA_FLAGS', '')
    )
    self.platforms_errors_helper(
        module_str=self.platforms_errors_module_str,
        platforms=['RANDOM_PLATFORM_1', 'RANDOM_PLATFORM_2'],
        expected_error=(
            None if platform_check_disabled_by_flags else errors.NotFoundError
        ),
        expected_error_message='current platform .* is not among the platforms'
    )

  def platforms_errors_dissabled_check(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: not collected by the test runner (name lacks the test prefix; also misspelled).
    # Uses made-up platform names but explicitly disables the platform check through the op's
    # disabled-checks argument; expects the call to succeed with no error.
    self.platforms_errors_helper(
        module_str=self.platforms_errors_module_str,
        platforms=('RANDOM_PLATFORM_1', 'RANDOM_PLATFORM_2'),
        disabled_checks=(xla.call_module_disable_check_platform(),),
        expected_error=None,
        expected_error_message='current platform .* is not among the platforms'
    )

  def platforms_errors_empty(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: not collected by the test runner (name lacks the test prefix).
    # Passes an empty platform list with the platform check disabled; expects success.
    self.platforms_errors_helper(
        module_str=self.platforms_errors_module_str,
        platforms=[],
        disabled_checks=[xla.call_module_disable_check_platform()],
        expected_error=None,
        expected_error_message='current platform .* is not among the platforms'
    )

  def test_shape_assertion_success(self):
    # [FUNCTION DOC]
    # Runs a module that asserts via a shape-assertion custom call that the first dimension equals
    # three, then returns that dimension. Input has three rows; asserts the result is three.
    x = np.ones((3, 5), dtype=np.int32)
    res = np.int32(x.shape[0])

    def f(x):  # x: f32[b, 5] and b = 3
      # [FUNCTION DOC]
      # Serializes the module with a satisfied shape assertion and calls the op; returns its output.
      # return x.shape[0]
      module, version = serialize("""
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x5xi32>) -> tensor<i32> {
    %b = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xi32>) -> tensor<i32>
    %3 = stablehlo.constant dense<3> : tensor<i32>
    %ok = stablehlo.compare  EQ, %b, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%ok) {
      error_message = "The error message",
      has_side_effect = true
    } : (tensor<i1>) -> ()
    return %b : tensor<i32>
  }

}
""")
      return xla.call_module([x,], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_shape_assertion_failure(self):
    # [FUNCTION DOC]
    # Runs a module whose shape assertion requires the first dimension to be four while the input
    # has three. Reads the XLA flags environment variable: if shape-assertion checks are disabled
    # there, asserts the call succeeds anyway; otherwise asserts an invalid-argument error whose
    # message has the formatted inputs substituted into the error template.
    x = np.ones((3, 5), dtype=np.int32)
    res = np.int32(x.shape[0])

    def f(x):  # x: f32[b, 5] and b = 3, with a constraint b == 4.
      # [FUNCTION DOC]
      # Serializes the module with a violated shape assertion and calls the op; returns its output.
      # return x.shape[0]
      module, version = serialize("""
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x5xi32>) -> tensor<i32> {
    %b = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xi32>) -> tensor<i32>
    %4 = stablehlo.constant dense<4> : tensor<i32>
    %5 = stablehlo.constant dense<5> : tensor<i32>
    %11 = stablehlo.constant dense<11> : tensor<i32>
    %ok = stablehlo.compare  EQ, %b, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%ok, %b, %4, %5, %4, %5, %4, %5, %4, %5, %4, %5, %11) {
      error_message = "Expecting {0} == {1}. Extra {2,=5}, {3}, {{0}, {4}, {5}, {6}, {7}, {11}.",
      has_side_effect = true
    } : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    return %b : tensor<i32>
  }
}
""")
      return xla.call_module([x,], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape],
                             platforms=[self.testing_platform()],)

    # This test runs as part of two targets, with and without
    # disabling shape_assertions.
    disabled_shape_assertions_check = (
        '--tf_xla_call_module_disabled_checks=shape_assertions'
        in os.getenv('TF_XLA_FLAGS', ''))
    if disabled_shape_assertions_check:
      # No error even though the constraint is false.
      self._assertOpOutputMatchesExpected(f, (x,), (res,))
    else:
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          re.escape('Expecting 3 == 4. Extra   5  , 4, {0}, 5, 4, 5, 4, 11.')):
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_invalid_shape_assertion(self):
    # [FUNCTION DOC]
    # Checks that malformed shape-assertion custom calls are rejected. Skipped when the XLA flags
    # environment variable disables shape assertions. Runs several sub-tests, each with a module
    # that is invalid in one way (condition not a boolean constant, non-static condition, side
    # effect flag false, message with a specifier but no input, non-static message input, message
    # input of wrong type) and asserts an invalid-argument error with the matching message.
    arg_i1 = np.bool_(True)
    arg_i32 = np.int32(2)
    res = arg_i32

    # This test runs as part of two targets, with and without
    # disabling shape_assertions.
    disabled_shape_assertions_check = (
        '--tf_xla_call_module_disabled_checks=shape_assertions'
        in os.getenv('TF_XLA_FLAGS', ''))
    if disabled_shape_assertions_check:
      self.skipTest('Test is N/A when shape_assertions are disabled')

    subtest_count = 1
    def one_subtest(error_msg: str, module_str: str):
      # [FUNCTION DOC]
      # HELPER: runs one invalid-module sub-test. Takes the expected error message pattern and the
      # StableHLO text. Increments a shared counter, then within a sub-test context asserts that
      # running the module with a boolean and an integer scalar raises an invalid-argument error
      # matching the pattern. Returns nothing.
      def f(*args):
        # [FUNCTION DOC]
        # Serializes the module under test and calls the op with all given arguments; returns the
        # op's output.
        module, version = serialize(module_str)
        return xla.call_module(
            list(args),
            version=version,
            module=module,
            Tout=[res.dtype],
            Sout=[res.shape],
            platforms=[self.testing_platform()],
        )

      nonlocal subtest_count
      subtest_count += 1
      with self.subTest(count=subtest_count, error_msg=error_msg):
        with self.assertRaisesRegex(errors.InvalidArgumentError, error_msg):
          self._assertOpOutputMatchesExpected(f, (arg_i1, arg_i32), (res,))

    one_subtest(
        'expects assert_what .* to be a constant of type tensor<i1>',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    %ok = stablehlo.constant dense<0> : tensor<i32>
    stablehlo.custom_call @shape_assertion(%ok) {
      error_message = "Some error",
      has_side_effect = true
    } : (tensor<i32>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

    one_subtest(
        'expects static assert_what',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    stablehlo.custom_call @shape_assertion(%arg_i1) {
      error_message = "Some error",
      has_side_effect = true
    } : (tensor<i1>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

    one_subtest(
        '`shape_assertion` custom calls must set `has_side_effect = true`.',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    %ok = stablehlo.constant dense<false> : tensor<i1>
    stablehlo.custom_call @shape_assertion(%ok) {
      error_message = "Some error",
      has_side_effect = false
    } : (tensor<i1>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

    one_subtest(
        'expects error_message .* Found specifier {0}',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    %ok = stablehlo.constant dense<false> : tensor<i1>
    stablehlo.custom_call @shape_assertion(%ok) {
      error_message = "Some error {0}",
      has_side_effect = true
    } : (tensor<i1>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

    one_subtest(
        'expects static error_message_input',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    %ok = stablehlo.constant dense<false> : tensor<i1>
    stablehlo.custom_call @shape_assertion(%ok, %arg_i32) {
      error_message = "Some error {0}",
      has_side_effect = true
    } : (tensor<i1>, tensor<i32>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

    one_subtest(
        'expects error_message_input .* to be a constant of type tensor<i32>',
        """
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {
    %ok = stablehlo.constant dense<false> : tensor<i1>
    %c = stablehlo.constant dense<2.0> : tensor<f32>
    stablehlo.custom_call @shape_assertion(%ok, %c) {
      error_message = "Some error {0}",
      has_side_effect = true
    } : (tensor<i1>, tensor<f32>) -> ()
    return %arg_i32 : tensor<i32>
  }
}
""",
    )

  def test_dynamic_iota(self):
    # [FUNCTION DOC]
    # Runs a module that builds an integer iota whose length is the dynamic first dimension of the
    # input (three by five). Asserts the output equals a numpy range of three elements.
    x = np.ones((3, 5), dtype=np.int32)
    res = np.arange(x.shape[0], dtype=np.int32)

    def f(x):  # x: f32[b, 5]
      # [FUNCTION DOC]
      # Serializes the dynamic-iota module and calls the op with an unknown-length output shape;
      # returns the op's output.
      # return np.arange(x.shape[0], dtype=np.int32)
      module, version = serialize("""
module @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x5xi32>) -> tensor<?xi32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xi32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x5xi32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x5xi32>) -> tensor<?xi32> {
    %0 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %1 = "stablehlo.dynamic_iota"(%0) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}
""")
      return xla.call_module([x,], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None,)],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_build_graph_with_any_platform(self):
    # [FUNCTION DOC]
    # Builds (without running) a graph function whose module lists only the TPU platform and
    # asserts the resulting graph definition text contains the call-module op, showing graph
    # construction works on any platform.
    """We can construct the tf.Graph on all platforms."""
    x = np.float32(0.)

    module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>) -> tensor<f32> {
    return %arg0 : tensor<f32>
  }
}
""")
    platforms = ['TPU']  # the module is compilable only on TPU
    def f(x):
      # [FUNCTION DOC]
      # Calls the op on the input with the TPU-only platform list; returns the op's output.
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[np.float32],
                             Sout=[()],
                             platforms=platforms)
    tf_graph = def_function.function(f).get_concrete_function(x).graph
    self.assertIn('XlaCallModule', str(tf_graph.as_graph_def()))

  def test_dynamic_reshape(self):
    # [FUNCTION DOC]
    # Runs a module that flattens a matrix with a dynamic first dimension (input four by three)
    # using a dynamic reshape. Asserts the output equals the flattened numpy array.
    x = np.ones((4, 3), dtype=np.float32)
    res = x.reshape((-1,))

    def f(x):  # x: f32[b, 3]
      # [FUNCTION DOC]
      # Serializes the dynamic-reshape module and calls the op; returns the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x3xf32>) -> tensor<?xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x3xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x3xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x3xf32>) -> tensor<?xf32> {
    %0 = stablehlo.constant dense<3> : tensor<i32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.dynamic_reshape %arg1, %2 : (tensor<?x3xf32>, tensor<1xi32>) -> tensor<?xf32>
    return %3 : tensor<?xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None,)],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_dynamic_gather(self):
    # [FUNCTION DOC]
    # Runs a module using a dynamic gather to take two columns from a matrix with dynamic rows
    # (input three by four of ones). Asserts the output is a three by two matrix of ones.
    x = np.ones((3, 4), dtype=np.float32)
    res = np.ones((3, 2), dtype=np.float32)

    def f(x):  # x: f32[b, 4]
      # [FUNCTION DOC]
      # Serializes the dynamic-gather module and calls the op; returns the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x4xf32>) -> tensor<?x2xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x4xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x4xf32>) -> tensor<?x2xf32>
    return %0 : tensor<?x2xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>) -> tensor<?x2xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1 = stablehlo.constant dense<0> : tensor<1xi64>
    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<2> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = "stablehlo.dynamic_gather"(%arg1, %1, %4) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], start_index_map = [1]>, indices_are_sorted = true} : (tensor<?x4xf32>, tensor<1xi64>, tensor<2xi32>) -> tensor<?x2xf32>
    return %5 : tensor<?x2xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None, 2)],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_real_dynamic_slice(self):
    # [FUNCTION DOC]
    # Runs a module that uses a real dynamic slice to extract the last row of a matrix with dynamic
    # rows (input three by four). Asserts the output equals the last row.
    x = np.ones((3, 4), dtype=np.float32)
    res = x[-1, :]

    def f(x):  # x: f32[b, 4]
      # [FUNCTION DOC]
      # Serializes the dynamic-slice module and calls the op; returns the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x4xf32>) -> tensor<4xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x4xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.constant dense<-1> : tensor<i32>
    %1 = stablehlo.add %arg0, %0 : tensor<i32>
    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<0> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %6 = stablehlo.constant dense<4> : tensor<1xi32>
    %7 = stablehlo.concatenate %5, %6, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %10 = stablehlo.constant dense<1> : tensor<2xi32>
    %11 = stablehlo.real_dynamic_slice %arg1, %4, %7, %10 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xf32>
    %12 = stablehlo.reshape %11 : (tensor<1x4xf32>) -> tensor<4xf32>
    return %12 : tensor<4xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[x.dtype],
                             Sout=[(4,)],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_dynamic_update_slice(self):
    # [FUNCTION DOC]
    # Runs a module that updates a matrix (with dynamic rows) with itself at a negative row index
    # that is normalized by adding the dynamic size. Input is a three by four matrix of ones and
    # index minus two. Asserts the output equals the input, since the update is a no-op.
    x = np.ones((3, 4), dtype=np.float32)
    idx = np.int32(-2)
    res = x   # The update should be a nop

    def f(x, idx):  # x: f32[b, 4]  idx: i32
      # [FUNCTION DOC]
      # Serializes the update-slice module and calls the op with the matrix and the index; returns
      # the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x4xf32>, %arg2: tensor<i32>) -> tensor<?x4xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x4xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1, %arg2) : (tensor<i32>, tensor<?x4xf32>, tensor<i32>) -> tensor<?x4xf32>
    return %0 : tensor<?x4xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>, %arg2: tensor<i32>) -> tensor<?x4xf32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.compare  LT, %arg2, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %2 = stablehlo.add %arg2, %arg0 : tensor<i32>
    %3 = stablehlo.select %1, %2, %arg2 : tensor<i1>, tensor<i32>
    %4 = stablehlo.constant dense<0> : tensor<i32>
    %5 = stablehlo.dynamic_update_slice %arg1, %arg1, %3, %4 : (tensor<?x4xf32>, tensor<?x4xf32>, tensor<i32>, tensor<i32>) -> tensor<?x4xf32>
    return %5 : tensor<?x4xf32>
  }
}
""")
      return xla.call_module([x, idx], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None, 4)],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x, idx), (res,))

  def test_dynamic_broadcast_in_dim(self):
    # [FUNCTION DOC]
    # Runs a module that dynamically broadcasts a matrix to match a three-dimensional tensor and
    # adds them. Inputs are a three by four and a two by three by four tensor of ones. Asserts the
    # two outputs equal the numpy broadcast and the sum.
    x = np.ones((3, 4), dtype=np.float32)
    y = np.ones((2, 3, 4), dtype=np.float32)
    res = (np.broadcast_to(x, y.shape), x + y)

    def f(x, y):  # x: f32[b, 4]  y: f32[2, b, 4]
      # [FUNCTION DOC]
      # Serializes the broadcast module and calls the op with both inputs, declaring two outputs
      # with an unknown middle dimension; returns the op's outputs.
      # return (np.broadcast_to(x, y.shape), x + y)
      module, version = serialize("""
module @jit_fun.0 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x4xf32>, %arg2: tensor<2x?x4xf32>) -> (tensor<2x?x4xf32>, tensor<2x?x4xf32>) {
    %arg0_new = "stablehlo.get_dimension_size"(%arg2) {dimension = 1 : i64} : (tensor<2x?x4xf32>) -> tensor<i32>
    %0, %1 = call @dyn_main(%arg0_new, %arg1, %arg2) : (tensor<i32>, tensor<?x4xf32>, tensor<2x?x4xf32>) -> (tensor<2x?x4xf32>, tensor<2x?x4xf32>)
    return %0, %1 : tensor<2x?x4xf32>, tensor<2x?x4xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>, %arg2: tensor<2x?x4xf32>) -> (tensor<2x?x4xf32>, tensor<2x?x4xf32>) {
    %0 = stablehlo.constant dense<2> : tensor<1xi32>
    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<4> : tensor<1xi32>
    %4 = "stablehlo.concatenate"(%0, %2, %3) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
    %5 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %4) {broadcast_dimensions = array<i64: 1, 2>} : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<2x?x4xf32>
    %6 = stablehlo.add %5, %arg2 : (tensor<2x?x4xf32>, tensor<2x?x4xf32>) -> tensor<2x?x4xf32>
    return %5, %6 : tensor<2x?x4xf32>, tensor<2x?x4xf32>
  }
}
""")
      return xla.call_module([x, y], version=version,
                             module=module,
                             Tout=[res[0].dtype, res[1].dtype],
                             Sout=[(2, None, 4), (2, None, 4)],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x, y), res)

  @unittest.skip('TODO(necula): test is flaky')
  def test_reduce(self):
    # [FUNCTION DOC]
    # Skipped unconditionally (marked flaky). Intended to run a module that sums an integer vector
    # with dynamic length and multiplies the sum by the length, expecting the numpy equivalent.
    # KNOWN ISSUE: the module text references an argument name that is not defined in the main
    # function, so it would likely be invalid even if the test were enabled.
    x = np.arange(5, dtype=np.int32)
    res = np.sum(x) * x.shape[0]

    def f(x):  # x: i32[b]
      # [FUNCTION DOC]
      # Serializes the reduce module and calls the op expecting a scalar output; returns it.
      module, version = serialize("""
module @jit_fun attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?xi32>) -> tensor<i32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg2) {dimension = 0 : i64} : (tensor<?xi32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xi32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xi32>) -> tensor<i32> {
    %0 = stablehlo.constant dense<0> : tensor<i32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [0] : (tensor<?xi32>, tensor<i32>) -> tensor<i32>
     reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {
      %4 = stablehlo.add %arg2, %arg3 : tensor<i32>
      "stablehlo.return"(%4) : (tensor<i32>) -> ()
    }
    %2 = stablehlo.multiply %1, %arg0 : tensor<i32>
    return %2 : tensor<i32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[res.shape],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_reduce_broadcast(self):
    # [FUNCTION DOC]
    # Runs a module that sums each row of a matrix with dynamic rows and reshapes the sums into a
    # column via a dynamic broadcast. Input has three rows of repeated values; asserts the output
    # column equals the row index times five.
    x = np.broadcast_to(np.arange(3, dtype=np.float32).reshape(3, 1), (3, 5))
    res = np.arange(3, dtype=np.float32).reshape(3, 1) * 5

    def f(x):  # x: f32[b, 5]
      # [FUNCTION DOC]
      # Serializes the reduce-and-broadcast module and calls the op; returns the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?x5xf32>) -> tensor<?x1xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x5xf32>) -> tensor<?x1xf32>
    return %0 : tensor<?x1xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x5xf32>) -> tensor<?x1xf32> {
    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x5xf32>, tensor<f32>) -> tensor<?xf32>
     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {
      %6 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %6 : tensor<f32>
    }
    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %3 = stablehlo.constant dense<1> : tensor<1xi32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %5 = stablehlo.dynamic_broadcast_in_dim %1, %4, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>
    return %5 : tensor<?x1xf32>
  }
}
""")
      return xla.call_module([x,], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[(None, 1)],
                             platforms=[self.testing_platform()],)

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_call(self):
    # [FUNCTION DOC]
    # Runs a module where the main function calls a dynamic-main function which in turn calls
    # another function that creates an iota of the dynamic length. Input is five ones; asserts the
    # output is the integer range of five elements.
    """A chain of calls."""
    x = np.ones((5,), dtype=np.float32)
    res = np.arange(x.shape[0], dtype=np.int32)

    def f(x):  # x: f32[b]
      # [FUNCTION DOC]
      # Serializes the chained-call module and calls the op; returns the op's output.
      module, version = serialize("""
module @jit_fun_3 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xi32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> tensor<?xi32> {
    %0 = call @f(%arg0, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }
  func.func private @f(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> tensor<?xi32> {
    %0 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
    %1 = "stablehlo.dynamic_iota"(%0) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}
""")
      return xla.call_module([x,], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             platforms=[self.testing_platform()])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_identity(self):
    # [FUNCTION DOC]
    # Runs a polymorphic module that returns its input unchanged (five ones) and asserts the output
    # equals the input.
    x = np.ones((5,), dtype=np.float32)
    res = x

    def f(x):  # x: f32[b]
      # [FUNCTION DOC]
      # Serializes the identity module and calls the op; returns the op's output.
      module, version = serialize("""
module @jit_fun_3 attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    return %arg1 : tensor<?xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             platforms=[self.testing_platform()])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_while(self):
    # [FUNCTION DOC]
    # Runs a module with a while loop carrying a dynamically shaped vector and a 64-bit counter;
    # each of five iterations adds a dynamic iota to the vector. Input is five ones; the expected
    # results are computed in numpy first. Asserts both the final vector and the counter value five.
    """A while loop with carryied dynamic shapes."""
    x = np.ones((5,), dtype=np.float32)
    # Compute the result in Python first
    res0 = np.copy(x)
    for _ in range(5):
      res0 += np.arange(x.shape[0], dtype=np.float32)
    res1 = np.int64(5)

    def f(x):  # x: f32[b]
      # [FUNCTION DOC]
      # Serializes the while-loop module and calls the op declaring two outputs (vector with
      # unknown length, and a scalar counter); returns the op's outputs.
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>) {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>
    %0, %1 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>)
    return %0, %1 : tensor<?xf32>, tensor<i64>
  }
  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>) {
    %0 = stablehlo.constant dense<0> : tensor<i64>
    %1:2 = "stablehlo.while"(%arg1, %0) ({
    ^bb0(%arg2: tensor<?xf32>, %arg3: tensor<i64>):
      %2 = stablehlo.constant dense<5> : tensor<i64>
      %3 = stablehlo.compare  LT, %arg3, %2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %3 : tensor<i1>
    }, {
    ^bb0(%arg2: tensor<?xf32>, %arg3: tensor<i64>):
      %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>
      %3 = stablehlo.dynamic_iota %2, dim = 0 : (tensor<1xi32>) -> tensor<?xf32>
      %4 = stablehlo.add %arg2, %3 : tensor<?xf32>
      %5 = stablehlo.constant dense<1> : tensor<i64>
      %6 = stablehlo.add %arg3, %5 : tensor<i64>
      stablehlo.return %4, %6 : tensor<?xf32>, tensor<i64>
    }) : (tensor<?xf32>, tensor<i64>) -> (tensor<?xf32>, tensor<i64>)
    return %1#0, %1#1 : tensor<?xf32>, tensor<i64>
  }
}
""")
      return xla.call_module([x,], version=version,
                             module=module,
                             Tout=[res0.dtype, res1.dtype],
                             Sout=[(None,), res1.shape],
                             platforms=[self.testing_platform()])

    self._assertOpOutputMatchesExpected(f, (x,), (res0, res1))

  def test_skip_shape_refinement(self):
    # [FUNCTION DOC]
    # Checks that a module with dynamic shapes but without the shape-polymorphism attribute (first
    # missing, then explicitly false) is rejected. Asserts an invalid-argument error saying the
    # module has dynamic shapes in both cases. The attribute text is changed between the two runs
    # through a variable shared with the nested function.
    # We skipped the shape refinement, but there are dynamic shapes.
    x = np.ones((5,), dtype=np.float32)
    res = x

    module_attrs = ''  # attribute is missing
    def f(x):  # x: f32[b]
      # [FUNCTION DOC]
      # Serializes the module with whatever attribute text is currently set and calls the op;
      # returns the op's output (or raises, when the module is rejected).
      module, version = serialize(f"""
module @jit_fun_3 {module_attrs} {{
  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xf32> {{
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {{dimension = 0 : i64}} : (tensor<?xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }}
  func.func private @dyn_main(%arg0: tensor<i32> {{jax.global_constant = "b"}}, %arg1: tensor<?xf32>) -> tensor<?xf32> {{
    return %arg1 : tensor<?xf32>
  }}
}}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             platforms=[self.testing_platform()])

    module_attrs = ''  # attribute is missing
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Module has dynamic shapes'):
      self._assertOpOutputMatchesExpected(f, (x,), (res,))

    module_attrs = 'attributes {jax.uses_shape_polymorphism = false}'
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Module has dynamic shapes'):
      self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_uses_shape_polymorphism_before_version_8(self):
    # [FUNCTION DOC]
    # Checks that with op version 7 a dynamic-shape module works even without the
    # shape-polymorphism attribute, because that attribute defaults to true for old versions.
    # Input is five ones; asserts the identity-style output equals the input.
    x = np.ones((5,), dtype=np.float32)
    res = x

    def f(x):  # x: f32[b]
      # [FUNCTION DOC]
      # Serializes the module (ignoring the returned maximum version) and calls the op explicitly
      # with version 7; returns the op's output.
      # No `uses_shape_polymorphism` attribute, but it default for version 7
      version = 7
      module, _ = serialize("""
module @jit_fun_3 {
  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xf32> {
    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>
    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>
    return %0 : tensor<?xf32>
  }
  func.func private @dyn_main(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    return %arg1 : tensor<?xf32>
  }
}
""")
      return xla.call_module([x], version=version,
                             module=module,
                             Tout=[res.dtype],
                             Sout=[()],
                             platforms=[self.testing_platform()])

    self._assertOpOutputMatchesExpected(f, (x,), (res,))

  def test_tf_call_function(self):
    # [FUNCTION DOC]
    # Runs a module whose custom call invokes a TensorFlow function (adds two integers) passed in
    # the op's function list. Inputs are two and three; asserts the result is five.
    """A TensorFlow function call inside StableHLO."""
    x = np.int32(2)
    y = np.int32(3)
    res = x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def foo(x, y):
      # [FUNCTION DOC]
      # TensorFlow function called from the module: returns the sum of its two integer inputs.
      return x + y

    def f(x, y):
      # [FUNCTION DOC]
      # Serializes the module with the TensorFlow-call custom call and calls the op, supplying the
      # function list containing the called function; returns the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 0}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(foo,),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res,))

  def test_tf_call_function_multiple_funcs(self):
    # [FUNCTION DOC]
    # Runs a module with three custom calls to two TensorFlow functions (an adder and a wrapper
    # that calls the adder), the third combining the first two results. Inputs two and three;
    # asserts the result is ten.
    """Multiple TensorFlow function calls inside StableHLO."""
    x = np.int32(2)
    y = np.int32(3)
    res = (x + y) + (x + y)

    @function.Defun(dtypes.int32, dtypes.int32)
    def foo(x, y):
      # [FUNCTION DOC]
      # TensorFlow function: returns the sum of its two integer inputs.
      return x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def bar(x, y):
      # [FUNCTION DOC]
      # TensorFlow function: delegates to the adder function and returns its result.
      return foo(x, y)

    def f(x, y):
      # [FUNCTION DOC]
      # Serializes the multi-call module and calls the op with both functions in the function list;
      # returns the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 0}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 1}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.custom_call @tf.call_tf_function(%0, %1) {
      tf.backend_config = {called_index = 1}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %2 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(foo, bar),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res,))

  def test_shape_polymorphic_tf_call_function(self):
    # [FUNCTION DOC]
    # Runs a shape-polymorphic module whose custom call to a TensorFlow function also receives the
    # dynamic dimension size as a shape operand. Inputs are two-element vectors of twos and threes;
    # asserts the element-wise sum.
    """A TensorFlow function call inside StableHLO."""
    x = np.full((2,), 2, dtype=np.int32)
    y = np.full((2,), 3, dtype=np.int32)
    res = x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def foo(x, y):
      # [FUNCTION DOC]
      # TensorFlow function: returns the sum of its two integer inputs.
      return x + y

    def f(x, y):
      # [FUNCTION DOC]
      # Serializes the polymorphic module and calls the op with the function list; returns the
      # op's output.
      module, version = serialize("""
module @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {
  func.func public @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xi32>) -> tensor<i32>
    %1 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1, %0) {
      tf.backend_config = {called_index = 0},
      indices_of_shape_operands = dense<[2]> : tensor<1xi64>
    } : (tensor<?xi32>, tensor<?xi32>, tensor<i32>) -> tensor<?xi32>
    return %1 : tensor<?xi32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(foo,),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res,))

  def test_tf_call_function_with_token(self):
    # [FUNCTION DOC]
    # Runs a module whose TensorFlow-function custom call takes and returns a token, flagged in the
    # backend configuration. Inputs two and three; asserts the numeric result is five.
    """A TensorFlow function call inside StableHLO."""
    x = np.int32(2)
    y = np.int32(3)
    res = x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def foo(x, y):
      # [FUNCTION DOC]
      # TensorFlow function: returns the sum of its two integer inputs.
      return x + y

    def f(x, y):
      # [FUNCTION DOC]
      # Serializes the token-carrying module and calls the op with the function list; returns the
      # op's output.
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: !stablehlo.token, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (!stablehlo.token, tensor<i32>) {
    %0:2 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1, %arg2) {
      tf.backend_config = {called_index = 0, has_token_input_output = true}
    } : (!stablehlo.token, tensor<i32>, tensor<i32>) -> (!stablehlo.token, tensor<i32>)
    return %0#0, %0#1 : !stablehlo.token, tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(foo,),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res,))

  def test_tf_call_function_nested(self):
    # [FUNCTION DOC]
    # Checks nesting: an outer call-module op calls a TensorFlow function that itself contains a
    # call-module op, which calls an adder TensorFlow function. Inputs two and three; asserts five.
    """Nested XlaCallModule inside TensorFlow function calls."""
    x = np.int32(2)
    y = np.int32(3)
    res = x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def add(x, y):
      # [FUNCTION DOC]
      # TensorFlow function: returns the sum of its two integer inputs.
      return x + y

    @function.Defun(dtypes.int32, dtypes.int32)
    def nested_xla_call(x, y):
      # [FUNCTION DOC]
      # TensorFlow function that serializes a module with a custom call to the adder and runs the
      # call-module op (with the adder in its function list); returns the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 0}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(add,),
      )

    @function.Defun(dtypes.int32, dtypes.int32)
    def call(x, y):
      # [FUNCTION DOC]
      # TensorFlow function: delegates to the nested call-module function and returns its result.
      return nested_xla_call(x, y)

    def f(x, y):
      # [FUNCTION DOC]
      # Serializes a module with a custom call to the delegating function and calls the op with it
      # in the function list; returns the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 0}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res.dtype],
          Sout=[res.shape],
          platforms=[self.testing_platform()],
          function_list=(call,),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res,))

  def test_tf_call_function_nested_func_renaming(self):
    # [FUNCTION DOC]
    # Checks that private functions with the same name but different bodies, in two inner modules
    # called from one outer module, are renamed properly on import. One inner function adds and the
    # other subtracts. Inputs two and three; asserts the outputs five and minus one.
    """Multiple custom calls with identically named private functions."""
    x = np.int32(2)
    y = np.int32(3)
    res0 = x + y
    res1 = x - y

    # Verify that multiple inner TF function calls with the same private
    # functions are properly renamed during StableHLO import. This test case is
    # carefully constructed such that one outer XlaCallModule op has two custom
    # calls, each of which has the same private "@call" function with different
    # body. This is to catch bugs in the func renaming logic.

    @function.Defun(dtypes.int32, dtypes.int32)
    def add(x, y):
      # [FUNCTION DOC]
      # TensorFlow function that runs a call-module op on a module adding its inputs through a
      # private function; returns the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func private @call(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }

  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = func.call @call(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res0.dtype],
          Sout=[res0.shape],
          platforms=[self.testing_platform()],
      )

    @function.Defun(dtypes.int32, dtypes.int32)
    def subtract(x, y):
      # [FUNCTION DOC]
      # TensorFlow function that runs a call-module op on a module subtracting its inputs through a
      # private function of the same name as in the adder; returns the op's output.
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func private @call(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.subtract %arg0, %arg1 : tensor<i32>
    return %0 : tensor<i32>
  }

  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
    %0 = func.call @call(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res1.dtype],
          Sout=[res1.shape],
          platforms=[self.testing_platform()],
      )

    def f(x, y):
      # [FUNCTION DOC]
      # Serializes the outer module with two custom calls (to the adder and the subtractor) and
      # calls the op with both functions in the function list; returns both outputs.
      module, version = serialize("""
module @jit_fun_flat_jax {
  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 0}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {
      tf.backend_config = {called_index = 1}
    } : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %0, %1 : tensor<i32>, tensor<i32>
  }
}
""")
      return xla.call_module(
          [x, y],
          version=version,
          module=module,
          Tout=[res0.dtype, res1.dtype],
          Sout=[res0.shape, res1.shape],
          platforms=[self.testing_platform()],
          function_list=(add, subtract),
      )

    self._assertOpOutputMatchesExpected(f, (x, y), (res0, res1))

  def test_op_backward_compatibility(self):
    # [FUNCTION DOC]
    # Creates the raw generated call-module op directly (without the wrapper that fills in default
    # attribute values) for the sine-of-cosine module and asserts the result matches numpy, showing
    # older graphs lacking newer attributes still work.
    """Test for ensuring XlaCallModuleOp backward compatibility."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def f(x):
      # [FUNCTION DOC]
      # Serializes the module and invokes the raw generated op with only the basic attributes;
      # returns the op's output.
      # sin(cos(x))
      module, version = serialize("""
module @jit_f.0 {
  func.func public @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    %0 = stablehlo.cosine %arg0 : tensor<3xf32>
    %1 = stablehlo.sine %0 : tensor<3xf32>
    return %1 : tensor<3xf32>
  }
}
""")
      # Create the raw XlaCallModule op directly instead of calling
      # `xla.call_module`, which handles default values for unpresent
      # attributes.
      return gen_xla_ops.xla_call_module(
          [x],
          version=version,
          module=module,
          Tout=[x.dtype],
          Sout=[x.shape],
          platforms=[self.testing_platform()],
      )

    self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

  def test_op_backward_incompatibility(self):
    # [FUNCTION DOC]
    # Passes an invalid module string directly to the raw op and asserts an invalid-argument error
    # whose message begins with a deserialization failure notice, emulating a backward
    # incompatibility. The call is made directly (not through the jit helper).
    """Test for ensuring XlaCallModuleOp with invalid bytecode."""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    def f(x):
      # [FUNCTION DOC]
      # Calls the raw op with a corrupted, unparseable module text and the maximum supported
      # version; returns the op's output (the call is expected to fail).
      # Use an invalid MLIR string that will fail to parse when loading the
      # call module op, emulating a backward incompatibility.
      corrupted_module = 'stablehlo.invalid_op'
      return gen_xla_ops.xla_call_module(
          [x],
          version=xla.call_module_maximum_supported_version(),
          module=corrupted_module,
          Tout=[x.dtype],
          Sout=[x.shape],
          platforms=[self.testing_platform()],
      )

    # Expect any error message to be included after `:`
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Cannot deserialize computation: .+',
    ):
      f(x)


if __name__ == '__main__':
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(log_device_placement=True)
  )
  googletest.main()
