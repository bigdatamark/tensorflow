# [MODULE DOC]
# Unit tests for the configure script. Each test builds a configuration options object for a
# backend/compiler combination, renders bazelrc lines from a pre-filled paths-and-versions object
# (so no host probing happens) and compares them with golden bazelrc files in the testdata
# directory next to the script. Defines shared constants for paths, versions and CUDA settings.
# Runs under the absl test runner when executed directly.
# Copyright 2024 The OpenXLA Authors.
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
import os

from absl.testing import absltest

from build_tools import test_utils
from build_tools.configure import configure


XLAConfigOptions = configure.XLAConfigOptions
DiscoverablePathsAndVersions = configure.DiscoverablePathsAndVersions
Backend = configure.Backend
HostCompiler = configure.HostCompiler
CudaCompiler = configure.CudaCompiler
RocmCompiler = configure.RocmCompiler
SyclCompiler = configure.SyclCompiler
OS = configure.OS

_PYTHON_BIN_PATH = "/usr/bin/python3"
_CLANG_PATH = "/usr/lib/llvm-18/bin/clang"
_GCC_PATH = "/usr/bin/gcc"
_COMPILER_OPTIONS = ("-Wno-sign-compare",)

# CUDA specific paths and versions
_CUDA_SPECIFIC_PATHS_AND_VERSIONS = {
    "cuda_version": '"12.8.0"',
    "cuda_compute_capabilities": ["7.5"],
    "cudnn_version": '"9.8.0"',
    "ld_library_path": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
}
_CUDA_COMPUTE_CAPABILITIES_AND_LD_LIBRARY_PATH = {
    "cuda_compute_capabilities": [
        "sm_50",
        "sm_60",
        "sm_70",
        "sm_80",
        "compute_90",
    ],
    "ld_library_path": "/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
}


class ConfigureTest(absltest.TestCase):
  # [CLASS DOC]
  # Tests of the bazelrc generation in the configure script. Class-level fixtures are the golden
  # line lists loaded once from testdata files (clang local, gcc, CUDA clang, CUDA clang local,
  # default CUDA clang, NVCC with local clang, NVCC with gcc).

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture. Loads each golden bazelrc file from the testdata directory under the
    # XLA source root, strips whitespace from every line and stores the lists as class attributes.
    # For the gcc based files, the fake gcc path in the file is replaced with the real resolved
    # path of that path on this machine. Reads files; raises a file error if any is missing.
    super().setUpClass()

    testdata = (
        test_utils.xla_src_root() / "build_tools" / "configure" / "testdata"
    )

    with (testdata / "clang_local.bazelrc").open() as f:
      cls.clang_local_bazelrc_lines = [line.strip() for line in f.readlines()]

    with (testdata / "gcc.bazelrc").open() as f:
      resolved_gcc_path = os.path.realpath(_GCC_PATH)
      cls.gcc_bazelrc_lines = [
          line.strip().replace(_GCC_PATH, resolved_gcc_path)
          for line in f.readlines()
      ]

    with (testdata / "cuda_clang.bazelrc").open() as f:
      cls.cuda_clang_bazelrc_lines = [line.strip() for line in f.readlines()]

    with (testdata / "cuda_clang_local.bazelrc").open() as f:
      cls.cuda_clang_local_bazelrc_lines = [
          line.strip() for line in f.readlines()
      ]

    with (testdata / "default_cuda_clang.bazelrc").open() as f:
      cls.default_cuda_clang_bazelrc_lines = [
          line.strip() for line in f.readlines()
      ]

    with (testdata / "nvcc_clang_local.bazelrc").open() as f:
      cls.nvcc_clang_local_bazelrc_lines = [
          line.strip() for line in f.readlines()
      ]

    with (testdata / "nvcc_gcc.bazelrc").open() as f:
      resolved_gcc_path = os.path.realpath(_GCC_PATH)
      cls.nvcc_gcc_bazelrc_lines = [
          line.strip().replace(_GCC_PATH, resolved_gcc_path)
          for line in f.readlines()
      ]

  def test_clang_bazelrc(self):
    # [FUNCTION DOC]
    # Scenario: CPU backend, Linux, clang host compiler, clang path preset with major version 18
    # and an empty library path. Asserts the generated lines equal the clang local golden file.
    config = XLAConfigOptions(
        backend=Backend.CPU,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.CLANG,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.NVCC,
        using_nccl=False,
        rocm_compiler=RocmCompiler.HIPCC,
        sycl_compiler=SyclCompiler.ICPX,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            clang_path=_CLANG_PATH,
            ld_library_path="",
            clang_major_version=18,
        )
    )

    self.assertEqual(bazelrc_lines, self.clang_local_bazelrc_lines)

  def test_gcc_bazelrc(self):
    # [FUNCTION DOC]
    # Scenario: CPU backend, Linux, gcc host compiler with a preset gcc path and empty library
    # path. Asserts the generated lines equal the gcc golden file.
    config = XLAConfigOptions(
        backend=Backend.CPU,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.GCC,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.NVCC,
        using_nccl=False,
        rocm_compiler=RocmCompiler.HIPCC,
        sycl_compiler=SyclCompiler.ICPX,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            gcc_path=_GCC_PATH,
            ld_library_path="",
        )
    )

    self.assertEqual(bazelrc_lines, self.gcc_bazelrc_lines)

  def test_cuda_clang_bazelrc(self):
    # [FUNCTION DOC]
    # Scenario: CUDA backend, CUDA clang compiler with clang host, no clang path (hermetic), and
    # preset CUDA version, capabilities, cuDNN version and library path. Asserts the lines equal
    # the CUDA clang golden file.
    config = XLAConfigOptions(
        backend=Backend.CUDA,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.CLANG,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.CLANG,
        using_nccl=False,
        rocm_compiler=RocmCompiler.HIPCC,
        sycl_compiler=SyclCompiler.ICPX,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            clang_path=None,
            clang_major_version=None,
            **_CUDA_SPECIFIC_PATHS_AND_VERSIONS,
        )
    )

    self.assertEqual(bazelrc_lines, self.cuda_clang_bazelrc_lines)

  def test_cuda_clang_local_bazelrc(self):
    # [FUNCTION DOC]
    # Scenario: CUDA backend with CUDA clang and clang host, local clang path preset with major
    # version 18, plus the shared CUDA settings. Asserts equality with the CUDA clang local file.
    config = XLAConfigOptions(
        backend=Backend.CUDA,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.CLANG,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.CLANG,
        using_nccl=False,
        rocm_compiler=RocmCompiler.HIPCC,
        sycl_compiler=SyclCompiler.ICPX,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            clang_path=_CLANG_PATH,
            clang_major_version=18,
            **_CUDA_SPECIFIC_PATHS_AND_VERSIONS,
        )
    )

    self.assertEqual(bazelrc_lines, self.cuda_clang_local_bazelrc_lines)

  def test_default_cuda_clang_bazelrc(self):
    # [FUNCTION DOC]
    # Scenario: CUDA clang with local clang of major version 17 and only compute capabilities and
    # library path set (no CUDA or cuDNN versions). Asserts equality with the default CUDA clang
    # golden file, covering the version-dependent warning option and omitted version lines.
    config = XLAConfigOptions(
        backend=Backend.CUDA,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.CLANG,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.CLANG,
        using_nccl=False,
        rocm_compiler=RocmCompiler.HIPCC,
        sycl_compiler=SyclCompiler.ICPX,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            clang_path=_CLANG_PATH,
            clang_major_version=17,
            **_CUDA_COMPUTE_CAPABILITIES_AND_LD_LIBRARY_PATH,
        )
    )

    self.assertEqual(bazelrc_lines, self.default_cuda_clang_bazelrc_lines)

  def test_nvcc_clang_local_bazelrc(self):
    # [FUNCTION DOC]
    # Scenario: CUDA backend with NVCC device compiler and a local clang host compiler (major
    # version 18) plus shared CUDA settings. Asserts equality with the NVCC clang local file.
    config = XLAConfigOptions(
        backend=Backend.CUDA,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.CLANG,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.NVCC,
        using_nccl=False,
        rocm_compiler=RocmCompiler.HIPCC,
        sycl_compiler=SyclCompiler.ICPX,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            clang_path=_CLANG_PATH,
            clang_major_version=18,
            **_CUDA_SPECIFIC_PATHS_AND_VERSIONS,
        )
    )

    self.assertEqual(bazelrc_lines, self.nvcc_clang_local_bazelrc_lines)

  def test_nvcc_gcc_bazelrc(self):
    # [FUNCTION DOC]
    # Scenario: CUDA backend with NVCC and gcc host compiler with preset gcc path and shared CUDA
    # settings. Asserts equality with the NVCC gcc golden file.
    config = XLAConfigOptions(
        backend=Backend.CUDA,
        os=OS.LINUX,
        python_bin_path=_PYTHON_BIN_PATH,
        host_compiler=HostCompiler.GCC,
        compiler_options=list(_COMPILER_OPTIONS),
        cuda_compiler=CudaCompiler.NVCC,
        using_nccl=False,
        rocm_compiler=RocmCompiler.HIPCC,
        sycl_compiler=SyclCompiler.ICPX,
    )

    bazelrc_lines = config.to_bazelrc_lines(
        DiscoverablePathsAndVersions(
            gcc_path=_GCC_PATH,
            **_CUDA_SPECIFIC_PATHS_AND_VERSIONS,
        )
    )

    self.assertEqual(bazelrc_lines, self.nvcc_gcc_bazelrc_lines)


if __name__ == "__main__":
  absltest.main()
