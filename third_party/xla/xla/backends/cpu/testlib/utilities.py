# [MODULE DOC]
# HELPER: boilerplate helper module for CPU kernel tests.
# Provides functions to turn an HLO module (parsed from text or built from instructions) into a
# scheduled module together with a buffer assignment, as needed to construct kernel emitters.
# Uses the CPU HLO compiler from the CPU testlib package and base HLO types from the codegen
# testlib. No import-time side effects.
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
"""Boilerplate utilities for cpu kernel testing."""

from collections.abc import Sequence

from xla.backends.cpu import testlib as testlib_cpu
from xla.codegen import testlib as testlib_base


def annotate_hlo_module(
    hlo_module: testlib_base.HloModule,
) -> tuple[testlib_base.HloModule, testlib_base.BufferAssignment]:
  # [FUNCTION DOC]
  # HELPER: finishes preparing an HLO module for kernel emission in CPU tests.
  # Input: hlo_module, an HLO module. Creates a CPU HLO compiler, computes a schedule for the
  # module and attaches it to the module (mutating the module), then creates a buffer assignment
  # for it. Returns a pair of the same module object and the buffer assignment.
  # Errors from scheduling or buffer assignment propagate from the native extension.
  hlo_compiler = testlib_cpu.HloCompiler()
  hlo_module.set_schedule(hlo_compiler.create_hlo_schedule(hlo_module))
  buffer_assignment = hlo_compiler.create_buffer_assignment(hlo_module)
  return hlo_module, buffer_assignment


def parse_hlo_module(
    hlo_string: str,
) -> tuple[testlib_base.HloModule, testlib_base.BufferAssignment]:
  # [FUNCTION DOC]
  # HELPER: parses HLO text and prepares it for CPU kernel emission in tests.
  # Input: hlo_string, the textual HLO module. Parses it into a module (raising whatever error
  # the parser raises on invalid text), then schedules it and creates a buffer assignment via
  # the annotation helper. Returns a pair of the module and its buffer assignment.
  hlo_module = testlib_base.HloModule.parse_from_string(hlo_string)
  return annotate_hlo_module(hlo_module)


def build_hlo_module(
    root: testlib_base.HloInstruction,
    *instructions: testlib_base.HloInstruction,
    extra_computations: Sequence[testlib_base.HloComputation] | None = None,
) -> tuple[testlib_base.HloModule, testlib_base.BufferAssignment]:
  # [FUNCTION DOC]
  # HELPER: builds a scheduled single-entry HLO module from instruction objects for CPU tests.
  # Inputs: root, the root instruction (its name becomes the module name); instructions, the
  # remaining instructions the root depends on (variadic); extra_computations, an optional
  # sequence of additional computations (default none) that are added to the module after the
  # entry computation. Builds the entry computation from the root and instructions, then
  # schedules and buffer-assigns the module. Returns a pair of module and buffer assignment.
  """Builds an HLO module from a root instruction and its dependencies.

  Args:
    root: The root instruction of the module.
    *instructions: The instructions that are dependencies of the root
      instruction.
    extra_computations: Any extra computations that should be added to the
      module.

  Returns:
    A tuple containing the HLO module and its buffer assignment.
  """
  hlo_module = testlib_base.HloModule(root.name())
  hlo_module.add_entry_computation(
      testlib_base.build_hlo_computation(root, *instructions)
  )
  if extra_computations is not None:
    for computation in extra_computations:
      hlo_module.add_computation(computation)
  return annotate_hlo_module(hlo_module)
